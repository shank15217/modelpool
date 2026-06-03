# ModelPool Architecture

## Core Concept

A **resource** is a fully configured model recipe -- the exact command to launch an inference server, tuned for specific hardware and a specific use case. Resources are loaded into **workers** (machines with GPU/CPU) on demand, and a **pool proxy** routes requests to the right worker based on task tags.

## Current Deployment

```
                    ┌─────────────────────────┐
                    │    modelpool-proxy       │
                    │    LXC 192.168.35.159    │
                    │    Hermes + Pool Proxy   │
                    │    :9000 (proxy)         │
                    └─────────┬───────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌───▼────────┐ ┌──▼──────────┐
        │  hwrouter    │ │  pvellm    │ │  Cloud APIs │
        │  .185        │ │  .17       │ │             │
        │  2x RX9070XT │ │  9850X3D   │ │  xAI / Z.ai │
        │  :8080/:9100 │ │  :8081     │ │             │
        └──────────────┘ └────────────┘ └─────────────┘
```

### Resource Table

| Resource | Type | Hardware | Context | Slots | Benchmark | Generalist |
|---|---|---|---|---|---|---|
| qwen36-27b_mtp_reasoning | GPU | 2x RX9070XT, 64GB VRAM | 131K | 2 | 728/33.8 t/s | Yes |
| qwen36-35b-a3b_mtp | GPU | 2x RX9070XT, 64GB VRAM | 262K | 1 | 2225/71.4 t/s | No |
| qwen36-35b-a3b_cpu | CPU | 9850X3D, 48GB RAM | 262K | 1 | 473/39.4 t/s | No |
| grok-4.3_general | Cloud | xAI API | 256K | - | N/A | No |
| glm-45-flash_general | Cloud | Z.ai free | 131K | - | N/A | No |

### Tag Routing Table

| Tag | Priority 1 | Priority 2 | Priority 3 | Priority 4 | Priority 5 |
|---|---|---|---|---|---|
| chat | 27B GPU (generalist) | 35B GPU | 35B CPU | grok-4.3 | glm-4.5-flash |
| compression | 35B GPU | 35B CPU | grok-4.3 | glm-4.5-flash | |
| title | 35B GPU | 35B CPU | | | |
| triage | 35B GPU | 35B CPU | glm-4.5-flash | | |
| agentic | 27B GPU | grok-4.3 | | | |
| reasoning | 27B GPU | | | | |
| code | 27B GPU | | | | |
| vision | grok-4.3 | | | | |

## Async Router

The router is fully async. All worker status lookups use `httpx.AsyncClient` with a configurable status cache TTL (default 2 seconds).

```
Request arrives
      │
      ▼
  resolve(tag) ──────────────────────────────────────┐
      │                                               │
      ▼                                               │
  Get candidates from registry (sync, in-memory)      │
      │                                               │
      ▼                                               │
  For each candidate (by priority):                   │
      │                                               │
      ├─ _resolve_resource()                          │
      │   ├─ external? -> return immediately          │
      │   └─ managed? -> _resolve_managed()           │
      │       │                                       │
      │       ├─ 1. _find_loaded_generalist()          │
      │       │   └─ check all generalist resources    │
      │       │      on all workers (async, cached)    │
      │       │                                       │
      │       ├─ 2. Already loaded? -> return          │
      │       ├─ 3. Idle worker? -> cold load          │
      │       └─ 4. Swap candidate? -> swap            │
      │                                               │
      ▼                                               │
  First match = primary resolution                     │
  Remaining matches = fallback chain                   │
      │                                               │
      ▼                                               │
  Resolution(resource, worker, needs_swap, chain)      │
```

### Status Cache

Worker status is cached with a 2-second TTL. This means:
- Rapid successive requests to the same tag don't re-query workers
- Cache is invalidated after swaps (proxy calls `invalidate_status_cache()`)
- Failed status checks are also cached (prevents hammering dead workers)
- `get_all_worker_statuses()` uses `asyncio.gather()` for concurrent queries

### Generalist Preference

The 27B dense model is marked `generalist: true`. When it's already loaded on a worker, it serves ANY tag -- even tags it doesn't explicitly have (compression, title, etc.). This avoids unnecessary swaps for light tasks.

Priority: loaded generalist > exact tag match > cold load > swap.

## CPU Inference

The pvellm worker runs pure CPU inference on an AMD Ryzen 9 9850X3D (12C/24T, 48GB RAM).

### Settings

```
Binary:        ik_llama.cpp (optimized fork with MoE/MTP support)
Model:         Qwen3.6-35B-A3B Q4_K_M (MoE: 35B total, 3B active params)
Context:       262,144 tokens (-c 262144)
GPU offload:   0 layers (-ngl 0) -- pure CPU
Threads:       8 (-threads 8) -- leaves 4 cores for OS + worker
Flash Attn:    ON (-fa on)
Reasoning:     OFF (--reasoning off)
Parallel:      1 slot (--parallel 1)
Batch:         2048 / ubatch 256
MTP Spec:      --spec-stage mtp:n_max=3,p_min=0.75
KV Cache:      q8_0 type, 16384 RAM cache (--cache-ram)
KV Checkpoints: 256 (--ctx-checkpoints 256)
MoE Optimize:  --merge-up-gate-experts
```

### Why ik_llama.cpp

ik_llama.cpp is an optimized fork of llama.cpp with significant CPU improvements:
- **MTP speculative decoding**: 19.5x faster generation than standard llama.cpp on CPU
- **MoE optimizations**: `--merge-up-gate-experts` fuses gate computations for sparse models
- **Context checkpoints**: `--ctx-checkpoints 256` enables chunked context processing for large KV caches

### Performance

| Metric | Value |
|---|---|
| Prompt eval | 473 t/s |
| Generation | 39.4 t/s |
| Model size | 21 GB (Q4_K_M) |
| Active params | 3B (MoE) |
| RAM usage | ~30 GB (model + KV cache) |

### Role in Pool

CPU serves as a **fallback tier** for all GPU resources:
- chat: priority 3 (after 27B GPU and 35B GPU)
- compression: priority 2 (after 35B GPU)
- title/triage: priority 2 (after 35B GPU)

When all GPUs are busy or unloaded, the CPU worker handles the overflow. It's slower than GPU (39 t/s vs 71 t/s for the same model) but always available.

---

## Planned: 4-GPU Architecture

The next hardware expansion adds dedicated GPU servers for higher throughput and model diversity.

### Hardware Topology

```
                         ┌─────────────────────────┐
                         │    modelpool-proxy       │
                         │    :9000                 │
                         └─────────┬───────────────┘
                                   │
           ┌───────────┬───────────┼───────────┬───────────┐
           │           │           │           │           │
    ┌──────▼─────┐ ┌───▼──────┐ ┌─▼────────┐ ┌▼─────────┐ ┌▼──────┐
    │  gpu-dense │ │gpu-sparse│ │gpu-sparse│ │ cpu-back │ │ Cloud │
    │  2x9700PRO │ │ 1x9070XT │ │ 1x9070XT │ │ 9850X3D  │ │       │
    │  64GB VRAM │ │ 16GB VRM │ │ 16GB VRM │ │ 48GB RAM │ │       │
    │  4 slots   │ │ 2 slots  │ │ 2 slots  │ │ 1 slot   │ │       │
    └────────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘
```

### Server Specs

| Server | GPUs | VRAM | RAM | Role |
|---|---|---|---|---|
| gpu-dense | 2x AMD RX 9700 PRO 32GB | 64 GB | 64 GB | Dense models, high parallelism |
| gpu-sparse-1 | 1x AMD RX 9070 XT 16GB | 16 GB | 32 GB | MoE/sparse with CPU offload |
| gpu-sparse-2 | 1x AMD RX 9070 XT 16GB | 16 GB | 32 GB | MoE/sparse with CPU offload |
| cpu-backup | (none) | - | 48 GB | CPU fallback (existing pvellm) |

### Model Placement

**gpu-dense (64 GB VRAM)**

```
Model:    Qwen3.6-27B Q4_K_M (~16 GB)
Layout:   Both GPUs, tensor-split 0.5,0.5
Slots:    4 parallel (--parallel 4)
Context:  131K (--c 131072)
KV:       q4_0, cache-ram 16384
Spec:     MTP n_max=3, p_min=0.75
Generalist: Yes
```

The 27B dense model at Q4 fits comfortably in 64 GB with room for the KV cache. 4 parallel slots means 4 concurrent coding agents without queueing. This is the workhorse -- it handles chat, agentic, code, and reasoning, and as the generalist it catches any tag to avoid swaps.

Estimated throughput: ~120-150 t/s combined generation across 4 slots (based on current 2-slot benchmarks of ~50 t/s).

**gpu-sparse-1 and gpu-sparse-2 (16 GB VRAM each)**

```
Model:    Qwen3.6-35B-A3B Q4_K_M (~21 GB)
Layout:   GPU offload + CPU offload (-ngl ~30, rest on CPU)
Slots:    2 parallel each (--parallel 2)
Context:  131K or 262K (depends on CPU RAM for KV)
KV:       q8_0, cache-ram 8192
Spec:     MTP n_max=3, p_min=0.75
```

The 35B MoE is 21 GB but only 3B params are active. With partial GPU offload (~30 layers on GPU, rest on CPU), the 16 GB card handles the hot layers while CPU handles the rest. Each server provides 2 slots.

Estimated throughput: ~25-35 t/s generation per server (less than full GPU due to CPU offload bottleneck, but MoE helps since only 3B params are computed per token).

**cpu-backup (existing pvellm)**

Stays as-is -- 35B MoE pure CPU at 39.4 t/s, fallback for everything.

### Planned Tag Routing

| Tag | Priority 1 | Priority 2 | Priority 3 | Priority 4 | Priority 5 |
|---|---|---|---|---|---|
| chat | 27B dense (generalist) | 35B sparse-1 | 35B sparse-2 | 35B CPU | cloud |
| agentic | 27B dense | 35B sparse-1 | 35B sparse-2 | cloud | |
| code | 27B dense | 35B sparse-1 | 35B sparse-2 | cloud | |
| reasoning | 27B dense | cloud | | | |
| compression | 35B sparse-1 | 35B sparse-2 | 35B CPU | grok-4.3 | glm-flash |
| title | 35B sparse-1 | 35B sparse-2 | 35B CPU | | |
| triage | 35B sparse-1 | 35B sparse-2 | 35B CPU | glm-flash | |
| vision | grok-4.3 | | | | |

### Routing Flow

```
chat request arrives
      │
      ├─ Is 27B loaded on gpu-dense? ──Yes──> Use it (generalist, 4 slots)
      │
      ├─ Is 35B loaded on gpu-sparse-1? ──Yes──> Use it (2 slots)
      │
      ├─ Is 35B loaded on gpu-sparse-2? ──Yes──> Use it (2 slots)
      │
      ├─ Is 35B loaded on cpu-backup? ──Yes──> Use it (1 slot)
      │
      └─ Send to cloud fallback
```

### Capacity Analysis

| Scenario | Slots Available | Total Throughput |
|---|---|---|
| Best case (all loaded) | 4 + 2 + 2 + 1 = 9 | ~250-350 t/s combined |
| Coding agents only (27B) | 4 | ~120-150 t/s |
| Compression burst | 2 + 2 + 1 = 5 | ~130-175 t/s |
| CPU-only fallback | 1 | ~39 t/s |

### Swap Strategy

With 4 workers, the pool can keep multiple models loaded simultaneously:
- **gpu-dense**: Always has 27B loaded (generalist, never swapped)
- **gpu-sparse-1**: Default 35B, can swap if a different model is needed
- **gpu-sparse-2**: Default 35B, mirrors sparse-1 for redundancy
- **cpu-backup**: Always has 35B loaded (never swapped)

This means the 27B and both 35Bs are typically always loaded, giving zero-swap routing for most requests.

### Partial GPU Offload Tuning

For the 16 GB cards running the 21 GB MoE model, the key tuning parameters are:

```
-ngl N              # Number of layers on GPU (rest on CPU)
-cache-ram 8192     # Smaller RAM cache (less RAM than dense server)
-threads 6          # Leave cores for GPU-CPU transfer
```

The exact `-ngl` value depends on layer sizes. For Qwen3.6-35B-A3B:
- ~60 transformer layers, ~350 MB each at Q4
- 16 GB VRAM - 2 GB overhead = ~14 GB for layers
- 14 GB / 350 MB = ~40 layers on GPU
- Remaining ~20 layers on CPU

This gives a good balance: the GPU handles the attention-heavy layers while CPU handles the FFN layers (which are small for MoE since only 3B params are active).

---

## Worker Internals

### Subprocess Lifecycle

```
idle ──load()──> loading ──_wait_healthy()──> ready
  ▲                  │                           │
  │                  │ (fail)                    │
  │                  ▼                           │
  │                error                         │
  │                  │                           │
  │                  └──stop()──> idle <──stop()─┘
  │                               ▲
  └──────revert()─────────────────┘
```

- `load()`: drain existing requests, stop subprocess, start new subprocess, wait for health check
- `stop()`: send SIGTERM, wait for clean exit, escalate to SIGKILL after timeout
- `revert()`: stop current model, load the worker's default resource
- All blocking operations wrapped in `asyncio.to_thread()` to avoid event loop freezes

### Auth Middleware

Worker endpoints use `X-Pool-Secret` header auth:
- `hmac.compare_digest()` for timing-safe comparison
- Path normalization strips trailing slashes
- `/worker/status` and `/worker/ready` are open (monitoring)
- All other endpoints require the secret

### Idle Shutdown

Workers auto-unload models after 15 minutes of inactivity (900s). This frees GPU memory for other workloads. When a request arrives, the pool proxy triggers a cold load.

---

## File Reference

| File | Purpose |
|---|---|
| `resources.yaml` | Resource definitions, worker configs, benchmarks |
| `src/modelpool/registry.py` | YAML parser, validation, lookups |
| `src/modelpool/pool/router.py` | Async tag-based priority router with status cache |
| `src/modelpool/pool/proxy.py` | HTTP proxy with streaming SSE, auth injection |
| `src/modelpool/pool/server.py` | FastAPI app (endpoints, idle timer) |
| `src/modelpool/worker/loader.py` | Subprocess manager, state machine |
| `src/modelpool/worker/watchdog.py` | Health monitor, auto-recovery |
| `src/modelpool/worker/server.py` | Worker FastAPI app, auth middleware |
