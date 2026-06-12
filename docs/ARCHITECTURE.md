# ModelPool Architecture

## Design Philosophy

ModelPool is a **static resource pool**. Each worker runs one model, started at boot, served forever. The pool proxy routes by tag priority with simple failover. No dynamic routing, no model swapping, no worker state queries.

This trades flexibility for simplicity and reliability. In a homelab with dedicated GPUs, each card has a job. The pool just needs to route requests to the right card.

## Architecture A: Static Pool

```
                    ┌─────────────────────────┐
                    │    Pool Proxy            │
                    │    :9000                 │
                    │                          │
                    │  resolve(tag) -> tiers   │
                    │  try tier 1, failover    │
                    └─────────┬───────────────┘
                              │
           ┌───────────┬──────┼──────┬───────────┐
           │           │      │      │           │
    ┌──────▼─────┐ ┌───▼───┐ ┌▼────┐ ┌▼────────┐ ┌────────┐
    │  hwrouter  │ │ pvellm│ │cloud│ │ cloud   │ │ cloud  │
    │  GPU: 27B  │ │ CPU:  │ │ xAI │ │ Z.ai    │ │ ...    │
    │  or 35B    │ │ 35B   │ │     │ │         │ │        │
    └────────────┘ └───────┘ └─────┘ └─────────┘ └────────┘
    (started at boot, always running)
```

## Routing Algorithm

```
1. Request arrives with tag (from header/model field/default "chat")

2. Router.resolve(tag) -> list of (resource, worker) sorted by priority
   - Purely in-memory lookup, no HTTP calls
   - Example for "chat": [(27B-GPU, tier1), (35B-CPU, tier3), (grok, tier4), (glm, tier5)]

3. Proxy tries candidates in order:
   - Try tier 1 (27B GPU)
   - If connection fails -> try tier 2 (35B GPU or CPU)
   - If that fails -> try tier 3 (cloud)
   - First successful connection wins

4. Proxy streams response back to client
```

No worker queries. No state checks. No swaps. Just try in order until one works.

## Resolution

```python
@dataclass
class Resolution:
    tag: str
    resource: Resource
    worker: Worker
```

Three fields. That's it. No swap state, no loaded model tracking, no fallback chains.

## Components

### Router (sync, in-memory)

Reads resources.yaml at startup. `resolve(tag)` returns all candidates sorted by priority. No network calls. No async. Pure computation.

### Pool Proxy (async HTTP)

Receives OpenAI-compatible requests. Resolves tag. Tries candidates in order. Proxies with streaming SSE. Injects auth for external resources. Handles failover on connection errors.

### Worker (subprocess manager)

Starts llama-server at boot with pre-tuned parameters. Serves requests. Reports health via `/worker/status` and `/worker/ready`. That's it.

## Tag Routing Table

| Tag | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|---|
| chat | 27B GPU | | 35B CPU | grok-4.3 | glm-4.5-flash |
| compression | 35B GPU | 35B CPU | grok-4.3 | glm-4.5-flash | |
| title | 35B GPU | 35B CPU | | | |
| triage | 35B GPU | 35B CPU | glm-4.5-flash | | |
| agentic | 27B GPU | grok-4.3 | | | |
| reasoning | 27B GPU | | | | |
| code | 27B GPU | | | | |
| vision | grok-4.3 | | | | |

## Planned: 4-GPU Expansion

```
gpu-dense:   2x RX 9700 PRO 32GB (64 GB VRAM) -> 27B dense, 4 slots
gpu-sparse-1: 1x RX 9070 XT 16GB              -> 35B MoE, 2 slots (partial GPU offload)
gpu-sparse-2: 1x RX 9070 XT 16GB              -> 35B MoE, 2 slots (partial GPU offload)
cpu-backup:   AMD 9850X3D, 48GB RAM            -> 35B MoE, 1 slot (pure CPU)
```

### Updated Tag Routing (4-GPU)

| Tag | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |
|---|---|---|---|---|---|
| chat | 27B gpu-dense | 35B gpu-sparse-1 | 35B gpu-sparse-2 | 35B CPU | cloud |
| agentic | 27B gpu-dense | 35B gpu-sparse-1 | 35B gpu-sparse-2 | cloud | |
| compression | 35B gpu-sparse-1 | 35B gpu-sparse-2 | 35B CPU | grok | glm |
| title | 35B gpu-sparse-1 | 35B gpu-sparse-2 | 35B CPU | | |

Same static routing. More hardware. More tiers. Zero additional complexity in the proxy.

## CPU Inference

The pvellm worker runs pure CPU inference on an AMD Ryzen 9 9850X3D.

```
Binary:        ik_llama.cpp
Model:         Qwen3.6-35B-A3B Q4_K_M (MoE: 3B active params)
Context:       262K (-c 262144)
Threads:       8
Flash Attn:    ON
MTP Spec:      --spec-stage mtp:n_max=3,p_min=0.75
KV Cache:      q8_0, 16384 RAM cache
MoE Optimize:  --merge-up-gate-experts
```

Performance: 473 t/s prompt eval, 39.4 t/s generation.

### Role

CPU is a fallback tier for all GPU resources. When all GPUs are busy or down, the CPU handles overflow. Slower than GPU but always available.

## File Reference

| File | Purpose |
|---|---|
| `resources.yaml` | Resource definitions, worker configs, benchmarks |
| `src/modelpool/registry.py` | YAML parser, validation, lookups |
| `src/modelpool/pool/router.py` | Sync tag-based priority router |
| `src/modelpool/pool/proxy.py` | HTTP proxy with streaming, failover |
| `src/modelpool/pool/server.py` | FastAPI app (/pool/status, /pool/routing) |
| `src/modelpool/worker/loader.py` | Subprocess manager (start/stop) |
| `src/modelpool/worker/watchdog.py` | Health monitor |
| `src/modelpool/worker/server.py` | Worker FastAPI app (/worker/status, /worker/ready) |
