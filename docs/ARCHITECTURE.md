# ModelPool Architecture

## Core Concept

A **resource** is a fully configured model recipe -- the GGUF weights plus every llama-server flag tuned for specific hardware and a specific use case. Resources are tested and benchmarked as a unit. The pool loads resources on demand, serves them via OpenAI-compatible API, and shuts them down when done.

```
Resource = Model weights + Hardware config + Launch params + Use case tags
```

Example resource: `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu`
- Model: Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf
- Hardware: Both GPUs (ROCm0, ROCm1), tensor split 50/50
- Params: 262K ctx, MTP speculative decoding, reasoning off, cont-batching
- Use cases: context compression, light agentic work, summarization
- Benchmark: 24 tok/s generation, 91 tok/s prompt eval on 100K context

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client     │────▶│   ModelPool      │────▶│   Worker        │
│   (Hermes)   │     │   (proxy/router) │     │   (llama-server)│
└─────────────┘     └──────┬───────────┘     └─────────────────┘
                          │
                    ┌─────▼──────┐
                    │  Resources │
                    │  (recipes) │
                    └────────────┘
```

## Data Flow

```
1. Client sends POST /v1/chat/completions
   Header: X-Task-Type: compression
   Body: { messages: [...], max_tokens: 4096 }

2. Pool resolves task_type -> resource
   compression -> qwen36-35b-a3b_mtp_no-reasoning_multi-gpu

3. Pool checks workers for resource availability
   hwrouter: serving qwen36-27b_mtp_vision_multi-gpu (different resource)

4. Pool tells worker to load resource
   POST http://hwrouter:9100/worker/load
   { "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }

5. Worker: drain -> stop current -> start llama-server with resource recipe
   ~45-60s

6. Worker reports ready

7. Pool proxies request to worker inference port

8. After idle timeout (5min) -> worker reverts to default resource
```

## Resource Registry (`resources.yaml`)

Each resource is a named, tested, benchmarked configuration.

```yaml
resources:
  # --- Qwen3.6-27B resources ---

  qwen36-27b_mtp_vision_multi-gpu:
    description: "Qwen3.6-27B with vision, MTP, 256K ctx, both GPUs"
    model: /AITOOLCHAIN/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf
    mmproj: /AITOOLCHAIN/models/Qwen3.6-27B/mmproj-BF16.gguf
    size_gb: 16
    ctx: 262144
    capabilities: [chat, vision, code, agentic]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: 91.3
      generation_tps: 24.0
      tested_at: 2026-05-25
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, /AITOOLCHAIN/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf]
        - [--mmproj, /AITOOLCHAIN/models/Qwen3.6-27B/mmproj-BF16.gguf]
        - [-c, "262144"]
        - [-ngl, "99"]
        - [-fa, "on"]
        - [--cache-type-k, q4_0]
        - [--cache-type-v, q4_0]
        - [--reasoning, "off"]
        - [--jinja]
        - [--chat-template-kwargs, '{"preserve_thinking":true}']
        - [--no-mmap]
        - [--kv-unified]
        - [--tensor-split, "0.5,0.5"]
        - [--device, "ROCm0,ROCm1"]
        - [--parallel, "4"]
        - [--cont-batching]
        - [--batch-size, "2048"]
        - [--ubatch-size, "1024"]
        - [--port, "8080"]
        - [--host, "0.0.0.0"]
        - [--temp, "0.7"]
        - [--top-p, "0.95"]
        - [--top-k, "40"]
        - [--min-p, "0.05"]
        - [--presence-penalty, "0.0"]
        - [--spec-type, draft-mtp]
        - [--spec-draft-n-max, "3"]
        - [--spec-draft-p-min, "0.75"]
        - [--cache-ram, "16384"]

  # --- Qwen3.6-35B-A3B resources ---

  qwen36-35b-a3b_mtp_no-reasoning_multi-gpu:
    description: "35B MoE (3B active), MTP, no reasoning, 256K ctx, both GPUs. Best for compression."
    model: /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf
    size_gb: 21
    ctx: 262144
    capabilities: [chat, summarize, compression]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: 3420    # 100K tokens in 32s
      generation_tps: 8.2      # MoE 3B active
      tested_at: 2026-05-25
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf]
        - [-c, "262144"]
        - [-ngl, "99"]
        - [-fa, "on"]
        - [--cache-type-k, q8_0]
        - [--cache-type-v, q8_0]
        - [--reasoning, "off"]
        - [--jinja]
        - [--chat-template-kwargs, '{"preserve_thinking":true}']
        - [--no-mmap]
        - [--kv-unified]
        - [--tensor-split, "0.5,0.5"]
        - [--device, "ROCm0,ROCm1"]
        - [--parallel, "1"]
        - [--cont-batching]
        - [--batch-size, "1024"]
        - [--ubatch-size, "512"]
        - [--port, "8080"]
        - [--host, "0.0.0.0"]
        - [--temp, "0.7"]
        - [--top-p, "0.95"]
        - [--top-k, "40"]
        - [--min-p, "0.05"]
        - [--presence-penalty, "0.0"]
        - [--spec-type, draft-mtp]
        - [--spec-draft-n-max, "3"]
        - [--spec-draft-p-min, "0.75"]
        - [--cache-ram, "16384"]

  qwen36-35b-a3b_no-reasoning_cpu:
    description: "35B MoE on CPU (9850X3D), no reasoning. Slow but free overflow."
    model: /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf
    size_gb: 21
    ctx: 262144
    capabilities: [chat, summarize]
    workers: [aitooolchain]
    benchmark:
      prompt_eval_tps: 138
      generation_tps: 2.0
      tested_at: 2026-05-25
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf]
        - [-c, "262144"]
        - [-ngl, "0"]
        - [--threads, "14"]
        - [-fa, "on"]
        - [--cache-type-k, q8_0]
        - [--cache-type-v, q8_0]
        - [--reasoning, "off"]
        - [--parallel, "1"]
        - [--batch-size, "512"]
        - [--ubatch-size, "256"]
        - [--port, "8081"]
        - [--host, "0.0.0.0"]
        - [--temp, "0.7"]
        - [--top-p, "0.95"]

  # --- Fine-tuned / specialized resources ---

  ornstein-27b-saber_q5_multi-gpu:
    description: "Ornstein-Hermes SABER fine-tune, Q5, for code review"
    model: /AITOOLCHAIN/models/Qwen3.6-27B/Ornstein-Hermes-3.6-27b-SABER-Q5_K_M.gguf
    size_gb: 18
    ctx: 131072
    capabilities: [chat, code, review]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: null
      generation_tps: null
      tested_at: null    # not yet benchmarked
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, /AITOOLCHAIN/models/Qwen3.6-27B/Ornstein-Hermes-3.6-27b-SABER-Q5_K_M.gguf]
        - [-c, "131072"]
        - [-ngl, "99"]
        - [-fa, "on"]
        - [--cache-type-k, q4_0]
        - [--cache-type-v, q4_0]
        - [--tensor-split, "0.5,0.5"]
        - [--device, "ROCm0,ROCm1"]
        - [--parallel, "2"]
        - [--cont-batching]
        - [--batch-size, "1024"]
        - [--ubatch-size, "512"]
        - [--port, "8080"]
        - [--host, "0.0.0.0"]

  qwen35-9b_q8_multi-gpu:
    description: "Small fast model for triage, approval, lightweight routing"
    model: /AITOOLCHAIN/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q8_0.gguf
    size_gb: 9
    ctx: 131072
    capabilities: [chat, triage]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: null
      generation_tps: null
      tested_at: null
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, /AITOOLCHAIN/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q8_0.gguf]
        - [-c, "131072"]
        - [-ngl, "99"]
        - [-fa, "on"]
        - [--cache-type-k, q4_0]
        - [--cache-type-v, q4_0]
        - [--tensor-split, "0.5,0.5"]
        - [--device, "ROCm0,ROCm1"]
        - [--parallel, "8"]
        - [--cont-batching]
        - [--batch-size, "2048"]
        - [--ubatch-size, "1024"]
        - [--port, "8080"]
        - [--host, "0.0.0.0"]

# --- Workers ---

workers:
  hwrouter:
    host: 192.168.35.185
    worker_port: 9100
    inference_port: 8080
    type: gpu
    vram_gb: 32
    max_model_gb: 28
    swap_timeout: 120
    drain_timeout: 30
    default_resource: qwen36-27b_mtp_vision_multi-gpu

  aitooolchain:
    host: 192.168.35.17
    worker_port: 9100
    inference_port: 8081
    type: cpu
    ram_gb: 48
    max_model_gb: 40
    swap_timeout: 60
    drain_timeout: 10
    default_resource: qwen36-35b-a3b_no-reasoning_cpu

# --- Routing ---

routing:
  compression:
    resource: qwen36-35b-a3b_mtp_no-reasoning_multi-gpu
    fallback_resource: qwen36-35b-a3b_no-reasoning_cpu
    timeout: 120
    idle_revert: 300
    swap_behavior: queue

  code-review:
    resource: ornstein-27b-saber_q5_multi-gpu
    fallback_resource: qwen36-27b_mtp_vision_multi-gpu
    timeout: 300
    idle_revert: 180
    swap_behavior: queue

  triage:
    resource: qwen35-9b_q8_multi-gpu
    fallback_resource: qwen36-27b_mtp_vision_multi-gpu
    timeout: 30
    idle_revert: 60
    swap_behavior: fallback

  chat:
    resource: qwen36-27b_mtp_vision_multi-gpu
    fallback_resource: none
    timeout: 120
    idle_revert: 0
    swap_behavior: fallback
```

## Components

### 1. Worker Agent

Runs on each inference host. Manages llama-server lifecycle.

**Responsibilities:**
- Accept load/unload requests with a resource name
- Look up the resource recipe from the registry
- Execute the exact command as defined in the resource (no generation needed)
- Report state: what resource is loaded, slot usage, uptime
- Graceful drain before stopping
- Watchdog for hung/OOM'd processes

**API:**

```
GET  /worker/status          -> { resource, state, slots, uptime }
POST /worker/load            -> { resource } -> drain -> stop -> start -> ready
POST /worker/unload          -> drain -> stop -> idle
GET  /worker/ready           -> 200 or 503
POST /worker/revert          -> load default resource
```

**Key design point:** The worker does NOT generate the command. It receives a resource name, looks up the command from the registry, and executes it exactly as specified. The resource IS the tested configuration. No parameter merging, no interpolation, no surprises.

### 2. Pool (Proxy/Router)

Sits between clients and workers. Routes by task type.

**Responsibilities:**
- Resolve task_type -> resource -> worker
- Check if resource is already loaded
- Trigger load if needed, proxy when ready
- Manage idle timers (revert to default after inactivity)
- Queue or fallback during swaps (per routing config)
- Streaming passthrough

**API:**

```
POST /v1/chat/completions     # X-Task-Type header -> route -> proxy
GET  /v1/models               # list loaded resources across workers
GET  /pool/status             # workers, loaded resources, idle timers
GET  /pool/routing            # current task -> resource mapping
POST /pool/swap               # manual swap for admin/testing
```

### 3. Resource Registry (`resources.yaml`)

Single file. Each resource is a named entry with:
- Description and use case tags
- Exact llama-server command (binary + flags)
- Size, context length, capabilities
- Which workers can serve it
- Benchmark results (optional, filled after testing)

**Adding a new resource:**
1. Test the llama-server command manually on the target hardware
2. Benchmark it
3. Add the exact working command as a new resource entry
4. Tag it with use cases in the routing table

No abstraction, no parameter generation. What you tested is what runs.

## State Machine

Each worker:

```
IDLE --[load(resource)]--> LOADING --[health OK]--> READY
READY --[load(new_resource)]--> DRAINING --> STOPPING --> LOADING --> READY
READY --[error]--> ERROR --[auto-recover]--> IDLE
Any state --[unload]--> DRAINING --> STOPPING --> IDLE
```

## Concurrency

During a swap, new requests either:
- **Queue** (batch tasks: compression, code-review) -- wait for optimal resource
- **Fallback** (interactive: chat) -- route to alternate resource immediately

Controlled per-task in routing config via `swap_behavior`.

## Deployment

```
Hermes Host                Workers
┌─────────────────┐        ┌──────────────┐  ┌──────────────┐
│ modelpool-pool  │───────▶│ hwrouter     │  │ aitooolchain │
│ port 9000       │        │ worker :9100 │  │ worker :9100 │
│                 │        │ llama :8080  │  │ llama :8081  │
└─────────────────┘        └──────────────┘  └──────────────┘
```

All config in one file: `resources.yaml`. Deployed to pool + all workers.
Workers read-only their own section. Pool reads routing + workers.
