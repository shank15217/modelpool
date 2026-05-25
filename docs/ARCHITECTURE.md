# ModelPool Architecture

## Overview

ModelPool is a task-routed LLM inference proxy. It sits between AI agents (like Hermes) and GPU inference servers (like llama.cpp). When a request arrives tagged with a task type, ModelPool ensures the optimal model is loaded on the optimal worker before proxying the request.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client     │────▶│   ModelPool      │────▶│   Worker (GPU)  │
│   (Hermes)   │     │   (proxy/router) │     │   llama-server  │
└─────────────┘     └──────┬───────────┘     └─────────────────┘
                          │
                    ┌─────▼──────┐
                    │  Registry  │
                    │  (models)  │
                    └────────────┘
```

## Design Principles

1. **Transparent proxy** -- Clients send standard OpenAI API requests. ModelPool handles everything else.
2. **Task-based routing** -- A request header or body field (`task_type`) tells ModelPool which model profile to use.
3. **Zero persistence** -- No database. State lives in memory and on disk (YAML configs). Workers are stateless.
4. **Fail fast, fallback often** -- If the optimal model can't load, fall back to the next best option immediately.
5. **Minimize swap time** -- Pre-warm workers, parallel operations, track load times per model+worker combo.

## Data Flow

```
1. Client sends POST /v1/chat/completions
   Header: X-Task-Type: compression
   Body: { messages: [...], max_tokens: 4096 }

2. Pool resolves task_type -> model_profile
   compression -> { model: qwen36-35b-a3b, worker: hwrouter, fallback: qwen36-27b }

3. Pool checks worker state
   GET http://hwrouter:9100/worker/status
   -> { loaded: "qwen36-27b", slots_idle: 3, vram_free: 2gb }

4. Model mismatch -> swap
   POST http://hwrouter:9100/worker/load
   { model: "qwen36-35b-a3b" }
   -> 202 Accepted, polling for ready...

5. Worker drains active slots, stops llama-server, starts new model
   ~45-60s depending on model size

6. Worker reports ready
   GET http://hwrouter:9100/worker/ready -> 200

7. Pool proxies the original request
   POST http://hwrouter:8080/v1/chat/completions
   -> response streams back to client

8. Pool starts idle timer for this model on this worker
   After 5min with no requests for this model -> revert to default
```

## Components

### 1. Registry (`models.yaml`)

The single source of truth for what models exist and how to run them.

```yaml
# Registry version
version: 1

# Default model loaded on worker boot
defaults:
  hwrouter: qwen36-27b-q4
  aitooolchain: qwen36-35b-a3b-q4-cpu

# Worker definitions
workers:
  hwrouter:
    host: 192.168.35.185
    port: 9100            # worker agent port
    inference_port: 8080   # llama-server port
    type: gpu
    vram_gb: 32
    devices: [Vulkan0, Vulkan1]
    max_model_gb: 28       # leave headroom for KV cache
    swap_timeout: 120      # max seconds to wait for model swap

  aitooolchain:
    host: 192.168.35.17
    port: 9100
    inference_port: 8081
    type: cpu
    cores: 16
    ram_gb: 48
    max_model_gb: 40
    swap_timeout: 30       # CPU models load faster (no VRAM transfer)

# Model definitions
models:
  qwen36-27b-q4:
    path: /AITOOLCHAIN/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf
    mmproj: /AITOOLCHAIN/models/Qwen3.6-27B/mmproj-BF16.gguf
    size_gb: 16
    ctx: 262144
    capabilities: [chat, vision, code]
    params:
      parallel: 4
      cont_batching: true
      tensor_split: [0.5, 0.5]
      cache_type: [q4_0, q4_0]
      flash_attn: true
      batch_size: 2048
      ubatch_size: 1024

  qwen36-35b-a3b-q4:
    path: /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf
    size_gb: 21
    ctx: 262144
    capabilities: [chat, summarize]
    params:
      parallel: 1
      reasoning: off
      tensor_split: [0.5, 0.5]
      cache_type: [q4_0, q4_0]
      flash_attn: true

  qwen36-35b-a3b-q4-cpu:
    path: /AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf
    size_gb: 21
    ctx: 262144
    capabilities: [chat, summarize]
    params:
      parallel: 1
      reasoning: off
      threads: 14
      batch_size: 512
      cache_type: [q8_0, q8_0]

  ornstein-27b-saber-q5:
    path: /AITOOLCHAIN/models/Qwen3.6-27B/Ornstein-Hermes-3.6-27b-SABER-Q5_K_M.gguf
    size_gb: 18
    ctx: 131072
    capabilities: [chat, code]
    params:
      parallel: 2
      cont_batching: true
      tensor_split: [0.5, 0.5]
      cache_type: [q4_0, q4_0]
      flash_attn: true

  qwen35-9b-q8:
    path: /AITOOLCHAIN/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q8_0.gguf
    size_gb: 9
    ctx: 131072
    capabilities: [chat, triage]
    params:
      parallel: 8
      cont_batching: true
      tensor_split: [0.5, 0.5]
      cache_type: [q4_0, q4_0]

# Task routing table
routing:
  compression:
    model: qwen36-35b-a3b-q4
    worker_preference: [hwrouter]
    fallback_model: qwen36-27b-q4      # if optimal won't load
    fallback_worker: [aitooolchain]     # CPU fallback
    timeout: 120
    idle_revert: 300                    # seconds before reverting to default

  code-review:
    model: ornstein-27b-saber-q5
    worker_preference: [hwrouter]
    fallback_model: qwen36-27b-q4
    timeout: 300
    idle_revert: 180

  triage:
    model: qwen35-9b-q8
    worker_preference: [hwrouter]
    fallback_model: qwen36-27b-q4
    timeout: 30
    idle_revert: 60                     # revert fast, triage is bursty

  chat:
    model: qwen36-27b-q4               # this is the default, never swapped
    worker_preference: [hwrouter]
    fallback_model: none
    timeout: 120
    idle_revert: 0                      # never revert, this IS the default
```

### 2. Worker Agent

A lightweight HTTP agent running on each inference host alongside llama-server.

**Responsibilities:**
- Start/stop/restart llama-server with specified model and parameters
- Report current state (loaded model, VRAM/RAM, slot utilization)
- Graceful drain: stop accepting new requests, wait for in-flight to complete
- Generate correct llama-server command line from model params
- Watchdog: detect hung/OOM'd processes, auto-recover

**API:**

```
GET  /worker/status
  Response: {
    "worker_id": "hwrouter",
    "state": "ready" | "loading" | "draining" | "error",
    "loaded_model": "qwen36-27b-q4",
    "vram_total_gb": 32,
    "vram_used_gb": 16.2,
    "slots_idle": 3,
    "slots_processing": 1,
    "uptime_s": 86400,
    "load_time_s": null  // only set during "loading" state
  }

POST /worker/load
  Body: { "model_id": "qwen36-35b-a3b-q4" }
  Response: 202 { "status": "loading", "estimated_s": 60 }
  Flow:
    1. Set state to "draining"
    2. Wait for all in-flight requests to complete (up to drain_timeout)
    3. Stop llama-server (SIGTERM, wait, SIGKILL)
    4. Set state to "loading"
    5. Start llama-server with new model params
    6. Poll /health until 200 OK
    7. Set state to "ready"
    8. Report loaded_model

POST /worker/unload
  Response: 202 { "status": "draining" }
  Flow: drain -> stop -> state = "idle"

GET  /worker/ready
  Response: 200 when state == "ready", 503 otherwise

POST /worker/revert
  Body: { "model_id": "qwen36-27b-q4" }
  Response: 202
  Flow: Same as load but uses the default model from registry
```

**Implementation:** Python, ~300 lines. Uses subprocess to manage llama-server, http.server or FastAPI for the API. Runs as systemd service `modelpool-worker.service`.

**Command generation:**

The worker builds the llama-server command from the model definition:

```python
def build_command(model_def: dict, worker_def: dict) -> list[str]:
    """Generate llama-server command from model + worker config."""
    binary = "/AITOOLCHAIN/llama.cpp/build/bin/llama-server"
    params = model_def["params"]

    cmd = [
        binary,
        "-m", model_def["path"],
        "-c", str(model_def["ctx"]),
        "--port", str(worker_def["inference_port"]),
        "--host", "0.0.0.0",
    ]

    # GPU-specific params
    if worker_def["type"] == "gpu":
        devices = params.get("devices", worker_def.get("devices", []))
        if devices:
            cmd.extend(["--device", ",".join(devices)])
        if params.get("tensor_split"):
            cmd.extend(["--tensor-split", ",".join(str(s) for s in params["tensor_split"])])

    # CPU-specific params
    if worker_def["type"] == "cpu":
        threads = params.get("threads", worker_def.get("cores", 4))
        cmd.extend(["--threads", str(threads)])

    # Common optional params
    if model_def.get("mmproj"):
        cmd.extend(["--mmproj", model_def["mmproj"]])
    if params.get("parallel"):
        cmd.extend(["--parallel", str(params["parallel"])])
    if params.get("cont_batching"):
        cmd.append("--cont-batching")
    if params.get("flash_attn"):
        cmd.extend(["-fa", "on"])
    if params.get("reasoning") == "off":
        cmd.extend(["--reasoning", "off"])
    if params.get("cache_type"):
        ct = params["cache_type"]
        cmd.extend(["--cache-type-k", ct[0], "--cache-type-v", ct[1]])
    if params.get("batch_size"):
        cmd.extend(["--batch-size", str(params["batch_size"])])
    if params.get("ubatch_size"):
        cmd.extend(["--ubatch-size", str(params["ubatch_size"])])

    return cmd
```

### 3. Pool (Proxy/Router)

The central coordinator. Sits on the network and accepts client requests.

**Responsibilities:**
- Accept OpenAI-format requests with task routing hints
- Resolve task -> model -> worker
- Check worker state, trigger model swaps if needed
- Proxy requests to the correct worker's inference port
- Manage idle timers (auto-revert models after inactivity)
- Queue management for concurrent requests during swaps
- Metrics: request counts, latencies, swap counts per task type

**API:**

```
# Standard OpenAI-compatible endpoints (pass-through to workers)
POST /v1/chat/completions
  Headers:
    X-Task-Type: compression | code-review | triage | chat | ...
    # OR pass in extra_body via the request JSON
  Body: standard OpenAI chat completions format
  Response: standard OpenAI response (streaming supported)

GET /v1/models
  Response: lists currently loaded models across all workers

# Pool management endpoints
GET /pool/status
  Response: {
    "workers": {
      "hwrouter": { "state": "ready", "loaded": "qwen36-27b-q4", ... },
      "aitooolchain": { "state": "ready", "loaded": "qwen36-35b-a3b-q4-cpu", ... }
    },
    "idle_timers": {
      "hwrouter": { "model": "qwen36-35b-a3b-q4", "expires_in_s": 142 }
    }
  }

GET /pool/routing
  Response: current routing table with resolved model+worker for each task

POST /pool/swap
  Body: { "worker": "hwrouter", "model": "qwen36-35b-a3b-q4" }
  Response: 202 { "status": "initiated" }
  # Manual swap endpoint for testing/admin
```

**Request routing logic:**

```python
async def route_request(request):
    task_type = request.headers.get("X-Task-Type", "chat")
    route = registry.get_route(task_type)

    # Try preferred worker with preferred model
    for worker_id in route.worker_preference:
        worker = workers[worker_id]
        status = await worker.get_status()

        if status.state == "error":
            continue

        if status.loaded_model == route.model:
            # Already loaded, proxy immediately
            return await proxy(request, worker)

        if status.state == "ready" and can_fit(route.model, worker):
            # Need to swap
            swap_ok = await worker.load_model(route.model, timeout=route.timeout)
            if swap_ok:
                reset_idle_timer(worker_id, route.model, route.idle_revert)
                return await proxy(request, worker)

    # Preferred model unavailable, try fallback
    if route.fallback_model:
        for worker_id in route.fallback_worker or route.worker_preference:
            worker = workers[worker_id]
            status = await worker.get_status()

            if status.loaded_model == route.fallback_model:
                return await proxy(request, worker)

            if status.state == "ready" and can_fit(route.fallback_model, worker):
                swap_ok = await worker.load_model(route.fallback_model, timeout=route.timeout)
                if swap_ok:
                    return await proxy(request, worker)

    # Everything failed
    return JSONResponse({"error": "No available worker for task"}, status_code=503)
```

**Idle timer management:**

```python
# Per-worker idle timer
idle_timers: dict[str, asyncio.Task] = {}

def reset_idle_timer(worker_id: str, model_id: str, revert_after_s: int):
    if revert_after_s == 0:
        return  # never revert (default model)

    # Cancel existing timer
    if worker_id in idle_timers:
        idle_timers[worker_id].cancel()

    async def revert():
        await asyncio.sleep(revert_after_s)
        worker = workers[worker_id]
        status = await worker.get_status()
        if status.loaded_model == model_id:
            await worker.load_model(registry.get_default(worker_id))

    idle_timers[worker_id] = asyncio.create_task(revert())
```

### 4. Streaming Support

ModelPool must support streaming responses (SSE). The proxy streams chunks from the worker back to the client without buffering the entire response.

```python
async def proxy_streaming(request, worker):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"http://{worker.host}:{worker.inference_port}/v1/chat/completions",
            json=request.body,
            timeout=300,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk
```

## State Machine

Each worker follows this state machine:

```
                  ┌───────────┐
          boot    │           │
        ─────────▶│   IDLE    │
                  │           │
                  └─────┬─────┘
                        │ load_model()
                        ▼
                  ┌───────────┐
                  │           │
                  │  LOADING  │──── health check OK
                  │           │
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │           │◀─────────────────┐
                  │   READY   │                  │
                  │           │──── load_model() │ (swap to different model)
                  └─────┬─────┘         │
                        │               ▼
                        │         ┌───────────┐
                        │         │           │
                        │         │ DRAINING  │
                        │         │           │
                        │         └─────┬─────┘
                        │               │ all slots idle
                        │               ▼
                        │         ┌───────────┐
                        │         │ STOPPING  │──▶ LOADING (new model)
                        │         └───────────┘
                        │
                  error │
                        ▼
                  ┌───────────┐
                  │           │── auto-recover ──▶ IDLE (load default)
                  │   ERROR   │
                  │           │
                  └───────────┘
```

## Concurrency Model

**During a model swap, what happens to incoming requests?**

Option A: **Queue** (default for non-interactive tasks)
- Pool holds the request in a per-worker queue
- When worker becomes ready, queued requests are processed in order
- Respects the task's `timeout` -- if timeout expires, try fallback or 503

Option B: **Fallback immediately** (default for interactive tasks)
- Pool routes to the fallback model on a different worker
- No waiting, no queueing
- Lower quality but immediate response

The routing config controls this per task:
```yaml
routing:
  compression:
    swap_behavior: queue        # wait for optimal model
    queue_timeout: 90
  chat:
    swap_behavior: fallback     # never block interactive
```

## Configuration

ModelPool uses three config files:

| File | Purpose | Changed by |
|---|---|---|
| `models.yaml` | Model registry, worker defs, routing table | Admin (manual edit) |
| `pool.yaml` | Pool process config (listen port, logging, metrics) | Admin |
| `worker.yaml` | Worker process config (per-host, deployed to each worker) | Admin |

### `pool.yaml`

```yaml
listen:
  host: 0.0.0.0
  port: 9000

registry_path: /etc/modelpool/models.yaml

logging:
  level: info
  file: /var/log/modelpool/pool.log

metrics:
  enabled: true
  port: 9100
  path: /metrics
```

### `worker.yaml`

```yaml
worker_id: hwrouter
registry_path: /etc/modelpool/models.yaml

llama_server:
  binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
  health_endpoint: /health
  drain_timeout: 30
  stop_timeout: 10
  startup_timeout: 120

logging:
  level: info
  file: /var/log/modelpool/worker.log
```

## Error Handling

| Scenario | Pool behavior | Worker behavior |
|---|---|---|
| Worker unreachable | Try next worker, then fallback model, then 503 | N/A |
| Model swap timeout | 503 + `Retry-After` header | Force-stop llama-server, report ERROR |
| OOM during load | Try smaller model (fallback) | Report ERROR, auto-recover to default |
| Inference timeout | 504 Gateway Timeout | N/A (client-side timeout) |
| Worker crashes mid-swap | Detect via health check, mark worker ERROR | systemd auto-restarts worker agent |
| llama-server OOM | Route to fallback worker | Watchdog detects, restarts with default model |

## Security

- Workers only accept connections from the pool (firewall by source IP)
- Pool only accepts connections from known clients (Hermes host)
- No authentication on inference endpoints (homelab, firewalled network)
- Registry files are read-only for worker processes

## Deployment

```
┌─────────────────────────────────────────────────┐
│  Hermes Host (192.168.35.x)                     │
│                                                  │
│  Hermes Agent ──▶ modelpool-pool (port 9000)     │
│                                                  │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│  hwrouter    │ │ AITOOL   │ │ (future) │
│  .185        │ │ .17      │ │          │
│              │ │          │ │          │
│ pool-worker  │ │pool-worker│ │          │
│ :9100        │ │ :9100    │ │          │
│ llama-server │ │llama-    │ │          │
│ :8080        │ │server    │ │          │
│ 2x RX 9070XT│ │:8081     │ │          │
│ 32GB VRAM   │ │9850X3D   │ │          │
└──────────────┘ └──────────┘ └──────────┘
```

Each component runs as a systemd service:
- `modelpool-pool.service` -- on Hermes host (or any central host)
- `modelpool-worker.service` -- on each inference host

## Metrics

Pool exposes Prometheus-compatible metrics:

```
modelpool_requests_total{task="compression",model="qwen36-35b-a3b-q4",worker="hwrouter"} 42
modelpool_request_duration_seconds{task="compression"} 0.5 2.1 5.8 12.4 45.0
modelpool_swaps_total{worker="hwrouter",from="qwen36-27b-q4",to="qwen36-35b-a3b-q4"} 7
modelpool_swap_duration_seconds{worker="hwrouter"} 12.0 45.0 60.0 90.0 120.0
modelpool_fallbacks_total{task="compression",reason="swap_timeout"} 2
modelpool_worker_state{worker="hwrouter"} 1  # 1=ready, 0=not
modelpool_vram_used_gb{worker="hwrouter"} 16.2
```

## Future Extensions

- **Hot standby:** Pre-load the next most likely model on idle workers
- **Batch queue:** Accumulate similar task-type requests and batch them
- **Auto-benchmark:** On first load, run a quick benchmark and record actual load/eval speeds
- **Multi-model:** Split a single worker's GPUs to run 2 small models simultaneously (e.g., 2x 9B on 32GB)
- **Priority preemption:** High-priority tasks can interrupt low-priority model loads
