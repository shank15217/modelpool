# ModelPool Architecture

## Core Concept

A **resource** is a fully configured model recipe -- the exact command to launch an inference server, tuned for specific hardware and a specific use case. Resources come in two flavors:

- **Managed resources** -- local llama-server processes the worker starts, monitors, and stops
- **External resources** -- cloud APIs or pre-existing endpoints the pool can route to

The pool routes by task type, resolves to a resource, and either manages the lifecycle (managed) or just proxies (external). Clients don't know the difference.

```
Resource = Name + Exact launch command (or external endpoint) + Hardware target + Use case tags + Benchmarks
```

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client     │     │              ModelPool                       │
│   (Hermes)   │────▶│                                              │
│              │     │  Router ──▶ Resource Registry ──▶ Workers    │
└─────────────┘     │                                              │
                    │  ┌─────────────┐  ┌──────────────────────┐   │
                    │  │  Managed    │  │  External            │   │
                    │  │  (local GPU)│  │  (cloud API)         │   │
                    │  │  lifecycle  │  │  proxy only           │   │
                    │  └─────────────┘  └──────────────────────┘   │
                    └──────────────────────────────────────────────┘
                              │                    │
                    ┌─────────▼──────┐   ┌─────────▼──────────┐
                    │  hwrouter      │   │  xAI cloud         │
                    │  llama-server  │   │  api.x.ai/v1       │
                    │  :8080         │   │  (already running)  │
                    └────────────────┘   └────────────────────┘
```

## Current Resources

### Managed Resources (hwrouter: 2x RX 9070 XT, 32GB VRAM)

| Resource | Model | Ctx | Reasoning | Parallel | Best For | Prompt Eval | Generation |
|---|---|---|---|---|---|---|---|
| `qwen36-27b_mtp_reasoning_multi-gpu` | Qwen3.6-27B MTP Q4_K_M | 131K | ON | 2 slots | Coding agents, chat, agentic work | 728 t/s | 33.8 t/s |
| `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu` | Qwen3.6-35B-A3B MoE (3B active) MTP Q4_K_M | 262K | OFF | 1 slot | Context compression, summarization | 2,225 t/s | 71.4 t/s |

### Managed Resources (AITOOLCHAIN: AMD 9850X3D, 48GB RAM, ik_llama.cpp)

| Resource | Model | Ctx | Reasoning | Best For | Prompt Eval | Generation |
|---|---|---|---|---|---|---|
| `qwen36-35b-a3b_no-reasoning_cpu` | Qwen3.6-35B-A3B MoE (3B active) MTP Q4_K_M | 262K | OFF | Title gen, triage, summarize, code review | 473 t/s | 39.4 t/s |

### External Resources (Cloud)

| Resource | Provider | Model | Ctx | Auth | Best For |
|---|---|---|---|---|---|
| `grok-4.3_general` | xAI | grok-4.3 | 256K | OAuth | General fallback, 256K ctx |
| `glm-45-flash_general` | Z.ai | glm-4.5-flash | 131K | API key (free) | Triage, small compression, summarize |

### Model Selection Guide

| Task | Primary Resource | Why | Swap Time |
|---|---|---|---|
| Coding / agentic work | `qwen36-27b_mtp_reasoning_multi-gpu` | Reasoning ON, 2 parallel slots, 33.8 t/s gen | Default (always loaded) |
| Context compression | `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu` | 262K ctx, 2225 t/s prompt eval, 71.4 t/s gen | ~8s swap |
| Title gen / triage | `qwen36-35b-a3b_no-reasoning_cpu` | Free CPU, 473 t/s prompt eval, sub-10s response | N/A (always on) |
| Triage / small tasks | `glm-45-flash_general` | Free cloud, fast, no GPU needed | N/A (external) |
| Large compression fallback | `grok-4.3_general` | 256K ctx cloud, no swap needed | N/A (external) |

### Swap Performance

Swapping between 27B and 35B-A3B on hwrouter (drain + stop + load + health check):

```
27B -> 35B-A3B: ~8s (model fits in GPU VRAM cache, fast load)
35B-A3B -> 27B: ~8s (same)
```

## How a Managed Resource Works

### Step 1: Define the resource in `resources.yaml`

Take the exact command you tested manually and put it in the resource definition:

```yaml
resources:
  qwen36-35b-a3b_mtp_no-reasoning_multi-gpu:
    description: "35B MoE, MTP speculative, reasoning off, both GPUs"
    type: managed
    size_gb: 21
    ctx: 262144
    capabilities: [compression, summarize, chat]
    workers: [hwrouter]
    benchmark:
      prompt_eval_tps: 2224.8
      generation_tps: 71.4
      tested_at: "2026-05-27"
    command:
      binary: /AITOOLCHAIN/llama.cpp/build/bin/llama-server
      flags:
        - [-m, "/AITOOLCHAIN/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf"]
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
        - [--port, "{inference_port}"]       # template: replaced with worker's port
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
```

The command is the exact thing you ran and tested. Flags are stored as a list of `[flag, value]` pairs (or `[flag]` for boolean flags). The only templating is `{inference_port}` -- everything else is literal.

### Step 2: Worker receives a load request

```
POST http://hwrouter:9100/worker/load
{ "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }
```

### Step 3: Worker executes the lifecycle

```
Worker state machine during load:

  READY (running qwen36-27b_mtp_reasoning_multi-gpu)
    │
    ▼
  DRAINING
    │  GET http://localhost:8080/health -> check slots_processing == 0
    │  Wait up to 30s for in-flight requests to finish
    │  If timeout: force stop anyway
    ▼
  STOPPING
    │  Send SIGTERM to llama-server process group
    │  Wait up to 10s for clean exit
    │  If still running: SIGKILL
    │  Verify port 8080 is released
    ▼
  LOADING
    │  Build command line from resource definition:
    │    /AITOOLCHAIN/llama.cpp/build/bin/llama-server \
    │      -m /AITOOLCHAIN/models/.../Qwen3.6-35B-A3B-UD-MTP-Q4_K_M.gguf \
    │      -c 262144 -ngl 99 -fa on ... --port 8080 --host 0.0.0.0 ...
    │
    │  Launch as subprocess (no shell, direct exec)
    │  Capture stdout/stderr to log file
    │
    │  Poll health: GET http://localhost:8080/health
    │  Every 2s, up to 120s startup timeout
    │
    │  Health check returns 200 after ~8s
    ▼
  READY (running qwen36-35b-a3b_mtp_no-reasoning_multi-gpu)
    Report to pool: resource loaded, ready for requests
```

### Step 4: Pool proxies requests

```
Client -> Pool (port 9000) -> Worker (port 8080) -> llama-server
                                 ^
                                 routes OpenAI /v1/chat/completions
```

The pool proxies the raw HTTP request to the worker's inference port. llama-server serves it directly. Responses stream back through the pool to the client.

### Step 5: Shutdown / revert

When the idle timer expires or pool requests a revert:

```
  READY (running qwen36-35b-a3b)
    │
    ▼
  DRAINING -> STOPPING -> LOADING (default resource) -> READY
```

The worker goes through the same drain -> stop -> start cycle to load the default resource.

### Why subprocess, not systemd?

The worker IS a systemd service (`modelpool-worker.service`). It's a long-running Python process that manages llama-server as child processes:

```
systemd
  └── modelpool-worker (Python, always running)
        ├── HTTP API on :9100 (management)
        └── child process management
              ├── llama-server (port 8080)  ← managed subprocess
              └── (only one at a time)
```

Benefits of subprocess over restarting systemd services:
- **No privilege escalation** -- worker runs as a user, doesn't need systemd reload
- **Faster** -- no systemd overhead on start/stop
- **Direct control** -- SIGTERM/SIGKILL, stdout/stderr capture, exit code checking
- **Portable** -- works on any Linux, no systemd-specific APIs

The worker process itself is managed by systemd (auto-restart on crash). The llama-server processes are managed by the worker (start/stop on demand).

## External Resources

Cloud APIs and pre-existing endpoints. No lifecycle management -- the pool just proxies.

```yaml
resources:
  grok-4.3_general:
    description: "Grok 4.3 via xAI OAuth, 256K ctx, cloud"
    type: external
    endpoint: https://api.x.ai/v1
    auth:
      method: xai-oauth       # uses Hermes auth store
    model: grok-4.3           # model name to send in requests
    ctx: 256000
    capabilities: [chat, compression, agentic, vision]
    workers: [cloud-xai]
    benchmark:
      prompt_eval_tps: null   # fill after testing
      generation_tps: null

  glm-45-flash_general:
    description: "GLM-4.5-Flash free tier, 131K ctx"
    type: external
    endpoint: https://open.bigmodel.cn/api/paas/v4
    auth:
      method: api_key
      env_var: LM_API_KEY
    model: glm-4.5-flash
    ctx: 131072
    capabilities: [chat, compression, summarize, triage]
    workers: [cloud-zai-free]
    benchmark:
      prompt_eval_tps: null
      generation_tps: null
```

External resources don't need a `command` section. They need:
- `endpoint` -- base URL for the API
- `auth` -- how to authenticate (OAuth reference, env var name, or static key)
- `model` -- model name to pass in the request body

The pool handles auth injection when proxying to external resources -- it reads the auth config, gets the current credentials, and adds them to the proxied request.

## Workers

Workers are targets that can serve resources. Two types:

```yaml
workers:
  # Managed: the worker agent runs on this host and controls llama-server
  hwrouter:
    host: 192.168.35.185
    worker_port: 9100          # worker agent API
    inference_port: 8080       # llama-server port
    type: managed
    vram_gb: 32
    max_model_gb: 28
    swap_timeout: 120
    drain_timeout: 30
    default_resource: qwen36-27b_mtp_reasoning_multi-gpu

  aitooolchain:
    host: 192.168.35.17
    worker_port: 9100
    inference_port: 8081
    type: managed
    ram_gb: 48
    max_model_gb: 40
    swap_timeout: 60
    drain_timeout: 10
    default_resource: qwen36-35b-a3b_no-reasoning_cpu

  # External: no worker agent, just an API endpoint
  cloud-xai:
    type: external
    # No host/ports -- the resource defines the endpoint

  cloud-zai-free:
    type: external
```

Managed workers run the worker agent. External workers are virtual -- they exist only in the routing table so the pool can direct traffic to them.

## Routing

Maps task types to resources. The pool tries the primary resource first, falls back if unavailable.

```yaml
routing:
  compression:
    resource: qwen36-35b-a3b_mtp_no-reasoning_multi-gpu
    fallback_resource: glm-45-flash_general     # free cloud first
    fallback_resource_2: grok-4.3_general       # then xAI
    fallback_resource_3: qwen36-35b-a3b_no-reasoning_cpu  # last resort: local CPU
    timeout: 120
    idle_revert: 300
    swap_behavior: queue

  chat:
    resource: qwen36-27b_mtp_reasoning_multi-gpu
    fallback_resource: grok-4.3_general
    timeout: 120
    idle_revert: 0                              # default resource, never revert
    swap_behavior: fallback
```

Fallback chain for compression:
```
1. Local 35B-A3B on GPU (free, 2225 t/s prompt eval, 262K ctx)
2. GLM-4.5-Flash cloud (free, fast, but 131K ctx limit)
3. Grok 4.3 cloud (OAuth, 256K ctx, subscription)
4. Local 35B-A3B on CPU (free, 473 t/s prompt eval, ik_llama.cpp)
```

The pool tries each in order. If a managed resource needs a swap, it waits (queue behavior). If a cloud resource fails, it immediately tries the next fallback.

## Data Flow (Full Example)

```
1. Hermes sends: POST http://localhost:9000/v1/chat/completions
   Header: X-Task-Type: compression
   Body: { messages: [...], max_tokens: 4096 }

2. Pool looks up routing: compression -> qwen36-35b-a3b_mtp_no-reasoning_multi-gpu

3. Pool checks: which worker serves this resource? -> hwrouter

4. Pool checks hwrouter: GET http://hwrouter:9100/worker/status
   Response: { resource: "qwen36-27b_mtp_reasoning_multi-gpu", state: "ready" }

5. Wrong resource loaded. Pool triggers swap:
   POST http://hwrouter:9100/worker/load
   { "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }
   Response: 202 { status: "loaded", resource: "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }

6. Worker lifecycle (~8s):
   - Drains active slots (30s max)
   - Stops llama-server (SIGTERM, 10s, then SIGKILL)
   - Builds command from resource flags
   - Starts subprocess: /AITOOLCHAIN/llama.cpp/build/bin/llama-server -m ... --port 8080 ...
   - Polls GET http://localhost:8080/health every 2s
   - Health returns 200 after ~8s
   - Reports: { state: "ready", resource: "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu" }

7. Pool proxies the original request:
   POST http://hwrouter:8080/v1/chat/completions
   Body: { messages: [...], max_tokens: 4096 }
   (streaming response passes through pool back to Hermes)

8. Pool starts idle timer: 300s
   No compression requests for 5 minutes -> revert to qwen36-27b default
```

## Worker Internals

### Subprocess Management

The `LlamaServerManager` class manages a single llama-server subprocess:

- **start(resource)**: Builds command from resource flags, launches subprocess via `Popen` with `os.setsid` for process group isolation, polls `/health` until 200 OK
- **stop(timeout=10)**: SIGTERM to process group, wait, SIGKILL if needed
- **drain(timeout=30)**: Polls `/health` for `slots_processing == 0`
- **load_resource(resource)**: Full drain -> stop -> start cycle
- **revert(registry, worker_name)**: Unload and start the worker's default resource
- **get_status()**: Returns state, loaded_resource, pid, uptime, slot info

State machine: `IDLE -> LOADING -> READY -> DRAINING -> STOPPING -> IDLE` (or ERROR at any point)

### Watchdog

Background asyncio task that monitors llama-server health every 15s. On 3 consecutive failures, it marks the worker as ERROR and auto-recovers by restarting with the default resource.

### Worker HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/worker/status` | GET | Current state, loaded resource, pid, uptime, slots |
| `/worker/load` | POST | Load a resource (drain -> stop -> start) |
| `/worker/unload` | POST | Drain and stop, leave idle |
| `/worker/ready` | GET | 200 if ready, 503 otherwise |
| `/worker/revert` | POST | Revert to default resource |

## Auth for External Resources

The pool needs to inject credentials when proxying to external resources. Auth methods:

| Method | How it works |
|---|---|
| `xai-oauth` | Read tokens from `~/.hermes/auth.json`, refresh if expired, inject as `Authorization: Bearer ***` |
| `api_key` | Read key from env var, inject as `Authorization: Bearer ***` |
| `none` | No auth (local endpoints) |

## Deployment

```
Hermes Host (192.168.35.x)
├── modelpool-pool (port 9000)     # systemd service (Phase 2, not yet implemented)
└── resources.yaml                  # shared config

Managed Workers
├── hwrouter (192.168.35.185)
│   ├── modelpool-worker (port 9100)  # systemd service -- ACTIVE
│   ├── llama-server (port 8080)      # managed subprocess
│   └── resources.yaml                # copy at /etc/modelpool/resources.yaml
└── aitooolchain (192.168.35.17)
    ├── modelpool-worker (port 9100)  # not yet deployed
    ├── llama-server (port 8081)      # managed subprocess
    └── resources.yaml

External Workers (no agent needed)
├── cloud-xai (api.x.ai)
└── cloud-zai-free (open.bigmodel.cn)
```

`resources.yaml` is the same file everywhere. Each worker reads only its own section.

## Security

- Managed workers: firewall worker port (9100) to pool host only
- Inference ports (8080/8081): firewall to pool host only
- External resources: auth tokens never logged, refreshed automatically
- Pool: firewall to Hermes host only
- No auth between pool and workers (homelab, firewalled)

## Adding a New Resource

1. SSH to the target worker
2. Test the llama-server command manually -- get it working perfectly
3. Benchmark it: `modelpool bench` or manual timing
4. Add the exact working command as a new resource in `resources.yaml`
5. Add routing entry mapping a task type to the new resource
6. Deploy updated `resources.yaml` to pool + workers
7. Test: `curl -H "X-Task-Type: <task>" http://pool:9000/v1/chat/completions -d '...'`

No parameter generation. No interpolation (except `{inference_port}`). What you tested is what runs.

## Future Extensions

- **Hot standby:** Pre-load the next most likely resource on idle workers
- **Auto-benchmark:** On first load, run a quick benchmark, store results in registry
- **Multi-model:** Split GPUs to run two small resources simultaneously
- **Priority preemption:** High-priority tasks interrupt low-priority loads
- **Resource versioning:** Track command changes, re-benchmark on update
