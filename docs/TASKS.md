# ModelPool Implementation Plan

Reference: [ARCHITECTURE.md](ARCHITECTURE.md) for full design document.

## Phase 1: Worker Agent

The worker agent is a Python HTTP server running on each inference host. It manages llama-server as a child process -- starts, stops, monitors, and reports status.

### 1.1 Project scaffolding

Create the Python package and build system.

- [ ] Create directory structure:
  ```
  modelpool/
  ├── pyproject.toml
  ├── resources.yaml
  ├── src/
  │   └── modelpool/
  │       ├── __init__.py
  │       ├── registry.py       # load/validate resources.yaml
  │       ├── config.py          # load worker.yaml, pool.yaml
  │       ├── worker/
  │       │   ├── __init__.py
  │       │   ├── loader.py      # subprocess management
  │       │   ├── server.py      # FastAPI worker API
  │       │   └── watchdog.py    # background health monitor
  │       └── pool/
  │           ├── __init__.py
  │           ├── router.py      # task -> resource resolution
  │           ├── proxy.py       # HTTP reverse proxy
  │           └── server.py      # FastAPI pool API
  ├── tests/
  │   ├── unit/
  │   ├── integration/
  │   └── bench/
  └── deploy/
      ├── modelpool-worker.service
      └── modelpool-pool.service
  ```
- [ ] `pyproject.toml`: dependencies (fastapi, uvicorn, httpx, pyyaml), entry points
  - `modelpool-worker` CLI command -> starts worker agent
  - `modelpool-pool` CLI command -> starts pool proxy
- [ ] Verify: `pip install -e .` works, both CLIs print help

### 1.2 Resource registry

Load and validate `resources.yaml`.

- [ ] `registry.py`: parse resources.yaml into Python dataclasses
  - `Resource` dataclass: name, type (managed/external), command, workers, capabilities, benchmark
  - `Worker` dataclass: host, ports, type, limits, default_resource
  - `Route` dataclass: task_type, resource, fallback chain, timeout, idle_revert, swap_behavior
- [ ] Validate on load:
  - All resources referenced in routing actually exist
  - All workers referenced in resources actually exist
  - Managed resources have a `command` section
  - External resources have `endpoint` and `auth`
  - No resource exceeds its worker's max_model_gb
- [ ] Lookup methods:
  - `get_resource(name) -> Resource`
  - `get_worker(name) -> Worker`
  - `get_route(task_type) -> Route`
  - `get_default_resource(worker_name) -> Resource`
  - `get_resources_for_worker(worker_name) -> list[Resource]`
- [ ] Unit tests: parse real resources.yaml, validate lookups, test validation errors

### 1.3 Worker subprocess manager

The core of managed resource lifecycle.

- [ ] `loader.py`: `LlamaServerManager` class
  - `start(resource_name)`:
    1. Look up resource in registry
    2. Build command: `[binary] + flatten(flags) + replace {inference_port}`
    3. Launch: `subprocess.Popen(cmd, stdout=log, stderr=STDOUT, preexec_fn=os.setsid)`
    4. Poll `GET http://localhost:{port}/health` every 2s until 200 OK
    5. On success: state = ready, loaded_resource = name
    6. On timeout: SIGKILL, state = error
  - `stop(timeout=10)`:
    1. `os.killpg(pgid, SIGTERM)` to process group
    2. `process.wait(timeout=10)`
    3. If still alive: `os.killpg(pgid, SIGKILL)`, wait 5s
    4. Set process = None, state = idle
  - `drain(timeout=30)`:
    1. Poll `GET http://localhost:{port}/props`
    2. Wait until `slots_processing == 0` or timeout
  - `load_resource(resource_name)`:
    1. If currently ready: drain -> stop -> start new
    2. If idle: start directly
    3. If loading/d raining: reject with 409 Conflict
  - `get_status() -> dict`: state, loaded_resource, pid, slots, uptime
  - `unload()`: drain -> stop -> state = idle
  - `revert()`: unload -> start default resource
- [ ] State machine enforcement (IDLE, LOADING, READY, DRAINING, STOPPING, ERROR)
  - Reject invalid transitions (e.g., load while already loading)
- [ ] Log output: capture subprocess stdout/stderr to `/var/log/modelpool/llama-server.log`
- [ ] Unit tests with mock subprocess (verify command building, state transitions)

### 1.4 Worker watchdog

Background health monitor for the managed llama-server.

- [ ] `watchdog.py`: asyncio background task
  - Every 15s, if state == ready: GET /health
  - If health check fails 3 times in a row: state = error
  - On error: stop broken process, load default resource
  - Log all state transitions
- [ ] Configurable check interval and failure threshold
- [ ] Unit tests: simulate health check failures, verify auto-recovery

### 1.5 Worker HTTP API

FastAPI server exposing the worker management endpoints.

- [ ] `server.py`: FastAPI app with endpoints:
  - `GET /worker/status` -> state, loaded_resource, slots, pid, uptime
  - `POST /worker/load` -> { resource: str } -> drain -> stop -> start -> 200
  - `POST /worker/unload` -> drain -> stop -> idle
  - `GET /worker/ready` -> 200 if ready, 503 otherwise
  - `POST /worker/revert` -> load default resource
- [ ] Request validation (resource exists, worker can serve it, size fits)
- [ ] Error responses:
  - 404: resource not found in registry
  - 409: worker busy (already loading/draining)
  - 422: resource not compatible with this worker
  - 503: swap failed (health check timeout)
- [ ] `modelpool-worker` CLI: argparse entry point
  - `--config worker.yaml` (default: ./worker.yaml)
  - `--registry resources.yaml` (default: ./resources.yaml)
  - Starts uvicorn with the FastAPI app
- [ ] Integration tests: start real llama-server, verify load/unload/status cycle

### 1.6 Worker systemd service

Deploy the worker as a system service.

- [ ] `deploy/modelpool-worker.service`:
  ```ini
  [Unit]
  Description=ModelPool Worker Agent
  After=network.target

  [Service]
  Type=simple
  User=root
  WorkingDirectory=/opt/modelpool
  ExecStart=/opt/modelpool/venv/bin/modelpool-worker --config /etc/modelpool/worker.yaml
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```
- [ ] `deploy/install-worker.sh`: install script
  - Copy source to `/opt/modelpool/`
  - Create venv, install package
  - Copy resources.yaml to `/etc/modelpool/`
  - Generate worker.yaml from args (worker_id, ports)
  - Enable and start systemd service
- [ ] Test on hwrouter: install, verify worker starts, verify status endpoint
- [ ] Test on AITOOLCHAIN: same

---

## Phase 2: Pool Proxy

The pool sits between Hermes and the workers. Routes by task type, manages swaps.

### 2.1 Pool router

Resolve task types to resources and workers.

- [ ] `router.py`: core routing logic
  - `resolve(task_type) -> (resource, worker, fallback_chain)`
  - Check which workers can serve the resource (from resource.workers list)
  - Check if resource already loaded on any matching worker
  - If loaded: return immediately
  - If not: pick best worker (fewest pending swaps, most capacity)
  - Build fallback chain from routing config
- [ ] `can_fit(resource, worker)` -- size check
- [ ] `pick_worker(resource) -> worker` -- preference + availability
- [ ] Unit tests: verify routing resolution with various scenarios

### 2.2 Pool HTTP proxy

Reverse proxy from clients to workers/external endpoints.

- [ ] `proxy.py`: FastAPI reverse proxy
  - `POST /v1/chat/completions`:
    1. Read `X-Task-Type` header (default: "chat")
    2. Resolve to resource + worker
    3. If managed + already loaded: proxy to worker inference port
    4. If managed + needs swap: trigger load, wait, then proxy
    5. If external: inject auth, proxy to endpoint
    6. If swap fails: try fallback chain
    7. Stream response back to client (SSE passthrough)
  - `GET /v1/models`: aggregate loaded resources across all workers
- [ ] Streaming: use `httpx.AsyncClient.stream()` with `aiter_bytes()`
- [ ] Timeout handling: per-route timeout from routing config
- [ ] Error responses: 502 (worker error), 503 (no available worker), 504 (timeout)

### 2.3 External resource proxy

Handle cloud API routing with auth injection.

- [ ] Auth injection for external resources:
  - `xai-oauth`: read from Hermes auth store, refresh if needed
  - `api_key`: read from env var
  - `none`: pass through
- [ ] Request transformation for external:
  - Override `model` field in request body with resource's model name
  - Inject `Authorization` header
  - Proxy to resource.endpoint
- [ ] No lifecycle management -- just proxy and return
- [ ] Unit tests: mock external endpoints, verify auth injection

### 2.4 Pool management API

Admin and status endpoints.

- [ ] `GET /pool/status`:
  ```json
  {
    "workers": {
      "hwrouter": { "state": "ready", "resource": "qwen36-27b_mtp_vision_multi-gpu" },
      "cloud-xai": { "state": "external", "available": true }
    },
    "idle_timers": {
      "hwrouter": { "resource": "qwen36-35b-a3b_mtp_no-reasoning_multi-gpu", "expires_in_s": 142 }
    }
  }
  ```
- [ ] `GET /pool/routing`: current task -> resource mapping with fallback chains
- [ ] `POST /pool/swap`: { worker, resource } -- manual swap for admin/testing
- [ ] `POST /pool/revert`: { worker } -- revert to default

### 2.5 Idle timer management

Auto-revert workers to default resource after inactivity.

- [ ] Per-worker asyncio timers
- [ ] On successful swap to non-default resource: start timer with route's `idle_revert` seconds
- [ ] On each request to that resource: reset timer
- [ ] On timer expiry: `POST /worker/revert` to the worker
- [ ] On manual swap or new request requiring different resource: cancel timer
- [ ] `idle_revert: 0` means never revert (default resource)
- [ ] Pool status shows active timers and remaining time

### 2.6 Queue management

Handle requests arriving during model swaps.

- [ ] Per-worker request queue (asyncio.Queue)
- [ ] `swap_behavior: queue`:
  - Request enters queue, waits for resource to be loaded
  - If queue_timeout expires: try fallback resource
  - When resource ready: process queued requests in order
- [ ] `swap_behavior: fallback`:
  - Skip queue, immediately try fallback resource
  - No waiting, lower quality but instant response
- [ ] Multiple concurrent requests for same resource: coalesce (single swap, all wait)
- [ ] Unit tests: verify queue/fallback behaviors

### 2.7 Pool CLI and service

Entry point and systemd service for the pool.

- [ ] `modelpool-pool` CLI:
  - `--config pool.yaml`
  - `--registry resources.yaml`
  - Starts uvicorn on configured port (default 9000)
- [ ] `deploy/modelpool-pool.service`: systemd unit
- [ ] `deploy/install-pool.sh`: install script

---

## Phase 3: Polish

### 3.1 Error handling

- [ ] Worker unreachable: try next worker in preference list
- [ ] Swap timeout: 503 + Retry-After header
- [ ] OOM during load: detect from process exit code, try fallback resource
- [ ] Worker crash mid-swap: detect from connection refused, mark ERROR
- [ ] llama-server OOM at runtime: watchdog detects, restarts with default
- [ ] External endpoint 401/403: log warning, try next fallback
- [ ] All fallbacks exhausted: 503 with descriptive error

### 3.2 Metrics

- [ ] Prometheus-compatible `/metrics` endpoint on pool
- [ ] Counters: `modelpool_requests_total{task, resource, worker, status}`
- [ ] Histograms: `modelpool_request_duration_seconds{task}`
- [ ] Counters: `modelpool_swaps_total{worker, from, to}`
- [ ] Histograms: `modelpool_swap_duration_seconds{worker}`
- [ ] Counters: `modelpool_fallbacks_total{task, reason}`
- [ ] Gauges: `modelpool_worker_state{worker}`, `modelpool_vram_used_gb{worker}`

### 3.3 Logging

- [ ] Structured JSON logging
- [ ] Per-request trace IDs
- [ ] Log all state transitions (worker, swap, route resolution)
- [ ] Log file rotation via config

---

## Phase 4: Benchmarking & Integration

### 4.1 Benchmark framework

- [ ] `modelpool bench <resource_name>` CLI command
  - Starts the resource on the target worker
  - Runs test prompts at multiple context sizes: 4K, 30K, 70K, 100K, 170K
  - Measures: prompt eval tok/s, generation tok/s, total latency, time-to-first-token
  - Outputs results as JSON
- [ ] `modelpool bench --all`: benchmark all managed resources
- [ ] `modelpool bench --update`: write results back into resources.yaml benchmark section
- [ ] Test scripts in `tests/bench/`

### 4.2 Populate real resource recipes

- [ ] Extract exact working command from hwrouter's current llama-server.service
- [ ] Extract exact working command from AITOOLCHAIN's current process
- [ ] Add xAI Grok external resource definition
- [ ] Add GLM-4.5-Flash external resource definition
- [ ] Benchmark each resource with the framework
- [ ] Fill in benchmark results in resources.yaml

### 4.3 End-to-end tests

- [ ] Start worker, load default resource, verify /worker/status
- [ ] Route compression request -> verify 35B-A3B loaded -> verify response
- [ ] Swap during active request -> verify drain + swap + re-route
- [ ] Idle timer expiry -> verify revert to default resource
- [ ] Worker failure -> verify fallback to alternate worker/resource
- [ ] External resource request -> verify auth injection + proxy
- [ ] Concurrent requests during swap -> verify queue behavior
- [ ] All workers down -> verify 503 with descriptive error

### 4.4 Hermes integration

- [ ] Configure Hermes auxiliary compression to point at pool proxy
- [ ] Test: send compression request, verify pool routes to correct resource
- [ ] Test: verify idle revert doesn't break subsequent requests
- [ ] Document config changes needed in Hermes config.yaml
- [ ] Update wiki-infra with modelpool deployment docs

### 4.5 Documentation

- [ ] README: quick start, architecture overview
- [ ] docs/DEPLOY.md: step-by-step deployment guide
- [ ] docs/ADDING-RESOURCES.md: how to add a new resource
- [ ] docs/BENCHMARKS.md: benchmark results for all resources
- [ ] docs/HERMES-INTEGRATION.md: configuring Hermes to use modelpool
- [ ] docs/TROUBLESHOOTING.md: common issues and fixes
