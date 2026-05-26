# ModelPool Implementation Tasks

## Phase 1: Foundation (Worker Agent)

### Task 1.1: Project scaffolding
- [ ] Create Python package structure (`modelpool/`)
- [ ] `pyproject.toml` with dependencies (fastapi, uvicorn, httpx, pyyaml)
- [ ] `modelpool/registry.py` -- load and validate `models.yaml`
- [ ] `modelpool/config.py` -- load `worker.yaml` / `pool.yaml`
- [ ] Unit tests for registry parsing and validation

### Task 1.2: Worker command builder
- [ ] `modelpool/worker/command.py` -- `build_command(model_def, worker_def)` 
- [ ] Generates correct llama-server CLI from model params + worker config
- [ ] Handles GPU params (devices, tensor-split), CPU params (threads), common params
- [ ] Unit tests with real model definitions from models.yaml

### Task 1.3: Worker process manager
- [ ] `modelpool/worker/process.py` -- start/stop/restart llama-server subprocess
- [ ] State machine: IDLE -> LOADING -> READY -> DRAINING -> STOPPING -> ERROR
- [ ] Health check polling (GET /health until 200 OK)
- [ ] Drain: check slots_idle via /props, wait up to drain_timeout
- [ ] Stop: SIGTERM -> wait stop_timeout -> SIGKILL
- [ ] Watchdog: detect hung/OOM'd processes, auto-recover to default

### Task 1.4: Worker HTTP API
- [ ] `modelpool/worker/server.py` -- FastAPI app exposing worker endpoints
- [ ] `GET /worker/status` -- state, loaded_model, vram, slots
- [ ] `POST /worker/load` -- drain -> stop -> start new model -> ready
- [ ] `POST /worker/unload` -- drain -> stop -> idle
- [ ] `GET /worker/ready` -- 200 if ready, 503 otherwise
- [ ] `POST /worker/revert` -- load default model from registry
- [ ] Integration tests against real llama-server

### Task 1.5: Worker systemd service
- [ ] `deploy/modelpool-worker.service` template
- [ ] Install script to deploy worker + config to remote hosts
- [ ] Verify on hwrouter and AITOOLCHAIN

---

## Phase 2: Core (Pool Proxy)

### Task 2.1: Pool registry + routing
- [ ] `modelpool/pool/registry.py` -- shared registry with routing table
- [ ] Resolve task_type -> { model, worker_preference, fallback, timeout, idle_revert }
- [ ] `can_fit(model, worker)` -- check model size vs worker capacity

### Task 2.2: Pool request router
- [ ] `modelpool/pool/router.py` -- core routing logic
- [ ] Resolve task -> model -> worker
- [ ] Model already loaded -> proxy immediately
- [ ] Model mismatch -> trigger swap, wait, proxy
- [ ] Swap failed -> try fallback model on fallback worker
- [ ] Everything failed -> 503

### Task 2.3: Pool HTTP proxy
- [ ] `modelpool/pool/proxy.py` -- FastAPI reverse proxy
- [ ] `POST /v1/chat/completions` -- route by X-Task-Type header, proxy to worker
- [ ] `GET /v1/models` -- aggregate models across all workers
- [ ] Streaming support (SSE passthrough)
- [ ] Request/response timeout handling

### Task 2.4: Pool management API
- [ ] `GET /pool/status` -- all workers, loaded models, idle timers
- [ ] `GET /pool/routing` -- current routing table
- [ ] `POST /pool/swap` -- manual swap for testing/admin

### Task 2.5: Idle timer management
- [ ] Per-worker idle timers (asyncio tasks)
- [ ] Reset on each request to that model
- [ ] On expiry: revert worker to default model
- [ ] Cancel timer on manual swap or new request

### Task 2.6: Queue management
- [ ] Per-worker request queue during swaps
- [ ] Queue behavior: wait for optimal model (compression, code-review)
- [ ] Fallback behavior: immediate redirect to fallback (chat)
- [ ] Queue timeout: if waited too long, try fallback or 503

---

## Phase 3: Polish

### Task 3.1: Pool systemd service
- [ ] `deploy/modelpool-pool.service`
- [ ] Config to run on Hermes host port 9000

### Task 3.2: Metrics
- [ ] Prometheus-compatible metrics endpoint
- [ ] Request count, duration, swap count, fallback count per task type
- [ ] Worker state, VRAM usage gauges

### Task 3.3: Error handling hardening
- [ ] Worker unreachable -> try next worker
- [ ] Swap timeout -> 503 + Retry-After
- [ ] OOM during load -> fallback model
- [ ] Worker crash mid-swap -> detect, mark ERROR, auto-recover
- [ ] llama-server OOM -> watchdog restart with default

---

## Phase 4: Integration & Testing

### Task 4.1: Benchmark framework
- [ ] `tests/bench/` -- standardized benchmark scripts
- [ ] Test matrix: model x task x context_size
- [ ] Measures: prompt eval speed, generation speed, total latency, swap time
- [ ] Output: markdown table + JSON results file

### Task 4.2: End-to-end tests
- [ ] Start worker, load default model, verify /worker/status
- [ ] Send task request -> verify correct model loaded -> verify response
- [ ] Model swap during active request -> verify drain + swap + re-route
- [ ] Idle timer expiry -> verify revert to default
- [ ] Worker failure -> verify fallback to alternate worker
- [ ] Concurrent requests during swap -> verify queue behavior

### Task 4.3: Hermes integration
- [ ] Configure Hermes auxiliary task to point at pool proxy
- [ ] Test compression task triggers 35B-A3B load on hwrouter
- [ ] Verify auto-revert after idle timeout
- [ ] Document Hermes config changes

### Task 4.4: Documentation
- [ ] README with quick start
- [ ] Deployment guide (worker install, pool install, Hermes config)
- [ ] Benchmark results
- [ ] Troubleshooting guide
