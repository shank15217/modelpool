# ModelPool Implementation Tasks

## Phase 1: Foundation (Worker Agent)

### Task 1.1: Project scaffolding
- [ ] Create Python package structure (`modelpool/`)
- [ ] `pyproject.toml` with dependencies (fastapi, uvicorn, httpx, pyyaml)
- [ ] `modelpool/registry.py` -- load and validate `resources.yaml`
- [ ] `modelpool/config.py` -- load worker/pool config
- [ ] Unit tests for registry parsing and resource lookup

### Task 1.2: Worker resource loader
- [ ] `modelpool/worker/loader.py` -- execute resource recipes
- [ ] Look up resource by name in registry
- [ ] Execute the exact command from `resource.command` (binary + flags)
- [ ] No parameter generation -- run what's defined, nothing more
- [ ] Unit tests with mock llama-server binary

### Task 1.3: Worker process manager
- [ ] `modelpool/worker/process.py` -- manage llama-server subprocess
- [ ] State machine: IDLE -> LOADING -> READY -> DRAINING -> STOPPING -> ERROR
- [ ] Health check polling (GET /health on inference port until 200)
- [ ] Drain: check slots_idle via /props, wait up to drain_timeout
- [ ] Stop: SIGTERM -> wait stop_timeout -> SIGKILL
- [ ] Watchdog: detect hung/OOM'd processes, auto-recover to default resource

### Task 1.4: Worker HTTP API
- [ ] `modelpool/worker/server.py` -- FastAPI app
- [ ] `GET /worker/status` -- state, loaded resource, slots, uptime
- [ ] `POST /worker/load` -- { resource } -> drain -> stop -> start -> ready
- [ ] `POST /worker/unload` -> drain -> stop -> idle
- [ ] `GET /worker/ready` -> 200 or 503
- [ ] `POST /worker/revert` -> load default resource
- [ ] Integration tests against real llama-server

### Task 1.5: Worker systemd service
- [ ] `deploy/modelpool-worker.service` template
- [ ] Install script to deploy worker + resources.yaml to remote hosts
- [ ] Verify on hwrouter and AITOOLCHAIN

---

## Phase 2: Core (Pool Proxy)

### Task 2.1: Pool routing
- [ ] `modelpool/pool/router.py` -- resolve task_type -> resource -> worker
- [ ] Check which workers can serve the resource (from resource.workers list)
- [ ] Check if resource already loaded on any worker
- [ ] `can_fit(resource, worker)` -- size check against worker capacity

### Task 2.2: Pool request router
- [ ] Core routing logic:
  - Resource already loaded -> proxy immediately
  - Resource not loaded -> trigger load, wait, proxy
  - Load failed -> try fallback resource
  - Everything failed -> 503
- [ ] Swap behavior: queue vs fallback (per routing config)

### Task 2.3: Pool HTTP proxy
- [ ] `modelpool/pool/proxy.py` -- FastAPI reverse proxy
- [ ] `POST /v1/chat/completions` -- X-Task-Type header -> route -> proxy
- [ ] `GET /v1/models` -- aggregate loaded resources across workers
- [ ] Streaming support (SSE passthrough via httpx)
- [ ] Request/response timeout handling

### Task 2.4: Pool management API
- [ ] `GET /pool/status` -- workers, loaded resources, idle timers
- [ ] `GET /pool/routing` -- current task -> resource mapping
- [ ] `POST /pool/swap` -- manual swap for admin/testing

### Task 2.5: Idle timer management
- [ ] Per-worker asyncio timers
- [ ] Reset on each request to that resource
- [ ] On expiry: revert worker to default resource
- [ ] Cancel on manual swap or new request requiring different resource

### Task 2.6: Queue management
- [ ] Per-worker request queue during swaps
- [ ] queue behavior: wait for optimal resource (with timeout)
- [ ] fallback behavior: immediate redirect to fallback resource
- [ ] Queue timeout expiry -> try fallback or 503

---

## Phase 3: Polish

### Task 3.1: Pool systemd service
- [ ] `deploy/modelpool-pool.service`
- [ ] Run on Hermes host port 9000

### Task 3.2: Metrics
- [ ] Prometheus-compatible `/metrics` endpoint
- [ ] Request count, duration, swap count, fallback count per task type
- [ ] Worker state, VRAM usage gauges

### Task 3.3: Error handling
- [ ] Worker unreachable -> next worker, then fallback, then 503
- [ ] Swap timeout -> 503 + Retry-After
- [ ] OOM during load -> fallback resource
- [ ] Worker crash mid-swap -> detect, mark ERROR, auto-recover
- [ ] llama-server OOM -> watchdog restart with default resource

---

## Phase 4: Benchmarking & Integration

### Task 4.1: Benchmark framework
- [ ] `tests/bench/` -- standardized scripts per resource
- [ ] Test matrix: resource x context_size (4K, 30K, 70K, 100K, 170K)
- [ ] Measures: prompt eval speed, generation speed, swap time
- [ ] Output: results stored back into resource.benchmark in resources.yaml
- [ ] Command: `modelpool bench <resource_name>`

### Task 4.2: Populate real resource recipes
- [ ] Extract exact working commands from hwrouter systemd service
- [ ] Extract exact working commands from AITOOLCHAIN
- [ ] Benchmark each resource with the bench framework
- [ ] Fill in benchmark results in resources.yaml

### Task 4.3: End-to-end tests
- [ ] Load resource -> verify /worker/status correct
- [ ] Route request -> correct resource loaded -> response returned
- [ ] Swap during active request -> drain + swap + re-route
- [ ] Idle timer -> revert to default
- [ ] Worker failure -> fallback to alternate worker
- [ ] Concurrent requests during swap -> queue behavior

### Task 4.4: Hermes integration
- [ ] Point Hermes auxiliary compression at pool proxy
- [ ] Test compression triggers 35B-A3B load on hwrouter
- [ ] Verify auto-revert after idle timeout
- [ ] Document config changes

### Task 4.5: Documentation
- [ ] README with quick start
- [ ] Deployment guide
- [ ] How to add a new resource
- [ ] Benchmark results
