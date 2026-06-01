# ModelPool Implementation Plan

Reference: [ARCHITECTURE.md](ARCHITECTURE.md) for full design document.

## Completed

### Phase 1: Worker Agent ✅

- [x] Project scaffolding (pyproject.toml, src/modelpool/, tests/)
- [x] Resource registry (resources.yaml parsing, validation, lookups)
- [x] Worker subprocess manager (LlamaServerManager, state machine, drain/stop/start)
- [x] Worker watchdog (health monitoring, auto-recovery)
- [x] Worker HTTP API (FastAPI endpoints, auth middleware, async to_thread)
- [x] Worker systemd service (deployed on hwrouter and pvellm)
- [x] Worker auth (pool_secret, timing-safe hmac.compare_digest, path normalization)

### Phase 2: Pool Proxy ✅

- [x] Pool router (tag-based priority routing)
- [x] Generalist preference (loaded workhorse serves any tag)
- [x] Worker capacity (max_concurrent_models enforcement)
- [x] No-rug-pull protection (busy workers not swapped)
- [x] Pool HTTP proxy (streaming SSE, auth injection for external resources)
- [x] External resource proxy (xAO OAuth, API key injection)
- [x] Pool management API (status, routing, swap, revert)
- [x] Pool CLI and systemd service (deployed on pool proxy host)

### Phase 3: Testing ✅

- [x] 76 unit tests covering registry, router, routing policy, worker auth, review fixes
- [x] Tag resolution with mocked worker status
- [x] Generalist preference tests (serves any tag when loaded)
- [x] Capacity enforcement tests (max_concurrent_models)
- [x] Rug-pull protection tests (busy workers skipped)
- [x] Fallback chain tests (unreachable workers, all-busy error)
- [x] Worker auth tests (secret middleware, timing-safe comparison)
- [x] Code review regression tests (secret headers, double parse, async, config)

### Phase 4: Documentation ✅

- [x] README: quick start, architecture, routing rules, Hermes integration
- [x] ARCHITECTURE.md: full design doc, data flow, routing algorithm
- [x] TASKS.md: implementation status

## Remaining Work

### Testing Gaps

- [ ] Worker loader tests (state machine, subprocess lifecycle, command building)
- [ ] Worker watchdog tests (health check failure, auto-recovery)
- [ ] Pool proxy tests (streaming proxy, swap triggering, fallback logic)
- [ ] Pool server endpoint tests (status, routing, swap, revert)
- [ ] Integration tests (live worker + pool end-to-end)
- [ ] Negative/error path tests (worker 500, swap timeout, health failure cascade)
- [ ] Concurrent load request tests (two requests to same worker simultaneously)

### Performance

- [ ] Router: use httpx.AsyncClient instead of sync requests.get (currently blocks event loop)
- [ ] Router: cache worker status with TTL (currently queries workers on every resolve)
- [ ] Pool proxy: connection pool limits on httpx.AsyncClient

### Features

- [ ] Request queuing during swaps (currently returns 503 if swap needed and worker busy)
- [ ] Idle timer implementation (currently a no-op in proxy, worker handles idle_shutdown)
- [ ] Prometheus /metrics endpoint
- [ ] Structured JSON logging with trace IDs
- [ ] Auto-benchmark on first resource load

### Operational

- [ ] Rerun benchmarks on current hardware (existing results from May 28)
- [ ] Verify deployment on all workers after latest changes
- [ ] Load testing under concurrent multi-agent scenarios
