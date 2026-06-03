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
- [x] External resource proxy (xAI OAuth, API key injection)
- [x] Pool management API (status, routing, swap, revert)
- [x] Pool CLI and systemd service (deployed on pool proxy host)
- [x] Async router (httpx.AsyncClient, status cache with TTL)

### Phase 3: Testing ✅

- [x] 200 unit tests covering all core components
- [x] Registry + router + routing policy tests
- [x] Worker loader (54 tests): state machine, command building, lifecycle
- [x] Worker watchdog (14 tests): health checks, auto-recovery
- [x] Worker server (27 tests): endpoints, middleware, auth
- [x] Pool proxy (18 tests): routing, streaming, auth injection
- [x] Pool server (13 tests): endpoints, idle timer
- [x] Code review regression tests (13 tests)
- [x] All tests converted to async (pytest-asyncio) for async router

### Phase 4: Documentation ✅

- [x] README: quick start, architecture, routing rules, Hermes integration
- [x] ARCHITECTURE.md: full design doc, data flow, async router, CPU inference, 4-GPU plan
- [x] TASKS.md: implementation status

## Remaining Work

### Operational

- [ ] Redeploy latest code to workers + proxy (async router, security fixes)
- [ ] Rerun benchmarks on current hardware
- [ ] Load testing under concurrent multi-agent scenarios

### Features

- [ ] Request queuing during swaps (currently returns 503 if swap needed and worker busy)
- [ ] Idle timer implementation in proxy (currently no-op, worker handles idle_shutdown)
- [ ] Prometheus /metrics endpoint
- [ ] Structured JSON logging with trace IDs
- [ ] Auto-benchmark on first resource load

### 4-GPU Expansion (planned)

- [ ] Provision gpu-dense server (2x RX 9700 PRO 32GB)
- [ ] Provision gpu-sparse-1 and gpu-sparse-2 servers (1x RX 9070 XT 16GB each)
- [ ] Tune partial GPU offload for 16GB cards (-ngl, thread count, cache sizes)
- [ ] Benchmark 27B dense with 4 slots on 64GB VRAM
- [ ] Benchmark 35B MoE with CPU offload on 16GB cards
- [ ] Update resources.yaml with new workers and resources
- [ ] Verify routing across 4 GPU + 1 CPU + cloud resources
