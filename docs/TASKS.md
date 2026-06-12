# ModelPool Implementation Plan

Reference: [ARCHITECTURE.md](ARCHITECTURE.md) for full design document.

## Architecture

**Architecture A: Static Pool** -- each worker runs one model, started at boot, served forever. Pool proxy routes by tag priority with failover. No dynamic routing, no swapping.

Dynamic pool code (Architecture B) preserved on `arch/dynamic-pool` branch for future use.

## Completed

### Core ✅

- [x] Project scaffolding (pyproject.toml, src/modelpool/, tests/)
- [x] Resource registry (resources.yaml parsing, validation, lookups)
- [x] Tag-based priority router (sync, in-memory, zero HTTP calls)
- [x] Pool HTTP proxy (streaming SSE, auth injection, failover)
- [x] External resource proxy (xAI OAuth, API key injection)
- [x] Pool management API (/pool/status, /pool/routing)
- [x] Pool CLI and systemd service

### Worker ✅

- [x] Subprocess manager (start/stop llama-server at boot)
- [x] Worker HTTP API (/worker/status, /worker/ready)
- [x] Worker auth (pool_secret, timing-safe hmac.compare_digest)
- [x] Worker CLI and systemd service
- [x] Health monitoring (watchdog)

### Testing ✅

- [x] 149 unit tests covering all components
- [x] Registry tests (parsing, validation, lookups)
- [x] Router tests (tag resolution, priority ordering)
- [x] Proxy tests (streaming, auth injection, failover)
- [x] Worker tests (subprocess lifecycle, command building, endpoints)
- [x] Watchdog tests (health checks)
- [x] Pool server tests (endpoints)

### Documentation ✅

- [x] README: quick start, architecture, routing rules, Hermes integration
- [x] ARCHITECTURE.md: design philosophy, routing algorithm, 4-GPU expansion plan
- [x] TASKS.md: implementation status

## Remaining Work

### Operational

- [ ] Deploy workers with assigned models on all hardware
- [ ] Deploy pool proxy and verify routing
- [ ] Verify Hermes integration end-to-end
- [ ] Rerun benchmarks on 9700 PRO hardware

### 4-GPU Expansion

- [ ] Provision gpu-dense server (2x RX 9700 PRO 32GB)
- [ ] Provision gpu-sparse-1 and gpu-sparse-2 servers (1x RX 9070 XT 16GB each)
- [ ] Benchmark 27B dense with 4 slots on 64GB VRAM
- [ ] Benchmark 35B MoE with CPU offload on 16GB cards
- [ ] Update resources.yaml with new workers and resources
- [ ] Verify routing across all tiers

### Optional Enhancements

- [ ] Prometheus /metrics endpoint
- [ ] Structured JSON logging with trace IDs
- [ ] Request queuing when all tiers are busy
- [ ] Round-robin for same-tier workers
