# modelpool

On-demand LLM resource orchestration for homelab GPU clusters.

A **resource** is a fully configured model recipe -- the GGUF weights file plus every llama-server launch flag, tuned for specific hardware and a specific use case. ModelPool routes tasks to the right resource and swaps models in and out of GPU memory as needed.

## Current Resources

### hwrouter (2x RX 9070 XT, 32GB VRAM, ROCm)

| Resource | Model | Ctx | Speed (prompt/gen) | Best For |
|---|---|---|---|---|
| `qwen36-27b_mtp_reasoning_multi-gpu` | 27B MTP Q4_K_M, reasoning ON | 131K | 728 / 33.8 t/s | Coding agents, agentic work (default) |
| `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu` | 35B MoE (3B active), reasoning OFF | 262K | 2,225 / 71.4 t/s | Context compression |

Swap time between resources: ~8 seconds.

### pvellm (AMD 9850X3D, 48GB RAM, ik_llama.cpp)

| Resource | Model | Ctx | Speed (prompt/gen) | Best For |
|---|---|---|---|---|
| `qwen36-35b-a3b_no-reasoning_cpu` | 35B MoE (3B active), reasoning OFF | 262K | 473 / 39.4 t/s | Title gen, triage, summarize (always on) |

### Cloud (External)

| Resource | Provider | Ctx | Auth | Best For |
|---|---|---|---|---|
| `grok-4.3_general` | xAI | 256K | OAuth | General fallback |
| `glm-45-flash_general` | Z.ai | 131K | API key (free) | Triage, summarize |

## How It Works

```
Hermes -> Pool Proxy (:9000) -> Worker Agent (:9100) -> llama-server (:8080)
                                        │
                                   swaps models on demand
```

1. **Define** resources in `resources.yaml` with the exact tested command
2. **Deploy** worker agents on each inference host
3. **Route** task types (compression, chat, code) to resources
4. **Pool proxy** handles routing, swapping, fallback chains, idle timers

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Start worker (manages llama-server on this host)
modelpool-worker --config worker.yaml --registry resources.yaml

# Load a specific resource
curl -X POST localhost:9100/worker/load -d '{"resource":"qwen36-35b-a3b_mtp_no-reasoning_multi-gpu"}'

# Check status
curl localhost:9100/worker/status

# Revert to default
curl -X POST localhost:9100/worker/revert
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- full design document
- [Implementation Tasks](docs/TASKS.md) -- development roadmap

## Status

- [x] Phase 1: Worker Agent (deployed on hwrouter)
- [ ] Phase 2: Pool Proxy (routing, streaming, idle timers)
- [ ] Phase 3: Polish (metrics, error handling)
- [ ] Phase 4: Benchmarking & Hermes integration
