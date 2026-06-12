# ModelPool

Static resource pool for homelab LLM inference. Routes OpenAI-compatible requests to pre-assigned hardware based on task tags with priority-based failover.

## How It Works

1. Each GPU/CPU worker runs **one model, always loaded, started at boot**
2. Tasks are tagged (chat, compression, title, triage, etc.) with priority tiers
3. Pool proxy resolves a tag to the highest-priority available worker
4. If tier 1 fails, tries tier 2, then tier 3, etc.
5. Zero dynamic routing, zero model swapping, zero worker state queries

## Resources

### hwrouter (2x RX 9700 PRO, 64GB VRAM, ROCm)

| Resource | Model | Ctx | Speed (prompt/gen) | Tags (priority) |
|---|---|---|---|---|
| `qwen36-27b_mtp_reasoning_multi-gpu` | 27B MTP Q4_K_M, reasoning ON | 131K | 728 / 33.8 t/s | chat:1, agentic:1, code:1, reasoning:1 |
| `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu` | 35B MoE (3B active), reasoning OFF | 262K | 2,225 / 71.4 t/s | compression:1, title:1, summarize:1 |

### pvellm (AMD 9850X3D, 48GB RAM, ik_llama.cpp)

| Resource | Model | Ctx | Speed (prompt/gen) | Tags (priority) |
|---|---|---|---|---|
| `qwen36-35b-a3b_no-reasoning_cpu` | 35B MoE (3B active), reasoning OFF | 262K | 473 / 39.4 t/s | compression:2, title:2, triage:2 |

### Cloud (External)

| Resource | Provider | Ctx | Auth | Tags (priority) |
|---|---|---|---|---|
| `grok-4.3_general` | xAI | 256K | OAuth | chat:4, compression:3, agentic:2, vision:1 |
| `glm-45-flash_general` | Z.ai | 131K | API key (free) | compression:4, summarize:3, triage:3, chat:5 |

## Architecture

```
Hermes -> Pool Proxy (:9000) -> Worker (:8080) -> llama-server
    |                              or
    +-- tag "chat" -------------> tier 1: 27B GPU
    +-- tag "compression" ------> tier 1: 35B GPU
    +-- if tier 1 fails --------> tier 2: 35B CPU
    +-- if tier 2 fails --------> tier 3: cloud
```

Workers start their assigned model on boot via systemd. The pool proxy never queries worker state, never triggers swaps. It routes by tag priority and fails over on connection errors.

1. **Define** resources in `resources.yaml` with tags and priorities
2. **Deploy** workers on each inference host (each boots with its assigned model)
3. **Deploy** one pool proxy per Hermes instance
4. **Pool proxy** resolves tags, proxies requests, handles failover

### Routing Rules

- Tag resolved from: `X-Task-Type` header > `model` field in body > default "chat"
- Router returns candidates in priority order (tier 1, tier 2, tier 3...)
- Proxy tries tier 1 first; on connection failure, tries tier 2, etc.
- External (cloud) resources always available as last resort

### Worker Pairing

```yaml
# resources.yaml
workers:
  gpu-host:
    host: 192.168.35.185
    pool_secret: mp-secret-homelab
```

- `/worker/status` and `/worker/ready` are open for monitoring
- Worker starts model at boot and serves until stopped

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Start worker (loads assigned model at boot)
modelpool-worker --config worker.yaml --registry resources.yaml

# Start pool proxy (routes by tags)
modelpool-pool --registry resources.yaml --port 9000

# Send a chat request (routed by tag "chat")
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"chat","messages":[{"role":"user","content":"Hello"}]}'

# Send a compression request (routed by tag "compression")
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"compression","messages":[{"role":"user","content":"Compress this..."}]}'
```

## Hermes Integration

```yaml
# ~/.hermes/config.yaml
model:
  default: chat
  provider: custom:modelpool

providers:
  modelpool:
    base_url: http://localhost:9000/v1
    api_key: no-key-required

auxiliary:
  compression:
    provider: custom:modelpool
    model: compression
  title_generation:
    provider: custom:modelpool
    model: compression
```

Hermes sends `model: chat` or `model: compression` -> pool resolves tag -> picks highest priority -> proxies.

## Testing

```bash
# Run all tests
pytest tests/ -q

# Run only routing tests
pytest tests/unit/test_router.py -v
```

149 tests covering:
- Registry parsing, validation, lookups
- Tag-based priority routing (sync, in-memory)
- Worker subprocess lifecycle (start, stop, command building)
- Pool proxy (streaming, auth injection, failover)
- Worker endpoints (status, ready, health)
- Pool endpoints (status, routing)

## Branches

- `main` -- Architecture A: static pool, priority-based failover
- `arch/dynamic-pool` -- Architecture B: dynamic swapping, generalist preference, worker state queries
