# modelpool

On-demand LLM resource orchestration for homelab GPU clusters.

A **resource** is a fully configured model recipe -- the GGUF weights file plus every llama-server launch flag, tuned for specific hardware and a specific use case. ModelPool routes tasks to the right resource using **tag-based priority routing** and swaps models in and out of GPU memory as needed.

## How Routing Works

Resources are tagged with priority levels. Lower number = preferred.

```yaml
resources:
  gpu-27b:
    tags:
      chat: 1       # best for chat
      agentic: 1    # best for agents
      compression: 3 # can do compression but not preferred

  gpu-35b:
    tags:
      compression: 1 # best for compression
      title: 1       # best for titles
      chat: 2         # decent for chat (fallback)
```

Router resolves a tag by sorting resources by priority, picking the first available. No separate routing table needed -- just tag your resources.

```
Hermes sends model: "compression" -> pool looks up tag "compression"
  -> gpu-35b (priority 1) -> worker available -> done
  -> if gpu-35b busy -> cpu-35b (priority 2) -> fallback
  -> if cpu busy -> cloud-glm (priority 3) -> last resort
```

## Current Resources

### hwrouter (2x RX 9070 XT, 32GB VRAM, ROCm)

| Resource | Model | Ctx | Speed (prompt/gen) | Tags (priority) |
|---|---|---|---|---|
| `qwen36-27b_mtp_reasoning_multi-gpu` | 27B MTP Q4_K_M, reasoning ON | 131K | 728 / 33.8 t/s | chat:1, agentic:1, code:1 |
| `qwen36-35b-a3b_mtp_no-reasoning_multi-gpu` | 35B MoE (3B active), reasoning OFF | 262K | 2,225 / 71.4 t/s | compression:1, title:1, summarize:1 |

Swap time between resources: ~8 seconds.

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
Hermes -> Pool Proxy (:9000) -> Worker Agent (:9100) -> llama-server (:8080)
                                        │
                                   swaps models on demand
```

Workers are **paired** with a single pool proxy using a shared secret (`pool_secret`). Management endpoints (load, unload, revert) require the secret -- preventing multiple pool proxies from fighting over the same GPU. Status and health endpoints stay open for monitoring.

1. **Define** resources in `resources.yaml` with tags, priorities, and worker secrets
2. **Deploy** worker agents on each inference host (each with `pool_secret`)
3. **Deploy** one pool proxy per Hermes instance (reads secrets from registry)
4. **Pool proxy** resolves tags, authenticates to workers, handles swapping

### Worker Pairing

```yaml
# resources.yaml
workers:
  gpu-host:
    host: 192.168.35.185
    pool_secret: mp-secret-homelab   # shared secret with pool proxy
```

- Worker rejects management commands without the correct `X-Pool-Secret` header
- Pool proxy reads the secret from the registry and sends it with every swap/load
- `GET /worker/status` shows `paired: true/false`
- One pool proxy per Hermes instance = no GPU fighting

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Start worker (manages llama-server on this host)
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
  -d '{"model":"compression","messages":[{"role":"user","content":"Compress this text..."}]}'
```

## Hermes Integration

Configure Hermes to use the pool proxy as a provider:

```yaml
# ~/.hermes/config.yaml
model:
  default: chat                  # maps to tag "chat" in pool
  provider: custom:modelpool

providers:
  modelpool:
    base_url: http://localhost:9000/v1
    api_key: no-key-required

auxiliary:
  compression:
    provider: custom:modelpool
    model: compression            # maps to tag "compression" in pool
  title_generation:
    provider: custom:modelpool
    model: compression
```

Hermes sends `model: chat` or `model: compression` -> pool resolves the tag -> picks the best available resource. Hermes never knows about specific model names or workers.

## Worker Management

```bash
# Status
curl http://localhost:9100/worker/status

# Load a resource (swaps model)
curl -X POST localhost:9100/worker/load -d '{"resource":"qwen36-35b-a3b_mtp_no-reasoning_multi-gpu"}'

# Revert to default
curl -X POST localhost:9100/worker/revert

# Unload (free GPU/CPU memory)
curl -X POST localhost:9100/worker/unload
```

Workers start idle (no model loaded) when `idle_shutdown > 0` is configured. Models load on demand, unload after 15 minutes of inactivity.
