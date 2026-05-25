# modelpool

On-demand LLM model orchestration for homelab GPU clusters. Routes inference tasks to the optimal model by swapping models in and out of GPU memory as needed.

**The problem:** Different AI tasks need different models, but GPU VRAM is finite. You can't load everything at once.

**The solution:** A proxy that accepts standard OpenAI API requests tagged with a task type, automatically loads the right model on the right GPU worker, proxies the request, and reverts when done.

## Quick Start

```bash
# Coming soon
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design document.

## Components

| Component | Description |
|---|---|
| **Registry** | YAML model catalog -- paths, sizes, capabilities, worker assignments |
| **Worker** | Agent on each GPU/CPU host that manages llama-server lifecycle |
| **Pool** | Proxy/router that maps tasks to models and manages swaps |

## Status

Pre-implementation. Architecture design in progress.
