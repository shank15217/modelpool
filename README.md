# modelpool

On-demand LLM resource orchestration for homelab GPU clusters.

A **resource** is a fully configured model recipe -- the GGUF weights file plus every llama-server flag, tuned for specific hardware and a specific use case. Resources are tested and benchmarked as a unit. The pool loads resources on demand, serves them via OpenAI-compatible API, and shuts them down when done.

**The key idea:** What you tested is what runs. No parameter generation, no merging, no interpolation. Each resource is an exact command that has been verified on real hardware.

## Quick Start

```bash
# Coming soon
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design document.

## Task Breakdown

See [docs/TASKS.md](docs/TASKS.md) for the implementation plan.

## Status

Pre-implementation. Architecture design in progress.
