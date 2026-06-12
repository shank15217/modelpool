"""Pool server - FastAPI app for the modelpool proxy.

Simplified for Architecture A: static pool, no dynamic swapping.

Exposes:
  POST /v1/chat/completions  -- routed inference proxy
  GET  /v1/models            -- list configured resources
  GET  /pool/status          -- pool configuration info
  GET  /pool/routing         -- current routing table
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from modelpool.registry import Registry
from modelpool.pool.router import Router
from modelpool.pool.proxy import PoolProxy

logger = logging.getLogger("modelpool.pool.server")

# Global state
_registry: Registry | None = None
_router: Router | None = None
_proxy: PoolProxy | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up on shutdown."""
    yield
    if _proxy:
        await _proxy.close()


app = FastAPI(title="ModelPool Proxy", lifespan=lifespan)


def configure(registry: Registry, router: Router, proxy: PoolProxy) -> None:
    """Configure the pool app with runtime dependencies."""
    global _registry, _router, _proxy
    _registry = registry
    _router = router
    _proxy = proxy


# --- Inference endpoints ---


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Routed inference proxy. Uses X-Task-Type header for routing."""
    if not _proxy:
        raise HTTPException(500, "Pool not configured")
    return await _proxy.handle_chat_completions(request)


@app.get("/v1/models")
async def list_models(request: Request):
    """List all configured resources."""
    if not _proxy:
        raise HTTPException(500, "Pool not configured")
    return await _proxy.handle_models(request)


# --- Pool management endpoints ---


@app.get("/pool/status")
async def pool_status():
    """Pool configuration info."""
    if not _registry:
        raise HTTPException(500, "Pool not configured")

    workers_info = {}
    for name, worker in _registry.workers.items():
        workers_info[name] = {
            "type": worker.type,
            "host": worker.host if worker.is_managed else "external",
        }

    return {
        "workers": workers_info,
        "resources": {name: {"type": res.type} for name, res in _registry.resources.items()},
    }


@app.get("/pool/routing")
async def pool_routing():
    """Current tag routing table with resource priorities."""
    if not _registry:
        raise HTTPException(500, "Pool not configured")

    all_tags = _registry.all_tags()
    result = {}
    for tag, candidates in all_tags.items():
        result[tag] = [
            {"resource": name, "priority": priority}
            for name, priority in candidates
        ]

    return {"routing": result}
