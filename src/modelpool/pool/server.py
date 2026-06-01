"""Pool server - FastAPI app for the modelpool proxy.

Exposes:
  POST /v1/chat/completions  -- routed inference proxy
  GET  /v1/models            -- aggregate loaded models
  GET  /pool/status          -- workers, loaded resources, idle timers
  GET  /pool/routing         -- current routing table
  POST /pool/swap            -- manual swap (admin)
  POST /pool/revert          -- manual revert (admin)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from modelpool.registry import Registry, RegistryError
from modelpool.pool.router import Router, RoutingError
from modelpool.pool.proxy import PoolProxy, SwapError

logger = logging.getLogger("modelpool.pool.server")

# Global state
_registry: Registry | None = None
_router: Router | None = None
_proxy: PoolProxy | None = None
_idle_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start idle timer checker on startup, clean up on shutdown."""
    global _idle_task
    _idle_task = asyncio.create_task(_idle_timer_loop())
    yield
    if _idle_task:
        _idle_task.cancel()
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
    """Aggregate loaded models across all workers."""
    if not _proxy:
        raise HTTPException(500, "Pool not configured")
    return await _proxy.handle_models(request)


# --- Pool management endpoints ---


@app.get("/pool/status")
async def pool_status():
    """Cluster-wide status: workers, loaded resources, idle timers."""
    if not _router or not _proxy:
        raise HTTPException(500, "Pool not configured")

    worker_statuses = _router.get_all_worker_statuses()
    idle_timers = _proxy.get_idle_timers()

    return {
        "workers": worker_statuses,
        "idle_timers": idle_timers,
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


@app.post("/pool/swap")
async def pool_swap(request: Request):
    """Manual swap: force a worker to load a specific resource."""
    if not _registry or not _router:
        raise HTTPException(500, "Pool not configured")

    body = await request.json()
    worker_name = body.get("worker")
    resource_name = body.get("resource")

    if not worker_name or not resource_name:
        raise HTTPException(400, "Must specify 'worker' and 'resource'")

    try:
        worker = _registry.get_worker(worker_name)
        resource = _registry.get_resource(resource_name)
    except RegistryError as e:
        raise HTTPException(404, str(e))

    if worker_name not in resource.workers:
        raise HTTPException(
            422, f"Resource '{resource_name}' cannot run on worker '{worker_name}'"
        )

    import requests as sync_requests
    headers = {}
    if worker.pool_secret:
        headers["X-Pool-Secret"] = worker.pool_secret
    try:
        resp = sync_requests.post(
            f"{worker.worker_url}/worker/load",
            json={"resource": resource_name},
            headers=headers,
            timeout=120,
        )
        return {"status": resp.json().get("status", "ok"), "worker": worker_name, "resource": resource_name}
    except Exception as e:
        raise HTTPException(503, f"Swap failed: {e}")


@app.post("/pool/revert")
async def pool_revert(request: Request):
    """Manual revert: return a worker to its default resource."""
    if not _registry or not _router:
        raise HTTPException(500, "Pool not configured")

    body = await request.json()
    worker_name = body.get("worker")

    if not worker_name:
        raise HTTPException(400, "Must specify 'worker'")

    try:
        worker = _registry.get_worker(worker_name)
    except RegistryError as e:
        raise HTTPException(404, str(e))

    import requests as sync_requests
    headers = {}
    if worker.pool_secret:
        headers["X-Pool-Secret"] = worker.pool_secret
    try:
        resp = sync_requests.post(
            f"{worker.worker_url}/worker/revert",
            headers=headers,
            timeout=120,
        )
        return {"status": resp.json().get("status", "ok"), "worker": worker_name}
    except Exception as e:
        raise HTTPException(503, f"Revert failed: {e}")


# --- Idle timer background task ---


async def _idle_timer_loop():
    """Check idle timers every 30s, revert workers that expired."""
    while True:
        await asyncio.sleep(30)
        if not _proxy or not _registry:
            continue

        try:
            timers = _proxy.get_idle_timers()
            for worker_name, timer_info in timers.items():
                if timer_info["expires_in_s"] <= 0:
                    logger.info(
                        f"Idle timer expired for worker '{worker_name}', "
                        f"reverting from '{timer_info['resource']}'"
                    )
                    try:
                        worker = _registry.get_worker(worker_name)
                        import requests as sync_requests
                        sync_requests.post(
                            f"{worker.worker_url}/worker/revert",
                            timeout=120,
                        )
                    except Exception as e:
                        logger.error(f"Idle revert failed for '{worker_name}': {e}")

                    # Clear the timer
                    _proxy._idle_timers.pop(worker_name, None)
        except Exception as e:
            logger.error(f"Idle timer check error: {e}")
