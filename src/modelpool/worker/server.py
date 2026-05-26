"""Worker HTTP API - FastAPI endpoints for worker management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modelpool.registry import Registry, RegistryError
from modelpool.worker.loader import LlamaServerManager, StateError, LoadError
from modelpool.worker.watchdog import Watchdog

logger = logging.getLogger("modelpool.worker.server")


class LoadRequest(BaseModel):
    resource: str


# Global state (set during app startup)
_manager: LlamaServerManager | None = None
_registry: Registry | None = None
_watchdog: Watchdog | None = None
_worker_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start watchdog on startup, stop on shutdown."""
    if _watchdog:
        _watchdog.start()
    yield
    if _watchdog:
        _watchdog.stop()
    if _manager:
        _manager.stop()


app = FastAPI(title="ModelPool Worker", lifespan=lifespan)


def configure(
    manager: LlamaServerManager,
    registry: Registry,
    watchdog: Watchdog,
    worker_name: str,
) -> None:
    """Configure the worker app with runtime dependencies."""
    global _manager, _registry, _watchdog, _worker_name
    _manager = manager
    _registry = registry
    _watchdog = watchdog
    _worker_name = worker_name


@app.get("/worker/status")
async def worker_status():
    """Current worker state, loaded resource, and slot info."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")
    return _manager.get_status()


@app.post("/worker/load", status_code=202)
async def worker_load(req: LoadRequest):
    """Load a resource: drain current -> stop -> start new."""
    if not _manager or not _registry:
        raise HTTPException(500, "Worker not configured")

    # Validate resource exists and this worker can serve it
    try:
        resource = _registry.get_resource(req.resource)
    except RegistryError:
        raise HTTPException(404, f"Resource not found: {req.resource}")

    if _worker_name not in resource.workers:
        raise HTTPException(
            422,
            f"Resource '{req.resource}' cannot run on worker '{_worker_name}'"
        )

    worker = _registry.get_worker(_worker_name)
    if worker.max_model_gb and resource.size_gb > worker.max_model_gb:
        raise HTTPException(
            422,
            f"Resource too large ({resource.size_gb}GB > {worker.max_model_gb}GB)"
        )

    # Already serving this resource
    if _manager.loaded_resource == req.resource and _manager.is_ready():
        return {"status": "already_loaded", "resource": req.resource}

    # Check state
    if _manager.state in ("loading", "draining", "stopping"):
        raise HTTPException(
            409,
            f"Worker busy (state: {_manager.state}). Try again later."
        )

    # Execute the swap
    try:
        _manager.load_resource(
            resource,
            drain_timeout=worker.drain_timeout,
            swap_timeout=worker.swap_timeout,
        )
        return {"status": "loaded", "resource": req.resource}
    except StateError as e:
        raise HTTPException(409, str(e))
    except LoadError as e:
        raise HTTPException(503, f"Failed to load resource: {e}")


@app.post("/worker/unload", status_code=202)
async def worker_unload():
    """Drain and stop the current resource."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")

    if _manager.state not in ("ready", "error"):
        raise HTTPException(409, f"Cannot unload from state: {_manager.state}")

    worker = _registry.get_worker(_worker_name)
    _manager.unload(drain_timeout=worker.drain_timeout if worker else 30)
    return {"status": "unloaded"}


@app.get("/worker/ready")
async def worker_ready():
    """Returns 200 if ready, 503 otherwise."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")
    if _manager.is_ready():
        return {"status": "ready", "resource": _manager.loaded_resource}
    raise HTTPException(503, f"Worker state: {_manager.state}")


@app.post("/worker/revert", status_code=202)
async def worker_revert():
    """Revert to the default resource for this worker."""
    if not _manager or not _registry:
        raise HTTPException(500, "Worker not configured")

    if _manager.state in ("loading", "draining", "stopping"):
        raise HTTPException(409, f"Worker busy (state: {_manager.state})")

    try:
        _manager.revert(_registry, _worker_name)
        default = _registry.get_worker(_worker_name).default_resource
        return {"status": "reverted", "resource": default}
    except (StateError, LoadError) as e:
        raise HTTPException(503, f"Revert failed: {e}")
