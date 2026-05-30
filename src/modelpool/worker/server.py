"""Worker HTTP API - FastAPI endpoints for worker management."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import requests as sync_requests
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
_idle_shutdown: int = 0  # seconds, 0 = disabled
_last_request_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start watchdog + idle shutdown monitor on startup, stop on shutdown."""
    if _watchdog:
        _watchdog.start()

    idle_task = None
    if _idle_shutdown > 0:
        idle_task = asyncio.create_task(_idle_shutdown_loop())
        logger.info(f"Idle shutdown monitor started ({_idle_shutdown}s)")

    yield

    if idle_task:
        idle_task.cancel()
    if _watchdog:
        _watchdog.stop()
    if _manager:
        _manager.stop()


def configure(
    manager: LlamaServerManager,
    registry: Registry,
    watchdog: Watchdog,
    worker_name: str,
    idle_shutdown: int = 0,
) -> None:
    """Configure the worker app with runtime dependencies."""
    global _manager, _registry, _watchdog, _worker_name, _idle_shutdown
    _manager = manager
    _registry = registry
    _watchdog = watchdog
    _worker_name = worker_name
    _idle_shutdown = idle_shutdown


def _touch_request_time():
    """Mark that a request was handled (resets idle timer)."""
    global _last_request_time
    _last_request_time = time.time()


async def _idle_shutdown_loop():
    """Background task: unload model after idle_shutdown seconds with no activity."""
    global _last_request_time

    check_interval = 30

    while True:
        await asyncio.sleep(check_interval)

        if not _manager or _idle_shutdown <= 0:
            continue

        if _manager.state != "ready":
            continue

        # Check for active slots
        try:
            resp = sync_requests.get(
                f"http://localhost:{_manager.inference_port}/health",
                timeout=2,
            )
            data = resp.json()
            if data.get("slots_processing", 0) > 0:
                _touch_request_time()  # active work resets timer
                continue
        except Exception:
            continue

        # No active work and no requests within the timeout
        if _last_request_time > 0:
            idle_duration = time.time() - _last_request_time
            if idle_duration >= _idle_shutdown:
                logger.info(
                    f"Idle shutdown: no activity for {int(idle_duration)}s >= "
                    f"{_idle_shutdown}s, unloading '{_manager.loaded_resource}'"
                )
                try:
                    worker = _registry.get_worker(_worker_name) if _registry else None
                    _manager.unload(drain_timeout=worker.drain_timeout if worker else 10)
                    _last_request_time = 0.0
                    logger.info("Model unloaded, worker now idle")
                except Exception as e:
                    logger.error(f"Idle shutdown unload failed: {e}")


app = FastAPI(title="ModelPool Worker", lifespan=lifespan)


@app.get("/worker/status")
async def worker_status():
    """Current worker state, loaded resource, and slot info."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")
    status = _manager.get_status()
    status["idle_shutdown"] = _idle_shutdown if _idle_shutdown > 0 else None
    return status


@app.post("/worker/load", status_code=202)
async def worker_load(req: LoadRequest):
    """Load a resource: drain current -> stop -> start new."""
    if not _manager or not _registry:
        raise HTTPException(500, "Worker not configured")

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
        _touch_request_time()
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
        _touch_request_time()
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

    worker = _registry.get_worker(_worker_name) if _registry else None
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
        _touch_request_time()
        return {"status": "reverted", "resource": default}
    except (StateError, LoadError) as e:
        raise HTTPException(503, f"Revert failed: {e}")
