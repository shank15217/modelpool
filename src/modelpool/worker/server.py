"""Worker HTTP API - FastAPI endpoints for worker management.

Simplified for Architecture A: status and ready endpoints only.
No dynamic load/unload/revert (models are loaded at boot).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from modelpool.worker.loader import LlamaServerManager
from modelpool.worker.watchdog import Watchdog

logger = logging.getLogger("modelpool.worker.server")

# Endpoints that require pool_secret authentication (none in static pool, kept for future admin use)
PROTECTED_ENDPOINTS: set[str] = set()
SECRET_HEADER = "X-Pool-Secret"

# Global state (set during app startup)
_manager: LlamaServerManager | None = None
_watchdog: Watchdog | None = None
_pool_secret: str | None = None


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


def configure(
    manager: LlamaServerManager,
    watchdog: Watchdog,
    pool_secret: str | None = None,
) -> None:
    """Configure the worker app with runtime dependencies."""
    global _manager, _watchdog, _pool_secret
    _manager = manager
    _watchdog = watchdog
    _pool_secret = pool_secret


app = FastAPI(title="ModelPool Worker", lifespan=lifespan)


@app.middleware("http")
async def pool_auth_middleware(request: Request, call_next):
    """Authenticate management requests using pool_secret.

    In static pool mode, only status/ready are exposed (both open).
    This middleware is kept for future admin endpoints.
    """
    if request.url.path.rstrip("/") in PROTECTED_ENDPOINTS:
        if _pool_secret:
            provided = request.headers.get(SECRET_HEADER, "")
            if not hmac.compare_digest(provided, _pool_secret):
                logger.warning(
                    f"Unauthorized management request: {request.url.path} "
                    f"from {request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid or missing pool secret"},
                )
    return await call_next(request)


@app.get("/worker/status")
async def worker_status():
    """Current worker state, loaded resource, slot info, and pairing status."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")
    status = _manager.get_status()
    status["paired"] = _pool_secret is not None
    return status


@app.get("/worker/ready")
async def worker_ready():
    """Returns 200 if ready, 503 otherwise."""
    if not _manager:
        raise HTTPException(500, "Worker not configured")
    if _manager.is_ready():
        return {"status": "ready", "resource": _manager.loaded_resource}
    raise HTTPException(503, f"Worker state: {_manager.state}")
