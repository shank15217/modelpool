"""Pool HTTP proxy - reverse proxy with streaming and task routing.

Accepts standard OpenAI /v1/chat/completions requests, resolves the task
type from X-Task-Type header, triggers model swaps if needed, and proxies
the request to the right worker or external endpoint.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from modelpool.registry import Registry, RegistryError
from modelpool.pool.router import Router, RoutingError, Resolution
from modelpool.worker.loader import LoadError

logger = logging.getLogger("modelpool.pool.proxy")


class PoolProxy:
    """The main pool proxy that routes and forwards inference requests."""

    def __init__(self, registry: Registry, router: Router):
        self.registry = registry
        self.router = router
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

        # Idle timer state: worker_name -> (resource_name, expires_at)
        self._idle_timers: dict[str, tuple[str, float]] = {}

    async def handle_chat_completions(self, request: Request) -> StreamingResponse | JSONResponse:
        """Handle /v1/chat/completions with task routing.

        Routing priority:
        1. X-Task-Type header (explicit, e.g. "compression")
        2. model field in request body (e.g. "compression" -> looks up routing table)
        3. Default "chat"
        """
        start_time = time.time()

        # Read request body
        try:
            body = await request.body()
        except Exception as e:
            raise HTTPException(400, f"Failed to read request body: {e}")

        body_json = _safe_json(body)

        # Determine tag: header > model field > default
        tag = request.headers.get("X-Task-Type")
        if not tag and body_json:
            model_field = body_json.get("model", "")
            # Check if the model field matches a known tag
            if model_field and model_field in self.router.tags:
                tag = model_field
                logger.info(f"Model-name routing: '{model_field}' -> tag '{tag}'")
        if not tag:
            tag = "chat"

        # Route to resource + worker
        try:
            resolution = self.router.resolve(tag)
        except RoutingError as e:
            logger.error(f"Routing failed for task '{tag}': {e}")
            raise HTTPException(503, str(e))
        except RegistryError as e:
            raise HTTPException(404, str(e))

        logger.info(
            f"Task '{tag}' -> resource '{resolution.resource.name}' "
            f"on worker '{resolution.worker.name}' "
            f"(swap={resolution.needs_swap}, external={resolution.is_external})"
        )

        # Handle swap if needed (managed resources only)
        if resolution.needs_swap and not resolution.is_external:
            try:
                await self._trigger_swap(resolution)
            except SwapError as e:
                logger.warning(f"Swap failed: {e}, trying fallbacks")
                resolution = await self._try_fallbacks(tag, resolution)
                if resolution is None:
                    raise HTTPException(503, f"Swap failed and no fallbacks available: {e}")

        # Reset idle timer for this worker
        if not resolution.is_external:
            self._reset_idle_timer(resolution)

        # Proxy the request
        target_url = f"{resolution.inference_url}/v1/chat/completions"
        headers = self._build_proxy_headers(request, resolution)

        # Check if streaming
        body_json = _safe_json(body)
        stream = body_json.get("stream", False) if body_json else False

        try:
            if stream:
                return await self._proxy_stream(target_url, headers, body, resolution, start_time)
            else:
                return await self._proxy_sync(target_url, headers, body, resolution, start_time)
        except httpx.ConnectError as e:
            logger.error(f"Connection failed to {target_url}: {e}")
            raise HTTPException(502, f"Worker unreachable: {resolution.worker.name}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout proxying to {target_url}: {e}")
            raise HTTPException(504, f"Worker timeout: {resolution.worker.name}")

    async def handle_models(self, request: Request) -> JSONResponse:
        """Handle /v1/models - aggregate loaded models across workers."""
        models = []
        statuses = self.router.get_all_worker_statuses()

        for worker_name, status in statuses.items():
            if status.get("state") == "ready" and status.get("loaded_resource"):
                resource_name = status["loaded_resource"]
                try:
                    resource = self.registry.get_resource(resource_name)
                    models.append({
                        "id": resource_name,
                        "object": "model",
                        "owned_by": "modelpool",
                        "worker": worker_name,
                        "ctx": resource.ctx,
                        "capabilities": resource.capabilities,
                    })
                except RegistryError:
                    pass

        # Add external resources
        for name, resource in self.registry.resources.items():
            if resource.type == "external":
                models.append({
                    "id": name,
                    "object": "model",
                    "owned_by": "external",
                    "worker": resource.workers[0] if resource.workers else "unknown",
                    "ctx": resource.ctx,
                    "capabilities": resource.capabilities,
                })

        return JSONResponse({"object": "list", "data": models})

    async def _trigger_swap(self, resolution: Resolution) -> None:
        """Trigger a model swap on the target worker (async)."""
        import asyncio
        timeout = 120  # default swap timeout

        logger.info(
            f"Triggering swap on '{resolution.worker.name}': "
            f"{resolution.currently_loaded} -> {resolution.resource.name}"
        )

        url = f"{resolution.worker_api_url}/worker/load"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
                resp = await client.post(
                    url,
                    json={"resource": resolution.resource.name},
                )
                if resp.status_code not in (200, 202):
                    raise SwapError(
                        f"Worker returned {resp.status_code}: {resp.text[:200]}"
                    )
            logger.info(f"Swap complete: {resolution.resource.name} loaded")
        except httpx.TimeoutException:
            raise SwapError(f"Swap timed out after {timeout}s")
        except httpx.ConnectError:
            raise SwapError(f"Worker unreachable during swap")

    async def _try_fallbacks(
        self, tag: str, failed_resolution: Resolution
    ) -> Optional[Resolution]:
        """Try fallback resources from the resolution's fallback chain."""
        for fb_resource, fb_worker in failed_resolution.fallback_chain:
            try:
                fb_resolution = self.router._resolve_resource(fb_resource)
                if fb_resolution:
                    logger.info(f"Fallback: using '{fb_resource.name}' instead")
                    fb_resolution.tag = tag
                    return fb_resolution
            except Exception as e:
                logger.warning(f"Fallback '{fb_resource.name}' failed: {e}")
        return None

    async def _proxy_stream(
        self, target_url: str, headers: dict, body: bytes,
        resolution: Resolution, start_time: float,
    ) -> StreamingResponse:
        """Proxy with SSE streaming passthrough.

        Uses an asyncio.Queue to decouple the upstream read from the
        downstream write, so the httpx stream context can stay alive
        while Starlette iterates the generator.
        """
        import asyncio

        queue: asyncio.Queue = asyncio.Queue()
        status_code_holder: list[int] = []
        content_type_holder: list[str] = []

        async def _read_upstream():
            try:
                async with self._http_client.stream(
                    "POST", target_url, headers=headers, content=body,
                ) as upstream:
                    status_code_holder.append(upstream.status_code)
                    content_type_holder.append(
                        upstream.headers.get("content-type", "text/event-stream")
                    )

                    if upstream.status_code != 200:
                        error_body = await upstream.aread()
                        logger.error(f"Upstream error {upstream.status_code}: {error_body[:200]}")
                        await queue.put(("error", upstream.status_code, error_body))
                        return

                    async for chunk in upstream.aiter_bytes():
                        await queue.put(("data", chunk))
            except Exception as e:
                logger.error(f"Upstream stream error: {e}")
                await queue.put(("error", 502, str(e).encode()))
            finally:
                await queue.put(None)  # sentinel

        # Start upstream reader as background task
        asyncio.create_task(_read_upstream())

        # Wait for status code to arrive
        # First item in queue tells us if it's data or error
        first = await queue.get()
        if first is None:
            return JSONResponse(status_code=502, content={"error": "Upstream closed unexpectedly"})

        if first[0] == "error":
            return JSONResponse(status_code=first[1], content={"error": first[2][:200].decode(errors="replace")})

        # It's data -- start streaming
        first_chunk = first[1]
        content_type = content_type_holder[0] if content_type_holder else "text/event-stream"

        async def generate():
            yield first_chunk
            while True:
                item = await queue.get()
                if item is None:
                    break
                if item[0] == "error":
                    break
                yield item[1]
            elapsed = time.time() - start_time
            logger.info(
                f"Stream completed: {resolution.resource.name} on "
                f"{resolution.worker.name} in {elapsed:.1f}s"
            )

        return StreamingResponse(
            generate(),
            media_type=content_type,
            headers={"X-Accel-Buffering": "no"},
        )

    async def _proxy_sync(
        self, target_url: str, headers: dict, body: bytes,
        resolution: Resolution, start_time: float,
    ) -> JSONResponse:
        """Proxy non-streaming request."""
        resp = await self._http_client.post(target_url, headers=headers, content=body)
        elapsed = time.time() - start_time
        logger.info(
            f"Request completed: {resolution.resource.name} on "
            f"{resolution.worker.name} in {elapsed:.1f}s (status={resp.status_code})"
        )
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text},
        )

    def _build_proxy_headers(self, request: Request, resolution: Resolution) -> dict:
        """Build headers for the upstream request."""
        headers = {
            "content-type": request.headers.get("content-type", "application/json"),
        }

        # Auth injection for external resources
        if resolution.is_external and resolution.resource.auth:
            auth = resolution.resource.auth
            if auth.method == "api_key" and auth.env_var:
                key = os.environ.get(auth.env_var, "")
                if key:
                    headers["authorization"] = f"Bearer {key}"
            elif auth.method == "xai-oauth":
                try:
                    from hermes_cli.auth import resolve_xai_oauth_runtime_credentials
                    creds = resolve_xai_oauth_runtime_credentials()
                    if creds.get("api_key"):
                        headers["authorization"] = f"Bearer {creds['api_key']}"
                except ImportError:
                    logger.warning("Hermes auth not available for xAI OAuth")

        return headers

    def _reset_idle_timer(self, resolution: Resolution) -> None:
        """Reset the idle timer for the worker after a successful request.

        Uses the worker's idle_shutdown setting. The worker manages its own
        idle timeout, so the pool proxy idle timer is just for revert-to-default
        behavior (disabled for now -- let the worker handle it).
        """
        pass

    def get_idle_timers(self) -> dict:
        """Get current idle timer state."""
        now = time.time()
        result = {}
        for worker_name, (resource, expires_at) in self._idle_timers.items():
            remaining = max(0, expires_at - now)
            if remaining > 0:
                result[worker_name] = {
                    "resource": resource,
                    "expires_in_s": round(remaining),
                }
        return result

    async def close(self):
        """Clean up HTTP client."""
        await self._http_client.aclose()


class SwapError(Exception):
    """Raised when a model swap fails."""


def _safe_json(body: bytes) -> Optional[dict]:
    """Safely parse JSON body."""
    try:
        import json
        return json.loads(body)
    except Exception:
        return None
