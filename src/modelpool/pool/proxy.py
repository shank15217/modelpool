"""Pool HTTP proxy - reverse proxy with streaming and task routing.

Accepts standard OpenAI /v1/chat/completions requests, resolves the task
type from X-Task-Type header or model field, and proxies the request to
the right worker or external endpoint. Tries candidates in priority order
for failover.
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

logger = logging.getLogger("modelpool.pool.proxy")


class PoolProxy:
    """The main pool proxy that routes and forwards inference requests."""

    def __init__(self, registry: Registry, router: Router):
        self.registry = registry
        self.router = router
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

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
            if model_field and model_field in self.router.tags:
                tag = model_field
                logger.info(f"Model-name routing: '{model_field}' -> tag '{tag}'")
        if not tag:
            tag = "chat"

        # Route to candidates (synchronous, in-memory)
        try:
            candidates = self.router.resolve(tag)
        except RoutingError as e:
            logger.error(f"Routing failed for task '{tag}': {e}")
            raise HTTPException(503, str(e))
        except RegistryError as e:
            raise HTTPException(404, str(e))

        primary = candidates[0]
        logger.info(
            f"Task '{tag}' -> resource '{primary.resource.name}' "
            f"on worker '{primary.worker.name}' (external={primary.is_external})"
        )

        # Try candidates in order for failover
        last_error = None
        for i, resolution in enumerate(candidates):
            target_url = f"{resolution.inference_url}/v1/chat/completions"
            headers = self._build_proxy_headers(request, resolution)

            # Check if streaming
            stream = body_json.get("stream", False) if body_json else False

            try:
                if stream:
                    return await self._proxy_stream(target_url, headers, body, resolution, start_time)
                else:
                    return await self._proxy_sync(target_url, headers, body, resolution, start_time)
            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    f"Connection failed to '{resolution.worker.name}' "
                    f"({target_url}): {e}, trying next candidate"
                )
                continue
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on '{resolution.worker.name}' "
                    f"({target_url}): {e}, trying next candidate"
                )
                continue

        # All candidates failed
        logger.error(f"All {len(candidates)} candidates failed for tag '{tag}'")
        raise HTTPException(502, f"All workers unreachable for tag '{tag}': {last_error}")

    async def handle_models(self, request: Request) -> JSONResponse:
        """Handle /v1/models - list all configured resources."""
        models = []

        for name, resource in self.registry.resources.items():
            models.append({
                "id": name,
                "object": "model",
                "owned_by": "modelpool" if resource.is_managed else "external",
                "worker": resource.workers[0] if resource.workers else "unknown",
                "ctx": resource.ctx,
                "capabilities": resource.capabilities,
            })

        return JSONResponse({"object": "list", "data": models})

    async def _proxy_stream(
        self, target_url: str, headers: dict, body: bytes,
        resolution: Resolution, start_time: float,
    ) -> StreamingResponse:
        """Proxy with SSE streaming passthrough."""
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
                await queue.put(None)

        asyncio.create_task(_read_upstream())

        first = await queue.get()
        if first is None:
            return JSONResponse(status_code=502, content={"error": "Upstream closed unexpectedly"})

        if first[0] == "error":
            return JSONResponse(status_code=first[1], content={"error": first[2][:200].decode(errors="replace")})

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

    async def close(self):
        """Clean up HTTP clients."""
        await self._http_client.aclose()


def _safe_json(body: bytes) -> Optional[dict]:
    """Safely parse JSON body."""
    try:
        import json
        return json.loads(body)
    except Exception:
        return None
