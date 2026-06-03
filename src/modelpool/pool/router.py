"""Pool router - resolves tags to resources and workers.

The router is the brain of the pool. Given a tag (e.g. "compression", "chat"),
it:
1. Finds all resources tagged with that tag, sorted by priority
2. For each candidate, checks if a worker is available
3. Returns the best resolution (resource + worker + swap status)

All worker status lookups are async and cached with a TTL to avoid
blocking the event loop or hammering workers with status requests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from modelpool.registry import Registry, Resource, Worker

logger = logging.getLogger("modelpool.pool.router")


@dataclass
class Resolution:
    """The result of routing a tag to a concrete resource + worker."""

    tag: str
    resource: Resource
    worker: Worker
    needs_swap: bool
    currently_loaded: Optional[str] = None
    fallback_chain: list[tuple[Resource, Worker]] = field(default_factory=list)

    @property
    def is_external(self) -> bool:
        """True if this resolution points to an external (cloud) resource."""
        return not self.worker.is_managed or not self.resource.is_managed

    @property
    def inference_url(self) -> str:
        """URL to proxy the request to."""
        if self.is_external:
            return self.resource.endpoint or ""
        return self.worker.inference_url

    @property
    def worker_api_url(self) -> str:
        """URL for the worker management API."""
        return self.worker.worker_url


class Router:
    """Routes tags to resources and workers using priority-based resolution.

    All methods that query workers are async. A status cache with configurable
    TTL avoids redundant HTTP calls on every resolve().
    """

    def __init__(
        self,
        registry: Registry,
        worker_timeout: float = 5.0,
        status_ttl: float = 2.0,
    ):
        self.registry = registry
        self.worker_timeout = worker_timeout
        self.status_ttl = status_ttl

        # Shared async HTTP client for all worker status lookups
        self._client: Optional[httpx.AsyncClient] = None

        # Status cache: worker_name -> (timestamp, status_dict | None)
        self._status_cache: dict[str, tuple[float, Optional[dict]]] = {}

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.worker_timeout, connect=3.0),
            )
        return self._client

    async def close(self):
        """Clean up the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def tags(self) -> dict:
        """Access all known tags for model-name lookups."""
        return self.registry.all_tags()

    def invalidate_status_cache(self, worker_name: str = ""):
        """Invalidate cached status for a specific worker, or all workers."""
        if worker_name:
            self._status_cache.pop(worker_name, None)
        else:
            self._status_cache.clear()

    async def resolve(self, tag: str) -> Resolution:
        """Resolve a tag to a resource + worker plan.

        Strategy:
        1. Get all resources with this tag, sorted by priority (lowest first)
        2. For each candidate, try to find an available worker
        3. First successful match wins (best priority that's available)
        4. Build fallback chain from remaining candidates
        """
        candidates = self.registry.resolve_tag(tag)

        if not candidates:
            raise RoutingError(
                f"No resources tagged '{tag}'. "
                f"Available tags: {list(self.registry.all_tags().keys())}"
            )

        resolution = None
        fallback_chain = []

        for resource, priority in candidates:
            result = await self._resolve_resource(resource, tag=tag)
            if result is not None:
                if resolution is None:
                    resolution = result
                    logger.info(
                        f"Tag '{tag}' -> resource '{resource.name}' "
                        f"(priority {priority}) on worker '{result.worker.name}' "
                        f"(swap={result.needs_swap})"
                    )
                else:
                    fallback_chain.append((resource, result.worker))
            else:
                logger.debug(
                    f"Tag '{tag}': resource '{resource.name}' (priority {priority}) "
                    f"has no available workers"
                )

        if resolution is None:
            raise RoutingError(
                f"No available workers for any resource tagged '{tag}' "
                f"(tried {len(candidates)} candidates)"
            )

        resolution.tag = tag
        resolution.fallback_chain = fallback_chain
        return resolution

    async def _resolve_resource(self, resource: Resource, tag: str = "") -> Optional[Resolution]:
        """Try to resolve a specific resource to a worker."""
        if resource.type == "external":
            return self._resolve_external(resource)
        return await self._resolve_managed(resource, tag=tag)

    def _resolve_external(self, resource: Resource) -> Resolution:
        """External resources always resolve -- no worker state to check."""
        workers = self.registry.get_workers_for_resource(resource.name)
        if not workers:
            raise RoutingError(
                f"External resource '{resource.name}' has no workers defined"
            )
        worker = workers[0]

        return Resolution(
            tag="",
            resource=resource,
            worker=worker,
            needs_swap=False,
        )

    async def _resolve_managed(self, resource: Resource, tag: str = "") -> Optional[Resolution]:
        """Resolve a managed resource to a worker.

        Priority:
        1. Any generalist resource that is already loaded with capacity
        2. The requested resource is already loaded on a worker
        3. Worker is idle (cold start)
        4. Worker is ready for swap (only if under max_concurrent_models)
        """
        # Step 1: Check for a loaded generalist with capacity
        gen = await self._find_loaded_generalist(tag=tag if tag else None)
        if gen:
            return gen

        workers = self.registry.get_workers_for_resource(resource.name)
        if not workers:
            return None

        ready_with_resource = None
        ready_for_swap = None
        idle_worker = None

        for worker in workers:
            status = await self._get_worker_status(worker)
            if status is None:
                logger.debug(f"Worker '{worker.name}' unreachable, skipping")
                continue

            state = status.get("state", "unknown")
            loaded = status.get("loaded_resource")
            current_models = status.get("loaded_models_count", 0)
            if loaded and current_models == 0:
                current_models = 1
            max_models = worker.max_concurrent_models

            # Best case: already loaded on this worker
            if state == "ready" and loaded == resource.name:
                logger.info(
                    f"Resource '{resource.name}' already loaded on "
                    f"worker '{worker.name}'"
                )
                return Resolution(
                    tag="",
                    resource=resource,
                    worker=worker,
                    needs_swap=False,
                    currently_loaded=loaded,
                )

            # Swap candidate: worker has a different model loaded
            if state == "ready" and loaded and ready_for_swap is None:
                ready_for_swap = (worker, loaded)

            # Cold load: worker is idle
            if state == "idle" and idle_worker is None:
                idle_worker = worker

        # Prefer idle worker (cold start is clean, no drain needed)
        if idle_worker:
            logger.info(
                f"Resource '{resource.name}' needs cold load on idle "
                f"worker '{idle_worker.name}'"
            )
            return Resolution(
                tag="",
                resource=resource,
                worker=idle_worker,
                needs_swap=True,
                currently_loaded=None,
            )

        # Use a worker that's ready for a swap
        if ready_for_swap:
            worker, currently_loaded = ready_for_swap
            logger.info(
                f"Resource '{resource.name}' needs swap on worker "
                f"'{worker.name}' (currently: {currently_loaded})"
            )
            return Resolution(
                tag="",
                resource=resource,
                worker=worker,
                needs_swap=True,
                currently_loaded=currently_loaded,
            )

        return None

    async def _find_loaded_generalist(self, tag: Optional[str] = None) -> Optional[Resolution]:
        """Find any generalist resource that is already loaded and has capacity."""
        for res in self.registry.resources.values():
            if not res.generalist or not res.is_managed:
                continue
            for wname in res.workers:
                try:
                    w = self.registry.get_worker(wname)
                except Exception:
                    continue
                status = await self._get_worker_status(w)
                if status is None:
                    continue
                if (status.get("state") == "ready"
                        and status.get("loaded_resource") == res.name):
                    current = status.get("loaded_models_count", 0)
                    if res.name and current == 0:
                        current = 1
                    if current <= w.max_concurrent_models:
                        logger.info(
                            f"Using loaded generalist '{res.name}' on "
                            f"'{w.name}' (capacity available)"
                        )
                        return Resolution(
                            tag="",
                            resource=res,
                            worker=w,
                            needs_swap=False,
                            currently_loaded=res.name,
                        )
        return None

    async def _get_worker_status(self, worker: Worker) -> Optional[dict]:
        """Query a worker's status API with caching.

        Returns cached status if within TTL, otherwise makes an async HTTP call.
        """
        if not worker.is_managed:
            return {"state": "external"}

        # Check cache
        now = time.monotonic()
        cached = self._status_cache.get(worker.name)
        if cached is not None:
            ts, status = cached
            if now - ts < self.status_ttl:
                return status

        # Cache miss: async HTTP call
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"{worker.worker_url}/worker/status",
            )
            if resp.status_code == 200:
                status = resp.json()
                self._status_cache[worker.name] = (now, status)
                return status
        except httpx.ConnectError:
            pass
        except httpx.TimeoutException:
            logger.warning(f"Worker '{worker.name}' status timeout")
        except Exception as e:
            logger.warning(f"Worker '{worker.name}' status error: {e}")

        # Cache the failure too (prevents hammering unreachable workers)
        self._status_cache[worker.name] = (now, None)
        return None

    async def get_all_worker_statuses(self) -> dict[str, dict]:
        """Get status from all managed workers."""
        import asyncio
        statuses = {}
        tasks = {}
        for name, worker in self.registry.workers.items():
            if worker.is_managed:
                tasks[name] = self._get_worker_status(worker)
            else:
                statuses[name] = {"state": "external"}

        # Query all workers concurrently
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                statuses[name] = {"state": "error", "error": str(result)}
            else:
                statuses[name] = result or {"state": "unreachable"}

        return statuses


class RoutingError(Exception):
    """Raised when routing fails for a tag."""
