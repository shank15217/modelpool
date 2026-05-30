"""Pool router - resolves task types to resources and workers.

The router is the brain of the pool. Given a task type, it:
1. Looks up the route (primary resource + fallback chain)
2. Finds the best worker that can serve the resource
3. Checks if the resource is already loaded on any worker
4. Returns a resolution plan (resource, worker, needs_swap, fallbacks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from modelpool.registry import Registry, Resource, Worker, Route

logger = logging.getLogger("modelpool.pool.router")


@dataclass
class Resolution:
    """The result of routing a task type to a concrete resource + worker."""

    task_type: str
    resource: Resource
    worker: Worker
    needs_swap: bool
    currently_loaded: Optional[str] = None  # what's on the worker now
    fallback_chain: list[tuple[Resource, Worker]] = field(default_factory=list)
    route: Optional[Route] = None

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
    """Routes task types to resources and workers."""

    def __init__(self, registry: Registry, worker_timeout: float = 5.0):
        self.registry = registry
        self.worker_timeout = worker_timeout

    def resolve(self, task_type: str) -> Resolution:
        """Resolve a task type to a resource + worker plan.

        Strategy:
        1. Look up the route for this task type
        2. For the primary resource, find workers that can serve it
        3. Check each worker: is the resource already loaded?
        4. If already loaded -> use it (no swap needed)
        5. If not loaded -> pick the best available worker (needs swap)
        6. Build the fallback chain from route config
        """
        route = self.registry.get_route(task_type)
        primary_resource = self.registry.get_resource(route.resource)

        # Try primary resource
        resolution = self._resolve_resource(primary_resource, route)
        if resolution:
            return resolution

        # Primary failed (no workers available) -> try fallbacks
        logger.info(
            f"Primary resource '{primary_resource.name}' unavailable for "
            f"task '{task_type}', trying fallbacks"
        )
        for fb_name in route.fallback_resources:
            fb_resource = self.registry.get_resource(fb_name)
            resolution = self._resolve_resource(fb_resource, route)
            if resolution:
                resolution.route = route
                return resolution

        # All resources failed
        raise RoutingError(
            f"No available resource for task type '{task_type}' "
            f"(tried primary + {len(route.fallback_resources)} fallbacks)"
        )

    def _resolve_resource(
        self, resource: Resource, route: Route
    ) -> Optional[Resolution]:
        """Try to resolve a specific resource to a worker."""
        if resource.type == "external":
            return self._resolve_external(resource, route)

        return self._resolve_managed(resource, route)

    def _resolve_external(
        self, resource: Resource, route: Route
    ) -> Resolution:
        """External resources always resolve -- no worker state to check."""
        workers = self.registry.get_workers_for_resource(resource.name)
        if not workers:
            raise RoutingError(
                f"External resource '{resource.name}' has no workers defined"
            )
        worker = workers[0]

        return Resolution(
            task_type=route.task_type,
            resource=resource,
            worker=worker,
            needs_swap=False,
            route=route,
        )

    def _resolve_managed(
        self, resource: Resource, route: Route
    ) -> Optional[Resolution]:
        """Resolve a managed resource to a worker.

        Checks each worker that can serve this resource:
        1. If already loaded -> use immediately (no swap)
        2. If worker is ready with something else -> candidate for swap
        3. If worker is idle (no model loaded) -> candidate for cold load
        4. If worker is busy (loading/draining) -> skip
        """
        workers = self.registry.get_workers_for_resource(resource.name)
        if not workers:
            return None

        ready_with_resource = None
        ready_for_swap = None
        idle_worker = None

        for worker in workers:
            status = self._get_worker_status(worker)
            if status is None:
                logger.debug(f"Worker '{worker.name}' unreachable, skipping")
                continue

            state = status.get("state", "unknown")
            loaded = status.get("loaded_resource")

            # Best case: already loaded on this worker
            if state == "ready" and loaded == resource.name:
                logger.info(
                    f"Resource '{resource.name}' already loaded on "
                    f"worker '{worker.name}'"
                )
                return Resolution(
                    task_type=route.task_type,
                    resource=resource,
                    worker=worker,
                    needs_swap=False,
                    currently_loaded=loaded,
                    route=route,
                )

            # Good: worker is ready but has a different resource
            if state == "ready" and loaded and ready_for_swap is None:
                ready_for_swap = (worker, loaded)

            # Good: worker is idle (no model loaded) -- cold start
            if state == "idle" and idle_worker is None:
                idle_worker = worker

        # Prefer idle worker (cold start is clean, no drain needed)
        if idle_worker:
            logger.info(
                f"Resource '{resource.name}' needs cold load on idle "
                f"worker '{idle_worker.name}'"
            )
            return Resolution(
                task_type=route.task_type,
                resource=resource,
                worker=idle_worker,
                needs_swap=True,
                currently_loaded=None,
                route=route,
            )

        # Use a worker that's ready for a swap
        if ready_for_swap:
            worker, currently_loaded = ready_for_swap
            logger.info(
                f"Resource '{resource.name}' needs swap on worker "
                f"'{worker.name}' (currently: {currently_loaded})"
            )
            return Resolution(
                task_type=route.task_type,
                resource=resource,
                worker=worker,
                needs_swap=True,
                currently_loaded=currently_loaded,
                route=route,
            )

        # No workers available
        return None

    def build_fallback_chain(self, route: Route) -> list[tuple[Resource, Worker]]:
        """Build the full fallback chain for a route.

        Returns list of (resource, worker) pairs in priority order.
        Only includes resources that have available workers.
        """
        chain = []
        for fb_name in route.fallback_resources:
            try:
                resource = self.registry.get_resource(fb_name)
                workers = self.registry.get_workers_for_resource(fb_name)
                if workers:
                    # For external resources, just use the first worker
                    # For managed, we'll check availability at request time
                    chain.append((resource, workers[0]))
            except Exception as e:
                logger.warning(f"Skipping fallback '{fb_name}': {e}")
        return chain

    def _get_worker_status(self, worker: Worker) -> Optional[dict]:
        """Query a worker's status API. Returns None if unreachable."""
        if not worker.is_managed:
            return {"state": "external"}  # virtual status for external workers

        try:
            resp = requests.get(
                f"{worker.worker_url}/worker/status",
                timeout=self.worker_timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            logger.warning(f"Worker '{worker.name}' status timeout")
        except Exception as e:
            logger.warning(f"Worker '{worker.name}' status error: {e}")

        return None

    def get_all_worker_statuses(self) -> dict[str, dict]:
        """Get status from all managed workers."""
        statuses = {}
        for name, worker in self.registry.workers.items():
            if worker.is_managed:
                statuses[name] = self._get_worker_status(worker) or {
                    "state": "unreachable"
                }
            else:
                statuses[name] = {"state": "external"}
        return statuses


class RoutingError(Exception):
    """Raised when routing fails for a task type."""
