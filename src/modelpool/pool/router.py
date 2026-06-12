"""Pool router - resolves tags to resources and workers.

The router is purely synchronous and in-memory. Given a tag (e.g. "compression",
"chat"), it returns all candidate (resource, worker) pairs sorted by priority.
No HTTP calls, no status queries, no network activity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from modelpool.registry import Registry, Resource, Worker

logger = logging.getLogger("modelpool.pool.router")


@dataclass
class Resolution:
    """The result of routing a tag to a concrete resource + worker."""

    tag: str
    resource: Resource
    worker: Worker

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


class Router:
    """Routes tags to resources and workers using priority-based resolution.

    All methods are synchronous, in-memory lookups. No network calls.
    """

    def __init__(self, registry: Registry):
        self.registry = registry

    @property
    def tags(self) -> dict:
        """Access all known tags for model-name lookups."""
        return self.registry.all_tags()

    def resolve(self, tag: str) -> list[Resolution]:
        """Resolve a tag to all candidates sorted by priority (best first).

        Returns a list of Resolution objects. The proxy tries them in order
        until one works (failover at proxy level).

        Raises RoutingError if no resources are tagged with the given tag.
        """
        candidates = self.registry.resolve_tag(tag)

        if not candidates:
            raise RoutingError(
                f"No resources tagged '{tag}'. "
                f"Available tags: {list(self.registry.all_tags().keys())}"
            )

        results: list[Resolution] = []
        for resource, priority in candidates:
            workers = self.registry.get_workers_for_resource(resource.name)
            for worker in workers:
                results.append(Resolution(tag=tag, resource=resource, worker=worker))

        if not results:
            raise RoutingError(
                f"No workers available for any resource tagged '{tag}' "
                f"(tried {len(candidates)} candidates)"
            )

        logger.info(
            f"Tag '{tag}' resolved to {len(results)} candidate(s), "
            f"primary: '{results[0].resource.name}' on '{results[0].worker.name}'"
        )
        return results


class RoutingError(Exception):
    """Raised when routing fails for a tag."""
