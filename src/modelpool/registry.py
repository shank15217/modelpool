"""Resource registry - load and validate resources.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class RegistryError(Exception):
    """Raised on invalid or missing registry configuration."""


@dataclass
class Benchmark:
    prompt_eval_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    tested_at: Optional[str] = None


@dataclass
class AuthConfig:
    method: str  # "xai-oauth", "api_key", "none"
    env_var: Optional[str] = None


@dataclass
class Resource:
    name: str
    type: str  # "managed" or "external"
    description: str = ""
    size_gb: float = 0.0
    ctx: int = 0
    capabilities: list[str] = field(default_factory=list)
    workers: list[str] = field(default_factory=list)
    benchmark: Benchmark = field(default_factory=Benchmark)

    # Managed resource fields
    binary: Optional[str] = None
    flags: list[list[str]] = field(default_factory=list)

    # External resource fields
    endpoint: Optional[str] = None
    auth: Optional[AuthConfig] = None
    model: Optional[str] = None  # model name for external requests

    @property
    def is_managed(self) -> bool:
        return self.type == "managed"


@dataclass
class Worker:
    name: str
    host: str = ""
    worker_port: int = 9100
    inference_port: int = 8080
    type: str = "managed"  # "managed" or "external"
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    max_model_gb: float = 0.0
    swap_timeout: int = 120
    drain_timeout: int = 30
    default_resource: Optional[str] = None

    @property
    def worker_url(self) -> str:
        return f"http://{self.host}:{self.worker_port}"

    @property
    def inference_url(self) -> str:
        return f"http://{self.host}:{self.inference_port}"

    @property
    def is_managed(self) -> bool:
        return self.type == "managed"


@dataclass
class Route:
    task_type: str
    resource: str
    fallback_resources: list[str] = field(default_factory=list)
    timeout: int = 120
    idle_revert: int = 300
    swap_behavior: str = "queue"  # "queue" or "fallback"


class Registry:
    """Loads and validates resources.yaml. Provides lookup methods."""

    def __init__(self, data: dict[str, Any]):
        self._raw = data
        self._resources: dict[str, Resource] = {}
        self._workers: dict[str, Worker] = {}
        self._routes: dict[str, Route] = {}

        self._parse(data)
        self._validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "Registry":
        """Load registry from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise RegistryError(f"Registry file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data or not isinstance(data, dict):
            data = {}  # Empty file is valid -- no resources, no workers, no routes
        return cls(data)

    # --- Lookup methods ---

    def get_resource(self, name: str) -> Resource:
        """Get a resource by name. Raises RegistryError if not found."""
        if name not in self._resources:
            raise RegistryError(f"Resource not found: {name}")
        return self._resources[name]

    def get_worker(self, name: str) -> Worker:
        """Get a worker by name. Raises RegistryError if not found."""
        if name not in self._workers:
            raise RegistryError(f"Worker not found: {name}")
        return self._workers[name]

    def get_route(self, task_type: str) -> Route:
        """Get a route by task type. Raises RegistryError if not found."""
        if task_type not in self._routes:
            raise RegistryError(f"No route defined for task type: {task_type}")
        return self._routes[task_type]

    def get_default_resource(self, worker_name: str) -> Resource:
        """Get the default resource for a worker."""
        worker = self.get_worker(worker_name)
        if not worker.default_resource:
            raise RegistryError(f"Worker {worker_name} has no default resource")
        return self.get_resource(worker.default_resource)

    def get_resources_for_worker(self, worker_name: str) -> list[Resource]:
        """Get all resources that can be served by a worker."""
        return [r for r in self._resources.values() if worker_name in r.workers]

    def get_workers_for_resource(self, resource_name: str) -> list[Worker]:
        """Get all workers that can serve a resource."""
        resource = self.get_resource(resource_name)
        return [self._workers[w] for w in resource.workers if w in self._workers]

    @property
    def resources(self) -> dict[str, Resource]:
        return self._resources

    @property
    def workers(self) -> dict[str, Worker]:
        return self._workers

    @property
    def routes(self) -> dict[str, Route]:
        return self._routes

    # --- Parsing ---

    def _parse(self, data: dict) -> None:
        self._parse_resources(data.get("resources", {}))
        self._parse_workers(data.get("workers", {}))
        self._parse_routing(data.get("routing", {}))

    def _parse_resources(self, raw: dict) -> None:
        for name, rdef in raw.items():
            benchmark = Benchmark(
                prompt_eval_tps=rdef.get("benchmark", {}).get("prompt_eval_tps"),
                generation_tps=rdef.get("benchmark", {}).get("generation_tps"),
                tested_at=rdef.get("benchmark", {}).get("tested_at"),
            )

            auth = None
            if "auth" in rdef:
                auth = AuthConfig(
                    method=rdef["auth"].get("method", "none"),
                    env_var=rdef["auth"].get("env_var"),
                )

            resource = Resource(
                name=name,
                type=rdef.get("type", "managed"),
                description=rdef.get("description", ""),
                size_gb=rdef.get("size_gb", 0.0),
                ctx=rdef.get("ctx", 0),
                capabilities=rdef.get("capabilities", []),
                workers=rdef.get("workers", []),
                benchmark=benchmark,
                binary=rdef.get("command", {}).get("binary"),
                flags=rdef.get("command", {}).get("flags", []),
                endpoint=rdef.get("endpoint"),
                auth=auth,
                model=rdef.get("model"),
            )
            self._resources[name] = resource

    def _parse_workers(self, raw: dict) -> None:
        for name, wdef in raw.items():
            worker = Worker(
                name=name,
                host=wdef.get("host", ""),
                worker_port=wdef.get("worker_port", 9100),
                inference_port=wdef.get("inference_port", 8080),
                type=wdef.get("type", "managed"),
                vram_gb=wdef.get("vram_gb", 0.0),
                ram_gb=wdef.get("ram_gb", 0.0),
                max_model_gb=wdef.get("max_model_gb", 0.0),
                swap_timeout=wdef.get("swap_timeout", 120),
                drain_timeout=wdef.get("drain_timeout", 30),
                default_resource=wdef.get("default_resource"),
            )
            self._workers[name] = worker

    def _parse_routing(self, raw: dict) -> None:
        for task_type, rdef in raw.items():
            # Collect fallback chain
            fallbacks = []
            for key in sorted(rdef.keys()):
                if key.startswith("fallback_resource") and rdef[key] and rdef[key] != "none":
                    fallbacks.append(rdef[key])

            route = Route(
                task_type=task_type,
                resource=rdef["resource"],
                fallback_resources=fallbacks,
                timeout=rdef.get("timeout", 120),
                idle_revert=rdef.get("idle_revert", 300),
                swap_behavior=rdef.get("swap_behavior", "queue"),
            )
            self._routes[task_type] = route

    # --- Validation ---

    def _validate(self) -> None:
        """Validate registry consistency."""
        errors = []

        # Validate resources
        for name, res in self._resources.items():
            if res.is_managed:
                if not res.binary:
                    errors.append(f"Resource '{name}': managed resource missing command.binary")
                if not res.flags:
                    errors.append(f"Resource '{name}': managed resource missing command.flags")
                if not res.workers:
                    errors.append(f"Resource '{name}': no workers assigned")
            else:
                if not res.endpoint:
                    errors.append(f"Resource '{name}': external resource missing endpoint")

            # Validate worker references
            for wname in res.workers:
                if wname not in self._workers:
                    errors.append(f"Resource '{name}': references unknown worker '{wname}'")

        # Validate workers
        for name, worker in self._workers.items():
            if worker.is_managed and not worker.host:
                errors.append(f"Worker '{name}': managed worker missing host")
            if worker.default_resource:
                if worker.default_resource not in self._resources:
                    errors.append(
                        f"Worker '{name}': default_resource '{worker.default_resource}' not found"
                    )

        # Validate size constraints
        for name, res in self._resources.items():
            if res.is_managed:
                for wname in res.workers:
                    if wname in self._workers:
                        worker = self._workers[wname]
                        if worker.max_model_gb and res.size_gb > worker.max_model_gb:
                            errors.append(
                                f"Resource '{name}' ({res.size_gb}GB) exceeds "
                                f"worker '{wname}' capacity ({worker.max_model_gb}GB)"
                            )

        # Validate routing
        for task_type, route in self._routes.items():
            if route.resource not in self._resources:
                errors.append(
                    f"Route '{task_type}': resource '{route.resource}' not found"
                )
            for fb in route.fallback_resources:
                if fb not in self._resources:
                    errors.append(
                        f"Route '{task_type}': fallback resource '{fb}' not found"
                    )

        if errors:
            raise RegistryError(
                "Registry validation failed:\n  - " + "\n  - ".join(errors)
            )
