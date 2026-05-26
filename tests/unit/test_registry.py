"""Unit tests for the resource registry."""

import pytest
import yaml
from pathlib import Path

from modelpool.registry import Registry, RegistryError, Resource, Worker, Route


@pytest.fixture
def valid_registry_yaml(tmp_path):
    """Create a minimal valid resources.yaml."""
    data = {
        "resources": {
            "test-resource": {
                "type": "managed",
                "description": "Test resource",
                "size_gb": 10,
                "ctx": 131072,
                "capabilities": ["chat"],
                "workers": ["test-worker"],
                "command": {
                    "binary": "/usr/bin/llama-server",
                    "flags": [
                        ["-m", "/models/test.gguf"],
                        ["-c", "131072"],
                        ["--port", "{inference_port}"],
                    ],
                },
            },
            "test-external": {
                "type": "external",
                "description": "External test",
                "endpoint": "https://api.test.com/v1",
                "auth": {"method": "api_key", "env_var": "TEST_KEY"},
                "model": "test-model",
                "ctx": 128000,
                "capabilities": ["chat"],
                "workers": ["cloud-test"],
            },
        },
        "workers": {
            "test-worker": {
                "host": "192.168.1.100",
                "worker_port": 9100,
                "inference_port": 8080,
                "type": "managed",
                "vram_gb": 24,
                "max_model_gb": 20,
                "default_resource": "test-resource",
            },
            "cloud-test": {
                "type": "external",
            },
        },
        "routing": {
            "chat": {
                "resource": "test-resource",
                "fallback_resource": "test-external",
                "timeout": 60,
                "idle_revert": 300,
                "swap_behavior": "fallback",
            },
        },
    }
    path = tmp_path / "resources.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def registry(valid_registry_yaml):
    return Registry.from_file(valid_registry_yaml)


class TestRegistryParsing:
    def test_load_from_file(self, valid_registry_yaml):
        reg = Registry.from_file(valid_registry_yaml)
        assert len(reg.resources) == 2
        assert len(reg.workers) == 2
        assert len(reg.routes) == 1

    def test_missing_file(self, tmp_path):
        with pytest.raises(RegistryError, match="not found"):
            Registry.from_file(tmp_path / "nonexistent.yaml")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        reg = Registry.from_file(path)
        assert len(reg.resources) == 0


class TestResourceLookup:
    def test_get_resource(self, registry):
        r = registry.get_resource("test-resource")
        assert r.name == "test-resource"
        assert r.type == "managed"
        assert r.is_managed
        assert r.size_gb == 10
        assert r.ctx == 131072
        assert "chat" in r.capabilities
        assert r.binary == "/usr/bin/llama-server"
        assert len(r.flags) == 3

    def test_get_resource_not_found(self, registry):
        with pytest.raises(RegistryError, match="not found"):
            registry.get_resource("nonexistent")

    def test_get_external_resource(self, registry):
        r = registry.get_resource("test-external")
        assert not r.is_managed
        assert r.endpoint == "https://api.test.com/v1"
        assert r.auth.method == "api_key"
        assert r.auth.env_var == "TEST_KEY"
        assert r.model == "test-model"


class TestWorkerLookup:
    def test_get_worker(self, registry):
        w = registry.get_worker("test-worker")
        assert w.name == "test-worker"
        assert w.host == "192.168.1.100"
        assert w.worker_port == 9100
        assert w.inference_port == 8080
        assert w.is_managed
        assert w.max_model_gb == 20

    def test_worker_urls(self, registry):
        w = registry.get_worker("test-worker")
        assert w.worker_url == "http://192.168.1.100:9100"
        assert w.inference_url == "http://192.168.1.100:8080"

    def test_external_worker(self, registry):
        w = registry.get_worker("cloud-test")
        assert not w.is_managed
        assert w.type == "external"

    def test_get_worker_not_found(self, registry):
        with pytest.raises(RegistryError, match="not found"):
            registry.get_worker("nonexistent")


class TestRouteLookup:
    def test_get_route(self, registry):
        route = registry.get_route("chat")
        assert route.task_type == "chat"
        assert route.resource == "test-resource"
        assert route.fallback_resources == ["test-external"]
        assert route.timeout == 60
        assert route.idle_revert == 300
        assert route.swap_behavior == "fallback"

    def test_get_route_not_found(self, registry):
        with pytest.raises(RegistryError, match="No route"):
            registry.get_route("nonexistent")


class TestAssociations:
    def test_get_default_resource(self, registry):
        r = registry.get_default_resource("test-worker")
        assert r.name == "test-resource"

    def test_get_resources_for_worker(self, registry):
        resources = registry.get_resources_for_worker("test-worker")
        assert len(resources) == 1
        assert resources[0].name == "test-resource"

    def test_get_workers_for_resource(self, registry):
        workers = registry.get_workers_for_resource("test-resource")
        assert len(workers) == 1
        assert workers[0].name == "test-worker"


class TestValidation:
    def test_unknown_worker_in_resource(self, tmp_path):
        data = {
            "resources": {
                "bad-resource": {
                    "type": "managed",
                    "workers": ["ghost-worker"],
                    "command": {"binary": "/bin/test", "flags": []},
                },
            },
            "workers": {},
            "routing": {},
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="unknown worker"):
            Registry.from_file(path)

    def test_resource_too_large(self, tmp_path):
        data = {
            "resources": {
                "big-model": {
                    "type": "managed",
                    "size_gb": 50,
                    "workers": ["small-worker"],
                    "command": {"binary": "/bin/test", "flags": []},
                },
            },
            "workers": {
                "small-worker": {
                    "host": "localhost",
                    "type": "managed",
                    "max_model_gb": 20,
                },
            },
            "routing": {},
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="exceeds"):
            Registry.from_file(path)

    def test_route_references_missing_resource(self, tmp_path):
        data = {
            "resources": {},
            "workers": {},
            "routing": {
                "chat": {"resource": "nonexistent"},
            },
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="not found"):
            Registry.from_file(path)
