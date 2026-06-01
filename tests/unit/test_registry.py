"""Unit tests for the resource registry and tag routing."""

import pytest
import yaml
from pathlib import Path

from modelpool.registry import Registry, RegistryError, Resource, Worker


@pytest.fixture
def valid_registry_yaml(tmp_path):
    """Create a minimal valid resources.yaml with tags."""
    data = {
        "resources": {
            "test-gpu-model": {
                "type": "managed",
                "description": "Test GPU resource",
                "size_gb": 10,
                "ctx": 131072,
                "capabilities": ["chat", "code"],
                "workers": ["test-worker"],
                "tags": {
                    "chat": 1,
                    "code": 1,
                    "compression": 2,
                },
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
                "capabilities": ["chat", "vision"],
                "workers": ["cloud-test"],
                "tags": {
                    "chat": 2,
                    "vision": 1,
                },
            },
            "test-cpu-model": {
                "type": "managed",
                "description": "Test CPU resource",
                "size_gb": 10,
                "ctx": 65536,
                "capabilities": ["compression"],
                "workers": ["test-worker"],
                "tags": {
                    "compression": 1,
                    "title": 1,
                },
                "command": {
                    "binary": "/usr/bin/llama-server",
                    "flags": [
                        ["-m", "/models/cpu-test.gguf"],
                        ["-c", "65536"],
                        ["--port", "{inference_port}"],
                    ],
                },
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
                "default_resource": "test-gpu-model",
                "pool_secret": "test-secret-123",
            },
            "cloud-test": {
                "type": "external",
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
        assert len(reg.resources) == 3
        assert len(reg.workers) == 2

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
        r = registry.get_resource("test-gpu-model")
        assert r.name == "test-gpu-model"
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


class TestTagLookup:
    def test_resolve_tag_returns_sorted_by_priority(self, registry):
        """Resources tagged 'chat' should return sorted by priority."""
        matches = registry.resolve_tag("chat")
        assert len(matches) == 2
        # Priority 1 first (test-gpu-model), then 2 (test-external)
        assert matches[0][0].name == "test-gpu-model"
        assert matches[0][1] == 1
        assert matches[1][0].name == "test-external"
        assert matches[1][1] == 2

    def test_resolve_tag_single_match(self, registry):
        matches = registry.resolve_tag("vision")
        assert len(matches) == 1
        assert matches[0][0].name == "test-external"
        assert matches[0][1] == 1

    def test_resolve_tag_unknown(self, registry):
        matches = registry.resolve_tag("nonexistent-tag")
        assert matches == []

    def test_all_tags(self, registry):
        tags = registry.all_tags()
        assert "chat" in tags
        assert "compression" in tags
        assert "vision" in tags
        assert "code" in tags
        assert "title" in tags
        # Chat should have 2 resources sorted by priority
        assert len(tags["chat"]) == 2
        assert tags["chat"][0] == ("test-gpu-model", 1)
        assert tags["chat"][1] == ("test-external", 2)
        # Compression should have 2 resources
        assert len(tags["compression"]) == 2
        assert tags["compression"][0] == ("test-cpu-model", 1)
        assert tags["compression"][1] == ("test-gpu-model", 2)

    def test_tags_on_resource(self, registry):
        r = registry.get_resource("test-gpu-model")
        assert r.tags == {"chat": 1, "code": 1, "compression": 2}

    def test_external_resource_tags(self, registry):
        r = registry.get_resource("test-external")
        assert r.tags == {"chat": 2, "vision": 1}


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

    def test_worker_pool_secret(self, registry):
        w = registry.get_worker("test-worker")
        assert w.pool_secret == "test-secret-123"

    def test_worker_no_secret(self, registry):
        w = registry.get_worker("cloud-test")
        assert w.pool_secret is None

    def test_external_worker(self, registry):
        w = registry.get_worker("cloud-test")
        assert not w.is_managed
        assert w.type == "external"

    def test_get_worker_not_found(self, registry):
        with pytest.raises(RegistryError, match="not found"):
            registry.get_worker("nonexistent")


class TestAssociations:
    def test_get_default_resource(self, registry):
        r = registry.get_default_resource("test-worker")
        assert r.name == "test-gpu-model"

    def test_get_resources_for_worker(self, registry):
        resources = registry.get_resources_for_worker("test-worker")
        assert len(resources) == 2
        names = {r.name for r in resources}
        assert "test-gpu-model" in names
        assert "test-cpu-model" in names

    def test_get_workers_for_resource(self, registry):
        workers = registry.get_workers_for_resource("test-gpu-model")
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
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="exceeds"):
            Registry.from_file(path)

    def test_invalid_tag_priority(self, tmp_path):
        data = {
            "resources": {
                "bad-tags": {
                    "type": "managed",
                    "workers": ["test-worker"],
                    "tags": {"chat": 0},  # must be >= 1
                    "command": {"binary": "/bin/test", "flags": []},
                },
            },
            "workers": {
                "test-worker": {"host": "localhost", "type": "managed"},
            },
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="positive integer"):
            Registry.from_file(path)

    def test_tag_priority_must_be_integer(self, tmp_path):
        data = {
            "resources": {
                "bad-tags": {
                    "type": "managed",
                    "workers": ["test-worker"],
                    "tags": {"chat": "high"},  # must be int
                    "command": {"binary": "/bin/test", "flags": []},
                },
            },
            "workers": {
                "test-worker": {"host": "localhost", "type": "managed"},
            },
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(RegistryError, match="positive integer"):
            Registry.from_file(path)

    def test_no_routes_section_required(self, tmp_path):
        """resources.yaml without a routing section should be valid."""
        data = {
            "resources": {
                "simple": {
                    "type": "external",
                    "endpoint": "https://api.test.com/v1",
                    "workers": ["cloud"],
                    "tags": {"chat": 1},
                },
            },
            "workers": {
                "cloud": {"type": "external"},
            },
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        reg = Registry.from_file(path)
        assert len(reg.resources) == 1
        assert reg.resolve_tag("chat")


class TestNewFields:
    """Tests for generalist and max_concurrent_models fields."""

    def test_resource_generalist_defaults_false(self, registry):
        """Resources without generalist flag should default to False."""
        for res in registry.resources.values():
            assert hasattr(res, "generalist")
            # In our test data none are marked generalist yet
            assert res.generalist is False

    def test_worker_max_concurrent_models_default(self, registry):
        """Workers should have max_concurrent_models with default 1."""
        for worker in registry.workers.values():
            assert hasattr(worker, "max_concurrent_models")
            assert worker.max_concurrent_models >= 1

    def test_generalist_can_be_set_true(self, tmp_path):
        """A resource can be marked as generalist."""
        data = {
            "resources": {
                "gen-model": {
                    "type": "managed",
                    "size_gb": 16,
                    "ctx": 131072,
                    "workers": ["gpu-worker"],
                    "tags": {"chat": 1},
                    "generalist": True,
                    "command": {"binary": "/bin/test", "flags": [["-m", "test.gguf"]]},
                }
            },
            "workers": {
                "gpu-worker": {
                    "host": "127.0.0.1",
                    "max_concurrent_models": 2,
                }
            },
        }
        path = tmp_path / "resources.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        reg = Registry.from_file(path)
        res = reg.resources["gen-model"]
        assert res.generalist is True
        w = reg.workers["gpu-worker"]
        assert w.max_concurrent_models == 2
