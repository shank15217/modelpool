"""Unit tests for the pool router (tag-based resolution).

Simplified for Architecture A: synchronous, in-memory routing only.
No HTTP calls, no mocks needed.
"""

import pytest
from modelpool.registry import Registry
from modelpool.pool.router import Router, Resolution, RoutingError


@pytest.fixture
def registry_data():
    return {
        "resources": {
            "gpu-chat-model": {
                "type": "managed",
                "size_gb": 16,
                "ctx": 131072,
                "capabilities": ["chat", "code"],
                "workers": ["gpu-worker"],
                "tags": {"chat": 1, "code": 1, "agentic": 1},
                "command": {"binary": "/usr/bin/llama-server", "flags": [["--port", "{inference_port}"]]},
            },
            "gpu-compress-model": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 262144,
                "capabilities": ["compression"],
                "workers": ["gpu-worker"],
                "tags": {"compression": 1, "chat": 2},
                "command": {"binary": "/usr/bin/llama-server", "flags": [["--port", "{inference_port}"]]},
            },
            "cpu-compress-model": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 65536,
                "capabilities": ["compression"],
                "workers": ["cpu-worker"],
                "tags": {"compression": 2, "title": 1},
                "command": {"binary": "/usr/bin/llama-server", "flags": [["--port", "{inference_port}"]]},
            },
            "cloud-model": {
                "type": "external",
                "endpoint": "https://api.cloud.com/v1",
                "auth": {"method": "api_key", "env_var": "CLOUD_KEY"},
                "model": "cloud-v1",
                "ctx": 128000,
                "capabilities": ["chat", "vision"],
                "workers": ["cloud-worker"],
                "tags": {"chat": 3, "vision": 1, "compression": 3},
            },
        },
        "workers": {
            "gpu-worker": {
                "host": "192.168.1.100",
                "worker_port": 9100,
                "inference_port": 8080,
                "type": "managed",
                "vram_gb": 32,
                "max_model_gb": 28,
            },
            "cpu-worker": {
                "host": "192.168.1.200",
                "worker_port": 9100,
                "inference_port": 8081,
                "type": "managed",
                "ram_gb": 48,
                "max_model_gb": 40,
            },
            "cloud-worker": {
                "type": "external",
            },
        },
    }


@pytest.fixture
def registry(registry_data):
    return Registry(registry_data)


@pytest.fixture
def router(registry):
    return Router(registry)


class TestTagResolution:
    """Tests for synchronous tag-based resolution (no HTTP calls)."""

    def test_resolve_chat_returns_candidates_sorted(self, router):
        """Chat tag should return candidates sorted by priority."""
        candidates = router.resolve("chat")
        assert len(candidates) == 3
        # Priority order: gpu-chat-model(1) < gpu-compress-model(2) < cloud-model(3)
        assert candidates[0].resource.name == "gpu-chat-model"
        assert candidates[1].resource.name == "gpu-compress-model"
        assert candidates[2].resource.name == "cloud-model"

    def test_resolve_compression_returns_sorted(self, router):
        """Compression tag should return sorted candidates."""
        candidates = router.resolve("compression")
        assert len(candidates) == 3
        assert candidates[0].resource.name == "gpu-compress-model"
        assert candidates[1].resource.name == "cpu-compress-model"
        assert candidates[2].resource.name == "cloud-model"

    def test_resolve_vision_only_cloud(self, router):
        """Vision tag only has cloud-model."""
        candidates = router.resolve("vision")
        assert len(candidates) == 1
        assert candidates[0].resource.name == "cloud-model"

    def test_resolve_unknown_tag_raises(self, router):
        """Unknown tag raises RoutingError."""
        with pytest.raises(RoutingError, match="No resources tagged"):
            router.resolve("nonexistent-tag")

    def test_all_tags(self, router):
        """All known tags are accessible."""
        tags = router.tags
        assert "chat" in tags
        assert "compression" in tags
        assert "code" in tags
        assert "agentic" in tags
        assert "title" in tags
        assert "vision" in tags

    def test_resolve_is_synchronous(self, router):
        """resolve() returns a plain list, no await needed."""
        result = router.resolve("chat")
        assert isinstance(result, list)

    def test_resolve_returns_resolutions(self, router):
        """Each candidate is a Resolution with tag, resource, worker."""
        candidates = router.resolve("chat")
        for c in candidates:
            assert isinstance(c, Resolution)
            assert c.tag == "chat"
            assert c.resource is not None
            assert c.worker is not None

    def test_resolve_external_resource(self, router):
        """External resource resolution."""
        candidates = router.resolve("vision")
        assert candidates[0].is_external
        assert candidates[0].inference_url == "https://api.cloud.com/v1"

    def test_resolve_managed_resource(self, router):
        """Managed resource has correct inference URL."""
        candidates = router.resolve("chat")
        primary = candidates[0]
        assert not primary.is_external
        assert primary.inference_url == "http://192.168.1.100:8080"

    def test_resolve_with_multiple_workers(self):
        """Resources with multiple workers get multiple candidates."""
        data = {
            "resources": {
                "shared-model": {
                    "type": "managed",
                    "size_gb": 10,
                    "workers": ["worker-a", "worker-b"],
                    "tags": {"chat": 1},
                    "command": {"binary": "/bin/test", "flags": [["-m", "x.gguf"]]},
                },
            },
            "workers": {
                "worker-a": {"host": "10.0.0.1", "inference_port": 8080},
                "worker-b": {"host": "10.0.0.2", "inference_port": 8081},
            },
        }
        reg = Registry(data)
        r = Router(reg)
        candidates = r.resolve("chat")
        assert len(candidates) == 2
        assert candidates[0].worker.name == "worker-a"
        assert candidates[1].worker.name == "worker-b"

    def test_tag_priority_ordering(self, router):
        """Lower priority number comes first."""
        candidates = router.resolve("chat")
        priorities = []
        for c in candidates:
            priorities.append(c.resource.tags["chat"])
        assert priorities == sorted(priorities)


class TestResolutionProperties:
    """Tests for Resolution dataclass properties."""

    def test_is_external_for_cloud(self, router):
        candidates = router.resolve("vision")
        assert candidates[0].is_external

    def test_is_not_external_for_managed(self, router):
        candidates = router.resolve("chat")
        assert not candidates[0].is_external

    def test_inference_url_external(self, router):
        candidates = router.resolve("vision")
        assert candidates[0].inference_url == "https://api.cloud.com/v1"

    def test_inference_url_managed(self, router):
        candidates = router.resolve("chat")
        assert candidates[0].inference_url == "http://192.168.1.100:8080"

    def test_resolution_has_no_swap_fields(self, router):
        """Architecture A: Resolution has no needs_swap, currently_loaded, fallback_chain."""
        candidates = router.resolve("chat")
        r = candidates[0]
        assert not hasattr(r, "needs_swap")
        assert not hasattr(r, "currently_loaded")
        assert not hasattr(r, "fallback_chain")
