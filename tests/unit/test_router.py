"""Unit tests for the pool router (tag-based resolution)."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

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
                "default_resource": "gpu-chat-model",
            },
            "cpu-worker": {
                "host": "192.168.1.200",
                "worker_port": 9100,
                "inference_port": 8081,
                "type": "managed",
                "ram_gb": 48,
                "max_model_gb": 40,
                "default_resource": "cpu-compress-model",
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
    def test_resolve_chat_priorities(self, router):
        """Chat tag should prefer gpu-chat-model (priority 1)."""
        matches = router.registry.resolve_tag("chat")
        assert len(matches) == 3
        assert matches[0][0].name == "gpu-chat-model"  # priority 1
        assert matches[1][0].name == "gpu-compress-model"  # priority 2
        assert matches[2][0].name == "cloud-model"  # priority 3

    def test_resolve_compression_priorities(self, router):
        """Compression tag should prefer gpu-compress-model (priority 1)."""
        matches = router.registry.resolve_tag("compression")
        assert len(matches) == 3
        assert matches[0][0].name == "gpu-compress-model"  # priority 1
        assert matches[1][0].name == "cpu-compress-model"  # priority 2
        assert matches[2][0].name == "cloud-model"  # priority 3

    def test_resolve_vision_only_cloud(self, router):
        """Vision tag only has cloud-model."""
        matches = router.registry.resolve_tag("vision")
        assert len(matches) == 1
        assert matches[0][0].name == "cloud-model"

    def test_resolve_unknown_tag_raises(self, router):
        with pytest.raises(RoutingError, match="No resources tagged"):
            router.resolve("nonexistent-tag")

    def test_all_tags(self, router):
        tags = router.tags
        assert "chat" in tags
        assert "compression" in tags
        assert "code" in tags
        assert "agentic" in tags
        assert "title" in tags
        assert "vision" in tags


class TestResolveWithWorkerStatus:
    @patch("modelpool.pool.router.requests.get")
    def test_resolve_with_resource_already_loaded(self, mock_get, router):
        """If the model is already loaded, no swap needed."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gpu-chat-model",
                          "loaded_models_count": 0},
        )

        resolution = router.resolve("chat")
        assert resolution.resource.name == "gpu-chat-model"
        assert resolution.worker.name == "gpu-worker"
        assert resolution.needs_swap is False
        assert resolution.currently_loaded == "gpu-chat-model"

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_idle_worker_cold_load(self, mock_get, router):
        """Idle worker gets cold load."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "idle", "loaded_resource": None},
        )

        resolution = router.resolve("chat")
        assert resolution.resource.name == "gpu-chat-model"
        assert resolution.worker.name == "gpu-worker"
        assert resolution.needs_swap is True
        assert resolution.currently_loaded is None

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_swap_needed(self, mock_get, router):
        """Worker has wrong model loaded -> needs swap."""
        def side_effect(url, **kwargs):
            m = MagicMock(status_code=200)
            if "192.168.1.100" in url:
                m.json = lambda: {"state": "ready", "loaded_resource": "gpu-compress-model",
                                  "loaded_models_count": 1}
            else:
                m.json = lambda: {"state": "idle", "loaded_resource": None,
                                  "loaded_models_count": 0}
            return m
        mock_get.side_effect = side_effect

        resolution = router.resolve("chat")
        assert resolution.resource.name == "gpu-chat-model"
        assert resolution.worker.name == "gpu-worker"
        assert resolution.needs_swap is True
        assert resolution.currently_loaded == "gpu-compress-model"

    @patch("modelpool.pool.router.requests.get")
    def test_fallback_to_cloud_when_worker_unreachable(self, mock_get, router):
        """Worker unreachable -> falls back to cloud."""
        mock_get.return_value = MagicMock(
            status_code=None,
            raise_exception=Exception("Connection refused"),
        )

        resolution = router.resolve("chat")
        assert resolution.resource.name == "cloud-model"
        assert resolution.is_external
        assert resolution.needs_swap is False

    @patch("modelpool.pool.router.requests.get")
    def test_fallback_chain_populated(self, mock_get, router):
        """Resolution should have fallback chain from lower-priority resources."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gpu-chat-model",
                          "loaded_models_count": 0},
        )

        resolution = router.resolve("chat")
        # Should have fallback chain from priority 2 and 3
        assert len(resolution.fallback_chain) >= 1


class TestResolutionProperties:
    @patch("modelpool.pool.router.requests.get")
    def test_managed_resolution(self, mock_get, router):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "idle", "loaded_resource": None},
        )

        resolution = router.resolve("chat")
        assert not resolution.is_external
        assert resolution.inference_url == "http://192.168.1.100:8080"
        assert resolution.worker_api_url == "http://192.168.1.100:9100"

    def test_external_resolution(self, router):
        # External doesn't query worker status
        resolution = router.resolve("vision")
        assert resolution.is_external
        assert resolution.inference_url == "https://api.cloud.com/v1"
        assert resolution.worker.name == "cloud-worker"
