"""Unit tests for the pool router."""

import pytest
import yaml
from unittest.mock import patch, MagicMock

from modelpool.registry import Registry
from modelpool.pool.router import Router, Resolution, RoutingError


@pytest.fixture
def registry_yaml(tmp_path):
    """Create a test resources.yaml with 2 managed + 2 external resources."""
    data = {
        "resources": {
            "resource-a": {
                "type": "managed",
                "description": "Managed A",
                "size_gb": 10,
                "ctx": 131072,
                "capabilities": ["chat"],
                "workers": ["worker-1"],
                "command": {
                    "binary": "/usr/bin/llama-server",
                    "flags": [["-m", "/models/a.gguf"], ["--port", "{inference_port}"]],
                },
            },
            "resource-b": {
                "type": "managed",
                "description": "Managed B",
                "size_gb": 15,
                "ctx": 262144,
                "capabilities": ["compression"],
                "workers": ["worker-1"],
                "command": {
                    "binary": "/usr/bin/llama-server",
                    "flags": [["-m", "/models/b.gguf"], ["--port", "{inference_port}"]],
                },
            },
            "external-cloud": {
                "type": "external",
                "description": "Cloud API",
                "endpoint": "https://api.cloud.com/v1",
                "auth": {"method": "api_key", "env_var": "CLOUD_KEY"},
                "model": "cloud-model",
                "ctx": 256000,
                "capabilities": ["chat", "compression"],
                "workers": ["cloud-virtual"],
            },
        },
        "workers": {
            "worker-1": {
                "host": "192.168.1.100",
                "worker_port": 9100,
                "inference_port": 8080,
                "type": "managed",
                "vram_gb": 32,
                "max_model_gb": 28,
                "default_resource": "resource-a",
            },
            "cloud-virtual": {
                "type": "external",
            },
        },
        "routing": {
            "chat": {
                "resource": "resource-a",
                "timeout": 60,
                "idle_revert": 0,
                "swap_behavior": "fallback",
            },
            "compression": {
                "resource": "resource-b",
                "fallback_resource": "external-cloud",
                "timeout": 120,
                "idle_revert": 300,
                "swap_behavior": "queue",
            },
            "external-only": {
                "resource": "external-cloud",
                "timeout": 30,
                "swap_behavior": "fallback",
            },
        },
    }
    path = tmp_path / "resources.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def registry(registry_yaml):
    return Registry.from_file(registry_yaml)


def _mock_worker_status(state="ready", loaded_resource="resource-a"):
    """Helper to mock worker status responses."""
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "state": state,
            "loaded_resource": loaded_resource,
            "pid": 12345,
        }
        return resp
    return mock_get


class TestRouterResolution:
    """Test the core resolve() method."""

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_already_loaded(self, mock_get, registry):
        """Resource already on worker -> no swap needed."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "state": "ready",
                "loaded_resource": "resource-a",
                "pid": 123,
            }),
        )
        router = Router(registry)
        result = router.resolve("chat")

        assert result.resource.name == "resource-a"
        assert result.worker.name == "worker-1"
        assert result.needs_swap is False
        assert result.task_type == "chat"

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_needs_swap(self, mock_get, registry):
        """Worker has wrong resource -> swap needed."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "state": "ready",
                "loaded_resource": "resource-a",  # has A, need B
                "pid": 123,
            }),
        )
        router = Router(registry)
        result = router.resolve("compression")

        assert result.resource.name == "resource-b"
        assert result.worker.name == "worker-1"
        assert result.needs_swap is True
        assert result.currently_loaded == "resource-a"

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_external(self, mock_get, registry):
        """External resource resolves immediately, no worker check."""
        router = Router(registry)
        result = router.resolve("external-only")

        assert result.resource.name == "external-cloud"
        assert result.worker.name == "cloud-virtual"
        assert result.needs_swap is False
        assert result.is_external is True
        assert result.inference_url == "https://api.cloud.com/v1"
        # Should not have called requests.get for worker status
        # (external workers don't have status endpoints)

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_fallback(self, mock_get, registry):
        """Primary resource unavailable -> fallback to external."""
        # Make worker unreachable
        mock_get.side_effect = Exception("connection refused")

        router = Router(registry)
        # compression primary is resource-b, fallback is external-cloud
        result = router.resolve("compression")

        # Should fall back to external-cloud since worker is unreachable
        assert result.resource.name == "external-cloud"
        assert result.is_external is True

    @patch("modelpool.pool.router.requests.get")
    def test_resolve_no_route(self, mock_get, registry):
        """Unknown task type raises error."""
        router = Router(registry)
        with pytest.raises(Exception, match="No route"):
            router.resolve("nonexistent-task")


class TestRouterResolutionProperties:
    """Test Resolution dataclass properties."""

    def test_inference_url_managed(self, registry):
        res = Resolution(
            task_type="chat",
            resource=registry.get_resource("resource-a"),
            worker=registry.get_worker("worker-1"),
            needs_swap=False,
        )
        assert res.inference_url == "http://192.168.1.100:8080"
        assert res.worker_api_url == "http://192.168.1.100:9100"
        assert res.is_external is False

    def test_inference_url_external(self, registry):
        res = Resolution(
            task_type="chat",
            resource=registry.get_resource("external-cloud"),
            worker=registry.get_worker("cloud-virtual"),
            needs_swap=False,
        )
        assert res.inference_url == "https://api.cloud.com/v1"
        assert res.is_external is True


class TestFallbackChain:
    """Test fallback chain building."""

    def test_build_fallback_chain(self, registry):
        router = Router(registry)
        route = registry.get_route("compression")
        chain = router.build_fallback_chain(route)

        assert len(chain) == 1
        assert chain[0][0].name == "external-cloud"
        assert chain[0][1].name == "cloud-virtual"

    def test_empty_fallback_chain(self, registry):
        router = Router(registry)
        route = registry.get_route("chat")
        chain = router.build_fallback_chain(route)
        assert len(chain) == 0  # chat route has no fallbacks


class TestWorkerStatuses:
    """Test get_all_worker_statuses."""

    @patch("modelpool.pool.router.requests.get")
    def test_all_statuses(self, mock_get, registry):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "state": "ready",
                "loaded_resource": "resource-a",
            }),
        )
        router = Router(registry)
        statuses = router.get_all_worker_statuses()

        assert "worker-1" in statuses
        assert statuses["worker-1"]["state"] == "ready"
        assert "cloud-virtual" in statuses
        assert statuses["cloud-virtual"]["state"] == "external"

    @patch("modelpool.pool.router.requests.get")
    def test_unreachable_worker(self, mock_get, registry):
        mock_get.side_effect = Exception("timeout")
        router = Router(registry)
        statuses = router.get_all_worker_statuses()

        assert statuses["worker-1"]["state"] == "unreachable"
        assert statuses["cloud-virtual"]["state"] == "external"
