"""Pool routing behavior tests.

Tests the generalist preference, max_concurrent_models enforcement,
and no-swap-when-busy policy using mocked worker statuses.
"""

import pytest
from unittest.mock import patch, MagicMock

from modelpool.registry import Registry
from modelpool.pool.router import Router, RoutingError


def _make_registry_data(**overrides):
    """Build a registry with a generalist GPU model, specialist GPU model,
    and a CPU fallback. Mirrors the real homelab topology."""
    data = {
        "resources": {
            "gpu-27b": {
                "type": "managed",
                "size_gb": 16,
                "ctx": 131072,
                "capabilities": ["chat", "code", "agentic", "reasoning"],
                "workers": ["hwrouter"],
                "tags": {"chat": 1, "agentic": 1, "code": 1, "reasoning": 1},
                "generalist": True,
                "command": {"binary": "/bin/llama-server", "flags": [["-m", "27b.gguf"]]},
            },
            "gpu-35b": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 262144,
                "capabilities": ["compression", "title", "summarize"],
                "workers": ["hwrouter"],
                "tags": {"compression": 1, "title": 1, "summarize": 1, "chat": 2},
                "command": {"binary": "/bin/llama-server", "flags": [["-m", "35b.gguf"]]},
            },
            "cpu-35b": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 65536,
                "capabilities": ["compression", "title"],
                "workers": ["pvellm"],
                "tags": {"compression": 2, "title": 2, "chat": 3},
                "command": {"binary": "/bin/llama-server", "flags": [["-m", "35b-cpu.gguf"]]},
            },
        },
        "workers": {
            "hwrouter": {
                "host": "192.168.35.185",
                "worker_port": 9100,
                "inference_port": 8080,
                "type": "managed",
                "max_concurrent_models": 1,
                "default_resource": "gpu-27b",
            },
            "pvellm": {
                "host": "192.168.35.17",
                "worker_port": 9100,
                "inference_port": 8081,
                "type": "managed",
                "max_concurrent_models": 1,
                "default_resource": "cpu-35b",
            },
        },
    }
    for k, v in overrides.items():
        data[k] = v
    return data


@pytest.fixture
def registry():
    return Registry(_make_registry_data())


@pytest.fixture
def router(registry):
    return Router(registry)


def _mock_status(worker_name, state, loaded_resource, loaded_models_count=0):
    """Helper: return a mock function that returns specific status for a worker."""
    def _get_status(worker, **kwargs):
        if worker.name == worker_name:
            return {"state": state, "loaded_resource": loaded_resource,
                    "loaded_models_count": loaded_models_count}
        return {"state": "idle", "loaded_resource": None, "loaded_models_count": 0}
    return _get_status


def _mock_multi_status(status_map):
    """Helper: return a mock function that returns different statuses per worker.
    status_map = {"hwrouter": ("ready", "gpu-27b", 0), "pvellm": ("idle", None, 0)}
    """
    def _get_status(worker, **kwargs):
        if worker.name in status_map:
            state, loaded, count = status_map[worker.name]
            return {"state": state, "loaded_resource": loaded,
                    "loaded_models_count": count}
        return {"state": "idle", "loaded_resource": None, "loaded_models_count": 0}
    return _get_status


# =========================================================================
# 1. Generalist preference tests
# =========================================================================

class TestGeneralistPreference:
    """When a generalist resource is loaded, prefer it for any tag."""

    @patch("modelpool.pool.router.requests.get")
    def test_generalist_loaded_serves_any_tag(self, mock_get, router):
        """27B (generalist) is loaded -> compression request uses it, no swap."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gpu-27b",
                          "loaded_models_count": 1},
        )
        res = router.resolve("compression")
        # Generalist 27B is loaded and has capacity -> should use it
        assert res.resource.generalist is True
        assert res.needs_swap is False

    @patch("modelpool.pool.router.requests.get")
    def test_generalist_loaded_serves_title(self, mock_get, router):
        """27B (generalist) is loaded -> title request uses it, no swap."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gpu-27b",
                          "loaded_models_count": 1},
        )
        res = router.resolve("title")
        assert res.resource.generalist is True
        assert res.needs_swap is False

    @patch("modelpool.pool.router.requests.get")
    def test_generalist_loaded_serves_chat(self, mock_get, router):
        """27B (generalist) is loaded -> chat request uses it (it's also tagged chat:1)."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gpu-27b",
                          "loaded_models_count": 1},
        )
        res = router.resolve("chat")
        assert res.resource.name == "gpu-27b"
        assert res.needs_swap is False

    @patch("modelpool.pool.router.requests.get")
    def test_no_generalist_loaded_uses_tag_priority(self, mock_get, router):
        """When nothing is loaded, use normal tag priority ordering."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "idle", "loaded_resource": None,
                          "loaded_models_count": 0},
        )
        res = router.resolve("compression")
        # Should pick gpu-35b (priority 1 for compression) since nothing is loaded
        assert res.resource.name == "gpu-35b"
        assert res.needs_swap is True

    @patch("modelpool.pool.router.requests.get")
    def test_specialist_loaded_but_higher_priority_wins_with_swap(self, mock_get, router):
        """35B loaded with chat:2, but 27B (chat:1) is higher priority -> swap to 27B."""
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            if "192.168.35.185" in url:
                mock_resp.json = lambda: {
                    "state": "ready", "loaded_resource": "gpu-35b",
                    "loaded_models_count": 0,
                }
            elif "192.168.35.17" in url:
                mock_resp.json = lambda: {
                    "state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0,
                }
            return mock_resp
        mock_get.side_effect = side_effect
        # 27B (generalist, chat:1) is higher priority than 35B (chat:2)
        # Generalist is NOT loaded, so preference doesn't apply
        # Router picks best priority (27B) and swaps from 35B
        res = router.resolve("chat")
        assert res.resource.name == "gpu-27b"
        assert res.resource.generalist is True
        assert res.needs_swap is True
        assert res.currently_loaded == "gpu-35b"


# =========================================================================
# 2. max_concurrent_models enforcement
# =========================================================================

class TestMaxConcurrentModels:
    """Workers at max_concurrent_models should not receive new loads/swaps."""

    @patch("modelpool.pool.router.requests.get")
    def test_worker_at_capacity_skipped(self, mock_get, router):
        """Worker with loaded_models_count >= max_concurrent_models is skipped."""
        # hwrouter has max_concurrent_models=1, and already has 1 model loaded
        # (a different model than what we're requesting)
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            if "hwrouter" in url or "192.168.35.185" in url:
                mock_resp.json = lambda: {
                    "state": "ready", "loaded_resource": "gpu-35b",
                    "loaded_models_count": 1,  # at capacity
                }
            elif "pvellm" in url or "192.168.35.17" in url:
                mock_resp.json = lambda: {
                    "state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0,
                }
            return mock_resp

        mock_get.side_effect = side_effect
        res = router.resolve("chat")
        # hwrouter is at capacity with 35B loaded (which has chat:2)
        # Should fall through to next available option
        assert res is not None

    @patch("modelpool.pool.router.requests.get")
    def test_worker_under_capacity_accepted(self, mock_get, router):
        """Worker with loaded_models_count < max_concurrent_models can load."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "idle", "loaded_resource": None,
                          "loaded_models_count": 0},
        )
        res = router.resolve("chat")
        assert res is not None
        assert res.needs_swap is True  # cold load needed


# =========================================================================
# 3. No-swap-when-busy (rug pull protection)
# =========================================================================

class TestNoSwapWhenBusy:
    """A worker that is serving requests should not be swapped."""

    @patch("modelpool.pool.router.requests.get")
    def test_busy_worker_not_swapped_for_different_tag(self, mock_get, router):
        """Worker in draining/loading state is skipped, falls to CPU."""
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            if "192.168.35.185" in url:
                # hwrouter: actively draining (busy, cannot accept swap)
                mock_resp.json = lambda: {
                    "state": "draining", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 1,
                }
            elif "192.168.35.17" in url:
                # pvellm: idle, can take compression
                mock_resp.json = lambda: {
                    "state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0,
                }
            return mock_resp

        mock_get.side_effect = side_effect
        res = router.resolve("compression")
        # hwrouter is draining -> not ready -> swap blocked -> skip
        # Falls to cpu-35b on pvellm
        assert res.resource.name == "cpu-35b"
        assert res.worker.name == "pvellm"
        assert res.needs_swap is True

    @patch("modelpool.pool.router.requests.get")
    def test_busy_generalist_with_capacity_still_serves(self, mock_get, router):
        """27B is loaded with capacity remaining -> still serves as generalist."""
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            if "192.168.35.185" in url:
                mock_resp.json = lambda: {
                    "state": "ready", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 0,  # under capacity
                }
            elif "192.168.35.17" in url:
                mock_resp.json = lambda: {
                    "state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0,
                }
            return mock_resp

        mock_get.side_effect = side_effect
        res = router.resolve("compression")
        # 27B is generalist and has capacity -> should use it
        assert res.resource.name == "gpu-27b"
        assert res.needs_swap is False

    @patch("modelpool.pool.router.requests.get")
    def test_all_workers_busy_raises_error(self, mock_get, router):
        """All workers in busy state -> RoutingError."""
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            mock_resp.json = lambda: {
                "state": "loading", "loaded_resource": "gpu-35b",
                "loaded_models_count": 1,
            }
            return mock_resp

        mock_get.side_effect = side_effect
        # "agentic" is only tagged on gpu-27b (worker hwrouter)
        # All workers are loading/draining -> no match -> RoutingError
        with pytest.raises(RoutingError):
            router.resolve("agentic")

    @patch("modelpool.pool.router.requests.get")
    def test_loaded_generalist_wins_over_swap_and_idle(self, mock_get, router):
        """Loaded generalist with capacity serves the request, no swap needed."""
        def side_effect(url, **kwargs):
            mock_resp = MagicMock(status_code=200)
            if "192.168.35.185" in url:
                # hwrouter: 27B generalist loaded with capacity
                mock_resp.json = lambda: {
                    "state": "ready", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 1,
                }
            elif "192.168.35.17" in url:
                # pvellm: idle
                mock_resp.json = lambda: {
                    "state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0,
                }
            return mock_resp

        mock_get.side_effect = side_effect
        res = router.resolve("title")
        # 27B generalist is loaded -> serves title directly (no swap)
        # Even though gpu-35b has higher priority for title,
        # avoiding a swap is preferred
        assert res.resource.name == "gpu-27b"
        assert res.resource.generalist is True
        assert res.needs_swap is False
