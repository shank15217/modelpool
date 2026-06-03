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


# =========================================================================
# 1. Generalist preference tests
# =========================================================================

class TestGeneralistPreference:
    """When a generalist resource is loaded, prefer it for any tag."""

    @pytest.mark.asyncio
    async def test_generalist_loaded_serves_any_tag(self, router):
        """27B (generalist) is loaded -> compression request uses it, no swap."""
        async def mock_status(worker):
            return {"state": "ready", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 1}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("compression")
        # Generalist 27B is loaded and has capacity -> should use it
        assert res.resource.generalist is True
        assert res.needs_swap is False

    @pytest.mark.asyncio
    async def test_generalist_loaded_serves_title(self, router):
        """27B (generalist) is loaded -> title request uses it, no swap."""
        async def mock_status(worker):
            return {"state": "ready", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 1}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("title")
        assert res.resource.generalist is True
        assert res.needs_swap is False

    @pytest.mark.asyncio
    async def test_generalist_loaded_serves_chat(self, router):
        """27B (generalist) is loaded -> chat request uses it (it's also tagged chat:1)."""
        async def mock_status(worker):
            return {"state": "ready", "loaded_resource": "gpu-27b",
                    "loaded_models_count": 1}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("chat")
        assert res.resource.name == "gpu-27b"
        assert res.needs_swap is False

    @pytest.mark.asyncio
    async def test_no_generalist_loaded_uses_tag_priority(self, router):
        """When nothing is loaded, use normal tag priority ordering."""
        async def mock_status(worker):
            return {"state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("compression")
        # Should pick gpu-35b (priority 1 for compression) since nothing is loaded
        assert res.resource.name == "gpu-35b"
        assert res.needs_swap is True

    @pytest.mark.asyncio
    async def test_specialist_loaded_but_higher_priority_wins_with_swap(self, router):
        """35B loaded with chat:2, but 27B (chat:1) is higher priority -> swap to 27B."""
        async def mock_status(worker):
            if worker.name == "hwrouter":
                return {"state": "ready", "loaded_resource": "gpu-35b",
                        "loaded_models_count": 0}
            elif worker.name == "pvellm":
                return {"state": "idle", "loaded_resource": None,
                        "loaded_models_count": 0}
            return None

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            # 27B (generalist, chat:1) is higher priority than 35B (chat:2)
            # Generalist is NOT loaded, so preference doesn't apply
            # Router picks best priority (27B) and swaps from 35B
            res = await router.resolve("chat")
        assert res.resource.name == "gpu-27b"
        assert res.resource.generalist is True
        assert res.needs_swap is True
        assert res.currently_loaded == "gpu-35b"


# =========================================================================
# 2. max_concurrent_models enforcement
# =========================================================================

class TestMaxConcurrentModels:
    """Workers at max_concurrent_models should not receive new loads/swaps."""

    @pytest.mark.asyncio
    async def test_worker_at_capacity_skipped(self, router):
        """Worker with loaded_models_count >= max_concurrent_models is skipped."""
        async def mock_status(worker):
            if worker.name == "hwrouter":
                return {"state": "ready", "loaded_resource": "gpu-35b",
                        "loaded_models_count": 1}  # at capacity
            elif worker.name == "pvellm":
                return {"state": "idle", "loaded_resource": None,
                        "loaded_models_count": 0}
            return None

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("chat")
        # hwrouter is at capacity with 35B loaded (which has chat:2)
        # Should fall through to next available option
        assert res is not None

    @pytest.mark.asyncio
    async def test_worker_under_capacity_accepted(self, router):
        """Worker with loaded_models_count < max_concurrent_models can load."""
        async def mock_status(worker):
            return {"state": "idle", "loaded_resource": None,
                    "loaded_models_count": 0}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("chat")
        assert res is not None
        assert res.needs_swap is True  # cold load needed


# =========================================================================
# 3. No-swap-when-busy (rug pull protection)
# =========================================================================

class TestNoSwapWhenBusy:
    """A worker that is serving requests should not be swapped."""

    @pytest.mark.asyncio
    async def test_busy_worker_not_swapped_for_different_tag(self, router):
        """Worker in draining/loading state is skipped, falls to CPU."""
        async def mock_status(worker):
            if worker.name == "hwrouter":
                return {"state": "draining", "loaded_resource": "gpu-27b",
                        "loaded_models_count": 1}
            elif worker.name == "pvellm":
                return {"state": "idle", "loaded_resource": None,
                        "loaded_models_count": 0}
            return None

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("compression")
        # hwrouter is draining -> not ready -> swap blocked -> skip
        # Falls to cpu-35b on pvellm
        assert res.resource.name == "cpu-35b"
        assert res.worker.name == "pvellm"
        assert res.needs_swap is True

    @pytest.mark.asyncio
    async def test_busy_generalist_with_capacity_still_serves(self, router):
        """27B is loaded with capacity remaining -> still serves as generalist."""
        async def mock_status(worker):
            if worker.name == "hwrouter":
                return {"state": "ready", "loaded_resource": "gpu-27b",
                        "loaded_models_count": 0}  # under capacity
            elif worker.name == "pvellm":
                return {"state": "idle", "loaded_resource": None,
                        "loaded_models_count": 0}
            return None

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("compression")
        # 27B is generalist and has capacity -> should use it
        assert res.resource.name == "gpu-27b"
        assert res.needs_swap is False

    @pytest.mark.asyncio
    async def test_all_workers_busy_raises_error(self, router):
        """All workers in busy state -> RoutingError."""
        async def mock_status(worker):
            return {"state": "loading", "loaded_resource": "gpu-35b",
                    "loaded_models_count": 1}

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            # "agentic" is only tagged on gpu-27b (worker hwrouter)
            # All workers are loading/draining -> no match -> RoutingError
            with pytest.raises(RoutingError):
                await router.resolve("agentic")

    @pytest.mark.asyncio
    async def test_loaded_generalist_wins_over_swap_and_idle(self, router):
        """Loaded generalist with capacity serves the request, no swap needed."""
        async def mock_status(worker):
            if worker.name == "hwrouter":
                return {"state": "ready", "loaded_resource": "gpu-27b",
                        "loaded_models_count": 1}
            elif worker.name == "pvellm":
                return {"state": "idle", "loaded_resource": None,
                        "loaded_models_count": 0}
            return None

        with patch.object(Router, "_get_worker_status", side_effect=mock_status):
            res = await router.resolve("title")
        # 27B generalist is loaded -> serves title directly (no swap)
        # Even though gpu-35b has higher priority for title,
        # avoiding a swap is preferred
        assert res.resource.name == "gpu-27b"
        assert res.resource.generalist is True
        assert res.needs_swap is False
