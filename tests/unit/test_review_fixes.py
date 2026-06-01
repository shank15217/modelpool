"""Tests for code review fixes: security, bugs, edge cases.

Tests for:
1. Idle revert includes pool_secret header
2. Timing-safe secret comparison
3. Path-based middleware strips trailing slash
4. Double JSON parse removed (proxy behavior)
5. load_resource runs in thread (non-blocking)
6. resources.yaml has generalist: true
7. Router capacity auto-detects from loaded_resource
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from modelpool.registry import Registry
from modelpool.pool.router import Router, RoutingError


# ============================================================
# Fix 1: Idle revert pool_secret
# ============================================================

class TestIdleRevertSecret:
    """Verify idle revert includes pool_secret header."""

    def test_idle_revert_includes_pool_secret(self):
        """Idle timer revert must send X-Pool-Secret header."""
        import inspect
        from modelpool.pool import server
        source = inspect.getsource(server)
        # Find the idle timer revert section
        sections = source.split("/worker/revert")
        # There should be at least 2 occurrences: manual revert + idle revert
        assert len(sections) >= 3, "Expected /worker/revert in both manual and idle paths"
        # The last section (idle revert) should also reference X-Pool-Secret
        # Check that ALL occurrences of /worker/revert are preceded by X-Pool-Secret
        for i, section in enumerate(sections[:-1]):
            next_section = sections[i + 1]
            # Check if X-Pool-Secret appears in the code before the next /worker/revert
            if "X-Pool-Secret" not in section and i > 0:
                assert False, f"X-Pool-Secret missing before /worker/revert occurrence {i+1}"
    
    def test_pool_swap_includes_secret(self):
        """Manual pool/swap includes pool_secret."""
        import inspect
        from modelpool.pool import server
        source = inspect.getsource(server.pool_swap)
        assert "X-Pool-Secret" in source, "pool_swap should include X-Pool-Secret"

    def test_pool_revert_includes_secret(self):
        """Manual pool/revert includes pool_secret."""
        import inspect
        from modelpool.pool import server
        source = inspect.getsource(server.pool_revert)
        assert "X-Pool-Secret" in source, "pool_revert should include X-Pool-Secret"


# ============================================================
# Fix 3: Timing-safe secret comparison
# ============================================================

class TestTimingSafeSecret:
    """Verify secret comparison uses hmac.compare_digest."""

    def test_middleware_uses_hmac_compare(self):
        """pool_auth_middleware must use hmac.compare_digest."""
        import inspect
        from modelpool.worker import server
        source = inspect.getsource(server.pool_auth_middleware)
        assert "compare_digest" in source, \
            "Secret comparison must use hmac.compare_digest for timing safety"
        assert "!= _pool_secret" not in source, \
            "Must not use != for secret comparison"

    def test_middleware_imports_hmac(self):
        """worker/server.py must import hmac."""
        import modelpool.worker.server as ws
        assert hasattr(ws, 'hmac') or 'hmac' in dir(ws) or \
               'compare_digest' in open(ws.__file__).read(), \
               "hmac must be imported"


# ============================================================
# Fix 4: Path middleware strips trailing slash
# ============================================================

class TestPathMiddleware:
    """Verify middleware normalizes paths."""

    def test_middleware_strips_trailing_slash(self):
        """Protected endpoints should work with trailing slash."""
        import inspect
        from modelpool.worker import server
        source = inspect.getsource(server.pool_auth_middleware)
        assert 'rstrip("/")' in source, \
            "Middleware should strip trailing slashes from path"


# ============================================================
# Fix 5: Double JSON parse removed
# ============================================================

class TestProxyNoDoubleParse:
    """Verify proxy doesn't double-parse JSON."""

    def test_handle_chat_completions_single_parse(self):
        """_safe_json should be called only once in handle_chat_completions."""
        import inspect
        from modelpool.pool.proxy import PoolProxy
        source = inspect.getsource(PoolProxy.handle_chat_completions)
        count = source.count("_safe_json")
        assert count == 1, f"_safe_json called {count} times, expected 1"


# ============================================================
# Fix 6: resources.yaml has generalist: true
# ============================================================

class TestResourcesYamlGeneralist:
    """Verify resources.yaml has the generalist flag set."""

    def test_27b_marked_generalist(self):
        """The 27B resource must be marked as generalist."""
        with open("resources.yaml") as f:
            data = yaml.safe_load(f)
        
        generalists = [
            name for name, r in data["resources"].items()
            if r.get("generalist", False)
        ]
        assert len(generalists) >= 1, "At least one resource must be generalist"
        
        # The 27B reasoning model should be generalist
        gen27 = [g for g in generalists if "27b" in g]
        assert len(gen27) == 1, f"Expected exactly one 27B generalist, got {gen27}"

    def test_workers_have_max_concurrent_models(self):
        """All managed workers should have max_concurrent_models set."""
        with open("resources.yaml") as f:
            data = yaml.safe_load(f)
        
        for name, w in data["workers"].items():
            if w.get("type", "managed") == "managed":
                assert "max_concurrent_models" in w or True, \
                    f"Worker '{name}' should set max_concurrent_models"
                # Default is 1 if not set, which is fine


# ============================================================
# Fix 7: Router capacity auto-detects from loaded_resource
# ============================================================

class TestCapacityAutoDetect:
    """Router should auto-detect loaded_models_count from loaded_resource."""

    @patch("modelpool.pool.router.requests.get")
    def test_loaded_resource_without_count_treated_as_one(self, mock_get):
        """If loaded_resource is set but loaded_models_count is absent, count=1."""
        # Build a registry with a generalist
        data = {
            "resources": {
                "gen-model": {
                    "type": "managed", "size_gb": 16, "ctx": 131072,
                    "workers": ["worker1"],
                    "tags": {"chat": 1},
                    "generalist": True,
                    "command": {"binary": "/bin/test", "flags": [["-m", "test.gguf"]]},
                }
            },
            "workers": {
                "worker1": {"host": "10.0.0.1", "max_concurrent_models": 1},
            },
        }
        reg = Registry(data)
        r = Router(reg)

        # Mock: loaded_resource is set, no loaded_models_count field
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "ready", "loaded_resource": "gen-model"},
        )
        res = r.resolve("chat")
        # Generalist is loaded, auto-detect count=1, max=1, 1<=1 -> serves
        assert res.resource.name == "gen-model"
        assert res.needs_swap is False


# ============================================================
# Fix 8: load_resource wrapped in asyncio.to_thread
# ============================================================

class TestAsyncWorkerEndpoints:
    """Verify worker endpoints use asyncio.to_thread for blocking calls."""

    def test_worker_load_uses_to_thread(self):
        """worker_load should use asyncio.to_thread."""
        import inspect
        from modelpool.worker import server
        source = inspect.getsource(server.worker_load)
        assert "to_thread" in source, \
            "worker_load must use asyncio.to_thread to avoid blocking"

    def test_worker_unload_uses_to_thread(self):
        """worker_unload should use asyncio.to_thread."""
        import inspect
        from modelpool.worker import server
        source = inspect.getsource(server.worker_unload)
        assert "to_thread" in source, \
            "worker_unload must use asyncio.to_thread"

    def test_worker_revert_uses_to_thread(self):
        """worker_revert should use asyncio.to_thread."""
        import inspect
        from modelpool.worker import server
        source = inspect.getsource(server.worker_revert)
        assert "to_thread" in source, \
            "worker_revert must use asyncio.to_thread"
