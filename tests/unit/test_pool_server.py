"""Tests for pool/server.py - pool management endpoints.

Tests cover:
- Pool status aggregation
- Routing table endpoint
- Manual swap and revert
- Idle timer loop (with pool_secret header)
- Configuration validation
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modelpool.registry import Registry
from modelpool.pool.router import Router
from modelpool.pool.proxy import PoolProxy
from modelpool.pool import server as pool_server


def make_test_registry():
    data = {
        "resources": {
            "gpu-27b": {
                "type": "managed",
                "size_gb": 16,
                "ctx": 131072,
                "workers": ["hwrouter"],
                "tags": {"chat": 1},
                "generalist": True,
                "command": {"binary": "/bin/test", "flags": [["-m", "27b.gguf"]]},
            },
            "gpu-35b": {
                "type": "managed",
                "size_gb": 21,
                "ctx": 262144,
                "workers": ["hwrouter"],
                "tags": {"compression": 1},
                "command": {"binary": "/bin/test", "flags": [["-m", "35b.gguf"]]},
            },
        },
        "workers": {
            "hwrouter": {
                "host": "192.168.35.185",
                "pool_secret": "mp-secret",
                "default_resource": "gpu-27b",
                "max_concurrent_models": 1,
            },
        },
    }
    return Registry(data)


def setup_pool():
    """Create a configured pool server for testing."""
    reg = make_test_registry()
    router = Router(reg)
    proxy = PoolProxy(reg, router)
    pool_server.configure(reg, router, proxy)
    return reg, router, proxy


class TestPoolRouting:
    """Tests for /pool/routing endpoint."""

    def test_routing_returns_all_tags(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.get("/pool/routing")
        assert resp.status_code == 200

        data = resp.json()
        assert "routing" in data
        routing = data["routing"]
        assert "chat" in routing
        assert "compression" in routing

    def test_routing_shows_priorities(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.get("/pool/routing")
        data = resp.json()

        chat_entries = data["routing"]["chat"]
        assert len(chat_entries) >= 1
        assert chat_entries[0]["resource"] == "gpu-27b"
        assert chat_entries[0]["priority"] == 1


class TestPoolStatus:
    """Tests for /pool/status endpoint."""

    @patch("modelpool.pool.router.requests.get")
    def test_status_returns_workers(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"state": "idle", "loaded_resource": None},
        )

        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.get("/pool/status")
        assert resp.status_code == 200

        data = resp.json()
        assert "workers" in data
        assert "hwrouter" in data["workers"]


class TestPoolSwap:
    """Tests for /pool/swap endpoint."""

    def test_swap_missing_params_returns_400(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/swap", json={})
        assert resp.status_code == 400

    def test_swap_unknown_worker_returns_404(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/swap", json={"worker": "nonexistent", "resource": "gpu-27b"})
        assert resp.status_code == 404

    def test_swap_resource_not_for_worker_returns_422(self):
        """Resource not compatible with worker."""
        # Add a resource for a different worker
        data = {
            "resources": {
                "gpu-27b": {
                    "type": "managed", "size_gb": 16, "ctx": 131072,
                    "workers": ["hwrouter"], "tags": {"chat": 1},
                    "command": {"binary": "/bin/test", "flags": [["-m", "27b.gguf"]]},
                },
                "cpu-model": {
                    "type": "managed", "size_gb": 10, "ctx": 4096,
                    "workers": ["other-worker"], "tags": {"triage": 1},
                    "command": {"binary": "/bin/test", "flags": [["-m", "cpu.gguf"]]},
                },
            },
            "workers": {
                "hwrouter": {"host": "10.0.0.1", "pool_secret": "secret"},
                "other-worker": {"host": "10.0.0.2", "pool_secret": "secret"},
            },
        }
        reg = Registry(data)
        router = Router(reg)
        proxy = PoolProxy(reg, router)
        pool_server.configure(reg, router, proxy)

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/swap", json={"worker": "hwrouter", "resource": "cpu-model"})
        assert resp.status_code == 422

    @patch("requests.post")
    def test_swap_includes_pool_secret(self, mock_post):
        """Manual swap must send pool_secret header to worker."""
        mock_post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"status": "loaded"},
        )

        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/swap", json={"worker": "hwrouter", "resource": "gpu-35b"})

        # Verify the POST to worker included secret
        post_call = mock_post.call_args
        headers = post_call[1].get("headers", {})
        assert headers.get("X-Pool-Secret") == "mp-secret"


class TestPoolRevert:
    """Tests for /pool/revert endpoint."""

    def test_revert_missing_worker_returns_400(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/revert", json={})
        assert resp.status_code == 400

    def test_revert_unknown_worker_returns_404(self):
        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/revert", json={"worker": "nonexistent"})
        assert resp.status_code == 404

    @patch("requests.post")
    def test_revert_includes_pool_secret(self, mock_post):
        """Manual revert must send pool_secret header."""
        mock_post.return_value = MagicMock(
            status_code=202,
            json=lambda: {"status": "reverted"},
        )

        reg, router, proxy = setup_pool()

        from starlette.testclient import TestClient as TC
        client = TC(pool_server.app)
        resp = client.post("/pool/revert", json={"worker": "hwrouter"})

        post_call = mock_post.call_args
        headers = post_call[1].get("headers", {})
        assert headers.get("X-Pool-Secret") == "mp-secret"


class TestIdleTimerLoop:
    """Tests for the idle timer background loop."""

    def test_idle_timer_loop_includes_pool_secret(self):
        """Verify the idle revert code includes X-Pool-Secret header."""
        import inspect
        from modelpool.pool import server
        source = inspect.getsource(server._idle_timer_loop)
        # The idle timer should include X-Pool-Secret in its revert POST
        assert "X-Pool-Secret" in source, "Idle timer revert must include X-Pool-Secret"
        assert "pool_secret" in source, "Idle timer must check pool_secret"
