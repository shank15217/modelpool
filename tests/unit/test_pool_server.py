"""Tests for pool/server.py - pool management endpoints.

Simplified for Architecture A: static pool, no dynamic swapping.
Tests cover:
- Pool routing table endpoint
- Pool status endpoint
- Configuration validation
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
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
        client = TestClient(pool_server.app)
        resp = client.get("/pool/routing")
        assert resp.status_code == 200

        data = resp.json()
        assert "routing" in data
        routing = data["routing"]
        assert "chat" in routing
        assert "compression" in routing

    def test_routing_shows_priorities(self):
        reg, router, proxy = setup_pool()
        client = TestClient(pool_server.app)
        resp = client.get("/pool/routing")
        data = resp.json()

        chat_entries = data["routing"]["chat"]
        assert len(chat_entries) >= 1
        assert chat_entries[0]["resource"] == "gpu-27b"
        assert chat_entries[0]["priority"] == 1


class TestPoolStatus:
    """Tests for /pool/status endpoint."""

    def test_status_returns_workers(self):
        reg, router, proxy = setup_pool()
        client = TestClient(pool_server.app)
        resp = client.get("/pool/status")
        assert resp.status_code == 200

        data = resp.json()
        assert "workers" in data
        assert "hwrouter" in data["workers"]

    def test_status_returns_resources(self):
        reg, router, proxy = setup_pool()
        client = TestClient(pool_server.app)
        resp = client.get("/pool/status")
        data = resp.json()
        assert "resources" in data
        assert "gpu-27b" in data["resources"]


class TestRemovedEndpoints:
    """Architecture A: dynamic pool endpoints removed."""

    def test_no_swap_endpoint(self):
        reg, router, proxy = setup_pool()
        client = TestClient(pool_server.app)
        resp = client.post("/pool/swap", json={"worker": "hwrouter", "resource": "gpu-35b"})
        assert resp.status_code == 404  # endpoint doesn't exist

    def test_no_revert_endpoint(self):
        reg, router, proxy = setup_pool()
        client = TestClient(pool_server.app)
        resp = client.post("/pool/revert", json={"worker": "hwrouter"})
        assert resp.status_code == 404  # endpoint doesn't exist

    def test_no_idle_timer_loop(self):
        """Architecture A: no idle timer background loop."""
        import inspect
        from modelpool.pool import server
        source = inspect.getsource(server)
        assert "_idle_timer_loop" not in source
