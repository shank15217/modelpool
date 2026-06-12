"""Tests for worker/server.py - worker HTTP endpoints and middleware.

Simplified for Architecture A: status and ready endpoints only.
No dynamic load/unload/revert endpoints.
Tests cover:
- Worker status endpoint
- Worker ready endpoint
- Pool auth middleware (kept for future admin endpoints)
- Configuration
"""

import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from modelpool.registry import Registry
from modelpool.worker.loader import LlamaServerManager
from modelpool.worker.watchdog import Watchdog
from modelpool.worker import server as worker_server


def make_registry():
    data = {
        "resources": {
            "test-model": {
                "type": "managed",
                "size_gb": 10,
                "ctx": 4096,
                "workers": ["test-worker"],
                "tags": {"chat": 1},
                "command": {
                    "binary": "/usr/bin/test-server",
                    "flags": [["-m", "/models/test.gguf"]],
                },
            },
        },
        "workers": {
            "test-worker": {
                "host": "10.0.0.1",
                "max_model_gb": 50,
                "pool_secret": "secret123",
            },
        },
    }
    return Registry(data)


def setup_worker(state="idle", loaded=None):
    """Create a configured worker server for testing."""
    reg = make_registry()
    mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
    mgr.state = state
    mgr.loaded_resource = loaded
    wd = Watchdog(mgr)

    worker_server.configure(
        manager=mgr,
        watchdog=wd,
        pool_secret="secret123",
    )
    return mgr, reg, wd


class TestWorkerStatus:
    """Tests for GET /worker/status."""

    def test_status_returns_state(self):
        setup_worker(state="idle")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"
        assert data["paired"] is True

    def test_status_shows_loaded_resource(self):
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/status")
        data = resp.json()
        assert data["state"] == "ready"
        assert data["loaded_resource"] == "test-model"


class TestWorkerReady:
    """Tests for GET /worker/ready."""

    def test_ready_when_ready(self):
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_when_idle_returns_503(self):
        setup_worker(state="idle")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/ready")
        assert resp.status_code == 503

    def test_ready_when_loading_returns_503(self):
        setup_worker(state="loading")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/ready")
        assert resp.status_code == 503

    def test_ready_no_auth_required(self):
        """Ready endpoint is open for monitoring (no secret needed)."""
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/ready")
        assert resp.status_code == 200  # no 403


class TestWorkerConfiguration:
    """Tests for worker configuration."""

    def test_configure_sets_globals(self):
        mgr = LlamaServerManager(8080)
        wd = Watchdog(mgr)

        worker_server.configure(mgr, wd, pool_secret="abc")

        assert worker_server._manager is mgr
        assert worker_server._watchdog is wd
        assert worker_server._pool_secret == "abc"


class TestRemovedEndpoints:
    """Architecture A: dynamic pool endpoints removed."""

    def test_no_load_endpoint(self):
        setup_worker(state="idle")
        client = TestClient(worker_server.app)
        resp = client.post("/worker/load", json={"resource": "test-model"})
        assert resp.status_code == 404  # endpoint doesn't exist

    def test_no_unload_endpoint(self):
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.post("/worker/unload")
        assert resp.status_code == 404  # endpoint doesn't exist

    def test_no_revert_endpoint(self):
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.post("/worker/revert")
        assert resp.status_code == 404  # endpoint doesn't exist

    def test_no_idle_shutdown(self):
        """Architecture A: no idle shutdown timer."""
        assert not hasattr(worker_server, '_idle_shutdown')
