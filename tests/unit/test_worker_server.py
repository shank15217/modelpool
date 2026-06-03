"""Tests for worker/server.py - worker HTTP endpoints and middleware.

Tests cover:
- Worker status endpoint
- Worker load/unload/revert/ready endpoints
- Pool auth middleware (secret, path normalization)
- Error responses (404, 409, 422, 503)
- Idle shutdown loop behavior
- Configuration
"""

import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from modelpool.registry import Registry
from modelpool.worker.loader import LlamaServerManager, StateError, LoadError
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
            "other-worker-model": {
                "type": "managed",
                "size_gb": 5,
                "ctx": 4096,
                "workers": ["other-worker"],
                "tags": {"triage": 1},
                "command": {
                    "binary": "/usr/bin/test-server",
                    "flags": [["-m", "/models/other.gguf"]],
                },
            },
        },
        "workers": {
            "test-worker": {
                "host": "10.0.0.1",
                "default_resource": "test-model",
                "max_model_gb": 50,
                "drain_timeout": 5,
                "swap_timeout": 30,
                "pool_secret": "secret123",
            },
            "other-worker": {
                "host": "10.0.0.2",
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
    wd = Watchdog(mgr, reg, "test-worker")

    worker_server.configure(
        manager=mgr,
        registry=reg,
        watchdog=wd,
        worker_name="test-worker",
        idle_shutdown=0,
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

    def test_status_shows_idle_shutdown(self):
        mgr, reg, wd = setup_worker()
        worker_server.configure(mgr, reg, wd, "test-worker", idle_shutdown=900, pool_secret="secret123")
        client = TestClient(worker_server.app)
        resp = client.get("/worker/status")
        data = resp.json()
        assert data["idle_shutdown"] == 900


class TestWorkerLoad:
    """Tests for POST /worker/load."""

    def test_load_requires_pool_secret(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post("/worker/load", json={"resource": "test-model"})
        assert resp.status_code == 403

    def test_load_with_wrong_secret_returns_403(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "wrong"},
        )
        assert resp.status_code == 403

    def test_load_unknown_resource_returns_404(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "nonexistent"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 404

    def test_load_resource_not_for_worker_returns_422(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "other-worker-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 422
        assert "cannot run on worker" in resp.json()["detail"]

    def test_load_too_large_resource_returns_422(self):
        """Resource too large for worker's max_model_gb should be rejected."""
        # Reconfigure with small max_model_gb
        mgr, reg, wd = setup_worker()
        worker_server.configure(mgr, reg, wd, "test-worker", pool_secret="secret123")
        # Override worker's max_model_gb
        reg.get_worker("test-worker").max_model_gb = 1
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 422
        assert "too large" in resp.json()["detail"].lower()

    def test_load_already_loaded_returns_already_loaded(self):
        setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "already_loaded"

    def test_load_while_busy_returns_409(self):
        mgr, reg, wd = setup_worker(state="loading")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 409
        assert "busy" in resp.json()["detail"].lower()

    @patch.object(LlamaServerManager, "load_resource")
    def test_load_success(self, mock_load):
        mgr, reg, wd = setup_worker(state="idle")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "loaded"
        mock_load.assert_called_once()

    @patch.object(LlamaServerManager, "load_resource", side_effect=LoadError("health failed"))
    def test_load_failure_returns_503(self, mock_load):
        mgr, reg, wd = setup_worker(state="idle")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/load",
            json={"resource": "test-model"},
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 503


class TestWorkerUnload:
    """Tests for POST /worker/unload."""

    def test_unload_requires_pool_secret(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post("/worker/unload")
        assert resp.status_code == 403

    @patch.object(LlamaServerManager, "unload")
    def test_unload_success(self, mock_unload):
        mgr, reg, wd = setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/unload",
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "unloaded"

    def test_unload_while_loading_returns_409(self):
        mgr, reg, wd = setup_worker(state="loading")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/unload",
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 409


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


class TestWorkerRevert:
    """Tests for POST /worker/revert."""

    def test_revert_requires_pool_secret(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post("/worker/revert")
        assert resp.status_code == 403

    def test_revert_while_busy_returns_409(self):
        mgr, reg, wd = setup_worker(state="loading")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/revert",
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 409

    @patch.object(LlamaServerManager, "revert")
    def test_revert_success(self, mock_revert):
        mgr, reg, wd = setup_worker(state="ready", loaded="test-model")
        client = TestClient(worker_server.app)
        resp = client.post(
            "/worker/revert",
            headers={"X-Pool-Secret": "secret123"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "reverted"
        mock_revert.assert_called_once()


class TestMiddleware:
    """Tests for pool_auth_middleware."""

    def test_trailing_slash_still_protected(self):
        """Path normalization: /worker/load/ should also require auth."""
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.post("/worker/load/", json={"resource": "test-model"})
        assert resp.status_code == 403

    def test_status_endpoint_no_auth_needed(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.get("/worker/status")
        assert resp.status_code == 200

    def test_ready_endpoint_no_auth_needed(self):
        setup_worker()
        client = TestClient(worker_server.app)
        resp = client.get("/worker/ready")
        assert resp.status_code == 503  # not ready, but NOT 403

    def test_no_secret_configured_allows_all(self):
        """If pool_secret is None, all endpoints are open."""
        mgr, reg, wd = setup_worker()
        worker_server.configure(mgr, reg, wd, "test-worker", pool_secret=None)
        client = TestClient(worker_server.app)
        resp = client.post("/worker/load", json={"resource": "test-model"})
        # Should NOT be 403 (no secret configured)
        assert resp.status_code != 403


class TestWorkerConfiguration:
    """Tests for worker configuration."""

    def test_configure_sets_globals(self):
        mgr = LlamaServerManager(8080)
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker")

        worker_server.configure(mgr, reg, wd, "test-worker", idle_shutdown=300, pool_secret="abc")

        assert worker_server._manager is mgr
        assert worker_server._registry is reg
        assert worker_server._watchdog is wd
        assert worker_server._worker_name == "test-worker"
        assert worker_server._idle_shutdown == 300
        assert worker_server._pool_secret == "abc"
