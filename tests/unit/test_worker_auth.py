"""Tests for worker pool_secret authentication middleware.

Simplified for Architecture A: status and ready endpoints only (both open).
No load/unload/revert endpoints to protect.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from modelpool.worker.server import app, configure, SECRET_HEADER
from modelpool.registry import Registry
from modelpool.worker.loader import LlamaServerManager
from modelpool.worker.watchdog import Watchdog


@pytest.fixture
def mock_deps():
    """Configure worker app with mock dependencies."""
    registry_data = {
        "resources": {
            "test-model": {
                "type": "managed",
                "size_gb": 5,
                "workers": ["test-worker"],
                "command": {"binary": "/bin/test", "flags": [["-m", "test.gguf"]]},
            },
        },
        "workers": {
            "test-worker": {
                "host": "localhost",
                "type": "managed",
            },
        },
    }
    registry = Registry(registry_data)
    manager = MagicMock(spec=LlamaServerManager)
    manager.state = "idle"
    manager.loaded_resource = None
    manager.is_ready.return_value = False
    manager.get_status.return_value = {"state": "idle", "loaded_resource": None}

    watchdog = MagicMock(spec=Watchdog)
    return registry, manager, watchdog


@pytest.fixture
def client_no_secret(mock_deps):
    """Worker without pool_secret -- open mode."""
    _, manager, watchdog = mock_deps
    configure(manager, watchdog, pool_secret=None)
    return TestClient(app)


@pytest.fixture
def client_with_secret(mock_deps):
    """Worker with pool_secret -- paired mode."""
    _, manager, watchdog = mock_deps
    configure(manager, watchdog, pool_secret="my-secret")
    return TestClient(app)


class TestWorkerAuth:
    def test_status_open_without_secret(self, client_no_secret):
        """Status endpoint is always open."""
        resp = client_no_secret.get("/worker/status")
        assert resp.status_code == 200

    def test_status_open_with_secret(self, client_with_secret):
        """Status endpoint is always open, even with secret set."""
        resp = client_with_secret.get("/worker/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["paired"] is True

    def test_status_shows_not_paired(self, client_no_secret):
        """Status shows paired=False when no secret is set."""
        resp = client_no_secret.get("/worker/status")
        assert resp.json()["paired"] is False

    def test_status_shows_paired(self, client_with_secret):
        """Status shows paired=True when secret is set."""
        resp = client_with_secret.get("/worker/status")
        assert resp.json()["paired"] is True

    def test_ready_open_without_secret(self, client_with_secret):
        """Ready endpoint stays open for health checks."""
        resp = client_with_secret.get("/worker/ready")
        # Should not be 403 (503 is expected since no model loaded)
        assert resp.status_code != 403

    def test_no_load_endpoint(self, client_with_secret):
        """Architecture A: no /worker/load endpoint."""
        resp = client_with_secret.post("/worker/load", json={"resource": "test-model"})
        assert resp.status_code == 404

    def test_no_unload_endpoint(self, client_with_secret):
        """Architecture A: no /worker/unload endpoint."""
        resp = client_with_secret.post("/worker/unload")
        assert resp.status_code == 404

    def test_no_revert_endpoint(self, client_with_secret):
        """Architecture A: no /worker/revert endpoint."""
        resp = client_with_secret.post("/worker/revert")
        assert resp.status_code == 404
