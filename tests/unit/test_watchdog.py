"""Tests for worker/watchdog.py - health monitor and auto-recovery.

Tests cover:
- Health check success and failure counting
- Auto-recovery after threshold failures
- No action when worker not in READY state
- Watchdog start/stop lifecycle
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from modelpool.registry import Registry
from modelpool.worker.loader import LlamaServerManager, READY, ERROR, IDLE, LOADING
from modelpool.worker.watchdog import Watchdog


def make_registry():
    data = {
        "resources": {
            "default-model": {
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
                "default_resource": "default-model",
                "swap_timeout": 30,
            },
        },
    }
    return Registry(data)


class TestWatchdogCheck:
    """Tests for the _check() method."""

    @patch("modelpool.worker.watchdog.requests.get")
    def test_healthy_check_resets_failure_count(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker")

        wd._consecutive_failures = 2  # had some failures
        wd._check()
        assert wd._consecutive_failures == 0

    @patch("modelpool.worker.watchdog.requests.get")
    def test_failed_check_increments_count(self, mock_get):
        mock_get.side_effect = Exception("connection refused")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=5)

        wd._check()
        assert wd._consecutive_failures == 1

        wd._check()
        assert wd._consecutive_failures == 2

    @patch("modelpool.worker.watchdog.requests.get")
    def test_non_200_increments_failure(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=5)

        wd._check()
        assert wd._consecutive_failures == 1

    def test_check_skips_when_not_ready(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = IDLE
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker")

        # Should be a no-op
        wd._check()
        assert wd._consecutive_failures == 0

    def test_check_skips_when_loading(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker")

        wd._check()
        assert wd._consecutive_failures == 0

    @patch("modelpool.worker.watchdog.requests.get")
    def test_connection_error_counts_as_failure(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("refused")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=3)

        wd._check()
        assert wd._consecutive_failures == 1


class TestWatchdogRecovery:
    """Tests for auto-recovery after threshold failures."""

    @patch.object(LlamaServerManager, "_wait_healthy", return_value=True)
    @patch("modelpool.worker.loader.subprocess.Popen")
    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.watchdog.requests.get")
    def test_triggers_recovery_after_threshold(self, mock_wd_get, mock_loader_get, mock_popen, mock_wait):
        # Health checks fail
        mock_wd_get.side_effect = Exception("down")
        # After recovery, Popen + _wait_healthy succeed
        mock_popen.return_value = MagicMock(pid=123)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = MagicMock(pid=999)
        mgr.loaded_resource = "broken-model"

        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=3)

        # Trigger 3 failures
        wd._check()  # 1
        wd._check()  # 2
        assert wd._consecutive_failures == 2

        wd._check()  # 3 -> triggers recovery
        assert mgr.state == READY
        assert mgr.loaded_resource == "default-model"
        assert wd._consecutive_failures == 0

    @patch.object(LlamaServerManager, "_wait_healthy", return_value=False)
    @patch("modelpool.worker.loader.subprocess.Popen")
    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.watchdog.requests.get")
    def test_recovery_failure_leaves_in_error(self, mock_wd_get, mock_loader_get, mock_popen, mock_wait):
        mock_wd_get.side_effect = Exception("down")
        mock_popen.return_value = MagicMock(pid=123)
        # _wait_healthy returns False -> start fails

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = MagicMock(pid=999)

        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=2)

        wd._check()
        wd._check()  # triggers recovery, which fails
        assert mgr.state == ERROR

    @patch("modelpool.worker.watchdog.requests.get")
    def test_failure_count_resets_on_recovery(self, mock_get):
        """After a successful check, failure count goes back to 0."""
        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("down")
            return MagicMock(status_code=200)

        mock_get.side_effect = side_effect

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY

        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", failure_threshold=5)

        wd._check()  # fail 1
        assert wd._consecutive_failures == 1
        wd._check()  # fail 2
        assert wd._consecutive_failures == 2
        wd._check()  # success -> reset
        assert wd._consecutive_failures == 0


class TestWatchdogLifecycle:
    """Tests for start/stop of the watchdog."""

    def test_start_creates_task(self):
        import asyncio

        async def run():
            mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
            reg = make_registry()
            wd = Watchdog(mgr, reg, "test-worker", check_interval=9999)

            wd.start()
            assert wd._running is True
            assert wd._task is not None
            wd.stop()
            assert wd._running is False

        asyncio.run(run())

    def test_stop_cancels_task(self):
        import asyncio

        async def run():
            mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
            reg = make_registry()
            wd = Watchdog(mgr, reg, "test-worker", check_interval=9999)

            wd.start()
            task = wd._task
            assert task is not None
            wd.stop()
            # After stop, task should be cancelled or done
            await asyncio.sleep(0.05)  # let cancellation propagate
            assert task.cancelled() or task.done()

        asyncio.run(run())

    def test_double_start_is_noop(self):
        import asyncio

        async def run():
            mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
            reg = make_registry()
            wd = Watchdog(mgr, reg, "test-worker", check_interval=9999)

            wd.start()
            first_task = wd._task
            wd.start()  # should be no-op
            assert wd._task is first_task
            wd.stop()

        asyncio.run(run())

    def test_default_params(self):
        mgr = LlamaServerManager(8080)
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker")
        assert wd.check_interval == 15
        assert wd.failure_threshold == 3

    def test_custom_params(self):
        mgr = LlamaServerManager(8080)
        reg = make_registry()
        wd = Watchdog(mgr, reg, "test-worker", check_interval=5, failure_threshold=10)
        assert wd.check_interval == 5
        assert wd.failure_threshold == 10
