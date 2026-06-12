"""Tests for worker/watchdog.py - health monitor.

Simplified for Architecture A: basic health monitoring only.
No auto-recovery (since there's no dynamic swapping).
Tests cover:
- Health check success and failure counting
- Failure threshold logging (but no auto-recovery)
- No action when worker not in READY state
- Watchdog start/stop lifecycle
"""

import pytest
from unittest.mock import MagicMock, patch

from modelpool.worker.loader import LlamaServerManager, READY, ERROR, IDLE, LOADING
from modelpool.worker.watchdog import Watchdog


def make_registry():
    """Create a minimal registry (not needed by simplified watchdog but kept for compat)."""
    from modelpool.registry import Registry
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
        wd = Watchdog(mgr, check_interval=15, failure_threshold=3)

        wd._consecutive_failures = 2  # had some failures
        wd._check()
        assert wd._consecutive_failures == 0

    @patch("modelpool.worker.watchdog.requests.get")
    def test_failed_check_increments_count(self, mock_get):
        mock_get.side_effect = Exception("connection refused")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        wd = Watchdog(mgr, failure_threshold=5)

        wd._check()
        assert wd._consecutive_failures == 1

        wd._check()
        assert wd._consecutive_failures == 2

    @patch("modelpool.worker.watchdog.requests.get")
    def test_non_200_increments_failure(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        wd = Watchdog(mgr, failure_threshold=5)

        wd._check()
        assert wd._consecutive_failures == 1

    def test_check_skips_when_not_ready(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = IDLE
        wd = Watchdog(mgr)

        # Should be a no-op
        wd._check()
        assert wd._consecutive_failures == 0

    def test_check_skips_when_loading(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        wd = Watchdog(mgr)

        wd._check()
        assert wd._consecutive_failures == 0

    @patch("modelpool.worker.watchdog.requests.get")
    def test_connection_error_counts_as_failure(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError("refused")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        wd = Watchdog(mgr, failure_threshold=3)

        wd._check()
        assert wd._consecutive_failures == 1

    @patch("modelpool.worker.watchdog.requests.get")
    def test_threshold_reached_resets_counter(self, mock_get):
        """When threshold is reached, counter resets (to avoid log spam)."""
        mock_get.side_effect = Exception("down")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        wd = Watchdog(mgr, failure_threshold=3)

        wd._check()  # 1
        wd._check()  # 2
        wd._check()  # 3 -> threshold reached, counter resets
        assert wd._consecutive_failures == 0  # reset after threshold

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
        wd = Watchdog(mgr, failure_threshold=5)

        wd._check()  # fail 1
        assert wd._consecutive_failures == 1
        wd._check()  # fail 2
        assert wd._consecutive_failures == 2
        wd._check()  # success -> reset
        assert wd._consecutive_failures == 0


class TestNoAutoRecovery:
    """Architecture A: watchdog has no auto-recovery logic."""

    def test_no_recover_method(self):
        """Watchdog should not have _recover method."""
        assert not hasattr(Watchdog, "_recover")

    def test_no_registry_dependency(self):
        """Simplified watchdog doesn't need registry."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        wd = Watchdog(mgr)
        assert wd.manager is mgr
        assert not hasattr(wd, "registry")
        assert not hasattr(wd, "worker_name")


class TestWatchdogLifecycle:
    """Tests for start/stop of the watchdog."""

    def test_start_creates_task(self):
        import asyncio

        async def run():
            mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
            wd = Watchdog(mgr, check_interval=9999)

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
            wd = Watchdog(mgr, check_interval=9999)

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
            wd = Watchdog(mgr, check_interval=9999)

            wd.start()
            first_task = wd._task
            wd.start()  # should be no-op
            assert wd._task is first_task
            wd.stop()

        asyncio.run(run())

    def test_default_params(self):
        mgr = LlamaServerManager(8080)
        wd = Watchdog(mgr)
        assert wd.check_interval == 15
        assert wd.failure_threshold == 3

    def test_custom_params(self):
        mgr = LlamaServerManager(8080)
        wd = Watchdog(mgr, check_interval=5, failure_threshold=10)
        assert wd.check_interval == 5
        assert wd.failure_threshold == 10
