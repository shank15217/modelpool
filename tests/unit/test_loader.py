"""Tests for worker/loader.py - subprocess manager state machine and lifecycle.

Tests cover:
- State machine transitions (valid and invalid)
- Command building from resource definitions
- start/stop/drain/load_resource/unload/revert lifecycle
- Health check polling
- Error handling and recovery
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from modelpool.registry import Registry, Resource, Worker
from modelpool.worker.loader import (
    LlamaServerManager,
    StateError,
    LoadError,
    IDLE,
    LOADING,
    READY,
    DRAINING,
    STOPPING,
    ERROR,
    VALID_TRANSITIONS,
)


# ============================================================
# Fixtures
# ============================================================

def make_resource(name="test-model", binary="/usr/bin/test-server", flags=None,
                  workers=None, tags=None, generalist=False):
    """Create a minimal Resource for testing."""
    return Resource(
        name=name,
        type="managed",
        binary=binary,
        flags=flags or [["-m", "/models/test.gguf"], ["-c", "4096"]],
        workers=workers or ["test-worker"],
        tags=tags or {"chat": 1},
        generalist=generalist,
    )


def make_worker(name="test-worker", host="10.0.0.1", default_resource="test-model",
                drain_timeout=10, swap_timeout=30, pool_secret="secret123"):
    """Create a minimal Worker for testing."""
    return Worker(
        name=name,
        host=host,
        worker_port=9100,
        inference_port=8080,
        type="managed",
        default_resource=default_resource,
        drain_timeout=drain_timeout,
        swap_timeout=swap_timeout,
        pool_secret=pool_secret,
    )


def make_registry():
    """Create a minimal registry with one resource and one worker."""
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
                    "flags": [["-m", "/models/test.gguf"], ["-c", "4096"]],
                },
            },
            "alt-model": {
                "type": "managed",
                "size_gb": 8,
                "ctx": 8192,
                "workers": ["test-worker"],
                "tags": {"compression": 1},
                "command": {
                    "binary": "/usr/bin/test-server",
                    "flags": [["-m", "/models/alt.gguf"], ["-c", "8192"]],
                },
            },
        },
        "workers": {
            "test-worker": {
                "host": "10.0.0.1",
                "default_resource": "test-model",
                "drain_timeout": 10,
                "swap_timeout": 30,
                "pool_secret": "secret123",
            },
        },
    }
    return Registry(data)


# ============================================================
# State Machine
# ============================================================

class TestStateMachine:
    """Tests for the state machine transition rules."""

    def test_all_valid_transitions_defined(self):
        """Every state has at least one valid transition."""
        for state in [IDLE, LOADING, READY, DRAINING, STOPPING, ERROR]:
            assert state in VALID_TRANSITIONS, f"State '{state}' has no transitions defined"

    def test_idle_can_transition_to_loading(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        assert mgr.state == IDLE
        mgr._transition(LOADING)
        assert mgr.state == LOADING

    def test_loading_can_transition_to_ready(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        mgr._transition(READY)
        assert mgr.state == READY

    def test_loading_can_transition_to_error(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        mgr._transition(ERROR)
        assert mgr.state == ERROR

    def test_ready_can_transition_to_draining(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr._transition(DRAINING)
        assert mgr.state == DRAINING

    def test_ready_can_transition_to_loading(self):
        """Direct reload from READY (bypasses drain -- used when forced)."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr._transition(LOADING)
        assert mgr.state == LOADING

    def test_ready_can_transition_to_error(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr._transition(ERROR)
        assert mgr.state == ERROR

    def test_draining_can_transition_to_stopping(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = DRAINING
        mgr._transition(STOPPING)
        assert mgr.state == STOPPING

    def test_stopping_can_transition_to_idle(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = STOPPING
        mgr._transition(IDLE)
        assert mgr.state == IDLE

    def test_stopping_can_transition_to_loading(self):
        """Fast path: stop and immediately load."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = STOPPING
        mgr._transition(LOADING)
        assert mgr.state == LOADING

    def test_error_can_transition_to_idle(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = ERROR
        mgr._transition(IDLE)
        assert mgr.state == IDLE

    def test_error_can_transition_to_loading(self):
        """Recovery: load from error state."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = ERROR
        mgr._transition(LOADING)
        assert mgr.state == LOADING

    def test_invalid_transition_raises_state_error(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        assert mgr.state == IDLE
        with pytest.raises(StateError, match="Invalid transition"):
            mgr._transition(READY)  # IDLE -> READY is not valid

    def test_idle_to_draining_invalid(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        with pytest.raises(StateError):
            mgr._transition(DRAINING)

    def test_ready_to_idle_invalid(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        with pytest.raises(StateError):
            mgr._transition(IDLE)


# ============================================================
# Command Building
# ============================================================

class TestCommandBuilding:
    """Tests for build_command method."""

    def test_basic_command(self):
        mgr = LlamaServerManager(8080)
        res = make_resource(binary="/usr/bin/llama-server",
                            flags=[["-m", "/models/test.gguf"], ["-c", "4096"]])
        cmd = mgr.build_command(res)
        assert cmd[0] == "/usr/bin/llama-server"
        assert "-m" in cmd
        assert "/models/test.gguf" in cmd
        assert "-c" in cmd
        assert "4096" in cmd

    def test_inference_port_template_replaced(self):
        mgr = LlamaServerManager(9090)
        res = make_resource(
            flags=[["--port", "{inference_port}"], ["-m", "/x.gguf"]]
        )
        cmd = mgr.build_command(res)
        assert "--port" in cmd
        idx = cmd.index("--port")
        assert cmd[idx + 1] == "9090"
        assert "{inference_port}" not in cmd

    def test_no_binary_raises_load_error(self):
        mgr = LlamaServerManager(8080)
        res = make_resource(binary=None)
        with pytest.raises(LoadError, match="no binary"):
            mgr.build_command(res)

    def test_boolean_flags(self):
        """Flags with single element (boolean) should work."""
        mgr = LlamaServerManager(8080)
        res = make_resource(flags=[["--jinja"], ["-m", "/x.gguf"]])
        cmd = mgr.build_command(res)
        assert "--jinja" in cmd

    def test_multi_value_flags(self):
        """Flags like --tensor-split 0.5,0.5."""
        mgr = LlamaServerManager(8080)
        res = make_resource(flags=[["--tensor-split", "0.5,0.5"], ["-m", "/x.gguf"]])
        cmd = mgr.build_command(res)
        assert "--tensor-split" in cmd
        assert "0.5,0.5" in cmd

    def test_empty_flags_produces_only_binary(self):
        mgr = LlamaServerManager(8080)
        res = Resource(name="test", type="managed", binary="/usr/bin/test-server", flags=[])
        cmd = mgr.build_command(res)
        assert cmd == ["/usr/bin/test-server"]


# ============================================================
# Start Lifecycle
# ============================================================

class TestStart:
    """Tests for the start() method."""

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_succeeds_on_health_check(self, mock_popen, mock_get):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        # Health check returns 200 on first try
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        res = make_resource()
        mgr.start(res, timeout=10)

        assert mgr.state == READY
        assert mgr.loaded_resource == "test-model"
        assert mgr.process == mock_proc
        assert mgr.started_at is not None

    @patch("modelpool.worker.loader.time.sleep")
    @patch("modelpool.worker.loader.time.time")
    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_fails_on_health_timeout(self, mock_popen, mock_get, mock_time, mock_sleep):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        # Health check never returns 200
        mock_get.side_effect = Exception("connection refused")

        # Simulate time advancing past timeout
        mock_time.side_effect = [0, 0, 2, 3]  # start, first check, second check (past deadline)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        res = make_resource()

        with pytest.raises(LoadError, match="Health check failed"):
            mgr.start(res, timeout=1)

        assert mgr.state == ERROR

    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_fails_on_popen_error(self, mock_popen):
        mock_popen.side_effect = OSError("Binary not found")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        res = make_resource()

        with pytest.raises(LoadError, match="Failed to start"):
            mgr.start(res, timeout=5)

        assert mgr.state == ERROR

    def test_start_from_ready_raises_state_error(self):
        """Cannot start when already in READY state directly."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        res = make_resource()
        with pytest.raises(StateError, match="Cannot start"):
            mgr.start(res)

    def test_start_from_loading_raises_state_error(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        res = make_resource()
        with pytest.raises(StateError):
            mgr.start(res)

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_from_error_state(self, mock_popen, mock_get):
        """Recovery: starting from ERROR should work (LOADING transition allowed)."""
        mock_proc = MagicMock(pid=999)
        mock_popen.return_value = mock_proc
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = ERROR
        res = make_resource()
        mgr.start(res, timeout=5)

        assert mgr.state == READY

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_uses_setsid(self, mock_popen, mock_get):
        """Process should be started with os.setsid for process group isolation."""
        mock_popen.return_value = MagicMock(pid=123)
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.start(make_resource(), timeout=5)

        _, kwargs = mock_popen.call_args
        assert kwargs["preexec_fn"] == os.setsid

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_start_captures_stderr_to_stdout(self, mock_popen, mock_get):
        mock_popen.return_value = MagicMock(pid=123)
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.start(make_resource(), timeout=5)

        _, kwargs = mock_popen.call_args
        assert kwargs["stderr"] == subprocess.STDOUT


# ============================================================
# Stop Lifecycle
# ============================================================

class TestStop:
    """Tests for the stop() method."""

    def test_stop_no_process_sets_idle(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = None
        mgr.loaded_resource = "something"
        mgr.stop()
        assert mgr.state == IDLE
        assert mgr.loaded_resource is None

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_stop_sends_sigterm_then_sigkill(self, mock_getpgid, mock_killpg):
        mock_getpgid.return_value = 100
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        # First wait raises TimeoutExpired (SIGTERM didn't work)
        # Second wait succeeds (SIGKILL worked)
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=1),
            None,  # after SIGKILL
        ]

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = mock_proc
        mgr.loaded_resource = "test"

        mgr.stop(timeout=1)

        # Should have sent SIGTERM first, then SIGKILL
        calls = mock_killpg.call_args_list
        assert len(calls) == 2
        assert calls[0] == call(100, signal.SIGTERM)
        assert calls[1] == call(100, signal.SIGKILL)

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_stop_clean_exit_no_sigkill(self, mock_getpgid, mock_killpg):
        mock_getpgid.return_value = 100
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0  # clean exit

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = mock_proc
        mgr.loaded_resource = "test"

        mgr.stop(timeout=5)

        # Only SIGTERM, no SIGKILL
        calls = mock_killpg.call_args_list
        assert len(calls) == 1
        assert calls[0] == call(100, signal.SIGTERM)

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_stop_process_already_dead(self, mock_getpgid, mock_killpg):
        mock_getpgid.side_effect = ProcessLookupError("No process")
        mock_proc = MagicMock()
        mock_proc.pid = 12345

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = mock_proc

        mgr.stop()

        # Cleanup should have happened
        assert mgr.process is None
        assert mgr.loaded_resource is None

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_stop_resets_state(self, mock_getpgid, mock_killpg):
        mock_getpgid.return_value = 100
        mock_proc = MagicMock(pid=123, wait=MagicMock(return_value=0))

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = STOPPING
        mgr.process = mock_proc
        mgr.loaded_resource = "test"
        mgr.started_at = 1000.0

        mgr.stop()

        assert mgr.process is None
        assert mgr.loaded_resource is None
        assert mgr.started_at is None


# ============================================================
# Drain Lifecycle
# ============================================================

class TestDrain:
    """Tests for the drain() method."""

    def test_drain_only_from_ready(self):
        """drain() is a no-op if not in READY state."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = IDLE
        mgr.drain(timeout=1)
        assert mgr.state == IDLE  # no transition

    @patch("modelpool.worker.loader.requests.get")
    def test_drain_succeeds_when_slots_zero(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"slots_processing": 0},
        )

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.drain(timeout=5)

        assert mgr.state == DRAINING

    @patch("modelpool.worker.loader.requests.get")
    def test_drain_waits_for_slots_to_clear(self, mock_get):
        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"slots_processing": 2},
                )
            return MagicMock(
                status_code=200,
                json=lambda: {"slots_processing": 0},
            )

        mock_get.side_effect = side_effect

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.drain(timeout=30)

        assert mgr.state == DRAINING
        assert call_count[0] >= 3

    @patch("modelpool.worker.loader.requests.get")
    def test_drain_timeout_forces_proceed(self, mock_get):
        """If slots never clear, drain() proceeds after timeout."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"slots_processing": 5},
        )

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.drain(timeout=1)

        assert mgr.state == DRAINING  # proceeds anyway

    @patch("modelpool.worker.loader.requests.get")
    def test_drain_connection_error_completes(self, mock_get):
        """If server is unreachable, drain completes (already stopped)."""
        mock_get.side_effect = Exception("connection refused")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.drain(timeout=5)

        assert mgr.state == DRAINING


# ============================================================
# Full Lifecycle (load_resource)
# ============================================================

class TestLoadResource:
    """Tests for the full load_resource cycle."""

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_load_resource_from_idle(self, mock_popen, mock_get):
        """Cold load: IDLE -> LOADING -> READY."""
        mock_popen.return_value = MagicMock(pid=123)
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        assert mgr.state == IDLE

        res = make_resource()
        mgr.load_resource(res)

        assert mgr.state == READY
        assert mgr.loaded_resource == "test-model"

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_load_resource_from_ready_drains_and_swaps(self, mock_popen, mock_get):
        """Hot swap: READY -> DRAINING -> STOPPING -> (stop) -> LOADING -> READY."""
        # First call: start the initial model
        mock_popen.return_value = MagicMock(pid=123)

        drain_call = {"count": 0}
        health_call = {"count": 0}

        def get_side_effect(url, **kwargs):
            if "/health" in url:
                health_call["count"] += 1
                return MagicMock(
                    status_code=200,
                    json=lambda: {"slots_processing": 0, "slots_idle": 2},
                )
            return MagicMock(status_code=200)

        mock_get.side_effect = get_side_effect

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")

        # Load first resource
        res1 = make_resource(name="model-a")
        mgr.load_resource(res1)
        assert mgr.state == READY
        assert mgr.loaded_resource == "model-a"

        # Swap to second resource
        res2 = make_resource(name="model-b")
        mgr.load_resource(res2, drain_timeout=5)
        assert mgr.state == READY
        assert mgr.loaded_resource == "model-b"

    def test_load_resource_rejects_when_loading(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        with pytest.raises(StateError, match="Already loading"):
            mgr.load_resource(make_resource())


# ============================================================
# Unload
# ============================================================

class TestUnload:

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    @patch("modelpool.worker.loader.requests.get")
    def test_unload_from_ready(self, mock_get, mock_getpgid, mock_killpg):
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"slots_processing": 0}
        )
        mock_getpgid.return_value = 100

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = MagicMock(pid=123, wait=MagicMock(return_value=0))
        mgr.loaded_resource = "test"

        mgr.unload(drain_timeout=5)
        assert mgr.process is None

    def test_unload_from_idle_is_noop(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        assert mgr.state == IDLE
        mgr.unload()
        assert mgr.state == IDLE


# ============================================================
# Revert
# ============================================================

class TestRevert:

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_revert_loads_default_resource(self, mock_popen, mock_get):
        mock_popen.return_value = MagicMock(pid=123)
        mock_get.return_value = MagicMock(status_code=200)

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        registry = make_registry()

        mgr.revert(registry, "test-worker")

        assert mgr.state == READY
        assert mgr.loaded_resource == "test-model"


# ============================================================
# Status and Helpers
# ============================================================

class TestStatus:

    def test_status_idle(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        status = mgr.get_status()
        assert status["state"] == IDLE
        assert status["loaded_resource"] is None
        assert status["pid"] is None
        assert status["uptime_s"] is None

    @patch("modelpool.worker.loader.requests.get")
    @patch("modelpool.worker.loader.subprocess.Popen")
    def test_status_ready_with_slot_info(self, mock_popen, mock_get):
        mock_popen.return_value = MagicMock(pid=123)
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"slots_idle": 1, "slots_processing": 0},
        )

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = MagicMock(pid=123)
        mgr.loaded_resource = "test"
        mgr.started_at = time.time() - 60

        status = mgr.get_status()
        assert status["state"] == READY
        assert status["loaded_resource"] == "test"
        assert status["pid"] == 123
        assert status["uptime_s"] >= 59
        assert status["slots_idle"] == 1
        assert status["slots_processing"] == 0

    @patch("modelpool.worker.loader.requests.get")
    def test_status_ready_health_check_fails_gracefully(self, mock_get):
        mock_get.side_effect = Exception("unreachable")

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = READY
        mgr.process = MagicMock(pid=123)
        mgr.loaded_resource = "test"

        status = mgr.get_status()
        assert status["health_check"] == "failed"

    def test_is_ready(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        assert not mgr.is_ready()
        mgr.state = READY
        assert mgr.is_ready()


# ============================================================
# Cleanup
# ============================================================

class TestCleanup:

    def test_cleanup_resets_state_to_idle(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = STOPPING
        mgr.process = MagicMock()
        mgr.loaded_resource = "test"
        mgr.started_at = 100.0

        mgr._cleanup()

        assert mgr.process is None
        assert mgr.loaded_resource is None
        assert mgr.started_at is None
        assert mgr.state == IDLE

    def test_cleanup_preserves_loading_state(self):
        """If in LOADING state, cleanup should not reset to IDLE."""
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = LOADING
        mgr.process = MagicMock()

        mgr._cleanup()
        assert mgr.state == LOADING  # preserved

    def test_cleanup_preserves_error_state(self):
        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.state = ERROR
        mgr.process = MagicMock()

        mgr._cleanup()
        assert mgr.state == ERROR  # preserved


# ============================================================
# Kill Process
# ============================================================

class TestKillProcess:

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_kill_sends_sigkill(self, mock_getpgid, mock_killpg):
        mock_getpgid.return_value = 100
        mock_proc = MagicMock()
        mock_proc.pid = 123

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.process = mock_proc
        mgr.loaded_resource = "test"

        mgr._kill_process()

        mock_killpg.assert_called_with(100, signal.SIGKILL)
        assert mgr.process is None

    @patch("modelpool.worker.loader.os.killpg")
    @patch("modelpool.worker.loader.os.getpgid")
    def test_kill_handles_already_dead(self, mock_getpgid, mock_killpg):
        mock_getpgid.side_effect = ProcessLookupError("gone")
        mock_proc = MagicMock()
        mock_proc.pid = 123

        mgr = LlamaServerManager(8080, log_dir="/tmp/mp-test")
        mgr.process = mock_proc
        mgr.loaded_resource = "test"

        mgr._kill_process()
        assert mgr.process is None  # cleanup still runs
