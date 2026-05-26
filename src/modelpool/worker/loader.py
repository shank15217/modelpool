"""Worker subprocess manager - lifecycle for llama-server child processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from modelpool.registry import Registry, Resource

logger = logging.getLogger("modelpool.worker")

# Worker states
IDLE = "idle"
LOADING = "loading"
READY = "ready"
DRAINING = "draining"
STOPPING = "stopping"
ERROR = "error"

VALID_TRANSITIONS = {
    IDLE: {LOADING},
    LOADING: {READY, ERROR},
    READY: {DRAINING, LOADING, ERROR},
    DRAINING: {STOPPING, ERROR},
    STOPPING: {IDLE, LOADING, ERROR},
    ERROR: {IDLE, LOADING},
}


class StateError(Exception):
    """Raised on invalid state transitions."""


class LoadError(Exception):
    """Raised when a resource fails to load."""


class LlamaServerManager:
    """Manages a single llama-server subprocess on this host."""

    def __init__(self, inference_port: int, log_dir: str = "/var/log/modelpool"):
        self.inference_port = inference_port
        self.log_dir = log_dir
        self.state: str = IDLE
        self.loaded_resource: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.started_at: Optional[float] = None
        self._log_file = None

        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def _transition(self, new_state: str) -> None:
        """Enforce valid state machine transitions."""
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            raise StateError(
                f"Invalid transition: {self.state} -> {new_state} "
                f"(allowed: {VALID_TRANSITIONS.get(self.state, set())})"
            )
        logger.info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state

    def build_command(self, resource: Resource) -> list[str]:
        """Build the exact command line from a resource definition."""
        if not resource.binary:
            raise LoadError(f"Resource '{resource.name}' has no binary defined")

        cmd = [resource.binary]
        for flag in resource.flags:
            cmd.extend(flag)

        # Replace template variables
        port_str = str(self.inference_port)
        cmd = [s.replace("{inference_port}", port_str) for s in cmd]

        return cmd

    def start(self, resource: Resource, timeout: int = 120) -> None:
        """Start llama-server with a resource's exact command."""
        if self.state not in (IDLE, ERROR):
            raise StateError(f"Cannot start from state {self.state}")

        self._transition(LOADING)

        cmd = self.build_command(resource)
        logger.info(f"Starting resource '{resource.name}': {' '.join(cmd[:5])}...")

        # Open log file for this resource
        log_path = Path(self.log_dir) / f"llama-server-{resource.name}.log"
        self._log_file = open(log_path, "w")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            logger.info(f"Process started: PID={self.process.pid}")
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            self.state = ERROR
            if self._log_file:
                self._log_file.close()
            raise LoadError(f"Failed to start: {e}")

        # Wait for health check
        if not self._wait_healthy(timeout):
            self._kill_process()
            self.state = ERROR
            raise LoadError(
                f"Health check failed after {timeout}s for resource '{resource.name}'"
            )

        self.loaded_resource = resource.name
        self.started_at = time.time()
        self._transition(READY)
        logger.info(f"Resource '{resource.name}' loaded and healthy")

    def stop(self, timeout: int = 10) -> None:
        """Stop the running llama-server process."""
        if not self.process:
            self.state = IDLE
            self.loaded_resource = None
            return

        logger.info(f"Stopping process PID={self.process.pid}")

        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            # Process already dead
            self._cleanup()
            return

        try:
            self.process.wait(timeout=timeout)
            logger.info("Process exited cleanly on SIGTERM")
        except subprocess.TimeoutExpired:
            logger.warning(f"Process did not exit in {timeout}s, sending SIGKILL")
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass

        self._cleanup()

    def drain(self, timeout: int = 30) -> None:
        """Wait for all in-flight requests to complete."""
        if self.state != READY:
            return

        self._transition(DRAINING)
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://localhost:{self.inference_port}/health",
                    timeout=2,
                )
                data = resp.json()
                slots_processing = data.get("slots_processing", 0)
                if slots_processing == 0:
                    logger.info("Drain complete: no in-flight requests")
                    return
                logger.debug(f"Draining: {slots_processing} slots still processing")
            except requests.ConnectionError:
                logger.info("Drain complete: server unreachable (already stopped)")
                return
            except Exception as e:
                logger.debug(f"Drain check error: {e}")
            time.sleep(1)

        logger.warning(f"Drain timeout ({timeout}s): forcing proceed")

    def load_resource(self, resource: Resource, drain_timeout: int = 30, swap_timeout: int = 120) -> None:
        """Full lifecycle: drain current -> stop -> start new resource."""
        if self.state == LOADING:
            raise StateError("Already loading a resource")

        # If currently serving, drain and stop first
        if self.state == READY:
            self.drain(timeout=drain_timeout)
            self._transition(STOPPING) if self.state == DRAINING else None
            self.stop()

        # Start the new resource
        self.start(resource, timeout=swap_timeout)

    def unload(self, drain_timeout: int = 30) -> None:
        """Drain and stop, leaving the worker idle."""
        if self.state == READY:
            self.drain(timeout=drain_timeout)
        if self.process:
            if self.state == DRAINING:
                self._transition(STOPPING)
            self.stop()

    def revert(self, registry: Registry, worker_name: str) -> None:
        """Revert to the default resource for this worker."""
        default = registry.get_default_resource(worker_name)
        worker = registry.get_worker(worker_name)
        self.load_resource(
            default,
            drain_timeout=worker.drain_timeout,
            swap_timeout=worker.swap_timeout,
        )

    def get_status(self) -> dict:
        """Current status for /worker/status endpoint."""
        status = {
            "state": self.state,
            "loaded_resource": self.loaded_resource,
            "pid": self.process.pid if self.process else None,
            "uptime_s": int(time.time() - self.started_at) if self.started_at else None,
        }

        if self.state == READY and self.process:
            try:
                resp = requests.get(
                    f"http://localhost:{self.inference_port}/health",
                    timeout=2,
                )
                data = resp.json()
                status["slots_idle"] = data.get("slots_idle", 0)
                status["slots_processing"] = data.get("slots_processing", 0)
            except Exception:
                status["health_check"] = "failed"

        return status

    def is_ready(self) -> bool:
        """Check if the worker is ready to serve requests."""
        return self.state == READY

    def _wait_healthy(self, timeout: int) -> bool:
        """Poll /health until 200 OK."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"http://localhost:{self.inference_port}/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
            except Exception:
                pass
            time.sleep(2)
        return False

    def _kill_process(self) -> None:
        """Force kill the process."""
        if self.process:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass
            self._cleanup()

    def _cleanup(self) -> None:
        """Reset state after process stops."""
        self.process = None
        self.loaded_resource = None
        self.started_at = None
        if self.state not in (LOADING, ERROR):
            self.state = IDLE
        if self._log_file:
            self._log_file.close()
            self._log_file = None
