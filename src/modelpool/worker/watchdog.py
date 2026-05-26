"""Worker watchdog - background health monitor for llama-server."""

from __future__ import annotations

import asyncio
import logging
import time

import requests

from modelpool.worker.loader import LlamaServerManager, READY, ERROR
from modelpool.registry import Registry

logger = logging.getLogger("modelpool.worker.watchdog")


class Watchdog:
    """Monitors llama-server health and auto-recovers on failure."""

    def __init__(
        self,
        manager: LlamaServerManager,
        registry: Registry,
        worker_name: str,
        check_interval: int = 15,
        failure_threshold: int = 3,
    ):
        self.manager = manager
        self.registry = registry
        self.worker_name = worker_name
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Start the watchdog background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(
            f"Watchdog started (interval={self.check_interval}s, "
            f"threshold={self.failure_threshold})"
        )

    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Watchdog stopped")

    async def _run(self) -> None:
        """Main watchdog loop."""
        while self._running:
            await asyncio.sleep(self.check_interval)
            try:
                self._check()
            except Exception as e:
                logger.error(f"Watchdog check error: {e}")

    def _check(self) -> None:
        """Perform a single health check."""
        if self.manager.state != READY:
            return

        try:
            resp = requests.get(
                f"http://localhost:{self.manager.inference_port}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                if self._consecutive_failures > 0:
                    logger.info("Health check recovered after failures")
                self._consecutive_failures = 0
                return
        except requests.ConnectionError:
            pass
        except Exception as e:
            logger.debug(f"Health check error: {e}")

        self._consecutive_failures += 1
        logger.warning(
            f"Health check failed ({self._consecutive_failures}/{self.failure_threshold})"
        )

        if self._consecutive_failures >= self.failure_threshold:
            logger.error(
                f"Health check failed {self.failure_threshold} times, "
                f"triggering auto-recovery"
            )
            self._recover()

    def _recover(self) -> None:
        """Attempt to recover by restarting with the default resource."""
        self.manager.state = ERROR
        self._consecutive_failures = 0

        try:
            # Force stop the broken process
            self.manager.stop(timeout=5)
        except Exception as e:
            logger.error(f"Failed to stop broken process: {e}")
            self.manager.process = None
            self.manager.state = ERROR

        try:
            # Load the default resource
            worker = self.registry.get_worker(self.worker_name)
            default = self.registry.get_default_resource(self.worker_name)
            logger.info(f"Auto-recovering with default resource: {default.name}")
            self.manager.start(default, timeout=worker.swap_timeout)
            logger.info("Auto-recovery successful")
        except Exception as e:
            logger.error(f"Auto-recovery failed: {e}")
