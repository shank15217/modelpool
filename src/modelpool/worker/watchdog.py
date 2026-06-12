"""Worker watchdog - background health monitor for llama-server.

Simplified for Architecture A: basic health monitoring only, no auto-recovery
(since there's no dynamic swapping to fall back to).
"""

from __future__ import annotations

import asyncio
import logging

import requests

from modelpool.worker.loader import LlamaServerManager, READY

logger = logging.getLogger("modelpool.worker.watchdog")


class Watchdog:
    """Monitors llama-server health and logs failures."""

    def __init__(
        self,
        manager: LlamaServerManager,
        check_interval: int = 15,
        failure_threshold: int = 3,
    ):
        self.manager = manager
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
                f"Health check failed {self.failure_threshold} times consecutively. "
                f"Manual intervention may be needed."
            )
            # Reset counter so we don't spam logs every check
            self._consecutive_failures = 0
