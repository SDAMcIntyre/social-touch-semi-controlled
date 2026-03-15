"""Manages lifecycle of a persistent Prefect server process for the GUI."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class PrefectServerManager:
    """Start and stop a local Prefect server process.

    The server is started with ``start()`` and stopped with ``stop()``.
    If a server is already running on the target port when ``start()`` is
    called, it is reused and ``stop()`` will not terminate it.
    """

    def __init__(self, port: int = 4200) -> None:
        self._port = port
        self._process: subprocess.Popen | None = None
        self._owned: bool = False  # True only if *this* instance spawned the process

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def api_url(self) -> str:
        """Base API URL for the managed Prefect server."""
        return f"http://127.0.0.1:{self._port}/api"

    def get_env(self) -> dict[str, str]:
        """Return a copy of ``os.environ`` with ``PREFECT_API_URL`` set."""
        return {**os.environ, "PREFECT_API_URL": self.api_url}

    def start(self) -> None:
        """Start the server if it is not already running on the target port.

        If a server is already reachable (e.g. started externally), this
        method reuses it and marks the instance as *not* owning it — so
        ``stop()`` will leave it running.
        """
        if self._probe_health():
            logger.info(
                "Prefect server already running on port %d — reusing", self._port
            )
            self._owned = False
            return

        # Clean up a previously owned but crashed process before re-spawning.
        if self._owned and self._process is not None and self._process.poll() is not None:
            self._process = None
            self._owned = False

        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "prefect",
                "server",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                str(self._port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        self._owned = True
        logger.info(
            "Started Prefect server (PID %d) on port %d",
            self._process.pid,
            self._port,
        )

    def wait_until_ready(self, timeout_seconds: float = 30.0) -> bool:
        """Block until the health endpoint responds or the timeout expires.

        Calls ``QApplication.processEvents()`` between polls so the GUI
        remains responsive during the wait.

        Returns:
            ``True`` if the server became ready within the timeout,
            ``False`` otherwise.
        """
        from PyQt5.QtWidgets import QApplication

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._probe_health():
                return True
            QApplication.processEvents()
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        """Terminate the owned server process.

        No-op if the server was pre-existing (not started by this instance).
        """
        if not self._owned or self._process is None:
            return
        logger.info("Stopping Prefect server (PID %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully — killing")
            self._process.kill()
            self._process.wait()
        self._process = None
        self._owned = False

    def is_running(self) -> bool:
        """Return ``True`` if the server health endpoint is reachable."""
        if self._owned and self._process is not None:
            if self._process.poll() is not None:
                return False  # process has exited
        return self._probe_health()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_health(self) -> bool:
        """Return ``True`` if the health endpoint responds with HTTP 200."""
        try:
            with urllib.request.urlopen(f"{self.api_url}/health", timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False
