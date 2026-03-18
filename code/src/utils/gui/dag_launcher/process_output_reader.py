"""QThread worker that reads subprocess stdout line-by-line and emits each line as a signal."""

from __future__ import annotations

import re
import subprocess

from PyQt5.QtCore import QThread, pyqtSignal

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class ProcessOutputReader(QThread):
    """Reads stdout from a running subprocess and emits each decoded line via signal.

    The thread terminates naturally when the subprocess pipe closes (on process
    exit or termination).  The subprocess must be created with ``stdout=PIPE``.
    """

    line_received = pyqtSignal(str)
    cr_line_received = pyqtSignal(str)

    def __init__(self, process: subprocess.Popen, parent=None) -> None:
        super().__init__(parent)
        self._process = process

    def run(self) -> None:
        try:
            for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                line = _ANSI_RE.sub("", line)
                if "\r" in line:
                    # tqdm-style carriage-return overwrite — take last segment
                    segment = line.rsplit("\r", 1)[-1]
                    if segment:
                        self.cr_line_received.emit(segment)
                else:
                    self.line_received.emit(line)
        except Exception:
            pass
