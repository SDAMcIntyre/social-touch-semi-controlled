from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_dependency_error import PipelineDependencyError

# Keep references to non-modal popups so they are not garbage-collected.
_active_popups: list = []


def show_dependency_error_popup(error: "PipelineDependencyError") -> None:
    """Show a non-blocking QMessageBox informing the user of a cross-pipeline dependency failure.

    The popup stays visible until the user dismisses it, but does **not** block
    execution — the pipeline continues to the next session immediately.
    """
    try:
        import sys
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtCore import Qt

        app = QApplication.instance() or QApplication(sys.argv)

        session_label = error.session_name or "unknown"

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(f"Dependency Error — {session_label}")
        msg.setText(
            f"<b>Session:</b> {session_label}<br><br>"
            f"<b>A required prerequisite step has not been completed.</b><br><br>"
            f"{error}<br><br>"
            f"<b>Required pipeline:</b> {error.required_pipeline}"
        )
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)

        # Remove from the reference list when the user closes the popup.
        msg.finished.connect(lambda: _active_popups.remove(msg))
        _active_popups.append(msg)

        msg.show()
        msg.raise_()
        msg.activateWindow()
    except Exception as popup_exc:
        print(f"[dependency_popup] Could not show popup: {popup_exc}")
