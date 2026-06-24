from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_dependency_error import PipelineDependencyError


def show_dependency_error_popup(error: "PipelineDependencyError") -> None:
    """Show a modal QMessageBox informing the user of a cross-pipeline dependency failure.

    Blocks the calling thread until the user dismisses the dialog; the
    pipeline then continues to the next session. Must be called from the
    main thread — Qt GUI operations are not thread-safe. If invoked from
    a worker thread (e.g. under Prefect parallel execution) the popup is
    skipped and the error is logged instead.
    """
    try:
        import sys
        import threading
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtCore import Qt

        if threading.current_thread() is not threading.main_thread():
            print(
                f"[dependency_popup] Skipping modal popup (called from worker thread): "
                f"session={error.session_name!r} required_pipeline={error.required_pipeline!r} msg={error}"
            )
            return

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

        msg.exec_()
    except Exception as popup_exc:
        print(f"[dependency_popup] Could not show popup: {popup_exc}")
