from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_dependency_error import PipelineDependencyError


def show_dependency_error_popup(error: "PipelineDependencyError") -> None:
    """Show a blocking QMessageBox informing the user of a cross-pipeline dependency failure."""
    try:
        import sys
        from PyQt5.QtWidgets import QApplication, QMessageBox

        from PyQt5.QtCore import Qt

        app = QApplication.instance() or QApplication(sys.argv)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Pipeline Dependency Error")
        msg.setText(
            f"<b>A required prerequisite step has not been completed.</b><br><br>"
            f"{error}<br><br>"
            f"<b>Required pipeline:</b> {error.required_pipeline}"
        )
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
        msg.show()
        msg.raise_()
        msg.activateWindow()
        msg.exec_()
    except Exception as popup_exc:
        print(f"[dependency_popup] Could not show popup: {popup_exc}")

    sys.exit(0)
