"""
Kinect RGB ↔ Depth co-registration audit tool.

Opens a small launcher window with a file picker.  Once an MKV is chosen the
viewer is embedded directly in the same window.

Usage
-----
    python audit_rgb_depth_registration.py

The viewer shows the raw color frame and the transformed-depth-Z channel
side-by-side in a shared absolute pixel coordinate system.  Drag the frame
slider to scrub through the recording.  Click on either panel to place a
synchronised crosshair.  Enable the "Canny overlay" checkbox to draw Canny
edges (from the color frame) over the depth panel; adjust the t1 / t2 sliders
to tune the thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add code/src so that preprocessing imports resolve correctly.
# This script lives under code/scripts/, so code/src is two levels up.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent  # _0_kinect_diagnostics/
_CODE_SRC = _SCRIPTS_DIR.parent.parent.parent / "src"  # code/src/
if str(_CODE_SRC) not in sys.path:
    sys.path.insert(0, str(_CODE_SRC))

from PyQt5.QtWidgets import (  # noqa: E402 — after path setup
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from preprocessing.common.gui import KinectRgbDepthViewer  # noqa: E402


class _AuditWindow(QMainWindow):
    """
    Launcher window for the RGB ↔ Depth registration audit.

    Shows a file-picker header at the top.  Once an MKV is selected the
    KinectRgbDepthViewer is embedded as the central widget.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RGB ↔ Depth Registration Audit")
        self.resize(1280, 800)

        # --- Header bar (always visible) ---
        self._lbl_path = QLabel("No file selected.")
        self._lbl_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_browse = QPushButton("Browse MKV…")
        btn_browse.setFixedWidth(130)
        btn_browse.clicked.connect(self._on_browse)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.addWidget(btn_browse)
        header_layout.addSpacing(12)
        header_layout.addWidget(self._lbl_path)

        # --- Placeholder shown before a file is chosen ---
        placeholder = QLabel("Open an MKV file to begin.")
        placeholder.setStyleSheet("color: grey; font-size: 14px;")

        # --- Root widget ---
        root = QWidget()
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)
        self._root_layout.addWidget(header)
        self._root_layout.addWidget(placeholder, stretch=1)

        self.setCentralWidget(root)
        self._viewer: KinectRgbDepthViewer | None = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Kinect MKV",
            str(Path.home()),
            "MKV files (*.mkv);;All files (*)",
        )
        if not path:
            return  # user cancelled

        mkv_path = Path(path)
        self._lbl_path.setText(str(mkv_path))
        self.setWindowTitle(f"RGB ↔ Depth Registration Audit — {mkv_path.name}")
        self._load_viewer(mkv_path)

    def _load_viewer(self, mkv_path: Path) -> None:
        """Replace (or create) the embedded viewer for ``mkv_path``."""
        # Close the previous viewer cleanly before replacing it so the old
        # MKV handle is released.
        if self._viewer is not None:
            self._viewer.close()

        # parallax_correction=False: the audit tool must measure the *raw*
        # RGB↔depth offset, not absorb the median-shift correction that is
        # applied by default to production data. See
        # docs/development/knowledge-base/note-kinect-depth-access-single-path.md
        # (§"Audit-mode opt-out") for the invariant this deliberately bypasses.
        viewer = KinectRgbDepthViewer(mkv_path, parallax_correction=False)
        self._viewer = viewer

        # Swap out everything below the header.
        layout = self._root_layout
        # Remove all items below index 0 (the header).
        while layout.count() > 1:
            item = layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        layout.addWidget(viewer, stretch=1)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = _AuditWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
