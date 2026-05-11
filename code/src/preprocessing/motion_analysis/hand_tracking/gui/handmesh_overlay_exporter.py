from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QStatusBar,
    QToolBar, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

from primary_processing import KinectConfig
from preprocessing.motion_analysis.hand_tracking.handmesh_overlay_renderer import (
    BatchVideoRenderer,
)

logger = logging.getLogger(__name__)


def launch_handmesh_overlay_exporter(blocks: list[KinectConfig]) -> None:
    """Open the exporter GUI and block until the window is closed."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = HandmeshOverlayExporter(blocks)
    win.show()
    app.exec_()


class HandmeshOverlayExporter(QMainWindow):
    def __init__(self, blocks: list[KinectConfig], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hand-Model Overlay Exporter")
        self.resize(700, 120)

        # Build lookup indices
        # _configs: (session_id, block_id) -> KinectConfig
        # _session_to_blocks: session_id -> sorted list[block_id]
        self._configs: dict[tuple[str, str], KinectConfig] = {}
        self._session_to_blocks: dict[str, list[str]] = {}
        for config in blocks:
            key = (config.session_id, config.block_id)
            self._configs[key] = config
            self._session_to_blocks.setdefault(config.session_id, []).append(config.block_id)
        for session in self._session_to_blocks:
            self._session_to_blocks[session].sort()

        self._build_ui()
        if self._session_combo.count() > 0:
            self._on_session_changed(0)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(200)
        for session_id in sorted(self._session_to_blocks):
            self._session_combo.addItem(session_id)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Block-order:"))
        self._block_combo = QComboBox()
        self._block_combo.setMinimumWidth(160)
        self._block_combo.currentIndexChanged.connect(self._on_block_changed)
        toolbar.addWidget(self._block_combo)

        toolbar.addSeparator()

        self._export_btn = QPushButton("Export Video")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        toolbar.addWidget(self._export_btn)

        central = QWidget()
        self.setCentralWidget(central)
        QVBoxLayout(central)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Select a session and block-order.")

    def _on_session_changed(self, index: int) -> None:
        session_id = self._session_combo.currentText()
        # Guard against signal re-entry while repopulating (note-qt-itemchanged-signal-recursion)
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for block_id in self._session_to_blocks.get(session_id, []):
            self._block_combo.addItem(block_id)
        self._block_combo.blockSignals(False)
        self._block_combo.setCurrentIndex(0)
        self._on_block_changed(0)

    def _on_block_changed(self, index: int) -> None:
        session_id = self._session_combo.currentText()
        block_id = self._block_combo.currentText()
        if not session_id or not block_id:
            self._export_btn.setEnabled(False)
            self._status_bar.showMessage("No block selected.")
            return

        config = self._configs.get((session_id, block_id))
        if config is None:
            self._export_btn.setEnabled(False)
            self._status_bar.showMessage("Block not found in index.")
            return

        rgb_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
        pkl_path = (
            config.video_processed_output_dir
            / "kinematics_analysis"
            / f"{config.source_video.stem}_handmodel_tracked_hands.pkl"
        )

        rgb_ok = rgb_path.exists()
        pkl_ok = pkl_path.exists()

        parts = []
        if not rgb_ok:
            parts.append("RGB: missing")
        if not pkl_ok:
            parts.append("Hand model: missing")

        if parts:
            self._status_bar.showMessage("  |  ".join(parts))
            self._export_btn.setEnabled(False)
        else:
            self._status_bar.showMessage(
                f"RGB: {rgb_path.name}  |  Hand model: {pkl_path.name}"
            )
            self._export_btn.setEnabled(True)

    def _on_export_clicked(self) -> None:
        session_id = self._session_combo.currentText()
        block_id = self._block_combo.currentText()
        config = self._configs[(session_id, block_id)]

        rgb_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
        pkl_path = (
            config.video_processed_output_dir
            / "kinematics_analysis"
            / f"{config.source_video.stem}_handmodel_tracked_hands.pkl"
        )

        if not rgb_path.exists() or not pkl_path.exists():
            missing = []
            if not rgb_path.exists():
                missing.append(str(rgb_path))
            if not pkl_path.exists():
                missing.append(str(pkl_path))
            QMessageBox.critical(
                self,
                "Missing files",
                "Required files are missing:\n" + "\n".join(missing),
            )
            return

        proposed_name = f"{config.source_video.stem}_handmodel_overlay.mp4"
        proposed_path = pkl_path.parent / proposed_name

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save overlay video",
            str(proposed_path),
            "MP4 Files (*.mp4)",
        )
        if not filename:
            return

        output_path = Path(filename)

        # Count frames via VideoMP4Manager to set the progress dialog max.
        # We don't have a fast frame count without opening the video, so
        # use a progress dialog with unknown max initially and update in the callback.
        progress = QProgressDialog("Rendering overlay video...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        cancelled = [False]

        def progress_cb(current: int, total: int) -> bool:
            if progress.maximum() != total:
                progress.setMaximum(total)
            progress.setValue(current)
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled[0] = True
                return True
            return False

        try:
            renderer = BatchVideoRenderer(rgb_path, pkl_path, output_path)
            renderer.render(progress_cb=progress_cb)
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Render failed", str(exc))
            return

        progress.close()

        if cancelled[0]:
            self._status_bar.showMessage(f"Cancelled — partial file: {output_path.name}")
        else:
            self._status_bar.showMessage(f"Saved: {output_path}")
            QMessageBox.information(self, "Export complete", f"Video saved to:\n{output_path}")
