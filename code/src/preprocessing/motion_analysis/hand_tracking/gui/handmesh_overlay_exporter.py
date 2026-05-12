from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Optional

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton,
    QSizePolicy, QSlider, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

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
        self.resize(960, 640)

        self._configs: dict[tuple[str, str], KinectConfig] = {}
        self._session_to_blocks: dict[str, list[str]] = {}
        for config in blocks:
            key = (config.session_id, config.block_id)
            self._configs[key] = config
            self._session_to_blocks.setdefault(config.session_id, []).append(config.block_id)
        for session in self._session_to_blocks:
            self._session_to_blocks[session].sort()

        self._preview_renderer: Optional[BatchVideoRenderer] = None

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
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_label.setStyleSheet("background-color: #111; color: #666;")
        self._preview_label.setText("Select a block to preview")
        layout.addWidget(self._preview_label, 1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._frame_slider)
        self._frame_label = QLabel("— / —")
        self._frame_label.setFixedWidth(90)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Select a session and block-order.")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._preview_renderer is not None:
            self._update_preview_frame(self._frame_slider.value())

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
        self._preview_renderer = None
        self._frame_slider.setEnabled(False)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(0)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText("— / —")
        self._export_btn.setEnabled(False)
        self._preview_label.setText("Select a block to preview")

        session_id = self._session_combo.currentText()
        block_id = self._block_combo.currentText()
        if not session_id or not block_id:
            self._status_bar.showMessage("No block selected.")
            return

        config = self._configs.get((session_id, block_id))
        if config is None:
            self._status_bar.showMessage("Block not found in index.")
            return

        rgb_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
        pkl_path = (
            config.video_processed_output_dir
            / "kinematics_analysis"
            / f"{config.source_video.stem}_handmodel_tracked_hands.pkl"
        )

        parts = []
        if not rgb_path.exists():
            parts.append("RGB: missing")
        if not pkl_path.exists():
            parts.append("Hand model: missing")

        if parts:
            msg = "  |  ".join(parts)
            self._status_bar.showMessage(msg)
            self._preview_label.setText(msg)
            return

        self._status_bar.showMessage(
            f"RGB: {rgb_path.name}  |  Hand model: {pkl_path.name}"
        )
        self._preview_label.setText("Loading…")
        QApplication.processEvents()

        self._preview_renderer = BatchVideoRenderer(rgb_path, pkl_path)
        n = self._preview_renderer.frame_count
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, n - 1)
        self._frame_slider.setValue(0)
        self._frame_slider.blockSignals(False)
        self._frame_slider.setEnabled(True)
        self._frame_label.setText(f"0 / {n - 1}")
        self._export_btn.setEnabled(True)
        self._update_preview_frame(0)

    def _on_slider_changed(self, value: int) -> None:
        if self._preview_renderer is None:
            return
        n = self._preview_renderer.frame_count
        self._frame_label.setText(f"{value} / {n - 1}")
        self._update_preview_frame(value)

    def _update_preview_frame(self, frame_idx: int) -> None:
        frame_bgr = self._preview_renderer.render_frame(frame_idx)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._preview_label.setPixmap(
            pixmap.scaled(
                self._preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

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
