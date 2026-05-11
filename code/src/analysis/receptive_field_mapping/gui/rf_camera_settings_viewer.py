"""RF Camera Settings Viewer — per-session camera orientation GUI.

Presents an interactive PyVista 3D forearm mesh with contact-point heatmap.
Allows the researcher to set and save camera orientation per session.
Saved settings are consumed by downstream RF rendering and projection tasks.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.rf_extraction_io import (
    load_rf_camera_settings,
    save_rf_camera_settings,
)
from analysis.receptive_field_mapping.touch_population_data import PopulationData

logger = logging.getLogger(__name__)


class RFCameraSettingsViewer(QMainWindow):
    def __init__(
        self,
        sessions: List[Tuple[str, PopulationData]],
        output_dir: Path,
        title: str = "RF Camera Settings",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._sessions = sessions
        self._output_dir = output_dir
        self._session_cameras: dict = {}
        self._current_session_id: Optional[str] = None
        self._initialized = False

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for session_id, _ in sessions:
            self._session_combo.addItem(session_id)
        toolbar.addWidget(self._session_combo)
        self.addToolBar(toolbar)

        self._plotter = QtInteractor(self)
        self._plotter.set_background("white")

        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(4, 4, 4, 4)
        self._save_btn = QPushButton("Save Camera Settings")
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bottom_layout.addWidget(self._save_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self._status_label)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plotter.interactor, stretch=1)
        layout.addWidget(bottom_bar)
        self.setCentralWidget(central)

        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._save_btn.clicked.connect(self._on_save_clicked)

        self._update_combo_backgrounds()

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self):
        try:
            self._plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self._plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self._plotter.render_window.SetSize(sz.width(), sz.height())
        self._load_session(0)

    def closeEvent(self, event):
        self._plotter.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _capture_camera(self) -> dict:
        cam = self._plotter.camera
        return {
            "camera_position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up_vector": list(cam.up),
            "view_angle": float(cam.view_angle),
        }

    def _restore_camera(self, params: dict) -> None:
        self._plotter.camera.position = params["camera_position"]
        self._plotter.camera.focal_point = params["focal_point"]
        self._plotter.camera.up = params["up_vector"]
        self._plotter.camera.view_angle = params["view_angle"]
        self._plotter.renderer.ResetCameraClippingRange()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _on_session_changed(self, index: int) -> None:
        if self._current_session_id is not None:
            self._session_cameras[self._current_session_id] = self._capture_camera()
        self._load_session(index)

    def _load_session(self, index: int) -> None:
        session_id, pop_data = self._sessions[index]
        self._current_session_id = session_id
        self._build_scene(pop_data, session_id)

        if session_id in self._session_cameras:
            self._restore_camera(self._session_cameras[session_id])
        else:
            cameras = load_rf_camera_settings(self._output_dir)
            if session_id in cameras:
                self._restore_camera(cameras[session_id])
            else:
                self._plotter.view_xy()
                self._plotter.reset_camera()

        self._status_label.setText("")

    def _build_scene(self, pop_data: PopulationData, session_id: str) -> None:
        vertices = pop_data.forearm_vertices
        heatmap = np.bincount(pop_data.cp_vertex_idx, minlength=len(vertices)).astype(float)
        heatmap[heatmap == 0] = np.nan
        cloud = pv.PolyData(vertices)
        cloud["contact_count"] = heatmap
        self._plotter.clear()
        self._plotter.add_mesh(
            cloud,
            scalars="contact_count",
            cmap="hot",
            nan_color="lightgrey",
            point_size=3,
            show_scalar_bar=True,
            render_points_as_spheres=False,
            name="forearm",
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        index = self._session_combo.currentIndex()
        session_id = self._session_combo.itemText(index)

        cameras = load_rf_camera_settings(self._output_dir)
        if session_id in cameras:
            reply = QMessageBox.question(
                self,
                "Overwrite camera settings?",
                f"Overwrite existing settings for {session_id}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        cameras[session_id] = self._capture_camera()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        save_rf_camera_settings(self._output_dir, cameras)
        self._update_combo_backgrounds()
        self._status_label.setText(f"Saved camera settings for {session_id}.")

    # ------------------------------------------------------------------
    # Combo background
    # ------------------------------------------------------------------

    def _update_combo_backgrounds(self) -> None:
        cameras = load_rf_camera_settings(self._output_dir)
        combo = self._session_combo
        combo.blockSignals(True)
        for i in range(combo.count()):
            sid = combo.itemText(i)
            if sid in cameras:
                combo.setItemData(i, QColor("#90EE90"), Qt.BackgroundRole)
            else:
                combo.setItemData(i, None, Qt.BackgroundRole)
        combo.blockSignals(False)
