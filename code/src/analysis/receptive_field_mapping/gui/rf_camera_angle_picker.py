"""Multi-session interactive viewer for picking RF camera angles.

Displays one session at a time via a dropdown, with a settings panel for
full rendering control.  Camera angles are saved per-session to
``camera_params.json``.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

logger = logging.getLogger(__name__)

# Available colormaps for contact points
_COLORMAPS = ["YlOrRd", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet"]


@dataclass
class _ViewerSettings:
    """Global rendering settings that persist across session switches."""

    bg_color: str = "white"
    forearm_color: str = "lightgrey"
    forearm_size: int = 2
    forearm_opacity: float = 1.0
    forearm_spheres: bool = True
    contact_cmap: str = "YlOrRd"
    contact_size: int = 6
    contact_opacity: float = 1.0
    contact_spheres: bool = True
    show_scalar_bar: bool = True
    show_axes: bool = True


class RFCameraAnglePicker(QMainWindow):
    """Multi-session PyVista viewer for interactively choosing camera angles.

    Accepts a dict of ``SessionSceneData`` (one per session).  A dropdown
    lets the researcher switch sessions; rendering settings are global and
    persist across switches.  "Save Camera" writes the current session's
    camera to disk; "Save All & Close" writes all modified cameras.

    Uses the deferred-render pattern from ``PostprocessedSceneViewer``.

    Parameters
    ----------
    sessions : dict
        Mapping of session_id → ``SessionSceneData``.
    """

    def __init__(self, sessions: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick RF Camera Angle")

        self._sessions = sessions
        self._session_ids: List[str] = sorted(sessions.keys())
        self._current_id: str = self._session_ids[0]
        self._settings = _ViewerSettings()
        self._initial_render_done = False

        # Per-session in-memory camera state: modified but not yet saved
        self._modified_cameras: Dict[str, dict] = {}
        # Cameras written to disk during this session
        self.saved_cameras: Dict[str, dict] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Top toolbar ---
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Session:"))
        self._combo = QComboBox()
        for sid in self._session_ids:
            self._combo.addItem(self._combo_label(sid))
        self._combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._combo, stretch=1)

        self._save_btn = QPushButton("Save Camera")
        self._save_btn.clicked.connect(self._on_save_camera)
        toolbar.addWidget(self._save_btn)

        self._save_all_btn = QPushButton("Save All && Close")
        self._save_all_btn.clicked.connect(self._on_save_all_and_close)
        toolbar.addWidget(self._save_all_btn)
        root.addLayout(toolbar)

        # --- Body: plotter + settings panel ---
        body = QHBoxLayout()

        self.plotter = QtInteractor(central)
        body.addWidget(self.plotter.interactor, stretch=4)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFixedWidth(260)
        settings_widget = QWidget()
        self._settings_layout = QVBoxLayout(settings_widget)
        self._build_settings_panel()
        self._settings_layout.addStretch()
        settings_scroll.setWidget(settings_widget)
        body.addWidget(settings_scroll)

        root.addLayout(body)

    def _build_settings_panel(self) -> None:
        s = self._settings

        # --- Background ---
        bg_box = QGroupBox("Background")
        bg_lay = QVBoxLayout(bg_box)
        self._bg_color_btn = self._color_button(s.bg_color)
        self._bg_color_btn.clicked.connect(lambda: self._pick_color("bg_color", self._bg_color_btn))
        bg_lay.addWidget(self._bg_color_btn)
        self._settings_layout.addWidget(bg_box)

        # --- Forearm ---
        fa_box = QGroupBox("Forearm")
        fa_lay = QVBoxLayout(fa_box)

        self._fa_color_btn = self._color_button(s.forearm_color)
        self._fa_color_btn.clicked.connect(lambda: self._pick_color("forearm_color", self._fa_color_btn))
        row = QHBoxLayout()
        row.addWidget(QLabel("Color:"))
        row.addWidget(self._fa_color_btn)
        fa_lay.addLayout(row)

        self._fa_size_slider = self._slider("Point size:", 1, 20, s.forearm_size, fa_lay)
        self._fa_size_slider.valueChanged.connect(lambda v: self._set_and_rebuild("forearm_size", v))

        self._fa_opacity_slider = self._slider("Opacity:", 0, 100, int(s.forearm_opacity * 100), fa_lay)
        self._fa_opacity_slider.valueChanged.connect(lambda v: self._set_and_rebuild("forearm_opacity", v / 100.0))

        self._fa_spheres_cb = QCheckBox("Render as spheres")
        self._fa_spheres_cb.setChecked(s.forearm_spheres)
        self._fa_spheres_cb.stateChanged.connect(lambda st: self._set_and_rebuild("forearm_spheres", st == Qt.Checked))
        fa_lay.addWidget(self._fa_spheres_cb)
        self._settings_layout.addWidget(fa_box)

        # --- Contact Points ---
        cp_box = QGroupBox("Contact Points")
        cp_lay = QVBoxLayout(cp_box)

        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setCurrentText(s.contact_cmap)
        self._cmap_combo.currentTextChanged.connect(lambda t: self._set_and_rebuild("contact_cmap", t))
        row_cmap.addWidget(self._cmap_combo)
        cp_lay.addLayout(row_cmap)

        self._cp_size_slider = self._slider("Point size:", 1, 30, s.contact_size, cp_lay)
        self._cp_size_slider.valueChanged.connect(lambda v: self._set_and_rebuild("contact_size", v))

        self._cp_opacity_slider = self._slider("Opacity:", 0, 100, int(s.contact_opacity * 100), cp_lay)
        self._cp_opacity_slider.valueChanged.connect(lambda v: self._set_and_rebuild("contact_opacity", v / 100.0))

        self._cp_spheres_cb = QCheckBox("Render as spheres")
        self._cp_spheres_cb.setChecked(s.contact_spheres)
        self._cp_spheres_cb.stateChanged.connect(lambda st: self._set_and_rebuild("contact_spheres", st == Qt.Checked))
        cp_lay.addWidget(self._cp_spheres_cb)
        self._settings_layout.addWidget(cp_box)

        # --- Display ---
        disp_box = QGroupBox("Display")
        disp_lay = QVBoxLayout(disp_box)

        self._scalar_bar_cb = QCheckBox("Scalar bar")
        self._scalar_bar_cb.setChecked(s.show_scalar_bar)
        self._scalar_bar_cb.stateChanged.connect(lambda st: self._set_and_rebuild("show_scalar_bar", st == Qt.Checked))
        disp_lay.addWidget(self._scalar_bar_cb)

        self._axes_cb = QCheckBox("Axes")
        self._axes_cb.setChecked(s.show_axes)
        self._axes_cb.stateChanged.connect(lambda st: self._set_and_rebuild("show_axes", st == Qt.Checked))
        disp_lay.addWidget(self._axes_cb)
        self._settings_layout.addWidget(disp_box)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _color_button(color_name: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        qc = QColor(color_name)
        btn.setStyleSheet(f"background-color: {qc.name()};")
        return btn

    @staticmethod
    def _slider(label: str, lo: int, hi: int, val: int, layout: QVBoxLayout) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(lo)
        sl.setMaximum(hi)
        sl.setValue(val)
        row.addWidget(sl)
        layout.addLayout(row)
        return sl

    def _pick_color(self, attr: str, btn: QPushButton) -> None:
        current = QColor(getattr(self._settings, attr))
        color = QColorDialog.getColor(current, self, f"Pick {attr}")
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()};")
            self._set_and_rebuild(attr, color.name())

    # ------------------------------------------------------------------
    # Settings change → rebuild
    # ------------------------------------------------------------------
    def _set_and_rebuild(self, attr: str, value) -> None:
        setattr(self._settings, attr, value)
        self._apply_setting_change()

    def _apply_setting_change(self) -> None:
        """Preserve camera, clear scene, rebuild with current settings."""
        cam = self._capture_camera()
        self.plotter.clear()
        self._build_scene()
        self._restore_camera(cam)

    # ------------------------------------------------------------------
    # Session switching
    # ------------------------------------------------------------------
    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._session_ids):
            return
        # Store current camera in modified state
        self._modified_cameras[self._current_id] = self._capture_camera()
        self._update_combo_display(self._current_id)

        self._current_id = self._session_ids[index]
        self.plotter.clear()
        self._build_scene()

        # Restore camera: modified > saved > default
        cam = self._modified_cameras.get(self._current_id)
        if cam is None:
            saved = self._sessions[self._current_id].saved_camera_params
            if saved is not None:
                cam = saved
        if cam is not None:
            self._restore_camera(cam)

    def _combo_label(self, session_id: str) -> str:
        if session_id in self.saved_cameras:
            return f"{session_id} (saved)"
        if session_id in self._modified_cameras:
            return f"{session_id} (modified)"
        data = self._sessions[session_id]
        if data.saved_camera_params is not None:
            return f"{session_id} (saved)"
        return session_id

    def _update_combo_display(self, session_id: Optional[str] = None) -> None:
        ids = [session_id] if session_id else self._session_ids
        self._combo.blockSignals(True)
        for sid in ids:
            idx = self._session_ids.index(sid)
            self._combo.setItemText(idx, self._combo_label(sid))
        self._combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Deferred render
    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._initial_render_done:
            self._initial_render_done = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        try:
            self.plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self.plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self.plotter.render_window.SetSize(sz.width(), sz.height())
        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        data = self._sessions[self._current_id]
        s = self._settings

        self.plotter.set_background(s.bg_color)

        # Determine centroid
        if data.contact_points is not None and len(data.contact_points) > 0:
            centroid = data.contact_points.mean(axis=0)
        else:
            centroid = data.forearm_points.mean(axis=0)

        # Bounding proxy
        box_half = 400.0
        cx, cy, cz = centroid.tolist()
        bounds_proxy = pv.Box(bounds=(
            cx - box_half, cx + box_half,
            cy - box_half, cy + box_half,
            cz - box_half, cz + box_half,
        ))
        self.plotter.add_mesh(
            bounds_proxy, opacity=0.001, name="_bounds_proxy", pickable=False,
        )

        # Forearm point cloud
        forearm_cloud = pv.PolyData(data.forearm_points)
        self.plotter.add_mesh(
            forearm_cloud,
            color=s.forearm_color,
            point_size=s.forearm_size,
            opacity=s.forearm_opacity,
            render_points_as_spheres=s.forearm_spheres,
            name="forearm",
        )

        # Contact overlay
        if data.contact_points is not None and len(data.contact_points) > 0:
            contact_cloud = pv.PolyData(data.contact_points)
            has_scalars = (
                data.selectivity_scores is not None
                and len(data.selectivity_scores) == len(data.contact_points)
            )
            if has_scalars:
                contact_cloud["selectivity"] = data.selectivity_scores
                sbar_args = {"title": "Selectivity"} if s.show_scalar_bar else None
                self.plotter.add_mesh(
                    contact_cloud,
                    scalars="selectivity",
                    cmap=s.contact_cmap,
                    clim=[0, 1],
                    point_size=s.contact_size,
                    opacity=s.contact_opacity,
                    render_points_as_spheres=s.contact_spheres,
                    scalar_bar_args=sbar_args,
                    show_scalar_bar=s.show_scalar_bar,
                    name="contacts",
                )
            else:
                self.plotter.add_mesh(
                    contact_cloud,
                    color="red",
                    point_size=s.contact_size,
                    opacity=s.contact_opacity,
                    render_points_as_spheres=s.contact_spheres,
                    name="contacts",
                )

        if s.show_axes:
            self.plotter.add_axes()

        # Default camera (only when no camera will be restored externally)
        offset_dist = 400.0
        if data.initial_normal is not None:
            cam_pos = centroid + data.initial_normal * offset_dist
        else:
            cam_pos = centroid + np.array([0.0, 0.0, offset_dist])

        self.plotter.camera.position = cam_pos.tolist()
        self.plotter.camera.focal_point = centroid.tolist()
        self.plotter.camera.up = (0.0, 1.0, 0.0)
        self.plotter.camera.view_angle = 30.0

        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------
    def _capture_camera(self) -> dict:
        cam = self.plotter.camera
        return {
            "camera_position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up_vector": list(cam.up),
            "view_angle": float(cam.view_angle),
        }

    @staticmethod
    def _restore_camera_to_plotter(plotter, params: dict) -> None:
        plotter.camera.position = params["camera_position"]
        plotter.camera.focal_point = params["focal_point"]
        plotter.camera.up = params["up_vector"]
        plotter.camera.view_angle = params["view_angle"]
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()

    def _restore_camera(self, params: dict) -> None:
        self._restore_camera_to_plotter(self.plotter, params)

    # ------------------------------------------------------------------
    # Save actions
    # ------------------------------------------------------------------
    def _on_save_camera(self) -> None:
        cam = self._capture_camera()
        self.saved_cameras[self._current_id] = cam
        self._modified_cameras.pop(self._current_id, None)
        self._update_combo_display(self._current_id)
        logger.info("Camera saved for %s", self._current_id)

    def _on_save_all_and_close(self) -> None:
        # Save current camera if modified
        if self._current_id not in self.saved_cameras:
            self._modified_cameras[self._current_id] = self._capture_camera()
        # Write all modified cameras
        for sid, cam in self._modified_cameras.items():
            if sid not in self.saved_cameras:
                self.saved_cameras[sid] = cam
        self.close()
