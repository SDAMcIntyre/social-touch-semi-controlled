"""
forearm_stage_inspector.py
--------------------------
Interactive 3D viewer for inspecting forearm pointclouds at each
postprocessing stage: Raw (Unified/Registered) → PCA-Calibrated → RF-Centered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)

STAGES: List[str] = [
    "Raw (Unified/Registered)",
    "PCA-Calibrated",
    "RF-Centered",
]
_STAGE_RAW = 0
_STAGE_PCA = 1
_STAGE_RF = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ForearmStagePaths:
    """PLY paths for all three postprocessing stages of one session."""

    session_id: str
    raw_ply: Optional[Path]
    pca_calibrated_ply: Optional[Path]
    rf_centered_ply: Optional[Path]


def resolve_all_session_stage_paths(
    session_map: Dict[str, list],  # Dict[str, List[KinectConfig]]
) -> Dict[str, ForearmStagePaths]:
    """Return a ForearmStagePaths for every session in *session_map*.

    Uses the first KinectConfig in each session group; all configs in a
    session share the same session-level output directories.
    """
    result: Dict[str, ForearmStagePaths] = {}
    for session_id, configs in session_map.items():
        if not configs:
            continue
        cfg = configs[0]

        # Raw: unified registered PLY, or any *.ply fallback
        raw_ply: Optional[Path] = None
        forearm_dir = cfg.session_processed_output_dir / "forearm_pointclouds"
        unified = forearm_dir / f"{session_id}_unified_registered.ply"
        if unified.exists():
            raw_ply = unified
        else:
            plies = sorted(forearm_dir.glob("*.ply"))
            if plies:
                raw_ply = plies[0]

        # PCA-Calibrated and RF-Centered require session_merged_output_dir
        pca_ply: Optional[Path] = None
        rf_ply: Optional[Path] = None
        if cfg.session_merged_output_dir is not None:
            pca_ply = (
                cfg.session_merged_output_dir
                / "forearm_pca_calibrated"
                / f"{session_id}_forearm.ply"
            )
            rf_ply = (
                cfg.session_merged_output_dir
                / "forearm_rf_centered"
                / f"{session_id}_forearm.ply"
            )

        result[session_id] = ForearmStagePaths(
            session_id=session_id,
            raw_ply=raw_ply,
            pca_calibrated_ply=pca_ply,
            rf_centered_ply=rf_ply,
        )
    return result


# ---------------------------------------------------------------------------
# PLY loading and mesh helpers
# ---------------------------------------------------------------------------


def load_forearm_ply(ply_path: Path) -> Optional[pv.PolyData]:
    """Load a forearm PLY via Open3D; return PyVista PolyData with 'colors' array.

    Vertex colors (RGB uint8) are stored in the 'colors' array for direct
    PyVista rgb rendering.  Falls back to uniform grey (160, 160, 160) when
    the PLY has no color data.  Returns None when the file is missing or empty.
    """
    if not ply_path.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        return None

    points = np.asarray(pcd.points, dtype=np.float32)
    cloud = pv.PolyData(points)

    if pcd.has_colors():
        colors_f = np.asarray(pcd.colors)
        colors_u8 = (np.clip(colors_f, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        colors_u8 = np.full((len(points), 3), 160, dtype=np.uint8)

    cloud["colors"] = colors_u8
    return cloud


def compute_delaunay_mesh(
    cloud: pv.PolyData,
    max_edge_length: Optional[float] = None,
) -> Optional[pv.PolyData]:
    """Build a 2.5D Delaunay mesh from a PyVista point cloud.

    Projects points to XY, triangulates with scipy.spatial.Delaunay, fixes
    winding via trimesh, then converts to PyVista.  Vertex colors from the
    'colors' array are propagated to the mesh.  Returns None on failure.

    Parameters
    ----------
    max_edge_length:
        When set, triangles whose longest edge exceeds this value (in the same
        units as the point cloud — typically mm) are removed before the mesh
        is returned.  None means keep all triangles.
    """
    points = np.asarray(cloud.points)
    if len(points) < 4:
        return None

    try:
        xy = points[:, 0:2]
        tri = Delaunay(xy)
        simplices = tri.simplices

        if max_edge_length is not None and max_edge_length > 0.0:
            p = points
            e01 = np.linalg.norm(p[simplices[:, 0]] - p[simplices[:, 1]], axis=1)
            e12 = np.linalg.norm(p[simplices[:, 1]] - p[simplices[:, 2]], axis=1)
            e02 = np.linalg.norm(p[simplices[:, 0]] - p[simplices[:, 2]], axis=1)
            keep = np.maximum(np.maximum(e01, e12), e02) <= max_edge_length
            simplices = simplices[keep]

        if len(simplices) == 0:
            return None

        mesh = trimesh.Trimesh(vertices=points, faces=simplices, process=False)
        if np.mean(mesh.face_normals[:, 2]) < 0:
            mesh.invert()
        mesh.fix_normals()
    except Exception:
        logger.exception("Delaunay triangulation failed")
        return None

    faces = mesh.faces
    faces_vtk = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh_pv = pv.PolyData(mesh.vertices.astype(np.float32), faces_vtk)

    if "colors" in cloud.point_data:
        mesh_pv["colors"] = cloud["colors"]

    return mesh_pv


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class ForearmStageInspector(QMainWindow):
    """Single-window 3D viewer for forearm pointclouds at each postprocessing stage.

    Accepts a mapping of session_id → ForearmStagePaths (built by
    resolve_all_session_stage_paths()) and presents two dropdowns (session,
    stage) plus a Delaunay mesh checkbox.

    Parameters
    ----------
    stage_index:
        Pre-built path index, one entry per session.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        stage_index: Dict[str, ForearmStagePaths],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Forearm Stage Inspector")

        self._stage_index = stage_index
        self._ply_cache: Dict[Path, Optional[pv.PolyData]] = {}
        self._mesh_cache: Dict[tuple, Optional[pv.PolyData]] = {}  # key: (path, threshold)
        self._session_cameras: Dict[str, dict] = {}
        self._initial_render_done = False
        self._has_rendered_cloud = False  # True only after first successful cloud render
        self._rebuilding = False  # re-entrancy guard for signal handlers

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        sessions = sorted(self._stage_index.keys())
        self._session_combo.addItems(sessions)
        bar.addWidget(self._session_combo)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        self._stage_combo.addItems(STAGES)
        bar.addWidget(self._stage_combo)

        bar.addSpacing(12)
        self._delaunay_checkbox = QCheckBox("Delaunay mesh")
        bar.addWidget(self._delaunay_checkbox)

        bar.addSpacing(8)
        bar.addWidget(QLabel("Max edge:"))
        self._threshold_spinbox = QDoubleSpinBox()
        self._threshold_spinbox.setRange(0.0, 9999.0)
        self._threshold_spinbox.setValue(0.0)
        self._threshold_spinbox.setSingleStep(1.0)
        self._threshold_spinbox.setDecimals(1)
        self._threshold_spinbox.setSuffix(" mm")
        self._threshold_spinbox.setSpecialValueText("no limit")
        self._threshold_spinbox.setFixedWidth(90)
        bar.addWidget(self._threshold_spinbox)

        bar.addStretch()
        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._on_recenter)
        bar.addWidget(recenter_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bar.addWidget(close_btn)
        root.addLayout(bar)

        self.plotter = QtInteractor(central)
        root.addWidget(self.plotter.interactor, stretch=1)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignLeft)
        root.addWidget(self._status_label)

        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        self._delaunay_checkbox.stateChanged.connect(self._on_delaunay_toggled)
        self._threshold_spinbox.valueChanged.connect(self._on_threshold_changed)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_session_changed(self, _index: int) -> None:
        self._rebuild_scene()

    def _on_stage_changed(self, _index: int) -> None:
        self._rebuild_scene()

    def _on_delaunay_toggled(self, _state: int) -> None:
        self._rebuild_scene()

    def _on_threshold_changed(self, _value: float) -> None:
        self._mesh_cache.clear()
        self._rebuild_scene()

    def _on_recenter(self) -> None:
        session_id = self._session_combo.currentText()
        self._session_cameras.pop(session_id, None)
        self.plotter.reset_camera()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _rebuild_scene(self) -> None:
        if self._rebuilding:
            return
        self._rebuilding = True
        try:
            self._do_rebuild_scene()
        finally:
            self._rebuilding = False

    def _do_rebuild_scene(self) -> None:
        session_id = self._session_combo.currentText()
        stage_idx = self._stage_combo.currentIndex()

        # Save camera only after a cloud has been successfully rendered at least once,
        # so we never store the default zeroed-out camera and restore it as if valid.
        if self._has_rendered_cloud and session_id:
            cam = self.plotter.camera
            self._session_cameras[session_id] = {
                "position": list(cam.position),
                "focal_point": list(cam.focal_point),
                "up": list(cam.up),
                "view_angle": float(cam.view_angle),
            }

        self.plotter.clear()
        self.plotter.set_background("black")

        if not session_id or session_id not in self._stage_index:
            self._status_label.setText("No session selected.")
            self.plotter.render()
            return

        ply_path = self._get_ply_path(session_id, stage_idx)

        if ply_path is None or not ply_path.exists():
            self.plotter.add_text(
                "No PLY available for this stage",
                position="upper_left",
                font_size=14,
                color="white",
                name="no_ply",
            )
            self._status_label.setText("No PLY found for this stage.")
            self.plotter.render()
            return

        cloud = self._load_or_cache_ply(ply_path)
        if cloud is None:
            self.plotter.add_text(
                "Failed to load PLY",
                position="upper_left",
                font_size=14,
                color="white",
                name="load_error",
            )
            self._status_label.setText(f"Failed to load: {ply_path}")
            self.plotter.render()
            return

        if self._delaunay_checkbox.isChecked():
            mesh_pv = self._compute_or_cache_delaunay(ply_path, cloud)
            if mesh_pv is not None:
                self.plotter.add_mesh(
                    mesh_pv,
                    scalars="colors",
                    rgb=True,
                    opacity=1.0,
                    name="delaunay_mesh",
                )
        else:
            self.plotter.add_points(
                cloud,
                scalars="colors",
                rgb=True,
                point_size=4,
                render_points_as_spheres=False,
                name="forearm",
            )
        self._has_rendered_cloud = True

        # Restore saved camera or reset to fit
        saved = self._session_cameras.get(session_id)
        if saved is not None:
            self.plotter.camera.position = saved["position"]
            self.plotter.camera.focal_point = saved["focal_point"]
            self.plotter.camera.up = saved["up"]
            self.plotter.camera.view_angle = saved["view_angle"]
            self.plotter.renderer.ResetCameraClippingRange()
        else:
            self.plotter.reset_camera()

        self._status_label.setText(str(ply_path))
        self.plotter.render()

    def _get_ply_path(self, session_id: str, stage_idx: int) -> Optional[Path]:
        paths = self._stage_index[session_id]
        if stage_idx == _STAGE_RAW:
            return paths.raw_ply
        if stage_idx == _STAGE_PCA:
            return paths.pca_calibrated_ply
        if stage_idx == _STAGE_RF:
            return paths.rf_centered_ply
        raise ValueError(f"Unknown stage index: {stage_idx}")

    def _load_or_cache_ply(self, ply_path: Path) -> Optional[pv.PolyData]:
        if ply_path not in self._ply_cache:
            self._ply_cache[ply_path] = load_forearm_ply(ply_path)
        return self._ply_cache[ply_path]

    def _compute_or_cache_delaunay(
        self, ply_path: Path, cloud: pv.PolyData
    ) -> Optional[pv.PolyData]:
        threshold = self._threshold_spinbox.value()
        max_edge = threshold if threshold > 0.0 else None
        key = (ply_path, threshold)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = compute_delaunay_mesh(cloud, max_edge_length=max_edge)
        return self._mesh_cache[key]

    # ------------------------------------------------------------------
    # Qt lifecycle
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
        self._rebuild_scene()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
