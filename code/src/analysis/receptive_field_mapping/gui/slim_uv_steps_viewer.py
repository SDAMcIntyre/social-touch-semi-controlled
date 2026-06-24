"""Interactive step-by-step viewer for the forearm SLIM UV pipeline.

Presents each cleaning sub-step + UV initialisation, SLIM optimisation, and
distortion as navigable entries in a PyQt5 + PyVista viewer.  View-only;
the meshes are captured upstream during a normal precompute run and passed
in via the ``steps`` argument.

Used by ``precompute_forearm_slim_uv(..., interactive=True)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

logger = logging.getLogger(__name__)


@dataclass
class SlimStep:
    """One inspectable step in the SLIM UV pipeline.

    V/F define the mesh to render.  For UV-space steps (init, final, distortion)
    the caller embeds the 2D UV as 3D vertices with z=0.

    Optional fields:
    - overlay_points: dict[label, (N, 3) array] of extra points to render as spheres
    - overlay_colors: dict[label, str] matching overlay_points keys
    - face_scalars: per-face scalar (M,) for distortion colouring
    - vertex_colors: per-vertex RGBA (N, 4) in [0, 1] for skin-colour rendering
    - cmap: matplotlib cmap name when face_scalars is set
    - clim: (low, high) colour limits when face_scalars is set
    - scalar_bar_title: label for the scalar bar
    """

    label: str
    info: str
    V: np.ndarray
    F: np.ndarray
    overlay_points: dict[str, np.ndarray] = field(default_factory=dict)
    overlay_colors: dict[str, str] = field(default_factory=dict)
    face_scalars: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None
    cmap: Optional[str] = None
    clim: Optional[tuple[float, float]] = None
    scalar_bar_title: Optional[str] = None


def _faces_to_vtk(F: np.ndarray) -> np.ndarray:
    """Convert (M, 3) face array to VTK flat format [3, v0, v1, v2, ...]."""
    return np.hstack(
        [np.full((len(F), 1), 3, dtype=np.int64), F.astype(np.int64)]
    ).ravel()


class SlimUvStepsViewer(QMainWindow):
    """Per-session step-by-step viewer for the SLIM UV pipeline."""

    def __init__(
        self,
        session_id: str,
        steps: list[SlimStep],
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not steps:
            raise ValueError("SlimUvStepsViewer: steps list is empty.")
        self._session_id = session_id
        self._steps = steps
        self._current_idx = 0
        self._initialized = False
        self.setWindowTitle(f"SLIM UV steps — {session_id}")

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel(f"Session: {session_id}    "))
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._step_counter = QLabel("")
        toolbar.addWidget(self._prev_btn)
        toolbar.addWidget(self._next_btn)
        toolbar.addSeparator()
        toolbar.addWidget(self._step_counter)
        self.addToolBar(toolbar)

        # Sidebar: step list
        self._step_list = QListWidget()
        self._step_list.setMinimumWidth(260)
        self._step_list.setMaximumWidth(320)
        for i, step in enumerate(steps):
            self._step_list.addItem(f"{i + 1:2d}. {step.label}")

        # Centre: PyVista plotter
        self._plotter = QtInteractor(self)
        self._plotter.set_background("white")

        # Central layout
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._step_list)
        layout.addWidget(self._plotter.interactor, stretch=1)
        self.setCentralWidget(central)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Signals
        self._prev_btn.clicked.connect(self._on_prev_clicked)
        self._next_btn.clicked.connect(self._on_next_clicked)
        self._step_list.currentRowChanged.connect(self._on_row_changed)

    # ------------------------------------------------------------------
    # Qt lifecycle (mirrors RFCameraSettingsViewer pattern)
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
        self._step_list.setCurrentRow(0)  # triggers _on_row_changed → render

    def closeEvent(self, event):
        self._plotter.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _capture_camera(self) -> dict:
        cam = self._plotter.camera
        return {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up": list(cam.up),
            "view_angle": float(cam.view_angle),
        }

    def _restore_camera(self, params: dict) -> None:
        self._plotter.camera.position = params["position"]
        self._plotter.camera.focal_point = params["focal_point"]
        self._plotter.camera.up = params["up"]
        self._plotter.camera.view_angle = params["view_angle"]
        self._plotter.renderer.ResetCameraClippingRange()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_prev_clicked(self):
        new_idx = max(0, self._current_idx - 1)
        self._step_list.setCurrentRow(new_idx)

    def _on_next_clicked(self):
        new_idx = min(len(self._steps) - 1, self._current_idx + 1)
        self._step_list.setCurrentRow(new_idx)

    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self._steps):
            return
        prev_idx = self._current_idx
        prev_step = self._steps[prev_idx] if prev_idx != row else None
        next_step = self._steps[row]
        # Preserve camera only when staying in the same coordinate space
        # (3D mesh vs 2D UV).  Crossing the boundary needs a fresh camera.
        same_space = (
            prev_step is not None
            and prev_step.V.shape[1] == next_step.V.shape[1]
        )
        cam_params = self._capture_camera() if same_space else None
        self._current_idx = row
        self._render_step(row)
        if cam_params is not None:
            self._restore_camera(cam_params)
        else:
            if next_step.V.shape[1] == 2:
                self._plotter.view_xy()
            self._plotter.reset_camera()
        self._plotter.render()
        self._prev_btn.setEnabled(row > 0)
        self._next_btn.setEnabled(row < len(self._steps) - 1)
        self._step_counter.setText(f"Step {row + 1} of {len(self._steps)}")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_step(self, idx: int) -> None:
        step = self._steps[idx]
        self._plotter.clear()

        V = step.V
        F = step.F
        if V.shape[1] == 2:
            V_3d = np.column_stack([V, np.zeros(len(V), dtype=V.dtype)])
        else:
            V_3d = V

        if len(F) == 0:
            if step.vertex_colors is not None:
                rgb = (np.asarray(step.vertex_colors)[:, :3] * 255).astype(np.uint8)
                pcd_data = pv.PolyData(V_3d)
                pcd_data.point_data["rgb"] = rgb
                self._plotter.add_mesh(
                    pcd_data, scalars="rgb", rgb=True,
                    point_size=3, render_points_as_spheres=False, name="forearm",
                )
            else:
                self._plotter.add_points(
                    V_3d, color="#888888", point_size=3,
                    render_points_as_spheres=False, name="forearm",
                )
            self._status.showMessage(f"{step.label} — {step.info}")
            return

        mesh = pv.PolyData(V_3d, _faces_to_vtk(F))

        if step.face_scalars is not None:
            mesh.cell_data["scalar"] = np.asarray(step.face_scalars, dtype=np.float64)
            self._plotter.add_mesh(
                mesh,
                scalars="scalar",
                cmap=step.cmap or "viridis",
                clim=step.clim,
                show_scalar_bar=True,
                scalar_bar_args={"title": step.scalar_bar_title or ""},
                show_edges=False,
                name="forearm",
            )
        elif step.vertex_colors is not None:
            rgb = (np.asarray(step.vertex_colors)[:, :3] * 255).astype(np.uint8)
            mesh.point_data["rgb"] = rgb
            self._plotter.add_mesh(
                mesh,
                scalars="rgb",
                rgb=True,
                show_edges=True,
                edge_color="#404040",
                line_width=0.3,
                name="forearm",
            )
        else:
            self._plotter.add_mesh(
                mesh,
                color="#d8c8b8",
                show_edges=True,
                edge_color="#404040",
                line_width=0.3,
                name="forearm",
            )

        # Overlay points (centroid / boundary vertices).
        for name, pts in step.overlay_points.items():
            if pts is None or len(pts) == 0:
                continue
            pts_arr = np.asarray(pts, dtype=np.float64)
            if pts_arr.ndim == 1:
                pts_arr = pts_arr[np.newaxis, :]
            if pts_arr.shape[1] == 2:
                pts_arr = np.column_stack(
                    [pts_arr, np.zeros(len(pts_arr), dtype=pts_arr.dtype)]
                )
            color = step.overlay_colors.get(name, "red")
            point_size = 18 if name == "centroid" else 6
            self._plotter.add_points(
                pts_arr,
                color=color,
                point_size=point_size,
                render_points_as_spheres=True,
                name=f"overlay_{name}",
            )

        self._status.showMessage(f"{step.label} — {step.info}")
