"""
postprocessing_stage_viewer.py
-------------------------------
Single-window PyQt5+PyVista viewer for postprocessed data with a stage
dropdown that switches between all 5 postprocessing coordinate stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import open3d as o3d
import pandas as pd
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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

from merging.gui.neural_kinect_scene_viewer import NeuralDataPanel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_LABELS: List[str] = [
    "Merged (Raw)",
    "ICP Registered",
    "Deduplicated",
    "Contact Projected",
    "PCA Calibrated",
    "RF Centered",
]

_CAMERA_FRAME_STAGES = {0, 1, 2, 3}
_PCA_FRAME_STAGES = {4, 5}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StagePaths:
    """Paths and metadata for one postprocessing stage."""

    stage_label: str
    csv_path: Optional[Path]
    forearm: Optional[Union[Path, "o3d.geometry.PointCloud"]]
    coordinate_frame: str  # "camera" or "pca"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_contact_points_cell(cell) -> Optional[np.ndarray]:
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if s in ("", "[]", "nan"):
        return None
    rows = re.findall(r"\[\s*([-\d\s.,eE+]+)\s*\]", s)
    points: List[List[float]] = []
    for row in rows:
        parts = row.replace(",", " ").split()
        if len(parts) == 3:
            try:
                points.append([float(p) for p in parts])
            except ValueError:
                pass
    return np.array(points, dtype=np.float64) if points else None


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class PostprocessingStageViewer(QMainWindow):
    """Single-window viewer with a stage dropdown for the 5 postprocessing stages.

    Parameters
    ----------
    stage_paths:
        One ``StagePaths`` per stage (length must equal ``len(STAGE_LABELS)``).
    recording_name:
        Displayed in the 3D scene text label and window title.
    initial_stage:
        Index of the stage to display on launch (default: 0).
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        stage_paths: List[StagePaths],
        recording_name: str,
        initial_stage: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        if len(stage_paths) != len(STAGE_LABELS):
            raise ValueError(
                f"stage_paths must have {len(STAGE_LABELS)} entries, "
                f"got {len(stage_paths)}"
            )
        if not (0 <= initial_stage < len(stage_paths)):
            raise ValueError(
                f"initial_stage {initial_stage} out of range [0, {len(stage_paths)})"
            )

        pv.global_theme.allow_empty_mesh = True

        self._stage_paths = stage_paths
        self._recording_name = recording_name
        self._current_stage_idx: int = initial_stage
        self.current_index: int = 0
        self._initial_render_done = False
        self._bounds_proxy_active = False

        self.setWindowTitle(f"Postprocessing Stage Viewer | {recording_name}")

        self._load_stage_data(initial_stage)
        self._build_ui()
        self._init_actors()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(80)
        self._pending_drag_frame: Optional[int] = None
        self._slider_dragging = False
        self._stage_switch_generation: int = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_stage_data(self, stage_idx: int) -> None:
        sp = self._stage_paths[stage_idx]

        # --- CSV ---
        if sp.csv_path is None or not sp.csv_path.exists():
            self._full_df = pd.DataFrame()
            self._kinect_df = pd.DataFrame()
            self._contact_pts_by_frame: List[Optional[np.ndarray]] = []
            self._total_frames = 0
            self._forearm_pv = None
            self._contact_centroid = np.zeros(3, dtype=np.float64)
            return

        full_df = pd.read_csv(sp.csv_path)
        self._full_df = full_df
        self._kinect_df = full_df.dropna(subset=["time_kinect"]).reset_index(drop=True)
        self._total_frames = len(self._kinect_df)

        if "contact_points" in self._kinect_df.columns:
            self._contact_pts_by_frame = [
                _parse_contact_points_cell(c) for c in self._kinect_df["contact_points"]
            ]
        else:
            self._contact_pts_by_frame = []

        # --- Forearm ---
        self._forearm_pv = self._load_forearm(sp.forearm)

        # --- Contact centroid ---
        self._contact_centroid = self._compute_contact_centroid()

    def _load_forearm(
        self, forearm: Optional[Union[Path, "o3d.geometry.PointCloud"]]
    ) -> Optional[pv.PolyData]:
        if forearm is None:
            return None

        if isinstance(forearm, Path):
            if not forearm.exists():
                return None
            o3d_pc = o3d.io.read_point_cloud(str(forearm))
        elif isinstance(forearm, o3d.geometry.PointCloud):
            o3d_pc = forearm
        else:
            raise ValueError(
                f"forearm must be None, Path, or o3d.geometry.PointCloud, got {type(forearm)}"
            )

        if not o3d_pc.has_points():
            return None

        pts = np.asarray(o3d_pc.points, dtype=np.float32)
        mesh = pv.PolyData(pts)
        if o3d_pc.has_colors():
            mesh["colors"] = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
        else:
            mesh["colors"] = np.full((len(pts), 3), 160, dtype=np.uint8)
        return mesh

    def _compute_contact_centroid(self) -> np.ndarray:
        if self._contact_pts_by_frame:
            valid = [
                pts
                for pts in self._contact_pts_by_frame
                if pts is not None and len(pts) > 0
            ]
            if valid:
                return np.nanmean(np.vstack(valid), axis=0)
        return np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        self._outer_layout = QVBoxLayout(central)

        # --- Top bar: stage selector ---
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(4, 4, 4, 4)
        top_bar_layout.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        self._stage_combo.addItems([sp.stage_label for sp in self._stage_paths])
        self._stage_combo.setCurrentIndex(self._current_stage_idx)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        top_bar_layout.addWidget(self._stage_combo)
        top_bar_layout.addStretch()
        self._outer_layout.addWidget(top_bar)

        # --- 3D area + right panel ---
        mid_widget = QWidget()
        mid_layout = QHBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)

        plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(plotter_widget)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(plotter_widget)
        self.plotter.set_background("black")
        plotter_layout.addWidget(self.plotter.interactor)
        mid_layout.addWidget(plotter_widget, stretch=4)

        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout(self._right_panel)
        self._visibility: dict = {}
        self._point_sizes: dict = {"forearm": 5.0, "contact_points": 15.0}
        self._build_right_panel_controls()
        scroll = QScrollArea()
        scroll.setWidget(self._right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)
        mid_layout.addWidget(scroll, stretch=1)

        self._outer_layout.addWidget(mid_widget, stretch=4)

        # --- Frame controls ---
        self._outer_layout.addWidget(self._build_frame_controls())

        # --- Neural panel ---
        self._neural_panel: Optional[NeuralDataPanel] = None
        self._neural_scale: float = 1.0
        self._maybe_create_neural_panel()

    def _build_right_panel_controls(self) -> None:
        while self._right_panel_layout.count():
            item = self._right_panel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._visibility = {}
        self._point_sizes = {"forearm": 5.0, "contact_points": 15.0}

        def _add_group(label: str, key: str, has_slider: bool, default_size: float) -> None:
            self._visibility[key] = True
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)
            cb = QCheckBox("Visible")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, k=key: self._on_visibility_changed(k, state))
            box_layout.addWidget(cb)
            if has_slider:
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(QLabel("Pt size"))
                sl = QSlider(Qt.Horizontal)
                sl.setMinimum(1)
                sl.setMaximum(20)
                sl.setValue(int(self._point_sizes.get(key, default_size)))
                sl.valueChanged.connect(lambda val, k=key: self._on_point_size_changed(k, val))
                rl.addWidget(sl)
                box_layout.addWidget(row)
            self._right_panel_layout.addWidget(box)

        _add_group("Forearm", "forearm", has_slider=True, default_size=5.0)
        if self._contact_pts_by_frame:
            _add_group("Contact Points", "contact_points", has_slider=True, default_size=15.0)
        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self._total_frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(self._total_frames > 0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.frame_slider)

        label_text = f"1 / {self._total_frames}" if self._total_frames > 0 else "0 / 0"
        self.frame_label = QLabel(label_text)
        self.frame_label.setFixedWidth(100)
        layout.addWidget(self.frame_label)

        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._recenter_view)
        layout.addWidget(recenter_btn)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)
        return widget

    def _maybe_create_neural_panel(self) -> None:
        has_neural = (
            len(self._full_df) > 0 and "Nerve_freq" in self._full_df.columns
        )
        if has_neural:
            self._neural_panel = NeuralDataPanel(self._full_df, self._total_frames)
            self._neural_panel.frame_requested.connect(self._on_neural_frame_requested)
            self._outer_layout.addWidget(self._neural_panel)
            self._neural_scale = (
                len(self._full_df) / self._total_frames if self._total_frames > 0 else 1.0
            )
        else:
            self._neural_panel = None
            self._neural_scale = 1.0

    # ------------------------------------------------------------------
    # VTK actors
    # ------------------------------------------------------------------

    def _init_actors(self) -> None:
        _seed = np.zeros((1, 3), dtype=np.float32)
        _seed_col = np.full((1, 3), 128, dtype=np.uint8)

        if self._forearm_pv is not None and self._visibility.get("forearm", True):
            self._actor_forearm = self.plotter.add_mesh(
                self._forearm_pv,
                scalars="colors",
                rgb=True,
                name="forearm",
                render_points_as_spheres=True,
                point_size=self._point_sizes["forearm"],
            )
        else:
            _empty_forearm = pv.PolyData(_seed.copy())
            _empty_forearm["colors"] = _seed_col.copy()
            self._actor_forearm = self.plotter.add_mesh(
                _empty_forearm,
                scalars="colors",
                rgb=True,
                name="forearm",
                render_points_as_spheres=True,
                point_size=self._point_sizes["forearm"],
            )

        self._mesh_contact = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._actor_contact = self.plotter.add_mesh(
            self._mesh_contact,
            name="contact_points",
            color="red",
            render_points_as_spheres=True,
            point_size=self._point_sizes["contact_points"],
        )

        if self._total_frames == 0:
            self.plotter.add_text(
                "No data available",
                position="upper_left",
                font_size=14,
                color="white",
                name="no_data_label",
            )
        else:
            self.plotter.add_text(
                self._recording_name,
                position="upper_left",
                font_size=10,
                color="white",
                name="recording_label",
            )

        self.plotter.add_axes(interactive=False, line_width=150, box=True)

        cx, cy, cz = self._contact_centroid.tolist()
        _cam_pos, _cam_focal, _cam_up = self._compute_camera_params()
        self.plotter.camera.focal_point = _cam_focal
        self.plotter.camera.position = _cam_pos
        self.plotter.camera.up = _cam_up
        self.plotter.camera_set = True

        _box_half = 400.0
        _bounds_proxy = pv.Box(
            bounds=(
                cx - _box_half, cx + _box_half,
                cy - _box_half, cy + _box_half,
                cz - _box_half, cz + _box_half,
            )
        )
        self._bounds_proxy_active = True
        self.plotter.add_mesh(_bounds_proxy, opacity=0.001, name="_bounds_proxy", pickable=False)

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> None:
        self.current_index = frame_idx

        if self._bounds_proxy_active and (
            self._forearm_pv is not None
            or (
                self._contact_pts_by_frame
                and any(
                    p is not None
                    for p in self._contact_pts_by_frame[: min(frame_idx + 1, 10)]
                )
            )
        ):
            self._bounds_proxy_active = False
            try:
                self.plotter.remove_actor("_bounds_proxy")
            except Exception:
                pass

        if self._contact_pts_by_frame:
            _cpts = None
            if self._visibility.get("contact_points", True):
                _cpts = (
                    self._contact_pts_by_frame[frame_idx]
                    if frame_idx < len(self._contact_pts_by_frame)
                    else None
                )
            if _cpts is not None and len(_cpts) > 0:
                self._mesh_contact.DeepCopy(pv.PolyData(_cpts.astype(np.float32)))
            else:
                self._mesh_contact.DeepCopy(pv.PolyData(np.empty((0, 3), dtype=np.float32)))

        if hasattr(self, "_actor_forearm") and self._actor_forearm is not None:
            if self._visibility.get("forearm", True):
                self._actor_forearm.VisibilityOn()
            else:
                self._actor_forearm.VisibilityOff()

        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

        if self._neural_panel is not None:
            self._neural_panel.update_cursor(int(frame_idx * self._neural_scale))

        if self._total_frames > 0:
            self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")
        else:
            self.frame_label.setText("0 / 0")

    # ------------------------------------------------------------------
    # Stage switching
    # ------------------------------------------------------------------

    def _on_stage_changed(self, index: int) -> None:
        if index == self._current_stage_idx:
            return

        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")

        saved_frame = self.current_index
        old_frame = self._stage_paths[self._current_stage_idx].coordinate_frame
        new_frame = self._stage_paths[index].coordinate_frame
        frame_changed = old_frame != new_frame

        self._current_stage_idx = index
        self._load_stage_data(index)

        self.plotter.clear()
        self.plotter.set_background("black")
        self._build_right_panel_controls()

        # Defer actor initialization and rendering to the next event loop tick
        # so the OpenGL context from clear() is fully released first.
        # Without this, wglMakeCurrent fails on Windows.
        self._stage_switch_generation += 1
        current_gen = self._stage_switch_generation

        def _deferred_stage_init():
            if self._stage_switch_generation != current_gen:
                return
            self._init_actors()

            if frame_changed:
                pos, focal, up = self._compute_camera_params()
                self.plotter.camera.position = pos
                self.plotter.camera.focal_point = focal
                self.plotter.camera.up = up
                self.plotter.camera_set = True

            # Recreate NeuralDataPanel
            old_panel = self._neural_panel
            if old_panel is not None:
                self._outer_layout.removeWidget(old_panel)
                old_panel.deleteLater()
                self._neural_panel = None
            self._maybe_create_neural_panel()

            # Update frame controls
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(max(self._total_frames - 1, 0))
            self.frame_slider.setEnabled(self._total_frames > 0)
            restored_frame = min(saved_frame, max(self._total_frames - 1, 0))
            self.frame_slider.setValue(restored_frame)
            self.frame_slider.blockSignals(False)

            self.current_index = restored_frame
            self._update_frame(restored_frame)

        QTimer.singleShot(0, _deferred_stage_init)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _compute_camera_params(self):
        cx, cy, cz = self._contact_centroid.tolist()
        position = [cx, cy, cz - 400.0]
        up = [0.375, -0.904, -0.201]
        return position, [cx, cy, cz], up

    def _recenter_view(self) -> None:
        position, focal_point, up = self._compute_camera_params()
        self.plotter.camera.position = position
        self.plotter.camera.focal_point = focal_point
        self.plotter.camera.up = up
        self.plotter.render()

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
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
        if self._total_frames > 0:
            self._update_frame(0)
        else:
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        self._drag_timer.stop()
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Slider handlers
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._pending_drag_frame = None
        self._drag_timer.start()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._drag_timer.stop()
        self._pending_drag_frame = None
        self._update_frame(self.frame_slider.value())

    def _on_drag_timer_fired(self) -> None:
        if self._pending_drag_frame is not None:
            frame = self._pending_drag_frame
            self._pending_drag_frame = None
            self._update_frame(frame)

    def _on_slider_change(self, value: int) -> None:
        if self._slider_dragging:
            if self._total_frames > 0:
                self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._pending_drag_frame = value
        else:
            self._update_frame(value)

    # ------------------------------------------------------------------
    # Visibility and point-size handlers
    # ------------------------------------------------------------------

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visibility[key] = state == Qt.Checked
        self._update_frame(self.current_index)

    def _on_point_size_changed(self, key: str, value: int) -> None:
        self._point_sizes[key] = float(value)
        actor_map = {
            "forearm": getattr(self, "_actor_forearm", None),
            "contact_points": getattr(self, "_actor_contact", None),
        }
        actor = actor_map.get(key)
        if actor is not None:
            actor.GetProperty().SetPointSize(float(value))
            self.plotter.render()
        else:
            self._update_frame(self.current_index)

    def _on_neural_frame_requested(self, frame: int) -> None:
        if 0 <= frame < self._total_frames:
            self.frame_slider.setValue(frame)

    # ------------------------------------------------------------------
    # Play/pause
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            self._play_timer.start(33)
            self.play_button.setText("⏸ Pause")

    def _play_advance(self) -> None:
        if self._total_frames == 0:
            return
        next_frame = self.current_index + 1
        if next_frame >= self._total_frames:
            next_frame = 0
        self.frame_slider.setValue(next_frame)
