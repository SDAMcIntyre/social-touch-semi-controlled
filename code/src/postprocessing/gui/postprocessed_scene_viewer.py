"""
postprocessed_scene_viewer.py
-----------------------------
Standalone viewer for postprocessed (PCA-calibrated) data.

Shows a PCA-calibrated forearm surface with projected contact points — without
raw Kinect MKV data.  All spatial data is already in the unified PCA-calibrated
coordinate frame.

Two modes:
  simple   — Forearm PLY (static) + contact points (per frame) + NeuralDataPanel
  advanced — Adds sticker spheres (from CSV) + hand mesh (ICP+PCA transformed
             at render time)

Design invariants (same as NeuralKinectViewer):
- NEVER calls plotter.clear() or plotter.add_mesh() inside _update_frame().
  All dynamic actors are registered once in _init_actors() and updated via:
    • PolyData.DeepCopy()  — contact points (in-place dataset update)
    • mesh.points = verts  — hand mesh when triangle count is unchanged
    • actor.SetPosition()  — sticker spheres (geometry stays at origin)
  plotter.render() propagates all VTK Modified() flags automatically.
- ResetCameraClippingRange() is called before every plotter.render() to prevent
  VTK from silently clipping geometry after in-place dataset updates.
- Deferred initial render via showEvent + QTimer.singleShot(0, ...) so that
  the VTK render window has valid pixel dimensions on first Render().

Transform chain for the hand mesh (advanced mode only):
  1. Load per-frame vertices from HandMotionManager[frame_idx]  (Kinect camera space)
  2. Apply the active ICP 4×4 rigid transform (from icp_schedule, bisect lookup)
  3. Apply PCA calibration via PCACalibrationEngine.apply_full_transform()
  Result: hand mesh vertices in the same PCA-calibrated space as stickers /
  contact points.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import bisect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import open3d as o3d
import pandas as pd
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from preprocessing.forearm_extraction import apply_rigid_transform
from preprocessing.motion_analysis import HandMotionManager

from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import (
    CalibrationResult,
    PCACalibrationEngine,
)

# NeuralDataPanel lives in merging.gui; import directly from the module.
# StickerVelocityCompass is also there.
from merging.gui.neural_kinect_scene_viewer import NeuralDataPanel
from merging.gui.sticker_velocity_compass import StickerVelocityCompass


# ---------------------------------------------------------------------------
# Sticker column descriptors
# ---------------------------------------------------------------------------

# Each tuple: (actor_name, x_col, y_col, z_col)
_STICKER_COL_GROUPS: List[Tuple[str, str, str, str]] = [
    ("sticker_blue",   "sticker_blue_position_x",   "sticker_blue_position_y",   "sticker_blue_position_z"),
    ("sticker_green",  "sticker_green_position_x",  "sticker_green_position_y",  "sticker_green_position_z"),
    ("sticker_yellow", "sticker_yellow_position_x", "sticker_yellow_position_y", "sticker_yellow_position_z"),
]

_STICKER_DISPLAY_COLORS: Dict[str, str] = {
    "sticker_blue":   "blue",
    "sticker_green":  "green",
    "sticker_yellow": "yellow",
}


# ---------------------------------------------------------------------------
# Contact-point cell parser (duplicated from neural_kinect_scene_viewer to
# avoid a hard dependency on a private function)
# ---------------------------------------------------------------------------

def _parse_contact_points_cell(cell) -> Optional[np.ndarray]:
    """Parse one CSV cell from the ``contact_points`` column.

    Handles the numpy str() representation: ``"[[ 12.3  45.6  78.9]\\n [...]"``
    or the empty form ``"[]"``.

    Returns an ``(N, 3)`` float64 array, or ``None`` when empty / unparseable.
    """
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if s in ('', '[]', 'nan'):
        return None
    rows = re.findall(r'\[\s*([-\d\s.,eE+]+)\s*\]', s)
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
# PostprocessedSceneViewer
# ---------------------------------------------------------------------------

class PostprocessedSceneViewer(QMainWindow):
    """
    Standalone viewer for PCA-calibrated postprocessed data.

    Parameters
    ----------
    postprocessed_csv_path:
        Path to the contact-projected CSV (``blocks_contact_projected``).
        Must contain ``time_kinect`` and ``contact_points`` columns.
    forearm_ply_path:
        Path to the PCA-calibrated forearm ``.ply`` point cloud (static).
    recording_name:
        Human-readable label shown in the window title and as a 3-D actor.
    mode:
        ``"simple"`` — forearm + contact points + neural panel.
        ``"advanced"`` — adds sticker spheres and hand mesh with ICP+PCA
        transform chain.
    hand_motion_path:
        Path to the ``*_handmodel_motion.npz`` file (required for advanced
        mode; ignored in simple mode).
    icp_schedule:
        Ordered ``[(start_frame, T_4x4), ...]`` list as returned by
        ``get_transform_schedule()``.  ``None`` or ``[]`` → hand mesh is
        rendered without an ICP transform.
    pca_calib:
        ``CalibrationResult`` loaded from ``pca-xyz_transformation-matrices.json``.
        ``None`` → hand mesh is rendered without PCA calibration.
    """

    def __init__(
        self,
        postprocessed_csv_path: Path,
        forearm_ply_path: Path,
        recording_name: str,
        mode: str = "simple",
        hand_motion_path: Optional[Path] = None,
        icp_schedule: Optional[List[Tuple[int, np.ndarray]]] = None,
        pca_calib: Optional[CalibrationResult] = None,
        parent=None,
    ):
        super().__init__(parent)

        pv.global_theme.allow_empty_mesh = True

        # ------------------------------------------------------------------
        # Basic state
        # ------------------------------------------------------------------
        self.setWindowTitle(f"Postprocessed Viewer | {recording_name}")
        self._recording_name: str = recording_name
        self._mode: str = mode
        self.current_index: int = 0

        # ------------------------------------------------------------------
        # 1. Load and partition the postprocessed CSV
        # ------------------------------------------------------------------
        _full_df = pd.read_csv(postprocessed_csv_path)

        # NeuralDataPanel uses the full-rate DataFrame (neural sampling rate)
        self._full_df: pd.DataFrame = _full_df

        # Kinect rows: one row per kinect frame (time_kinect is not NaN)
        self._kinect_df: pd.DataFrame = (
            _full_df.dropna(subset=['time_kinect']).reset_index(drop=True)
        )
        self._total_frames: int = len(self._kinect_df)

        # ------------------------------------------------------------------
        # 2. Pre-parse contact points (PCA-calibrated, one entry per kinect frame)
        # ------------------------------------------------------------------
        self._contact_pts_by_frame: Optional[List[Optional[np.ndarray]]] = None
        if 'contact_points' in self._kinect_df.columns:
            self._contact_pts_by_frame = [
                _parse_contact_points_cell(cell)
                for cell in self._kinect_df['contact_points']
            ]

        # ------------------------------------------------------------------
        # 3. Load forearm PLY (static, rendered once)
        # ------------------------------------------------------------------
        self._forearm_pv: Optional[pv.PolyData] = None
        try:
            o3d_pc = o3d.io.read_point_cloud(str(forearm_ply_path))
            if o3d_pc.has_points():
                pts_fa = np.asarray(o3d_pc.points, dtype=np.float32)
                mesh_fa = pv.PolyData(pts_fa)
                if o3d_pc.has_colors():
                    mesh_fa['colors'] = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
                else:
                    mesh_fa['colors'] = np.full((len(pts_fa), 3), 160, dtype=np.uint8)
                self._forearm_pv = mesh_fa
        except Exception as exc:
            print(f"Warning: Could not load forearm PLY at {forearm_ply_path}: {exc}")

        # ------------------------------------------------------------------
        # 4. Advanced-mode data
        # ------------------------------------------------------------------
        self._icp_schedule: List[Tuple[int, np.ndarray]] = icp_schedule or []
        self._icp_starts: List[int] = [s for s, _ in self._icp_schedule]
        self._pca_calib: Optional[CalibrationResult] = pca_calib

        # Sticker per-frame positions (N_frames × 3) from kinect_df columns
        self._sticker_positions: Dict[str, np.ndarray] = {}
        if mode == "advanced":
            for name, xcol, ycol, zcol in _STICKER_COL_GROUPS:
                if all(c in self._kinect_df.columns for c in (xcol, ycol, zcol)):
                    self._sticker_positions[name] = (
                        self._kinect_df[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)
                    )

        # Hand motion manager
        self._hand_manager: Optional[HandMotionManager] = None
        self._last_hand_frame: int = -1
        self._last_hand_mesh = None
        self._last_hand_tri_count: int = -1
        if mode == "advanced" and hand_motion_path is not None:
            try:
                self._hand_manager = HandMotionManager()
                self._hand_manager.load(str(hand_motion_path))
            except Exception as exc:
                print(f"Warning: Could not load hand motion data ({exc}). Hand mesh disabled.")
                self._hand_manager = None

        # ------------------------------------------------------------------
        # 5. Compute contact centroid for camera positioning
        # ------------------------------------------------------------------
        self._contact_centroid: np.ndarray = self._compute_contact_centroid()

        # ------------------------------------------------------------------
        # 6. Build Qt UI
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # 7. Initialise named VTK actors
        # ------------------------------------------------------------------
        self._init_actors()

        # ------------------------------------------------------------------
        # 8. Play timer
        # ------------------------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        # Drag-throttle timer (80 ms ≈ 12 fps during slider drag)
        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(80)
        self._pending_drag_frame: Optional[int] = None
        self._slider_dragging: bool = False

        # Guard: defer first render to showEvent
        self._initial_render_done: bool = False

    # ------------------------------------------------------------------
    # Contact centroid
    # ------------------------------------------------------------------

    def _compute_contact_centroid(self) -> np.ndarray:
        """Return the mean XYZ of all non-empty contact points, or (0,0,0)."""
        if self._contact_pts_by_frame:
            valid = [pts for pts in self._contact_pts_by_frame if pts is not None and len(pts)]
            if valid:
                return np.nanmean(np.vstack(valid), axis=0)
        if self._sticker_positions:
            all_pos = np.vstack(list(self._sticker_positions.values()))
            finite_rows = all_pos[np.isfinite(all_pos).all(axis=1)]
            if len(finite_rows):
                return np.nanmean(finite_rows, axis=0)
        return np.zeros(3)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # --- Top row: 3D plotter + right scroll panel ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(plotter_widget)
        self.plotter = QtInteractor(plotter_widget)
        self.plotter.set_background('black')
        plotter_layout.addWidget(self.plotter.interactor)
        top_layout.addWidget(plotter_widget, stretch=4)

        # Right panel (scrollable, fixed 220 px)
        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout(self._right_panel)
        self._build_right_panel_controls()
        scroll = QScrollArea()
        scroll.setWidget(self._right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)
        top_layout.addWidget(scroll, stretch=1)

        outer.addWidget(top_widget, stretch=4)

        # --- Middle row: frame controls ---
        outer.addWidget(self._build_frame_controls())

        # --- Bottom row: neural panel (when Nerve_freq present in full CSV) ---
        self._neural_panel: Optional[NeuralDataPanel] = None
        _has_neural = (
            len(self._full_df) > 0
            and 'Nerve_freq' in self._full_df.columns
        )
        if _has_neural:
            self._neural_panel = NeuralDataPanel(self._full_df, self._total_frames)
            outer.addWidget(self._neural_panel)

        # Neural scale: ratio of full-CSV rows to kinect frames
        self._neural_scale: float = (
            len(self._full_df) / self._total_frames
            if self._total_frames > 0 else 1.0
        )

    def _build_right_panel_controls(self) -> None:
        """Populate the scrollable right panel."""
        self._visibility: Dict[str, bool] = {}
        self._point_sizes: Dict[str, float] = {
            'forearm': 5.0,
            'contact_points': 15.0,
        }
        self._compass_widgets: Dict[str, StickerVelocityCompass] = {}

        def _add_group(label: str, key: str, has_slider: bool = False,
                       default_size: float = 3.0):
            self._visibility[key] = True
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)
            cb = QCheckBox("Visible")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, k=key: self._on_visibility_changed(k, state)
            )
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
                sl.valueChanged.connect(
                    lambda val, k=key: self._on_point_size_changed(k, val)
                )
                rl.addWidget(sl)
                box_layout.addWidget(row)
            self._right_panel_layout.addWidget(box)

        _add_group("Forearm",        "forearm",         has_slider=True,  default_size=5.0)
        if self._contact_pts_by_frame is not None:
            _add_group("Contact Points", "contact_points", has_slider=True,  default_size=15.0)
        if self._mode == "advanced":
            if self._hand_manager is not None:
                _add_group("Hand Mesh",      "hand_meshes",     has_slider=False)
            for name in self._sticker_positions:
                color = _STICKER_DISPLAY_COLORS.get(name, 'magenta')
                _add_group(name, name)

                compass = StickerVelocityCompass(name, color)
                self._compass_widgets[name] = compass
                compass_box = QGroupBox(f"{name} velocity")
                compass_layout = QVBoxLayout(compass_box)
                compass_layout.addWidget(compass)
                self._right_panel_layout.addWidget(compass_box)

        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        """Return a QWidget row with slider, labels, and control buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self._total_frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.frame_slider)

        self.frame_label = QLabel(f"1 / {self._total_frames}")
        self.frame_label.setFixedWidth(100)
        layout.addWidget(self.frame_label)

        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._recenter_view)
        layout.addWidget(recenter_btn)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)

        return widget

    # ------------------------------------------------------------------
    # Actor initialisation (called once; actors mutated in-place thereafter)
    # ------------------------------------------------------------------

    def _init_actors(self) -> None:
        """
        Register every named actor exactly once.

        After this call ``_update_frame()`` mutates datasets in-place via
        DeepCopy / mesh.points / SetPosition without touching the camera.
        """
        # Seed PolyData with one dummy point so VTK registers the 'colors'
        # array; overwritten immediately by the first _update_frame().
        _seed = np.zeros((1, 3), dtype=np.float32)
        _seed_col = np.full((1, 3), 128, dtype=np.uint8)

        # Static forearm (registered once; never mutated)
        if self._forearm_pv is not None and self._visibility.get('forearm', True):
            self._actor_forearm = self.plotter.add_mesh(
                self._forearm_pv,
                scalars='colors',
                rgb=True,
                name='forearm',
                render_points_as_spheres=True,
                point_size=self._point_sizes['forearm'],
            )
        else:
            _empty = pv.PolyData(_seed.copy())
            _empty['colors'] = _seed_col.copy()
            self._actor_forearm = self.plotter.add_mesh(
                _empty, scalars='colors', rgb=True, name='forearm',
                render_points_as_spheres=True,
                point_size=self._point_sizes['forearm'],
            )

        # Dynamic contact points
        self._mesh_contact = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._actor_contact = self.plotter.add_mesh(
            self._mesh_contact,
            name='contact_points',
            color='red',
            render_points_as_spheres=True,
            point_size=self._point_sizes['contact_points'],
        )

        # Advanced-mode actors
        self._sticker_actors: Dict[str, Any] = {}
        self._mesh_hand: Optional[pv.PolyData] = None

        if self._mode == "advanced":
            # Sticker spheres (sphere stays at origin; actor translated via SetPosition)
            for name in self._sticker_positions:
                color = _STICKER_DISPLAY_COLORS.get(name, 'magenta')
                sphere = pv.Sphere(radius=4.0, center=(0.0, 0.0, 0.0))
                actor = self.plotter.add_mesh(sphere, color=color, name=f'sticker_{name}')
                self._sticker_actors[name] = actor

            # Hand mesh
            if self._hand_manager is not None:
                self._mesh_hand = pv.PolyData(np.empty((0, 3), dtype=np.float32))
                self.plotter.add_mesh(self._mesh_hand, name='hand_meshes', style='wireframe')

        # Static decorations
        self.plotter.add_text(
            self._recording_name,
            position='upper_left',
            font_size=10,
            color='white',
            name='recording_label',
        )
        self.plotter.add_axes(interactive=False, line_width=150, box=True)
        self.plotter.add_mesh(
            pv.Sphere(radius=2.0),
            color='yellow',
            name='origin_sphere',
            pickable=False,
        )

        # Default camera — centred on the contact centroid
        cx, cy, cz = self._contact_centroid.tolist()
        self.plotter.camera.focal_point = [cx, cy, cz]
        self.plotter.camera.position    = [cx, cy, cz - 400.0]
        self.plotter.camera.up          = [0.375, -0.904, -0.201]
        self.plotter.camera_set = True

        # Invisible bounding proxy so VTK has a plausible clipping range on
        # the first frame before real geometry is rendered.
        _box_half = 400.0
        _bounds_proxy = pv.Box(bounds=(
            cx - _box_half, cx + _box_half,
            cy - _box_half, cy + _box_half,
            cz - _box_half, cz + _box_half,
        ))
        self._bounds_proxy_active = True
        self.plotter.add_mesh(
            _bounds_proxy, opacity=0.001, name='_bounds_proxy', pickable=False,
        )

    # ------------------------------------------------------------------
    # Frame update — the hot path
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> None:
        """
        Update all dynamic actors for *frame_idx* and render exactly once.

        1. Contact points  — DeepCopy into self._mesh_contact
        2. Stickers        — actor.SetPosition() / VisibilityOn/Off()  [advanced]
        3. Hand mesh       — ICP+PCA transform, then mesh.points or DeepCopy [advanced]
        4. Neural panel    — cursor update
        5. Render          — ResetCameraClippingRange() + plotter.render()
        """
        self.current_index = frame_idx

        # Remove bounding proxy once we have real geometry
        if self._bounds_proxy_active and (
            self._forearm_pv is not None
            or (self._contact_pts_by_frame and any(
                p is not None for p in self._contact_pts_by_frame[:min(frame_idx + 1, 10)]
            ))
        ):
            self._bounds_proxy_active = False
            try:
                self.plotter.remove_actor('_bounds_proxy')
            except Exception:
                pass

        # 1. Contact points -----------------------------------------------
        if self._contact_pts_by_frame is not None:
            _cpts: Optional[np.ndarray] = None
            if self._visibility.get('contact_points', True):
                _cpts = (
                    self._contact_pts_by_frame[frame_idx]
                    if frame_idx < len(self._contact_pts_by_frame)
                    else None
                )
            if _cpts is not None and len(_cpts) > 0:
                self._mesh_contact.DeepCopy(
                    pv.PolyData(_cpts.astype(np.float32))
                )
            else:
                self._mesh_contact.DeepCopy(
                    pv.PolyData(np.empty((0, 3), dtype=np.float32))
                )

        # 2. Stickers [advanced] -----------------------------------------
        for name, actor in self._sticker_actors.items():
            positions = self._sticker_positions.get(name)
            pos = (
                positions[frame_idx]
                if positions is not None and frame_idx < len(positions)
                else None
            )
            visible = (
                pos is not None
                and np.isfinite(pos).all()
                and self._visibility.get(name, True)
            )
            if visible:
                actor.SetPosition(*pos.tolist())
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()

            # Compass velocity update
            if name in self._compass_widgets and frame_idx > 0 and positions is not None:
                prev_pos = positions[frame_idx - 1]
                if (
                    np.isfinite(prev_pos).all()
                    and pos is not None
                    and np.isfinite(pos).all()
                ):
                    self._compass_widgets[name].update_velocity(pos - prev_pos)

        # 3. Hand mesh [advanced] ----------------------------------------
        if self._mode == "advanced" and self._mesh_hand is not None:
            if not self._visibility.get('hand_meshes', True):
                if self._last_hand_tri_count != 0:
                    self._mesh_hand.DeepCopy(
                        pv.PolyData(np.empty((0, 3), dtype=np.float32))
                    )
                    self._last_hand_tri_count = 0
            else:
                o3d_mesh = self._get_hand_mesh(frame_idx)
                if o3d_mesh is not None and o3d_mesh.has_triangles():
                    verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
                    verts = self._transform_hand_to_pca(verts, frame_idx)
                    tris = np.asarray(o3d_mesh.triangles)
                    n_tris = len(tris)
                    if n_tris == self._last_hand_tri_count:
                        self._mesh_hand.points = verts.astype(np.float32)
                        self._mesh_hand.Modified()
                    else:
                        faces = np.hstack([
                            np.full((n_tris, 1), 3, dtype=tris.dtype), tris
                        ])
                        self._mesh_hand.DeepCopy(
                            pv.PolyData(verts.astype(np.float32), faces)
                        )
                        self._last_hand_tri_count = n_tris
                else:
                    if self._last_hand_tri_count != 0:
                        self._mesh_hand.DeepCopy(
                            pv.PolyData(np.empty((0, 3), dtype=np.float32))
                        )
                        self._last_hand_tri_count = 0

        # 4. Forearm visibility toggle (static mesh — only toggle actor visibility)
        if hasattr(self, '_actor_forearm') and self._actor_forearm is not None:
            if self._visibility.get('forearm', True):
                self._actor_forearm.VisibilityOn()
            else:
                self._actor_forearm.VisibilityOff()

        # 5. Render -------------------------------------------------------
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

        # 6. Neural panel cursor ----------------------------------------
        if self._neural_panel is not None:
            self._neural_panel.update_cursor(frame_idx, self._neural_scale)

        # 7. Frame label -------------------------------------------------
        self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")

    # ------------------------------------------------------------------
    # Hand mesh helpers
    # ------------------------------------------------------------------

    def _get_hand_mesh(self, frame_idx: int):
        """Return the Open3D TriangleMesh for *frame_idx* (Kinect camera space),
        using a simple one-frame LRU cache."""
        if self._hand_manager is None:
            return None
        if frame_idx == self._last_hand_frame:
            return self._last_hand_mesh
        try:
            mesh = self._hand_manager[frame_idx]
            self._last_hand_frame = frame_idx
            self._last_hand_mesh = mesh
            return mesh
        except Exception:
            return None

    def _transform_hand_to_pca(self, vertices: np.ndarray, frame_idx: int) -> np.ndarray:
        """Apply ICP rigid transform + PCA calibration to *vertices*.

        Step 1: Apply the active ICP 4×4 transform for *frame_idx* (bisect
                lookup on the icp_schedule).
        Step 2: Apply PCA calibration via PCACalibrationEngine.apply_full_transform().

        Returns the transformed vertices as float64.
        """
        coords = vertices.copy()

        # Step 1: ICP transform
        if self._icp_starts:
            i = bisect.bisect_right(self._icp_starts, frame_idx) - 1
            if i >= 0:
                T_icp = self._icp_schedule[i][1]
                coords = apply_rigid_transform(coords, T_icp)

        # Step 2: PCA calibration
        if self._pca_calib is not None:
            coords = PCACalibrationEngine.apply_full_transform(coords, self._pca_calib)

        return coords

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _recenter_view(self) -> None:
        cx, cy, cz = self._contact_centroid.tolist()
        self.plotter.camera.focal_point = [cx, cy, cz]
        self.plotter.camera.position    = [cx, cy, cz - 400.0]
        self.plotter.camera.up          = [0.375, -0.904, -0.201]
        self.plotter.render()

    # ------------------------------------------------------------------
    # showEvent / deferred start
    # ------------------------------------------------------------------

    def _deferred_start(self) -> None:
        """Initialise the VTK interactor geometry, then render frame 0."""
        try:
            self.plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self.plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self.plotter.render_window.SetSize(sz.width(), sz.height())
        self._update_frame(0)

    def showEvent(self, event) -> None:  # noqa: N802
        """Trigger the first render once the window has real geometry."""
        super().showEvent(event)
        if not self._initial_render_done:
            self._initial_render_done = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    # ------------------------------------------------------------------
    # Slot handlers
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
            self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._pending_drag_frame = value
        else:
            self._update_frame(value)

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visibility[key] = state == Qt.Checked
        self._update_frame(self.current_index)

    def _on_point_size_changed(self, key: str, value: int) -> None:
        self._point_sizes[key] = float(value)
        actor_map = {
            'forearm':         getattr(self, '_actor_forearm', None),
            'contact_points':  getattr(self, '_actor_contact', None),
        }
        actor = actor_map.get(key)
        if actor is not None:
            actor.GetProperty().SetPointSize(float(value))
            self.plotter.render()
        else:
            self._update_frame(self.current_index)

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            self._play_timer.start(33)  # ~30 fps
            self.play_button.setText("⏸ Pause")

    def _play_advance(self) -> None:
        next_frame = self.current_index + 1
        if next_frame >= self._total_frames:
            next_frame = 0
        self.frame_slider.setValue(next_frame)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop timers cleanly on close."""
        self._play_timer.stop()
        self._drag_timer.stop()
        super().closeEvent(event)
