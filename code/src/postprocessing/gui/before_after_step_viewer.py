"""
before_after_step_viewer.py
---------------------------
Side-by-side 3D viewer that displays the before and after state of contact
points (and optionally the forearm surface) for a single postprocessing step.

Two synchronized PyVista QtInteractor widgets are placed side by side.
Rotating or zooming in one viewport is mirrored to the other via VTK observer
callbacks with a re-entrancy guard.

Design invariants (same as PostprocessedSceneViewer):
- NEVER calls plotter.clear() or plotter.add_mesh() inside _update_frame().
  All dynamic actors are registered once in _init_actors() and updated via
  PolyData.DeepCopy() — contact points (in-place dataset update).
  plotter.render() propagates all VTK Modified() flags automatically.
- ResetCameraClippingRange() is called before every plotter.render() to
  prevent VTK from silently clipping geometry after in-place dataset updates.
- Deferred initial render via showEvent + QTimer.singleShot(0, ...) so that
  the VTK render window has valid pixel dimensions on first Render().
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
# NeuralDataPanel lives in merging.gui; import directly from the module.
from merging.gui.neural_kinect_scene_viewer import NeuralDataPanel


# ---------------------------------------------------------------------------
# Contact-point cell parser (copied verbatim from postprocessed_scene_viewer
# to avoid a hard dependency on a private function)
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
# BeforeAfterStepViewer
# ---------------------------------------------------------------------------

class BeforeAfterStepViewer(QMainWindow):
    """
    Side-by-side 3D viewer comparing before and after a postprocessing step.

    Parameters
    ----------
    before_csv_path:
        Path to the pre-transform CSV (must contain ``time_kinect`` column).
    after_csv_path:
        Path to the post-transform CSV (must contain ``time_kinect`` column).
    before_forearm:
        Forearm for the before side — either a ``Path`` to a ``.ply`` file or
        an Open3D ``PointCloud`` / ``TriangleMesh`` object already in memory.
        Pass ``None`` to omit the forearm on the before side.
    after_forearm:
        Forearm for the after side — same accepted types as ``before_forearm``.
    step_label:
        Human-readable name of the step (e.g. ``"ICP Registration"``), shown
        in the window title and as a 3-D text actor.
    recording_name:
        Human-readable session/block identifier, also shown in the title.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        before_csv_path: Path,
        after_csv_path: Path,
        before_forearm: Union[Path, object, None] = None,
        after_forearm: Union[Path, object, None] = None,
        step_label: str = "",
        recording_name: str = "",
        parent=None,
    ):
        super().__init__(parent)

        pv.global_theme.allow_empty_mesh = True

        # ------------------------------------------------------------------
        # Basic state
        # ------------------------------------------------------------------
        self._step_label: str = step_label
        self._recording_name: str = recording_name
        self.current_index: int = 0

        title_parts = [p for p in (recording_name, step_label) if p]
        self.setWindowTitle("Before/After Viewer" + (" | " + " — ".join(title_parts) if title_parts else ""))

        # ------------------------------------------------------------------
        # 1. Load both CSVs
        # ------------------------------------------------------------------
        _before_full_df = pd.read_csv(before_csv_path)
        _after_full_df = pd.read_csv(after_csv_path)

        # NeuralDataPanel uses the full-rate before DataFrame
        self._full_df: pd.DataFrame = _before_full_df

        # Kinect rows: one row per kinect frame (time_kinect is not NaN)
        self._before_kinect_df: pd.DataFrame = (
            _before_full_df.dropna(subset=['time_kinect']).reset_index(drop=True)
        )
        self._after_kinect_df: pd.DataFrame = (
            _after_full_df.dropna(subset=['time_kinect']).reset_index(drop=True)
        )

        # Use the shorter of the two to avoid index-out-of-bounds
        self._total_frames: int = min(
            len(self._before_kinect_df),
            len(self._after_kinect_df),
        )

        # ------------------------------------------------------------------
        # 2. Pre-parse contact points for both sides
        # ------------------------------------------------------------------
        self._before_contact_pts: Optional[List[Optional[np.ndarray]]] = None
        if 'contact_points' in self._before_kinect_df.columns:
            self._before_contact_pts = [
                _parse_contact_points_cell(cell)
                for cell in self._before_kinect_df['contact_points']
            ]

        self._after_contact_pts: Optional[List[Optional[np.ndarray]]] = None
        if 'contact_points' in self._after_kinect_df.columns:
            self._after_contact_pts = [
                _parse_contact_points_cell(cell)
                for cell in self._after_kinect_df['contact_points']
            ]

        # ------------------------------------------------------------------
        # 3. Load forearm geometry (optional, static)
        # ------------------------------------------------------------------
        self._before_forearm_pv: Optional[pv.PolyData] = self._load_forearm(
            before_forearm
        )
        self._after_forearm_pv: Optional[pv.PolyData] = self._load_forearm(
            after_forearm
        )

        # ------------------------------------------------------------------
        # 4. Compute contact centroid for camera positioning
        # ------------------------------------------------------------------
        self._contact_centroid: np.ndarray = self._compute_contact_centroid()

        # ------------------------------------------------------------------
        # 5. Camera sync re-entrancy guard
        # ------------------------------------------------------------------
        self._syncing: bool = False

        # ------------------------------------------------------------------
        # 6. Build Qt UI (creates self.plotter_before, self.plotter_after)
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # 7. Initialise named VTK actors on both plotters
        # ------------------------------------------------------------------
        self._init_actors()

        # ------------------------------------------------------------------
        # 8. Set up VTK camera sync observers
        # ------------------------------------------------------------------
        self._setup_camera_sync()

        # ------------------------------------------------------------------
        # 9. Play timer (~30 fps) and drag-throttle timer (80 ms)
        # ------------------------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(80)
        self._pending_drag_frame: Optional[int] = None
        self._slider_dragging: bool = False

        # Guard: defer first render to showEvent
        self._initial_render_done: bool = False

    # ------------------------------------------------------------------
    # Forearm loading helpers
    # ------------------------------------------------------------------

    def _load_forearm(self, forearm) -> Optional[pv.PolyData]:
        """Convert a forearm source to a ``pv.PolyData``, or return ``None``.

        Accepted inputs:
        - ``None``                     → returns None
        - ``pathlib.Path``             → loaded from disk via Open3D
        - Open3D ``PointCloud``        → converted directly
        - Open3D ``TriangleMesh``      → vertices converted directly
        """
        if forearm is None:
            return None
        if isinstance(forearm, Path):
            return self._load_forearm_from_ply(forearm)
        # Treat as Open3D geometry object
        try:
            if isinstance(forearm, o3d.geometry.TriangleMesh):
                pts = np.asarray(forearm.vertices, dtype=np.float32)
                has_colors = forearm.has_vertex_colors()
                colors_src = np.asarray(forearm.vertex_colors) if has_colors else None
            else:
                # PointCloud (or any other geometry with .points)
                pts = np.asarray(forearm.points, dtype=np.float32)
                has_colors = forearm.has_colors()
                colors_src = np.asarray(forearm.colors) if has_colors else None
            if len(pts) == 0:
                return None
            mesh_fa = pv.PolyData(pts)
            if colors_src is not None and len(colors_src) == len(pts):
                mesh_fa['colors'] = (colors_src * 255).astype(np.uint8)
            else:
                mesh_fa['colors'] = np.full((len(pts), 3), 160, dtype=np.uint8)
            return mesh_fa
        except Exception as exc:
            print(f"Warning: Could not convert Open3D forearm geometry: {exc}")
            return None

    def _load_forearm_from_ply(self, ply_path: Path) -> Optional[pv.PolyData]:
        """Load a forearm PLY via Open3D and return a ``pv.PolyData``, or None."""
        try:
            o3d_pc = o3d.io.read_point_cloud(str(ply_path))
            if not o3d_pc.has_points():
                return None
            pts_fa = np.asarray(o3d_pc.points, dtype=np.float32)
            mesh_fa = pv.PolyData(pts_fa)
            if o3d_pc.has_colors():
                mesh_fa['colors'] = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
            else:
                mesh_fa['colors'] = np.full((len(pts_fa), 3), 160, dtype=np.uint8)
            return mesh_fa
        except Exception as exc:
            print(f"Warning: Could not load forearm PLY at {ply_path}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Contact centroid
    # ------------------------------------------------------------------

    def _compute_contact_centroid(self) -> np.ndarray:
        """Return the mean XYZ of all non-empty contact points, or (0,0,0)."""
        all_pts: List[np.ndarray] = []
        for pts_list in (self._before_contact_pts, self._after_contact_pts):
            if pts_list:
                for pts in pts_list:
                    if pts is not None and len(pts):
                        all_pts.append(pts)
        if all_pts:
            return np.nanmean(np.vstack(all_pts), axis=0)
        return np.zeros(3)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # --- Top row: two 3D plotters + right scroll panel ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Left plotter (Before)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("Before")
        left_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_label)
        self.plotter_before = QtInteractor(left_container)
        self.plotter_before.set_background('black')
        left_layout.addWidget(self.plotter_before.interactor)
        top_layout.addWidget(left_container, stretch=4)

        # Right plotter (After)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_label = QLabel("After")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_label)
        self.plotter_after = QtInteractor(right_container)
        self.plotter_after.set_background('black')
        right_layout.addWidget(self.plotter_after.interactor)
        top_layout.addWidget(right_container, stretch=4)

        # Right scroll panel (220 px fixed)
        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout(self._right_panel)
        self._build_right_panel_controls()
        scroll = QScrollArea()
        scroll.setWidget(self._right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)
        top_layout.addWidget(scroll, stretch=0)

        outer.addWidget(top_widget, stretch=4)

        # --- Middle row: frame controls ---
        outer.addWidget(self._build_frame_controls())

        # --- Bottom row: neural panel (when Nerve_freq present in before_df) ---
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
        """Populate the scrollable right panel with visibility and size controls."""
        self._visibility: Dict[str, bool] = {}
        self._point_sizes: Dict[str, float] = {
            'forearm_before':    5.0,
            'forearm_after':     5.0,
            'contact_before':   15.0,
            'contact_after':    15.0,
        }

        def _add_group(label: str, key: str, default_size: float = 3.0):
            self._visibility[key] = True
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)

            cb = QCheckBox("Visible")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, k=key: self._on_visibility_changed(k, state)
            )
            box_layout.addWidget(cb)

            # Point-size slider
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

        _add_group("Forearm Before",    "forearm_before",  default_size=5.0)
        _add_group("Forearm After",     "forearm_after",   default_size=5.0)
        _add_group("Contacts Before",   "contact_before",  default_size=15.0)
        _add_group("Contacts After",    "contact_after",   default_size=15.0)

        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        """Return a QWidget row with slider, label, Recenter, and Play/Pause."""
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
        Register every named actor exactly once on both plotters.

        After this call ``_update_frame()`` mutates datasets in-place via
        DeepCopy without touching the camera.
        """
        _seed = np.zeros((1, 3), dtype=np.float32)
        _seed_col = np.full((1, 3), 128, dtype=np.uint8)

        title_text = " — ".join(p for p in (self._recording_name, self._step_label) if p)

        for plotter, forearm_pv, side_label, forearm_vis_key in (
            (self.plotter_before, self._before_forearm_pv, "Before", "forearm_before"),
            (self.plotter_after,  self._after_forearm_pv,  "After",  "forearm_after"),
        ):
            # Static forearm
            if forearm_pv is not None:
                _actor_fa = plotter.add_mesh(
                    forearm_pv,
                    scalars='colors',
                    rgb=True,
                    name='forearm',
                    render_points_as_spheres=True,
                    point_size=self._point_sizes[forearm_vis_key],
                )
            else:
                _empty_fa = pv.PolyData(_seed.copy())
                _empty_fa['colors'] = _seed_col.copy()
                _actor_fa = plotter.add_mesh(
                    _empty_fa,
                    scalars='colors',
                    rgb=True,
                    name='forearm',
                    render_points_as_spheres=True,
                    point_size=self._point_sizes[forearm_vis_key],
                )

            if side_label == "Before":
                self._actor_forearm_before = _actor_fa
            else:
                self._actor_forearm_after = _actor_fa

            # Text label
            plotter.add_text(
                title_text or side_label,
                position='upper_left',
                font_size=10,
                color='white',
                name='recording_label',
            )
            plotter.add_axes(interactive=False, line_width=150, box=True)

            # Invisible bounding proxy for valid initial clipping range
            cx, cy, cz = self._contact_centroid.tolist()
            _box_half = 400.0
            _bounds_proxy = pv.Box(bounds=(
                cx - _box_half, cx + _box_half,
                cy - _box_half, cy + _box_half,
                cz - _box_half, cz + _box_half,
            ))
            plotter.add_mesh(
                _bounds_proxy,
                opacity=0.001,
                name='_bounds_proxy',
                pickable=False,
            )

        # Dynamic contact PolyData — cyan for before, red for after
        self._mesh_contact_before = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._actor_contact_before = self.plotter_before.add_mesh(
            self._mesh_contact_before,
            name='contact_points',
            color='cyan',
            render_points_as_spheres=True,
            point_size=self._point_sizes['contact_before'],
        )

        self._mesh_contact_after = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._actor_contact_after = self.plotter_after.add_mesh(
            self._mesh_contact_after,
            name='contact_points',
            color='red',
            render_points_as_spheres=True,
            point_size=self._point_sizes['contact_after'],
        )

        # Track whether bounds proxy is still active on each plotter
        self._bounds_proxy_active_before: bool = True
        self._bounds_proxy_active_after: bool = True

        # Default camera on both plotters
        cam_pos, cam_focal, cam_up = self._compute_camera_params()
        for plotter in (self.plotter_before, self.plotter_after):
            plotter.camera.focal_point = cam_focal
            plotter.camera.position    = cam_pos
            plotter.camera.up          = cam_up
            plotter.camera_set = True

    # ------------------------------------------------------------------
    # Camera sync
    # ------------------------------------------------------------------

    def _setup_camera_sync(self) -> None:
        """Register VTK observer callbacks to keep both cameras in sync."""
        _SYNC_EVENTS = (
            'InteractionEvent',
            'EndInteractionEvent',
            'MouseWheelForwardEvent',
            'MouseWheelBackwardEvent',
        )

        def _make_sync_callback(source_plotter, target_plotter):
            def _cb(caller, event):
                if self._syncing:
                    return
                self._syncing = True
                try:
                    cam_src = source_plotter.camera
                    cam_tgt = target_plotter.camera
                    cam_tgt.position    = cam_src.position
                    cam_tgt.focal_point = cam_src.focal_point
                    cam_tgt.up          = cam_src.up
                    target_plotter.renderer.ResetCameraClippingRange()
                finally:
                    self._syncing = False
                # Defer render to next Qt event loop tick so the source plotter's
                # OpenGL context is released before wglMakeCurrent is called on
                # the target window (two concurrent wglMakeCurrent → ERROR).
                QTimer.singleShot(0, target_plotter.render)
            return _cb

        cb_before_to_after = _make_sync_callback(self.plotter_before, self.plotter_after)
        cb_after_to_before = _make_sync_callback(self.plotter_after,  self.plotter_before)

        for event_name in _SYNC_EVENTS:
            self.plotter_before.iren.interactor.AddObserver(event_name, cb_before_to_after)
            self.plotter_after.iren.interactor.AddObserver(event_name, cb_after_to_before)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _compute_camera_params(self) -> Tuple[List[float], List[float], List[float]]:
        """Return (position, focal_point, up) centred on the contact centroid."""
        cx, cy, cz = self._contact_centroid.tolist()
        _up = [0.375, -0.904, -0.201]
        _pos = [cx, cy, cz - 400.0]
        return _pos, [cx, cy, cz], _up

    def _recenter_view(self) -> None:
        """Reset both cameras to the default view."""
        cam_pos, cam_focal, cam_up = self._compute_camera_params()
        for plotter in (self.plotter_before, self.plotter_after):
            plotter.camera.focal_point = cam_focal
            plotter.camera.position    = cam_pos
            plotter.camera.up          = cam_up
            plotter.renderer.ResetCameraClippingRange()
            plotter.render()

    # ------------------------------------------------------------------
    # Frame update — the hot path
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> None:
        """
        Update all dynamic actors for *frame_idx* and render both plotters.

        1. Remove bounding proxies once real geometry is present
        2. DeepCopy contact points for before and after sides
        3. Honour visibility flags for forearm and contact actors
        4. ResetCameraClippingRange() + render() on both plotters
        5. Update NeuralDataPanel cursor
        6. Update frame label
        """
        self.current_index = frame_idx

        # 1. Remove bounding proxies once real geometry is available
        if self._bounds_proxy_active_before and (
            self._before_forearm_pv is not None
            or (self._before_contact_pts and any(
                p is not None
                for p in self._before_contact_pts[:min(frame_idx + 1, 10)]
            ))
        ):
            self._bounds_proxy_active_before = False
            try:
                self.plotter_before.remove_actor('_bounds_proxy')
            except Exception:
                pass

        if self._bounds_proxy_active_after and (
            self._after_forearm_pv is not None
            or (self._after_contact_pts and any(
                p is not None
                for p in self._after_contact_pts[:min(frame_idx + 1, 10)]
            ))
        ):
            self._bounds_proxy_active_after = False
            try:
                self.plotter_after.remove_actor('_bounds_proxy')
            except Exception:
                pass

        # 2. Before contact points
        if self._before_contact_pts is not None:
            _cpts_before: Optional[np.ndarray] = None
            if self._visibility.get('contact_before', True):
                _cpts_before = (
                    self._before_contact_pts[frame_idx]
                    if frame_idx < len(self._before_contact_pts)
                    else None
                )
            if _cpts_before is not None and len(_cpts_before) > 0:
                self._mesh_contact_before.DeepCopy(
                    pv.PolyData(_cpts_before.astype(np.float32))
                )
            else:
                self._mesh_contact_before.DeepCopy(
                    pv.PolyData(np.empty((0, 3), dtype=np.float32))
                )

        # 3. After contact points
        if self._after_contact_pts is not None:
            _cpts_after: Optional[np.ndarray] = None
            if self._visibility.get('contact_after', True):
                _cpts_after = (
                    self._after_contact_pts[frame_idx]
                    if frame_idx < len(self._after_contact_pts)
                    else None
                )
            if _cpts_after is not None and len(_cpts_after) > 0:
                self._mesh_contact_after.DeepCopy(
                    pv.PolyData(_cpts_after.astype(np.float32))
                )
            else:
                self._mesh_contact_after.DeepCopy(
                    pv.PolyData(np.empty((0, 3), dtype=np.float32))
                )

        # 4. Forearm visibility
        if hasattr(self, '_actor_forearm_before') and self._actor_forearm_before is not None:
            if self._visibility.get('forearm_before', True):
                self._actor_forearm_before.VisibilityOn()
            else:
                self._actor_forearm_before.VisibilityOff()

        if hasattr(self, '_actor_forearm_after') and self._actor_forearm_after is not None:
            if self._visibility.get('forearm_after', True):
                self._actor_forearm_after.VisibilityOn()
            else:
                self._actor_forearm_after.VisibilityOff()

        # 5. Contact visibility (actor-level toggle when points absent anyway,
        #    but also honour the checkbox when set explicitly to hidden)
        if hasattr(self, '_actor_contact_before') and self._actor_contact_before is not None:
            if self._visibility.get('contact_before', True):
                self._actor_contact_before.VisibilityOn()
            else:
                self._actor_contact_before.VisibilityOff()

        if hasattr(self, '_actor_contact_after') and self._actor_contact_after is not None:
            if self._visibility.get('contact_after', True):
                self._actor_contact_after.VisibilityOn()
            else:
                self._actor_contact_after.VisibilityOff()

        # 6. Render both plotters
        self.plotter_before.renderer.ResetCameraClippingRange()
        self.plotter_before.render()

        self.plotter_after.renderer.ResetCameraClippingRange()
        self.plotter_after.render()

        # 7. Neural panel cursor
        if self._neural_panel is not None:
            self._neural_panel.update_cursor(frame_idx, self._neural_scale)

        # 8. Frame label
        self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")

    # ------------------------------------------------------------------
    # showEvent / deferred start
    # ------------------------------------------------------------------

    def _deferred_start(self) -> None:
        """Initialise both VTK interactors, then render frame 0."""
        for plotter in (self.plotter_before, self.plotter_after):
            try:
                plotter.interactor.Initialize()
            except Exception:
                pass
            sz = plotter.interactor.size()
            if sz.width() > 0 and sz.height() > 0:
                plotter.render_window.SetSize(sz.width(), sz.height())
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
        actor_map: Dict[str, object] = {
            'forearm_before':  getattr(self, '_actor_forearm_before', None),
            'forearm_after':   getattr(self, '_actor_forearm_after',  None),
            'contact_before':  getattr(self, '_actor_contact_before', None),
            'contact_after':   getattr(self, '_actor_contact_after',  None),
        }
        actor = actor_map.get(key)
        if actor is not None:
            actor.GetProperty().SetPointSize(float(value))
            # Render the relevant plotter only
            if key in ('forearm_before', 'contact_before'):
                self.plotter_before.render()
            else:
                self.plotter_after.render()
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
        """Stop timers and finalize both VTK render windows on close.

        Explicitly calling plotter.close() releases the OpenGL context
        (wglMakeCurrent / Finalize) before the next viewer window is created.
        Without this, the second and subsequent windows in a sequential loop
        encounter a stale context from the previous viewer and emit
        'wglMakeCurrent failed' errors.
        """
        self._play_timer.stop()
        self._drag_timer.stop()
        try:
            self.plotter_before.close()
        except Exception:
            pass
        try:
            self.plotter_after.close()
        except Exception:
            pass
        super().closeEvent(event)
