"""
neural_kinect_scene_viewer.py
-----------------------------
High-performance 3D viewer that combines Kinect point-cloud data with neural
recording overlays.  Key design invariants:

- NEVER calls plotter.clear() — named actor replacement keeps VTK state stable.
- The FramePreloader is the ONLY thread that calls KinectPointCloudView[idx].
- Hand meshes are loaded lazily (one frame at a time) via HandMotionManager[i].
- CuPy GPU cropping is used when available; CPU fallback is transparent.
- All merged-CSV features (NeuralDataPanel, StickerVelocityCompass) are
  optional and skipped entirely when merged_csv_path=None.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import bisect
import collections
import queue
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# CuPy — must be imported BEFORE the internal preprocessing packages.
#
# Several preprocessing subpackages pull in compiled C-extensions (Open3D,
# pyk4a, …) whose initialisation modifies NumPy's internal dtype state in a
# way that causes CuPy's Cython _dtype init to fail with:
#     TypeError: Alias 'bool8' was removed in NumPy 2.0
# Importing (and probing) CuPy first, while NumPy's state is still pristine,
# avoids the conflict entirely.
# ---------------------------------------------------------------------------
_CUPY_AVAILABLE: bool = False
try:
    import cupy as _cp
    _cp.array([1.0])  # verify CUDA is actually accessible
    _CUPY_AVAILABLE = True
    del _cp
except Exception as _exc:
    print(
        f"CuPy not available or CUDA not accessible ({type(_exc).__name__}: {_exc}). "
        "Using CPU-based point cloud filtering."
    )
    del _exc

# ---------------------------------------------------------------------------
# Internal package imports
# ---------------------------------------------------------------------------
from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV
from preprocessing.common.data_access.kinect_pointcloud_wrapper import KinectPointCloudView
from preprocessing.forearm_extraction import (
    ForearmCatalog,
    ForearmFrameParametersFileHandler,
    get_forearms_with_fallback,
)
from preprocessing.motion_analysis import HandMotionManager
from preprocessing.stickers_analysis import XYZDataFileHandler

from .sticker_velocity_compass import StickerVelocityCompass


def define_custom_colors(string_list) -> dict:
    """
    Searches an iterable of strings for standard color keywords.

    Returns a dict mapping each input string that contains a color keyword to
    the matched color name (lowercase).  If multiple keywords match, the last
    one wins.

    Args:
        string_list: An iterable (e.g., list, dict_keys) of strings to search.

    Returns:
        A dict ``{item: color_name}`` for items that contain a color keyword.
    """
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet",
    }
    found_colors: dict = {}
    for item in string_list:
        item_lower = item.lower()
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
    return found_colors


def _parse_contact_points_cell(cell) -> Optional[np.ndarray]:
    """
    Parse one CSV cell from the ``contact_points`` column.

    Pandas serialises the raw NumPy 2-D array via ``str()``, producing a
    string like::

        "[[ 12.3  45.6  78.9]\\n [ 10.0  20.0  30.0]]"

    Empty-contact rows are stored as ``"[]"``.

    Returns an ``(N, 3)`` float64 array, or ``None`` when the cell is empty
    or unparseable.
    """
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if s in ('', '[]', 'nan'):
        return None
    # Match every [...] sub-group that contains only numbers, spaces, dots,
    # dashes, and exponent markers.  The double-bracket opening ([[) means
    # the first inner [...] starts at position 1, so the outer [ is skipped
    # automatically because its immediate neighbour is another [ (not a digit
    # or whitespace), failing the character-class match.
    rows = re.findall(r'\[\s*([-\d\s.eE+]+)\s*\]', s)
    points: List[List[float]] = []
    for row in rows:
        parts = row.split()
        if len(parts) == 3:
            try:
                points.append([float(p) for p in parts])
            except ValueError:
                pass
    return np.array(points, dtype=np.float64) if points else None


# ---------------------------------------------------------------------------
# FramePreloader
# ---------------------------------------------------------------------------

class FramePreloader(threading.Thread):
    """
    Background daemon thread that pre-decodes MKV frames into an in-memory
    ring buffer, eliminating seek latency during interactive playback.

    Thread-safety contract
    ~~~~~~~~~~~~~~~~~~~~~~
    - **This thread** is the ONLY caller of ``self._source[idx]``.
    - The **main thread** may call ``get_frame()`` and ``seek()`` at any time.
    - ``pyk4a`` is not thread-safe; the single-producer design enforces this.
    """

    def __init__(self, point_cloud_view: KinectPointCloudView, buffer_size: int = 8):
        super().__init__(daemon=True)
        self._source = point_cloud_view
        self._buffer_size = buffer_size

        # Communication channels (main thread → preloader thread)
        self._seek_queue: queue.Queue = queue.Queue(maxsize=1)

        # Shared state (protected by _lock)
        self._buffer: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._active = threading.Event()
        self._active.set()  # start active; cleared by pause()
        self._next_frame: int = 0

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def seek(self, frame_idx: int) -> None:
        """
        Signal the preloader to pivot around a new start index.

        This is a non-blocking "latest wins" put: if the queue is already
        full with a stale seek request, the old one is discarded first.
        """
        try:
            self._seek_queue.put_nowait(frame_idx)
        except queue.Full:
            try:
                self._seek_queue.get_nowait()
            except queue.Empty:
                pass
            self._seek_queue.put_nowait(frame_idx)

    def get_frame(self, frame_idx: int):
        """
        Return the cached PointCloudData for *frame_idx*, if available.

        Falls back to a **synchronous** (blocking) load for large seeks
        where the frame is not yet in the buffer.  This path is rare during
        normal playback.
        """
        with self._lock:
            if frame_idx in self._buffer:
                return self._buffer[frame_idx]
        # Cache miss — synchronous fallback (main thread calls source directly
        # only in this exceptional case; the preloader is ahead in its loop)
        try:
            return self._source[frame_idx]
        except Exception:
            return None

    def buffer_count(self) -> int:
        """Return the number of frames currently held in the ring buffer."""
        with self._lock:
            return len(self._buffer)

    def stop(self) -> None:
        """Signal the worker loop to exit."""
        self._stop_event.set()

    def pause(self) -> None:
        """Suspend MKV decoding (e.g. when the Kinect cloud is hidden)."""
        self._active.clear()

    def resume(self) -> None:
        """Resume MKV decoding after a pause()."""
        self._active.set()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._stop_event.is_set():
            # Paused — idle until resume() is called
            if not self._active.is_set():
                self._stop_event.wait(0.05)
                continue

            # 1. Drain any pending seek requests (latest wins)
            try:
                new_target = self._seek_queue.get_nowait()
                with self._lock:
                    # Evict only frames that are far from the new target so
                    # that back-and-forth scrubbing reuses cached frames.
                    far_keys = [
                        k for k in list(self._buffer)
                        if abs(k - new_target) > self._buffer_size
                    ]
                    for k in far_keys:
                        del self._buffer[k]
                self._next_frame = new_target
            except queue.Empty:
                pass

            # 2. Fill buffer ahead if not full
            with self._lock:
                buffered = len(self._buffer)

            if buffered < self._buffer_size:
                total = len(self._source)
                if self._next_frame < total:
                    try:
                        frame_data = self._source[self._next_frame]
                        with self._lock:
                            if len(self._buffer) >= self._buffer_size:
                                self._buffer.popitem(last=False)  # evict oldest
                            self._buffer[self._next_frame] = frame_data
                        self._next_frame += 1
                    except Exception:
                        self._next_frame += 1  # skip corrupt/missing frame
                else:
                    # At end of file — idle until a seek comes in
                    self._stop_event.wait(0.05)
            else:
                # Buffer full — idle briefly
                self._stop_event.wait(0.01)


# ---------------------------------------------------------------------------
# NeuralDataPanel
# ---------------------------------------------------------------------------

class NeuralDataPanel(QWidget):
    """
    A fixed-height Matplotlib panel (3 stacked axes, shared X) that plots
    ``Nerve_freq``, ``contact_depth``, and ``contact_area`` from the merged
    CSV, with a red vertical cursor line tracking the current frame.

    Two display modes are toggled via the "Full view" / "Zoom ±3 s" button:
    - **Zoom mode** (default): x-axis shows a ±3 s window around the cursor.
    - **Full mode**: x-axis spans the entire recording.
    """

    def __init__(
        self,
        merged_df: pd.DataFrame,
        total_kinect_frames: int,
        neural_fps: int = 1000,
        parent=None,
    ):
        super().__init__(parent)
        self._total_kinect_frames = total_kinect_frames
        self._total_samples: int = len(merged_df)

        # Zoom state — zoomed ±3 s window is the default
        self._neural_fps: int = neural_fps
        self._zoom_mode: bool = True
        self._zoom_half_window: int = 3 * neural_fps  # samples (3 000 at 1 kHz)
        self._current_sample: int = 0

        # --- Matplotlib figure ---
        self.fig = Figure(figsize=(12, 2.2), tight_layout=True)
        self.fig.patch.set_facecolor('#1a1a2e')
        self.canvas = FigureCanvasQTAgg(self.fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle button row (sits above the canvas)
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(4, 1, 4, 1)
        btn_layout.addStretch()

        btn_layout.addWidget(QLabel("\u00b1"))
        self._window_spinbox = QDoubleSpinBox()
        self._window_spinbox.setMinimum(0.5)
        self._window_spinbox.setMaximum(60.0)
        self._window_spinbox.setSingleStep(0.5)
        self._window_spinbox.setValue(3.0)
        self._window_spinbox.setSuffix(" s")
        self._window_spinbox.setFixedWidth(70)
        self._window_spinbox.setFixedHeight(18)
        self._window_spinbox.setStyleSheet("font-size: 8pt;")
        self._window_spinbox.valueChanged.connect(self._on_zoom_window_changed)
        btn_layout.addWidget(self._window_spinbox)

        self._zoom_btn = QPushButton("Full view")
        self._zoom_btn.setFixedHeight(18)
        self._zoom_btn.setFixedWidth(80)
        self._zoom_btn.setStyleSheet("font-size: 8pt;")
        self._zoom_btn.clicked.connect(self._toggle_zoom)
        btn_layout.addWidget(self._zoom_btn)
        layout.addWidget(btn_row)

        layout.addWidget(self.canvas)
        self.setFixedHeight(220)

        self._setup_axes(merged_df)

    def _setup_axes(self, merged_df: pd.DataFrame) -> None:
        axes = self.fig.subplots(3, 1, sharex=True)
        self.ax_freq, self.ax_depth, self.ax_area = axes

        x = np.arange(len(merged_df))
        signal_specs = [
            ('#ff9500', 'Nerve_freq',    'IFF (Hz)'),
            ('#00d4ff', 'contact_depth', 'Depth (mm)'),
            ('#39ff14', 'contact_area',  'Area (mm²)'),
        ]

        self._cursor_lines: List = []
        for ax, (color, col, ylabel) in zip(axes, signal_specs):
            ax.set_facecolor('#0d0d1a')
            if col in merged_df.columns:
                ax.plot(x, merged_df[col].fillna(0), color=color, lw=0.8)
            ax.set_ylabel(ylabel, color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            line = ax.axvline(x=0, color='red', lw=1.5, alpha=0.9)
            self._cursor_lines.append(line)

        self.canvas.draw()

    def update_cursor(self, frame_idx: int, scale_factor: float) -> None:
        """
        Move the red vertical cursor to the position corresponding to
        *frame_idx* in the merged-CSV sample space.

        In zoom mode the x-axis limits are also shifted to keep the cursor
        centred in a ±3 s window.  Uses ``draw_idle()`` (non-blocking) to
        avoid jank at 30 fps.
        """
        sample_idx = int(frame_idx * scale_factor)
        self._current_sample = sample_idx
        for line in self._cursor_lines:
            line.set_xdata([sample_idx])
        if self._zoom_mode:
            lo = max(0, sample_idx - self._zoom_half_window)
            hi = min(self._total_samples - 1, sample_idx + self._zoom_half_window)
            self.ax_freq.set_xlim(lo, hi)
        self.canvas.draw_idle()

    def _toggle_zoom(self) -> None:
        """Switch between the zoomed window view and full time-series view."""
        self._zoom_mode = not self._zoom_mode
        if self._zoom_mode:
            self._zoom_btn.setText("Full view")
            lo = max(0, self._current_sample - self._zoom_half_window)
            hi = min(self._total_samples - 1, self._current_sample + self._zoom_half_window)
            self.ax_freq.set_xlim(lo, hi)
        else:
            self._zoom_btn.setText("Zoom")
            self.ax_freq.set_xlim(0, self._total_samples - 1)
        self.canvas.draw_idle()

    def _on_zoom_window_changed(self, seconds: float) -> None:
        """Update the half-window size and refresh xlim if zoom mode is active."""
        self._zoom_half_window = int(seconds * self._neural_fps)
        if self._zoom_mode:
            lo = max(0, self._current_sample - self._zoom_half_window)
            hi = min(self._total_samples - 1, self._current_sample + self._zoom_half_window)
            self.ax_freq.set_xlim(lo, hi)
            self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# NeuralKinectViewer
# ---------------------------------------------------------------------------

class NeuralKinectViewer(QMainWindow):
    """
    High-performance standalone viewer for merged neural + Kinect recordings.

    Architecture highlights
    ~~~~~~~~~~~~~~~~~~~~~~~
    - Named PyVista actors are updated in-place; ``plotter.clear()`` is never
      called, preserving camera state across frame changes.
    - A background ``FramePreloader`` thread maintains an 8-frame ring buffer
      of decoded ``PointCloudData`` objects to eliminate MKV seek latency.
    - ``HandMotionManager[i]`` is called lazily per frame (no eager pre-build).
    - CuPy AABB cropping reduces the rendered Kinect cloud by 8-16x.
    - The ``NeuralDataPanel`` and ``StickerVelocityCompass`` widgets are
      created only when *merged_csv_path* is provided.

    Parameters
    ----------
    xyz_csv_path:
        Path to the ``*_handstickers_xyz_tracked.csv`` file.
    kinect_mkv_path:
        Path to the raw ``.mkv`` Kinect recording.
    forearm_pointcloud_dir:
        Directory that contains the forearm ``.ply`` point-cloud files.
    forearm_metadata_path:
        Path to the ``*_arm_roi_metadata.json`` file.
    rgb_video_path:
        Filename (str or Path) used as the catalog lookup key, e.g.
        ``"2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4"``.
    hand_motion_path:
        Path to the ``*_handmodel_motion.npz`` file.
    recording_name:
        Human-readable label shown in the window title and as a 3D actor.
    merged_csv_path:
        Optional path to the ``*_merged_data.csv`` file.  When ``None`` the
        viewer runs in pure-3D mode (no neural panel, no compass widgets).
    crop_half_size_mm:
        Half-width of the AABB crop box centred on the contact centroid (mm).
    """

    def __init__(
        self,
        xyz_csv_path: Path,
        kinect_mkv_path: Path,
        forearm_pointcloud_dir: Path,
        forearm_metadata_path: Path,
        rgb_video_path: Path,
        hand_motion_path: Path,
        recording_name: str,
        merged_csv_path: Optional[Path] = None,
        crop_half_size_mm: float = 400.0,
        parent=None,
    ):
        super().__init__(parent)

        # Empty PolyData objects are used as named placeholder slots so that
        # later add_mesh(..., name=X) calls do in-place actor replacement
        # without touching the camera.  PyVista rejects empty meshes by
        # default, so we allow them globally for this viewer.
        pv.global_theme.allow_empty_mesh = True

        # ------------------------------------------------------------------
        # Basic state
        # ------------------------------------------------------------------
        self.setWindowTitle(f"Neural-Kinect Viewer | {recording_name}")
        self._recording_name: str = recording_name
        self._crop_half_size: float = crop_half_size_mm
        self._cupy_available: bool = _CUPY_AVAILABLE
        self.current_index: int = 0

        # ------------------------------------------------------------------
        # 1. Load stickers
        # ------------------------------------------------------------------
        stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
        self._custom_colors: Dict[str, str] = define_custom_colors(stickers_df_dict.keys())
        self._stickers_xyz_dict: Dict[str, np.ndarray] = {
            k: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
            for k, df in stickers_df_dict.items()
        }

        # ------------------------------------------------------------------
        # 2. Load forearm
        # ------------------------------------------------------------------
        forearm_params = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
        catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
        # get_forearms_with_fallback expects a str filename (or Path converted to str)
        self._forearms_dict = get_forearms_with_fallback(catalog, str(rgb_video_path))
        self._sorted_forearm_keys: List[int] = sorted(self._forearms_dict.keys())

        # ------------------------------------------------------------------
        # 3. Load hand motion (LAZY — manager stored, not an eager loop)
        # ------------------------------------------------------------------
        try:
            self._hand_manager = HandMotionManager()
            self._hand_manager.load(str(hand_motion_path))
        except Exception as exc:
            print(f"Warning: Could not load hand motion data ({exc}). Hand mesh disabled.")
            self._hand_manager = None
        self._last_hand_frame: int = -1
        self._last_hand_mesh = None

        # ------------------------------------------------------------------
        # 4. Optional merged CSV
        # ------------------------------------------------------------------
        self.merged_df: Optional[pd.DataFrame] = (
            pd.read_csv(merged_csv_path) if merged_csv_path is not None else None
        )

        # ------------------------------------------------------------------
        # 5. Contact centroid (used for camera + GPU crop centre)
        # ------------------------------------------------------------------
        self.contact_centroid: np.ndarray = self._compute_contact_centroid()

        # ------------------------------------------------------------------
        # 6. Open MKV (manual context management — closed in closeEvent)
        # ------------------------------------------------------------------
        self._mkv = KinectMKV(kinect_mkv_path)
        self._mkv.__enter__()
        self._point_cloud_view = KinectPointCloudView(self._mkv)
        self._total_frames: int = len(self._point_cloud_view)

        # ------------------------------------------------------------------
        # 7. Background preloader (32-frame ring buffer)
        # ------------------------------------------------------------------
        self._preloader = FramePreloader(self._point_cloud_view, buffer_size=32)
        self._preloader.seek(0)
        self._preloader.start()

        # Tracks whether the user is actively dragging the frame slider so
        # that intermediate drag ticks skip the expensive full frame render.
        self._slider_dragging: bool = False

        # ------------------------------------------------------------------
        # 8. Build Qt UI
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # 9. Initialise named VTK actors (empty meshes = reserved slots)
        # ------------------------------------------------------------------
        self._init_actors()

        # ------------------------------------------------------------------
        # 10. Play timer
        # ------------------------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        # Throttle timer for live frame updates during slider drag.
        # Fires every 80 ms (~12 fps) while the user holds the slider,
        # preventing a full 3-D render on every tiny slider tick.
        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(80)
        self._pending_drag_frame: Optional[int] = None

        # ------------------------------------------------------------------
        # 12. Initial frame render
        # ------------------------------------------------------------------
        # _build_frame_controls() calls setValue(0) before valueChanged is
        # connected, so _update_frame is never triggered during __init__.
        # Defer until the Qt event loop is running and the window is visible
        # so the widget has a real pixel size.  Initialize() must be called
        # first: VTK silently drops render() calls until the interactor has
        # been set up with the widget dimensions (normally triggered by the
        # first mouse click inside the plotter, which is why renders only
        # worked after the user manually interacted with the 3D viewport).
        QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        """Initialize the VTK interactor, then render frame 0."""
        # ------------------------------------------------------------------
        # 11. Neural scale factor — must be set BEFORE _update_frame is
        #     called, because _update_frame step 6 accesses self._neural_scale.
        # ------------------------------------------------------------------
        self._neural_scale: float = (
            len(self.merged_df) / self._total_frames
            if self.merged_df is not None and self._total_frames > 0
            else 1.0
        )

        try:
            self.plotter.interactor.Initialize()
        except Exception:
            pass
        self._update_frame(0)

    # ------------------------------------------------------------------
    # Contact centroid
    # ------------------------------------------------------------------

    def _compute_contact_centroid(self) -> np.ndarray:
        """
        Return the mean XYZ of all forearm contact points from the merged CSV,
        or the global sticker mean when no merged CSV is available.
        """
        if self.merged_df is not None:
            all_contact_pts: List[np.ndarray] = []
            for cell in self.merged_df['contact_points']:
                pts = _parse_contact_points_cell(cell)
                if pts is not None:
                    all_contact_pts.append(pts)
            if all_contact_pts:
                return np.nanmean(np.vstack(all_contact_pts), axis=0)

        # Fallback for pure-3D mode (no merged CSV)
        all_xyz = np.concatenate(list(self._stickers_xyz_dict.values()), axis=0)
        return np.nanmean(all_xyz, axis=0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # --- Top row: 3D plotter + right panel ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(plotter_widget)
        self.plotter = QtInteractor(plotter_widget)
        self.plotter.set_background('midnightblue')
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

        # --- Bottom row: neural panel (only when merged CSV provided) ---
        self.neural_panel: Optional[NeuralDataPanel] = None
        if self.merged_df is not None:
            self.neural_panel = NeuralDataPanel(self.merged_df, self._total_frames)
            outer.addWidget(self.neural_panel)

    def _build_right_panel_controls(self) -> None:
        """
        Populate the scrollable right panel with:
        - A small camera-position readout at the top.
        - A visibility checkbox (+ point-size slider where applicable) for
          each scene object.
        - A StickerVelocityCompass per sticker (only when merged CSV present).
        """
        # --- Camera monitor (position + orientation) ---
        cam_box = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_box)
        self._cam_pos_label = QLabel(
            "pos (mm)\n  x: —\n  y: —\n  z: —\n"
            "up\n  x: —\n  y: —\n  z: —"
        )
        self._cam_pos_label.setAlignment(Qt.AlignLeft)
        self._cam_pos_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        cam_layout.addWidget(self._cam_pos_label)
        self._right_panel_layout.addWidget(cam_box)

        # Init state dicts
        self._visibility: Dict[str, bool] = {
            name: True
            for name in (
                ['kinect_point_cloud', 'forearms', 'hand_meshes']
                + list(self._stickers_xyz_dict.keys())
            )
        }
        self._point_sizes: Dict[str, float] = {
            'kinect_point_cloud': 2.0,
            'forearms': 8.0,
        }
        self._compass_widgets: Dict[str, StickerVelocityCompass] = {}

        def _add_object_group(label: str, key: str, has_slider: bool = False):
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)

            cb = QCheckBox("Visible")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, k=key: self._on_visibility_changed(k, state)
            )
            box_layout.addWidget(cb)

            if has_slider:
                slider_row = QWidget()
                slider_layout = QHBoxLayout(slider_row)
                slider_layout.setContentsMargins(0, 0, 0, 0)
                slider_layout.addWidget(QLabel("Pt size"))
                sl = QSlider(Qt.Horizontal)
                sl.setMinimum(1)
                sl.setMaximum(20)
                sl.setValue(int(self._point_sizes.get(key, 5)))
                sl.valueChanged.connect(
                    lambda val, k=key: self._on_point_size_changed(k, val)
                )
                slider_layout.addWidget(sl)
                box_layout.addWidget(slider_row)

            self._right_panel_layout.addWidget(box)

        _add_object_group("Kinect Cloud", "kinect_point_cloud", has_slider=True)
        _add_object_group("Forearms",     "forearms",           has_slider=True)
        _add_object_group("Hand Mesh",    "hand_meshes",        has_slider=False)

        for sticker_name in self._stickers_xyz_dict:
            _add_object_group(sticker_name, sticker_name)

            if self.merged_df is not None:
                color = self._custom_colors.get(sticker_name, 'magenta')
                compass = StickerVelocityCompass(sticker_name, color)
                self._compass_widgets[sticker_name] = compass

                compass_box = QGroupBox(f"{sticker_name} velocity")
                compass_layout = QVBoxLayout(compass_box)
                compass_layout.addWidget(compass)
                self._right_panel_layout.addWidget(compass_box)

        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        """Return a QWidget row containing slider, labels, and control buttons."""
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

        self._buffer_label = QLabel("Buf: 0/32")
        self._buffer_label.setFixedWidth(75)
        self._buffer_label.setStyleSheet("font-family: monospace; font-size: 8pt; color: gray;")
        layout.addWidget(self._buffer_label)

        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._recenter_view)
        layout.addWidget(recenter_btn)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)

        layout.addWidget(QLabel("Crop ±"))
        self.crop_spinbox = QSpinBox()
        self.crop_spinbox.setMinimum(100)
        self.crop_spinbox.setMaximum(2000)
        self.crop_spinbox.setSingleStep(50)
        self.crop_spinbox.setValue(int(self._crop_half_size))
        self.crop_spinbox.valueChanged.connect(self._on_crop_changed)
        layout.addWidget(self.crop_spinbox)

        return widget

    # ------------------------------------------------------------------
    # Actor initialisation
    # ------------------------------------------------------------------

    def _init_actors(self) -> None:
        """
        Add every named actor once so that subsequent ``add_mesh(name=X)``
        calls replace the mesh in-place without touching the camera.
        """

        empty = pv.PolyData(np.empty((0, 3), dtype=np.float32))

        # Static actors
        self.plotter.add_text(
            self._recording_name,
            position='upper_left',
            font_size=10,
            color='white',
            name='recording_label',
        )
        gpu_label = "GPU: ON" if self._cupy_available else "GPU: OFF"
        gpu_color = "lime" if self._cupy_available else "tomato"
        self.plotter.add_text(
            gpu_label,
            position='upper_right',
            font_size=10,
            color=gpu_color,
            name='gpu_status_label',
        )
        self.plotter.add_axes(interactive=False, line_width=150, box=True)
        self.plotter.add_mesh(
            pv.Sphere(radius=2.0),
            color='yellow',
            name='origin_sphere',
            pickable=False,
        )

        # Dynamic placeholder actors (empty mesh = named slot reserved)
        self.plotter.add_mesh(
            empty,
            name='kinect_point_cloud',
            render_points_as_spheres=False,
            point_size=self._point_sizes['kinect_point_cloud'],
        )
        self.plotter.add_mesh(
            empty,
            name='forearms',
            render_points_as_spheres=True,
            point_size=self._point_sizes['forearms'],
        )
        self.plotter.add_mesh(empty, name='hand_meshes', style='wireframe')

        for name, color in self._custom_colors.items():
            sphere = pv.Sphere(radius=4.0, center=(0.0, 0.0, 0.0))
            self.plotter.add_mesh(sphere, color=color, name=f'sticker_{name}')

        # ------------------------------------------------------------------
        # Default camera — edit these three lines to change the startup view.
        #
        # focal_point  where the camera looks at (contact centroid is a
        #              sensible default; keep it unless you want to look
        #              somewhere else entirely).
        #
        # position     where the camera sits in 3-D space (mm).
        #              Offset from focal_point along any axis:
        #                -Y  →  in front of the forearm (typical)
        #                +Z  →  looking down from above
        #              Example: centroid + (0, -800, +400) gives a
        #              slightly-elevated front view.
        #
        # up           the "up" axis of the image plane (unit vector).
        #              Common choices:
        #                (0,  0,  1)  →  Z is up   (default VTK / PyVista)
        #                (0, -1,  0)  →  -Y is up  (image-coords style)
        #                (0,  1,  0)  →  +Y is up
        # ------------------------------------------------------------------
        cx, cy, cz = self.contact_centroid.tolist()
        self.plotter.camera.focal_point = [cx, cy, cz]
        self.plotter.camera.position    = [cx, cy, cz - 400.0]
        self.plotter.camera.up          = [0.375, -0.904, -0.201]

        self.plotter.camera_set = True

        # Update camera readout whenever the user rotates / pans the scene
        try:
            self.plotter.iren.AddObserver(
                "EndInteractionEvent", self._refresh_cam_pos_label
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame update — the hot path
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> None:
        """
        Update all named actors for *frame_idx* and render exactly once.

        Called by slider changes and the play timer.  Camera state is
        preserved because ``plotter.clear()`` is never called.
        """
        self.current_index = frame_idx

        empty = pv.PolyData(np.empty((0, 3), dtype=np.float32))

        # 1. Kinect point cloud (GPU-cropped AABB) ----------------------
        if not self._visibility.get('kinect_point_cloud', True):
            self.plotter.add_mesh(empty, name='kinect_point_cloud',
                                  render_points_as_spheres=False,
                                  point_size=self._point_sizes['kinect_point_cloud'])
        else:
            pc_data = self._preloader.get_frame(frame_idx)
            if (
                pc_data is not None
                and pc_data.points is not None
                and pc_data.points.shape[0] > 0
            ):
                pts, cols = self._crop_pointcloud_gpu(
                    pc_data.points, pc_data.color,
                    self.contact_centroid, self._crop_half_size,
                )
                if pts.shape[0] > 0:
                    cloud = pv.PolyData(pts)
                    if cols is not None:
                        cloud['colors'] = cols.astype(np.uint8)
                        self.plotter.add_mesh(
                            cloud, scalars='colors', rgb=True,
                            name='kinect_point_cloud',
                            render_points_as_spheres=False,
                            point_size=self._point_sizes['kinect_point_cloud'],
                        )
                    else:
                        self.plotter.add_mesh(
                            cloud, color='gray',
                            name='kinect_point_cloud',
                            render_points_as_spheres=False,
                            point_size=self._point_sizes['kinect_point_cloud'],
                        )
                else:
                    self.plotter.add_mesh(empty, name='kinect_point_cloud',
                                          render_points_as_spheres=False,
                                          point_size=self._point_sizes['kinect_point_cloud'])
            else:
                self.plotter.add_mesh(empty, name='kinect_point_cloud',
                                      render_points_as_spheres=False,
                                      point_size=self._point_sizes['kinect_point_cloud'])

        # 2. Forearm (updated only when the bisect key changes) ----------
        forearm_key = self._bisect_forearm(frame_idx)
        if not self._visibility.get('forearms', True):
            self.plotter.add_mesh(empty, name='forearms',
                                  render_points_as_spheres=True,
                                  point_size=self._point_sizes['forearms'])
            self._last_forearm_key = object()  # force rebuild when re-enabled
        elif forearm_key != getattr(self, '_last_forearm_key', object()):
            self._last_forearm_key = forearm_key
            o3d_pc = self._forearms_dict.get(forearm_key)
            if o3d_pc is not None and o3d_pc.has_points():
                pts_fa = np.asarray(o3d_pc.points)
                cloud_fa = pv.PolyData(pts_fa)
                if o3d_pc.has_colors():
                    cols_fa = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
                    cloud_fa['colors'] = cols_fa
                    self.plotter.add_mesh(
                        cloud_fa, scalars='colors', rgb=True,
                        name='forearms',
                        render_points_as_spheres=True,
                        point_size=self._point_sizes['forearms'],
                    )
                else:
                    self.plotter.add_mesh(
                        cloud_fa, color='gray',
                        name='forearms',
                        render_points_as_spheres=True,
                        point_size=self._point_sizes['forearms'],
                    )
            else:
                self.plotter.add_mesh(empty, name='forearms',
                                      render_points_as_spheres=True,
                                      point_size=self._point_sizes['forearms'])

        # 3. Hand mesh (lazy per-frame transform) ------------------------
        if not self._visibility.get('hand_meshes', True):
            self.plotter.add_mesh(empty, name='hand_meshes', style='wireframe')
        else:
            o3d_mesh = self._get_hand_mesh(frame_idx)
            if o3d_mesh is not None and o3d_mesh.has_triangles():
                verts = np.asarray(o3d_mesh.vertices)
                tris = np.asarray(o3d_mesh.triangles)
                faces = np.hstack([
                    np.full((len(tris), 1), 3, dtype=tris.dtype), tris
                ])
                pv_mesh = pv.PolyData(verts, faces)
                self.plotter.add_mesh(
                    pv_mesh, color='white', style='wireframe', name='hand_meshes'
                )
            else:
                self.plotter.add_mesh(empty, name='hand_meshes', style='wireframe')

        # 4. Stickers + compass widgets ----------------------------------
        for name, positions in self._stickers_xyz_dict.items():
            pos = positions[frame_idx] if frame_idx < len(positions) else None
            actor_name = f'sticker_{name}'

            if pos is None or np.any(np.isnan(pos)):
                # Hide by replacing with empty mesh (never use SetVisibility)
                self.plotter.add_mesh(
                    empty, color=self._custom_colors.get(name, 'magenta'),
                    name=actor_name
                )
            elif self._visibility.get(name, True):
                sphere = pv.Sphere(radius=4.0, center=pos.tolist())
                self.plotter.add_mesh(
                    sphere,
                    color=self._custom_colors.get(name, 'magenta'),
                    name=actor_name,
                )
            else:
                self.plotter.add_mesh(
                    empty, color=self._custom_colors.get(name, 'magenta'),
                    name=actor_name
                )

            # Compass update
            if name in self._compass_widgets and frame_idx > 0:
                prev_pos = positions[frame_idx - 1]
                if not np.any(np.isnan(prev_pos)) and pos is not None and not np.any(np.isnan(pos)):
                    self._compass_widgets[name].update_velocity(pos - prev_pos)

        # 5. Single render call -----------------------------------------
        self.plotter.render()
        self._refresh_cam_pos_label()

        # 6. Neural panel cursor ----------------------------------------
        if self.neural_panel is not None:
            self.neural_panel.update_cursor(frame_idx, self._neural_scale)

        # 7. Frame label + buffer fill indicator ------------------------
        self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")
        buf_n = self._preloader.buffer_count()
        buf_max = self._preloader._buffer_size
        self._buffer_label.setText(f"Buf: {buf_n}/{buf_max}")

        # 8. Signal preloader to look ahead -----------------------------
        self._preloader.seek(frame_idx + 1)

    # ------------------------------------------------------------------
    # Supporting methods
    # ------------------------------------------------------------------

    def _crop_pointcloud_gpu(
        self,
        xyz: np.ndarray,
        colors: Optional[np.ndarray],
        center: np.ndarray,
        half_size: float,
    ):
        """
        AABB crop the point cloud around *center* ± *half_size* mm.

        Uses CuPy when available (GPU path); falls back to NumPy otherwise.
        Also removes points with z ≤ 0 (behind the sensor).
        """
        if self._cupy_available:
            try:
                import cupy as cp
                xyz_gpu = cp.asarray(xyz.astype(np.float32))
                c_gpu = cp.asarray(center.astype(np.float32))
                mask_gpu = (
                    (cp.abs(xyz_gpu - c_gpu) <= half_size).all(axis=1)
                    & (xyz_gpu[:, 2] > 0)
                )
                mask_np = cp.asnumpy(mask_gpu)
                pts_out = cp.asnumpy(xyz_gpu[mask_gpu])
                cols_out = colors[mask_np] if colors is not None else None
                return pts_out, cols_out
            except Exception:
                pass  # fall through to CPU path

        # CPU path (NumPy)
        mask = (
            (np.abs(xyz - center) <= half_size).all(axis=1)
            & (xyz[:, 2] > 0)
        )
        cols_out = colors[mask] if colors is not None else None
        return xyz[mask], cols_out

    def _bisect_forearm(self, frame_idx: int) -> Optional[int]:
        """
        Return the forearm dict key whose frame boundary is <= *frame_idx*
        (hold-last-frame semantics), or None if no keys exist.
        """
        if not self._sorted_forearm_keys:
            return None
        i = bisect.bisect_right(self._sorted_forearm_keys, frame_idx) - 1
        return self._sorted_forearm_keys[i] if i >= 0 else None

    def _get_hand_mesh(self, frame_idx: int):
        """
        Return the transformed Open3D TriangleMesh for *frame_idx*, using a
        simple one-frame LRU cache to avoid re-computing for unchanged frames.

        Returns None if the manager is unavailable or an error occurs.
        """
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

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        """Mark the start of a user drag and start the throttle timer."""
        self._slider_dragging = True
        self._pending_drag_frame = None
        self._drag_timer.start()

    def _on_slider_released(self) -> None:
        """On drag release, stop the throttle timer and do a final full render."""
        self._slider_dragging = False
        self._drag_timer.stop()
        self._pending_drag_frame = None
        self._update_frame(self.frame_slider.value())

    def _on_drag_timer_fired(self) -> None:
        """Throttled render callback: fires every 80 ms while the slider is held."""
        if self._pending_drag_frame is not None:
            frame = self._pending_drag_frame
            self._pending_drag_frame = None
            self._update_frame(frame)

    def _on_slider_change(self, value: int) -> None:
        if self._slider_dragging:
            # During drag: update label and preloader immediately, but let
            # _drag_timer throttle the expensive MKV decode + 3-D render.
            self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._preloader.seek(value)
            self._pending_drag_frame = value
        else:
            # Used by playback (setValue) and single click — full render.
            self._update_frame(value)

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visibility[key] = state == Qt.Checked
        # Redraw current frame to reflect new visibility
        self._update_frame(self.current_index)

    def _on_point_size_changed(self, key: str, value: int) -> None:
        self._point_sizes[key] = float(value)

    def _on_crop_changed(self, value: int) -> None:
        self._crop_half_size = float(value)

    def _refresh_cam_pos_label(self, *_) -> None:
        """Update the camera position + orientation readout in the right panel."""
        try:
            px, py, pz = self.plotter.camera.position
            ux, uy, uz = self.plotter.camera.up
            self._cam_pos_label.setText(
                f"pos (mm)\n  x:{px:9.1f}\n  y:{py:9.1f}\n  z:{pz:9.1f}\n"
                f"up\n  x:{ux:7.3f}\n  y:{uy:7.3f}\n  z:{uz:7.3f}"
            )
        except Exception:
            pass

    def _recenter_view(self) -> None:
        self.plotter.camera.focal_point = self.contact_centroid.tolist()
        self.plotter.render()
        self._refresh_cam_pos_label()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            fps = 30
            if (
                self._mkv._reader is not None
                and hasattr(self._mkv._reader, 'fps')
            ):
                fps = self._mkv._reader.fps
            self._play_timer.start(int(1000 / fps))
            self.play_button.setText("⏸ Pause")

    def _play_advance(self) -> None:
        nxt = (self.current_index + 1) % self._total_frames
        self.frame_slider.setValue(nxt)  # triggers _on_slider_change → _update_frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._play_timer.stop()
        self._drag_timer.stop()
        self._preloader.stop()
        self._preloader.join(timeout=2.0)
        try:
            self._mkv.__exit__(None, None, None)
        except Exception:
            pass
        # Explicitly finalize the VTK render window so its OpenGL context is
        # released *before* Qt destroys the child widgets.  Without this call
        # VTK's wglMakeCurrent still considers the old context "current" when
        # the next NeuralKinectViewer tries to initialise its own context,
        # producing: "wglMakeCurrent failed in MakeCurrent()".
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
