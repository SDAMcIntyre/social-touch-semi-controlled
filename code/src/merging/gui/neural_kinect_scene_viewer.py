"""
neural_kinect_scene_viewer.py
-----------------------------
High-performance 3D viewer that combines Kinect point-cloud data with neural
recording overlays.  Key design invariants:

- NEVER calls plotter.clear() or plotter.add_mesh() inside _update_frame().
  All dynamic actors are registered once in _init_actors() and updated via:
    • PolyData.DeepCopy()  — point clouds and contact points (updates dataset
                              in-place via VTK DeepCopy; same mapper/actor).
    • mesh.points = verts   — hand mesh when triangle count is unchanged
                              (cheapest path: only vertex positions updated).
    • actor.SetPosition()   — sticker spheres (geometry stays at origin;
                              only the actor transform changes).
  plotter.render() propagates all modifications automatically; Modified() is
  called internally by overwrite() and the mesh.points setter, so no explicit
  mapper.Update() calls are needed.
- The FramePreloader is the ONLY thread that calls KinectPointCloudView[idx].
- Hand meshes are loaded lazily (one frame at a time) via HandMotionManager[i].
- CuPy GPU cropping is used when available; CPU fallback is transparent.
- During slider drag or playback the Kinect cloud is subsampled by
  ``_interactive_stride`` (default 4) to cut render time; a single
  full-resolution frame is rendered on pause / slider release.
- All merged-CSV features (NeuralDataPanel, StickerVelocityCompass) are
  optional and skipped entirely when merged_csv_path=None.

Contact depth colouring
~~~~~~~~~~~~~~~~~~~~~~~
When the block spec carries a ``ContactDepthFieldSeries``, the contact actor is
driven by that field instead of the CSV's ``contact_points`` blob, and is
coloured by penetration depth.  This viewer is a **pure sink** for that data:

- The colour range arrives precomputed and **global** over the whole recording
  (``series.clim_penetration_mm``).  Per-frame autoscaling is prohibited — it
  would make the animation lie about relative depth, so the same frame would
  look different depending on whether it was reached by scrubbing forward or
  back.  Nothing here derives a statistic from the field.
- ``penetration_depth_mm = -signed_depth_mm`` is applied by the producer, once.
  No arithmetic is performed on the depths here.
- ``actor.mapper.scalar_range`` is re-asserted after **every** dataset swap.
  On PyVista 0.47.1 a mapper with no explicit clim silently reverts to
  per-frame autoscale (see ``contact_depth_field_viewer.py`` for the measured
  behaviour), so the assertion is what keeps a given depth the same colour at
  every frame.
- The colourmap is ``inferno``, never ``jet``: ``jet`` has non-monotonic
  lightness, invents boundaries the data does not contain, and is hostile to
  colour-vision deficiency
  (``investigation-jet-colormap-perceptual-problems.md``).
- In registered-frame mode the depth *points* are transformed by the block's
  ICP matrix exactly like the cloud, hand mesh and stickers; the depth *values*
  are invariant under a rigid transform and are never recomputed.
- The field is columnar, so it needs no string parsing:
  ``_parse_contact_points_cell`` is not involved in this path at all.

Hot-swap navigation
~~~~~~~~~~~~~~~~~~~
``NeuralKinectViewer`` now accepts an *all_blocks* dict
``{(session_id, block_id): NeuralKinectBlockSpec}`` rather than per-block
constructor arguments.  Two toolbar dropdowns let the user switch between
sessions and blocks without closing the window:

    viewer = NeuralKinectViewer(
        all_blocks={
            ("ST14-01", "block-order-01"): spec_01,
            ("ST14-01", "block-order-02"): spec_02,
        },
        initial_session_id="ST14-01",
        initial_block_id="block-order-01",
    )

On dropdown change ``_teardown_current_block()`` closes the old MKV and stops
the old preloader; then ``_load_block()`` loads the new spec.  The PyVista
plotter is kept alive (no ``plotter.clear()``); named actors are re-seeded via
``_init_actors()`` so the camera survives the swap.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import bisect
import collections
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolBar,
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
from preprocessing.common import define_custom_colors
from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV
from preprocessing.common.data_access.kinect_pointcloud_wrapper import KinectPointCloudView
from preprocessing.forearm_extraction import (
    apply_rigid_transform,
    ForearmCatalog,
    ForearmFrameParametersFileHandler,
    get_forearms_with_fallback,
)
from preprocessing.motion_analysis import HandMotionManager
# Imported from the module, not the package facade: the facade resolves this
# name lazily and the direct import keeps the dependency narrow.  These three
# constants are shared with the tactile-quantification depth viewer so the
# scalar array name, the colourbar title and the colourmap have exactly one
# definition between the two windows.
from preprocessing.motion_analysis.tactile_quantification.gui.contact_depth_field_viewer import (
    COLORMAP as CONTACT_DEPTH_COLORMAP,
    CONTACT_SCALAR_NAME,
    SCALAR_BAR_TITLE as CONTACT_DEPTH_SCALAR_BAR_TITLE,
)
from preprocessing.stickers_analysis import XYZDataFileHandler

from ..contact_depth_field_series import ContactDepthFieldSeries
from .sticker_velocity_compass import StickerVelocityCompass


# ---------------------------------------------------------------------------
# NeuralKinectBlockSpec
# ---------------------------------------------------------------------------

@dataclass
class NeuralKinectBlockSpec:
    """
    All paths and metadata needed to display one session block in
    ``NeuralKinectViewer``.

    Attributes
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
        Filename used as the ``ForearmCatalog`` lookup key, e.g.
        ``Path("2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4")``.
        Only the filename part is used; the full path is not required.
    hand_motion_path:
        Path to the ``*_handmodel_motion.npz`` file.
    recording_name:
        Human-readable label shown in the window title and as a 3D actor.
    merged_csv_path:
        Optional path to the ``*_merged_data.csv`` file.  ``None`` → pure-3D
        mode (no neural panel, no compass widgets).
    registration_transforms_by_forearm_key:
        Optional mapping ``{forearm_key: 4×4 transform}`` for registered-frame
        display.  ``None`` → identity (no registration applied).
    contact_depth_field:
        Optional per-vertex contact depth field, already indexed by frame and
        carrying its own **global** colour range.  ``None`` is an explicit
        *absent* state — the producer resolved the sidecar, did not find it, and
        said so — not a display preference.  When present it becomes the source
        of the contact geometry, replacing the CSV's ``contact_points`` blob.
    """
    xyz_csv_path: Path
    kinect_mkv_path: Path
    forearm_pointcloud_dir: Path
    forearm_metadata_path: Path
    rgb_video_path: Path
    hand_motion_path: Path
    recording_name: str
    merged_csv_path: Optional[Path]
    registration_transforms_by_forearm_key: Optional[Dict[int, np.ndarray]] = field(default=None)
    contact_depth_field: Optional[ContactDepthFieldSeries] = field(default=None)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


def _empty_contact_polydata() -> pv.PolyData:
    """A zero-point contact dataset that still carries the mapped scalar array.

    The array must exist even when empty, or the mapper loses its binding to
    ``CONTACT_SCALAR_NAME`` the first time a no-contact frame is displayed and
    the colours never come back.
    """
    mesh = pv.PolyData(np.empty((0, 3), dtype=np.float32))
    mesh[CONTACT_SCALAR_NAME] = np.empty((0,), dtype=np.float64)
    return mesh


def contact_polydata(
    points: np.ndarray,
    penetration_depth_mm: Optional[np.ndarray] = None,
) -> pv.PolyData:
    """Build the contact-actor dataset from points and optional depth scalars.

    Parameters
    ----------
    points:
        ``(N, 3)`` contact-vertex positions in millimetres.
    penetration_depth_mm:
        ``(N,)`` positive-is-deeper penetration depths, already sign-flipped by
        the producer, or ``None`` when this block has no depth field at all
        (contact geometry then comes from the CSV and renders flat).

        ``None`` carries no scalar array rather than a zero-filled one: the
        actor in that configuration was never bound to ``CONTACT_SCALAR_NAME``,
        and fabricating depths of zero for points whose depth is simply unknown
        would put invented numbers on screen.

    Raises
    ------
    ValueError
        If *points* is not ``(N, 3)``, or the depths are not index-aligned with
        it.  A misaligned pair would paint one vertex with another's depth,
        which is worse than not drawing at all.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"contact points must be (N, 3), got {pts.shape}.")

    mesh = pv.PolyData(pts)
    if penetration_depth_mm is None:
        return mesh

    depths = np.asarray(penetration_depth_mm, dtype=np.float64)
    if depths.ndim != 1 or len(depths) != len(pts):
        raise ValueError(
            f"{len(pts)} contact points vs {depths.shape} penetration depths; "
            "the field must be index-aligned with the points."
        )
    mesh[CONTACT_SCALAR_NAME] = depths
    mesh.set_active_scalars(CONTACT_SCALAR_NAME)
    return mesh


def extract_touch_boundaries(touch_ids: np.ndarray) -> List[tuple]:
    """
    Extract contiguous non-zero blocks from *touch_ids* and return them as a
    list of ``(start_idx, end_idx, touch_id)`` tuples.

    *start_idx* is inclusive; *end_idx* is exclusive (suitable for axvspan).
    Blocks of zero are ignored (gaps between touches).

    Parameters
    ----------
    touch_ids:
        1-D integer array of ``single_touch_id`` values (0 = no touch).

    Returns
    -------
    list of (int, int, int)
        One tuple per contiguous non-zero run: (start_idx, end_idx, touch_id).
    """
    if len(touch_ids) == 0:
        return []

    change_points = np.where(np.diff(touch_ids) != 0)[0] + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends   = np.concatenate((change_points, [len(touch_ids)]))
    run_values = touch_ids[run_starts]

    return [
        (int(s), int(e), int(v))
        for s, e, v in zip(run_starts, run_ends, run_values)
        if v != 0
    ]


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
        Return ``(frame_data, is_exact)`` for *frame_idx*.

        ``is_exact=True``  — the returned data is for exactly *frame_idx*.
        ``is_exact=False`` — the returned data is the nearest cached frame;
                             the caller should schedule a re-render once the
                             exact frame arrives in the buffer.

        Falls back to a synchronous (blocking) load only when the buffer is
        completely empty (e.g. the very first access before the preloader has
        decoded any frames).
        """
        with self._lock:
            if frame_idx in self._buffer:
                return self._buffer[frame_idx], True
            if self._buffer:
                # Nearest cached frame — avoids blocking the main thread on
                # large seeks while the preloader catches up.
                nearest_key = min(
                    self._buffer.keys(), key=lambda k: abs(k - frame_idx)
                )
                return self._buffer[nearest_key], False
        # Buffer empty — synchronous fallback (rare; only at startup)
        try:
            return self._source[frame_idx], True
        except Exception:
            return None, True

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

    Zoom is controlled continuously via the mouse wheel over the canvas
    (scroll up → zoom in, scroll down → zoom out) and via the ± spinbox.
    The spinbox and wheel stay in sync; zoom is clamped between 0.5 s and
    half the full recording length.
    """

    frame_requested = pyqtSignal(int)

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

        # Zoom state — zoomed ±30 s window is the default
        self._neural_fps: int = neural_fps
        self._zoom_half_window: int = 30.0 * neural_fps  # samples; must match _window_spinbox.setValue(30.0) below
        self._max_half_window: int = self._total_samples // 2
        self._current_sample: int = 0

        # Blit state — background snapshot for fast cursor updates
        self._bg_full = None          # copy_from_bbox snapshot (None = needs capture)
        self._bg_xlim: tuple = (0, 1) # xlim at which the snapshot was taken
        # Resnap threshold: centred=25 % drift from centre; edge-pan=5 % margin
        self._blit_threshold_frac: float = 0.25
        self._centered_mode: bool = False   # True = centred, False = edge-pan
        self._supports_blit: bool = True   # set False on non-blitting backends

        # --- Matplotlib figure ---
        self.fig = Figure(figsize=(12, 2.2))
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

        self._touch_bands_checkbox = QCheckBox("Touch bands")
        self._touch_bands_checkbox.setChecked(True)
        self._touch_bands_checkbox.setStyleSheet("font-size: 8pt;")
        self._touch_bands_checkbox.stateChanged.connect(
            lambda state: self._on_touch_bands_toggled(state == Qt.Checked)
        )
        btn_layout.addWidget(self._touch_bands_checkbox)

        self._centered_checkbox = QCheckBox("Centred")
        self._centered_checkbox.setChecked(self._centered_mode)
        self._centered_checkbox.setStyleSheet("font-size: 8pt;")
        self._centered_checkbox.stateChanged.connect(
            lambda state: self._on_centered_toggled(state == Qt.Checked)
        )
        btn_layout.addWidget(self._centered_checkbox)

        btn_layout.addWidget(QLabel("±"))
        self._window_spinbox = QDoubleSpinBox()
        self._window_spinbox.setMinimum(0.5)
        _max_secs = min(self._total_samples / self._neural_fps / 2.0, 3600.0)
        self._window_spinbox.setMaximum(_max_secs)
        self._window_spinbox.setSingleStep(0.5)
        self._window_spinbox.setValue(30.0)
        self._window_spinbox.setSuffix(" s")
        self._window_spinbox.setFixedWidth(70)
        self._window_spinbox.setFixedHeight(18)
        self._window_spinbox.setStyleSheet("font-size: 8pt;")
        self._window_spinbox.valueChanged.connect(self._on_zoom_window_changed)
        btn_layout.addWidget(self._window_spinbox)

        layout.addWidget(btn_row)

        self.canvas.installEventFilter(self)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('resize_event', self._invalidate_background)
        layout.addWidget(self.canvas)
        self.setFixedHeight(220)

        self._setup_axes(merged_df)

        # Extract touch boundaries from single_touch_id column if present
        self._touch_spans: List = []
        if 'single_touch_id' in merged_df.columns:
            touch_ids = merged_df['single_touch_id'].ffill().to_numpy(dtype=np.int64, na_value=0)
            self._touch_boundaries: List[tuple] = extract_touch_boundaries(touch_ids)
        else:
            self._touch_boundaries = []

        # Hide the checkbox when there are no touch boundaries to show
        if not self._touch_boundaries:
            self._touch_bands_checkbox.hide()

        self._draw_touch_bands()

    def _setup_axes(self, merged_df: pd.DataFrame) -> None:
        axes = self.fig.subplots(3, 1, sharex=True)
        self.ax_freq, self.ax_depth, self.ax_area = axes
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.10, hspace=0.10)

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
                ax.plot(x, merged_df[col].ffill(), color=color, lw=0.8)
            ax.set_ylabel(ylabel, color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            line = ax.axvline(x=0, color='red', lw=1.5, alpha=0.9)
            self._cursor_lines.append(line)

        self._capture_background()

    def _draw_touch_bands(self) -> None:
        """
        Draw alternating red/green semi-transparent background bands for each
        contiguous non-zero ``single_touch_id`` block on all three axes.

        Band colour alternates by touch order (index % 2), not by ID value:
        even-indexed touches → ``'#44ff44'`` (green), odd-indexed → ``'#ff4444'`` (red).
        Each span is extended by 33 samples past the last touch row to match the
        neural signal tail.
        All span artists are stored in ``self._touch_spans`` so they can be
        toggled via ``_on_touch_bands_toggled()``.

        No-op when ``self._touch_boundaries`` is empty.
        """
        axes = [self.ax_freq, self.ax_depth, self.ax_area]
        for idx, (start, end, _touch_id) in enumerate(self._touch_boundaries):
            color = '#44ff44' if idx % 2 == 0 else '#ff4444'
            for ax in axes:
                span = ax.axvspan(start, end + 33, color=color, alpha=0.12, zorder=0)
                self._touch_spans.append(span)

        if self._touch_spans:
            self._bg_full = None
            self.canvas.draw_idle()

    def _on_touch_bands_toggled(self, checked: bool) -> None:
        """Show or hide all touch-band span artists and refresh the canvas."""
        for artist in self._touch_spans:
            artist.set_visible(checked)
        self._bg_full = None
        self.canvas.draw_idle()

    def _on_centered_toggled(self, checked: bool) -> None:
        """Switch between centred and edge-pan scroll modes."""
        self._centered_mode = checked
        self._bg_full = None  # invalidate snapshot; next update_cursor recaptures

    # ------------------------------------------------------------------
    # Blit helpers
    # ------------------------------------------------------------------

    def _capture_background(self) -> None:
        """
        Take a full-canvas snapshot with cursor lines hidden.

        After snapshotting, the cursor lines are redrawn via the fast
        blit path so the canvas always shows the current cursor position.
        Records the xlim of the first axes at snapshot time in ``_bg_xlim``.
        """
        # Hide cursor lines
        for line in self._cursor_lines:
            line.set_visible(False)

        # Full draw without cursors
        self.canvas.draw()

        # Snapshot
        self._bg_full = self.canvas.copy_from_bbox(self.fig.bbox)
        self._bg_xlim = tuple(self.ax_freq.get_xlim())

        # Re-show cursor lines
        for line in self._cursor_lines:
            line.set_visible(True)

        # Draw cursor lines via blit
        for ax, line in zip(
            [self.ax_freq, self.ax_depth, self.ax_area],
            self._cursor_lines,
        ):
            ax.draw_artist(line)
        self.canvas.blit(self.fig.bbox)

    def _invalidate_background(self, *_) -> None:
        """Discard the cached background snapshot (e.g. on canvas resize)."""
        self._bg_full = None

    # ------------------------------------------------------------------
    # Cursor update — hot path
    # ------------------------------------------------------------------

    def update_cursor(self, frame_idx: int, scale_factor: float) -> None:
        """
        Move the red vertical cursor to *frame_idx* in merged-CSV sample space.

        Fast path: restore the cached background bitmap, draw only the cursor
        lines via ``ax.draw_artist`` + ``canvas.blit`` — ~1–3 ms per call.

        Full-redraw path: triggered when the cursor drifts outside the safe
        zone (configurable via ``_blit_threshold_frac``), when the zoom window
        changes, or when the canvas was resized.  After a full redraw a new
        snapshot is captured so subsequent frames hit the fast path.

        Mode ``_centered_mode=True``:  resnap when cursor drifts > 25 % from
        visible centre.
        Mode ``_centered_mode=False`` (edge-pan): resnap when cursor reaches
        within 5 % of the left or right edge of the current view.
        """
        sample_idx = int(frame_idx * scale_factor)
        self._current_sample = sample_idx

        # Update cursor line positions
        for line in self._cursor_lines:
            line.set_xdata([sample_idx])

        # Decide whether to shift xlim (and recapture) or use the fast path
        needs_resnap = False

        if self._bg_full is None:
            # No snapshot yet — compute xlim then capture
            needs_resnap = True
        else:
            lo_snap, hi_snap = self._bg_xlim
            span = hi_snap - lo_snap
            if span <= 0:
                needs_resnap = True
            elif self._centered_mode:
                centre_snap = (lo_snap + hi_snap) * 0.5
                drift_frac = abs(sample_idx - centre_snap) / (span * 0.5 + 1e-9)
                needs_resnap = drift_frac > self._blit_threshold_frac
            else:
                # Edge-pan mode: resnap only when cursor leaves a 5 % margin
                margin = 0.05 * span
                needs_resnap = sample_idx < lo_snap + margin or sample_idx > hi_snap - margin

        if needs_resnap:
            # Shift xlim to centre around the cursor, then capture
            lo = max(0, sample_idx - self._zoom_half_window)
            hi = min(self._total_samples - 1, sample_idx + self._zoom_half_window)
            self.ax_freq.set_xlim(lo, hi)
            self._capture_background()
            return

        # Fast blit path -------------------------------------------------------
        if not self._supports_blit:
            self.canvas.draw_idle()
            return
        try:
            self.canvas.restore_region(self._bg_full)
            for ax, line in zip(
                [self.ax_freq, self.ax_depth, self.ax_area],
                self._cursor_lines,
            ):
                ax.draw_artist(line)
            self.canvas.blit(self.fig.bbox)
        except AttributeError:
            # Backend does not support blitting — fall back permanently
            self._supports_blit = False
            self.canvas.draw_idle()

    def _on_zoom_window_changed(self, seconds: float) -> None:
        """Update the half-window size and refresh xlim."""
        self._zoom_half_window = int(seconds * self._neural_fps)
        lo = max(0, self._current_sample - self._zoom_half_window)
        hi = min(self._total_samples - 1, self._current_sample + self._zoom_half_window)
        self.ax_freq.set_xlim(lo, hi)
        self._bg_full = None
        self.canvas.draw_idle()

    def _on_canvas_click(self, event) -> None:
        """Convert a matplotlib left-click to a kinect frame and emit frame_requested."""
        if event.inaxes is None or event.button != 1:
            return
        sample_idx = event.xdata
        if sample_idx is None:
            return
        scale = self._total_samples / self._total_kinect_frames
        frame = int(round(sample_idx / scale))
        frame = max(0, min(frame, self._total_kinect_frames - 1))
        self.frame_requested.emit(frame)

    def eventFilter(self, obj, event) -> bool:
        """Intercept mouse-wheel events on the canvas for continuous zoom."""
        if obj is self.canvas and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            notches = delta / 120.0          # typically ±1 per detent
            factor = 1.25 ** (-notches)      # scroll-up shrinks, scroll-down grows
            new_half = int(self._zoom_half_window * factor)
            min_half = max(1, int(0.5 * self._neural_fps))
            new_half = max(min_half, min(new_half, self._max_half_window))
            self._zoom_half_window = new_half
            # Sync spinbox without re-triggering _on_zoom_window_changed
            self._window_spinbox.blockSignals(True)
            self._window_spinbox.setValue(new_half / self._neural_fps)
            self._window_spinbox.blockSignals(False)
            lo = max(0, self._current_sample - self._zoom_half_window)
            hi = min(self._total_samples - 1, self._current_sample + self._zoom_half_window)
            self.ax_freq.set_xlim(lo, hi)
            self._bg_full = None
            self.canvas.draw_idle()
            return True   # consume event — don't let Matplotlib handle it
        return False


# ---------------------------------------------------------------------------
# NeuralKinectViewer
# ---------------------------------------------------------------------------

class NeuralKinectViewer(QMainWindow):
    """
    High-performance standalone viewer for merged neural + Kinect recordings.

    Accepts an *all_blocks* dict ``{(session_id, block_id): NeuralKinectBlockSpec}``
    so that the user can navigate between blocks via toolbar dropdowns without
    closing the window (hot-swap).

    Architecture highlights
    ~~~~~~~~~~~~~~~~~~~~~~~
    - Named PyVista actors are updated in-place; ``plotter.clear()`` is never
      called, preserving camera state across frame changes.
    - A background ``FramePreloader`` thread maintains a 512-frame ring buffer
      of decoded ``PointCloudData`` objects to eliminate MKV seek latency.
    - ``HandMotionManager[i]`` is called lazily per frame (no eager pre-build).
    - CuPy AABB cropping reduces the rendered Kinect cloud by 8-16x.
    - The ``NeuralDataPanel`` and ``StickerVelocityCompass`` widgets are
      created only when *merged_csv_path* is provided.
    - Hot-swap: ``_teardown_current_block()`` → ``_load_block()`` swaps data
      without closing the plotter; named actors are re-seeded by
      ``_init_actors()`` after the swap.

    Parameters
    ----------
    all_blocks:
        Mapping ``{(session_id, block_id): NeuralKinectBlockSpec}`` for every
        block the viewer should be able to display.
    initial_session_id:
        Session dropdown value to display on first open.
    initial_block_id:
        Block dropdown value to display on first open.
    crop_half_size_mm:
        Half-width of the AABB crop box centred on the contact centroid (mm).
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        all_blocks: Dict[Tuple[str, str], "NeuralKinectBlockSpec"],
        initial_session_id: str,
        initial_block_id: str,
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
        # Basic immutable state
        # ------------------------------------------------------------------
        self._all_blocks: Dict[Tuple[str, str], NeuralKinectBlockSpec] = all_blocks
        self._crop_half_size: float = crop_half_size_mm
        self._cupy_available: bool = _CUPY_AVAILABLE
        self._preloader_buf_size: int = 512

        # Build session → [block_id, …] index (sorted for deterministic order)
        self._session_to_blocks: Dict[str, List[str]] = {}
        for (session_id, block_id) in all_blocks:
            self._session_to_blocks.setdefault(session_id, []).append(block_id)
        for session in self._session_to_blocks:
            self._session_to_blocks[session].sort()

        # ------------------------------------------------------------------
        # Per-block mutable state — populated by _load_block()
        # ------------------------------------------------------------------
        # These are set to sentinel values so that _teardown_current_block()
        # is safe to call before _load_block() has ever run.
        self._mkv = None
        self._preloader: Optional[FramePreloader] = None

        # ------------------------------------------------------------------
        # Persistent panel state — seeded once; survive session/block switches
        # ------------------------------------------------------------------
        self._visibility: Dict[str, bool] = {}
        self._point_sizes: Dict[str, float] = {
            'kinect_point_cloud': 2.0,
            'forearms': 8.0,
            'contact_points': 15.0,
        }
        self._interactive_stride: int = 4
        self._is_interactive: bool = False

        # Depth colouring is ON by default whenever the block carries a field:
        # the field is the more informative rendering, and flat colour exists
        # only as a comparison mode.  The preference persists across hot-swaps
        # like the visibility flags and point sizes beside it.
        self._colour_contact_by_depth: bool = True
        # Per-block; re-seeded by _load_block / _init_actors.  Declared here so
        # that an interim _update_frame fired during a hot-swap (before the new
        # block's actors exist) cannot hit an undefined attribute.
        self._depth_series: Optional[ContactDepthFieldSeries] = None
        self._contact_pts_by_frame: Optional[List[Optional[np.ndarray]]] = None
        self._contact_clim: Optional[Tuple[float, float]] = None

        # ------------------------------------------------------------------
        # Build the Qt UI (plotter + static right panel + frame controls)
        # ------------------------------------------------------------------
        self._build_ui()

        # ------------------------------------------------------------------
        # Timers (created once; re-used across hot-swaps)
        # ------------------------------------------------------------------
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        # Throttle timer for live frame updates during slider drag.
        # Fires every 33 ms (~30 fps) while the user holds the slider.
        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(33)
        self._pending_drag_frame: Optional[int] = None

        # Polling timer: fires every 50 ms after a cache-miss.
        self._exact_frame_timer = QTimer(self)
        self._exact_frame_timer.setInterval(50)
        self._exact_frame_timer.timeout.connect(self._on_exact_frame_poll)

        # Guard: first render is deferred to showEvent.
        self._initial_render_done: bool = False

        # ------------------------------------------------------------------
        # Load the initial block by driving the toolbar dropdowns.
        #
        # Setting the session combo index triggers _on_session_changed, which
        # populates the block combo and calls _on_block_changed, which calls
        # _load_block.  This ensures the dropdowns are in sync with the
        # displayed data from the very first open.
        # ------------------------------------------------------------------
        sorted_sessions = sorted(self._session_to_blocks)
        initial_session_index = sorted_sessions.index(initial_session_id) if initial_session_id in sorted_sessions else 0
        # Block signals during setup so we control exactly when _load_block fires.
        self._session_combo.blockSignals(True)
        self._block_combo.blockSignals(True)
        self._session_combo.setCurrentIndex(initial_session_index)
        self._block_combo.clear()
        for block_id in self._session_to_blocks.get(initial_session_id, []):
            self._block_combo.addItem(block_id)
        blocks_for_session = self._session_to_blocks.get(initial_session_id, [])
        initial_block_index = blocks_for_session.index(initial_block_id) if initial_block_id in blocks_for_session else 0
        self._block_combo.setCurrentIndex(initial_block_index)
        self._session_combo.blockSignals(False)
        self._block_combo.blockSignals(False)
        # Now load explicitly — signals are unblocked so future user interactions work.
        self._load_block(initial_session_id, initial_block_id)

    # ======================================================================
    # Block lifecycle — teardown + load
    # ======================================================================

    def _teardown_current_block(self) -> None:
        """Stop timers, preloader, and close the current MKV handle."""
        self._play_timer.stop()
        self._drag_timer.stop()
        self._exact_frame_timer.stop()

        if self._preloader is not None:
            self._preloader.stop()
            self._preloader.join(timeout=2.0)
            self._preloader = None

        if self._mkv is not None:
            try:
                self._mkv.__exit__(None, None, None)
            except Exception:
                pass
            self._mkv = None

    def _load_block(self, session_id: str, block_id: str) -> None:
        """
        Load all data for the given block and reinitialise the viewer state.

        Raises ``KeyError`` immediately if ``(session_id, block_id)`` is not
        present in ``self._all_blocks`` (fail-fast).
        """
        spec: NeuralKinectBlockSpec = self._all_blocks[(session_id, block_id)]

        # ------------------------------------------------------------------
        # 1. Load stickers
        # ------------------------------------------------------------------
        stickers_df_dict = XYZDataFileHandler.load(spec.xyz_csv_path)
        self._custom_colors: Dict[str, str] = define_custom_colors(stickers_df_dict.keys())
        self._stickers_xyz_dict: Dict[str, np.ndarray] = {
            k: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
            for k, df in stickers_df_dict.items()
        }

        # ------------------------------------------------------------------
        # 2. Load forearm
        # ------------------------------------------------------------------
        forearm_params = ForearmFrameParametersFileHandler.load(spec.forearm_metadata_path)
        catalog = ForearmCatalog(forearm_params, spec.forearm_pointcloud_dir)
        self._forearms_dict = get_forearms_with_fallback(catalog, str(spec.rgb_video_path))
        self._sorted_forearm_keys: List[int] = sorted(self._forearms_dict.keys())

        # ------------------------------------------------------------------
        # 3. Load hand motion (LAZY — manager stored, not an eager loop)
        # ------------------------------------------------------------------
        try:
            self._hand_manager = HandMotionManager()
            self._hand_manager.load(str(spec.hand_motion_path))
        except Exception as exc:
            print(f"Warning: Could not load hand motion data ({exc}). Hand mesh disabled.")
            self._hand_manager = None
        self._last_hand_frame: int = -1
        self._last_hand_mesh = None
        self._last_hand_tri_count: int = -1

        # ------------------------------------------------------------------
        # 4. Optional merged CSV
        # ------------------------------------------------------------------
        self.merged_df: Optional[pd.DataFrame] = (
            pd.read_csv(spec.merged_csv_path) if spec.merged_csv_path is not None else None
        )

        # ------------------------------------------------------------------
        # 4a. Contact depth field (columnar; needs no parsing of any kind)
        #
        # Taken verbatim from the spec.  ``None`` means the producer looked for
        # the sidecar and did not find it; it has already reported that.  When
        # present this series — not the CSV blob — supplies the contact
        # geometry, so the drawn points and the drawn depths cannot disagree.
        # ------------------------------------------------------------------
        self._depth_series: Optional[ContactDepthFieldSeries] = spec.contact_depth_field

        # ------------------------------------------------------------------
        # 4b. Pre-extract contact points indexed by kinect frame
        # ------------------------------------------------------------------
        self._contact_pts_by_frame: Optional[List[Optional[np.ndarray]]] = None
        if (
            self.merged_df is not None
            and 'contact_points' in self.merged_df.columns
            and 'time_kinect' in self.merged_df.columns
        ):
            kinect_rows = self.merged_df.dropna(subset=['time_kinect'])
            self._contact_pts_by_frame = [
                _parse_contact_points_cell(cell)
                for cell in kinect_rows['contact_points']
            ]

        # ------------------------------------------------------------------
        # 5. Contact centroid (used for GPU crop centre)
        # ------------------------------------------------------------------
        self._transforms_by_forearm_key: Optional[Dict[int, np.ndarray]] = (
            spec.registration_transforms_by_forearm_key or None
        )
        self.contact_centroid: np.ndarray = self._compute_contact_centroid()

        _T0: Optional[np.ndarray] = None
        if self._transforms_by_forearm_key:
            _T0 = self._transforms_by_forearm_key.get(
                min(self._transforms_by_forearm_key)
            )
        if _T0 is not None:
            _c = apply_rigid_transform(
                self.contact_centroid.reshape(1, 3), _T0
            )[0]
            self._camera_focal_point: np.ndarray = _c
        else:
            self._camera_focal_point = self.contact_centroid

        # Invalidate the CuPy crop-centre cache so it is re-uploaded for the
        # new block (the centroid changes between blocks).
        self._c_gpu = None

        # ------------------------------------------------------------------
        # 6. Open MKV (manual context management — closed in _teardown)
        # ------------------------------------------------------------------
        self._mkv = KinectMKV(spec.kinect_mkv_path).__enter__()
        self._point_cloud_view = KinectPointCloudView(self._mkv)
        self._total_frames: int = len(self._point_cloud_view)

        # ------------------------------------------------------------------
        # 7. Background preloader (512-frame ring buffer)
        # ------------------------------------------------------------------
        self._preloader = FramePreloader(self._point_cloud_view, buffer_size=self._preloader_buf_size)
        self._preloader.seek(0)
        self._preloader.start()
        if not self._visibility.get('kinect_point_cloud', True):
            self._preloader.pause()

        # ------------------------------------------------------------------
        # 8. Reset per-block interaction state
        # ------------------------------------------------------------------
        self.current_index: int = 0
        self._slider_dragging: bool = False
        self._exact_frame_pending: Optional[int] = None
        self._recording_name: str = spec.recording_name

        # Cache of last-rendered sticker positions, keyed by sticker name.
        # Used in _update_frame to detect when a position actually changes.
        self._last_sticker_pos: Dict[str, np.ndarray] = {}

        # Contact-point identity cache — skip DeepCopy when the frame's
        # contact data is unchanged since the last render.
        self._last_contact_frame: int = -1
        self._last_contact_empty: bool = True

        # ------------------------------------------------------------------
        # 9. Update frame slider range + window title
        # ------------------------------------------------------------------
        self.frame_slider.setRange(0, max(self._total_frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"1 / {self._total_frames}")
        self._buffer_label.setText(f"Buf: 0/{self._preloader_buf_size}")
        self.crop_spinbox.blockSignals(True)
        self.crop_spinbox.setValue(int(self._crop_half_size))
        self.crop_spinbox.blockSignals(False)
        self.lod_spinbox.blockSignals(True)
        self.lod_spinbox.setValue(self._interactive_stride)
        self.lod_spinbox.blockSignals(False)
        self.setWindowTitle(f"Neural-Kinect Viewer | {spec.recording_name}")

        # ------------------------------------------------------------------
        # 10. Rebuild data-dependent right-panel widgets
        #     (clear indices 1+ to keep the camera monitor at index 0)
        # ------------------------------------------------------------------
        while self._right_panel_layout.count() > 1:
            item = self._right_panel_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._build_right_panel_data()

        # ------------------------------------------------------------------
        # 11. Replace neural panel
        # ------------------------------------------------------------------
        if hasattr(self, 'neural_panel') and self.neural_panel is not None:
            self._outer_layout.removeWidget(self.neural_panel)
            self.neural_panel.deleteLater()
            self.neural_panel = None

        if self.merged_df is not None:
            self.neural_panel = NeuralDataPanel(self.merged_df, self._total_frames)
            self.neural_panel.frame_requested.connect(self.frame_slider.setValue)
            self._outer_layout.addWidget(self.neural_panel)
        else:
            self.neural_panel = None

        # ------------------------------------------------------------------
        # 12. Reset named VTK actor slots
        # ------------------------------------------------------------------
        self._init_actors()

        # ------------------------------------------------------------------
        # 13. Deferred render (if already shown)
        # ------------------------------------------------------------------
        if self._initial_render_done:
            QTimer.singleShot(0, lambda: self._update_frame(0))

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self) -> None:
        """
        Build the Qt widget hierarchy once.

        Static part: toolbar (session + block dropdowns), plotter, camera
        monitor, frame controls.  Data-dependent right-panel widgets are
        added later by ``_build_right_panel_data()`` inside ``_load_block()``.
        """
        # --- Toolbar with session / block dropdowns ---
        toolbar = QToolBar("Session controls")
        toolbar.setMovable(False)

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

        self.addToolBar(toolbar)

        # --- Central widget ---
        central = QWidget()
        self.setCentralWidget(central)
        self._outer_layout = QVBoxLayout(central)

        # --- Top row: 3D plotter + right panel ---
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(plotter_widget)
        self.plotter = QtInteractor(plotter_widget)
        self.plotter.set_background('black')
        plotter_layout.addWidget(self.plotter.interactor)
        top_layout.addWidget(plotter_widget, stretch=4)

        # Register once: refresh camera label whenever the user ends a rotate/pan
        self._cam_end_observer_tag = self.plotter.iren.add_observer(
            'EndInteractionEvent', self._refresh_cam_pos_label
        )

        # Right panel (scrollable, fixed 220 px)
        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout(self._right_panel)
        self._build_right_panel_static()  # camera monitor only (index 0)
        scroll = QScrollArea()
        scroll.setWidget(self._right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)
        top_layout.addWidget(scroll, stretch=1)

        self._outer_layout.addWidget(top_widget, stretch=4)

        # --- Middle row: frame controls ---
        self._outer_layout.addWidget(self._build_frame_controls())

        # neural_panel placeholder — will be set by _load_block()
        self.neural_panel: Optional[NeuralDataPanel] = None

    def _build_right_panel_static(self) -> None:
        """
        Add the static camera-position readout (index 0 in the right panel).

        Called once from ``_build_ui()``.  All subsequent items (visibility
        checkboxes, sticker controls, compasses) are added by
        ``_build_right_panel_data()`` and cleared/rebuilt on each hot-swap.

        Note: no trailing stretch is added here.  ``_build_right_panel_data()``
        adds the stretch at the end of the data items so it is naturally
        removed and re-added on each hot-swap.
        """
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

    @property
    def _has_contact_source(self) -> bool:
        """Whether this block has any source of contact geometry at all.

        Either the depth field sidecar or the merged CSV's ``contact_points``
        column will do; the field takes precedence when both are present.
        """
        return self._depth_series is not None or self._contact_pts_by_frame is not None

    @property
    def _depth_colouring_active(self) -> bool:
        """Whether contact vertices are currently coloured by penetration depth.

        Requires both a field to colour by and the user's checkbox.  These are
        different facts: the first is about the data, the second about the view.
        """
        return self._depth_series is not None and self._colour_contact_by_depth

    def _build_right_panel_data(self) -> None:
        """
        Populate the data-dependent portion of the right panel.

        Called from ``_load_block()`` after clearing indices 1+ (the camera
        monitor at index 0 is preserved).  Adds visibility checkboxes, point-
        size sliders, and ``StickerVelocityCompass`` widgets for the current
        block.

        A trailing stretch is appended at the very end so the widgets stay
        top-aligned.  On each hot-swap, ``_load_block()`` removes all items
        from index 1 onward (including the previous stretch), then this method
        re-adds the new items + a fresh stretch.
        """
        # Register any keys not yet in the persisted dicts (new block/session
        # may introduce sticker names that weren't seen before).
        _canonical_keys = (
            ['kinect_point_cloud', 'forearms', 'hand_meshes', 'contact_points']
            + list(self._stickers_xyz_dict.keys())
        )
        for _key in _canonical_keys:
            self._visibility.setdefault(_key, True)
        self._point_sizes.setdefault('kinect_point_cloud', 2.0)
        self._point_sizes.setdefault('forearms', 8.0)
        self._point_sizes.setdefault('contact_points', 15.0)
        self._compass_widgets: Dict[str, StickerVelocityCompass] = {}

        def _add_object_group(
            label: str,
            key: str,
            has_slider: bool = False,
            point_size: int = 3,
            extra_widgets: Tuple[QWidget, ...] = (),
        ):
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)

            cb = QCheckBox("Visible")
            cb.setChecked(self._visibility.get(key, True))
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
                sl.setValue(int(self._point_sizes.get(key, point_size)))
                sl.valueChanged.connect(
                    lambda val, k=key: self._on_point_size_changed(k, val)
                )
                slider_layout.addWidget(sl)
                box_layout.addWidget(slider_row)

            for extra in extra_widgets:
                box_layout.addWidget(extra)

            self._right_panel_layout.addWidget(box)

        _add_object_group("Kinect Cloud",    "kinect_point_cloud", has_slider=True)
        _add_object_group("Forearms",        "forearms",           has_slider=True)
        _add_object_group("Hand Mesh",       "hand_meshes",        has_slider=False)
        if self._has_contact_source:
            # "Colour by depth" sits beside the point-size slider so flat colour
            # stays one click away for comparison.  The checkbox is created only
            # when a field exists: offering a control that could not do anything
            # would suggest the data is there when it is not.
            _contact_extras: Tuple[QWidget, ...] = ()
            if self._depth_series is not None:
                depth_cb = QCheckBox("Colour by depth")
                depth_cb.setChecked(self._colour_contact_by_depth)
                low, high = self._depth_series.clim_penetration_mm
                depth_cb.setToolTip(
                    "Colour contact vertices by penetration depth (inferno), on a "
                    f"colour scale fixed over the whole recording: {low:.2f} to "
                    f"{high:.2f} mm. Unchecked renders them in flat red."
                )
                depth_cb.stateChanged.connect(self._on_contact_depth_colour_changed)
                _contact_extras = (depth_cb,)
            _add_object_group(
                "Contact Points",
                "contact_points",
                has_slider=True,
                extra_widgets=_contact_extras,
            )

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

        # Trailing stretch — keeps widgets top-aligned.  Removed with the rest
        # of the data items on hot-swap and re-added by the next call to this method.
        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        """Return a QWidget row containing slider, labels, and control buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("1 / 0")
        self.frame_label.setFixedWidth(100)
        layout.addWidget(self.frame_label)

        self._buffer_label = QLabel(f"Buf: 0/{self._preloader_buf_size}")
        self._buffer_label.setFixedWidth(75)
        self._buffer_label.setStyleSheet("font-family: monospace; font-size: 8pt; color: gray;")
        layout.addWidget(self._buffer_label)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)

        self._speed_spinbox = QDoubleSpinBox()
        self._speed_spinbox.setRange(0.1, 10.0)
        self._speed_spinbox.setSingleStep(0.1)
        self._speed_spinbox.setValue(1.0)
        self._speed_spinbox.setDecimals(1)
        self._speed_spinbox.setPrefix("x")
        self._speed_spinbox.setFixedWidth(65)
        self._speed_spinbox.setToolTip("Playback speed multiplier (1.0 = native FPS)")
        layout.addWidget(self._speed_spinbox)

        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._recenter_view)
        layout.addWidget(recenter_btn)

        layout.addWidget(QLabel("Crop ±"))
        self.crop_spinbox = QSpinBox()
        self.crop_spinbox.setMinimum(100)
        self.crop_spinbox.setMaximum(2000)
        self.crop_spinbox.setSingleStep(50)
        self.crop_spinbox.setValue(int(self._crop_half_size))
        self.crop_spinbox.valueChanged.connect(self._on_crop_changed)
        layout.addWidget(self.crop_spinbox)

        layout.addWidget(QLabel("LOD"))
        self.lod_spinbox = QSpinBox()
        self.lod_spinbox.setMinimum(1)
        self.lod_spinbox.setMaximum(8)
        self.lod_spinbox.setValue(4)
        self.lod_spinbox.setToolTip(
            "Point-cloud stride during interaction (1 = full resolution, "
            "higher = faster but sparser)"
        )
        self.lod_spinbox.valueChanged.connect(self._on_lod_changed)
        layout.addWidget(self.lod_spinbox)

        return widget

    # ======================================================================
    # Actor initialisation
    # ======================================================================

    def _init_actors(self) -> None:
        """
        Add every named actor once so that subsequent in-place mutations of
        the persistent ``PolyData`` objects update the scene without touching
        the camera.

        Called from ``_load_block()`` on each hot-swap.  Named actors are
        replaced in-place by PyVista when the same ``name=`` is reused, so the
        camera is preserved across block transitions.
        """
        _seed = np.zeros((1, 3), dtype=np.float32)
        _seed_col = np.zeros((1, 3), dtype=np.uint8)
        self._mesh_kinect = pv.PolyData(_seed.copy())
        self._mesh_kinect['colors'] = _seed_col.copy()
        self._mesh_forearm = pv.PolyData(_seed.copy())
        self._mesh_forearm['colors'] = _seed_col.copy()
        self._mesh_hand = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        self._mesh_contact = _empty_contact_polydata()
        # Force a full DeepCopy on the next _update_frame call so the freshly
        # created PolyData objects (no faces, no points) get populated correctly.
        # Without this reset, an interim _update_frame fired by frame_slider.setValue(0)
        # during _load_block sets these counters before _init_actors replaces the
        # PolyData objects, causing the deferred render to take the no-op path.
        self._last_hand_tri_count: int = -1
        self._last_contact_frame: int = -1
        self._last_contact_empty: bool = True

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

        # Dynamic actors — registered once (or replaced by name), mutated in-place.
        self._actor_kinect = self.plotter.add_mesh(
            self._mesh_kinect,
            scalars='colors',
            rgb=True,
            name='kinect_point_cloud',
            render_points_as_spheres=False,
            point_size=self._point_sizes['kinect_point_cloud'],
        )
        self._actor_forearm = self.plotter.add_mesh(
            self._mesh_forearm,
            scalars='colors',
            rgb=True,
            name='forearms',
            render_points_as_spheres=True,
            point_size=self._point_sizes['forearms'],
        )
        self.plotter.add_mesh(self._mesh_hand, name='hand_meshes', style='wireframe')

        # --- Contact points -------------------------------------------------
        # Registered ONCE per block load.  The colour range is the producer's,
        # computed over the whole recording; nothing here derives it.  The
        # previous block's scalar bar is removed first: two blocks have
        # different global ranges, and a bar shared between their mappers would
        # label one of them wrongly.
        if CONTACT_DEPTH_SCALAR_BAR_TITLE in self.plotter.scalar_bars:
            self.plotter.remove_scalar_bar(CONTACT_DEPTH_SCALAR_BAR_TITLE)

        self._contact_clim: Optional[Tuple[float, float]] = (
            self._depth_series.clim_penetration_mm
            if self._depth_series is not None else None
        )
        if self._contact_clim is None:
            self._actor_contact = self.plotter.add_mesh(
                self._mesh_contact,
                name='contact_points',
                color='red',
                render_points_as_spheres=True,
                point_size=self._point_sizes['contact_points'],
            )
        else:
            self._actor_contact = self.plotter.add_mesh(
                self._mesh_contact,
                name='contact_points',
                scalars=CONTACT_SCALAR_NAME,
                cmap=CONTACT_DEPTH_COLORMAP,
                clim=self._contact_clim,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': CONTACT_DEPTH_SCALAR_BAR_TITLE,
                    'vertical': True,
                    'n_labels': 6,
                    'fmt': '%.2f',
                    'title_font_size': 16,
                    'label_font_size': 13,
                    # Explicit white: the theme default is black, which is
                    # invisible against this viewer's dark background — the bar
                    # renders but its title and ticks do not.
                    'color': 'white',
                    'position_x': 0.88,
                    'position_y': 0.12,
                    'width': 0.05,
                    'height': 0.72,
                },
                render_points_as_spheres=True,
                point_size=self._point_sizes['contact_points'],
            )
            # Flat red is what the mapper falls back to when scalar visibility
            # is switched off, so it must be set even in depth-colouring mode.
            self._actor_contact.GetProperty().SetColor(1.0, 0.0, 0.0)
            self._apply_contact_scalar_mode()

        # Register ALL stickers and cache actor refs.
        self._sticker_actors: Dict[str, Any] = {}
        _small_sticker_colors = {'blue', 'green', 'yellow'}
        for name in self._stickers_xyz_dict:
            color = self._custom_colors.get(name, 'magenta')
            radius = 4.0 / 3.0 if color in _small_sticker_colors else 4.0
            sphere = pv.Sphere(radius=radius, center=(0.0, 0.0, 0.0))
            actor = self.plotter.add_mesh(sphere, color=color, name=f'sticker_{name}')
            self._sticker_actors[name] = actor

        # Default camera
        cx, cy, cz = self._camera_focal_point.tolist()
        self.plotter.camera.focal_point = [cx, cy, cz]
        self.plotter.camera.position    = [cx, cy, cz - 400.0]
        self.plotter.camera.up          = [0.375, -0.904, -0.201]
        self.plotter.camera_set = True

        # Invisible bounding proxy
        self._bounds_proxy_active = True
        _bounds_proxy = pv.Box(bounds=(
            cx - self._crop_half_size, cx + self._crop_half_size,
            cy - self._crop_half_size, cy + self._crop_half_size,
            cz - self._crop_half_size, cz + self._crop_half_size,
        ))
        self.plotter.add_mesh(
            _bounds_proxy, opacity=0.001, name='_bounds_proxy', pickable=False,
        )

        # Also reset the last-forearm-key sentinel so the forearm is redrawn
        self._last_forearm_key: Any = object()

        # Neural scale: ratio of merged-CSV rows to kinect frames.
        # Must be set here (not only in _deferred_start) so that hot-swap
        # renders triggered by _load_block use the correct scale for the
        # new block.
        self._neural_scale: float = (
            len(self.merged_df) / self._total_frames
            if self.merged_df is not None and self._total_frames > 0
            else 1.0
        )

    # ======================================================================
    # Frame update — the hot path
    # ======================================================================

    def _update_frame(self, frame_idx: int) -> None:
        """
        Update all dynamic actors for *frame_idx* and render exactly once.

        All five dynamic sections use in-place mutation rather than
        ``plotter.add_mesh()``:

        1. Kinect cloud  — ``self._mesh_kinect.DeepCopy(new_cloud)``
        2. Forearm       — ``self._mesh_forearm.DeepCopy(new_cloud)``
           (only when the bisect forearm key changes)
        3. Hand mesh     — ``self._mesh_hand.points = verts + Modified()``
           when topology is unchanged; ``overwrite()`` when it changes
        4. Stickers      — ``actor.SetPosition(*pos) / VisibilityOn/Off()``
        5. Contact pts   — ``self._mesh_contact.DeepCopy(new_cloud)``

        Camera state is preserved because ``plotter.clear()`` is never called.
        ``plotter.render()`` at the end propagates all VTK Modified() flags.
        """
        self.current_index = frame_idx

        # Dirty flags — set by each per-actor block when data actually changes.
        # dirty:       at least one actor was modified → need plotter.render()
        # bounds_dirty: geometry bounds changed (empty↔non-empty) → need
        #               ResetCameraClippingRange() before render.
        dirty: bool = False
        bounds_dirty: bool = False

        # Remove the invisible bounding proxy once a real kinect frame is available.
        if self._bounds_proxy_active and self._visibility.get('kinect_point_cloud', True):
            pc_data, _ = self._preloader.get_frame(frame_idx)
            if (
                pc_data is not None
                and pc_data.points is not None
                and pc_data.points.shape[0] > 0
            ):
                self._bounds_proxy_active = False
                try:
                    self.plotter.remove_actor('_bounds_proxy')
                except Exception:
                    pass

        # Resolve active forearm snapshot and its transform for this frame.
        forearm_key = self._bisect_forearm(frame_idx)
        T: Optional[np.ndarray] = (
            self._transforms_by_forearm_key.get(forearm_key)
            if self._transforms_by_forearm_key
            else None
        )

        # 1. Kinect point cloud (GPU-cropped AABB) --------------------------
        _kcloud: Optional[pv.PolyData] = None
        _got_exact: bool = True
        if self._visibility.get('kinect_point_cloud', True):
            pc_data, _got_exact = self._preloader.get_frame(frame_idx)
            if (
                pc_data is not None
                and pc_data.points is not None
                and pc_data.points.shape[0] > 0
            ):
                pts, cols = self._crop_pointcloud_gpu(
                    pc_data.points, pc_data.color,
                    self.contact_centroid, self._crop_half_size,
                )
                if T is not None and len(pts) > 0:
                    pts = apply_rigid_transform(
                        pts.astype(np.float64), T
                    ).astype(pts.dtype)
                if self._is_interactive and self._interactive_stride > 1:
                    s = self._interactive_stride
                    pts = pts[::s]
                    if cols is not None:
                        cols = cols[::s]
                if pts.shape[0] > 0:
                    _kcloud = pv.PolyData(pts.astype(np.float32))
                    _kcloud['colors'] = (
                        cols.astype(np.uint8) if cols is not None
                        else np.full((len(pts), 3), 128, dtype=np.uint8)
                    )
        if _kcloud is None:
            _kcloud = pv.PolyData(np.empty((0, 3), dtype=np.float32))
        # Only DeepCopy (and mark dirty) when the cloud or the current mesh
        # has real content — avoids a wasted copy of two empty PolyDatas.
        if _kcloud.n_points > 0 or self._mesh_kinect.n_points > 0:
            self._mesh_kinect.DeepCopy(_kcloud)
            dirty = True
            bounds_dirty = True

        # 2. Forearm (updated only when the bisect key changes) -------------
        if not self._visibility.get('forearms', True):
            _fa_empty = pv.PolyData(np.empty((0, 3), dtype=np.float32))
            self._mesh_forearm.DeepCopy(_fa_empty)
            self._last_forearm_key = object()  # force rebuild when re-enabled
            dirty = True
            bounds_dirty = True
        elif forearm_key != getattr(self, '_last_forearm_key', object()):
            self._last_forearm_key = forearm_key
            o3d_pc = self._forearms_dict.get(forearm_key)
            _fa_cloud: Optional[pv.PolyData] = None
            if o3d_pc is not None and o3d_pc.has_points():
                pts_fa = np.asarray(o3d_pc.points, dtype=np.float32)
                if T is not None:
                    pts_fa = apply_rigid_transform(
                        pts_fa.astype(np.float64), T
                    ).astype(np.float32)
                _fa_cloud = pv.PolyData(pts_fa)
                if o3d_pc.has_colors():
                    _fa_cloud['colors'] = (
                        np.asarray(o3d_pc.colors) * 255
                    ).astype(np.uint8)
                else:
                    _fa_cloud['colors'] = np.full(
                        (len(pts_fa), 3), 128, dtype=np.uint8
                    )
            if _fa_cloud is None:
                _fa_cloud = pv.PolyData(np.empty((0, 3), dtype=np.float32))
            self._mesh_forearm.DeepCopy(_fa_cloud)
            dirty = True
            bounds_dirty = True

        # 3. Hand mesh (lazy per-frame transform) ---------------------------
        if not self._visibility.get('hand_meshes', True):
            if self._last_hand_tri_count != 0:
                self._mesh_hand.DeepCopy(pv.PolyData(np.empty((0, 3), dtype=np.float32)))
                self._last_hand_tri_count = 0
                dirty = True
                bounds_dirty = True
        else:
            o3d_mesh = self._get_hand_mesh(frame_idx)
            if o3d_mesh is not None and o3d_mesh.has_triangles():
                verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
                if T is not None and len(verts) > 0:
                    verts = apply_rigid_transform(
                        verts.astype(np.float64), T
                    ).astype(np.float32)
                tris = np.asarray(o3d_mesh.triangles)
                n_tris = len(tris)
                if n_tris == self._last_hand_tri_count:
                    self._mesh_hand.points = verts
                    self._mesh_hand.Modified()
                    dirty = True
                else:
                    faces = np.hstack([
                        np.full((n_tris, 1), 3, dtype=tris.dtype), tris
                    ])
                    self._mesh_hand.DeepCopy(pv.PolyData(verts, faces))
                    self._last_hand_tri_count = n_tris
                    dirty = True
                    bounds_dirty = True
            else:
                if self._last_hand_tri_count != 0:
                    self._mesh_hand.DeepCopy(
                        pv.PolyData(np.empty((0, 3), dtype=np.float32))
                    )
                    self._last_hand_tri_count = 0
                    dirty = True
                    bounds_dirty = True

        # 4. Stickers + compass widgets -------------------------------------
        for name, positions in self._stickers_xyz_dict.items():
            pos = positions[frame_idx] if frame_idx < len(positions) else None
            if (
                T is not None
                and pos is not None
                and not np.any(np.isnan(pos))
            ):
                pos = apply_rigid_transform(
                    pos.reshape(1, 3).astype(np.float64), T
                )[0]
            actor = self._sticker_actors.get(name)

            if actor is not None:
                valid_pos = pos is not None and not np.any(np.isnan(pos))
                desired_visible = valid_pos and self._visibility.get(name, True)
                was_visible = actor.GetVisibility() != 0

                if desired_visible:
                    # Compare new position against last rendered position.
                    prev = self._last_sticker_pos.get(name)
                    pos_changed = (
                        prev is None
                        or not np.allclose(prev, pos, equal_nan=True)
                    )
                    if pos_changed or not was_visible:
                        actor.SetPosition(*pos.tolist())
                        actor.VisibilityOn()
                        self._last_sticker_pos[name] = pos.copy()
                        dirty = True
                else:
                    if was_visible:
                        actor.VisibilityOff()
                        self._last_sticker_pos.pop(name, None)
                        dirty = True

            if name in self._compass_widgets and frame_idx > 0:
                prev_pos = positions[frame_idx - 1]
                if (
                    T is not None
                    and not np.any(np.isnan(prev_pos))
                ):
                    prev_pos = apply_rigid_transform(
                        prev_pos.reshape(1, 3).astype(np.float64),
                        T,
                    )[0]
                if not np.any(np.isnan(prev_pos)) and pos is not None and not np.any(np.isnan(pos)):
                    self._compass_widgets[name].update_velocity(pos - prev_pos)

        # 5. Contact points -------------------------------------------------
        # Source precedence: the depth field when the block has one, otherwise
        # the CSV's pre-parsed contact_points.  Never a mix — the two are
        # different point sets (the CSV blob is quantised to 0.1 mm) and pairing
        # them would mean matching rows by coordinate value, which the sidecar's
        # design record explicitly forbids.
        if self._has_contact_source:
            _cpts: Optional[np.ndarray] = None
            _cdepths: Optional[np.ndarray] = None
            if self._visibility.get('contact_points', True):
                if self._depth_series is not None:
                    _pair = self._depth_series.frame(frame_idx)
                    if _pair is not None:
                        _cpts, _cdepths = _pair
                elif frame_idx < len(self._contact_pts_by_frame):
                    _cpts = self._contact_pts_by_frame[frame_idx]
            _contact_is_empty = _cpts is None or len(_cpts) == 0

            if frame_idx == self._last_contact_frame:
                # Same frame revisited (e.g. exact-frame poll) — skip DeepCopy.
                pass
            elif _contact_is_empty and self._last_contact_empty:
                # Transitioning empty → empty: nothing to update, no dirty set.
                self._last_contact_frame = frame_idx
            else:
                # Data actually changed (or empty↔non-empty transition).
                if not _contact_is_empty:
                    if T is not None:
                        # A rigid transform moves the vertices; it cannot change
                        # a distance, so _cdepths is carried through untouched.
                        _cpts = apply_rigid_transform(
                            _cpts.astype(np.float64), T
                        ).astype(np.float32)
                    self._mesh_contact.DeepCopy(
                        contact_polydata(_cpts, _cdepths)
                    )
                else:
                    self._mesh_contact.DeepCopy(_empty_contact_polydata())
                # Re-asserted after every dataset swap: without it the mapper
                # reverts to per-frame autoscale on PyVista 0.47.1 and the same
                # depth would take a different colour on a different frame.
                if self._contact_clim is not None:
                    self._actor_contact.mapper.scalar_range = self._contact_clim
                dirty = True
                if _contact_is_empty != self._last_contact_empty:
                    # Bounds changed: empty↔non-empty transition.
                    bounds_dirty = True
                self._last_contact_frame = frame_idx
                self._last_contact_empty = _contact_is_empty

        # 6. Single render call (gated on dirty flag) -----------------------
        if dirty:
            if bounds_dirty:
                self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()
        if not (self._is_interactive or self._play_timer.isActive()):
            self._refresh_cam_pos_label()

        # 7. Neural panel cursor --------------------------------------------
        if self.neural_panel is not None:
            self.neural_panel.update_cursor(frame_idx, self._neural_scale)

        # 8. Frame label + buffer fill indicator ----------------------------
        self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")
        buf_n = self._preloader.buffer_count()
        buf_max = self._preloader._buffer_size
        if not self._play_timer.isActive() or (frame_idx % 5 == 0):
            self._buffer_label.setText(f"Buf: {buf_n}/{buf_max}")

        # 9. Signal preloader to look ahead (skip when cloud is paused) -----
        if self._visibility.get('kinect_point_cloud', True):
            self._preloader.seek(frame_idx + 1)

        # 10. Schedule a re-render if we displayed an approximate frame -----
        # When the cloud is hidden _got_exact stays True (default line above),
        # so this branch is naturally skipped — no extra guard needed.
        if not _got_exact:
            self._schedule_exact_frame(frame_idx)

    # ======================================================================
    # Supporting methods
    # ======================================================================

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

        all_xyz = np.concatenate(list(self._stickers_xyz_dict.values()), axis=0)
        return np.nanmean(all_xyz, axis=0)

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
                if self._c_gpu is None:
                    self._c_gpu = cp.asarray(center.astype(np.float32))
                c_gpu = self._c_gpu
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

    # ======================================================================
    # Slot handlers — dropdowns
    # ======================================================================

    def _on_session_changed(self, index: int) -> None:
        """Repopulate the block combo for the newly-selected session."""
        if not hasattr(self, '_all_blocks'):
            return
        session_id = self._session_combo.currentText()
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for block_id in self._session_to_blocks.get(session_id, []):
            self._block_combo.addItem(block_id)
        self._block_combo.blockSignals(False)
        self._block_combo.setCurrentIndex(0)
        self._on_block_changed(0)

    def _on_block_changed(self, index: int) -> None:
        """Tear down the current block and load the newly-selected one."""
        if not hasattr(self, '_all_blocks'):
            return
        session_id = self._session_combo.currentText()
        block_id = self._block_combo.currentText()
        if not session_id or not block_id:
            return
        self._teardown_current_block()
        self._load_block(session_id, block_id)
        # Trigger a deferred render now that _initial_render_done is True
        # (_load_block does this itself; this call is a belt-and-suspenders guard).

    # ======================================================================
    # Slot handlers — frame controls
    # ======================================================================

    def _on_slider_pressed(self) -> None:
        """Mark the start of a user drag and start the throttle timer."""
        self._is_interactive = True
        self._slider_dragging = True
        self._pending_drag_frame = None
        self._drag_timer.start()

    def _on_slider_released(self) -> None:
        """On drag release, stop the throttle timer and do a final full render."""
        self._is_interactive = False
        self._slider_dragging = False
        self._drag_timer.stop()
        self._pending_drag_frame = None
        self._update_frame(self.frame_slider.value())
        self._refresh_cam_pos_label()

    def _on_drag_timer_fired(self) -> None:
        """Throttled render callback: fires every 33 ms while the slider is held."""
        if self._pending_drag_frame is not None:
            frame = self._pending_drag_frame
            self._pending_drag_frame = None
            self._update_frame(frame)

    def _on_slider_change(self, value: int) -> None:
        if self._slider_dragging:
            self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._preloader.seek(value)
            self._pending_drag_frame = value
        else:
            self._update_frame(value)

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visibility[key] = state == Qt.Checked
        if key == 'kinect_point_cloud' and self._preloader is not None:
            if self._visibility[key]:
                self._preloader.seek(self.current_index)
                self._preloader.resume()
            else:
                self._preloader.pause()
        self._update_frame(self.current_index)

    def _apply_contact_scalar_mode(self) -> None:
        """Switch the contact actor between depth colouring and flat red.

        Deliberately *not* an ``add_mesh`` / ``remove_actor`` cycle: re-adding
        the mesh re-enters PyVista's scalar-bar range logic, which does not
        preserve a global ``clim``.  Toggling ``scalar_visibility`` leaves the
        actor, its mapper and its lookup table exactly where they are, so the
        colour scale is identical before and after the round trip.
        """
        actor = getattr(self, '_actor_contact', None)
        if actor is None or self._contact_clim is None:
            return

        show_scalars = self._depth_colouring_active
        actor.mapper.scalar_visibility = show_scalars
        # Re-assert on every mode change for the same reason it is re-asserted
        # after every dataset swap.
        actor.mapper.scalar_range = self._contact_clim

        if CONTACT_DEPTH_SCALAR_BAR_TITLE in self.plotter.scalar_bars:
            # A colourbar with nothing mapped to it would claim the flat-red
            # points mean something on that scale.
            self.plotter.scalar_bars[CONTACT_DEPTH_SCALAR_BAR_TITLE].SetVisibility(
                bool(show_scalars)
            )

    def _on_contact_depth_colour_changed(self, state: int) -> None:
        """Handle the 'Colour by depth' checkbox."""
        self._colour_contact_by_depth = state == Qt.Checked
        self._apply_contact_scalar_mode()
        self.plotter.render()

    def _on_point_size_changed(self, key: str, value: int) -> None:
        self._point_sizes[key] = float(value)
        actor_map = {
            'kinect_point_cloud': getattr(self, '_actor_kinect', None),
            'forearms':           getattr(self, '_actor_forearm', None),
            'contact_points':     getattr(self, '_actor_contact', None),
        }
        actor = actor_map.get(key)
        if actor is not None:
            actor.GetProperty().SetPointSize(float(value))
            self.plotter.render()
        else:
            self._update_frame(self.current_index)

    def _on_crop_changed(self, value: int) -> None:
        self._crop_half_size = float(value)
        # Invalidate the CuPy crop-centre so it is not re-used with a stale half_size.
        # (centre doesn't change, but the crop box size does — GPU path rechecks both)
        self._c_gpu = None

    def _on_lod_changed(self, value: int) -> None:
        self._interactive_stride = value

    def _schedule_exact_frame(self, frame_idx: int) -> None:
        """
        Start polling the preloader buffer until *frame_idx* is cached
        exactly, then re-render it at full resolution.
        """
        if self._is_interactive:
            return
        self._exact_frame_pending = frame_idx
        if not self._exact_frame_timer.isActive():
            self._exact_frame_timer.start()

    def _on_exact_frame_poll(self) -> None:
        """Poll the preloader buffer for the pending exact frame."""
        if self._exact_frame_pending is None or self._is_interactive:
            self._exact_frame_timer.stop()
            self._exact_frame_pending = None
            return
        frame_idx = self._exact_frame_pending
        _, is_exact = self._preloader.get_frame(frame_idx)
        if is_exact:
            self._exact_frame_timer.stop()
            self._exact_frame_pending = None
            if self.current_index == frame_idx:
                self._update_frame(frame_idx)

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
        self.plotter.camera.focal_point = self._camera_focal_point.tolist()
        self.plotter.render()
        self._refresh_cam_pos_label()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
            self._is_interactive = False
            self._update_frame(self.current_index)
            self._refresh_cam_pos_label()
        else:
            self._is_interactive = True
            fps = 30
            if (
                self._mkv is not None
                and self._mkv._reader is not None
                and hasattr(self._mkv._reader, 'fps')
            ):
                fps = self._mkv._reader.fps
            self._play_timer.start(int(1000 / (fps * self._speed_spinbox.value())))
            self.play_button.setText("⏸ Pause")

    def _play_advance(self) -> None:
        nxt = (self.current_index + 1) % self._total_frames
        self.frame_slider.setValue(nxt)

    # ======================================================================
    # Show / close
    # ======================================================================

    def _deferred_start(self) -> None:
        """Initialize the VTK interactor (once), then render frame 0.

        ``_neural_scale`` is already set by ``_load_block()`` so it does not
        need to be recomputed here.  The VTK interactor initialisation is
        guarded so it only runs once even when this method is called on the
        first show.
        """
        if not getattr(self, '_vtk_interactor_initialized', False):
            try:
                self.plotter.interactor.Initialize()
                self._vtk_interactor_initialized = True
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

    def closeEvent(self, event) -> None:  # noqa: N802
        self._play_timer.stop()
        self._drag_timer.stop()
        self._exact_frame_timer.stop()
        if self._preloader is not None:
            self._preloader.stop()
            self._preloader.join(timeout=2.0)
        try:
            if self._mkv is not None:
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
