"""
KinectRgbDepthViewer — sticker-independent Kinect co-registration diagnostic widget.

This module is intentionally free of any import from
``preprocessing.stickers_analysis``.  Its sole purpose is auditing pixel-level
RGB↔depth co-registration: given an MKV file it renders the raw color frame
and the transformed-depth-Z channel side-by-side in a shared absolute pixel
coordinate system so that any spatial offset between the two streams can be
directly observed.

Phase 2 adds a synchronised crosshair: clicking on either panel places
``axhline(v)`` and ``axvline(u)`` lines on both panels simultaneously without
redrawing the full image.  A "Clear" button removes the crosshair.

Phase 3 adds a toggleable Canny-edge overlay (checkbox + two sliders for
``t1`` / ``t2``) drawn on the depth axis only as a red-tinted RGBA image.
"""

from __future__ import annotations

import csv
import logging
import math
import subprocess
from pathlib import Path
from typing import Optional

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except ImportError:  # Python < 3.8 fallback
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore

_log = logging.getLogger(__name__)

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV


class KinectRgbDepthViewer(QWidget):
    """
    Two-panel PyQt5 widget for auditing Kinect RGB ↔ depth co-registration.

    Both panels share a single absolute pixel coordinate system — their axes
    are linked via ``sharex`` / ``sharey`` and each image is rendered with
    ``extent=(0, W, H, 0)`` — so that any spatial offset between the color
    stream and the transformed depth stream is directly visible.

    Parameters
    ----------
    mkv_path:
        Path to the ``.mkv`` file produced by an Azure Kinect recording.
    parent:
        Optional parent ``QWidget`` (Qt ownership).
    parallax_correction:
        Forwarded verbatim to ``KinectMKV``.  Defaults to ``True`` so that
        the widget displays the parallax-corrected depth that production
        pipelines consume.  Pass ``False`` only for RGB↔depth registration
        audits that must measure the raw pyk4a offset (see
        ``docs/development/knowledge-base/note-kinect-depth-access-single-path.md``).
    """

    def __init__(
        self,
        mkv_path: str | Path,
        parent: Optional[QWidget] = None,
        *,
        parallax_correction: bool = True,
    ) -> None:
        super().__init__(parent)

        self._mkv_path = Path(mkv_path)

        # Open the MKV for the widget's lifetime; closed in closeEvent.
        # parallax_correction=True by default so the widget shows corrected
        # depth by default; the audit tool opts out explicitly.
        self._mkv = KinectMKV(self._mkv_path, parallax_correction=parallax_correction)
        self._mkv.__enter__()

        total_frames = len(self._mkv)

        # --- Figure and axes ---
        self._figure = Figure(figsize=(12, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)

        self._ax_rgb = self._figure.add_subplot(1, 2, 1)
        self._ax_depth = self._figure.add_subplot(
            1, 2, 2,
            sharex=self._ax_rgb,
            sharey=self._ax_rgb,
        )

        # Placeholder imshow artists (will be replaced on first frame load).
        self._im_rgb = None
        self._im_depth = None
        self._color_status_text: Optional[object] = None
        self._depth_status_text: Optional[object] = None

        # --- Crosshair state (Phase 2) ---
        # Stores the current crosshair pixel coords as (u, v), or None when
        # no crosshair is active.  u = column index, v = row index.
        self._crosshair: Optional[tuple[int, int]] = None

        # Four artist handles — two per axis (hline + vline).  Stored so
        # they can be removed without redrawing the full image.
        self._crosshair_lines: list = []

        # --- Canny overlay state (Phase 3) ---
        # The last color frame in BGR (set by _display_frame) so that Canny
        # can be recomputed without re-fetching from the MKV.
        self._current_color_bgr: Optional[np.ndarray] = None

        # Artist handle for the red-tinted Canny RGBA overlay on the depth
        # axis.  None when no overlay is currently drawn.
        self._canny_artist = None

        # --- Phase 1: depth source + measure offset state ---
        # The last transformed_depth_point_cloud (H×W×3) for z readout.
        self._current_point_cloud: Optional[np.ndarray] = None

        # Measure-offset state machine.
        # Step 0: collecting N RGB clicks into _measure_rgb_pts.
        # Step 1: collecting N depth clicks into _measure_depth_pts.
        # When both lists reach N, one CSV row is committed with median statistics.
        self._measure_step: int = 0
        self._measure_rgb_pts: list[tuple[int, int]] = []
        self._measure_depth_pts: list[tuple[int, int]] = []
        self._measure_z_samples: list[float] = []  # z_mm per depth click
        self._near_edge: bool = False  # depth-gradient warning flag for current row
        self._measure_markers: list = []  # "+" artist handles on both axes
        self._measurements: list[dict] = []  # accumulated completed pairs

        # --- Pan state (right-button drag) ---
        # When a right-drag is active, stores (x0, y0, xlim0, ylim0, inaxes)
        # captured on press.  Storing the originals — rather than re-reading
        # get_xlim()/get_ylim() each motion — avoids cumulative drift caused
        # by sharex/sharey double-application.
        self._pan_state: Optional[tuple] = None

        self._configure_axes()

        # Mouse interactions: left-click → crosshair, right-drag → pan,
        # scroll wheel → zoom-at-cursor.  All coords arrive pre-transformed
        # to data space via matplotlib's event system.
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        # --- Frame index controls ---
        # Full-width slider for scrubbing + compact spinbox for precise entry.
        # They are kept in sync: moving one updates the other without double-
        # triggering _display_frame.
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(max(0, total_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setTickPosition(QSlider.TicksBelow)
        self._frame_slider.setTickInterval(max(1, total_frames // 20))

        self._spinbox = QSpinBox()
        self._spinbox.setMinimum(0)
        self._spinbox.setMaximum(max(0, total_frames - 1))
        self._spinbox.setValue(0)
        self._spinbox.setSuffix(f" / {total_frames - 1}")
        self._spinbox.setFixedWidth(110)

        # Connect slider ↔ spinbox bidirectionally, guarded by _block_frame_sync
        # to avoid recursive updates.
        self._block_frame_sync = False
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)

        # --- Clear crosshair button ---
        self._btn_clear = QPushButton("Clear crosshair")
        self._btn_clear.setToolTip("Remove the crosshair from both panels.")
        self._btn_clear.clicked.connect(self._on_clear_crosshair)

        # --- Reset view button ---
        self._reset_view_btn = QPushButton("Reset view")
        self._reset_view_btn.setToolTip(
            "Restore both panels to the full image extent."
        )
        self._reset_view_btn.clicked.connect(self._on_reset_view)

        # --- Canny overlay controls (Phase 3) ---
        self._chk_canny = QCheckBox("Canny overlay")
        self._chk_canny.setToolTip(
            "Draw Canny edges from the color frame over the depth panel."
        )
        self._chk_canny.setChecked(False)
        self._chk_canny.toggled.connect(self._on_canny_control_changed)

        # t1 slider (0–500, default 50)
        self._lbl_t1 = QLabel("t1: 50")
        self._slider_t1 = QSlider(Qt.Horizontal)
        self._slider_t1.setMinimum(0)
        self._slider_t1.setMaximum(500)
        self._slider_t1.setValue(50)
        self._slider_t1.setFixedWidth(120)
        self._slider_t1.valueChanged.connect(self._on_t1_changed)

        # t2 slider (0–500, default 150)
        self._lbl_t2 = QLabel("t2: 150")
        self._slider_t2 = QSlider(Qt.Horizontal)
        self._slider_t2.setMinimum(0)
        self._slider_t2.setMaximum(500)
        self._slider_t2.setValue(150)
        self._slider_t2.setFixedWidth(120)
        self._slider_t2.valueChanged.connect(self._on_t2_changed)

        # --- Status label ---
        self._status_label = QLabel("")

        # --- Depth-source combobox ---
        self._combo_depth_source = QComboBox()
        self._combo_depth_source.addItem("Point cloud Z")   # index 0
        self._combo_depth_source.addItem("transformed_depth")  # index 1
        self._combo_depth_source.currentIndexChanged.connect(
            lambda _: self._display_frame(self._frame_slider.value())
        )

        # --- Clicks-per-feature spinbox (Task 1.1) ---
        # Selects N: the state machine collects N RGB clicks then N depth clicks
        # before committing one CSV row using median statistics.
        self._spinbox_clicks = QSpinBox()
        self._spinbox_clicks.setMinimum(1)
        self._spinbox_clicks.setMaximum(9)
        self._spinbox_clicks.setValue(1)
        self._spinbox_clicks.setFixedWidth(46)
        self._spinbox_clicks.setToolTip(
            "Number of clicks per feature per axis. When > 1 the viewer "
            "accumulates N RGB clicks then N depth clicks and commits one row "
            "with median (Δu, Δv, |Δ|) and per-click std."
        )
        # Changing N mid-sequence clears the in-progress measurement.
        self._spinbox_clicks.valueChanged.connect(self._on_clicks_per_feature_changed)

        # --- Measure offset button ---
        self._btn_measure = QPushButton("Measure offset")
        self._btn_measure.setCheckable(True)
        self._btn_measure.setToolTip(
            "Click a point on the RGB panel then the corresponding point on "
            "the depth panel to measure the pixel offset."
        )
        self._btn_measure.toggled.connect(self._on_measure_toggled)

        self._btn_export = QPushButton("Export CSV")
        self._btn_export.setToolTip("Export all collected measurements to a CSV file.")
        self._btn_export.clicked.connect(self._on_export_csv)
        self._btn_export.setEnabled(False)

        self._btn_clear_measurements = QPushButton("Clear measurements")
        self._btn_clear_measurements.setToolTip("Remove all collected measurement points.")
        self._btn_clear_measurements.clicked.connect(self._on_clear_measurements)
        self._btn_clear_measurements.setEnabled(False)

        # --- Layout ---
        # Row 1: full-width frame scrubber
        scrubber_row = QWidget()
        scrubber_layout = QHBoxLayout(scrubber_row)
        scrubber_layout.setContentsMargins(4, 0, 4, 0)
        scrubber_layout.addWidget(QLabel("Frame:"))
        scrubber_layout.addWidget(self._spinbox)
        scrubber_layout.addSpacing(8)
        scrubber_layout.addWidget(self._frame_slider, stretch=1)

        # Row 2: crosshair + Canny controls
        controls_row = QWidget()
        controls_layout = QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(4, 2, 4, 2)
        controls_layout.addWidget(self._btn_measure)
        controls_layout.addWidget(QLabel("×"))
        controls_layout.addWidget(self._spinbox_clicks)
        controls_layout.addWidget(self._btn_export)
        controls_layout.addWidget(self._btn_clear_measurements)
        controls_layout.addSpacing(16)
        controls_layout.addWidget(self._btn_clear)
        controls_layout.addWidget(self._reset_view_btn)
        controls_layout.addSpacing(24)
        controls_layout.addWidget(QLabel("Depth:"))
        controls_layout.addWidget(self._combo_depth_source)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self._chk_canny)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self._lbl_t1)
        controls_layout.addWidget(self._slider_t1)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(self._lbl_t2)
        controls_layout.addWidget(self._slider_t2)
        controls_layout.addStretch()

        outer = QVBoxLayout(self)
        outer.addWidget(self._canvas, stretch=1)
        outer.addWidget(scrubber_row)
        outer.addWidget(self._status_label)
        outer.addWidget(controls_row)

        # Task 1.3 — one-shot equivalence check: if transformed_depth and
        # transformed_depth_point_cloud[..., 2] are identical, the depth-source
        # toggle is purely diagnostic and we can hide it.
        self._run_equivalence_check()

        # Load the first frame.
        self._display_frame(0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_axes(self) -> None:
        """Apply labels and tick configuration to both axes."""
        for ax, title in ((self._ax_rgb, "Color (RGB)"), (self._ax_depth, "Depth-Z (mm)")):
            ax.set_title(title)
            ax.set_xlabel("u [px]")
            ax.set_ylabel("v [px]")
            ax.xaxis.set_major_locator(
                __import__("matplotlib.ticker", fromlist=["AutoLocator"]).AutoLocator()
            )
            ax.yaxis.set_major_locator(
                __import__("matplotlib.ticker", fromlist=["AutoLocator"]).AutoLocator()
            )

    # Threshold (mm) for local depth span over 5×5 neighbourhood that triggers
    # the near-edge warning (Task 1.2).  Module-level constant for easy tuning.
    _NEAR_EDGE_THRESHOLD_MM: float = 20.0

    def _run_equivalence_check(self) -> None:
        """One-shot check: are transformed_depth and point-cloud Z identical?

        Runs on the first frame at MKV load (Task 1.3).  If the two arrays
        agree (``np.allclose(..., equal_nan=True)``) the depth-source combobox
        is hidden — both sources are equivalent so the toggle is purely
        diagnostic.  The result is always logged at INFO or WARNING level.
        """
        try:
            frame = self._mkv[0]
            td = frame.transformed_depth
            pc = frame.transformed_depth_point_cloud
            if td is None or pc is None:
                _log.warning(
                    "KinectRgbDepthViewer: equivalence check skipped — "
                    "first frame has None transformed_depth or point_cloud."
                )
                return
            equivalent = np.allclose(
                td.astype(np.float32),
                pc[..., 2].astype(np.float32),
                equal_nan=True,
            )
            if equivalent:
                _log.info(
                    "KinectRgbDepthViewer: transformed_depth ≡ point_cloud[...,2] "
                    "on frame 0 — depth-source combobox hidden (diagnostic only)."
                )
                self._combo_depth_source.setVisible(False)
            else:
                _log.warning(
                    "KinectRgbDepthViewer: transformed_depth ≠ point_cloud[...,2] "
                    "on frame 0 — depth-source combobox kept visible."
                )
        except Exception as exc:
            _log.warning("KinectRgbDepthViewer: equivalence check failed: %s", exc)

    def _compute_near_edge(self, v: int, u: int) -> bool:
        """Return True if the 5×5 depth neighbourhood around (u, v) spans > threshold.

        Samples ``self._current_point_cloud[..., 2]`` in a 5×5 patch centred
        on the clicked pixel.  NaN values are ignored.  Falls back to False
        when the point cloud is absent or the patch is all-NaN (Task 1.2).
        """
        if self._current_point_cloud is None:
            return False
        z_all = self._current_point_cloud[..., 2]
        v0, v1 = max(0, v - 2), min(z_all.shape[0], v + 3)
        u0, u1 = max(0, u - 2), min(z_all.shape[1], u + 3)
        patch = z_all[v0:v1, u0:u1].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return False
        return float(valid.max() - valid.min()) > self._NEAR_EDGE_THRESHOLD_MM

    def _display_frame(self, frame_index: int) -> None:
        """Fetch ``frame_index`` from the MKV and refresh both panels."""
        try:
            frame = self._mkv[frame_index]
        except (IndexError, ValueError) as exc:
            # Surface the error on the canvas rather than crashing.
            self._ax_rgb.set_title(f"Color (RGB) — error: {exc}")
            self._canvas.draw_idle()
            return

        color_bgr = frame.color  # H×W×3, BGR, uint8 or None

        # Store for Canny overlay recomputation (Phase 3).
        self._current_color_bgr = color_bgr

        # Convert BGR → RGB for matplotlib display.
        if color_bgr is not None:
            color_rgb = color_bgr[:, :, ::-1]
        else:
            color_rgb = None

        # Store point cloud for z_mm readout in measure mode (Phase 1).
        self._current_point_cloud = frame.transformed_depth_point_cloud  # H×W×3 or None

        # Prepare the 2D depth array based on the selected source.
        if self._combo_depth_source.currentIndex() == 0:
            depth_2d = (
                self._current_point_cloud[:, :, 2].astype(np.float32)
                if self._current_point_cloud is not None
                else None
            )
        else:
            td = frame.transformed_depth
            depth_2d = td.astype(np.float32) if td is not None else None

        # Safety: assert same spatial shape when both arrays are present.
        if color_rgb is not None and depth_2d is not None:
            assert color_rgb.shape[:2] == depth_2d.shape[:2], (
                f"Shape mismatch: color {color_rgb.shape[:2]} vs "
                f"depth {depth_2d.shape[:2]}. Axes linkage would be invalid."
            )

        # ax.clear() inside the render helpers destroys any existing crosshair
        # artists and the Canny artist without going through our remove logic,
        # so we discard the stale references first and re-draw afterwards.
        self._crosshair_lines.clear()
        self._canny_artist = None  # will be invalidated by ax.clear() inside helpers

        # Clear measure markers (frame scrub resets the visual state).
        self._clear_measure_state()
        if self._btn_measure.isChecked():
            self._status_label.setText("Measure: click on RGB panel")

        self._render_color_panel(color_rgb)
        self._render_depth_panel(depth_2d)

        # Re-apply Canny overlay if it was enabled before the frame change.
        if self._chk_canny.isChecked():
            self._draw_canny_overlay()

        # Re-apply crosshair if one was active before the frame change.
        if self._crosshair is not None:
            self._draw_crosshair()

        # Rescale depth colormap to whatever is currently visible — when zoomed
        # in this overrides the full-frame clim set inside _render_depth_panel.
        self._update_depth_clim_to_view()

        self._canvas.draw_idle()

    def _render_color_panel(self, color_rgb: Optional[np.ndarray]) -> None:
        """Update the RGB axis with ``color_rgb`` (H×W×3, uint8)."""
        ax = self._ax_rgb
        if color_rgb is None:
            # ax.clear() invalidates any existing artists — null refs first.
            self._color_status_text = None
            self._im_rgb = None
            ax.clear()
            self._configure_axes()
            ax.set_title("Color (RGB)")
            self._color_status_text = ax.text(
                0.5, 0.5, "No color data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="red",
            )
            return

        # Remove stale "No color data" text when color becomes available.
        if self._color_status_text is not None:
            try:
                self._color_status_text.remove()
            except ValueError:
                pass
            self._color_status_text = None

        H, W = color_rgb.shape[:2]
        extent = (0, W, H, 0)

        if self._im_rgb is None:
            self._im_rgb = ax.imshow(
                color_rgb,
                extent=extent,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
        else:
            self._im_rgb.set_data(color_rgb)
            self._im_rgb.set_extent(extent)

    def _render_depth_panel(self, depth_2d: Optional[np.ndarray]) -> None:
        """Update the depth-Z axis with a precomputed 2D float32 depth array (mm)."""
        ax = self._ax_depth

        if depth_2d is None:
            ax.clear()
            self._configure_axes()
            ax.set_title("Depth-Z (mm)")
            if self._depth_status_text is None:
                self._depth_status_text = ax.text(
                    0.5, 0.5,
                    "depth data unavailable",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=10, color="orange",
                    wrap=True,
                )
            self._im_depth = None
            return

        # Remove stale status text if depth becomes available.
        if self._depth_status_text is not None:
            try:
                self._depth_status_text.remove()
            except ValueError:
                pass
            self._depth_status_text = None

        depth_z = depth_2d.copy()

        # Mask zero / invalid pixels with NaN so they render as transparent.
        depth_z[depth_z <= 0] = np.nan

        H, W = depth_z.shape
        extent = (0, W, H, 0)

        if self._im_depth is None:
            self._im_depth = ax.imshow(
                depth_z,
                extent=extent,
                origin="upper",
                aspect="equal",
                cmap="gray",
                interpolation="nearest",
            )
        else:
            self._im_depth.set_data(depth_z)
            self._im_depth.set_extent(extent)
            # Recompute color limits to span the new frame's valid range.
            valid = depth_z[~np.isnan(depth_z)]
            if valid.size > 0:
                self._im_depth.set_clim(valid.min(), valid.max())

    def _remove_crosshair_artists(self) -> None:
        """Remove all stored crosshair line artists from their axes."""
        for line in self._crosshair_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self._crosshair_lines.clear()

    def _draw_crosshair(self) -> None:
        """Draw axhline/axvline on both axes for the current crosshair state.

        Must only be called when ``self._crosshair`` is not ``None``.
        Assumes any previous crosshair artists have already been removed.
        """
        u, v = self._crosshair
        for ax in (self._ax_rgb, self._ax_depth):
            hline = ax.axhline(v, color="red", linewidth=0.8, linestyle="--")
            vline = ax.axvline(u, color="red", linewidth=0.8, linestyle="--")
            self._crosshair_lines.extend([hline, vline])

    def _draw_canny_overlay(self) -> None:
        """Compute and draw the Canny-edge overlay on the depth axis.

        Uses ``self._current_color_bgr`` so no new MKV fetch is needed.
        The overlay is a red-tinted RGBA image whose alpha channel equals
        the Canny edge mask (0 = transparent, 255 = opaque red).

        Any previously drawn Canny artist is removed first.  If no color
        frame is available the method is a no-op.
        """
        # Remove stale artist (if any).
        if self._canny_artist is not None:
            try:
                self._canny_artist.remove()
            except ValueError:
                pass
            self._canny_artist = None

        if self._current_color_bgr is None:
            return

        t1 = self._slider_t1.value()
        t2 = self._slider_t2.value()

        gray = cv2.cvtColor(self._current_color_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)  # uint8, 0 or 255

        H, W = edges.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255      # red channel fully on
        rgba[:, :, 3] = edges    # alpha = edge mask

        self._canny_artist = self._ax_depth.imshow(
            rgba,
            extent=(0, W, H, 0),
            origin="upper",
            aspect="auto",
            alpha=0.7,
        )

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event) -> None:
        """Handle a mouse-button press on the figure canvas.

        Ignores clicks outside any axis or outside the image extent.
        Only left-button presses act; right-button is reserved for pan.

        Measure-offset state machine (N = ``_spinbox_clicks.value()``):
          Step 0 — collect N RGB clicks into ``_measure_rgb_pts``.
                   Transitions to step 1 when the list reaches N.
          Step 1 — collect N depth clicks into ``_measure_depth_pts``.
                   When the list reaches N, computes per-click offsets,
                   takes medians / std, and commits one row to
                   ``_measurements``.  Resets to step 0 (markers kept).
        """
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in (self._ax_rgb, self._ax_depth):
            return

        # Round to integer pixel coordinates.
        u = int(round(event.xdata))
        v = int(round(event.ydata))

        # Determine image bounds; both axes share (H, W) — prefer RGB.
        ref_artist = self._im_rgb if self._im_rgb is not None else self._im_depth
        if ref_artist is None:
            return

        data = ref_artist.get_array()
        H, W = data.shape[:2]
        u = max(0, min(u, W - 1))
        v = max(0, min(v, H - 1))

        # --- Measure offset mode ---
        if self._btn_measure.isChecked():
            N = self._spinbox_clicks.value()

            if self._measure_step == 0:
                # Step 0: collecting RGB clicks — only accept ax_rgb.
                if event.inaxes is not self._ax_rgb:
                    return
                self._measure_rgb_pts.append((u, v))
                artist, = self._ax_rgb.plot(
                    u, v, '+', color='cyan', markersize=10, markeredgewidth=2
                )
                self._measure_markers.append(artist)
                n_rgb = len(self._measure_rgb_pts)
                if n_rgb < N:
                    self._status_label.setText(
                        f"Measure: RGB {n_rgb}/{N} — keep clicking RGB panel"
                    )
                else:
                    # Collected all RGB clicks; transition to depth collection.
                    self._measure_step = 1
                    self._near_edge = False  # reset for this row
                    self._status_label.setText(
                        f"Measure: RGB {N}/{N} done — now click depth panel (1/{N})"
                    )
                self._canvas.draw_idle()

            elif self._measure_step == 1:
                # Step 1: collecting depth clicks — only accept ax_depth.
                if event.inaxes is not self._ax_depth:
                    return
                self._measure_depth_pts.append((u, v))
                artist, = self._ax_depth.plot(
                    u, v, '+', color='cyan', markersize=10, markeredgewidth=2
                )
                self._measure_markers.append(artist)

                # Read z_mm from the point cloud at the click position.
                z_mm = float('nan')
                if self._current_point_cloud is not None:
                    try:
                        z_val = float(self._current_point_cloud[v, u, 2])
                        if z_val > 0 and math.isfinite(z_val):
                            z_mm = z_val
                    except (IndexError, ValueError):
                        pass
                self._measure_z_samples.append(z_mm)

                # Task 1.2 — depth-gradient warning: fire on first depth click
                # that lands on a depth edge; latch for the whole row.
                if not self._near_edge:
                    self._near_edge = self._compute_near_edge(v, u)

                n_depth = len(self._measure_depth_pts)
                if n_depth < N:
                    edge_tag = " [EDGE]" if self._near_edge else ""
                    self._status_label.setText(
                        f"Measure: depth {n_depth}/{N}{edge_tag}"
                    )
                    if self._near_edge:
                        self._status_label.setStyleSheet("color: #b86a00;")
                    self._canvas.draw_idle()
                else:
                    # Collected all depth clicks — commit one row.
                    self._commit_measurement_row()

            return  # Do NOT update crosshair in measure mode.

        # Update state and redraw crosshair without touching imshow artists.
        self._remove_crosshair_artists()
        self._crosshair = (u, v)
        self._draw_crosshair()
        self._canvas.draw_idle()

    def _commit_measurement_row(self) -> None:
        """Compute median statistics and append one row to ``_measurements``.

        Called once N RGB clicks and N depth clicks have been collected.
        Computes per-click (Δu, Δv, |Δ|) for each matched pair (indexed by
        order), then records median/std of those quantities.  Resets the
        in-progress lists and returns the state machine to step 0.
        """
        rgb_pts = self._measure_rgb_pts
        dep_pts = self._measure_depth_pts
        z_samples = self._measure_z_samples
        N = len(dep_pts)

        # Per-click offset vectors and magnitudes.
        du_list = [dep_pts[i][0] - rgb_pts[i][0] for i in range(N)]
        dv_list = [dep_pts[i][1] - rgb_pts[i][1] for i in range(N)]
        mag_list = [math.sqrt(du ** 2 + dv ** 2) for du, dv in zip(du_list, dv_list)]

        du_med = float(np.median(du_list))
        dv_med = float(np.median(dv_list))
        mag_med = float(np.median(mag_list))
        mag_std = float(np.std(mag_list))

        # Median z_mm from collected depth samples (NaN-safe).
        z_valid = [z for z in z_samples if math.isfinite(z)]
        z_med = float(np.median(z_valid)) if z_valid else float('nan')
        z_str = f"{z_med:.0f}" if math.isfinite(z_med) else "nan"

        # Representative RGB and depth pixel (medians of accumulated points).
        u_rgb_med = int(round(float(np.median([p[0] for p in rgb_pts]))))
        v_rgb_med = int(round(float(np.median([p[1] for p in rgb_pts]))))
        u_dep_med = int(round(float(np.median([p[0] for p in dep_pts]))))
        v_dep_med = int(round(float(np.median([p[1] for p in dep_pts]))))

        self._measurements.append({
            "frame": self._frame_slider.value(),
            "u_rgb": u_rgb_med,
            "v_rgb": v_rgb_med,
            "u_depth": u_dep_med,
            "v_depth": v_dep_med,
            "delta_u": round(du_med, 2),
            "delta_v": round(dv_med, 2),
            "delta_mag": round(mag_med, 2),
            "z_mm": round(z_med, 1) if math.isfinite(z_med) else "",
            "depth_source": self._combo_depth_source.currentText(),
            # Median-mode columns (N=1 rows have std=0, n_clicks=1).
            "delta_u_median": round(du_med, 2),
            "delta_v_median": round(dv_med, 2),
            "delta_mag_median": round(mag_med, 2),
            "delta_mag_std": round(mag_std, 3),
            "n_clicks": N,
            "near_edge": self._near_edge,
        })

        self._btn_export.setEnabled(True)
        self._btn_clear_measurements.setEnabled(True)
        n_rows = len(self._measurements)

        edge_tag = "  [EDGE]" if self._near_edge else ""
        self._status_label.setStyleSheet("")
        self._status_label.setText(
            f"[{n_rows} pts]  Δu={du_med:+.1f}  Δv={dv_med:+.1f}"
            f"  |Δ|={mag_med:.1f}±{mag_std:.2f} px  z={z_str} mm{edge_tag}"
        )

        # Reset in-progress state; keep "+" markers visible.
        self._measure_step = 0
        self._measure_rgb_pts = []
        self._measure_depth_pts = []
        self._measure_z_samples = []
        self._near_edge = False
        self._canvas.draw_idle()

    def _on_clear_crosshair(self) -> None:
        """Clear button handler — remove crosshair from both panels."""
        redraw = False
        if self._crosshair is not None:
            self._remove_crosshair_artists()
            self._crosshair = None
            redraw = True
        if self._btn_measure.isChecked():
            self._clear_measure_state()
            self._btn_measure.setChecked(False)
            n = len(self._measurements)
            self._status_label.setText(f"[{n} pts] — measurement mode off" if n else "")
            redraw = True
        if redraw:
            self._canvas.draw_idle()

    def _clear_measure_state(self) -> None:
        """Remove "+" marker artists and reset measure state variables."""
        for artist in self._measure_markers:
            try:
                artist.remove()
            except ValueError:
                pass
        self._measure_markers.clear()
        self._measure_step = 0
        self._measure_rgb_pts = []
        self._measure_depth_pts = []
        self._measure_z_samples = []
        self._near_edge = False
        self._status_label.setStyleSheet("")

    def _on_measure_toggled(self, checked: bool) -> None:
        """Slot for the Measure offset button toggle."""
        if checked:
            self._clear_measure_state()
            n = len(self._measurements)
            prefix = f"[{n} pts]  " if n else ""
            N = self._spinbox_clicks.value()
            clicks_hint = f"(1/{N})" if N > 1 else ""
            self._status_label.setText(f"{prefix}Measure: click RGB panel {clicks_hint}")
        else:
            self._clear_measure_state()
            n = len(self._measurements)
            self._status_label.setText(f"[{n} pts] — measurement mode off" if n else "")
        self._canvas.draw_idle()

    def _on_clicks_per_feature_changed(self, value: int) -> None:
        """Clear any in-progress measurement when clicks_per_feature changes."""
        if self._measure_rgb_pts or self._measure_depth_pts:
            self._clear_measure_state()
            if self._btn_measure.isChecked():
                n = len(self._measurements)
                prefix = f"[{n} pts]  " if n else ""
                self._status_label.setText(
                    f"{prefix}Measure reset — click RGB panel (1/{value})"
                )
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # CSV export helpers (Task 1.4)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_viewer_git_sha() -> str:
        """Return the short git SHA of HEAD, or ``"unknown"`` on any failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3,
            )
            sha = result.stdout.strip()
            return sha if sha else "unknown"
        except Exception:
            return "unknown"

    def _get_device_serial(self) -> str:
        """Return the device serial from MKV record configuration, or ``"unknown"``."""
        try:
            cfg = self._mkv.playback.get_record_configuration()
            serial = getattr(cfg, "device_serial_number", None)
            if serial:
                return str(serial)
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _get_pyk4a_version() -> str:
        """Return the installed pyk4a version string, or ``"unknown"``."""
        try:
            return _pkg_version("pyk4a")
        except PackageNotFoundError:
            return "unknown"

    def _on_export_csv(self) -> None:
        """Open a save dialog and write all accumulated measurements to CSV.

        The output file starts with a single ``#``-prefixed metadata header
        line (readable by ``pandas.read_csv(comment="#")``), followed by the
        column header row, then one data row per completed measurement.

        New columns (Task 1.4): ``delta_u_median``, ``delta_v_median``,
        ``delta_mag_median``, ``delta_mag_std``, ``n_clicks``, ``near_edge``.
        Legacy single-click columns (``delta_u``, ``delta_v``, ``delta_mag``,
        ``z_mm``, ``depth_source``) are kept for backward compatibility with
        Phase-1 CSV files.
        """
        if not self._measurements:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", "", "CSV files (*.csv)"
        )
        if not path:
            return

        total_frames = len(self._mkv)
        meta = {
            "recording_path": str(self._mkv_path),
            "frame_range": f"0-{total_frames - 1}",
            "device_serial": self._get_device_serial(),
            "pyk4a_version": self._get_pyk4a_version(),
            "viewer_git_sha": self._get_viewer_git_sha(),
            "clicks_per_feature": self._spinbox_clicks.value(),
            "csv_schema_version": 2,
        }
        header_line = "# " + ", ".join(f"{k}={v}" for k, v in meta.items())

        fieldnames = [
            # Legacy columns (schema v1 compatibility).
            "frame", "u_rgb", "v_rgb", "u_depth", "v_depth",
            "delta_u", "delta_v", "delta_mag", "z_mm", "depth_source",
            # Median-mode columns (schema v2).
            "delta_u_median", "delta_v_median",
            "delta_mag_median", "delta_mag_std",
            "n_clicks", "near_edge",
        ]
        with open(path, "w", newline="") as fh:
            fh.write(header_line + "\n")
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._measurements)

    def _on_clear_measurements(self) -> None:
        """Discard all accumulated measurements."""
        self._measurements.clear()
        self._btn_export.setEnabled(False)
        self._btn_clear_measurements.setEnabled(False)
        if self._btn_measure.isChecked():
            self._status_label.setText("Measure: click on RGB panel")
        else:
            self._status_label.setText("")

    # ------------------------------------------------------------------
    # Zoom & pan handlers
    # ------------------------------------------------------------------

    _ZOOM_STEP = 1.2
    _MIN_SPAN = 1.0  # pixels — prevents collapse/inversion

    def _on_scroll(self, event) -> None:
        """Wheel-zoom centered on the cursor. Shared axes mirror the change."""
        if event.inaxes not in (self._ax_rgb, self._ax_depth):
            return
        if event.xdata is None or event.ydata is None:
            return

        # Scroll up → zoom in (scale < 1); scroll down → zoom out (scale > 1).
        scale = 1.0 / self._ZOOM_STEP if event.step > 0 else self._ZOOM_STEP

        x0, x1 = self._ax_rgb.get_xlim()
        y0, y1 = self._ax_rgb.get_ylim()

        new_x0 = event.xdata + (x0 - event.xdata) * scale
        new_x1 = event.xdata + (x1 - event.xdata) * scale
        new_y0 = event.ydata + (y0 - event.ydata) * scale
        new_y1 = event.ydata + (y1 - event.ydata) * scale

        if abs(new_x1 - new_x0) < self._MIN_SPAN:
            return
        if abs(new_y1 - new_y0) < self._MIN_SPAN:
            return

        self._ax_rgb.set_xlim(new_x0, new_x1)
        self._ax_rgb.set_ylim(new_y0, new_y1)
        self._update_depth_clim_to_view()
        self._canvas.draw_idle()

    def _on_mouse_press(self, event) -> None:
        """Right-button press starts a pan; left-button is handled elsewhere."""
        if event.button != 3:
            return
        if event.inaxes not in (self._ax_rgb, self._ax_depth):
            return
        if event.xdata is None or event.ydata is None:
            return

        self._pan_state = (
            event.xdata,
            event.ydata,
            self._ax_rgb.get_xlim(),
            self._ax_rgb.get_ylim(),
            event.inaxes,
        )

    def _on_mouse_release(self, event) -> None:
        """Right-button release ends the pan."""
        if event.button == 3:
            self._pan_state = None

    def _on_mouse_motion(self, event) -> None:
        """Translate both panels while right-button is held."""
        if self._pan_state is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0, xlim0, ylim0, _ = self._pan_state
        dx = x0 - event.xdata
        dy = y0 - event.ydata

        self._ax_rgb.set_xlim(xlim0[0] + dx, xlim0[1] + dx)
        self._ax_rgb.set_ylim(ylim0[0] + dy, ylim0[1] + dy)
        self._update_depth_clim_to_view()
        self._canvas.draw_idle()

    def _on_reset_view(self) -> None:
        """Restore both panels to the full image extent."""
        ref = self._im_rgb if self._im_rgb is not None else self._im_depth
        if ref is None:
            return
        data = ref.get_array()
        H, W = data.shape[:2]
        self._ax_rgb.set_xlim(0, W)
        self._ax_rgb.set_ylim(H, 0)  # origin='upper'
        self._update_depth_clim_to_view()
        self._canvas.draw_idle()

    def _update_depth_clim_to_view(self) -> None:
        """Rescale the depth colormap to the currently-visible pixels only."""
        if self._im_depth is None:
            return
        data = self._im_depth.get_array()
        if data is None:
            return
        arr = np.asarray(data)
        H, W = arr.shape[:2]

        x0, x1 = self._ax_depth.get_xlim()
        y0, y1 = self._ax_depth.get_ylim()

        col_lo = max(0, int(np.floor(min(x0, x1))))
        col_hi = min(W, int(np.ceil(max(x0, x1))))
        row_lo = max(0, int(np.floor(min(y0, y1))))
        row_hi = min(H, int(np.ceil(max(y0, y1))))
        if col_hi <= col_lo or row_hi <= row_lo:
            return

        view = arr[row_lo:row_hi, col_lo:col_hi]
        finite = np.isfinite(view)
        if not finite.any():
            return
        vmin = float(view[finite].min())
        vmax = float(view[finite].max())
        if vmax <= vmin:
            return
        self._im_depth.set_clim(vmin, vmax)

    def _on_canny_control_changed(self) -> None:
        """Slot for checkbox toggle — add or remove the Canny overlay."""
        if self._chk_canny.isChecked():
            self._draw_canny_overlay()
        else:
            if self._canny_artist is not None:
                try:
                    self._canny_artist.remove()
                except ValueError:
                    pass
                self._canny_artist = None
        self._canvas.draw_idle()

    def _on_t1_changed(self, value: int) -> None:
        """Slot for t1 slider movement — update label and redraw overlay."""
        self._lbl_t1.setText(f"t1: {value}")
        if self._chk_canny.isChecked():
            self._draw_canny_overlay()
            self._canvas.draw_idle()

    def _on_t2_changed(self, value: int) -> None:
        """Slot for t2 slider movement — update label and redraw overlay."""
        self._lbl_t2.setText(f"t2: {value}")
        if self._chk_canny.isChecked():
            self._draw_canny_overlay()
            self._canvas.draw_idle()

    def _on_slider_changed(self, value: int) -> None:
        """Frame scrubber moved — sync spinbox and load frame."""
        if self._block_frame_sync:
            return
        self._block_frame_sync = True
        self._spinbox.setValue(value)
        self._block_frame_sync = False
        self._display_frame(value)

    def _on_spinbox_changed(self, value: int) -> None:
        """Spinbox changed — sync slider and load frame."""
        if self._block_frame_sync:
            return
        self._block_frame_sync = True
        self._frame_slider.setValue(value)
        self._block_frame_sync = False
        self._display_frame(value)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Close the MKV on widget destruction."""
        try:
            self._mkv.__exit__(None, None, None)
        except Exception:
            pass
        super().closeEvent(event)
