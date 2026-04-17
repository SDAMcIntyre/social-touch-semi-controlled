"""
Interactive Tkinter GUI for inspecting the depth-weighted XYZ aggregation
pipeline frame-by-frame.

Provides a timeline slider, sticker selector dropdown, and a 3×2 embedded
matplotlib figure covering: mask overlay, sampled pixels, depth histogram,
weight profile, weighted-vs-plain scatter, and summary statistics.
"""

from __future__ import annotations

import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .depth_aggregation_recomputer import recompute_frame_diagnostics

# TYPE_CHECKING imports keep runtime dependency-free
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from preprocessing.common import VideoMP4Manager
    from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV
    from preprocessing.stickers_analysis.common.models.consolidated_tracks_manager import (
        ConsolidatedTracksManager,
    )


class DepthAggregationDiagnosticsGUI:
    """Tkinter-based interactive diagnostics viewer for the depth-weighted XYZ
    aggregation pipeline.

    Scrub through session frames with a timeline slider and inspect every
    intermediate variable: ellipse mask, sampled depth pixels, depth
    distribution, exponential weights, weighted-vs-plain scatter, and a
    numeric summary.

    The numeric values in the Summary panel are guaranteed to match the
    corresponding output of ``EllipseDepthExtractor`` because
    ``recompute_frame_diagnostics`` reuses the extractor's static helpers
    verbatim.

    Args:
        video_manager: ``VideoMP4Manager`` opened on the RGB video, used for
            total frame count and FPS.
        mkv_path: Path to the raw Kinect MKV file (source of point clouds).
        tracks_manager: ``ConsolidatedTracksManager`` loaded from the session's
            ``*_summary_2d_coordinates.csv``.
        sticker_diameter_mm: Physical sticker diameter used as the z-range
            plausibility guard (mm).  Must match the value used by the
            extractor.
        depth_weight_sigma: Exponential-decay sigma for depth weighting.
            Must match the extractor.
        title: Window title string.
        windowState: Initial window state — ``'maximized'`` or ``'normal'``.
    """

    _PC_CACHE_SIZE = 64

    def __init__(
        self,
        *,
        video_manager: "VideoMP4Manager",
        mkv_path: Path,
        tracks_manager: "ConsolidatedTracksManager",
        sticker_diameter_mm: float = 10.0,
        depth_weight_sigma: float = 0.3,
        title: str = "Depth Aggregation Diagnostics",
        windowState: str = "maximized",
    ) -> None:
        self._video_manager = video_manager
        self._mkv_path = Path(mkv_path)
        self._tracks_manager = tracks_manager
        self._sticker_diameter_mm = sticker_diameter_mm
        self._depth_weight_sigma = depth_weight_sigma
        self._title = title
        self._windowState = windowState

        # Colorbars added to self.fig must be explicitly removed before each
        # redraw — ax.cla() clears the plot axes but leaves colorbar axes alive.
        self._colorbars: list = []

        self._sticker_names: List[str] = tracks_manager.object_names
        self._selected_sticker: Optional[str] = (
            self._sticker_names[0] if self._sticker_names else None
        )

        self._is_paused = True
        self._update_job: Optional[str] = None
        self._pending_update_id: Optional[str] = None
        self._playback_delay_ms = int(1000 / self._video_manager.fps)

        # Opened in start(), closed after mainloop() returns.
        self._mkv: Optional[Any] = None

        # LRU cache: frame_index -> (H, W, 3) point cloud ndarray or None
        self._pc_cache: OrderedDict = OrderedDict()

        # Tkinter / matplotlib handles — populated in _setup_ui()
        self.root: Optional[tk.Tk] = None
        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.scale_var: Optional[tk.IntVar] = None
        self.speed_var: Optional[tk.DoubleVar] = None
        self.sticker_var: Optional[tk.StringVar] = None
        self.play_pause_button: Optional[ttk.Button] = None
        self.current_frame_label: Optional[ttk.Label] = None
        self.speed_label: Optional[ttk.Label] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the MKV, build the UI, and enter the Tkinter event loop."""
        from preprocessing.common.data_access.kinect_mkv_manager import KinectMKV

        with KinectMKV(self._mkv_path, seek_strategy="sequential") as mkv:
            self._mkv = mkv
            self._setup_ui()
            self.root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(self._title)

        self.scale_var = tk.IntVar(value=0)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.sticker_var = tk.StringVar(value=self._selected_sticker or "")

        if self._windowState.upper() in ("MAXIMIZED", "MAXIMISED"):
            self.root.state("zoomed")
        else:
            self.root.geometry("1440x900")

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.rowconfigure(0, weight=0)  # top bar
        self.root.rowconfigure(1, weight=1)  # canvas
        self.root.rowconfigure(2, weight=0)  # bottom bar
        self.root.columnconfigure(0, weight=1)

        self._build_top_bar()
        self._build_canvas()
        self._build_bottom_bar()
        self._bind_keys()

        # Render frame 0 once the window has settled
        self.root.after(100, lambda: self._update_ui_for_frame(0))

    def _build_top_bar(self) -> None:
        top_bar = ttk.Frame(self.root, padding=5)
        top_bar.grid(row=0, column=0, sticky="ew")

        ttk.Label(top_bar, text="Sticker:").pack(side=tk.LEFT, padx=5)
        combo = ttk.Combobox(
            top_bar,
            textvariable=self.sticker_var,
            values=self._sticker_names,
            state="readonly",
            width=24,
        )
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", self._on_sticker_change)

        info = ttk.Label(
            top_bar,
            text=(
                f"  |  diameter={self._sticker_diameter_mm} mm  "
                f"sigma={self._depth_weight_sigma}"
            ),
            foreground="gray",
        )
        info.pack(side=tk.LEFT, padx=10)

    def _build_canvas(self) -> None:
        self.fig, axes_array = plt.subplots(2, 4, figsize=(20, 10))
        self.axes = axes_array.flatten()
        # All 8 slots are used
        self.fig.tight_layout(pad=1.5)

        frame = ttk.Frame(self.root)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_bottom_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=5)
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(3, weight=1)

        total = len(self._video_manager)

        self.play_pause_button = ttk.Button(
            bar, text="▶ Play", command=self.toggle_play_pause, width=10
        )
        self.play_pause_button.grid(row=0, column=0, padx=5)

        self.current_frame_label = ttk.Label(bar, text=f"Frame: 0 / {total - 1}")
        self.current_frame_label.grid(row=0, column=1, padx=10)

        speed_frame = ttk.Frame(bar)
        speed_frame.grid(row=0, column=2, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="Speed: 1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(
            speed_frame,
            from_=0.25,
            to=8.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self._on_speed_change,
            length=120,
        ).pack(side=tk.LEFT)

        ttk.Scale(
            bar,
            from_=0,
            to=total - 1,
            orient=tk.HORIZONTAL,
            variable=self.scale_var,
            command=self._on_slider_change,
        ).grid(row=0, column=3, sticky="ew", padx=5)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda e: self.seek_to_frame(self.scale_var.get() - 1))
        self.root.bind("<Right>", lambda e: self.seek_to_frame(self.scale_var.get() + 1))
        self.root.bind(
            "<Control-Left>", lambda e: self.seek_to_frame(self.scale_var.get() - 10)
        )
        self.root.bind(
            "<Control-Right>", lambda e: self.seek_to_frame(self.scale_var.get() + 10)
        )
        self.root.bind("<space>", lambda e: self.toggle_play_pause())

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def toggle_play_pause(self) -> None:
        self._is_paused = not self._is_paused
        self.play_pause_button.config(
            text="▶ Play" if self._is_paused else "❚❚ Pause"
        )
        if not self._is_paused:
            self._run_playback_loop()

    def seek_to_frame(self, frame_val: Any) -> None:
        try:
            frame_num = int(float(frame_val))
            if not (0 <= frame_num < len(self._video_manager)):
                return
            self.scale_var.set(frame_num)
            self._update_ui_for_frame(frame_num)
        except (ValueError, IndexError):
            pass

    def quit(self) -> None:
        """Cancel pending callbacks, close the figure, and destroy the window."""
        if self._update_job:
            self.root.after_cancel(self._update_job)
        if self._pending_update_id:
            self.root.after_cancel(self._pending_update_id)
        if self.fig is not None:
            plt.close(self.fig)
        self.root.destroy()

    def _run_playback_loop(self) -> None:
        if self._is_paused:
            return
        current = self.scale_var.get()
        nxt = current + 1
        if nxt < len(self._video_manager):
            self.scale_var.set(nxt)
            self._update_ui_for_frame(nxt)
            self._update_job = self.root.after(
                self._playback_delay_ms, self._run_playback_loop
            )
        else:
            self._is_paused = True
            self.play_pause_button.config(text="▶ Play")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider_change(self, val: str) -> None:
        if self._pending_update_id:
            self.root.after_cancel(self._pending_update_id)
        frame = int(float(val))
        self._pending_update_id = self.root.after_idle(
            lambda: self._update_ui_for_frame(frame)
        )

    def _on_sticker_change(self, _event: Any) -> None:
        self._selected_sticker = self.sticker_var.get()
        self._update_ui_for_frame(self.scale_var.get())

    def _on_speed_change(self, val: str) -> None:
        speed = float(val)
        base_delay = 1000 / self._video_manager.fps
        self._playback_delay_ms = max(1, int(base_delay / speed))
        if self.speed_label:
            self.speed_label.config(text=f"Speed: {speed:.2f}x")

    # ------------------------------------------------------------------
    # Point-cloud LRU cache
    # ------------------------------------------------------------------

    def _get_frame_data(
        self, frame_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(point_cloud, color_bgr)`` for *frame_index*, using an LRU cache.

        Both arrays are read from the same ``KinectFrame`` to avoid double seeks.
        ``point_cloud`` is (H, W, 3) XYZ in mm; ``color_bgr`` is (H, W, 3) uint8.
        Either may be ``None`` if the sensor returned no data for that frame.
        """
        if frame_index in self._pc_cache:
            self._pc_cache.move_to_end(frame_index)
            return self._pc_cache[frame_index]

        pc: Optional[np.ndarray] = None
        color_bgr: Optional[np.ndarray] = None
        try:
            kinect_frame = self._mkv[frame_index]
            pc = kinect_frame.transformed_depth_point_cloud
            color_bgr = kinect_frame.color
        except Exception:
            pass

        if len(self._pc_cache) >= self._PC_CACHE_SIZE:
            self._pc_cache.popitem(last=False)
        self._pc_cache[frame_index] = (pc, color_bgr)
        return pc, color_bgr

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def _update_ui_for_frame(self, frame_num: int) -> None:
        total = len(self._video_manager)
        self.current_frame_label.config(text=f"Frame: {frame_num} / {total - 1}")

        if self._selected_sticker is None:
            self._render_status_message("No sticker selected.")
            return

        sticker_df = self._tracks_manager.get_coordinates_for_object(
            self._selected_sticker
        )
        if sticker_df is None:
            self._render_status_message(
                f"No data for sticker '{self._selected_sticker}'."
            )
            return

        row_df = sticker_df[sticker_df["frame_number"] == frame_num]
        if row_df.empty:
            self._render_status_message(
                f"No tracking data for frame {frame_num}\n"
                f"(sticker: {self._selected_sticker})"
            )
            return

        row = row_df.iloc[0]

        status = row.get("status", "Valid")
        if status in ("Failed", "Black Frame", "Ignored"):
            self._render_status_message(
                f"Frame {frame_num}  —  row status: '{status}'\n"
                f"Sticker: {self._selected_sticker}"
            )
            return

        point_cloud, color_bgr = self._get_frame_data(frame_num)
        if point_cloud is None:
            self._render_status_message(
                f"Point cloud unavailable for frame {frame_num}."
            )
            return

        diag = recompute_frame_diagnostics(
            point_cloud=point_cloud,
            tracked_obj_row=row,
            sticker_diameter_mm=self._sticker_diameter_mm,
            depth_weight_sigma=self._depth_weight_sigma,
        )
        diag = self._augment_pixel_coords(diag, point_cloud)

        crop = self._compute_crop_bounds(row, point_cloud)
        self._render_panels(diag, row, frame_num, point_cloud, color_bgr, crop)
        self.canvas.draw_idle()

    def _augment_pixel_coords(
        self,
        diag: Dict[str, Any],
        point_cloud: np.ndarray,
    ) -> Dict[str, Any]:
        """Add pixel positions of valid/invalid mask pixels to *diag*."""
        mask = diag.get("mask")
        if mask is None:
            empty_i = np.array([], dtype=int)
            empty_f = np.array([], dtype=float)
            diag.update(
                valid_pix_rows=empty_i,
                valid_pix_cols=empty_i,
                invalid_pix_rows=empty_i,
                invalid_pix_cols=empty_i,
                sampled_z_at_pix=empty_f,
            )
            return diag

        h, w, _ = point_cloud.shape
        clipped = mask[:h, :w]
        rows_all, cols_all = np.where(clipped)
        sampled_all = point_cloud[clipped]
        nonzero = np.any(sampled_all != 0, axis=1)
        nonan = ~np.any(np.isnan(sampled_all), axis=1)
        valid = nonzero & nonan

        valid_rows = rows_all[valid]
        valid_cols = cols_all[valid]
        valid_z = sampled_all[valid, 2]

        diag["valid_pix_rows"] = valid_rows
        diag["valid_pix_cols"] = valid_cols
        diag["invalid_pix_rows"] = rows_all[~valid]
        diag["invalid_pix_cols"] = cols_all[~valid]
        diag["sampled_z_at_pix"] = valid_z

        # Split valid pixels into candidates (kept by guard) vs guard-clipped.
        # When the guard did not fire, all valid pixels are candidates.
        z_min = diag.get("z_min", float("nan"))
        if len(valid_z) > 0 and not np.isnan(z_min) and diag.get("z_range_clipped", False):
            guard_kept = valid_z <= z_min + self._sticker_diameter_mm
            diag["candidate_pix_rows"] = valid_rows[guard_kept]
            diag["candidate_pix_cols"] = valid_cols[guard_kept]
            diag["guard_clipped_pix_rows"] = valid_rows[~guard_kept]
            diag["guard_clipped_pix_cols"] = valid_cols[~guard_kept]
        else:
            diag["candidate_pix_rows"] = valid_rows
            diag["candidate_pix_cols"] = valid_cols
            diag["guard_clipped_pix_rows"] = np.array([], dtype=int)
            diag["guard_clipped_pix_cols"] = np.array([], dtype=int)

        return diag

    def _compute_crop_bounds(
        self,
        row: pd.Series,
        point_cloud: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) crop around the ellipse centre."""
        H, W, _ = point_cloud.shape
        px = row.get("center_x", np.nan)
        py = row.get("center_y", np.nan)
        axes_major = row.get("axes_major", np.nan)
        axes_minor = row.get("axes_minor", np.nan)

        cx = int(float(px)) if pd.notna(px) else W // 2
        cy = int(float(py)) if pd.notna(py) else H // 2

        max_axis = max(
            float(axes_major) if pd.notna(axes_major) else 30.0,
            float(axes_minor) if pd.notna(axes_minor) else 30.0,
        )
        pad = max(30, int(max_axis * 2.2))

        x1 = max(0, cx - pad)
        x2 = min(W, cx + pad)
        y1 = max(0, cy - pad)
        y2 = min(H, cy + pad)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # Panel rendering
    # ------------------------------------------------------------------

    def _render_panels(
        self,
        diag: Dict[str, Any],
        row: pd.Series,
        frame_num: int,
        point_cloud: np.ndarray,
        color_bgr: Optional[np.ndarray],
        crop: Tuple[int, int, int, int],
    ) -> None:
        for cb in self._colorbars:
            cb.remove()
        self._colorbars.clear()

        for ax in self.axes:
            ax.cla()
        self.axes[7].set_axis_off()

        x1, y1, x2, y2 = crop
        self._panel0_rgb(self.axes[0], row, color_bgr, x1, y1, x2, y2)
        self._panel1_mask(self.axes[1], diag, row, point_cloud, x1, y1, x2, y2)
        self._panel2_sampled(self.axes[2], diag, row, point_cloud, x1, y1, x2, y2)
        self._panel3_histogram(self.axes[3], diag)
        self._panel4_weights(self.axes[4], diag)
        self._panel5_scatter(self.axes[5], diag)
        self._panel6_selected(self.axes[6], diag, point_cloud, x1, y1, x2, y2)
        self._panel7_summary(self.axes[7], diag, row, frame_num)

        self.fig.tight_layout(pad=1.5)

    def _render_status_message(self, message: str) -> None:
        for ax in self.axes:
            ax.cla()
            ax.set_axis_off()
        self.axes[2].text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=11,
            transform=self.axes[2].transAxes,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"),
        )
        self.canvas.draw_idle()

    # ---- Panel 0: RGB Overlay -------------------------------------------

    def _panel0_rgb(
        self,
        ax: plt.Axes,
        row: pd.Series,
        color_bgr: Optional[np.ndarray],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        ax.set_title("RGB Overlay")

        if color_bgr is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No colour frame", transform=ax.transAxes,
                    ha="center", va="center")
            return

        rgb_crop = color_bgr[y1:y2, x1:x2, ::-1]  # BGR → RGB, no copy
        ax.imshow(
            rgb_crop,
            aspect="auto",
            origin="upper",
            extent=(x1, x2, y2, y1),  # absolute image-pixel coords
        )
        ax.tick_params(labelsize=7)

        px = row.get("center_x", np.nan)
        py = row.get("center_y", np.nan)
        axes_major = row.get("axes_major", np.nan)
        axes_minor = row.get("axes_minor", np.nan)
        angle = row.get("angle", 0.0)

        if pd.notna(px) and pd.notna(py):
            cx = float(px)
            cy = float(py)

            if pd.notna(axes_major) and pd.notna(axes_minor):
                # axes_minor is the axis rotated by `angle` (cv2.fitEllipse convention;
                # see consolidated_tracks_gui.py:280).
                ellipse = mpatches.Ellipse(
                    (cx, cy),
                    width=float(axes_minor),
                    height=float(axes_major),
                    angle=float(angle) if pd.notna(angle) else 0.0,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)

            ax.axhline(cy, color="lime", linewidth=0.8, alpha=0.7)
            ax.axvline(cx, color="lime", linewidth=0.8, alpha=0.7)
            ax.plot(cx, cy, marker="+", color="lime", markersize=8, zorder=5)

    # ---- Panel 1: Mask Overlay ----------------------------------------

    def _panel1_mask(
        self,
        ax: plt.Axes,
        diag: Dict[str, Any],
        row: pd.Series,
        point_cloud: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        z_crop = point_cloud[y1:y2, x1:x2, 2].astype(float)
        z_display = z_crop.copy()
        z_display[z_display == 0] = np.nan

        extent = (x1, x2, y2, y1)  # absolute image-pixel coords (origin upper)
        if not np.all(np.isnan(z_display)):
            ax.imshow(
                z_display, cmap="gray", aspect="auto", origin="upper", extent=extent
            )
        else:
            ax.imshow(
                np.zeros_like(z_crop),
                cmap="gray",
                aspect="auto",
                origin="upper",
                extent=extent,
            )
        ax.tick_params(labelsize=7)

        px = row.get("center_x", np.nan)
        py = row.get("center_y", np.nan)
        axes_major = row.get("axes_major", np.nan)
        axes_minor = row.get("axes_minor", np.nan)
        angle = row.get("angle", 0.0)

        if pd.notna(px) and pd.notna(py):
            cx = float(px)
            cy = float(py)

            # Ellipse outline — axes_minor is the axis rotated by `angle`
            # (cv2.fitEllipse convention; see consolidated_tracks_gui.py:280).
            if pd.notna(axes_major) and pd.notna(axes_minor):
                ellipse = mpatches.Ellipse(
                    (cx, cy),
                    width=float(axes_minor),
                    height=float(axes_major),
                    angle=float(angle) if pd.notna(angle) else 0.0,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)

            # Crosshair at centroid
            ax.axhline(cy, color="lime", linewidth=0.8, alpha=0.7)
            ax.axvline(cx, color="lime", linewidth=0.8, alpha=0.7)
            ax.plot(cx, cy, marker="+", color="lime", markersize=8, zorder=5)

        ax.set_title("Mask Overlay")

        if diag["fallback_triggered"]:
            ax.text(
                0.5,
                0.05,
                "FALLBACK",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="orange",
                fontsize=9,
                fontweight="bold",
            )

    # ---- Panel 2: Sampled Pixels ----------------------------------------

    def _panel2_sampled(
        self,
        ax: plt.Axes,
        diag: Dict[str, Any],
        row: pd.Series,
        point_cloud: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        ax.set_title("Sampled Pixels")

        z_crop = point_cloud[y1:y2, x1:x2, 2].astype(float)
        ax.imshow(
            np.zeros_like(z_crop),
            cmap="gray",
            aspect="auto",
            origin="upper",
            vmin=0,
            vmax=1,
            extent=(x1, x2, y2, y1),  # absolute image-pixel coords
        )
        ax.tick_params(labelsize=7)

        # Invalid mask pixels (black dots)
        inv_r = diag.get("invalid_pix_rows", np.array([], dtype=int))
        inv_c = diag.get("invalid_pix_cols", np.array([], dtype=int))
        if len(inv_r):
            ax.scatter(
                inv_c,
                inv_r,
                c="black",
                s=2,
                alpha=0.6,
                linewidths=0,
            )

        # Valid mask pixels coloured by z
        val_r = diag.get("valid_pix_rows", np.array([], dtype=int))
        val_c = diag.get("valid_pix_cols", np.array([], dtype=int))
        val_z = diag.get("sampled_z_at_pix", np.array([], dtype=float))
        if len(val_r):
            sc = ax.scatter(
                val_c,
                val_r,
                c=val_z,
                cmap="coolwarm",
                s=4,
                linewidths=0,
            )
            self._colorbars.append(
                self.fig.colorbar(sc, ax=ax, shrink=0.6, label="z (mm)")
            )

        if diag["fallback_triggered"]:
            ax.text(
                0.5,
                0.5,
                "FALLBACK",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="orange",
                fontsize=14,
                fontweight="bold",
                alpha=0.8,
            )

    # ---- Panel 3: Depth Distribution ------------------------------------

    def _panel3_histogram(self, ax: plt.Axes, diag: Dict[str, Any]) -> None:
        ax.set_title("Depth Distribution")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("count")

        zs = diag["zs"]
        if len(zs) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        z_min = diag["z_min"]
        z_max = diag["z_max"]
        threshold = z_min + self._sticker_diameter_mm
        fired = diag["z_range_clipped"]

        bins = min(40, max(10, len(zs) // 3))
        counts, edges, patches = ax.hist(zs, bins=bins, color="steelblue", alpha=0.8)

        # Shade clipped bars red
        if fired:
            for patch, left in zip(patches, edges[:-1]):
                if left > threshold:
                    patch.set_facecolor("red")
                    patch.set_alpha(0.6)
            ax.axvline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"guard={threshold:.1f}",
            )

        ax.axvline(z_min, color="navy", linestyle=":", linewidth=1.0, label=f"z_min={z_min:.1f}")
        ax.axvline(z_max, color="darkred", linestyle=":", linewidth=1.0, label=f"z_max={z_max:.1f}")
        ax.legend(fontsize=7, loc="upper right")

        if fired:
            ax.text(
                0.02,
                0.97,
                "GUARD FIRED",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                fontsize=8,
                fontweight="bold",
            )

    # ---- Panel 4: Weight Profile -----------------------------------------

    def _panel4_weights(self, ax: plt.Axes, diag: Dict[str, Any]) -> None:
        ax.set_title("Weight Profile")
        ax.set_xlabel("z_norm (0 = nearest)")
        ax.set_ylabel("weight")

        z_norm = diag["z_norm"]
        weights = diag["weights"]
        if len(z_norm) == 0:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        if diag["uniform_weights"]:
            ax.axhline(1.0, color="steelblue", linewidth=1.5)
            ax.scatter(z_norm, weights, c="steelblue", s=8, zorder=5)
            ax.text(
                0.5,
                0.5,
                "Uniform weights",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )
        else:
            # Smooth exponential curve
            t = np.linspace(0.0, 1.0, 200)
            w_curve = np.exp(-t / self._depth_weight_sigma)
            ax.plot(t, w_curve, color="steelblue", linewidth=1.5, label="exp decay")

            ax.scatter(z_norm, weights, c="steelblue", s=10, zorder=5)

            # Vertical lines at weighted and plain median z_norm
            cz_min = diag["cz_min"]
            cz_max = diag["cz_max"]
            if cz_max > cz_min:
                wz_n = (diag["weighted_z"] - cz_min) / (cz_max - cz_min)
                pz_n = (diag["plain_z"] - cz_min) / (cz_max - cz_min)
                ax.axvline(wz_n, color="red", linewidth=1.2, linestyle="--",
                           label=f"w-med z_n={wz_n:.2f}")
                ax.axvline(pz_n, color="green", linewidth=1.2, linestyle="--",
                           label=f"plain z_n={pz_n:.2f}")

            ax.legend(fontsize=7, loc="upper right")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.15)

    # ---- Panel 5: Weighted vs Plain Scatter ------------------------------

    def _panel5_scatter(self, ax: plt.Axes, diag: Dict[str, Any]) -> None:
        ax.set_title("Weighted vs Plain")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm) — low = near camera")
        ax.invert_yaxis()  # camera looks downward: small z = closest = top

        cxs = diag["candidate_xs"]
        czs = diag["candidate_zs"]
        weights = diag["weights"]

        if len(cxs) == 0:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        # Candidate pixels
        w_norm = weights / weights.max() if weights.max() > 0 else weights
        sc = ax.scatter(
            cxs,
            czs,
            c=weights,
            cmap="coolwarm",
            s=20 * (0.3 + w_norm),
            alpha=0.7,
            linewidths=0,
            label="candidates",
        )
        self._colorbars.append(
            self.fig.colorbar(sc, ax=ax, shrink=0.6, label="weight")
        )

        # Clipped pixels
        clip_xs = diag["clipped_xs"]
        clip_zs = diag["clipped_zs"]
        if len(clip_xs):
            ax.scatter(
                clip_xs,
                clip_zs,
                c="red",
                marker="x",
                s=15,
                alpha=0.4,
                label="clipped",
            )

        wx = diag["weighted_x"]
        wz = diag["weighted_z"]
        px_ = diag["plain_x"]
        pz = diag["plain_z"]

        if not np.isnan(wx):
            ax.scatter(
                [wx], [wz], c="red", marker="*", s=120, zorder=6, label=f"w-med ({wx:.1f}, {wz:.1f})"
            )
        if not np.isnan(px_):
            ax.scatter(
                [px_], [pz], c="green", marker="D", s=60, zorder=6,
                label=f"plain ({px_:.1f}, {pz:.1f})"
            )

        # Arrow from plain to weighted, annotated with shift
        if not (np.isnan(wx) or np.isnan(px_)):
            dx = wx - px_
            dz = wz - pz
            shift_mm = float(np.sqrt(dx ** 2 + dz ** 2))
            if shift_mm > 0.01:
                ax.annotate(
                    "",
                    xy=(wx, wz),
                    xytext=(px_, pz),
                    arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
                )
                ax.text(
                    (wx + px_) / 2,
                    (wz + pz) / 2,
                    f" {shift_mm:.1f}mm",
                    color="purple",
                    fontsize=7,
                )

        ax.legend(fontsize=7, loc="best")

    # ---- Panel 6: Selected Pixels ----------------------------------------

    def _panel6_selected(
        self,
        ax: plt.Axes,
        diag: Dict[str, Any],
        point_cloud: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Depth crop with guard-clipped pixels in grey and candidate pixels in red."""
        ax.set_title("Selected Pixels")
        ax.set_axis_off()

        z_crop = point_cloud[y1:y2, x1:x2, 2].astype(float)
        z_display = z_crop.copy()
        z_display[z_display == 0] = np.nan

        if not np.all(np.isnan(z_display)):
            ax.imshow(z_display, cmap="gray", aspect="auto", origin="upper")
        else:
            ax.imshow(np.zeros_like(z_crop), cmap="gray", aspect="auto", origin="upper")

        # Guard-clipped pixels (excluded by sticker-size guard) — grey
        gc_r = diag.get("guard_clipped_pix_rows", np.array([], dtype=int))
        gc_c = diag.get("guard_clipped_pix_cols", np.array([], dtype=int))
        if len(gc_r):
            ax.scatter(gc_c - x1, gc_r - y1, c="grey", s=6, alpha=0.6,
                       linewidths=0, label="guard-clipped")

        # Candidate pixels (used for aggregation) — red
        cand_r = diag.get("candidate_pix_rows", np.array([], dtype=int))
        cand_c = diag.get("candidate_pix_cols", np.array([], dtype=int))
        if len(cand_r):
            ax.scatter(cand_c - x1, cand_r - y1, c="red", s=6, alpha=0.8,
                       linewidths=0, label="candidates")

        # Weighted median position (in pixel space, approximated via point cloud lookup)
        wx = diag.get("weighted_x", float("nan"))
        wy = diag.get("weighted_y", float("nan"))
        if not (np.isnan(wx) or np.isnan(wy)):
            # Find the pixel whose XY mm values are closest to the weighted median
            if len(cand_r):
                pc_cand = point_cloud[cand_r, cand_c]  # (N, 3)
                dist = (pc_cand[:, 0] - wx) ** 2 + (pc_cand[:, 1] - wy) ** 2
                idx = int(np.argmin(dist))
                med_col = int(cand_c[idx]) - x1
                med_row = int(cand_r[idx]) - y1
                ax.scatter([med_col], [med_row], c="red", marker="*", s=120,
                           zorder=6, label="w-median")

        if diag.get("fallback_triggered", False):
            ax.text(0.5, 0.5, "FALLBACK", transform=ax.transAxes,
                    ha="center", va="center", color="orange",
                    fontsize=14, fontweight="bold", alpha=0.8)

    # ---- Panel 7: Summary Text -------------------------------------------

    def _panel7_summary(  # was _panel6_summary
        self,
        ax: plt.Axes,
        diag: Dict[str, Any],
        row: pd.Series,
        frame_num: int,
    ) -> None:
        ax.set_axis_off()
        ax.set_title("Summary")

        def _fmt(v: Any, decimals: int = 2) -> str:
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                return "NaN"
            if isinstance(v, float):
                return f"{v:.{decimals}f}"
            return str(v)

        px = _fmt(row.get("center_x", np.nan))
        py = _fmt(row.get("center_y", np.nan))
        am = _fmt(row.get("axes_major", np.nan))
        an = _fmt(row.get("axes_minor", np.nan))
        ang = _fmt(row.get("angle", np.nan), 1)

        n_total = len(diag["zs"])
        n_cand = len(diag["candidate_xs"])
        n_clipped = len(diag["clipped_xs"])

        wx = _fmt(diag["weighted_x"])
        wy = _fmt(diag["weighted_y"])
        wz = _fmt(diag["weighted_z"])
        px_ = _fmt(diag["plain_x"])
        py_ = _fmt(diag["plain_y"])
        pz_ = _fmt(diag["plain_z"])

        dz = diag["weighted_z"] - diag["plain_z"]
        dz_str = _fmt(dz) if not np.isnan(dz) else "NaN"

        lines = [
            f"Frame: {frame_num}",
            "",
            "── Inputs ──",
            f"centre:      ({px}, {py}) px",
            f"axes:        major={am}  minor={an}",
            f"angle:       {ang} deg",
            "",
            "── Depth Stats ──",
            f"z_min:       {_fmt(diag['z_min'])} mm",
            f"z_max:       {_fmt(diag['z_max'])} mm",
            f"z_range:     {_fmt(diag['z_range'])} mm",
            f"z_std:       {_fmt(diag['z_std'])} mm",
            f"n_pixels:    {n_total}",
            "",
            "── Guard ──",
            f"diameter:    {self._sticker_diameter_mm} mm",
            f"guard fired: {diag['z_range_clipped']}",
            f"n_clipped:   {n_clipped}  (kept {n_cand})",
            "",
            "── Weighting ──",
            f"sigma:       {self._depth_weight_sigma}",
            f"uniform:     {diag['uniform_weights']}",
            f"w_range:     [{_fmt(diag['weights'].min() if len(diag['weights']) else np.nan)}"
            f", {_fmt(diag['weights'].max() if len(diag['weights']) else np.nan)}]",
            "",
            "── Results ──",
            f"weighted:    ({wx}, {wy}, {wz}) mm",
            f"plain:       ({px_}, {py_}, {pz_}) mm",
            f"Δz (w-plain): {dz_str} mm",
        ]

        if diag["fallback_triggered"]:
            lines.insert(0, "*** FALLBACK TRIGGERED ***")
            lines.insert(1, "")

        text = "\n".join(lines)
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            fontfamily="monospace",
            verticalalignment="top",
        )
