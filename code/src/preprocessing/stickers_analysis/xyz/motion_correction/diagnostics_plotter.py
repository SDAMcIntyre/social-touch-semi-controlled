# file: diagnostics_plotter.py
"""Diagnostic plots for motion correction.

Produces three plot files per sticker:
1. **position_overlay** — raw vs corrected (or raw vs all filters in compare mode)
   for each axis (x, y, z).
2. **velocity_acceleration** — velocity and acceleration magnitude of raw vs corrected.
3. **outlier_summary** — raw signal with outlier frames marked by detection origin.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless / script runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .motion_filter_interface import MotionFilterInterface

# Columns that carry position data (axes).
_AXIS_COLUMNS = ("x_mm", "y_mm", "z_mm")


def _velocity_and_accel(
    position: np.ndarray, sampling_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return velocity (mm/s) and acceleration (mm/s²) via np.gradient."""
    dt = 1.0 / sampling_rate_hz
    vel = np.gradient(position, dt)
    acc = np.gradient(vel, dt)
    return vel, acc


class DiagnosticsPlotter:
    """Saves diagnostic PNG files for one sticker's correction results.

    Args:
        output_dir: Directory where PNG files will be written.
        sampling_rate_hz: Recording frame rate (default 30.0 for Azure Kinect).
        dpi: Figure resolution for saved PNGs (default 120).
    """

    def __init__(
        self,
        output_dir: str | Path,
        sampling_rate_hz: float = 30.0,
        dpi: int = 120,
    ) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._fps = sampling_rate_hz
        self._dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_position_overlay(
        self,
        sticker_name: str,
        raw_df: pd.DataFrame,
        corrected_series: Dict[str, pd.DataFrame],
        filename_suffix: str = "",
    ) -> None:
        """Save a position overlay plot.

        Args:
            sticker_name: e.g. ``"sticker_blue"``.
            raw_df: DataFrame with columns x_mm / y_mm / z_mm for the raw signal.
            corrected_series: Mapping of label → DataFrame (same columns).
                              In *correct* mode this has one entry labelled
                              with the filter name; in *compare* mode it has
                              one entry per filter.
        """
        axes = [c for c in _AXIS_COLUMNS if c in raw_df.columns]
        n_axes = len(axes)
        if n_axes == 0:
            return

        fig, axs = plt.subplots(n_axes, 1, figsize=(14, 3 * n_axes), sharex=True)
        if n_axes == 1:
            axs = [axs]

        frames = raw_df.index.to_numpy()

        for ax, col in zip(axs, axes):
            ax.plot(frames, raw_df[col].to_numpy(), color="grey", lw=0.8,
                    alpha=0.7, label="Raw")
            for label, df in corrected_series.items():
                ax.plot(frames, df[col].to_numpy(), lw=1.2, label=label)
            ax.set_ylabel(col)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, lw=0.4, alpha=0.5)

        axs[-1].set_xlabel("Frame")
        fig.suptitle(f"{sticker_name} — Position overlay", fontsize=11)
        fig.tight_layout()
        fname = self._out / f"{sticker_name}_position_overlay{filename_suffix}.png"
        fig.savefig(fname, dpi=self._dpi)
        plt.close(fig)

    def plot_velocity_acceleration(
        self,
        sticker_name: str,
        raw_df: pd.DataFrame,
        corrected_df: pd.DataFrame,
        corrected_label: str = "Corrected",
    ) -> None:
        """Save velocity and acceleration magnitude comparison plot.

        Only saves the comparison for the *chosen* corrected signal (not all
        filters); call once per run.
        """
        axes = [c for c in _AXIS_COLUMNS if c in raw_df.columns]
        if not axes:
            return

        frames = raw_df.index.to_numpy()

        def _magnitude(df: pd.DataFrame, axes: list) -> Tuple[np.ndarray, np.ndarray]:
            vels, accs = [], []
            for col in axes:
                v, a = _velocity_and_accel(df[col].to_numpy(), self._fps)
                vels.append(v ** 2)
                accs.append(a ** 2)
            return np.sqrt(np.sum(vels, axis=0)), np.sqrt(np.sum(accs, axis=0))

        raw_vel, raw_acc = _magnitude(raw_df, axes)
        cor_vel, cor_acc = _magnitude(corrected_df, axes)

        fig, (ax_vel, ax_acc) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        ax_vel.plot(frames, raw_vel, color="grey", lw=0.8, alpha=0.7, label="Raw")
        ax_vel.plot(frames, cor_vel, lw=1.2, label=corrected_label)
        ax_vel.set_ylabel("Speed (mm/s)")
        ax_vel.legend(fontsize=8)
        ax_vel.grid(True, lw=0.4, alpha=0.5)

        ax_acc.plot(frames, raw_acc, color="grey", lw=0.8, alpha=0.7, label="Raw")
        ax_acc.plot(frames, cor_acc, lw=1.2, label=corrected_label)
        ax_acc.set_ylabel("Acceleration (mm/s²)")
        ax_acc.set_xlabel("Frame")
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, lw=0.4, alpha=0.5)

        fig.suptitle(f"{sticker_name} — Velocity & acceleration", fontsize=11)
        fig.tight_layout()
        fname = self._out / f"{sticker_name}_velocity_acceleration.png"
        fig.savefig(fname, dpi=self._dpi)
        plt.close(fig)

    def plot_outlier_summary(
        self,
        sticker_name: str,
        raw_df: pd.DataFrame,
        hard_mask: np.ndarray | None,
        stat_mask: np.ndarray | None,
    ) -> None:
        """Save an outlier summary plot for each axis.

        Outlier frames are overlaid as scatter points colour-coded by the
        detection method that flagged them (hard threshold, statistical, or
        both).
        """
        axes = [c for c in _AXIS_COLUMNS if c in raw_df.columns]
        n_axes = len(axes)
        if n_axes == 0:
            return

        frames = raw_df.index.to_numpy()

        # Combine masks
        hard = hard_mask if hard_mask is not None else np.zeros(len(frames), dtype=bool)
        stat = stat_mask if stat_mask is not None else np.zeros(len(frames), dtype=bool)
        both = hard & stat
        only_hard = hard & ~stat
        only_stat = stat & ~hard

        fig, axs = plt.subplots(n_axes, 1, figsize=(14, 3 * n_axes), sharex=True)
        if n_axes == 1:
            axs = [axs]

        for ax, col in zip(axs, axes):
            signal = raw_df[col].to_numpy()
            ax.plot(frames, signal, color="grey", lw=0.8, alpha=0.7, label="Raw")

            if np.any(only_hard):
                ax.scatter(frames[only_hard], signal[only_hard],
                           c="orange", s=20, zorder=3, label="Hard threshold")
            if np.any(only_stat):
                ax.scatter(frames[only_stat], signal[only_stat],
                           c="dodgerblue", s=20, zorder=3, label="Statistical")
            if np.any(both):
                ax.scatter(frames[both], signal[both],
                           c="red", s=25, zorder=4, label="Both")

            ax.set_ylabel(col)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, lw=0.4, alpha=0.5)

        axs[-1].set_xlabel("Frame")
        fig.suptitle(f"{sticker_name} — Outlier summary", fontsize=11)
        fig.tight_layout()
        fname = self._out / f"{sticker_name}_outlier_summary.png"
        fig.savefig(fname, dpi=self._dpi)
        plt.close(fig)
