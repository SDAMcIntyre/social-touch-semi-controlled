# file: motion_correction_orchestrator.py
"""Orchestrates the full motion-correction pipeline for XYZ sticker data.

Two modes:
- **correct** — apply one chosen filter, save ``_xyz_corrected.csv`` and
  diagnostics for the chosen filter.
- **compare** — run all registered filters, produce overlay plots only
  (no CSV is written).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..data_access.xyz_data_filehandler import XYZDataFileHandler
from .motion_filter_factory import FilterChoice, MotionFilterFactory
from .outlier_detector import OutlierConfig, OutlierDetector
from .diagnostics_plotter import DiagnosticsPlotter

_AXIS_COLUMNS = ("x_mm", "y_mm", "z_mm")


class MotionCorrectionOrchestrator:
    """Coordinates loading, outlier removal, filtering, saving, and plotting.

    Args:
        outlier_config: Configuration for the :class:`OutlierDetector`.
                        Defaults to :class:`OutlierConfig` built-in defaults.
        sampling_rate_hz: Recording frame rate (default 30.0).
    """

    def __init__(
        self,
        outlier_config: OutlierConfig | None = None,
        sampling_rate_hz: float = 30.0,
    ) -> None:
        self._detector = OutlierDetector(outlier_config)
        self._fps = sampling_rate_hz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_correct(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        diagnostics_dir: str | Path,
        filter_method: str | FilterChoice = "butterworth",
        filter_params: Dict[str, Any] | None = None,
    ) -> None:
        """Apply the chosen filter and save the corrected CSV + diagnostics.

        Args:
            input_csv: Path to the raw ``_xyz_tracked.csv`` file.
            output_csv: Destination path for the corrected ``_xyz_corrected.csv``.
            diagnostics_dir: Directory for PNG diagnostic plots.
            filter_method: Filter to apply (``"butterworth"`` or ``"savgol"``).
            filter_params: Optional dict with keys ``"butterworth"`` / ``"savgol"``
                           mapping to constructor keyword arguments.
        """
        sticker_data = XYZDataFileHandler.load(str(input_csv))
        filt = MotionFilterFactory.get_filter(filter_method, filter_params)
        plotter = DiagnosticsPlotter(diagnostics_dir, self._fps)

        corrected_data: Dict[str, pd.DataFrame] = {}

        for sticker_name, raw_df in sticker_data.items():
            corrected_df, hard_mask, stat_mask = self._process_sticker(
                sticker_name, raw_df, [filt]
            )
            # Use the single filtered result
            corrected_data[sticker_name] = corrected_df[filt.name()]

            plotter.plot_position_overlay(
                sticker_name,
                raw_df,
                {filt.name(): corrected_df[filt.name()]},
            )
            plotter.plot_velocity_acceleration(
                sticker_name,
                raw_df,
                corrected_df[filt.name()],
                corrected_label=filt.name(),
            )
            plotter.plot_outlier_summary(
                sticker_name, raw_df, hard_mask, stat_mask
            )

        XYZDataFileHandler.save(corrected_data, str(output_csv))

    def run_compare(
        self,
        input_csv: str | Path,
        diagnostics_dir: str | Path,
        filter_params: Dict[str, Any] | None = None,
    ) -> None:
        """Run all filters and produce overlay comparison plots (no CSV saved).

        Args:
            input_csv: Path to the raw ``_xyz_tracked.csv`` file.
            diagnostics_dir: Directory for PNG diagnostic plots.
            filter_params: Optional dict with per-filter constructor arguments.
        """
        sticker_data = XYZDataFileHandler.load(str(input_csv))
        filters = MotionFilterFactory.get_all_filters(filter_params)
        plotter = DiagnosticsPlotter(diagnostics_dir, self._fps)

        for sticker_name, raw_df in sticker_data.items():
            corrected_df, hard_mask, stat_mask = self._process_sticker(
                sticker_name, raw_df, filters
            )

            for label, corr_df in corrected_df.items():
                safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
                plotter.plot_position_overlay(
                    sticker_name,
                    raw_df,
                    {label: corr_df},
                    filename_suffix=f"_{safe_label}",
                )
            # Velocity/acceleration: compare raw vs first filter as representative
            first_name = next(iter(corrected_df))
            plotter.plot_velocity_acceleration(
                sticker_name,
                raw_df,
                corrected_df[first_name],
                corrected_label=first_name,
            )
            plotter.plot_outlier_summary(
                sticker_name, raw_df, hard_mask, stat_mask
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_sticker(
        self,
        sticker_name: str,
        raw_df: pd.DataFrame,
        filters,
    ):
        """Detect outliers and apply each filter to every axis.

        Returns:
            Tuple of:
            - ``corrected_results`` — dict mapping filter name → corrected DataFrame
            - ``hard_mask`` — boolean array of hard-threshold outliers (per axis OR-combined)
            - ``stat_mask`` — boolean array of statistical outliers (per axis OR-combined)
        """
        n = len(raw_df)
        axes = [c for c in _AXIS_COLUMNS if c in raw_df.columns]

        if not axes or n == 0:
            # Nothing to process; pass through raw for every filter
            return {f.name(): raw_df.copy() for f in filters}, None, None

        # Accumulate masks across axes (OR-combined)
        combined_hard = np.zeros(n, dtype=bool)
        combined_stat = np.zeros(n, dtype=bool)

        # Per-axis, per-filter cleaned signals
        axis_results: Dict[str, Dict[str, np.ndarray]] = {col: {} for col in axes}

        for col in axes:
            signal = raw_df[col].to_numpy(dtype=float)

            # Step 1 — interpolate existing NaN gaps
            signal_no_nan = self._fill_nans(signal)

            # Step 2 — detect outliers on the NaN-filled signal
            cfg = self._detector._cfg
            hard_mask = None
            stat_mask = None

            if cfg.enable_hard_threshold:
                # Temporarily enable only hard threshold
                from .outlier_detector import OutlierConfig, OutlierDetector
                hard_cfg = OutlierConfig(
                    enable_hard_threshold=True,
                    max_velocity_mm_per_s=cfg.max_velocity_mm_per_s,
                    max_acceleration_mm_per_s2=cfg.max_acceleration_mm_per_s2,
                    enable_statistical=False,
                )
                hard_mask = OutlierDetector(hard_cfg).detect(signal_no_nan, self._fps)
                combined_hard |= hard_mask

            if cfg.enable_statistical:
                from .outlier_detector import OutlierConfig, OutlierDetector
                stat_cfg = OutlierConfig(
                    enable_hard_threshold=False,
                    enable_statistical=True,
                    mad_multiplier=cfg.mad_multiplier,
                )
                stat_mask = OutlierDetector(stat_cfg).detect(signal_no_nan, self._fps)
                combined_stat |= stat_mask

            full_mask = self._detector.detect(signal_no_nan, self._fps)

            # Step 3 — replace outliers by interpolation
            signal_clean = self._detector.interpolate_outliers(signal_no_nan, full_mask)

            # Guard: if signal is too short for any filter, fall back to cleaned signal
            if len(signal_clean) < 3:
                for filt in filters:
                    axis_results[col][filt.name()] = signal_clean.copy()
                continue

            # Step 4 — apply each filter
            for filt in filters:
                try:
                    axis_results[col][filt.name()] = filt.filter(signal_clean, self._fps)
                except Exception:
                    # Graceful degradation: use the cleaned (non-filtered) signal
                    axis_results[col][filt.name()] = signal_clean.copy()

        # Build corrected DataFrames (one per filter)
        corrected_results: Dict[str, pd.DataFrame] = {}
        for filt in filters:
            df_copy = raw_df.copy()
            for col in axes:
                df_copy[col] = axis_results[col][filt.name()]
            corrected_results[filt.name()] = df_copy

        return corrected_results, combined_hard, combined_stat

    @staticmethod
    def _fill_nans(signal: np.ndarray) -> np.ndarray:
        """Linear interpolation of NaN values; edge-clamps at boundaries."""
        if not np.any(np.isnan(signal)):
            return signal.copy()
        result = signal.copy()
        nan_mask = np.isnan(result)
        valid = np.flatnonzero(~nan_mask)
        if len(valid) == 0:
            result[:] = 0.0
            return result
        result[nan_mask] = np.interp(
            np.flatnonzero(nan_mask), valid, result[valid]
        )
        return result
