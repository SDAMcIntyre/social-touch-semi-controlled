"""Pipeline orchestrator for IFF tuning curve plots.

For each selected touch feature, bins touches into equal-width ranges across
all sessions and renders:

- per-session dual Y-axis plots (IFF line left, touch count bars right)
- all-sessions overlay plots (one IFF line per session, shared axis)

Output layout::

    4_analysed/stimulus_iff_tuning_curves/
        iff_tuning_sentinel.json
        {feature}/
            {gesture_subset}/
                {session_id}_tuning.png
                overlay_tuning.png
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import STIMULUS_IFF_TUNING_CURVES
from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,
    IFF_METRICS,
    TOUCH_ID_COLS,
    session_id_from_path,
)
from analysis.receptive_field_mapping.pipelines.rf_touch_feature_radar_pipeline import (
    _find_feature_csv,
)
from analysis.receptive_field_mapping.rendering.rf_iff_tuning_renderer import (
    _bin_data,
    _cat_display,
    render_session_tuning_curve,
    render_overlay_tuning_curve,
)
from analysis.receptive_field_mapping.rendering.rf_stimulus_session_comparison_renderer import (
    assign_session_colors,
)

logger = logging.getLogger(__name__)

_GESTURE_COL = "gesture_type"

_GESTURE_SUBSETS = ["all", "tap", "stroke", "stroke_proximal", "stroke_distal"]
_MIN_ROWS = 5


# ---------------------------------------------------------------------------
# Gesture filtering
# ---------------------------------------------------------------------------


def _filter_gesture(df: pd.DataFrame, gesture_subset: str) -> pd.DataFrame:
    """Return rows of *df* matching *gesture_subset*.

    Parameters
    ----------
    df:
        DataFrame that must contain the ``gesture_type`` column.
    gesture_subset:
        One of ``"all"``, ``"tap"``, ``"stroke"``, ``"stroke_proximal"``,
        ``"stroke_distal"``.

    Returns
    -------
    pd.DataFrame
        Filtered (or unfiltered) DataFrame.

    Raises
    ------
    ValueError
        If *gesture_subset* is not one of the recognised values.
    """
    if gesture_subset == "all":
        return df
    if gesture_subset == "tap":
        return df[df[_GESTURE_COL] == "tap"]
    if gesture_subset == "stroke":
        return df[df[_GESTURE_COL].isin({"stroke_proximal", "stroke_distal"})]
    if gesture_subset in {"stroke_proximal", "stroke_distal"}:
        return df[df[_GESTURE_COL] == gesture_subset]
    raise ValueError(
        f"_filter_gesture: unrecognised gesture_subset '{gesture_subset}'. "
        f"Valid values: {_GESTURE_SUBSETS}"
    )


# ---------------------------------------------------------------------------
# Bin edge / window computation
# ---------------------------------------------------------------------------


def _compute_bin_windows(
    pooled_series: pd.Series,
    n_bins: int,
    clip_percentile: float,
    overlap_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute parallel low/high edge arrays for sliding-window bins.

    Returns (bin_low, bin_high), each shape (n_bins,).
    overlap_ratio=0 reproduces disjoint half-open windows (last window inclusive).
    """
    if not (0.0 <= overlap_ratio <= 0.5):
        raise ValueError(
            f"_compute_bin_windows: overlap_ratio must be in [0.0, 0.5], "
            f"got {overlap_ratio}."
        )

    values = pooled_series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError(
            "_compute_bin_windows: pooled series is entirely NaN — "
            "cannot compute bin windows."
        )

    if clip_percentile > 0.0:
        lo = float(np.percentile(values, clip_percentile))
        hi = float(np.percentile(values, 100.0 - clip_percentile))
    else:
        lo = float(values.min())
        hi = float(values.max())

    if lo >= hi:
        raise ValueError(
            f"_compute_bin_windows: after clipping at {clip_percentile}th / "
            f"{100.0 - clip_percentile}th percentile, feature range collapsed to "
            f"[{lo}, {hi}] — all values are effectively identical. "
            f"Increase clip_percentile or remove this feature from tuning_features."
        )

    w = (hi - lo) / n_bins
    stride = w * (1.0 - overlap_ratio)
    bin_low = np.array([lo + i * stride for i in range(n_bins)], dtype=float)
    bin_high = bin_low + w
    bin_high[-1] = hi
    return bin_low, bin_high


def _compute_bin_edges(
    pooled_series: pd.Series,
    n_bins: int,
    clip_percentile: float,
) -> np.ndarray:
    """Compute global bin edges for a feature column across all sessions.

    Parameters
    ----------
    pooled_series:
        All values for this feature concatenated across sessions (may contain NaN).
    n_bins:
        Number of equal-width bins.
    clip_percentile:
        Percentile used for symmetric clipping.  The series is clipped to
        ``[clip_percentile, 100 - clip_percentile]`` before computing edges.
        Pass ``0.0`` to disable clipping.

    Returns
    -------
    np.ndarray
        Array of ``n_bins + 1`` bin edges (float64).

    Raises
    ------
    ValueError
        If fewer than 2 unique values remain after clipping (degenerate feature).
    """
    values = pooled_series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError(
            f"_compute_bin_edges: pooled series is entirely NaN — "
            f"cannot compute bin edges."
        )

    if clip_percentile > 0.0:
        lo = float(np.percentile(values, clip_percentile))
        hi = float(np.percentile(values, 100.0 - clip_percentile))
    else:
        lo = float(values.min())
        hi = float(values.max())

    if lo >= hi:
        raise ValueError(
            f"_compute_bin_edges: after clipping at {clip_percentile}th / "
            f"{100.0 - clip_percentile}th percentile, feature range collapsed to "
            f"[{lo}, {hi}] — all values are effectively identical. "
            f"Increase clip_percentile or remove this feature from tuning_features."
        )

    return np.linspace(lo, hi, n_bins + 1)


# ---------------------------------------------------------------------------
# Global IFF ylim and count max
# ---------------------------------------------------------------------------


def _compute_global_iff_ylim(
    session_dfs: list[pd.DataFrame],
    clip_percentile: float,
    iff_col: str,
) -> tuple[float, float]:
    """Return ``(0.0, upper_bound)`` where upper bound is the pooled IFF max.

    Parameters
    ----------
    session_dfs:
        All per-session DataFrames (must contain *iff_col*).
    clip_percentile:
        Upper percentile for clipping — the upper bound is set to
        ``np.nanpercentile(all_iff, 100 - clip_percentile)``.
    iff_col:
        Name of the IFF column (e.g. ``"Nerve_freq_mean"`` or
        ``"Nerve_freq_max"``).

    Returns
    -------
    tuple[float, float]
        ``(0.0, upper_percentile_value)``
    """
    all_iff = np.concatenate([
        df[iff_col].dropna().to_numpy(dtype=float)
        for df in session_dfs
        if iff_col in df.columns
    ])
    if len(all_iff) == 0:
        raise ValueError(
            f"_compute_global_iff_ylim: '{iff_col}' contains no finite values "
            f"across all sessions."
        )
    upper = float(np.nanpercentile(all_iff, 100.0 - clip_percentile))
    return (0.0, upper)


def _compute_global_count_max(
    session_dfs: list[pd.DataFrame],
    feature_col: str,
    bin_low: np.ndarray,
    bin_high: np.ndarray,
    iff_col: str,
    category_col: str | None = None,
    category_levels: list | None = None,
) -> float:
    """Return the maximum per-bin touch count across all sessions and gesture subsets.

    Used to set a consistent right Y-axis scale on per-session plots.

    Parameters
    ----------
    session_dfs:
        All per-session DataFrames.
    feature_col:
        Feature column to bin.
    bin_low:
        Pre-computed global bin low edges for this feature.
    bin_high:
        Pre-computed global bin high edges for this feature.
    iff_col:
        Name of the IFF column (e.g. ``"Nerve_freq_mean"`` or
        ``"Nerve_freq_max"``).
    category_col:
        Optional metadata column used for per-level composition.
    category_levels:
        Global ordered level list for *category_col*; passed through to
        ``_bin_data``.

    Returns
    -------
    float
        Max count (as float for matplotlib ylim compatibility).  Returns 1.0 if
        no data is available (prevents a zero-height axis).
    """
    global_max = 0
    for df in session_dfs:
        for gesture_subset in _GESTURE_SUBSETS:
            filtered = _filter_gesture(df, gesture_subset)
            if len(filtered) == 0:
                continue
            bin_result = _bin_data(
                filtered, feature_col, iff_col, bin_low, bin_high,
                category_col=category_col, category_levels=category_levels,
            )
            bin_max = int(bin_result.counts.max())
            if bin_max > global_max:
                global_max = bin_max
    return float(max(global_max, 1))


# ---------------------------------------------------------------------------
# Sentinel helper
# ---------------------------------------------------------------------------


def _write_sentinel(sentinel: Path, n_sessions: int, n_features: int) -> None:
    data = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_sessions": n_sessions,
        "n_features": n_features,
    }
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    with open(sentinel, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Metadata CSV loader
# ---------------------------------------------------------------------------


def _load_session_metadata_df(
    database_path: Path,
    metadata_dir: str,
    metadata_filename: str,
    session_id: str,
    needed_cols: list[str],
) -> pd.DataFrame:
    """Load a per-session metadata CSV and validate that required columns exist.

    Builds the path as::

        database_path / '4_analysed' / metadata_dir /
            metadata_filename.format(session_id=session_id)

    Raises
    ------
    ValueError
        If the file does not exist or any column in *needed_cols* is missing.
    """
    csv_path = (
        database_path
        / "4_analysed"
        / metadata_dir
        / metadata_filename.format(session_id=session_id)
    )
    if not csv_path.exists():
        raise ValueError(
            f"_load_session_metadata_df: metadata CSV not found for session "
            f"'{session_id}': {csv_path}"
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"_load_session_metadata_df: metadata CSV for session '{session_id}' "
            f"is missing required columns {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    return df


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def _write_bin_csv(
    bin_result: Any,
    session_id: str,
    feature: str,
    gesture_subset: str,
    overlap_ratio: float,
    out_path: Path,
) -> None:
    """Write per-bin data from *bin_result* to a CSV at *out_path*.

    Columns written:
        bin_center, bin_low, bin_high, count, iff_mean, iff_std,
        overlap_ratio, session_id, feature, gesture_subset,
        and for each level lv in bin_result.level_counts:
            count_{lv}, prop_{lv}.
    """
    rows: dict[str, Any] = {
        "bin_center": bin_result.bin_centers,
        "bin_low": bin_result.bin_low,
        "bin_high": bin_result.bin_high,
        "count": bin_result.counts,
        "iff_mean": bin_result.mean_iff,
        "iff_std": bin_result.std_iff,
        "overlap_ratio": overlap_ratio,
        "session_id": session_id,
        "feature": feature,
        "gesture_subset": gesture_subset,
    }
    for lv in bin_result.level_counts:
        rows[f"count_{lv}"] = bin_result.level_counts[lv]
        rows[f"prop_{lv}"] = bin_result.level_props[lv]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_iff_tuning_curves(
    session_config_paths: list[tuple[Path, Path]],
    options: dict,
    output_base_dir: Path,
) -> None:
    """Render IFF tuning curve PNGs for all sessions.

    Two-pass pipeline:

    1. **Load** — for each session, find the ``mean`` aggregation CSV under
       ``stimulus_extract_features/``, validate requested feature columns,
       and concatenate into a pooled DataFrame.
    2. **Render** — for each feature × gesture subset, call
       ``render_session_tuning_curve`` (per session) and
       ``render_overlay_tuning_curve`` (cross-session), using globally
       consistent axis limits.

    Idempotency is provided by ``iff_tuning_sentinel.json`` in *output_base_dir*.
    When the sentinel exists and ``force_processing`` is ``False`` the entire task
    is skipped.

    Parameters
    ----------
    session_config_paths:
        List of ``(csv_path, database_path)`` tuples.
    options:
        Task options from the DAG config.  Required keys:

        - ``tuning_features`` — list of feature column names (e.g.
          ``['contact_area_mean', 'pressure_mean']``).
        - ``n_bins`` — number of equal-width bins (default ``20``).
        - ``clip_percentile`` — symmetric percentile clip (default ``1.0``).
        - ``force_processing`` — if ``True``, re-render even when sentinel exists.
    output_base_dir:
        Root output directory, e.g.
        ``database_path / '4_analysed' / STIMULUS_IFF_TUNING_CURVES``.

    Raises
    ------
    ValueError
        If ``tuning_features`` is empty, if a requested feature is absent from
        all sessions, or if the IFF column is missing from any session's DataFrame.
    """
    tuning_features: list[str] = list(options.get("tuning_features") or [])
    if not tuning_features:
        raise ValueError(
            "run_iff_tuning_curves: 'tuning_features' is empty. "
            "Select at least one feature in the DAG config."
        )

    n_bins: int = int(options.get("n_bins", 20))
    clip_percentile: float = float(options.get("clip_percentile", 1.0))
    force_processing: bool = bool(options.get("force_processing", False))
    smoothing_sigma: float = float(options.get("smoothing_sigma", 0.0))
    overlap_ratio: float = float(options.get("overlap_ratio", 0.0))
    metadata_dir: str = str(options.get("metadata_dir", "touch_prepare_sessions"))
    metadata_filename: str = str(options.get("metadata_filename", "{session_id}_prepared.csv"))
    count_category_by: dict = dict(options.get("count_category_by") or {})

    iff_metric: str = str(options.get("iff_metric", "mean"))
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"run_iff_tuning_curves: invalid 'iff_metric' value '{iff_metric}'. "
            f"Expected one of {IFF_METRICS}."
        )
    iff_col: str = f"Nerve_freq_{iff_metric}"
    iff_ylabel: str = "Mean IFF (Hz)" if iff_metric == "mean" else "Max IFF (Hz)"

    sentinel = output_base_dir / "iff_tuning_sentinel.json"

    if sentinel.exists() and not force_processing:
        logger.info(
            "[IFF Tuning Curves] Up-to-date — skipping (sentinel: %s).", sentinel
        )
        return

    # =========================================================================
    # Pass 1 — Load all sessions
    # =========================================================================
    session_data: list[dict] = []

    for csv_path, database_path in session_config_paths:
        csv_path = Path(csv_path)
        database_path = Path(database_path)
        session_id = session_id_from_path(csv_path)

        mean_csv = _find_feature_csv(database_path, "mean", session_id)
        df = pd.read_csv(mean_csv)

        if iff_metric == "max":
            # Load the max aggregation CSV to obtain Nerve_freq_max, then merge
            # on TOUCH_ID_COLS. X-axis features always come from the mean CSV.
            max_csv = _find_feature_csv(database_path, "max", session_id)
            df_max = pd.read_csv(max_csv)

            touch_id_cols = list(TOUCH_ID_COLS)
            missing_in_mean = [c for c in touch_id_cols if c not in df.columns]
            if missing_in_mean:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: TOUCH_ID_COLS columns "
                    f"{missing_in_mean} not found in mean CSV {mean_csv}."
                )
            missing_in_max = [c for c in touch_id_cols if c not in df_max.columns]
            if missing_in_max:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: TOUCH_ID_COLS columns "
                    f"{missing_in_max} not found in max CSV {max_csv}."
                )
            if iff_col not in df_max.columns:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: column '{iff_col}' not found in "
                    f"{max_csv}. Available columns: {sorted(df_max.columns)}. "
                    f"Ensure 'stimulus_extract_features' with aggregation 'max' has run."
                )

            n_mean_rows = len(df)
            df = df.merge(
                df_max[touch_id_cols + [iff_col]],
                on=touch_id_cols,
                how="inner",
            )
            n_merged_rows = len(df)
            if n_merged_rows != n_mean_rows:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: inner merge on TOUCH_ID_COLS "
                    f"yielded {n_merged_rows} rows but the mean CSV had {n_mean_rows} rows. "
                    f"Touch IDs must be identical across aggregation CSVs — check that "
                    f"'stimulus_extract_features' (mean and max) was run on the same data."
                )
        else:
            # mean metric: IFF column comes from the mean CSV directly
            if iff_col not in df.columns:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: column '{iff_col}' not found in "
                    f"{mean_csv}. Available columns: {sorted(df.columns)}. "
                    f"Ensure 'stimulus_extract_features' with aggregation 'mean' has run."
                )

        if _GESTURE_COL not in df.columns:
            raise ValueError(
                f"[IFF Tuning Curves] {session_id}: column '{_GESTURE_COL}' not found in "
                f"the loaded DataFrame. Available columns: {sorted(df.columns)}."
            )

        if count_category_by:
            needed_cols = list(dict.fromkeys(count_category_by.values()))
            meta_df = _load_session_metadata_df(
                database_path, metadata_dir, metadata_filename, session_id, needed_cols
            )
            touch_id_cols = list(TOUCH_ID_COLS)
            meta_deduped = (
                meta_df[touch_id_cols + needed_cols]
                .groupby(touch_id_cols)[needed_cols]
                .first()
                .reset_index()
            )
            n_before = len(df)
            df = df.merge(meta_deduped, on=touch_id_cols, how="inner")
            n_after = len(df)
            if n_after != n_before:
                raise ValueError(
                    f"[IFF Tuning Curves] {session_id}: inner merge with metadata "
                    f"dropped rows ({n_before} → {n_after}). "
                    f"Touch IDs in the metadata CSV must cover every touch in the "
                    f"feature CSV."
                )

        session_data.append({
            "session_id": session_id,
            "df": df,
        })

    if not session_data:
        logger.info("[IFF Tuning Curves] No sessions to process.")
        return

    # =========================================================================
    # Validate that each requested feature exists in at least one session
    # =========================================================================
    all_cols: set[str] = set()
    for entry in session_data:
        all_cols.update(entry["df"].columns)

    absent_features = [f for f in tuning_features if f not in all_cols]
    if absent_features:
        raise ValueError(
            f"[IFF Tuning Curves] The following features are absent from all session "
            f"DataFrames: {absent_features}. "
            f"Ensure 'stimulus_extract_features' (mean) has run and the feature names "
            f"are correct. Available columns (sample): {sorted(all_cols)[:20]}"
        )

    # Restrict to features actually present
    valid_features = [f for f in tuning_features if f in all_cols]

    if count_category_by:
        unmapped = [f for f in valid_features if f not in count_category_by]
        if unmapped:
            raise ValueError(
                f"[IFF Tuning Curves] The following tuning features are not mapped in "
                f"'count_category_by': {unmapped}. "
                f"Add each feature → designed-metadata column entry to 'count_category_by' "
                f"in the DAG config, or leave 'count_category_by' empty to disable "
                f"composition colouring."
            )

    session_dfs = [entry["df"] for entry in session_data]
    session_ids = [entry["session_id"] for entry in session_data]

    # =========================================================================
    # Compute global bin windows per feature
    # =========================================================================
    pooled = pd.concat(session_dfs, ignore_index=True)

    global_levels: dict[str, list] = {}
    for cat_col in dict.fromkeys(count_category_by.values()):
        if cat_col not in pooled.columns:
            raise ValueError(
                f"[IFF Tuning Curves] Category column '{cat_col}' from "
                f"'count_category_by' is not present in the pooled DataFrame. "
                f"Available columns: {sorted(pooled.columns.tolist())[:20]}"
            )
        raw_values = pooled[cat_col].dropna().unique()
        try:
            numeric_vals = [float(v) for v in raw_values]
            ordered = [v for _, v in sorted(zip(numeric_vals, raw_values))]
        except (ValueError, TypeError):
            ordered = sorted(str(v) for v in raw_values)
        global_levels[cat_col] = ordered

    bin_windows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature in valid_features:
        bin_windows[feature] = _compute_bin_windows(
            pooled[feature], n_bins, clip_percentile, overlap_ratio=overlap_ratio
        )

    # =========================================================================
    # Compute global IFF ylim and per-feature count maxima
    # =========================================================================
    iff_ylim = _compute_global_iff_ylim(session_dfs, clip_percentile, iff_col)

    count_maxima: dict[str, float] = {}
    for feature in valid_features:
        bin_low, bin_high = bin_windows[feature]
        cat_col = count_category_by.get(feature) if count_category_by else None
        cat_levels = global_levels.get(cat_col, []) if cat_col else None
        count_maxima[feature] = _compute_global_count_max(
            session_dfs, feature, bin_low, bin_high, iff_col,
            category_col=cat_col, category_levels=cat_levels,
        )

    colors = assign_session_colors(session_ids)

    # =========================================================================
    # Pass 2 — Render
    # =========================================================================
    for feature in valid_features:
        bin_low, bin_high = bin_windows[feature]
        count_ymax = count_maxima[feature]
        cat_col = count_category_by.get(feature) if count_category_by else None
        cat_levels = global_levels.get(cat_col, []) if cat_col else None

        for gesture_subset in _GESTURE_SUBSETS:
            overlay_session_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            overlay_csv_dfs: list[pd.DataFrame] = []

            for entry in session_data:
                session_id = entry["session_id"]
                df = entry["df"]

                filtered = _filter_gesture(df, gesture_subset)
                if len(filtered) < _MIN_ROWS:
                    logger.warning(
                        "[IFF Tuning Curves] %s / %s / %s: only %d row(s) after "
                        "gesture filter — skipping (need ≥%d).",
                        feature, gesture_subset, session_id,
                        len(filtered), _MIN_ROWS,
                    )
                    continue

                bin_result = _bin_data(
                    filtered, feature, iff_col, bin_low, bin_high,
                    category_col=cat_col, category_levels=cat_levels,
                )
                bin_centers = bin_result.bin_centers
                mean_iff = bin_result.mean_iff
                counts = bin_result.counts

                out_path = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / f"{session_id}_tuning.png"
                )
                render_session_tuning_curve(
                    bin_centers=bin_centers,
                    mean_iff=mean_iff,
                    counts=counts,
                    feature_name=feature,
                    session_id=session_id,
                    gesture_subset=gesture_subset,
                    out_path=out_path,
                    iff_ylim=iff_ylim,
                    count_ymax=count_ymax,
                    iff_ylabel=iff_ylabel,
                    smoothing_sigma=smoothing_sigma,
                    std_iff=bin_result.std_iff,
                    level_counts=bin_result.level_counts,
                    level_props=bin_result.level_props,
                    category_levels=cat_levels or [],
                    category_display_name=_cat_display(cat_col) if cat_col else "",
                )
                csv_out_path = out_path.with_suffix(".csv")
                _write_bin_csv(
                    bin_result, session_id, feature, gesture_subset,
                    overlap_ratio=overlap_ratio, out_path=csv_out_path,
                )
                print(
                    f"[IFF Tuning Curves] {feature} / {gesture_subset} / {session_id}: "
                    f"saved {out_path.name}",
                    flush=True,
                )

                overlay_session_data[session_id] = (bin_centers, mean_iff, bin_result.std_iff)
                overlay_csv_dfs.append(pd.read_csv(csv_out_path))

            if len(overlay_session_data) < 2:
                logger.warning(
                    "[IFF Tuning Curves] %s / %s: fewer than 2 sessions have ≥%d rows "
                    "— skipping overlay.",
                    feature, gesture_subset, _MIN_ROWS,
                )
                continue

            overlay_path = (
                output_base_dir / feature / gesture_subset / "overlay_tuning.png"
            )
            render_overlay_tuning_curve(
                session_data=overlay_session_data,
                feature_name=feature,
                gesture_subset=gesture_subset,
                out_path=overlay_path,
                iff_ylim=iff_ylim,
                session_colors=colors,
                iff_ylabel=iff_ylabel,
                smoothing_sigma=smoothing_sigma,
            )
            overlay_csv_path = overlay_path.with_suffix(".csv")
            pd.concat(overlay_csv_dfs, ignore_index=True).to_csv(
                overlay_csv_path, index=False
            )
            print(
                f"[IFF Tuning Curves] {feature} / {gesture_subset}: "
                f"saved {overlay_path.name}",
                flush=True,
            )

    # =========================================================================
    # Write sentinel
    # =========================================================================
    _write_sentinel(sentinel, n_sessions=len(session_data), n_features=len(valid_features))
    print(
        f"[IFF Tuning Curves] Done — {len(valid_features)} feature(s), "
        f"{len(session_data)} session(s). Sentinel written.",
        flush=True,
    )
