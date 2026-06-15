"""Pipeline orchestrator for response tuning curve plots.

For each selected touch feature, bins touches into equal-width ranges across
all sessions and renders:

- per-session dual Y-axis plots (response line left, touch count bars right)
- all-sessions overlay plots (one response line per session, shared axis)

Output layout::

    4_analysed/stimulus_response_tuning/
        response_tuning_sentinel.json
        {feature}/
            {gesture_subset}/
                {overlap_dir}/
                    {metric_subdir}/
                        {session_id}_{feature}_{gesture_subset}_{metric_subdir}_tuning.png
                        overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_type.png
                        overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_session.png
                        overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning.csv
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import STIMULUS_RESPONSE_TUNING
from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,
    TOUCH_ID_COLS,
    session_id_from_path,
)
from analysis.receptive_field_mapping.pipelines.rf_touch_feature_radar_pipeline import (
    _find_feature_csv,
)
from analysis.receptive_field_mapping.rendering.rf_response_tuning_renderer import (
    _bin_data,
    _cat_display,
    render_session_tuning_curve,
    render_overlay_tuning_curve,
    render_session_raw_dots,
    render_overlay_raw_dots,
)
from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    build_session_color_scheme,
    SessionColorScheme,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response-metric registry
# ---------------------------------------------------------------------------
#
# Each entry maps a token → (response_col, agg_folder, bin_agg, ylabel, subdir).
#
#   response_col  — column name in the aggregation CSV
#   agg_folder    — subfolder under stimulus_extract_features/ to load when the
#                   response_col is not present in the mean CSV
#   bin_agg       — "mean" or "sum"; passed to _bin_data
#   ylabel        — Y-axis label for the tuning curve
#   subdir        — output sub-directory name under {feature}/{gesture_subset}/

_ResponseMetricSpec = tuple[str, str, str, str, str]

RESPONSE_METRICS: dict[str, _ResponseMetricSpec] = {
    "iff_mean": (
        "Nerve_freq_mean",
        "mean",
        "mean",
        "Mean IFF (Hz)",
        "iff_mean",
    ),
    "iff_max": (
        "Nerve_freq_max",
        "max",
        "mean",
        "Max IFF (Hz)",
        "iff_max",
    ),
    "spike_count_mean": (
        "Nerve_spike_count",
        "spike_count",
        "mean",
        "Mean spikes / touch",
        "spike_count_mean",
    ),
}

_ALL_RESPONSE_METRICS = ("iff_mean", "iff_max", "spike_count_mean")


def _resolve_response_metric(token: str) -> list[_ResponseMetricSpec]:
    """Return the list of metric specs for *token*.

    ``"all"`` expands to all entries in ``RESPONSE_METRICS``.

    Raises
    ------
    ValueError
        If *token* is not a recognised key or ``"all"``.
    """
    if token == "all":
        return [RESPONSE_METRICS[k] for k in _ALL_RESPONSE_METRICS]
    if token not in RESPONSE_METRICS:
        raise ValueError(
            f"_resolve_response_metric: unknown response_metric token '{token}'. "
            f"Valid values: {sorted(RESPONSE_METRICS)} or 'all'."
        )
    return [RESPONSE_METRICS[token]]


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


def _compute_global_response_ylim(
    session_dfs: list[pd.DataFrame],
    clip_percentile: float,
    response_col: str,
    bin_agg: str,
    bin_windows: dict[str, tuple[np.ndarray, np.ndarray]],
    valid_features: list[str],
) -> tuple[float, float]:
    """Return ``(0.0, upper_bound)`` for the response Y axis.

    For ``bin_agg="mean"``: upper bound is the pooled raw-value percentile.
    For ``bin_agg="sum"``: upper bound is the maximum per-bin sum across all
    sessions × gesture subsets × features (mirrors what the plot will show).

    Parameters
    ----------
    session_dfs:
        All per-session DataFrames.
    clip_percentile:
        Percentile for the mean-mode upper bound
        (``np.nanpercentile(all_vals, 100 - clip_percentile)``).
        Not used for ``"sum"`` mode.
    response_col:
        Response column name (e.g. ``"Nerve_freq_mean"``).
    bin_agg:
        ``"mean"`` or ``"sum"``.
    bin_windows:
        Pre-computed ``{feature: (bin_low, bin_high)}`` mapping.
    valid_features:
        Features to iterate over when computing per-bin sums.

    Returns
    -------
    tuple[float, float]
        ``(0.0, upper_bound)``
    """
    if bin_agg == "mean":
        all_vals = np.concatenate([
            df[response_col].dropna().to_numpy(dtype=float)
            for df in session_dfs
            if response_col in df.columns
        ])
        if len(all_vals) == 0:
            raise ValueError(
                f"_compute_global_response_ylim: '{response_col}' contains no "
                f"finite values across all sessions."
            )
        upper = float(np.nanpercentile(all_vals, 100.0 - clip_percentile))
        return (0.0, upper)

    # bin_agg == "sum": find the largest per-bin sum across sessions / subsets / features
    global_max = 0.0
    for df in session_dfs:
        if response_col not in df.columns:
            continue
        for feature in valid_features:
            if feature not in df.columns:
                continue
            bin_low, bin_high = bin_windows[feature]
            for gesture_subset in _GESTURE_SUBSETS:
                filtered = _filter_gesture(df, gesture_subset)
                if len(filtered) == 0:
                    continue
                valid_rows = filtered[[feature, response_col]].dropna()
                feature_vals = valid_rows[feature].to_numpy(dtype=float)
                response_vals = valid_rows[response_col].to_numpy(dtype=float)
                n_bins = len(bin_low)
                for i in range(n_bins):
                    if i < n_bins - 1:
                        mask = (feature_vals >= bin_low[i]) & (feature_vals < bin_high[i])
                    else:
                        mask = (feature_vals >= bin_low[i]) & (feature_vals <= bin_high[i])
                    if mask.any():
                        bin_sum = float(np.sum(response_vals[mask]))
                        if bin_sum > global_max:
                            global_max = bin_sum
    if global_max == 0.0:
        raise ValueError(
            f"_compute_global_response_ylim: '{response_col}' produced no "
            f"non-empty bins across all sessions (sum mode)."
        )
    return (0.0, global_max)


def _compute_global_count_max(
    session_dfs: list[pd.DataFrame],
    feature_col: str,
    bin_low: np.ndarray,
    bin_high: np.ndarray,
    response_col: str,
    bin_agg: str,
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
    response_col:
        Name of the response column.
    bin_agg:
        Aggregation mode passed through to ``_bin_data``.
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
                filtered, feature_col, response_col, bin_low, bin_high,
                category_col=category_col, category_levels=category_levels,
                bin_agg=bin_agg,
            )
            bin_max = int(bin_result.counts.max())
            if bin_max > global_max:
                global_max = bin_max
    return float(max(global_max, 1))


def _compute_raw_response_ylim(
    session_dfs: list[pd.DataFrame],
    response_col: str,
    clip_percentile: float,
) -> tuple[float, float]:
    """Return ``(0.0, upper_percentile)`` for the raw-dots response Y axis.

    Pools every raw response value across all sessions and clips the upper
    bound at the ``(100 - clip_percentile)``-th percentile.  This mirrors the
    ``bin_agg="mean"`` branch of :func:`_compute_global_response_ylim` but does
    not require pre-computed ``bin_windows`` (the raw-dots strategy has no bins).

    Parameters
    ----------
    session_dfs:
        All per-session DataFrames.
    response_col:
        Response column name (e.g. ``"Nerve_freq_mean"``).
    clip_percentile:
        Symmetric percentile clip; the upper bound is
        ``np.nanpercentile(all_vals, 100 - clip_percentile)``.

    Returns
    -------
    tuple[float, float]
        ``(0.0, upper_percentile)``

    Raises
    ------
    ValueError
        If *response_col* contains no finite values across all sessions.
    """
    all_vals = np.concatenate([
        df[response_col].dropna().to_numpy(dtype=float)
        for df in session_dfs
        if response_col in df.columns
    ])
    if len(all_vals) == 0:
        raise ValueError(
            f"_compute_raw_response_ylim: '{response_col}' contains no "
            f"finite values across all sessions."
        )
    upper = float(np.nanpercentile(all_vals, 100.0 - clip_percentile))
    return (0.0, upper)


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
    metric: str,
    out_path: Path,
) -> None:
    """Write per-bin data from *bin_result* to a CSV at *out_path*.

    Columns written:
        bin_center, bin_low, bin_high, count, response_value, response_std,
        metric, overlap_ratio, session_id, feature, gesture_subset,
        and for each level lv in bin_result.level_counts:
            count_{lv}, prop_{lv}.
    """
    rows: dict[str, Any] = {
        "bin_center": bin_result.bin_centers,
        "bin_low": bin_result.bin_low,
        "bin_high": bin_result.bin_high,
        "count": bin_result.counts,
        "response_value": bin_result.mean_iff,
        "response_std": bin_result.std_iff,
        "metric": metric,
        "overlap_ratio": overlap_ratio,
        "session_id": session_id,
        "feature": feature,
        "gesture_subset": gesture_subset,
    }
    for lv, arr in bin_result.level_counts.items():
        rows[f"count_{lv}"] = arr
    for lv, arr in bin_result.level_props.items():
        rows[f"prop_{lv}"] = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_raw_dots_csv(
    feature_vals: np.ndarray,
    response_vals: np.ndarray,
    session_id: str,
    feature: str,
    gesture_subset: str,
    fit_degree: int,
    metric: str,
    out_path: Path,
) -> None:
    """Write per-touch raw-dots data to a CSV at *out_path*.

    One row per touch.  Columns written (in order):
        session_id, feature_value, response_value, gesture_subset,
        fit_degree, metric.

    Parameters
    ----------
    feature_vals:
        Per-touch feature values (x axis).
    response_vals:
        Per-touch response values (y axis); must be parallel to *feature_vals*.
    session_id:
        Session identifier (constant across all rows).
    feature:
        Feature column name (used for messaging / validation only).
    gesture_subset:
        Gesture subset label (constant across all rows).
    fit_degree:
        Polynomial fit degree (constant across all rows).
    metric:
        Response-metric subdir token (constant across all rows).
    out_path:
        Destination CSV path.

    Raises
    ------
    ValueError
        If *feature_vals* and *response_vals* differ in length.
    """
    if len(feature_vals) != len(response_vals):
        raise ValueError(
            f"_write_raw_dots_csv: feature_vals ({len(feature_vals)}) and "
            f"response_vals ({len(response_vals)}) have mismatched lengths for "
            f"session '{session_id}', feature '{feature}'."
        )
    rows: dict[str, Any] = {
        "session_id": session_id,
        "feature_value": feature_vals,
        "response_value": response_vals,
        "gesture_subset": gesture_subset,
        "fit_degree": fit_degree,
        "metric": metric,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_response_tuning(
    session_config_paths: list[tuple[Path, Path]],
    options: dict,
    output_base_dir: Path,
) -> None:
    """Render tuning curve PNGs for all sessions and response metrics.

    Two-pass pipeline:

    1. **Load** — for each session, find the ``mean`` aggregation CSV under
       ``stimulus_extract_features/``, validate requested feature columns.
       When the response column is not in the mean CSV, load its dedicated
       aggregation folder and inner-merge on ``TOUCH_ID_COLS``.
    2. **Render** — for each metric × feature × gesture subset, call
       ``render_session_tuning_curve`` (per session) and
       ``render_overlay_tuning_curve`` (cross-session), using globally
       consistent axis limits.

    Idempotency is provided by ``response_tuning_sentinel.json`` in *output_base_dir*.
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
        - ``response_metric`` — one of the ``RESPONSE_METRICS`` keys, or
          ``"all"`` to render all four metrics.  Defaults to ``"iff_mean"``.
        - ``n_bins`` — number of equal-width bins (default ``20``).
        - ``clip_percentile`` — symmetric percentile clip (default ``1.0``).
        - ``force_processing`` — if ``True``, re-render even when sentinel exists.
    output_base_dir:
        Root output directory, e.g.
        ``database_path / '4_analysed' / STIMULUS_RESPONSE_TUNING``.

    Raises
    ------
    ValueError
        If ``tuning_features`` is empty, if ``response_metric`` is unknown, if a
        requested feature is absent from all sessions, or if the response column
        is missing from the expected aggregation CSV.
    """
    tuning_features: list[str] = list(options.get("tuning_features") or [])
    if not tuning_features:
        raise ValueError(
            "run_response_tuning: 'tuning_features' is empty. "
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

    response_metric_token: str = str(options.get("response_metric", "iff_mean"))
    metric_specs = _resolve_response_metric(response_metric_token)

    binning_strategy: str = str(options.get("binning_strategy", "sliding_window"))
    if binning_strategy not in {"sliding_window", "raw_dots"}:
        raise ValueError(
            f"run_response_tuning: unknown binning_strategy '{binning_strategy}'. "
            f"Valid values: 'sliding_window', 'raw_dots'."
        )
    _fit_degree_raw = options.get("fit_degree", 1)
    if isinstance(_fit_degree_raw, int):
        fit_degrees: list[int] = [_fit_degree_raw]
    elif isinstance(_fit_degree_raw, list):
        if not all(isinstance(d, int) for d in _fit_degree_raw):
            raise ValueError(
                f"run_response_tuning: 'fit_degree' list must contain only integers, "
                f"got {_fit_degree_raw!r}."
            )
        fit_degrees = list(_fit_degree_raw)
    else:
        raise ValueError(
            f"run_response_tuning: 'fit_degree' must be an int or list[int], "
            f"got {type(_fit_degree_raw).__name__!r}: {_fit_degree_raw!r}."
        )
    dot_alpha: float = float(options.get("dot_alpha", 0.35))
    show_fit_ci: bool = bool(options.get("show_fit_ci", False))
    secondary_color_by: dict = dict(options.get("secondary_color_by") or {})
    normalize_per_neuron: bool = bool(options.get("normalize_per_neuron", False))

    sentinel = output_base_dir / "response_tuning_sentinel.json"

    if sentinel.exists() and not force_processing:
        logger.info(
            "[Response Tuning] Up-to-date — skipping (sentinel: %s).", sentinel
        )
        return

    if binning_strategy == "raw_dots":
        degrees_tag = "_".join(str(d) for d in fit_degrees)
        overlap_dir = f"raw_dots_d{degrees_tag}"
    else:
        overlap_dir = f"b{n_bins}_ov{overlap_ratio:.2f}"

    total_features_rendered = 0

    for response_col, agg_folder, bin_agg, response_ylabel, metric_subdir in metric_specs:

        # =====================================================================
        # Pass 1 — Load all sessions for this metric
        # =====================================================================
        session_data: list[dict] = []

        for csv_path, database_path in session_config_paths:
            csv_path = Path(csv_path)
            database_path = Path(database_path)
            session_id = session_id_from_path(csv_path)

            mean_csv = _find_feature_csv(database_path, "mean", session_id)
            df = pd.read_csv(mean_csv)

            if response_col not in df.columns:
                # Response column lives in a dedicated aggregation folder —
                # load it and inner-merge on TOUCH_ID_COLS.
                agg_csv = _find_feature_csv(database_path, agg_folder, session_id)
                df_agg = pd.read_csv(agg_csv)

                touch_id_cols = list(TOUCH_ID_COLS)
                missing_in_mean = [c for c in touch_id_cols if c not in df.columns]
                if missing_in_mean:
                    raise ValueError(
                        f"[Response Tuning] {session_id}: TOUCH_ID_COLS columns "
                        f"{missing_in_mean} not found in mean CSV {mean_csv}."
                    )
                missing_in_agg = [c for c in touch_id_cols if c not in df_agg.columns]
                if missing_in_agg:
                    raise ValueError(
                        f"[Response Tuning] {session_id}: TOUCH_ID_COLS columns "
                        f"{missing_in_agg} not found in '{agg_folder}' CSV {agg_csv}."
                    )
                if response_col not in df_agg.columns:
                    raise ValueError(
                        f"[Response Tuning] {session_id}: column '{response_col}' "
                        f"not found in '{agg_folder}' CSV {agg_csv}. "
                        f"Available columns: {sorted(df_agg.columns)}. "
                        f"Ensure 'stimulus_extract_features' with aggregation "
                        f"'{agg_folder}' has run."
                    )

                n_mean_rows = len(df)
                df = df.merge(
                    df_agg[touch_id_cols + [response_col]],
                    on=touch_id_cols,
                    how="inner",
                )
                n_merged_rows = len(df)
                if n_merged_rows != n_mean_rows:
                    raise ValueError(
                        f"[Response Tuning] {session_id}: inner merge on TOUCH_ID_COLS "
                        f"yielded {n_merged_rows} rows but the mean CSV had {n_mean_rows} rows. "
                        f"Touch IDs must be identical across aggregation CSVs — check that "
                        f"'stimulus_extract_features' (mean and '{agg_folder}') was run on "
                        f"the same data."
                    )

            if _GESTURE_COL not in df.columns:
                raise ValueError(
                    f"[Response Tuning] {session_id}: column '{_GESTURE_COL}' not found in "
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
                        f"[Response Tuning] {session_id}: inner merge with metadata "
                        f"dropped rows ({n_before} → {n_after}). "
                        f"Touch IDs in the metadata CSV must cover every touch in the "
                        f"feature CSV."
                    )

            r_min = float(df[response_col].min()) if response_col in df.columns else 0.0
            r_max = float(df[response_col].max()) if response_col in df.columns else 1.0
            session_data.append({
                "session_id": session_id,
                "df": df,
                "response_range": (r_min, r_max),
            })

        if not session_data:
            logger.info("[Response Tuning] No sessions to process.")
            return

        # =====================================================================
        # Validate that each requested feature exists in at least one session
        # =====================================================================
        all_cols: set[str] = set()
        for entry in session_data:
            all_cols.update(entry["df"].columns)

        absent_features = [f for f in tuning_features if f not in all_cols]
        if absent_features:
            raise ValueError(
                f"[Response Tuning] The following features are absent from all session "
                f"DataFrames: {absent_features}. "
                f"Ensure 'stimulus_extract_features' (mean) has run and the feature names "
                f"are correct. Available columns (sample): {sorted(all_cols)[:20]}"
            )

        valid_features = [f for f in tuning_features if f in all_cols]

        if count_category_by:
            unmapped = [f for f in valid_features if f not in count_category_by]
            if unmapped:
                raise ValueError(
                    f"[Response Tuning] The following tuning features are not mapped in "
                    f"'count_category_by': {unmapped}. "
                    f"Add each feature → designed-metadata column entry to 'count_category_by' "
                    f"in the DAG config, or leave 'count_category_by' empty to disable "
                    f"composition colouring."
                )

        session_dfs = [entry["df"] for entry in session_data]
        session_ids = [entry["session_id"] for entry in session_data]

        # =====================================================================
        # Compute global bin windows per feature
        # =====================================================================
        pooled = pd.concat(session_dfs, ignore_index=True)

        global_levels: dict[str, list] = {}
        for cat_col in dict.fromkeys(count_category_by.values()):
            if cat_col not in pooled.columns:
                raise ValueError(
                    f"[Response Tuning] Category column '{cat_col}' from "
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

        if binning_strategy == "sliding_window":
            bin_windows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for feature in valid_features:
                bin_windows[feature] = _compute_bin_windows(
                    pooled[feature], n_bins, clip_percentile, overlap_ratio=overlap_ratio
                )

            # =================================================================
            # Compute global response ylim and per-feature count maxima
            # =================================================================
            response_ylim = _compute_global_response_ylim(
                session_dfs, clip_percentile, response_col, bin_agg,
                bin_windows, valid_features,
            )

            count_maxima: dict[str, float] = {}
            for feature in valid_features:
                bin_low, bin_high = bin_windows[feature]
                cat_col = count_category_by.get(feature) if count_category_by else None
                cat_levels = global_levels.get(cat_col, []) if cat_col else None
                count_maxima[feature] = _compute_global_count_max(
                    session_dfs, feature, bin_low, bin_high,
                    response_col, bin_agg,
                    category_col=cat_col, category_levels=cat_levels,
                )
        else:
            # raw_dots strategy: no bins, no count bars. Pool raw response
            # values for a consistent Y axis (same response_col / clip_percentile
            # the sliding_window path uses for its mean-mode ylim).
            bin_windows = {}
            count_maxima = {}
            response_ylim = _compute_raw_response_ylim(
                session_dfs, response_col, clip_percentile,
            )

        neuron_summary_xlsx_str: str | None = options.get("neuron_summary_xlsx") or None
        if not neuron_summary_xlsx_str:
            raise ValueError(
                "run_response_tuning: 'neuron_summary_xlsx' is not set in the task options. "
                "Set configs/analyse_workflow_processing_dag.yaml parameters.neuron_summary_xlsx "
                "to the path of MNG-DataSummary.xlsx (absolute, or relative to the database root)."
            )
        xlsx_path = Path(neuron_summary_xlsx_str)
        if not xlsx_path.is_absolute():
            xlsx_path = database_path / xlsx_path
        if not xlsx_path.is_file():
            raise FileNotFoundError(
                f"run_response_tuning: neuron_summary_xlsx not found: {xlsx_path}"
            )
        scheme: SessionColorScheme = build_session_color_scheme(session_ids, xlsx_path)

        # =====================================================================
        # Pass 2 — Render
        # =====================================================================
        for feature in valid_features:
            if binning_strategy == "sliding_window":
                bin_low, bin_high = bin_windows[feature]
                count_ymax = count_maxima[feature]
            cat_col = count_category_by.get(feature) if count_category_by else None
            cat_levels = global_levels.get(cat_col, []) if cat_col else None

            for gesture_subset in _GESTURE_SUBSETS:
                overlay_session_data: dict = {}
                overlay_csv_dfs: list[pd.DataFrame] = []

                for entry in session_data:
                    session_id = entry["session_id"]
                    df = entry["df"]

                    filtered = _filter_gesture(df, gesture_subset)
                    if len(filtered) < _MIN_ROWS:
                        logger.warning(
                            "[Response Tuning] %s / %s / %s / %s: only %d row(s) after "
                            "gesture filter — skipping (need ≥%d).",
                            metric_subdir, feature, gesture_subset, session_id,
                            len(filtered), _MIN_ROWS,
                        )
                        continue

                    if binning_strategy == "sliding_window":
                        bin_result = _bin_data(
                            filtered, feature, response_col, bin_low, bin_high,
                            category_col=cat_col, category_levels=cat_levels,
                            bin_agg=bin_agg,
                        )
                        bin_centers = bin_result.bin_centers
                        response_values = bin_result.mean_iff
                        counts = bin_result.counts

                        out_path = (
                            output_base_dir
                            / feature
                            / gesture_subset
                            / overlap_dir
                            / metric_subdir
                            / f"{session_id}_{feature}_{gesture_subset}_{metric_subdir}_tuning.png"
                        )
                        render_session_tuning_curve(
                            bin_centers=bin_centers,
                            mean_iff=response_values,
                            counts=counts,
                            feature_name=feature,
                            session_id=session_id,
                            gesture_subset=gesture_subset,
                            out_path=out_path,
                            iff_ylim=response_ylim,
                            count_ymax=count_ymax,
                            iff_ylabel=response_ylabel,
                            smoothing_sigma=smoothing_sigma,
                            std_iff=bin_result.std_iff,
                            level_counts=bin_result.level_counts,
                            level_props=bin_result.level_props,
                            category_levels=cat_levels or [],
                            category_display_name=_cat_display(cat_col) if cat_col else "",
                            line_color=scheme.session_color[session_id],
                        )
                        csv_out_path = out_path.with_suffix(".csv")
                        _write_bin_csv(
                            bin_result, session_id, feature, gesture_subset,
                            overlap_ratio=overlap_ratio,
                            metric=metric_subdir,
                            out_path=csv_out_path,
                        )
                        print(
                            f"[Response Tuning] {metric_subdir} / {feature} / "
                            f"{gesture_subset} / {session_id}: saved {out_path.name}",
                            flush=True,
                        )

                        overlay_session_data[session_id] = (
                            bin_centers, response_values, bin_result.std_iff
                        )
                        overlay_csv_dfs.append(pd.read_csv(csv_out_path))
                    else:
                        # raw_dots strategy: extract the same raw feature/response
                        # arrays the sliding_window path bins, drop NaN pairs, and
                        # plot every touch as a dot with a polynomial fit line.
                        secondary_col = secondary_color_by.get(feature)
                        if secondary_col is not None and secondary_col in filtered.columns:
                            cols_to_load = [feature, response_col, secondary_col]
                        else:
                            cols_to_load = [feature, response_col]
                            secondary_col = None

                        valid_rows = filtered[cols_to_load].dropna(subset=[feature, response_col])
                        feat_arr = valid_rows[feature].to_numpy(dtype=float)
                        resp_arr = valid_rows[response_col].to_numpy(dtype=float)
                        sec_arr = valid_rows[secondary_col].to_numpy(dtype=float) if secondary_col is not None else None

                        out_path = (
                            output_base_dir
                            / feature
                            / gesture_subset
                            / overlap_dir
                            / metric_subdir
                            / f"{session_id}_{feature}_{gesture_subset}_{metric_subdir}_tuning.png"
                        )
                        render_session_raw_dots(
                            feature_vals=feat_arr,
                            response_vals=resp_arr,
                            feature_name=feature,
                            session_id=session_id,
                            gesture_subset=gesture_subset,
                            out_path=out_path,
                            iff_ylim=response_ylim,
                            iff_ylabel=response_ylabel,
                            fit_degrees=fit_degrees,
                            dot_alpha=dot_alpha,
                            show_fit_ci=show_fit_ci,
                            line_color=scheme.session_color[session_id],
                            secondary_vals=sec_arr,
                            secondary_label=secondary_col or "",
                            secondary_cmap="viridis",
                        )
                        csv_out_path = out_path.with_suffix(".csv")
                        _write_raw_dots_csv(
                            feature_vals=feat_arr,
                            response_vals=resp_arr,
                            session_id=session_id,
                            feature=feature,
                            gesture_subset=gesture_subset,
                            fit_degree=fit_degrees[0],
                            metric=metric_subdir,
                            out_path=csv_out_path,
                        )
                        print(
                            f"[Response Tuning] {metric_subdir} / {feature} / "
                            f"{gesture_subset} / {session_id}: saved {out_path.name}",
                            flush=True,
                        )

                        overlay_session_data[session_id] = (feat_arr, resp_arr, sec_arr)
                        overlay_csv_dfs.append(pd.read_csv(csv_out_path))

                if len(overlay_session_data) < 2:
                    logger.warning(
                        "[Response Tuning] %s / %s / %s: fewer than 2 sessions "
                        "have ≥%d rows — skipping overlay.",
                        metric_subdir, feature, gesture_subset, _MIN_ROWS,
                    )
                    continue

                overlay_path_by_type = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / overlap_dir
                    / metric_subdir
                    / f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_type.png"
                )
                overlay_path_by_session = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / overlap_dir
                    / metric_subdir
                    / f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_session.png"
                )
                if binning_strategy == "sliding_window":
                    render_overlay_tuning_curve(
                        session_data=overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=overlay_path_by_type,
                        iff_ylim=response_ylim,
                        session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                        iff_ylabel=response_ylabel,
                        smoothing_sigma=smoothing_sigma,
                        legend_mode="by_type",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                    )
                    render_overlay_tuning_curve(
                        session_data=overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=overlay_path_by_session,
                        iff_ylim=response_ylim,
                        session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                        iff_ylabel=response_ylabel,
                        smoothing_sigma=smoothing_sigma,
                        legend_mode="by_session",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                    )
                else:
                    render_overlay_raw_dots(
                        session_data=overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=overlay_path_by_type,
                        iff_ylim=response_ylim,
                        session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                        iff_ylabel=response_ylabel,
                        fit_degrees=fit_degrees,
                        dot_alpha=0.15,
                        show_fit_ci=show_fit_ci,
                        legend_mode="by_type",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                        secondary_label=secondary_color_by.get(feature) or "",
                        secondary_cmap="viridis",
                    )
                    render_overlay_raw_dots(
                        session_data=overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=overlay_path_by_session,
                        iff_ylim=response_ylim,
                        session_colors={sid: scheme.session_color[sid] for sid in overlay_session_data},
                        iff_ylabel=response_ylabel,
                        fit_degrees=fit_degrees,
                        dot_alpha=0.15,
                        show_fit_ci=show_fit_ci,
                        legend_mode="by_session",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                        secondary_label=secondary_color_by.get(feature) or "",
                        secondary_cmap="viridis",
                    )
                overlay_csv_path = overlay_path_by_type.parent / f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning.csv"
                pd.concat(overlay_csv_dfs, ignore_index=True).to_csv(
                    overlay_csv_path, index=False
                )
                print(
                    f"[Response Tuning] {metric_subdir} / {feature} / "
                    f"{gesture_subset}: saved {overlay_path_by_type.name} + "
                    f"{overlay_path_by_session.name}",
                    flush=True,
                )

        # =====================================================================
        # Pass 3 — Normalized output (optional)
        # =====================================================================
        if normalize_per_neuron and binning_strategy == "raw_dots":
            for feature in valid_features:
                for gesture_subset in _GESTURE_SUBSETS:
                    norm_overlay_session_data: dict = {}

                    for entry in session_data:
                        session_id = entry["session_id"]
                        df = entry["df"]
                        r_min, r_max = entry["response_range"]

                        filtered = _filter_gesture(df, gesture_subset)
                        if len(filtered) < _MIN_ROWS:
                            continue

                        secondary_col = secondary_color_by.get(feature)
                        if secondary_col is not None and secondary_col in filtered.columns:
                            cols_to_load = [feature, response_col, secondary_col]
                        else:
                            cols_to_load = [feature, response_col]
                            secondary_col = None

                        valid_rows = filtered[cols_to_load].dropna(subset=[feature, response_col])
                        feat_arr = valid_rows[feature].to_numpy(dtype=float)
                        resp_arr = valid_rows[response_col].to_numpy(dtype=float)
                        sec_arr = valid_rows[secondary_col].to_numpy(dtype=float) if secondary_col is not None else None

                        if r_max > r_min:
                            norm_resp = (resp_arr - r_min) / (r_max - r_min)
                        else:
                            norm_resp = np.zeros_like(resp_arr)

                        norm_out_path = (
                            output_base_dir
                            / feature
                            / gesture_subset
                            / overlap_dir
                            / metric_subdir
                            / "normalized"
                            / f"{session_id}_{feature}_{gesture_subset}_{metric_subdir}_normalized.png"
                        )
                        render_session_raw_dots(
                            feature_vals=feat_arr,
                            response_vals=norm_resp,
                            feature_name=feature,
                            session_id=session_id,
                            gesture_subset=gesture_subset,
                            out_path=norm_out_path,
                            iff_ylim=(0.0, 1.0),
                            iff_ylabel="Normalized response",
                            fit_degrees=fit_degrees,
                            dot_alpha=dot_alpha,
                            show_fit_ci=show_fit_ci,
                            line_color=scheme.session_color[session_id],
                            secondary_vals=sec_arr,
                            secondary_label=secondary_col or "",
                            secondary_cmap="viridis",
                        )
                        print(
                            f"[Response Tuning] {metric_subdir} / {feature} / "
                            f"{gesture_subset} / {session_id}: saved normalized {norm_out_path.name}",
                            flush=True,
                        )

                        norm_overlay_session_data[session_id] = (feat_arr, norm_resp, sec_arr)

                    if len(norm_overlay_session_data) < 2:
                        continue

                    norm_overlay_path_by_type = (
                        output_base_dir
                        / feature
                        / gesture_subset
                        / overlap_dir
                        / metric_subdir
                        / "normalized"
                        / f"overlay_{feature}_{gesture_subset}_{metric_subdir}_normalized_by_type.png"
                    )
                    norm_overlay_path_by_session = (
                        output_base_dir
                        / feature
                        / gesture_subset
                        / overlap_dir
                        / metric_subdir
                        / "normalized"
                        / f"overlay_{feature}_{gesture_subset}_{metric_subdir}_normalized_by_session.png"
                    )
                    render_overlay_raw_dots(
                        session_data=norm_overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=norm_overlay_path_by_type,
                        iff_ylim=(0.0, 1.0),
                        session_colors={sid: scheme.session_color[sid] for sid in norm_overlay_session_data},
                        iff_ylabel="Normalized response",
                        fit_degrees=fit_degrees,
                        dot_alpha=0.15,
                        show_fit_ci=show_fit_ci,
                        legend_mode="by_type",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                        secondary_label=secondary_color_by.get(feature) or "",
                        secondary_cmap="viridis",
                    )
                    render_overlay_raw_dots(
                        session_data=norm_overlay_session_data,
                        feature_name=feature,
                        gesture_subset=gesture_subset,
                        out_path=norm_overlay_path_by_session,
                        iff_ylim=(0.0, 1.0),
                        session_colors={sid: scheme.session_color[sid] for sid in norm_overlay_session_data},
                        iff_ylabel="Normalized response",
                        fit_degrees=fit_degrees,
                        dot_alpha=0.15,
                        show_fit_ci=show_fit_ci,
                        legend_mode="by_session",
                        session_neuron_types=scheme.session_neuron_type,
                        type_colors=scheme.type_color,
                        secondary_label=secondary_color_by.get(feature) or "",
                        secondary_cmap="viridis",
                    )
                    print(
                        f"[Response Tuning] {metric_subdir} / {feature} / "
                        f"{gesture_subset}: saved normalized overlays.",
                        flush=True,
                    )

        total_features_rendered += len(valid_features)

    # =========================================================================
    # Write sentinel
    # =========================================================================
    _write_sentinel(
        sentinel,
        n_sessions=len(session_config_paths),
        n_features=total_features_rendered,
    )
    print(
        f"[Response Tuning] Done — {total_features_rendered} feature×metric "
        f"combination(s), {len(session_config_paths)} session(s). Sentinel written.",
        flush=True,
    )
