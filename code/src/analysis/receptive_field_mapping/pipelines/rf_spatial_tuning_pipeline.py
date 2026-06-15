"""Pipeline orchestrator for RF spatial tuning curves.

For each session, bins touches by a selected stimulus parameter (velocity,
depth, contact_area), computes a population RF heatmap per bin, extracts the
inflection boundary, and collects RF spatial metrics (area_mm2, circularity,
pca_aspect_ratio, pca_orientation_deg) as a function of the bin center.

Output layout::

    4_analysed/spatial_tuning_rf_metrics/
        spatial_tuning_sentinel.json
        {feature}/
            {gesture_subset}/
                {session_id}_{feature}_{gesture_subset}_spatial_tuning.png
                overlay_{feature}_{gesture_subset}_spatial_tuning_by_type.png
                overlay_{feature}_{gesture_subset}_spatial_tuning_by_session.png
                {session_id}_{feature}_{gesture_subset}_spatial_tuning.csv
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from analysis.pipeline.output_dirs import (
    SPATIAL_EXTRACT_BOUNDARIES,
    SPATIAL_MAP_SINGLE_TOUCH,
    SPATIAL_SLIM_UV,
    STIMULUS_EXTRACT_FEATURES,
    TOUCH_COMPUTE_SERIES,
)
from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,
    IFF_METRICS,
    session_id_from_path,
    single_touch_npz_filename,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_colors,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
)
from analysis.receptive_field_mapping.metrics.rf_pca_alignment import (
    apply_uv_alignment,
    compute_rf_pca_alignment,
)
from analysis.receptive_field_mapping.pipelines.rf_touch_feature_radar_pipeline import (
    _find_feature_csv,
)
from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    SessionColorScheme,
    build_session_color_scheme,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_interpolated_grid,
    compute_uv_to_mm_scale,
)
from analysis.receptive_field_mapping.rendering.rf_spatial_tuning_renderer import (
    render_session_spatial_tuning,
    render_overlay_spatial_tuning,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import load_slim_uv_cache

logger = logging.getLogger(__name__)

_GESTURE_SUBSETS = ["all", "tap", "stroke_proximal", "stroke_distal"]
_GESTURE_COL = "gesture_type"

# ---------------------------------------------------------------------------
# Gesture filtering
# ---------------------------------------------------------------------------


def _filter_gesture(df: pd.DataFrame, gesture_subset: str) -> pd.DataFrame:
    """Return rows of *df* matching *gesture_subset*.

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
# Bin edge computation (reuses logic from rf_response_tuning_pipeline)
# ---------------------------------------------------------------------------


def _compute_bin_edges(
    values: np.ndarray,
    n_bins: int,
    clip_percentile: float,
) -> np.ndarray:
    """Return n_bins+1 equal-width bin edges for *values* after percentile clipping.

    Raises
    ------
    ValueError
        If *values* is empty or collapses to a single unique value after clipping.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(
            "_compute_bin_edges: no finite values — cannot compute bin edges."
        )

    if clip_percentile > 0.0:
        lo = float(np.percentile(finite, clip_percentile))
        hi = float(np.percentile(finite, 100.0 - clip_percentile))
    else:
        lo = float(finite.min())
        hi = float(finite.max())

    if lo >= hi:
        raise ValueError(
            f"_compute_bin_edges: feature range collapsed to [{lo}, {hi}] after "
            f"clipping at {clip_percentile}th / {100.0 - clip_percentile}th "
            f"percentile — all values are effectively identical."
        )
    return np.linspace(lo, hi, n_bins + 1)


# ---------------------------------------------------------------------------
# Per-bin RF metric extraction
# ---------------------------------------------------------------------------


def _extract_bins_for_session(
    session_id: str,
    feature_df: pd.DataFrame,
    gesture_subset: str,
    tuning_feature: str,
    bin_edges: np.ndarray,
    min_touches_per_bin: int,
    rf_vertex_indices: list,
    rf_values: list,
    n_verts: int,
    cp_vertex_idx: np.ndarray,
    cp_touch_idx: np.ndarray,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    nearest_orig_for_slim: np.ndarray,
    vertex_threshold_ratio: float,
    inflection_sigma: float,
    touch_triple_keys: np.ndarray,
) -> list[dict]:
    """Return per-bin RF metric dicts for one session × gesture_subset × feature.

    Each returned dict has keys:
        ``bin_center``, ``area_mm2``, ``circularity``,
        ``pca_aspect_ratio``, ``pca_orientation_deg``.

    Bins are skipped (not returned) when:
    - fewer than ``min_touches_per_bin`` touches fall in the bin, OR
    - ``compute_inflection_boundary`` returns None.

    Parameters
    ----------
    session_id:
        Session identifier (used only for logging).
    feature_df:
        Per-touch feature DataFrame for this session.  Must contain
        ``gesture_type`` and ``tuning_feature`` columns, plus
        ``block_order_id``, ``trial_id``, ``single_touch_id`` for indexing.
    gesture_subset:
        Gesture filter to apply before binning.
    tuning_feature:
        Column name of the stimulus feature to bin by.
    bin_edges:
        (n_bins+1,) array of global bin edges.
    min_touches_per_bin:
        Minimum number of touches required in a bin.
    rf_vertex_indices:
        Per-touch list of contacted vertex index arrays (from NPZ).
    rf_values:
        Per-touch list of RF value arrays parallel to ``rf_vertex_indices``.
    n_verts:
        Total number of forearm vertices in the original PLY.
    cp_vertex_idx:
        Per-contact-point vertex index array from ``PopulationData``.
    cp_touch_idx:
        Per-contact-point touch index array from ``PopulationData``.
    forearm_uv:
        (V_slim, 2) SLIM UV coordinates.
    forearm_faces:
        (F_slim, 3) SLIM face array.
    forearm_V:
        (V_slim, 3) SLIM vertex positions in mm.
    nearest_orig_for_slim:
        (V_slim,) index array mapping SLIM vertices to original PLY vertices.
    vertex_threshold_ratio:
        Fraction of touches that must contact a vertex for it to be kept.
    inflection_sigma:
        Gaussian sigma for ``compute_inflection_boundary``.
    touch_triple_keys:
        (T, 3) int64 array of (block_order_id, trial_id, single_touch_id)
        used to cross-reference the feature CSV with the population data.

    Returns
    -------
    list[dict]
        One dict per bin that yielded a valid inflection boundary.
    """
    mm_scale = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)

    filtered_df = _filter_gesture(feature_df, gesture_subset)
    if len(filtered_df) == 0:
        logger.info(
            "[Spatial Tuning] %s / %s / %s: no touches after gesture filter.",
            session_id, gesture_subset, tuning_feature,
        )
        return []

    if tuning_feature not in filtered_df.columns:
        raise ValueError(
            f"[Spatial Tuning] {session_id}: column '{tuning_feature}' not found in "
            f"feature DataFrame. Available columns: {sorted(filtered_df.columns)}"
        )

    # Build a lookup: (block_order_id, trial_id, single_touch_id) → touch_index
    triple_to_touch_idx: dict[tuple, int] = {
        tuple(row): i for i, row in enumerate(touch_triple_keys)
    }

    id_cols = ["block_order_id", "trial_id", "single_touch_id"]
    for col in id_cols:
        if col not in filtered_df.columns:
            raise ValueError(
                f"[Spatial Tuning] {session_id}: required column '{col}' not found in "
                f"feature CSV. Available: {sorted(filtered_df.columns)}"
            )

    feature_vals = filtered_df[tuning_feature].to_numpy(dtype=float)
    valid_feature_mask = np.isfinite(feature_vals)
    filtered_df = filtered_df.iloc[valid_feature_mask]
    feature_vals = feature_vals[valid_feature_mask]

    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    result_bins: list[dict] = []

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (feature_vals >= lo) & (feature_vals < hi)
        else:
            mask = (feature_vals >= lo) & (feature_vals <= hi)

        bin_df = filtered_df.iloc[mask]
        n_bin_touches = int(mask.sum())

        if n_bin_touches < min_touches_per_bin:
            logger.debug(
                "[Spatial Tuning] %s / %s / %s bin %d: %d touches < min %d — skip.",
                session_id, gesture_subset, tuning_feature,
                i, n_bin_touches, min_touches_per_bin,
            )
            continue

        # Map feature CSV rows to population-data touch indices.
        touch_indices: list[int] = []
        for _, row in bin_df.iterrows():
            key = (int(row["block_order_id"]), int(row["trial_id"]), int(row["single_touch_id"]))
            idx = triple_to_touch_idx.get(key)
            if idx is not None:
                touch_indices.append(idx)

        touch_indices = list(dict.fromkeys(touch_indices))
        if len(touch_indices) < min_touches_per_bin:
            logger.debug(
                "[Spatial Tuning] %s / %s / %s bin %d: only %d matched population "
                "touch indices (< min %d) — skip.",
                session_id, gesture_subset, tuning_feature,
                i, len(touch_indices), min_touches_per_bin,
            )
            continue

        touch_idx_arr = np.array(touch_indices, dtype=np.int64)

        heatmap = compute_rf_heatmap(
            touch_idx_arr,
            rf_vertex_indices,
            rf_values,
            n_verts,
        )

        cp_mask = np.isin(cp_touch_idx, touch_idx_arr)
        unique_count = compute_unique_touch_count(
            cp_vertex_idx, cp_touch_idx, cp_mask, n_verts,
        )

        threshold = compute_threshold_from_ratio(
            vertex_threshold_ratio * 100.0, len(touch_indices)
        )
        thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)
        slim_heatmap = thresholded[nearest_orig_for_slim]

        try:
            grid_u, grid_v, grid_z = compute_interpolated_grid(
                forearm_uv, forearm_faces, forearm_V, slim_heatmap,
            )
        except ValueError as exc:
            logger.info(
                "[Spatial Tuning] %s / %s / %s bin %d: interpolation failed (%s) — skip.",
                session_id, gesture_subset, tuning_feature, i, exc,
            )
            continue

        boundary = compute_inflection_boundary(
            grid_u, grid_v, grid_z, inflection_sigma,
        )
        if boundary is None:
            logger.info(
                "[Spatial Tuning] %s / %s / %s bin %d: no inflection boundary — skip.",
                session_id, gesture_subset, tuning_feature, i,
            )
            continue

        area_mm2 = boundary.area_uv * (mm_scale ** 2)
        pca_major_mm = boundary.pca_major_uv * mm_scale
        pca_minor_mm = boundary.pca_minor_uv * mm_scale
        pca_aspect_ratio = (
            pca_major_mm / pca_minor_mm
            if pca_minor_mm > 1e-12
            else float("nan")
        )

        result_bins.append({
            "bin_center":        float(bin_centers[i]),
            "area_mm2":          float(area_mm2),
            "circularity":       float(boundary.circularity),
            "pca_aspect_ratio":  float(pca_aspect_ratio),
            "pca_orientation_deg": float(boundary.pca_orientation_deg),
            "n_touches":         n_bin_touches,
        })

    logger.info(
        "[Spatial Tuning] %s / %s / %s: %d / %d bins yielded a boundary.",
        session_id, gesture_subset, tuning_feature,
        len(result_bins), n_bins,
    )
    return result_bins


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
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_spatial_tuning(
    session_config_paths: list[tuple[Path, Path]],
    options: dict,
    output_base_dir: Path,
) -> None:
    """Render RF spatial tuning scatter plots for all sessions.

    Two-pass pipeline:

    1. **Load** — for each session, load population data, single-touch RF NPZ,
       SLIM UV cache, and per-touch feature CSV.  Compute globally consistent
       bin edges per tuning feature across all sessions.
    2. **Render** — for each feature × gesture subset × session, extract
       per-bin RF metrics, render a 4-subplot figure, and collect data for the
       cross-session overlay.

    Idempotency is provided by ``spatial_tuning_sentinel.json`` in
    *output_base_dir*.  When the sentinel exists and ``force_processing`` is
    False the entire task is skipped.

    Parameters
    ----------
    session_config_paths:
        List of ``(csv_path, database_path)`` tuples.
    options:
        Task options from the DAG config.  Required keys:

        - ``tuning_features`` — list of feature column names.
        - ``gesture_subsets`` — list of gesture subset strings.
        - ``n_bins`` — number of equal-width bins (default 8).
        - ``min_touches_per_bin`` — minimum touches per bin (default 5).
        - ``fit_degrees`` — int or list[int] polynomial degrees (default [1, 2]).
        - ``vertex_threshold`` — vertex overlap ratio in [0, 1] (default 0.05).
        - ``inflection_sigma`` — Gaussian sigma for boundary detection (default 4.0).
        - ``clip_percentile`` — symmetric percentile clip for bin edges (default 1.0).
        - ``force_processing`` — if True, re-render even when sentinel exists.
        - ``iff_metric`` — ``"mean"`` or ``"max"``; which NPZ to consume.

    output_base_dir:
        Root output directory
        (e.g. ``database_path / '4_analysed' / SPATIAL_TUNING_RF_METRICS``).

    Raises
    ------
    ValueError
        If required options are missing or invalid.
    FileNotFoundError
        If expected input files (NPZ, SLIM UV cache, feature CSV) are missing.
    """
    tuning_features: list[str] = list(options.get("tuning_features") or [])
    if not tuning_features:
        raise ValueError(
            "run_spatial_tuning: 'tuning_features' is empty. "
            "Select at least one feature in the DAG config."
        )

    gesture_subsets: list[str] = list(options.get("gesture_subsets") or _GESTURE_SUBSETS)
    for gs in gesture_subsets:
        if gs not in _GESTURE_SUBSETS:
            raise ValueError(
                f"run_spatial_tuning: unknown gesture_subset '{gs}'. "
                f"Valid values: {_GESTURE_SUBSETS}"
            )

    n_bins: int = int(options.get("n_bins", 8))
    min_touches_per_bin: int = int(options.get("min_touches_per_bin", 5))
    vertex_threshold: float = float(options.get("vertex_threshold", 0.05))
    inflection_sigma: float = float(options.get("inflection_sigma", 4.0))
    clip_percentile: float = float(options.get("clip_percentile", 1.0))
    force_processing: bool = bool(options.get("force_processing", False))
    dot_alpha: float = float(options.get("dot_alpha", 0.7))
    iff_metric: str = str(options.get("iff_metric", "mean"))

    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"run_spatial_tuning: unknown iff_metric '{iff_metric}'. "
            f"Valid values: {IFF_METRICS}."
        )

    _fit_degree_raw = options.get("fit_degrees", [1, 2])
    if isinstance(_fit_degree_raw, int):
        fit_degrees: list[int] = [_fit_degree_raw]
    elif isinstance(_fit_degree_raw, list):
        if not all(isinstance(d, int) for d in _fit_degree_raw):
            raise ValueError(
                f"run_spatial_tuning: 'fit_degrees' list must contain only integers, "
                f"got {_fit_degree_raw!r}."
            )
        fit_degrees = list(_fit_degree_raw)
    else:
        raise ValueError(
            f"run_spatial_tuning: 'fit_degrees' must be an int or list[int], "
            f"got {type(_fit_degree_raw).__name__!r}: {_fit_degree_raw!r}."
        )

    neuron_summary_xlsx_str: str | None = options.get("neuron_summary_xlsx") or None
    if not neuron_summary_xlsx_str:
        raise ValueError(
            "run_spatial_tuning: 'neuron_summary_xlsx' is not set in the task options. "
            "Set configs/analyse_workflow_processing_dag.yaml parameters.neuron_summary_xlsx "
            "to the path of MNG-DataSummary.xlsx (absolute, or relative to the database root)."
        )
    xlsx_path = Path(neuron_summary_xlsx_str)

    sentinel = output_base_dir / "spatial_tuning_sentinel.json"
    if sentinel.exists() and not force_processing:
        logger.info(
            "[Spatial Tuning] Up-to-date — skipping (sentinel: %s).", sentinel
        )
        return

    npz_filename = single_touch_npz_filename(iff_metric)

    # =========================================================================
    # Pass 1 — Load session data
    # =========================================================================
    session_data: list[dict] = []

    for csv_path, database_path in session_config_paths:
        csv_path = Path(csv_path)
        database_path = Path(database_path)
        session_id = session_id_from_path(csv_path)

        print(f"[Spatial Tuning] Loading session {session_id}...", flush=True)

        series_csv_path = (
            database_path / "4_analysed" / TOUCH_COMPUTE_SERIES
            / f"{session_id}_series_augmented.csv"
        )
        if not series_csv_path.exists():
            raise FileNotFoundError(
                f"[Spatial Tuning] {session_id}: series CSV not found: {series_csv_path}. "
                f"Run 'touch_compute_series' first."
            )

        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise FileNotFoundError(
                f"[Spatial Tuning] {session_id}: forearm PLY not found in {csv_path.parent}."
            )

        npz_path = (
            database_path / "4_analysed" / SPATIAL_MAP_SINGLE_TOUCH
            / session_id / npz_filename
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Spatial Tuning] {session_id}: {npz_filename} not found: {npz_path}. "
                f"Enable 'spatial_map_single_touch' in the DAG config."
            )

        slim_cache_path = (
            database_path / "4_analysed" / SPATIAL_SLIM_UV
            / session_id / f"{session_id}_slim_uv.npz"
        )
        if not slim_cache_path.exists():
            raise FileNotFoundError(
                f"[Spatial Tuning] {session_id}: SLIM UV cache not found: {slim_cache_path}. "
                f"Enable 'spatial_precompute_slim_uv' in the DAG config."
            )

        mean_csv = _find_feature_csv(database_path, "mean", session_id)
        feature_df = pd.read_csv(mean_csv)

        if _GESTURE_COL not in feature_df.columns:
            raise ValueError(
                f"[Spatial Tuning] {session_id}: column '{_GESTURE_COL}' not found "
                f"in feature CSV {mean_csv}. Available: {sorted(feature_df.columns)}"
            )

        pop_data = load_population_data(series_csv_path, forearm_ply_path)
        n_verts = len(pop_data.forearm_vertices)
        rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_verts)

        cache = load_slim_uv_cache(slim_cache_path)
        forearm_uv_raw = cache.uv
        slim_V = cache.V
        slim_faces = cache.F

        orig_tree = KDTree(pop_data.forearm_vertices)
        distances, nearest_orig_for_slim = orig_tree.query(slim_V)
        max_dist_mm = float(distances.max())
        _ERROR_THRESHOLD_MM = 5.0
        if max_dist_mm > _ERROR_THRESHOLD_MM:
            raise ValueError(
                f"[Spatial Tuning] {session_id}: KDTree max distance "
                f"{max_dist_mm:.3f} mm > {_ERROR_THRESHOLD_MM} mm — "
                f"SLIM mesh and forearm PLY are misaligned."
            )

        all_touch_indices = np.arange(len(pop_data.touch_triple_keys))
        heatmap_all = compute_rf_heatmap(
            all_touch_indices,
            rf_data.rf_vertex_indices,
            rf_data.rf_values,
            n_verts,
        )
        slim_heatmap_all = heatmap_all[nearest_orig_for_slim]
        alignment_center, alignment_rotation_matrix, _ = compute_rf_pca_alignment(
            forearm_uv_raw, slim_heatmap_all
        )
        forearm_uv = apply_uv_alignment(
            forearm_uv_raw, alignment_center, alignment_rotation_matrix
        )

        session_data.append({
            "session_id":            session_id,
            "database_path":         database_path,
            "feature_df":            feature_df,
            "pop_data":              pop_data,
            "rf_data":               rf_data,
            "n_verts":               n_verts,
            "forearm_uv":            forearm_uv,
            "slim_faces":            slim_faces,
            "slim_V":                slim_V,
            "nearest_orig_for_slim": nearest_orig_for_slim,
            "touch_triple_keys":     pop_data.touch_triple_keys,
        })

    if not session_data:
        logger.info("[Spatial Tuning] No sessions to process.")
        return

    session_ids = [s["session_id"] for s in session_data]

    if not xlsx_path.is_absolute():
        xlsx_path = Path(session_config_paths[0][1]) / xlsx_path
    if not xlsx_path.is_file():
        raise FileNotFoundError(
            f"run_spatial_tuning: neuron_summary_xlsx not found: {xlsx_path}"
        )
    scheme: SessionColorScheme = build_session_color_scheme(session_ids, xlsx_path)

    # Validate that each requested feature exists in at least one session.
    all_cols: set[str] = set()
    for entry in session_data:
        all_cols.update(entry["feature_df"].columns)

    absent_features = [f for f in tuning_features if f not in all_cols]
    if absent_features:
        raise ValueError(
            f"[Spatial Tuning] The following features are absent from all session "
            f"DataFrames: {absent_features}. "
            f"Ensure 'stimulus_extract_features' (mean) has run. "
            f"Available columns (sample): {sorted(all_cols)[:20]}"
        )

    valid_features = [f for f in tuning_features if f in all_cols]

    # =========================================================================
    # Compute global bin edges per feature
    # =========================================================================
    pooled_feature_vals: dict[str, np.ndarray] = {}
    for feature in valid_features:
        all_vals = np.concatenate([
            entry["feature_df"][feature].dropna().to_numpy(dtype=float)
            for entry in session_data
            if feature in entry["feature_df"].columns
        ])
        pooled_feature_vals[feature] = all_vals

    global_bin_edges: dict[str, np.ndarray] = {}
    for feature in valid_features:
        global_bin_edges[feature] = _compute_bin_edges(
            pooled_feature_vals[feature], n_bins, clip_percentile
        )

    total_features_rendered = 0

    # =========================================================================
    # Pass 2 — Render
    # =========================================================================
    for feature in valid_features:
        bin_edges = global_bin_edges[feature]

        for gesture_subset in gesture_subsets:
            all_sessions_overlay_data: dict[str, list[dict]] = {}
            all_sessions_csv_dfs: list[pd.DataFrame] = []

            for entry in session_data:
                session_id = entry["session_id"]
                pop_data = entry["pop_data"]
                rf_data = entry["rf_data"]

                bins_data = _extract_bins_for_session(
                    session_id=session_id,
                    feature_df=entry["feature_df"],
                    gesture_subset=gesture_subset,
                    tuning_feature=feature,
                    bin_edges=bin_edges,
                    min_touches_per_bin=min_touches_per_bin,
                    rf_vertex_indices=rf_data.rf_vertex_indices,
                    rf_values=rf_data.rf_values,
                    n_verts=entry["n_verts"],
                    cp_vertex_idx=pop_data.cp_vertex_idx,
                    cp_touch_idx=pop_data.cp_touch_idx,
                    forearm_uv=entry["forearm_uv"],
                    forearm_faces=entry["slim_faces"],
                    forearm_V=entry["slim_V"],
                    nearest_orig_for_slim=entry["nearest_orig_for_slim"],
                    vertex_threshold_ratio=vertex_threshold,
                    inflection_sigma=inflection_sigma,
                    touch_triple_keys=entry["touch_triple_keys"],
                )

                all_sessions_overlay_data[session_id] = bins_data

                if not bins_data:
                    logger.info(
                        "[Spatial Tuning] %s / %s / %s: no valid bins — skipping session figure.",
                        session_id, gesture_subset, feature,
                    )
                    continue

                out_path = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / f"{session_id}_{feature}_{gesture_subset}_spatial_tuning.png"
                )
                render_session_spatial_tuning(
                    session_id=session_id,
                    bins_data=bins_data,
                    tuning_feature=feature,
                    gesture_subset=gesture_subset,
                    out_path=out_path,
                    fit_degrees=fit_degrees,
                    dot_alpha=dot_alpha,
                    line_color=scheme.session_color[session_id],
                )
                print(
                    f"[Spatial Tuning] {feature} / {gesture_subset} / {session_id}: "
                    f"saved {out_path.name}",
                    flush=True,
                )

                csv_rows = [{**b, "session_id": session_id, "feature": feature,
                             "gesture_subset": gesture_subset} for b in bins_data]
                csv_df = pd.DataFrame(csv_rows)
                csv_out = out_path.with_suffix(".csv")
                csv_out.parent.mkdir(parents=True, exist_ok=True)
                csv_df.to_csv(csv_out, index=False)
                all_sessions_csv_dfs.append(csv_df)

            non_empty_sessions = {
                sid: data
                for sid, data in all_sessions_overlay_data.items()
                if data
            }

            if len(non_empty_sessions) < 2:
                logger.info(
                    "[Spatial Tuning] %s / %s: fewer than 2 sessions have valid bins — "
                    "skipping overlay.",
                    feature, gesture_subset,
                )
            else:
                overlay_path_by_type = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / f"overlay_{feature}_{gesture_subset}_spatial_tuning_by_type.png"
                )
                render_overlay_spatial_tuning(
                    all_sessions_data=non_empty_sessions,
                    tuning_feature=feature,
                    gesture_subset=gesture_subset,
                    out_path=overlay_path_by_type,
                    fit_degrees=fit_degrees,
                    session_colors={sid: scheme.session_color[sid] for sid in non_empty_sessions},
                    dot_alpha=0.4,
                    legend_mode="by_type",
                    session_neuron_types=scheme.session_neuron_type,
                    type_colors=scheme.type_color,
                )
                overlay_path_by_session = (
                    output_base_dir
                    / feature
                    / gesture_subset
                    / f"overlay_{feature}_{gesture_subset}_spatial_tuning_by_session.png"
                )
                render_overlay_spatial_tuning(
                    all_sessions_data=non_empty_sessions,
                    tuning_feature=feature,
                    gesture_subset=gesture_subset,
                    out_path=overlay_path_by_session,
                    fit_degrees=fit_degrees,
                    session_colors={sid: scheme.session_color[sid] for sid in non_empty_sessions},
                    dot_alpha=0.4,
                    legend_mode="by_session",
                    session_neuron_types=scheme.session_neuron_type,
                    type_colors=scheme.type_color,
                )
                print(
                    f"[Spatial Tuning] {feature} / {gesture_subset}: "
                    f"saved overlays (by_type + by_session).",
                    flush=True,
                )

                if all_sessions_csv_dfs:
                    overlay_csv_path = (
                        output_base_dir
                        / feature
                        / gesture_subset
                        / f"overlay_{feature}_{gesture_subset}_spatial_tuning.csv"
                    )
                    overlay_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.concat(all_sessions_csv_dfs, ignore_index=True).to_csv(
                        overlay_csv_path, index=False
                    )

        total_features_rendered += 1

    _write_sentinel(
        sentinel,
        n_sessions=len(session_config_paths),
        n_features=total_features_rendered,
    )
    print(
        f"[Spatial Tuning] Done — {total_features_rendered} feature(s), "
        f"{len(session_config_paths)} session(s). Sentinel written.",
        flush=True,
    )
