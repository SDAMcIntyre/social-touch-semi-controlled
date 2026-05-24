"""Pipeline orchestrator for per-session touch feature radar plots.

Discovers feature extraction CSVs, computes per-gesture statistics with
min-max normalization, and delegates rendering to
``rf_touch_feature_radar_renderer``.

Two normalization scopes are produced for each group:

- **session** — min/max computed within each session independently.
- **global** — min/max computed across all sessions, making sessions directly
  comparable on a shared scale.

Output layout per group and session::

    4_analysed/touch_feature_radar/{group_name}/session/{session_id}/
        {session_id}_radar_{gesture_name}.png
        radar_sentinel.json

    4_analysed/touch_feature_radar/{group_name}/global/{session_id}/
        {session_id}_radar_{gesture_name}.png
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import (
    GESTURE_TYPES,
    TOUCH_ID_COLS_WITH_SESSION,
    session_id_from_path,
)
from analysis.receptive_field_mapping.rendering.rf_touch_feature_radar_renderer import (
    GESTURE_COLORS,
    render_gesture_radar,
)
from analysis.touch_analytics.clustering_pipeline import DATA_TYPE_TO_COLUMNS
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display name map — data type -> human-readable axis label
# ---------------------------------------------------------------------------

#: Mapping from data type name (as used in ``features`` spec) to a short
#: human-readable label with physical unit for radar axis annotations.
_DISPLAY_NAMES: dict[str, str] = {
    "contact_area":            "Contact Area (mm²)",
    "contact_depth":           "Depth (mm)",
    "hand_velocity":           "Hand Velocity (mm/s)",
    "hand_velocity_amplitude": "Hand Velocity (mm/s)",
    "hand_acceleration":       "Hand Accel. (mm/s²)",
    "pressure":                "Pressure (N/mm²)",
    "hand_position":           "Hand Position (mm)",
    "mos_strain":              "Strain",
    "mos_stress_kpa":          "Stress (kPa)",
    "mos_strain_rate":         "Strain Rate (1/s)",
    "mos_elastic_energy_mj":   "Elastic E. (mJ)",
    "mos_impulse_mns":         "Impulse (mN·s)",
    "mechanics_of_solids":     "MoS",
    "location":                "Location (mm)",
}

# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------


def _find_feature_csv(db_path: Path, aggregation: str, session_id: str) -> Path:
    """Return the touch-summary CSV for *session_id* under *aggregation*.

    Parameters
    ----------
    db_path:
        Root database path (the directory that contains ``4_analysed/``).
    aggregation:
        Aggregation folder name (e.g. ``"mean_during_iff"``).
    session_id:
        Session identifier prefix (e.g. ``"P01_semicontrolled"``).

    Returns
    -------
    Path
        Resolved path to the single matching CSV.

    Raises
    ------
    ValueError
        If no CSV is found, or if more than one CSV matches the pattern.
    """
    feature_dir = db_path / "4_analysed" / "touch_features" / aggregation
    candidates = sorted(feature_dir.glob(f"{session_id}_*_touch_summary.csv"))
    if not candidates:
        raise ValueError(
            f"No touch feature CSV found for session '{session_id}' under "
            f"{feature_dir}. "
            f"Ensure 'touch_feature_extraction' with aggregation '{aggregation}' has run."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple feature CSVs found for session '{session_id}': {candidates}. "
            f"Expected exactly one."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Aggregation folder resolution
# ---------------------------------------------------------------------------


def _resolve_required_aggregation_folders(features_spec: dict) -> list[str]:
    """Return deduplicated folder names required for *features_spec*.

    Each value in ``features_spec`` is a list of aggregation method names,
    which correspond to subfolder names under ``4_analysed/touch_features/``.
    The ``location: [mean]`` special case does not add ``mean`` to the folder
    list because mean location values are already present as shared columns in
    every per-aggregation CSV — this mirrors the clustering pipeline's
    ``_resolve_required_feature_folders`` behaviour.

    Parameters
    ----------
    features_spec:
        Dict mapping data type -> list of aggregation methods, as written in
        the ``radar_groups`` YAML config.

    Returns
    -------
    list[str]
        Unique aggregation folder names (order-preserving deduplication).
    """
    folders: list[str] = []

    for data_type, aggregations in features_spec.items():
        if data_type == "location":
            non_mean_aggs = [a for a in aggregations if a != "mean"]
            folders.extend(non_mean_aggs)
        else:
            folders.extend(aggregations)

    seen: set[str] = set()
    deduped: list[str] = []
    for f in folders:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ---------------------------------------------------------------------------
# Multi-folder CSV loading / merging
# ---------------------------------------------------------------------------


def _load_and_merge_feature_csvs(
    db_path: Path,
    agg_folders: list[str],
    session_id: str,
) -> pd.DataFrame:
    """Load and merge feature CSVs from multiple aggregation folders.

    Parameters
    ----------
    db_path:
        Root database path (the directory that contains ``4_analysed/``).
    agg_folders:
        Ordered list of aggregation folder names to load.
    session_id:
        Session identifier prefix.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from each folder.  When only one
        folder is requested the CSV is returned directly without merging.

    Raises
    ------
    ValueError
        If any requested CSV is missing, or if overlapping columns (outside
        the join key) are found across CSVs.
    """
    if not agg_folders:
        raise ValueError(
            f"_load_and_merge_feature_csvs: no aggregation folders provided for "
            f"session '{session_id}'."
        )

    # Single-folder fast path — no merge needed
    if len(agg_folders) == 1:
        csv_path = _find_feature_csv(db_path, agg_folders[0], session_id)
        return pd.read_csv(csv_path)

    # Multi-folder: load each CSV and check for column overlaps before merging
    join_key = list(TOUCH_ID_COLS_WITH_SESSION)
    frames: list[pd.DataFrame] = []
    for folder in agg_folders:
        csv_path = _find_feature_csv(db_path, folder, session_id)
        frames.append(pd.read_csv(csv_path))

    # Validate: no non-key column should appear in more than one CSV
    seen_cols: set[str] = set(join_key)
    for i, df in enumerate(frames):
        non_key = [c for c in df.columns if c not in seen_cols]
        duplicates = [c for c in non_key if c in seen_cols]
        if duplicates:
            raise ValueError(
                f"_load_and_merge_feature_csvs: column(s) {duplicates} appear in "
                f"folder '{agg_folders[i]}' and in a previously loaded CSV for "
                f"session '{session_id}'. Disambiguate by choosing non-overlapping "
                f"aggregation methods."
            )
        seen_cols.update(non_key)

    # Merge iteratively on join key
    merged = frames[0]
    for i, other in enumerate(frames[1:], start=1):
        merge_keys = [k for k in join_key if k in merged.columns and k in other.columns]
        non_key_other = [c for c in other.columns if c not in join_key]
        subset = other[merge_keys + non_key_other]
        merged = merged.merge(subset, on=merge_keys, how="inner")

    return merged


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def _resolve_radar_columns(
    df: pd.DataFrame,
    features_spec: dict,
) -> tuple[list[str], list[str]]:
    """Resolve radar column names and display labels from *features_spec*.

    For each data type in *features_spec*, looks up the corresponding base
    column name(s) via ``DATA_TYPE_TO_COLUMNS``.  For each aggregation method,
    the full column name is ``{base_col}_{aggregation}``.  When a data type
    maps to multiple base columns (e.g. ``hand_velocity`` -> 3 axes) all of
    them are expanded.

    Parameters
    ----------
    df:
        Merged feature DataFrame for the session.
    features_spec:
        Dict mapping data type -> list of aggregation methods.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(column_names, display_labels)`` — one entry per radar axis.

    Raises
    ------
    ValueError
        If a data type is unknown (not in ``DATA_TYPE_TO_COLUMNS``), or if a
        resolved column name is not present in *df* (fail-fast).
    """
    column_names: list[str] = []
    display_labels: list[str] = []

    for data_type, aggregations in features_spec.items():
        base_cols = DATA_TYPE_TO_COLUMNS.get(data_type)
        if base_cols is None:
            raise ValueError(
                f"_resolve_radar_columns: unknown data type '{data_type}'. "
                f"Known types: {sorted(DATA_TYPE_TO_COLUMNS)}."
            )

        base_label = _DISPLAY_NAMES.get(data_type, data_type)
        multi_agg = len(aggregations) > 1

        for agg in aggregations:
            for base_col in base_cols:
                col = f"{base_col}_{agg}"
                if col not in df.columns:
                    raise ValueError(
                        f"_resolve_radar_columns: expected column '{col}' not found "
                        f"in DataFrame. Available columns: {sorted(df.columns)}. "
                        f"Ensure 'touch_feature_extraction' with aggregation '{agg}' "
                        f"has run for this session."
                    )
                column_names.append(col)

                if len(base_cols) > 1:
                    # Multi-column type: qualify with axis suffix (e.g. "_x")
                    axis_suffix = base_col[len(data_type):] if base_col.startswith(data_type) else f"_{base_col}"
                    if multi_agg:
                        display_labels.append(f"{base_label}{axis_suffix} ({agg})")
                    else:
                        display_labels.append(f"{base_label}{axis_suffix}")
                else:
                    if multi_agg:
                        display_labels.append(f"{base_label} ({agg})")
                    else:
                        display_labels.append(base_label)

    return column_names, display_labels


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _normalize_features(
    raw: np.ndarray,
    col_min: np.ndarray,
    col_max: np.ndarray,
) -> np.ndarray:
    """Min-max normalize *raw* using externally supplied bounds.

    Zero-variance columns (where ``col_max == col_min``) are set to 0.0.
    """
    col_range = col_max - col_min
    safe_range = np.where(col_range == 0.0, 1.0, col_range)
    return np.where(col_range == 0.0, 0.0, (raw - col_min) / safe_range)


def _compute_gesture_stats(
    normalized: np.ndarray,
    raw: np.ndarray,
    gesture_types: np.ndarray,
) -> dict:
    """Compute per-gesture median / Q25 / Q75 on already-normalised data.

    Parameters
    ----------
    normalized:
        2-D array of shape ``(n_touches, n_features)`` in [0, 1].
    raw:
        2-D array of same shape — un-normalised feature values.
    gesture_types:
        1-D string array of length ``n_touches`` (the ``gesture_type`` column).

    Returns
    -------
    dict
        Mapping of gesture name → ``{"medians", "q25", "q75", "raw_medians"}``.
        Includes ``"all"`` for all rows combined.
        Gesture types with fewer than 2 rows are omitted.
    """
    results: dict = {}

    subsets: list[tuple[str, np.ndarray | None]] = [("all", None)] + [
        (gtype, gesture_types == gtype) for gtype in GESTURE_TYPES
    ]

    for name, mask in subsets:
        norm_rows = normalized if mask is None else normalized[mask]
        raw_rows = raw if mask is None else raw[mask]

        if len(norm_rows) < 2:
            logger.info(
                "[Touch Feature Radar] Gesture '%s' has %d row(s) — need ≥2 to "
                "compute statistics; skipping.",
                name, len(norm_rows),
            )
            continue

        results[name] = {
            "medians": np.nanmedian(norm_rows, axis=0),
            "q25": np.nanpercentile(norm_rows, 25, axis=0),
            "q75": np.nanpercentile(norm_rows, 75, axis=0),
            "raw_medians": np.nanmedian(raw_rows, axis=0),
        }

    return results


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


def _write_sentinel(sentinel: Path, session_id: str, produced: list[Path]) -> None:
    data = {
        "session_id": session_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_pngs": len(produced),
        "pngs": [str(p) for p in produced],
    }
    with open(sentinel, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def _render_radar_pngs(
    stats: dict,
    display_labels: list[str],
    axis_max_values: np.ndarray,
    session_id: str,
    group_name: str,
    scope_label: str,
    output_dir: Path,
) -> list[Path]:
    """Render per-gesture radar PNGs into *output_dir* and return produced paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for gesture_name, gstats in stats.items():
        color = GESTURE_COLORS.get(gesture_name, "#ffffff")
        title = f"{session_id} | {gesture_name} | {scope_label}"
        out_path = output_dir / f"{session_id}_radar_{gesture_name}.png"

        render_gesture_radar(
            labels=display_labels,
            medians=gstats["medians"],
            q25=gstats["q25"],
            q75=gstats["q75"],
            title=title,
            color=color,
            out_path=out_path,
            axis_max_values=axis_max_values,
            raw_medians=gstats["raw_medians"],
        )
        produced.append(out_path)
        print(
            f"[Touch Feature Radar] {group_name} ({scope_label}) / {session_id}: "
            f"saved {out_path.name}",
            flush=True,
        )

    return produced


def run_touch_feature_radar(
    session_configs: list,
    radar_groups: dict,
    force_processing: bool = False,
) -> None:
    """Render per-session touch feature radar plots for each enabled radar group.

    Two normalization scopes are produced:

    - **session** — min/max computed within each session independently;
      idempotency via ``should_process_task`` sentinel.
    - **global** — min/max computed across all sessions (always recomputed).

    Outputs are written to
    ``{db}/4_analysed/touch_feature_radar/{group_name}/{scope}/{session_id}/``.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, db_path)`` tuples, following the same
        convention as other pipelines in this package.
    radar_groups:
        Dict mapping group name -> group spec.  Each spec must have an
        ``enabled`` bool and a ``features`` dict (data type -> list of
        aggregation methods).  Groups with ``enabled: false`` are skipped.
    force_processing:
        If ``True``, reprocess sessions even when the sentinel file is fresh.

    Raises
    ------
    ValueError
        If *radar_groups* is empty.
    """
    if not radar_groups:
        raise ValueError(
            "run_touch_feature_radar: 'radar_groups' is empty. "
            "Define at least one radar group in the DAG config."
        )

    for group_name, group_spec in radar_groups.items():
        if not group_spec.get("enabled", True):
            logger.info(
                "[Touch Feature Radar] Group '%s': disabled — skipping.", group_name
            )
            continue

        features_spec: dict = group_spec.get("features", {})
        if not features_spec:
            raise ValueError(
                f"[Touch Feature Radar] Group '{group_name}': 'features' is empty or missing. "
                f"Define at least one data type in the group spec."
            )

        agg_folders = _resolve_required_aggregation_folders(features_spec)
        if not agg_folders:
            raise ValueError(
                f"[Touch Feature Radar] Group '{group_name}': no aggregation folders "
                f"resolved from features spec. "
                f"A group with only 'location: [mean]' is not supported — "
                f"add at least one non-mean-location feature or aggregation."
            )

        print(
            f"[Touch Feature Radar] Group '{group_name}' — folders: {agg_folders}",
            flush=True,
        )

        # =================================================================
        # Pass 1 — Load all sessions, resolve columns, collect raw matrices
        # =================================================================
        session_data: list[dict] = []
        resolved_cols: list[str] | None = None
        display_labels: list[str] | None = None

        for csv_path, database_path in session_configs:
            csv_path = Path(csv_path)
            database_path = Path(database_path)
            session_id = session_id_from_path(csv_path)

            df = _load_and_merge_feature_csvs(database_path, agg_folders, session_id)

            if "gesture_type" not in df.columns:
                raise ValueError(
                    f"[Touch Feature Radar] {session_id}: 'gesture_type' column not "
                    f"found in merged feature DataFrame for group '{group_name}'. "
                    f"Available columns: {list(df.columns)}"
                )

            cols, labels = _resolve_radar_columns(df, features_spec)
            if resolved_cols is None:
                resolved_cols, display_labels = cols, labels

            raw = df[cols].to_numpy(dtype=float)
            gesture_types = df["gesture_type"].to_numpy()

            session_data.append({
                "session_id": session_id,
                "database_path": database_path,
                "raw": raw,
                "gesture_types": gesture_types,
                "feature_csv": _find_feature_csv(
                    database_path, agg_folders[0], session_id
                ),
            })

        if not session_data:
            logger.info(
                "[Touch Feature Radar] Group '%s': no sessions to process.",
                group_name,
            )
            continue

        # Global min/max across all sessions
        all_raw = np.vstack([s["raw"] for s in session_data])
        global_min = np.nanmin(all_raw, axis=0)
        global_max = np.nanmax(all_raw, axis=0)

        # =================================================================
        # Pass 2 — Render both normalization scopes per session
        # =================================================================
        for entry in session_data:
            session_id = entry["session_id"]
            database_path = entry["database_path"]
            raw = entry["raw"]
            gesture_types = entry["gesture_types"]
            base_dir = (
                database_path / "4_analysed" / "touch_feature_radar" / group_name
            )

            # --- Session-normalized scope (idempotency via sentinel) ---
            session_dir = base_dir / "session" / session_id
            session_sentinel = session_dir / "radar_sentinel.json"

            session_skip = not should_process_task(
                output_paths=session_sentinel,
                input_paths=entry["feature_csv"],
                force=force_processing,
            )
            if session_skip:
                logger.info(
                    "[Touch Feature Radar] %s (session) / %s: up-to-date — skipping.",
                    group_name, session_id,
                )
            else:
                session_min = np.nanmin(raw, axis=0)
                session_max = np.nanmax(raw, axis=0)
                norm_session = _normalize_features(raw, session_min, session_max)
                stats_session = _compute_gesture_stats(
                    norm_session, raw, gesture_types,
                )

                if not stats_session:
                    logger.info(
                        "[Touch Feature Radar] %s (session) / %s: no gesture type has "
                        "≥2 touches — no PNGs produced.",
                        group_name, session_id,
                    )
                    session_dir.mkdir(parents=True, exist_ok=True)
                    _write_sentinel(session_sentinel, session_id, produced=[])
                else:
                    produced = _render_radar_pngs(
                        stats_session, display_labels, session_max,
                        session_id, group_name, "session", session_dir,
                    )
                    _write_sentinel(session_sentinel, session_id, produced=produced)
                    print(
                        f"[Touch Feature Radar] {group_name} (session) / {session_id}: "
                        f"done — {len(produced)} PNG(s) written.",
                        flush=True,
                    )

            # --- Global-normalized scope (always recomputed) ---
            global_dir = base_dir / "global" / session_id
            norm_global = _normalize_features(raw, global_min, global_max)
            stats_global = _compute_gesture_stats(
                norm_global, raw, gesture_types,
            )

            if not stats_global:
                logger.info(
                    "[Touch Feature Radar] %s (global) / %s: no gesture type has "
                    "≥2 touches — no PNGs produced.",
                    group_name, session_id,
                )
                global_dir.mkdir(parents=True, exist_ok=True)
            else:
                produced = _render_radar_pngs(
                    stats_global, display_labels, global_max,
                    session_id, group_name, "global", global_dir,
                )
                print(
                    f"[Touch Feature Radar] {group_name} (global) / {session_id}: "
                    f"done — {len(produced)} PNG(s) written.",
                    flush=True,
                )
