"""Pipeline orchestrator for per-session touch feature radar plots.

Discovers feature extraction CSVs, computes per-gesture statistics with
session-wide min-max normalization, and delegates rendering to
``rf_touch_feature_radar_renderer``.

Outputs per group and session
(in ``4_analysed/touch_feature_radar/{group_name}/{session_id}/``):
  - ``{session_id}_radar_{gesture_name}.png``  — one per gesture type present
  - ``{session_id}_radar_composite.png``        — overlay when ≥2 gesture types present
  - ``radar_sentinel.json``                     — idempotency sentinel
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
    render_gesture_radar_composite,
)
from analysis.touch_analytics.clustering_pipeline import DATA_TYPE_TO_COLUMNS
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display name map — data type -> human-readable axis label
# ---------------------------------------------------------------------------

#: Mapping from data type name (as used in ``features`` spec) to a short
#: human-readable label for radar axis annotations.
_DISPLAY_NAMES: dict[str, str] = {
    "contact_area":            "Contact Area",
    "contact_depth":           "Depth",
    "hand_velocity":           "Hand Velocity",
    "hand_velocity_amplitude": "Hand Velocity",
    "hand_acceleration":       "Hand Accel.",
    "pressure":                "Pressure",
    "hand_position":           "Hand Position",
    "mos_strain":              "Strain",
    "mos_stress_kpa":          "Stress",
    "mos_strain_rate":         "Strain Rate",
    "mos_elastic_energy_mj":   "Elastic E.",
    "mos_impulse_mns":         "Impulse",
    "mechanics_of_solids":     "MoS",
    "location":                "Location",
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


def _compute_session_stats(df: pd.DataFrame, resolved_cols: list[str]) -> dict:
    """Compute session-wide min-max normalised statistics per gesture type.

    Normalization is computed across **all** rows (session-wide), so all
    gesture-type subsets share the same [0, 1] scale and are directly
    comparable.

    Parameters
    ----------
    df:
        Feature summary DataFrame (all rows for the session).
    resolved_cols:
        Resolved column names (one per feature axis).

    Returns
    -------
    dict
        Mapping of gesture name → ``{"medians": np.ndarray, "q25": np.ndarray,
        "q75": np.ndarray}``.  Includes an ``"all"`` key for all rows combined.
        Gesture types with fewer than 2 rows are omitted (logged at INFO level).
    """
    # --- Session-wide min-max normalisation ---
    raw = df[resolved_cols].to_numpy(dtype=float)  # shape: (n_touches, n_features)
    col_min = np.nanmin(raw, axis=0)
    col_max = np.nanmax(raw, axis=0)
    col_range = col_max - col_min

    # Zero-variance guard: if max == min, set normalized value to 0.0
    safe_range = np.where(col_range == 0.0, 1.0, col_range)
    normalized = np.where(
        col_range == 0.0,
        0.0,
        (raw - col_min) / safe_range,
    )

    results: dict = {}

    subsets: list[tuple[str, pd.Series | None]] = [("all", None)] + [
        (gtype, df["gesture_type"] == gtype) for gtype in GESTURE_TYPES
    ]

    for name, mask in subsets:
        if mask is None:
            rows = normalized
        else:
            row_indices = mask.to_numpy()
            rows = normalized[row_indices]

        n_rows = len(rows)
        if n_rows < 2:
            logger.info(
                "[Touch Feature Radar] Gesture '%s' has %d row(s) — need ≥2 to compute "
                "statistics; skipping.",
                name, n_rows,
            )
            continue

        medians = np.nanmedian(rows, axis=0)
        q25 = np.nanpercentile(rows, 25, axis=0)
        q75 = np.nanpercentile(rows, 75, axis=0)

        results[name] = {
            "medians": medians,
            "q25": q25,
            "q75": q75,
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


def run_touch_feature_radar(
    session_configs: list,
    radar_groups: dict,
    force_processing: bool = False,
) -> None:
    """Render per-session touch feature radar plots for each enabled radar group.

    For each enabled group and each session, produces one radar PNG per gesture
    type present (≥2 touches) plus a composite overlay PNG when ≥2 gesture
    types are available.

    Outputs are written to
    ``{db}/4_analysed/touch_feature_radar/{group_name}/{session_id}/``.

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

        for csv_path, database_path in session_configs:
            csv_path = Path(csv_path)
            database_path = Path(database_path)

            session_id = session_id_from_path(csv_path)
            output_dir = (
                database_path
                / "4_analysed"
                / "touch_feature_radar"
                / group_name
                / session_id
            )
            sentinel = output_dir / "radar_sentinel.json"

            # --- Idempotency check ---
            # Use the first aggregation folder's CSV as the representative input.
            feature_csv = _find_feature_csv(database_path, agg_folders[0], session_id)
            if not should_process_task(
                output_paths=sentinel,
                input_paths=feature_csv,
                force=force_processing,
            ):
                logger.info(
                    "[Touch Feature Radar] %s / %s: outputs up-to-date — skipping.",
                    group_name,
                    session_id,
                )
                continue

            print(
                f"[Touch Feature Radar] {group_name} / {session_id}: processing...",
                flush=True,
            )

            # --- Load + merge feature CSV(s) ---
            df = _load_and_merge_feature_csvs(database_path, agg_folders, session_id)

            # --- Validate gesture_type column ---
            if "gesture_type" not in df.columns:
                raise ValueError(
                    f"[Touch Feature Radar] {session_id}: 'gesture_type' column not found "
                    f"in merged feature DataFrame for group '{group_name}'. "
                    f"Available columns: {list(df.columns)}"
                )

            # --- Resolve feature columns and display labels ---
            resolved_cols, display_labels = _resolve_radar_columns(df, features_spec)

            # --- Compute per-gesture statistics ---
            stats = _compute_session_stats(df, resolved_cols)

            if not stats:
                logger.info(
                    "[Touch Feature Radar] %s / %s: no gesture type has ≥2 touches — "
                    "no PNGs produced.",
                    group_name,
                    session_id,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_sentinel(sentinel, session_id, produced=[])
                continue

            # --- Render outputs ---
            output_dir.mkdir(parents=True, exist_ok=True)
            produced: list[Path] = []

            # Individual per-gesture radar plots
            for gesture_name, gstats in stats.items():
                color = GESTURE_COLORS.get(gesture_name, "#ffffff")
                title = f"{session_id} | {gesture_name} | touch feature radar"
                out_path = output_dir / f"{session_id}_radar_{gesture_name}.png"

                render_gesture_radar(
                    labels=display_labels,
                    medians=gstats["medians"],
                    q25=gstats["q25"],
                    q75=gstats["q75"],
                    title=title,
                    color=color,
                    out_path=out_path,
                )
                produced.append(out_path)
                print(
                    f"[Touch Feature Radar] {group_name} / {session_id}: "
                    f"saved {out_path.name}",
                    flush=True,
                )

            # Composite overlay when ≥2 gesture types (excluding "all")
            non_all_gesture_names = [g for g in stats if g != "all"]
            if len(non_all_gesture_names) >= 2:
                composite_stats: dict[str, dict] = {}
                for gesture_name, gstats in stats.items():
                    if gesture_name == "all":
                        continue
                    composite_stats[gesture_name] = {
                        "medians": gstats["medians"],
                        "q25": gstats["q25"],
                        "q75": gstats["q75"],
                        "color": GESTURE_COLORS.get(gesture_name, "#ffffff"),
                    }

                composite_path = output_dir / f"{session_id}_radar_composite.png"
                render_gesture_radar_composite(
                    labels=display_labels,
                    gesture_stats=composite_stats,
                    title=f"{session_id} | all gesture types | touch feature radar",
                    out_path=composite_path,
                )
                produced.append(composite_path)
                print(
                    f"[Touch Feature Radar] {group_name} / {session_id}: "
                    f"saved {composite_path.name}",
                    flush=True,
                )

            # --- Write sentinel ---
            _write_sentinel(sentinel, session_id, produced=produced)
            print(
                f"[Touch Feature Radar] {group_name} / {session_id}: "
                f"done — {len(produced)} PNG(s) written.",
                flush=True,
            )
