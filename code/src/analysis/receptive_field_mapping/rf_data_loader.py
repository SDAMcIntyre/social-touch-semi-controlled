"""Data loader for receptive field mapping.

Joins the unified touch summary CSV with per-trial raw CSVs to produce
grouped spatial data (spike counts and total counts per 3D contact point),
partitioned by user-chosen grouping variables (e.g. touch type, direction,
depth bin).
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.touch_analytics.touch_config import DISCRETIZATION_CONFIG

from .rf_mapping_config import GroupedSpatialData, RFMappingConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PLY path resolution
# ------------------------------------------------------------------

def load_forearm_vertices(ply_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load forearm point-cloud vertices with a .npy sidecar cache.

    Cache path: ``{ply_path.parent}/{ply_path.stem}_vertices.npy``.
    Returns ``None`` when *ply_path* is ``None`` or the file does not exist.
    On cache hit (sidecar mtime ≥ PLY mtime) loads the .npy directly.
    On cache miss, loads via open3d, saves the sidecar, then returns the array.
    Raises ``ValueError`` for a cached .npy with the wrong shape.
    """
    if ply_path is None or not ply_path.exists():
        return None

    npy_path = ply_path.parent / f"{ply_path.stem}_vertices.npy"

    if npy_path.exists() and npy_path.stat().st_mtime >= ply_path.stat().st_mtime:
        arr = np.load(npy_path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"load_forearm_vertices: cached .npy has unexpected shape {arr.shape} "
                f"(expected (N, 3)): {npy_path}"
            )
        return arr

    import open3d as o3d  # type: ignore
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None

    try:
        np.save(npy_path, pts.astype(np.float64))
    except Exception:
        logger.warning("Could not save forearm vertices cache: %s", npy_path)

    return pts


def resolve_forearm_ply(session_dir: Path, session_id: str) -> Optional[Path]:
    """Resolve the forearm PLY path for a session in RF-centered coordinate space.

    The aggregated session CSV always contains contact points in RF-centered
    space (produced by ``blocks_rf_centered/``).  The forearm PLY must be in
    the same space to avoid misalignment.  Returns ``None`` if the
    RF-centered PLY does not exist.
    """
    rf_centered = session_dir / 'forearm_rf_centered' / f'{session_id}_forearm.ply'
    if rf_centered.exists():
        return rf_centered

    return None


# ------------------------------------------------------------------
# Contact-point parsing (canonical copy)
# ------------------------------------------------------------------

def parse_contact_points(point_str: str) -> List[Tuple[float, float, float]]:
    """Parse a string representation of 3D points.

    Expected format: ``[[x1 y1 z1] [x2 y2 z2]]`` or ``"[]"``.
    Returns a list of ``(x, y, z)`` tuples.
    """
    if pd.isna(point_str) or point_str.strip() == "[]" or not isinstance(point_str, str):
        return []

    points: List[Tuple[float, float, float]] = []
    matches = re.findall(r'\[([^\]]+)\]', point_str)

    for match in matches:
        try:
            parts = match.strip().lstrip("[").split()
            if len(parts) == 3:
                pt = (float(parts[0]), float(parts[1]), float(parts[2]))
                points.append(pt)
        except ValueError:
            continue

    return points


# ------------------------------------------------------------------
# Discretization helpers
# ------------------------------------------------------------------

def _discretize_column(
    series: pd.Series,
    q: int,
) -> pd.Series:
    """Discretize a continuous series into *q* quantile bins.

    Returns a Series of string labels.  Rows with NaN values are kept as
    ``"unknown"``.
    """
    try:
        binned = pd.qcut(series, q=q, duplicates="drop")
        return binned.astype(str).fillna("unknown")
    except (ValueError, TypeError):
        # All identical values or not enough distinct values for q bins.
        logger.warning(
            "Could not discretize column '%s' into %d bins; using 'all' as label.",
            series.name,
            q,
        )
        return pd.Series("all", index=series.index)


def _build_group_lookup(
    summary_df: pd.DataFrame,
    grouping_columns: List[str],
    trial_id_col: str,
    touch_id_col: str,
) -> Dict[Tuple, str]:
    """Build a ``(trial_id, single_touch_id) -> group_label`` lookup dict.

    Continuous columns are discretized according to
    ``DISCRETIZATION_CONFIG``; categorical columns are used as-is.
    """
    continuous_cfg = DISCRETIZATION_CONFIG.get("continuous_vars", {})
    categorical_vars = set(DISCRETIZATION_CONFIG.get("categorical_vars", []))

    # Validate that requested grouping columns exist in the summary
    available = set(summary_df.columns)
    valid_columns: List[str] = []
    for col in grouping_columns:
        if col not in available:
            logger.warning(
                "Grouping column '%s' not found in summary CSV (available: %s). "
                "Skipping this column.",
                col,
                ", ".join(sorted(available)),
            )
        else:
            valid_columns.append(col)

    if not valid_columns:
        logger.warning(
            "No valid grouping columns remain. All data will be assigned "
            "to a single group ('all')."
        )
        return {
            (row[trial_id_col], row[touch_id_col]): "all"
            for _, row in summary_df.iterrows()
        }

    # Prepare label series for each grouping column
    label_parts: Dict[str, pd.Series] = {}
    for col in valid_columns:
        if col in continuous_cfg:
            q = continuous_cfg[col].get("q", 3)
            label_parts[col] = _discretize_column(summary_df[col], q)
        elif col in categorical_vars:
            label_parts[col] = summary_df[col].astype(str).fillna("unknown")
        else:
            # Unknown column type: treat as categorical
            label_parts[col] = summary_df[col].astype(str).fillna("unknown")

    # Concatenate labels into a single group string per row
    combined_labels = pd.DataFrame(label_parts)
    group_strings = combined_labels.apply(lambda row: "_".join(row.values), axis=1)

    lookup: Dict[Tuple, str] = {}
    for idx, group_label in group_strings.items():
        trial_id = summary_df.at[idx, trial_id_col]
        touch_id = summary_df.at[idx, touch_id_col]
        lookup[(trial_id, touch_id)] = group_label

    return lookup


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_grouped_spatial_data(
    raw_csv_paths: List[Path],
    summary_csv_path: Path,
    grouping_columns: List[str],
    config: RFMappingConfig,
) -> Dict[str, GroupedSpatialData]:
    """Load and group spatial contact-point data for RF mapping.

    Parameters
    ----------
    raw_csv_paths:
        Paths to per-trial raw CSVs containing frame-level rows with
        ``single_touch_id``, ``Nerve_spike``, and ``contact_points``
        columns.
    summary_csv_path:
        Path to the unified touch summary CSV that contains one row per
        touch with the grouping metadata (e.g. ``type_metadata``,
        ``direction``, ``max_depth``).
    grouping_columns:
        Column names from the summary CSV to group touches by.
        Continuous columns are discretized via ``pd.qcut``; categorical
        columns are used as-is.
    config:
        ``RFMappingConfig`` providing column-name mappings.

    Returns
    -------
    dict mapping group label strings to ``GroupedSpatialData`` objects.
    """
    col = config.columns

    # --- Read summary CSV ---
    if not summary_csv_path.exists():
        logger.error("Summary CSV not found: %s", summary_csv_path)
        return {}

    try:
        summary_df = pd.read_csv(summary_csv_path)
    except Exception:
        logger.exception("Failed to read summary CSV: %s", summary_csv_path)
        return {}

    if summary_df.empty:
        logger.warning("Summary CSV is empty: %s", summary_csv_path)
        return {}

    # Ensure key columns exist in summary
    for required in (col.trial_id, col.touch_id):
        if required not in summary_df.columns:
            logger.error(
                "Required column '%s' missing from summary CSV.", required
            )
            return {}

    # --- Build group lookup ---
    group_lookup = _build_group_lookup(
        summary_df,
        grouping_columns,
        trial_id_col=col.trial_id,
        touch_id_col=col.touch_id,
    )

    # --- Initialize grouped data containers ---
    groups: Dict[str, GroupedSpatialData] = {}
    for label in set(group_lookup.values()):
        groups[label] = GroupedSpatialData(group_label=label)

    # Track unique (trial_id, touch_id) pairs seen per group for touch_count
    group_touch_sets: Dict[str, set] = {label: set() for label in groups}

    # --- Iterate raw CSVs ---
    for csv_path in raw_csv_paths:
        if not csv_path.exists():
            logger.warning("Raw CSV not found, skipping: %s", csv_path)
            continue

        try:
            points_col = col.points
            needed_cols = [col.trial_id, col.touch_id, col.spike, points_col]
            # Only read columns that actually exist (trial_id may be absent
            # in very old CSVs — handle gracefully)
            available_cols = set(pd.read_csv(csv_path, nrows=0).columns)
            missing = [c for c in needed_cols if c not in available_cols]
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s",
                    csv_path.name,
                    missing,
                )
                continue

            df = pd.read_csv(csv_path, usecols=needed_cols)
        except KeyError as exc:
            logger.warning(
                "Skipping %s: column resolution failed (%s)", csv_path.name, exc
            )
            continue
        except Exception:
            logger.exception("Error reading %s", csv_path.name)
            continue

        if df.empty:
            continue

        trial_ids = df[col.trial_id].to_numpy()
        touch_ids = df[col.touch_id].to_numpy()
        spikes = df[col.spike].to_numpy()
        raw_points = df[points_col]

        for idx in range(len(df)):
            # Skip background / noise rows
            if touch_ids[idx] == 0:
                continue

            parsed = parse_contact_points(raw_points.iat[idx])
            if not parsed:
                continue

            key = (trial_ids[idx], touch_ids[idx])
            group_label = group_lookup.get(key)
            if group_label is None:
                # Touch not present in summary — skip silently
                continue

            gsd = groups[group_label]
            gsd.total_counts.update(parsed)

            if spikes[idx] == 1:
                gsd.spike_counts.update(parsed)

            group_touch_sets[group_label].add(key)

    # --- Finalise touch counts ---
    for label, touch_set in group_touch_sets.items():
        groups[label].touch_count = len(touch_set)

    # Log summary
    total_touches = sum(g.touch_count for g in groups.values())
    total_points = sum(len(g.total_counts) for g in groups.values())
    logger.info(
        "Loaded %d groups, %d total unique touches, %d total unique spatial points.",
        len(groups),
        total_touches,
        total_points,
    )

    return groups
