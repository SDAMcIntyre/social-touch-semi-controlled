"""Spatial transformation utilities for somatosensory CSV files.

Provides helpers for parsing and serialising ``contact_points`` cells,
resolving transform keys, and applying 4x4 rigid transforms to spatial
columns in-place.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns containing 3D coordinates that must be transformed
# Each inner tuple is one (x, y, z) triplet that shares a single rigid transform.
_XYZ_GROUPS = (
    ("contact_location_x",       "contact_location_y",       "contact_location_z"),
    ("sticker_blue_position_x",  "sticker_blue_position_y",  "sticker_blue_position_z"),
    ("sticker_green_position_x", "sticker_green_position_y", "sticker_green_position_z"),
    ("sticker_yellow_position_x","sticker_yellow_position_y","sticker_yellow_position_z"),
)
_POINTS_COL = "contact_points"


# ------------------------------------------------------------------
# contact_points parsing / serialization
# ------------------------------------------------------------------

def parse_contact_points(point_str: str) -> List[Tuple[float, float, float]]:
    """Parse a ``contact_points`` cell into a list of (x, y, z) tuples.

    Mirrors the logic of
    ``determine_receptive_field.py:parse_contact_points`` so that
    round-trip compatibility is guaranteed.
    """
    if pd.isna(point_str) or not isinstance(point_str, str):
        return []
    stripped = point_str.strip()
    if stripped == "[]":
        return []

    points = []
    for match in re.findall(r"\[([^\]]+)\]", stripped):
        parts = match.strip().lstrip("[").replace(",", " ").split()
        if len(parts) == 3:
            try:
                points.append(
                    (float(parts[0]), float(parts[1]), float(parts[2]))
                )
            except ValueError:
                continue
    return points


def serialize_contact_points(
    points: List[Tuple[float, float, float]],
) -> str:
    """Re-serialize a list of (x, y, z) tuples back to the CSV string format.

    Produces the numpy-style representation used by the rest of the pipeline::

        [[x1 y1 z1] [x2 y2 z2] ...]

    An empty list becomes ``"[]"``.
    """
    if not points:
        return "[]"
    #inner = " ".join(f"[{x} {y} {z}]" for x, y, z in points)
    inner = " ".join(f"[{x:.1f} {y:.1f} {z:.1f}]" for x, y, z in points)
    return f"[{inner}]"


# Keep private aliases for internal backward compatibility
_parse_contact_points = parse_contact_points
_serialize_contact_points = serialize_contact_points


# ------------------------------------------------------------------
# Rigid transform helpers
# ------------------------------------------------------------------

def apply_rigid_transform(
    points: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


# ------------------------------------------------------------------
# Transform key resolution
# ------------------------------------------------------------------

def find_applicable_transform_key(
    transforms: Dict,
    current_video_stem: str,
) -> Optional[str]:
    """Return the most applicable transform key for *current_video_stem*.

    Each transform key ``"<video_stem>:<frame_id>"`` defines a forearm snapshot
    that is valid from its capture point **forward** in the session timeline.
    Block-order ordering determines the timeline (``_block-order01`` precedes
    ``_block-order02``, etc.).

    Selection rules:
    1. If the current video has one or more snapshot keys, return the **earliest**
       (lowest frame_id) one — it covers the widest range of the block.
    2. Otherwise, return the key from the **most recent preceding** block (largest
       block-order number still less than the current one, and within that block
       the largest frame_id).  This correctly propagates a snapshot forward across
       subsequent blocks that have no snapshot of their own.
    3. If no preceding snapshot exists (current block is before all snapshots),
       return ``None`` — the caller will pass through unchanged.

    For non-block-order videos an exact stem match is attempted instead.
    """
    block_match = re.match(r"(.*_block-order)(\d+)", current_video_stem)

    if block_match is None:
        # Non-block-order video: look for an exact stem match
        for key in transforms:
            key_stem = key.rsplit(":", 1)[0]
            if key_stem == current_video_stem:
                return key
        return None

    current_prefix = block_match.group(1)
    current_block = int(block_match.group(2))

    # Parse all keys that share the same session/block-order prefix
    candidates = []  # (block_num, frame_num, key)
    for key in transforms:
        key_stem, _, frame_str = key.rpartition(":")
        m = re.match(r"(.*_block-order)(\d+)", key_stem)
        if m and m.group(1) == current_prefix:
            try:
                candidates.append((int(m.group(2)), int(frame_str), key))
            except ValueError:
                continue

    if not candidates:
        return None

    candidates.sort()  # ascending by (block_num, frame_num)

    # Rule 1: same block — use the earliest snapshot
    same_block = [(b, f, k) for b, f, k in candidates if b == current_block]
    if same_block:
        return same_block[0][2]

    # Rule 2: most recent preceding block — latest snapshot within that block
    preceding = [(b, f, k) for b, f, k in candidates if b < current_block]
    if preceding:
        return preceding[-1][2]  # latest (block_num, frame_num) before current

    # Rule 3: no preceding snapshot — cannot apply a transform
    return None


# ------------------------------------------------------------------
# Transform schedule
# ------------------------------------------------------------------

def get_transform_schedule(
    transforms: Dict,
    current_video_stem: str,
    max_frame: int,
) -> List[Tuple[int, np.ndarray]]:
    """Return an ordered transform schedule for *current_video_stem*.

    Each entry ``(start_frame, T)`` means: apply *T* to all rows whose
    ``frame_index`` satisfies ``start_frame <= frame_index < next_start_frame``
    (last entry covers up to *max_frame*).

    Schedule construction:
    - Entry at frame 0: transform from the most recent preceding block
      (same logic as Rule 2 of :func:`find_applicable_transform_key`).
      Omitted when no preceding snapshot exists.
    - One entry per same-block snapshot whose ``frame_id <= max_frame``,
      in ascending frame order.

    Returns an empty list when no transforms apply (caller should pass through
    unchanged, consistent with the existing Rule 3 behaviour).
    """
    block_match = re.match(r"(.*_block-order)(\d+)", current_video_stem)

    if block_match is None:
        # Non-block-order video: exact stem match only, single entry at frame 0
        for key, entry in transforms.items():
            key_stem = key.rsplit(":", 1)[0]
            if key_stem == current_video_stem:
                T = np.asarray(entry["matrix_4x4"], dtype=np.float64)
                return [(0, T)]
        return []

    current_prefix = block_match.group(1)
    current_block  = int(block_match.group(2))

    # Collect all candidates sharing the same session prefix
    candidates = []  # (block_num, frame_num, key)
    for key, entry in transforms.items():
        key_stem, _, frame_str = key.rpartition(":")
        m = re.match(r"(.*_block-order)(\d+)", key_stem)
        if m and m.group(1) == current_prefix:
            try:
                candidates.append((int(m.group(2)), int(frame_str), key))
            except ValueError:
                continue

    if not candidates:
        return []

    candidates.sort()  # ascending (block_num, frame_num)

    schedule: List[Tuple[int, np.ndarray]] = []

    # --- Initial transform at frame 0 (preceding block, Rule 2) ---
    preceding = [(b, f, k) for b, f, k in candidates if b < current_block]
    if preceding:
        _, _, prec_key = preceding[-1]
        T_prec = np.asarray(transforms[prec_key]["matrix_4x4"], dtype=np.float64)
        schedule.append((0, T_prec))

    # --- Intra-block snapshots within [0, max_frame] ---
    same_block = [(b, f, k) for b, f, k in candidates
                  if b == current_block and f <= max_frame]
    for _, frame_id, key in same_block:
        T = np.asarray(transforms[key]["matrix_4x4"], dtype=np.float64)
        schedule.append((frame_id, T))

    return schedule


# ------------------------------------------------------------------
# In-place spatial transform
# ------------------------------------------------------------------

def transform_spatial_columns_in_place(
    df: pd.DataFrame, transform_4x4: np.ndarray
) -> pd.DataFrame:
    """Apply a rigid transform to spatial columns of *df*, overwriting originals.

    Applies *transform_4x4* to all XYZ triplet columns
    (``contact_location_x/y/z``, ``sticker_blue/green/yellow_position_x/y/z``)
    and ``contact_points``, overwriting the original values.

    If *transform_4x4* is the identity matrix the DataFrame is returned
    unchanged.

    Args:
        df: DataFrame to transform (modified in-place and returned).
        transform_4x4: A 4x4 rigid-body transformation matrix.

    Returns:
        The modified DataFrame.
    """
    if np.allclose(transform_4x4, np.eye(4)):
        return df

    # --- transform XYZ triplets ---
    for group in _XYZ_GROUPS:
        if all(c in df.columns for c in group):
            xyz = df[list(group)].to_numpy(dtype=np.float64)
            mask = ~np.isnan(xyz).any(axis=1)
            if mask.any():
                xyz[mask] = apply_rigid_transform(xyz[mask], transform_4x4)
            for i, col in enumerate(group):
                df[col] = xyz[:, i]

    # --- transform contact_points ---
    if _POINTS_COL in df.columns:
        new_points_col = []
        for cell in df[_POINTS_COL]:
            pts = parse_contact_points(cell)
            if pts:
                arr = np.asarray(pts, dtype=np.float64)
                arr = apply_rigid_transform(arr, transform_4x4)
                new_points_col.append(
                    serialize_contact_points([tuple(row) for row in arr.tolist()])
                )
            else:
                new_points_col.append(cell)
        df[_POINTS_COL] = new_points_col

    return df


# ------------------------------------------------------------------
# Scheduled spatial transform
# ------------------------------------------------------------------

def transform_spatial_columns_scheduled(
    df: pd.DataFrame,
    schedule: List[Tuple[int, np.ndarray]],
    *,
    frame_col: str = "frame_index",
) -> pd.DataFrame:
    """Apply per-segment rigid transforms to spatial columns of *df*.

    Each ``(start_frame, T)`` entry in *schedule* defines a half-open interval
    ``[start_frame, next_start_frame)`` over which *T* is applied.  The final
    entry covers ``[start_frame, +inf)``.

    Rows whose ``frame_col`` value is NaN are skipped (neural-only rows).
    If *schedule* is empty the DataFrame is returned unchanged.

    Args:
        df: DataFrame to transform (modified in-place and returned).
        schedule: Ordered ``[(start_frame, T_4x4), ...]`` list as returned by
            :func:`get_transform_schedule`.
        frame_col: Name of the integer frame-index column.  Default
            ``"frame_index"``.

    Returns:
        The modified DataFrame.
    """
    if not schedule or frame_col not in df.columns:
        return df

    # Validate that all expected spatial columns are present before doing any work.
    for group in _XYZ_GROUPS:
        missing = [c for c in group if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required spatial columns: {missing}. "
                "All XYZ groups must be present before applying a scheduled transform."
            )
    if _POINTS_COL not in df.columns:
        raise ValueError(
            f"DataFrame is missing required column '{_POINTS_COL}'. "
            "All spatial columns must be present before applying a scheduled transform."
        )

    frames = df[frame_col].to_numpy(dtype=np.float64)  # NaN for neural rows
    valid  = ~np.isnan(frames)

    for seg_idx, (start_frame, T) in enumerate(schedule):
        if np.allclose(T, np.eye(4)):
            continue  # identity — nothing to do for this segment

        end_frame = schedule[seg_idx + 1][0] if seg_idx + 1 < len(schedule) else np.inf
        seg_mask  = valid & (frames >= start_frame) & (frames < end_frame)

        if not seg_mask.any():
            continue

        # Transform XYZ triplets
        for group in _XYZ_GROUPS:
            xyz = df.loc[seg_mask, list(group)].to_numpy(dtype=np.float64)
            row_valid = ~np.isnan(xyz).any(axis=1)
            if row_valid.any():
                xyz[row_valid] = apply_rigid_transform(xyz[row_valid], T)
            df.loc[seg_mask, list(group)] = xyz

        # Transform contact_points
        indices = np.where(seg_mask)[0]
        new_vals = []
        for i in indices:
            cell = df.iloc[i][_POINTS_COL]
            pts  = parse_contact_points(cell)
            if pts:
                arr = apply_rigid_transform(np.asarray(pts, dtype=np.float64), T)
                new_vals.append(serialize_contact_points([tuple(r) for r in arr.tolist()]))
            else:
                new_vals.append(cell)
        df.iloc[indices, df.columns.get_loc(_POINTS_COL)] = new_vals

    return df


