"""Spatial transformation of somatosensory CSV files.

Applies a 4x4 rigid transform to the ``contact_location_x/y/z`` and
``contact_points`` columns of a per-block unified CSV, producing a
``_unified_registered.csv`` that contains both the original spatial columns
and new ``*_transformed`` columns expressed in the session-level registered
coordinate frame.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns containing 3D coordinates that must be transformed
_LOCATION_COLS = ("contact_location_x", "contact_location_y", "contact_location_z")
_POINTS_COL = "contact_points"


# ------------------------------------------------------------------
# contact_points parsing / serialization
# ------------------------------------------------------------------

def _parse_contact_points(point_str: str) -> List[Tuple[float, float, float]]:
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


def _serialize_contact_points(
    points: List[Tuple[float, float, float]],
) -> str:
    """Re-serialize a list of (x, y, z) tuples back to the CSV string format.

    Produces the numpy-style representation used by the rest of the pipeline::

        [[x1 y1 z1] [x2 y2 z2] ...]

    An empty list becomes ``"[]"``.
    """
    if not points:
        return "[]"
    inner = " ".join(f"[{x} {y} {z}]" for x, y, z in points)
    return f"[{inner}]"


# ------------------------------------------------------------------
# Rigid transform helpers
# ------------------------------------------------------------------

def _apply_rigid_transform(
    points: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def transform_unified_csv(
    input_csv: Path,
    output_csv: Path,
    transform_4x4: np.ndarray,
) -> Path:
    """Apply a rigid transform to the spatial columns of a somatosensory CSV.

    The function reads *input_csv*, adds ``contact_location_x/y/z_transformed``
    and ``contact_points_transformed`` columns with the transformed coordinates,
    and writes the result to *output_csv*.  Original columns are preserved
    unchanged.  The source file is never modified.

    If the transform is the identity matrix the file is still written
    (to keep the pipeline uniform), but no arithmetic is performed on
    the data.

    Args:
        input_csv: Path to the source ``_unified.csv``.
        output_csv: Path for the transformed output.
        transform_4x4: A 4x4 rigid-body transformation matrix.

    Returns:
        *output_csv* for convenient chaining.
    """
    df = pd.read_csv(input_csv)

    is_identity = np.allclose(transform_4x4, np.eye(4))

    # --- transform contact_location_x/y/z ---
    has_location_cols = all(c in df.columns for c in _LOCATION_COLS)
    if has_location_cols:
        loc = df[list(_LOCATION_COLS)].to_numpy(dtype=np.float64)
        if not is_identity:
            mask = ~np.isnan(loc).any(axis=1)
            if mask.any():
                loc[mask] = _apply_rigid_transform(loc[mask], transform_4x4)
        for i, col in enumerate(_LOCATION_COLS):
            df[col + "_transformed"] = loc[:, i]

    # --- transform contact_points ---
    if _POINTS_COL in df.columns:
        new_points_col = []
        for cell in df[_POINTS_COL]:
            pts = _parse_contact_points(cell)
            if pts and not is_identity:
                arr = np.asarray(pts, dtype=np.float64)
                arr = _apply_rigid_transform(arr, transform_4x4)
                new_points_col.append(
                    _serialize_contact_points(
                        [tuple(row) for row in arr.tolist()]
                    )
                )
            else:
                # Identity transform or empty / NaN — copy original representation
                new_points_col.append(cell)
        df[_POINTS_COL + "_transformed"] = new_points_col

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote transformed CSV to %s", output_csv)
    return output_csv
