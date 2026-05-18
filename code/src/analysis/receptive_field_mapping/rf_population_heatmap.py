"""Shared computation functions for population RF heatmaps."""

import numpy as np

from analysis.touch_analytics.clustering_pipeline import GESTURE_TYPES  # noqa: F401


def compute_rf_heatmap(
    touch_indices: list[int],
    rf_vertex_indices: list[np.ndarray],
    rf_values: list[np.ndarray],
    n_verts: int,
) -> np.ndarray:
    """Mean RF value per vertex across selected touches. NaN for uncontacted."""
    result = np.zeros(n_verts, dtype=np.float64)
    count = np.zeros(n_verts, dtype=np.int64)
    for idx in touch_indices:
        verts = rf_vertex_indices[idx]
        vals = rf_values[idx]
        if len(verts) == 0:
            continue
        np.add.at(result, verts, vals)
        np.add.at(count, verts, 1)
    nonzero = count > 0
    result[nonzero] /= count[nonzero]
    result[~nonzero] = np.nan
    return result


def compute_unique_touch_count(
    cp_vertex_idx: np.ndarray,
    cp_touch_idx: np.ndarray,
    cp_mask: np.ndarray,
    n_verts: int,
) -> np.ndarray:
    """Per-vertex count of distinct touches among masked contact points."""
    vertex_idx = cp_vertex_idx[cp_mask]
    touch_idx = cp_touch_idx[cp_mask]

    if len(vertex_idx) == 0:
        return np.zeros(n_verts, dtype=np.int64)

    # Encode (vertex, touch) pairs as a single integer for fast deduplication.
    max_touch = int(touch_idx.max()) + 1
    key = vertex_idx * max_touch + touch_idx
    unique_keys = np.unique(key)

    # Decode vertex indices from deduplicated keys.
    unique_verts = unique_keys // max_touch

    return np.bincount(unique_verts, minlength=n_verts).astype(np.int64)


def compute_threshold_from_ratio(ratio_pct: float, n_filtered: int) -> int:
    """Convert a percentage threshold to an absolute integer count."""
    return max(1, round(ratio_pct / 100 * n_filtered))


def apply_vertex_threshold(
    heatmap_val: np.ndarray,
    unique_touch_count: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Set below-threshold contacted vertices to -1.0 (grey marker)."""
    result = heatmap_val.copy()
    # A vertex is "contacted" when it has a positive unique-touch count.
    contacted = unique_touch_count > 0
    below_threshold = contacted & (unique_touch_count < threshold)
    result[below_threshold] = -1.0
    return result


def build_gesture_touch_indices(gesture_types: np.ndarray, gtype: str) -> np.ndarray:
    """Return touch indices where gesture_types == gtype."""
    return np.where(gesture_types == gtype)[0]
