"""Pure-function deviation metrics between a condition RF and a baseline RF.

All functions operate on flat metric dicts (as returned by
``compute_grid_cell_metrics()``) and 1-D RF map arrays of shape (V,) where
NaN indicates an inactive vertex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BaselineDeviationMetrics:
    """8 scalar deviation metrics comparing a condition RF to a baseline RF."""

    deviation_area_ratio: float = float("nan")
    deviation_hotspot_area_ratio: float = float("nan")
    deviation_centroid_shift_mm: float = float("nan")
    deviation_centroid_shift_normalized: float = float("nan")
    deviation_mean_iff_ratio: float = float("nan")
    deviation_peak_iff_ratio: float = float("nan")
    deviation_vertex_overlap_jaccard: float = float("nan")
    deviation_vertex_containment: float = float("nan")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero or NaN."""
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _centroid_3d_distance(metrics_a: dict, metrics_b: dict) -> float:
    """Return 3D Euclidean distance between weighted centroids in two metric dicts.

    Keys expected: ``weighted_centroid_3d_x``, ``weighted_centroid_3d_y``,
    ``weighted_centroid_3d_z``. Returns NaN if any key is missing or NaN in
    either dict.
    """
    keys = ("weighted_centroid_3d_x", "weighted_centroid_3d_y", "weighted_centroid_3d_z")
    coords_a = []
    coords_b = []
    for k in keys:
        va = metrics_a.get(k, float("nan"))
        vb = metrics_b.get(k, float("nan"))
        if math.isnan(va) or math.isnan(vb):
            return float("nan")
        coords_a.append(va)
        coords_b.append(vb)
    delta = np.array(coords_a) - np.array(coords_b)
    return float(np.sqrt(np.dot(delta, delta)))


def _vertex_set_overlap(
    rf_map_a: np.ndarray,
    rf_map_b: np.ndarray,
) -> tuple[float, float]:
    """Compute Jaccard similarity and containment between two active vertex sets.

    Active vertices are those with non-NaN values in the RF map array.

    Parameters
    ----------
    rf_map_a:
        (V,) array; NaN = inactive vertex.
    rf_map_b:
        (V,) array (same length as ``rf_map_a``); NaN = inactive vertex.

    Returns
    -------
    (jaccard, containment) where
        jaccard     = |A ∩ B| / |A ∪ B|, NaN if union is empty
        containment = |A ∩ B| / |B|,     NaN if B is empty
    """
    active_a = set(np.where(~np.isnan(rf_map_a))[0].tolist())
    active_b = set(np.where(~np.isnan(rf_map_b))[0].tolist())

    intersection = len(active_a & active_b)
    union = len(active_a | active_b)
    n_b = len(active_b)

    jaccard = float(intersection) / float(union) if union > 0 else float("nan")
    containment = float(intersection) / float(n_b) if n_b > 0 else float("nan")

    return jaccard, containment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_baseline_deviation(
    condition_metrics: dict,
    baseline_metrics: dict,
    condition_rf_map: np.ndarray,
    baseline_rf_map: np.ndarray,
) -> dict:
    """Compute 8 deviation metrics comparing a condition RF to a baseline RF.

    Parameters
    ----------
    condition_metrics:
        Flat metric dict for the condition cell (output of
        ``compute_grid_cell_metrics()``).
    baseline_metrics:
        Flat metric dict for the baseline (same schema as ``condition_metrics``).
    condition_rf_map:
        (V,) IFF array for the condition; NaN = inactive vertex.
    baseline_rf_map:
        (V,) IFF array for the baseline; same length as ``condition_rf_map``.

    Returns
    -------
    Flat dict with all 8 ``deviation_*`` keys, suitable for merging into a
    ``pd.DataFrame`` row. All entries default to NaN when inputs are
    insufficient.
    """
    area_ratio = _safe_ratio(
        condition_metrics.get("convex_hull_area_mm2", float("nan")),
        baseline_metrics.get("convex_hull_area_mm2", float("nan")),
    )

    hotspot_area_ratio = _safe_ratio(
        condition_metrics.get("hotspot_area_mm2", float("nan")),
        baseline_metrics.get("hotspot_area_mm2", float("nan")),
    )

    centroid_shift_mm = _centroid_3d_distance(condition_metrics, baseline_metrics)

    baseline_diameter = baseline_metrics.get("equivalent_diameter_mm", float("nan"))
    centroid_shift_normalized = _safe_ratio(centroid_shift_mm, baseline_diameter)

    mean_iff_ratio = _safe_ratio(
        condition_metrics.get("mean_iff", float("nan")),
        baseline_metrics.get("mean_iff", float("nan")),
    )

    peak_iff_ratio = _safe_ratio(
        condition_metrics.get("max_iff", float("nan")),
        baseline_metrics.get("max_iff", float("nan")),
    )

    jaccard, containment = _vertex_set_overlap(condition_rf_map, baseline_rf_map)

    return {
        "deviation_area_ratio": area_ratio,
        "deviation_hotspot_area_ratio": hotspot_area_ratio,
        "deviation_centroid_shift_mm": centroid_shift_mm,
        "deviation_centroid_shift_normalized": centroid_shift_normalized,
        "deviation_mean_iff_ratio": mean_iff_ratio,
        "deviation_peak_iff_ratio": peak_iff_ratio,
        "deviation_vertex_overlap_jaccard": jaccard,
        "deviation_vertex_containment": containment,
    }
