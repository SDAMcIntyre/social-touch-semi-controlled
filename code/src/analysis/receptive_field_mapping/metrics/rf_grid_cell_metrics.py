"""Per-cell scalar metrics for population RF grid heatmaps.

Operates on vertex-level IFF arrays ``(V,)`` produced by ``map_population_rf_grid``.
Reuses ``compute_rf_metrics()`` for all spatial/shape/Gaussian metrics and adds
IFF-specific intensity, topographic, distributional, and boundary-shape metrics.

All spatial coordinates are in mm (Azure Kinect SDK native), so areas are in mm²
and lengths are in mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import entropy, kurtosis, skew

from .rf_metrics import RFMetrics, compute_rf_metrics
from analysis.receptive_field_mapping.surface.rf_projection import project_to_2d


# ---------------------------------------------------------------------------
# GridCellRFMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class GridCellRFMetrics:
    """IFF-specific scalar metrics for a single population RF grid cell."""

    # IFF intensity
    max_iff: float = float("nan")
    mean_iff: float = float("nan")
    median_iff: float = float("nan")
    std_iff: float = float("nan")
    iff_range: float = float("nan")
    n_active_vertices: int = 0

    # Topographic
    hypsometric_integral: float = float("nan")
    coefficient_of_variation: float = float("nan")

    # Distribution
    iff_skewness: float = float("nan")
    iff_kurtosis: float = float("nan")
    gini_coefficient: float = float("nan")
    shannon_entropy: float = float("nan")

    # Boundary shape
    perimeter_mm: float = float("nan")
    circularity: float = float("nan")
    eccentricity: float = float("nan")


# ---------------------------------------------------------------------------
# Task 1.2: IFF intensity metrics
# ---------------------------------------------------------------------------


def compute_iff_intensity_metrics(iff_values: np.ndarray) -> dict:
    """Compute basic intensity statistics from active IFF values.

    Parameters
    ----------
    iff_values:
        (N,) array of non-NaN IFF values for active vertices.

    Returns
    -------
    Dict with keys: max_iff, mean_iff, median_iff, std_iff, iff_range,
    n_active_vertices.
    """
    n = len(iff_values)
    if n == 0:
        return {
            "max_iff": float("nan"),
            "mean_iff": float("nan"),
            "median_iff": float("nan"),
            "std_iff": float("nan"),
            "iff_range": float("nan"),
            "n_active_vertices": 0,
        }

    return {
        "max_iff": float(np.max(iff_values)),
        "mean_iff": float(np.mean(iff_values)),
        "median_iff": float(np.median(iff_values)),
        "std_iff": float(np.std(iff_values)),
        "iff_range": float(np.max(iff_values) - np.min(iff_values)),
        "n_active_vertices": n,
    }


# ---------------------------------------------------------------------------
# Task 1.3: Topographic metrics
# ---------------------------------------------------------------------------


def compute_topographic_metrics(iff_values: np.ndarray) -> dict:
    """Compute hypsometric integral and coefficient of variation.

    Parameters
    ----------
    iff_values:
        (N,) array of non-NaN IFF values.

    Returns
    -------
    Dict with keys: hypsometric_integral, coefficient_of_variation.
    """
    nan_result = {
        "hypsometric_integral": float("nan"),
        "coefficient_of_variation": float("nan"),
    }

    if len(iff_values) < 2:
        return nan_result

    v_min = float(np.min(iff_values))
    v_max = float(np.max(iff_values))
    v_mean = float(np.mean(iff_values))
    v_std = float(np.std(iff_values))

    if v_max == v_min:
        hi = float("nan")
    else:
        hi = (v_mean - v_min) / (v_max - v_min)

    cv = float("nan") if v_mean == 0.0 else v_std / v_mean

    return {
        "hypsometric_integral": hi,
        "coefficient_of_variation": cv,
    }


# ---------------------------------------------------------------------------
# Task 1.4: Distribution metrics
# ---------------------------------------------------------------------------


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.

    Returns NaN if the sum is zero. Result in [0, 1]:
    0 = perfectly uniform, 1 = fully concentrated.
    """
    total = float(values.sum())
    if total == 0.0:
        return float("nan")
    n = len(values)
    sorted_v = np.sort(values)
    # Linear closed-form formula: sum of |xi - xj| / (2 * n * sum(xi))
    # Equivalent: 2 * sum((i+1) * v[i]) / (n * sum) - (n+1)/n
    rank = np.arange(1, n + 1)
    return float((2.0 * np.sum(rank * sorted_v)) / (n * total) - (n + 1) / n)


def compute_distribution_metrics(iff_values: np.ndarray) -> dict:
    """Compute skewness, kurtosis, Gini coefficient, and Shannon entropy.

    Parameters
    ----------
    iff_values:
        (N,) array of non-NaN IFF values.

    Returns
    -------
    Dict with keys: iff_skewness, iff_kurtosis, gini_coefficient, shannon_entropy.
    """
    nan_result = {
        "iff_skewness": float("nan"),
        "iff_kurtosis": float("nan"),
        "gini_coefficient": float("nan"),
        "shannon_entropy": float("nan"),
    }

    n = len(iff_values)
    if n < 2:
        if n == 1:
            return {
                "iff_skewness": float("nan"),
                "iff_kurtosis": float("nan"),
                "gini_coefficient": 0.0,
                "shannon_entropy": 0.0,
            }
        return nan_result

    total = float(iff_values.sum())

    skewness = float(skew(iff_values))
    kurt = float(kurtosis(iff_values))
    gini = _gini_coefficient(iff_values)

    if total == 0.0:
        shannon = float("nan")
    else:
        probabilities = iff_values / total
        shannon = float(entropy(probabilities, base=2))

    return {
        "iff_skewness": skewness,
        "iff_kurtosis": kurt,
        "gini_coefficient": gini,
        "shannon_entropy": shannon,
    }


# ---------------------------------------------------------------------------
# Task 1.5: Boundary shape metrics
# ---------------------------------------------------------------------------


def compute_boundary_shape_metrics(
    uv_2d: np.ndarray,
    hull_area: float,
    ellipse_major: float,
    ellipse_minor: float,
) -> dict:
    """Compute perimeter, circularity, and eccentricity from 2D projected vertices.

    Parameters
    ----------
    uv_2d:
        (N, 2) array of 2D projected active vertex positions in mm.
    hull_area:
        Convex hull area in mm² (from RFMetrics.convex_hull_area_mm2).
    ellipse_major:
        PCA ellipse major axis length in mm (from RFMetrics.ellipse_major_mm).
    ellipse_minor:
        PCA ellipse minor axis length in mm (from RFMetrics.ellipse_minor_mm).

    Returns
    -------
    Dict with keys: perimeter_mm, circularity, eccentricity.

    Notes
    -----
    scipy ConvexHull 2D convention: hull.volume = area, hull.area = perimeter.
    """
    perimeter_mm = float("nan")
    circularity = float("nan")

    if uv_2d.shape[0] >= 3:
        try:
            hull = ConvexHull(uv_2d)
            perimeter_mm = float(hull.area)
        except QhullError:
            pass

    if not math.isnan(perimeter_mm) and perimeter_mm != 0.0 and hull_area != 0.0:
        circularity = 4.0 * math.pi * hull_area / (perimeter_mm ** 2)

    if (
        not math.isnan(ellipse_major)
        and not math.isnan(ellipse_minor)
        and ellipse_major > 0.0
        and ellipse_minor <= ellipse_major
    ):
        eccentricity = float(math.sqrt(1.0 - (ellipse_minor / ellipse_major) ** 2))
    else:
        eccentricity = float("nan")

    return {
        "perimeter_mm": perimeter_mm,
        "circularity": circularity,
        "eccentricity": eccentricity,
    }


# ---------------------------------------------------------------------------
# Empty-row sentinel
# ---------------------------------------------------------------------------

_RF_METRICS_KEPT_FIELDS = [
    "convex_hull_area_mm2",
    "threshold_area_mm2",
    "gaussian_sigma_major_mm",
    "gaussian_sigma_minor_mm",
    "equivalent_diameter_mm",
    "aspect_ratio",
    "ellipse_major_mm",
    "ellipse_minor_mm",
    "ellipse_orientation_deg",
    "weighted_centroid_3d_x",
    "weighted_centroid_3d_y",
    "weighted_centroid_3d_z",
    "weighted_centroid_2d_u",
    "weighted_centroid_2d_v",
    "hotspot_area_mm2",
    "hotspot_fraction",
    "sparsity_index",
    "half_peak_n_points",
    "convex_hull_n_vertices",
    "gaussian_converged",
    "gaussian_r_squared",
    "gaussian_explained_variance",
    "gaussian_amplitude",
    "gaussian_x0_mm",
    "gaussian_y0_mm",
    "gaussian_theta_deg",
]

_RF_METRICS_INT_FIELDS = {"half_peak_n_points", "convex_hull_n_vertices"}
_RF_METRICS_BOOL_FIELDS = {"gaussian_converged"}


def _empty_rf_metrics_dict() -> dict:
    """Return a dict of all kept RFMetrics columns filled with NaN / zero defaults."""
    result = {}
    for key in _RF_METRICS_KEPT_FIELDS:
        if key in _RF_METRICS_BOOL_FIELDS:
            result[key] = False
        elif key in _RF_METRICS_INT_FIELDS:
            result[key] = 0
        else:
            result[key] = float("nan")
    return result


def _empty_grid_cell_metrics_dict() -> dict:
    """Return a dict of all GridCellRFMetrics columns filled with NaN / zero defaults."""
    result = {}
    for f_name, f_info in GridCellRFMetrics.__dataclass_fields__.items():
        default = f_info.default
        result[f_name] = default
    return result


def _empty_row_dict() -> dict:
    """Return a fully-keyed all-NaN dict for an empty (no active vertices) cell."""
    row = {}
    row.update(_empty_rf_metrics_dict())
    row.update(_empty_grid_cell_metrics_dict())
    return row


# ---------------------------------------------------------------------------
# Task 1.6: Orchestrator
# ---------------------------------------------------------------------------


def _rf_metrics_to_kept_dict(rf_metrics: RFMetrics) -> dict:
    """Extract and flatten the kept RFMetrics fields into a flat dict.

    Drops the 6 misleading response/metadata fields and expands tuple fields.
    """
    result = {}
    for f_name in rf_metrics.__dataclass_fields__:
        # Drop misleading spike-count and metadata fields
        if f_name in {
            "peak_spike_count",
            "mean_spike_count",
            "total_spikes",
            "n_points",
            "projection_method",
            "projection_fallback",
        }:
            continue

        val = getattr(rf_metrics, f_name)

        if f_name == "weighted_centroid_3d":
            result["weighted_centroid_3d_x"] = float(val[0])
            result["weighted_centroid_3d_y"] = float(val[1])
            result["weighted_centroid_3d_z"] = float(val[2])
        elif f_name == "weighted_centroid_2d":
            result["weighted_centroid_2d_u"] = float(val[0])
            result["weighted_centroid_2d_v"] = float(val[1])
        else:
            result[f_name] = val

    return result


def compute_grid_cell_metrics(
    rf_map: np.ndarray,
    forearm_vertices: np.ndarray,
    projection_method: str = "tangent_plane",
    rotation_matrix: np.ndarray = None,
    slim_cache_path=None,
) -> dict:
    """Reduce a single grid cell RF heatmap to a flat dict of scalar metrics.

    Parameters
    ----------
    rf_map:
        (V,) array of IFF values for each forearm vertex. NaN indicates no
        data for that vertex (inactive).
    forearm_vertices:
        (V, 3) forearm mesh vertices in mm, matching the ``rf_map`` length.
    projection_method:
        Projection method key passed through to ``compute_rf_metrics()`` and
        ``project_to_2d()``.

    Returns
    -------
    Flat dict containing all kept RFMetrics-derived columns (with tuple fields
    expanded) plus all GridCellRFMetrics fields.
    """
    active_mask = ~np.isnan(rf_map)
    active_positions = forearm_vertices[active_mask]
    active_values = rf_map[active_mask]

    if active_values.size == 0:
        return _empty_row_dict()

    # --- Spatial metrics via compute_rf_metrics bridge ---
    df = pd.DataFrame(
        {
            "x": active_positions[:, 0],
            "y": active_positions[:, 1],
            "z": active_positions[:, 2],
            "spike_count": active_values,
        }
    )
    rf_metrics = compute_rf_metrics(df, forearm_vertices, projection_method, rotation_matrix=rotation_matrix, slim_cache_path=slim_cache_path)
    rf_dict = _rf_metrics_to_kept_dict(rf_metrics)

    # --- IFF-specific metrics ---
    intensity = compute_iff_intensity_metrics(active_values)
    topographic = compute_topographic_metrics(active_values)
    distributional = compute_distribution_metrics(active_values)

    # --- Boundary shape metrics: project active positions to 2D ---
    centroid = active_positions.mean(axis=0)
    uv_2d = project_to_2d(
        active_positions,
        forearm_vertices,
        centroid,
        method=projection_method,
        rotation_matrix=rotation_matrix,
        slim_cache_path=slim_cache_path,
    )
    shape = compute_boundary_shape_metrics(
        uv_2d,
        hull_area=rf_dict.get("convex_hull_area_mm2", float("nan")),
        ellipse_major=rf_dict.get("ellipse_major_mm", float("nan")),
        ellipse_minor=rf_dict.get("ellipse_minor_mm", float("nan")),
    )

    # --- Merge into single flat dict ---
    result = {}
    result.update(rf_dict)
    result.update(intensity)
    result.update(topographic)
    result.update(distributional)
    result.update(shape)
    return result
