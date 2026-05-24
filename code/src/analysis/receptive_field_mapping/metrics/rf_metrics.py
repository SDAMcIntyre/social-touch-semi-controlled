"""Quantitative receptive-field metrics for the RF cluster pipeline.

Phase 1 — core metrics computed from a spike_counts DataFrame
(columns: x, y, z, spike_count) and optional forearm PLY vertices.

All spatial coordinates are in mm (Azure Kinect SDK native), so areas are
in mm² and lengths are in mm.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from analysis.receptive_field_mapping.surface.rf_projection import project_to_2d


# ---------------------------------------------------------------------------
# RFMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class RFMetrics:
    """All quantitative RF metrics for a single cluster."""

    # --- Size metrics ---
    convex_hull_area_mm2: float = 0.0
    threshold_area_mm2: float = 0.0
    gaussian_sigma_major_mm: float = float("nan")
    gaussian_sigma_minor_mm: float = float("nan")
    equivalent_diameter_mm: float = 0.0

    # --- Shape metrics ---
    aspect_ratio: float = float("nan")
    ellipse_major_mm: float = float("nan")
    ellipse_minor_mm: float = float("nan")
    ellipse_orientation_deg: float = float("nan")

    # --- Center metrics ---
    weighted_centroid_3d: Tuple[float, float, float] = field(
        default_factory=lambda: (float("nan"), float("nan"), float("nan"))
    )
    weighted_centroid_2d: Tuple[float, float] = field(
        default_factory=lambda: (float("nan"), float("nan"))
    )

    # --- Response metrics ---
    peak_spike_count: int = 0
    mean_spike_count: float = 0.0
    total_spikes: int = 0

    # --- Concentration metrics ---
    hotspot_area_mm2: float = 0.0
    hotspot_fraction: float = float("nan")
    sparsity_index: float = float("nan")

    # --- Boundary metrics ---
    half_peak_n_points: int = 0
    convex_hull_n_vertices: int = 0

    # --- Gaussian fit metrics ---
    gaussian_converged: bool = False
    gaussian_r_squared: float = float("nan")
    gaussian_explained_variance: float = float("nan")
    gaussian_amplitude: float = float("nan")
    gaussian_x0_mm: float = float("nan")
    gaussian_y0_mm: float = float("nan")
    gaussian_theta_deg: float = float("nan")

    # --- Metadata ---
    n_points: int = 0
    projection_method: str = "tangent_plane"
    projection_fallback: bool = False


# ---------------------------------------------------------------------------
# Sub-function: weighted centroid (3D)
# ---------------------------------------------------------------------------


def _compute_weighted_centroid(
    points_3d: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return the spike-count-weighted centroid of 3D points.

    Parameters
    ----------
    points_3d:
        (N, 3) array in mm.
    weights:
        (N,) non-negative weights (spike counts).

    Returns
    -------
    (3,) centroid array.
    """
    w_sum = weights.sum()
    if w_sum == 0:
        return points_3d.mean(axis=0)
    return (weights[:, np.newaxis] * points_3d).sum(axis=0) / w_sum


# ---------------------------------------------------------------------------
# Sub-function: convex hull area
# ---------------------------------------------------------------------------


def _compute_convex_hull_area(uv_2d: np.ndarray) -> Tuple[float, int]:
    """Compute the convex hull area of 2D projected points.

    Parameters
    ----------
    uv_2d:
        (N, 2) array.

    Returns
    -------
    (area_mm2, n_vertices) — (0.0, 0) for degenerate cases.
    """
    if uv_2d.shape[0] < 3:
        return 0.0, 0
    try:
        hull = ConvexHull(uv_2d)
        return float(hull.volume), len(hull.vertices)  # volume == area in 2D
    except QhullError:
        return 0.0, 0


# ---------------------------------------------------------------------------
# Sub-function: threshold boundary area
# ---------------------------------------------------------------------------


def _compute_threshold_boundary(
    uv_2d: np.ndarray,
    weights: np.ndarray,
    threshold_frac: float = 0.5,
) -> Tuple[float, int]:
    """Convex hull area of points whose weight exceeds threshold_frac * peak.

    Parameters
    ----------
    uv_2d:
        (N, 2) projected coordinates.
    weights:
        (N,) spike counts.
    threshold_frac:
        Fraction of peak weight used as the lower bound (default 0.5 → half-peak).

    Returns
    -------
    (area_mm2, n_points_above_threshold)
    """
    if weights.size == 0:
        return 0.0, 0
    threshold = threshold_frac * float(weights.max())
    mask = weights >= threshold
    n_above = int(mask.sum())
    if n_above == 0:
        return 0.0, 0
    area, _ = _compute_convex_hull_area(uv_2d[mask])
    return area, n_above


# ---------------------------------------------------------------------------
# Sub-function: hotspot
# ---------------------------------------------------------------------------


def _compute_hotspot(
    uv_2d: np.ndarray,
    weights: np.ndarray,
    total_area: float,
    top_frac: float = 0.2,
) -> Tuple[float, float]:
    """Convex hull area of the top top_frac points by spike count.

    Parameters
    ----------
    uv_2d:
        (N, 2) projected coordinates.
    weights:
        (N,) spike counts.
    total_area:
        Total convex hull area (mm²) — used to compute the area fraction.
    top_frac:
        Fraction of points to keep (default 0.2 → top 20%).

    Returns
    -------
    (hotspot_area_mm2, hotspot_fraction)
    """
    n = len(weights)
    if n == 0:
        return 0.0, float("nan")

    n_top = max(1, math.ceil(top_frac * n))
    top_indices = np.argsort(weights)[-n_top:]
    hotspot_area, _ = _compute_convex_hull_area(uv_2d[top_indices])

    if total_area > 0:
        fraction = hotspot_area / total_area
    else:
        fraction = float("nan")

    return hotspot_area, fraction


# ---------------------------------------------------------------------------
# Sub-function: weighted PCA ellipse fit
# ---------------------------------------------------------------------------


def _fit_ellipse(
    uv_2d: np.ndarray, weights: np.ndarray
) -> Tuple[float, float, float, float]:
    """Fit a weighted PCA ellipse to 2D projected points.

    Returns
    -------
    (major_mm, minor_mm, orientation_deg, aspect_ratio)
    All NaN on degenerate input.
    """
    nan4 = (float("nan"), float("nan"), float("nan"), float("nan"))

    if uv_2d.shape[0] < 2:
        return nan4

    w = weights.astype(float)
    w_sum = w.sum()
    if w_sum == 0:
        w = np.ones(len(w))
        w_sum = w.sum()

    centroid = (w[:, np.newaxis] * uv_2d).sum(axis=0) / w_sum
    diff = uv_2d - centroid

    # Weighted covariance matrix
    cov = (w[:, np.newaxis] * diff).T @ diff / w_sum

    # Eigendecomposition (eigh guarantees real eigenvalues for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order; largest = major axis
    major_var = float(eigenvalues[-1])
    minor_var = float(eigenvalues[0])
    major_vec = eigenvectors[:, -1]

    # 2-sigma axis lengths
    major_mm = 2.0 * math.sqrt(max(major_var, 0.0))
    minor_mm = 2.0 * math.sqrt(max(minor_var, 0.0))

    orientation_deg = float(math.degrees(math.atan2(major_vec[1], major_vec[0])))

    if minor_mm > 0:
        aspect_ratio = major_mm / minor_mm
    else:
        aspect_ratio = float("nan")

    return major_mm, minor_mm, orientation_deg, aspect_ratio


# ---------------------------------------------------------------------------
# Sub-function: 2D Gaussian fit
# ---------------------------------------------------------------------------


def _gaussian_2d_model(
    xy: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    amp: float,
    offset: float,
) -> np.ndarray:
    """Rotated 2D Gaussian evaluated on a (2, N) coordinate array."""
    x, y = xy
    xp = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    yp = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    return amp * np.exp(-0.5 * (xp**2 / sigma_x**2 + yp**2 / sigma_y**2)) + offset


def _fit_2d_gaussian(
    uv_2d: np.ndarray, weights: np.ndarray
) -> Tuple[bool, float, float, float, float, float, float, float, float]:
    """Fit a 2D rotated Gaussian to weighted 2D data.

    Returns
    -------
    (converged, r_squared, explained_variance, amplitude,
     x0_mm, y0_mm, theta_deg, sigma_major_mm, sigma_minor_mm)

    On failure all float fields are NaN and converged=False.
    """
    nan_result = (
        False,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )

    n = uv_2d.shape[0]
    if n < 7:  # need at least as many points as free parameters
        return nan_result

    w = weights.astype(float)
    w_sum = w.sum()
    if w_sum == 0:
        return nan_result

    centroid = (w[:, np.newaxis] * uv_2d).sum(axis=0) / w_sum
    x0_init, y0_init = float(centroid[0]), float(centroid[1])

    dists = np.linalg.norm(uv_2d - centroid, axis=1)
    sigma_init = float(np.sqrt((w * dists**2).sum() / w_sum))
    if sigma_init <= 0:
        sigma_init = 1.0

    amp_init = float(w.max() - w.min())
    offset_init = float(w.min())

    p0 = [x0_init, y0_init, sigma_init, sigma_init, 0.0, amp_init, offset_init]

    xy = uv_2d.T  # shape (2, N)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                _gaussian_2d_model,
                xy,
                w,
                p0=p0,
                maxfev=5000,
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        return nan_result

    x0, y0, sigma_x, sigma_y, theta, amp, offset = popt

    # R² and explained variance
    w_pred = _gaussian_2d_model(xy, *popt)
    ss_res = float(np.sum((w - w_pred) ** 2))
    ss_tot = float(np.sum((w - w.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Explained variance score
    var_res = float(np.var(w - w_pred))
    var_tot = float(np.var(w))
    explained_variance = 1.0 - var_res / var_tot if var_tot > 0 else float("nan")

    sigma_major = float(max(abs(sigma_x), abs(sigma_y)))
    sigma_minor = float(min(abs(sigma_x), abs(sigma_y)))
    theta_deg = float(math.degrees(theta))

    return (
        True,
        r_squared,
        explained_variance,
        float(amp),
        float(x0),
        float(y0),
        theta_deg,
        sigma_major,
        sigma_minor,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def compute_rf_metrics(
    spike_counts_df,
    forearm_vertices: Optional[np.ndarray],
    projection_method: str = "tangent_plane",
    rotation_matrix: np.ndarray = None,
    slim_cache_path=None,
) -> RFMetrics:
    """Compute all RF metrics from a spike_counts DataFrame.

    Parameters
    ----------
    spike_counts_df:
        DataFrame with columns ``x``, ``y``, ``z``, ``spike_count`` (mm coords).
    forearm_vertices:
        (M, 3) PLY forearm mesh vertices in mm, or None to fall back to raw XY.
    projection_method:
        Projection method key passed to ``project_to_2d()``.

    Returns
    -------
    RFMetrics instance with all fields populated.
    """
    # --- Edge case: empty dataframe ---
    if spike_counts_df is None or len(spike_counts_df) == 0:
        return RFMetrics(
            n_points=0,
            projection_method=projection_method,
            projection_fallback=forearm_vertices is None,
        )

    points_3d = spike_counts_df[["x", "y", "z"]].to_numpy(dtype=float)
    weights = spike_counts_df["spike_count"].to_numpy(dtype=float)
    n = len(weights)

    # --- Response metrics (cheap, no projection needed) ---
    peak_spike_count = int(weights.max())
    mean_spike_count = float(weights.mean())
    total_spikes = int(weights.sum())

    sparsity_index = (
        1.0 - mean_spike_count / peak_spike_count if peak_spike_count > 0 else float("nan")
    )

    # --- Weighted 3D centroid ---
    centroid_3d = _compute_weighted_centroid(points_3d, weights)
    weighted_centroid_3d = (
        float(centroid_3d[0]),
        float(centroid_3d[1]),
        float(centroid_3d[2]),
    )

    # --- 2D projection ---
    projection_fallback = False
    if forearm_vertices is None:
        projection_fallback = True
        uv_2d = points_3d[:, :2].copy()
    else:
        uv_2d = project_to_2d(
            points_3d,
            forearm_vertices,
            centroid_3d,
            method=projection_method,
            rotation_matrix=rotation_matrix,
            slim_cache_path=slim_cache_path,
        )

    # Weighted 2D centroid
    w_sum = weights.sum()
    if w_sum > 0:
        centroid_2d = (weights[:, np.newaxis] * uv_2d).sum(axis=0) / w_sum
    else:
        centroid_2d = uv_2d.mean(axis=0)
    weighted_centroid_2d = (float(centroid_2d[0]), float(centroid_2d[1]))

    # --- Convex hull (all points) ---
    hull_area, hull_n_vertices = _compute_convex_hull_area(uv_2d)
    equivalent_diameter = math.sqrt(4.0 * hull_area / math.pi) if hull_area > 0 else 0.0

    # --- Threshold boundary (50% of peak) ---
    threshold_area, half_peak_n_points = _compute_threshold_boundary(
        uv_2d, weights, threshold_frac=0.5
    )

    # --- Hotspot (top 20%) ---
    hotspot_area, hotspot_fraction = _compute_hotspot(
        uv_2d, weights, total_area=hull_area, top_frac=0.2
    )

    # --- Ellipse fit ---
    ellipse_major, ellipse_minor, ellipse_orientation_deg, aspect_ratio = _fit_ellipse(
        uv_2d, weights
    )

    # --- 2D Gaussian fit ---
    (
        gauss_converged,
        gauss_r2,
        gauss_ev,
        gauss_amp,
        gauss_x0,
        gauss_y0,
        gauss_theta_deg,
        gauss_sigma_major,
        gauss_sigma_minor,
    ) = _fit_2d_gaussian(uv_2d, weights)

    return RFMetrics(
        # size
        convex_hull_area_mm2=hull_area,
        threshold_area_mm2=threshold_area,
        gaussian_sigma_major_mm=gauss_sigma_major,
        gaussian_sigma_minor_mm=gauss_sigma_minor,
        equivalent_diameter_mm=equivalent_diameter,
        # shape
        aspect_ratio=aspect_ratio,
        ellipse_major_mm=ellipse_major,
        ellipse_minor_mm=ellipse_minor,
        ellipse_orientation_deg=ellipse_orientation_deg,
        # center
        weighted_centroid_3d=weighted_centroid_3d,
        weighted_centroid_2d=weighted_centroid_2d,
        # response
        peak_spike_count=peak_spike_count,
        mean_spike_count=mean_spike_count,
        total_spikes=total_spikes,
        # concentration
        hotspot_area_mm2=hotspot_area,
        hotspot_fraction=hotspot_fraction,
        sparsity_index=sparsity_index,
        # boundary
        half_peak_n_points=half_peak_n_points,
        convex_hull_n_vertices=hull_n_vertices,
        # gaussian fit
        gaussian_converged=gauss_converged,
        gaussian_r_squared=gauss_r2,
        gaussian_explained_variance=gauss_ev,
        gaussian_amplitude=gauss_amp,
        gaussian_x0_mm=gauss_x0,
        gaussian_y0_mm=gauss_y0,
        gaussian_theta_deg=gauss_theta_deg,
        # metadata
        n_points=n,
        projection_method=projection_method,
        projection_fallback=projection_fallback,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_python(val):
    """Convert numpy scalar or float to a plain Python type; NaN → None."""
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val


def metrics_to_dict(m: RFMetrics) -> dict:
    """Serialize RFMetrics to a JSON-safe dict (NaN → None, tuples → lists).

    Parameters
    ----------
    m:
        An RFMetrics instance.

    Returns
    -------
    Plain Python dict suitable for JSON serialization.
    """
    result = {}
    for f in m.__dataclass_fields__:
        val = getattr(m, f)
        if isinstance(val, tuple):
            result[f] = [_to_python(v) for v in val]
        else:
            result[f] = _to_python(val)
    return result


def metrics_to_row(
    m: RFMetrics,
    cluster_label,
    combo,
    clusterer,
) -> dict:
    """Serialize RFMetrics to a flat dict row suitable for a DataFrame.

    Tuple fields are expanded to separate columns:
    - ``weighted_centroid_3d`` → ``weighted_centroid_3d_x/y/z``
    - ``weighted_centroid_2d`` → ``weighted_centroid_2d_u/v``

    Parameters
    ----------
    m:
        RFMetrics instance.
    cluster_label:
        Cluster identifier (int or str).
    combo:
        Feature combination label.
    clusterer:
        Clusterer name/label.

    Returns
    -------
    Flat dict row.
    """
    row: dict = {
        "cluster_label": cluster_label,
        "feature_combination": combo,
        "clusterer": clusterer,
    }

    for f in m.__dataclass_fields__:
        val = getattr(m, f)
        if f == "weighted_centroid_3d":
            row["weighted_centroid_3d_x"] = _to_python(val[0])
            row["weighted_centroid_3d_y"] = _to_python(val[1])
            row["weighted_centroid_3d_z"] = _to_python(val[2])
        elif f == "weighted_centroid_2d":
            row["weighted_centroid_2d_u"] = _to_python(val[0])
            row["weighted_centroid_2d_v"] = _to_python(val[1])
        else:
            row[f] = _to_python(val)

    return row
