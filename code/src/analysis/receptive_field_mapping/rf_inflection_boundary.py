"""Inflection boundary detection for population RF heatmaps.

Detects the Laplacian zero-crossing on a 2D IFF heatmap, which marks
where the surface transitions from concave (near the peak) to convex
(on the flanks) — the ring of steepest descent around the peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, gaussian_filter, laplace, map_coordinates
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# InflectionBoundary dataclass
# ---------------------------------------------------------------------------


@dataclass
class InflectionBoundary:
    """Geometric metrics for the inflection boundary of a population RF map."""

    contour_uv: np.ndarray
    """(N, 2) UV coordinates of the boundary contour."""

    area_uv: float
    """Polygon area in UV space (shoelace formula)."""

    perimeter_uv: float
    """Polygon perimeter in UV space."""

    circularity: float
    """Shape compactness: 4π·area / perimeter². 1.0 = perfect circle."""

    centroid_uv: tuple[float, float]
    """Polygon centroid in UV space."""

    pca_major_uv: float
    """PCA major axis length (2σ) in UV space."""

    pca_minor_uv: float
    """PCA minor axis length (2σ) in UV space."""

    pca_orientation_deg: float
    """Major axis orientation in degrees."""

    mean_iff_on_contour: float
    """Mean IFF value sampled along the contour path."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_masked_laplacian(
    grid_z: np.ndarray,
    gaussian_sigma: float,
) -> np.ndarray:
    """NaN-aware Laplacian of a 2D grid.

    Replaces NaN cells with 0 for smoothing, uses a binary weight mask to
    normalize near NaN edges, then applies the Laplacian. Re-masks NaN regions
    plus a 2-pixel border around them to prevent finite-difference artifacts.

    Returns an array of the same shape with NaN where masked.
    """
    nan_mask = np.isnan(grid_z)

    values = np.where(nan_mask, 0.0, grid_z)
    weight = (~nan_mask).astype(float)

    smoothed_values = gaussian_filter(values, sigma=gaussian_sigma)
    smoothed_weight = gaussian_filter(weight, sigma=gaussian_sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(smoothed_weight > 0, smoothed_values / smoothed_weight, np.nan)

    laplacian = laplace(np.where(np.isnan(normalized), 0.0, normalized))

    dilation_kernel = np.ones((5, 5), dtype=bool)
    expanded_nan_mask = binary_dilation(nan_mask, structure=dilation_kernel)

    laplacian = np.where(expanded_nan_mask, np.nan, laplacian)
    return laplacian


def _find_peak_location(grid_z: np.ndarray) -> tuple[int, int] | None:
    """Return (row, col) of the global maximum ignoring NaN.

    Returns None if grid_z is all NaN.
    """
    if np.all(np.isnan(grid_z)):
        return None
    flat_idx = np.nanargmax(grid_z)
    row, col = np.unravel_index(flat_idx, grid_z.shape)
    return int(row), int(col)


def _is_contour_closed(contour_rc: np.ndarray, tolerance_px: float = 2.0) -> bool:
    """Return True if the first and last contour points are within tolerance_px."""
    dist = np.linalg.norm(contour_rc[0] - contour_rc[-1])
    return bool(dist <= tolerance_px)


def _contour_area_pixels(contour_rc: np.ndarray) -> float:
    """Shoelace area in pixel coordinates (row, col)."""
    r = contour_rc[:, 0]
    c = contour_rc[:, 1]
    return abs(float(np.dot(r, np.roll(c, 1)) - np.dot(c, np.roll(r, 1)))) * 0.5


def _select_enclosing_contour(
    contours: list[np.ndarray],
    peak_rc: tuple[int, int],
) -> np.ndarray | None:
    """Return the smallest closed contour (by shoelace area) enclosing the peak.

    Returns None if no contour is both closed and encloses the peak.
    """
    peak_point = np.array([peak_rc[0], peak_rc[1]], dtype=float)

    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        if len(contour) < 4:
            continue
        if not _is_contour_closed(contour):
            continue
        path = Path(contour)
        if path.contains_point(peak_point):
            area = _contour_area_pixels(contour)
            candidates.append((area, contour))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _contour_pixels_to_uv(
    contour_rc: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
) -> np.ndarray:
    """Convert find_contours (row, col) coordinates to UV space.

    grid_u and grid_v are both (R, C) arrays where axis 0 is the U dimension
    and axis 1 is the V dimension (produced by np.mgrid[u_min:u_max:150j, ...]).
    Row index maps to U, col index maps to V.
    """
    n_rows, n_cols = grid_u.shape
    rows = contour_rc[:, 0]
    cols = contour_rc[:, 1]

    u_min = float(grid_u[0, 0])
    u_max = float(grid_u[-1, 0])
    v_min = float(grid_v[0, 0])
    v_max = float(grid_v[0, -1])

    u_coords = u_min + rows * (u_max - u_min) / (n_rows - 1)
    v_coords = v_min + cols * (v_max - v_min) / (n_cols - 1)

    return np.column_stack([u_coords, v_coords])


# ---------------------------------------------------------------------------
# Polygon metrics
# ---------------------------------------------------------------------------


def _compute_polygon_area(contour_uv: np.ndarray) -> float:
    """Shoelace formula polygon area in UV space."""
    u = contour_uv[:, 0]
    v = contour_uv[:, 1]
    return abs(float(np.dot(u, np.roll(v, 1)) - np.dot(v, np.roll(u, 1)))) * 0.5


def _compute_polygon_perimeter(contour_uv: np.ndarray) -> float:
    """Polygon perimeter: sum of Euclidean segment lengths."""
    diffs = np.diff(contour_uv, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _compute_polygon_centroid(contour_uv: np.ndarray) -> tuple[float, float]:
    """Polygon centroid via the shoelace-based area centroid formula."""
    u = contour_uv[:, 0]
    v = contour_uv[:, 1]
    u_next = np.roll(u, -1)
    v_next = np.roll(v, -1)

    cross = u * v_next - u_next * v
    signed_area = float(np.sum(cross)) * 0.5

    if abs(signed_area) < 1e-12:
        return float(u.mean()), float(v.mean())

    cu = float(np.sum((u + u_next) * cross)) / (6.0 * signed_area)
    cv = float(np.sum((v + v_next) * cross)) / (6.0 * signed_area)
    return cu, cv


def _compute_contour_pca(
    contour_uv: np.ndarray,
) -> tuple[float, float, float]:
    """Unweighted PCA of contour vertices.

    Returns (major_2sigma, minor_2sigma, orientation_deg).
    """
    centroid = contour_uv.mean(axis=0)
    diff = contour_uv - centroid
    cov = (diff.T @ diff) / len(contour_uv)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    major_var = float(eigenvalues[-1])
    minor_var = float(eigenvalues[0])
    major_vec = eigenvectors[:, -1]

    major_2sigma = 2.0 * math.sqrt(max(major_var, 0.0))
    minor_2sigma = 2.0 * math.sqrt(max(minor_var, 0.0))
    orientation_deg = float(math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))))

    return major_2sigma, minor_2sigma, orientation_deg


# ---------------------------------------------------------------------------
# Contour IFF sampling
# ---------------------------------------------------------------------------


def _sample_grid_along_contour(
    grid_z: np.ndarray,
    contour_rc: np.ndarray,
) -> float:
    """Bilinear interpolation of grid_z at contour (row, col) positions.

    Uses scipy.ndimage.map_coordinates with order=1. NaN cells in grid_z
    are replaced with the grid mean before sampling to avoid propagating NaN
    through the interpolation kernel; sampled positions over original NaN
    cells are excluded from the mean.
    """
    nan_mask = np.isnan(grid_z)
    fill_value = float(np.nanmean(grid_z)) if not np.all(nan_mask) else 0.0
    filled = np.where(nan_mask, fill_value, grid_z)

    coords = np.array([contour_rc[:, 0], contour_rc[:, 1]])
    sampled = map_coordinates(filled, coords, order=1, mode="nearest")

    return float(np.mean(sampled))


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compute_inflection_boundary(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    gaussian_sigma: float = 2.0,
) -> InflectionBoundary | None:
    """Detect the inflection boundary on a 2D IFF population heatmap.

    The inflection boundary is the Laplacian zero-crossing of the smoothed
    heatmap — the ring where the surface transitions from concave (near the
    peak) to convex (on the flanks). Gaussian smoothing is applied first to
    suppress noise; ``gaussian_sigma`` controls the trade-off between
    sensitivity and smoothness.

    Parameters
    ----------
    grid_u:
        (R, C) U-coordinate grid (axis 0 = U dimension).
    grid_v:
        (R, C) V-coordinate grid (axis 1 = V dimension).
    grid_z:
        (R, C) IFF values; NaN where no data exists.
    gaussian_sigma:
        Standard deviation for the NaN-aware Gaussian pre-smoothing step.

    Returns
    -------
    InflectionBoundary or None if no valid boundary can be extracted
    (all-NaN input, flat surface, no closed contour enclosing the peak,
    or peak within 2 pixels of the grid border).
    """
    if grid_z is None:
        return None

    if np.all(np.isnan(grid_z)):
        return None

    peak_rc = _find_peak_location(grid_z)
    if peak_rc is None:
        return None

    n_rows, n_cols = grid_z.shape
    pr, pc = peak_rc
    if pr < 2 or pr >= n_rows - 2 or pc < 2 or pc >= n_cols - 2:
        return None

    laplacian = _compute_masked_laplacian(grid_z, gaussian_sigma)

    valid_lap = laplacian[~np.isnan(laplacian)]
    if valid_lap.size == 0 or float(np.ptp(valid_lap)) < 1e-12:
        return None

    lap_for_contour = np.where(np.isnan(laplacian), np.nanmin(laplacian) - 1.0, laplacian)
    contours = find_contours(lap_for_contour, 0.0)

    if not contours:
        return None

    selected = _select_enclosing_contour(contours, peak_rc)
    if selected is None:
        return None

    contour_uv = _contour_pixels_to_uv(selected, grid_u, grid_v)

    area_uv = _compute_polygon_area(contour_uv)
    perimeter_uv = _compute_polygon_perimeter(contour_uv)

    if perimeter_uv > 0:
        circularity = 4.0 * math.pi * area_uv / (perimeter_uv ** 2)
    else:
        circularity = float("nan")

    centroid_uv = _compute_polygon_centroid(contour_uv)
    pca_major, pca_minor, pca_orientation_deg = _compute_contour_pca(contour_uv)
    mean_iff = _sample_grid_along_contour(grid_z, selected)

    return InflectionBoundary(
        contour_uv=contour_uv,
        area_uv=area_uv,
        perimeter_uv=perimeter_uv,
        circularity=circularity,
        centroid_uv=centroid_uv,
        pca_major_uv=pca_major,
        pca_minor_uv=pca_minor,
        pca_orientation_deg=pca_orientation_deg,
        mean_iff_on_contour=mean_iff,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_json_safe(val: float) -> float | None:
    """Convert float to JSON-safe value; NaN → None."""
    if isinstance(val, float) and math.isnan(val):
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def inflection_boundary_to_dict(boundary: InflectionBoundary) -> dict:
    """Serialize InflectionBoundary to a JSON-safe dict.

    contour_uv is serialized as a list of [u, v] pairs. All scalar fields are
    plain Python floats; NaN is converted to None.
    """
    return {
        "contour_uv": [[float(pt[0]), float(pt[1])] for pt in boundary.contour_uv],
        "area_uv": _to_json_safe(boundary.area_uv),
        "perimeter_uv": _to_json_safe(boundary.perimeter_uv),
        "circularity": _to_json_safe(boundary.circularity),
        "centroid_uv": [_to_json_safe(boundary.centroid_uv[0]), _to_json_safe(boundary.centroid_uv[1])],
        "pca_major_uv": _to_json_safe(boundary.pca_major_uv),
        "pca_minor_uv": _to_json_safe(boundary.pca_minor_uv),
        "pca_orientation_deg": _to_json_safe(boundary.pca_orientation_deg),
        "mean_iff_on_contour": _to_json_safe(boundary.mean_iff_on_contour),
    }
