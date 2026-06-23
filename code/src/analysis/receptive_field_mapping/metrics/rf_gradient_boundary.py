"""Gradient ridge boundary detection for population RF heatmaps.

Detects the gradient magnitude ridge on a 2D IFF heatmap — the ring where
the slope (first spatial derivative) is steepest around the peak.  This
complements the Laplacian zero-crossing (inflection boundary) by marking
where the firing rate changes fastest, rather than where curvature changes
sign.
"""

from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_polygon_area,
    compute_polygon_perimeter,
    compute_polygon_centroid,
    compute_contour_pca,
    contour_pixels_to_uv,
    find_peak_location,
    sample_grid_along_contour,
    sample_grid_at_uv_point,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradientBoundary dataclass
# ---------------------------------------------------------------------------


@dataclass
class GradientBoundary:
    """Geometric metrics for the gradient ridge boundary of a population RF map."""

    contour_uv: np.ndarray
    """(N, 2) UV coordinates of the gradient ridge contour."""

    area_uv: float
    """Polygon area in UV space (shoelace formula)."""

    perimeter_uv: float
    """Polygon perimeter in UV space."""

    circularity: float
    """Shape compactness: 4pi * area / perimeter^2.  1.0 = perfect circle."""

    centroid_uv: tuple[float, float]
    """Polygon centroid in UV space."""

    peak_uv: tuple[float, float]
    """UV coordinates of the global response maximum (grid-cell resolution)."""

    pca_major_uv: float
    """PCA major axis length (2 sigma) in UV space."""

    pca_minor_uv: float
    """PCA minor axis length (2 sigma) in UV space."""

    pca_orientation_deg: float
    """Major axis orientation in degrees."""

    mean_iff_on_contour: float
    """Mean IFF value sampled along the contour path."""

    iff_at_centroid: float
    """IFF value sampled at the polygon centroid via bilinear interpolation."""

    gradient_magnitude: np.ndarray
    """(R, C) gradient magnitude field used for this boundary."""


# ---------------------------------------------------------------------------
# Gradient magnitude computation
# ---------------------------------------------------------------------------


def compute_gradient_magnitude(
    smoothed: np.ndarray,
    nan_mask: np.ndarray,
) -> np.ndarray:
    """Gradient magnitude |nabla z| from a smoothed field.

    NaN cells are filled with zero before computing np.gradient, then
    re-masked to NaN in the output.

    Returns
    -------
    (R, C) gradient magnitude array, NaN where *nan_mask* is True.
    """
    filled = np.where(nan_mask, 0.0, smoothed)
    grad_u = np.gradient(filled, axis=0)
    grad_v = np.gradient(filled, axis=1)
    grad_mag = np.sqrt(grad_u ** 2 + grad_v ** 2)
    grad_mag[nan_mask] = np.nan
    return grad_mag


# ---------------------------------------------------------------------------
# Radial profiling
# ---------------------------------------------------------------------------


def _extract_ridge_via_radial_profiling(
    grad_mag: np.ndarray,
    peak_rc: tuple[int, int],
    n_angles: int = 360,
    savgol_window: int | None = 31,
    savgol_polyorder: int = 3,
) -> np.ndarray | None:
    """Extract gradient ridge contour by radial profiling from the peak.

    Casts *n_angles* rays from *peak_rc* outward and finds the pixel
    distance of maximum gradient magnitude along each ray.  Returns
    an (N, 2) array of sub-pixel (row, col) contour points, or None
    if the gradient field is too flat to define a ridge.

    Parameters
    ----------
    grad_mag:
        (R, C) gradient magnitude (NaN outside data region).
    peak_rc:
        (row, col) peak location of the response field.
    n_angles:
        Number of evenly-spaced radial rays.
    savgol_window:
        Savitzky-Golay window for smoothing the radial-distance profile
        (applied in polar-angle space with circular padding).  ``None``
        to skip smoothing.
    savgol_polyorder:
        Polynomial order for the Savitzky-Golay filter.
    """
    n_rows, n_cols = grad_mag.shape
    peak_r, peak_c = peak_rc
    max_radius = int(math.ceil(math.hypot(n_rows, n_cols)))

    grad_filled = np.where(np.isnan(grad_mag), 0.0, grad_mag)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_radii = np.zeros(n_angles)

    for i, angle in enumerate(angles):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        radii = np.arange(1, max_radius, dtype=np.float64)
        rows = peak_r + radii * cos_a
        cols = peak_c + radii * sin_a

        valid = (
            (rows >= 0) & (rows < n_rows - 1)
            & (cols >= 0) & (cols < n_cols - 1)
        )
        if not np.any(valid):
            best_radii[i] = 1.0
            continue

        radii = radii[valid]
        rows = rows[valid]
        cols = cols[valid]

        coords = np.array([rows, cols])
        sampled = map_coordinates(grad_filled, coords, order=1, mode="nearest")

        if len(sampled) == 0 or float(np.max(sampled)) < 1e-12:
            best_radii[i] = 1.0
            continue

        best_radii[i] = float(radii[int(np.argmax(sampled))])

    # ---- Optional Savitzky-Golay smoothing on the radial-distance profile ---
    if savgol_window is not None and len(best_radii) >= savgol_window:
        pad = savgol_window // 2
        padded = np.concatenate([best_radii[-pad:], best_radii, best_radii[:pad]])
        smoothed_radii = savgol_filter(padded, savgol_window, savgol_polyorder)
        best_radii = smoothed_radii[pad:-pad]

    if np.all(best_radii <= 1.0):
        logger.warning(
            "gradient_ridge: all radii <= 1.0 — gradient too flat to define a ridge"
        )
        return None

    contour_rows = peak_r + best_radii * np.cos(angles)
    contour_cols = peak_c + best_radii * np.sin(angles)
    return np.column_stack([contour_rows, contour_cols])


# ---------------------------------------------------------------------------
# Snapshot rendering
# ---------------------------------------------------------------------------


def _render_gradient_snapshot(
    grid_z: np.ndarray,
    grad_mag: np.ndarray,
    contour_rc: np.ndarray,
    peak_rc: tuple[int, int],
    contour_color: str = "green",
) -> "plt.Figure":
    """Gradient magnitude heatmap with ridge contour overlaid."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_z, ax_grad) = plt.subplots(1, 2, figsize=(12, 5))

    ax_z.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax_z.plot(contour_rc[:, 1], contour_rc[:, 0], "-", color=contour_color, linewidth=1.5)
    ax_z.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax_z.set_title("IFF field + gradient ridge")

    vmax_grad = float(np.nanmax(grad_mag))
    ax_grad.imshow(
        np.ma.masked_invalid(grad_mag), cmap="inferno",
        origin="upper", vmin=0, vmax=max(vmax_grad, 1e-12),
    )
    ax_grad.plot(contour_rc[:, 1], contour_rc[:, 0], "-", color=contour_color, linewidth=1.5)
    ax_grad.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax_grad.set_title("|∇z| gradient magnitude")

    fig.suptitle("Gradient ridge boundary", fontsize=10)
    fig.tight_layout()
    return fig


def _render_gradient_uv(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    contour_uv: np.ndarray,
    centroid_uv: tuple[float, float],
    area_uv: float,
    circularity: float,
    contour_color: str = "green",
) -> "plt.Figure":
    """Gradient ridge contour in UV space with metric annotations."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(grid_u, grid_v, np.ma.masked_invalid(grid_z), cmap=jet, shading="auto")

    closed = np.vstack([contour_uv, contour_uv[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color=contour_color, linewidth=1.5)

    cu, cv = centroid_uv
    ax.plot(cu, cv, "+", color=contour_color, markersize=12, markeredgewidth=2)

    ax.text(
        0.02, 0.98,
        f"area={area_uv:.4f}  circ={circularity:.3f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_aspect("equal")
    ax.set_title("UV-space gradient ridge contour")

    fig.tight_layout()
    return fig


def _save_gradient_snapshots(
    grid_z: np.ndarray,
    grad_mag: np.ndarray,
    contour_rc: np.ndarray,
    contour_uv: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    peak_rc: tuple[int, int],
    centroid_uv: tuple[float, float],
    area_uv: float,
    circularity: float,
    snapshot_dir: pathlib.Path,
    snapshot_label: str,
    contour_color: str = "green",
) -> None:
    """Save diagnostic PNGs for the gradient ridge boundary pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = f"gradient_{snapshot_label}" if snapshot_label else "gradient"

    fig1 = _render_gradient_snapshot(grid_z, grad_mag, contour_rc, peak_rc, contour_color)
    path1 = pathlib.Path(snapshot_dir) / f"{prefix}_ridge.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    fig2 = _render_gradient_uv(
        grid_u, grid_v, grid_z, contour_uv, centroid_uv,
        area_uv, circularity, contour_color,
    )
    path2 = pathlib.Path(snapshot_dir) / f"{prefix}_ridge_uv.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    logger.debug("gradient_snapshots: saved %s, %s", path1, path2)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compute_gradient_ridge(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    n_angles: int = 360,
    savgol_window: int | None = 31,
    snapshot_dir: pathlib.Path | None = None,
    snapshot_label: str = "",
    contour_color: str = "green",
) -> GradientBoundary | None:
    """Detect the gradient ridge boundary on a 2D IFF population heatmap.

    The gradient ridge is the ring of maximum gradient magnitude |nabla z|
    — where the slope of the smoothed field is steepest.  For bell-shaped
    responses it lies closer to the peak than the Laplacian zero-crossing
    (inflection boundary).

    The contour is extracted via radial profiling: N rays are cast from the
    peak location outward, and the distance of maximum |nabla z| along each
    ray defines the contour.

    Parameters
    ----------
    grid_u, grid_v:
        (R, C) coordinate grids (axis 0 = U, axis 1 = V).
    grid_z:
        (R, C) raw IFF values; NaN where no data exists.
    smoothed:
        (R, C) Gaussian-smoothed field (typically from ``compute_laplacian_arrays``).
    n_angles:
        Number of radial rays for contour extraction.
    savgol_window:
        Savitzky-Golay window for contour smoothing; ``None`` to skip.
    snapshot_dir:
        Directory for diagnostic PNGs; ``None`` to skip.
    snapshot_label:
        Label appended to snapshot filenames.
    contour_color:
        Color for the contour in diagnostic snapshots.

    Returns
    -------
    GradientBoundary or None if no valid boundary can be extracted.
    """
    if grid_z is None:
        logger.warning("gradient_ridge: grid_z is None")
        return None

    if np.all(np.isnan(grid_z)):
        logger.debug("gradient_ridge: all-NaN grid, shape=%s", grid_z.shape)
        return None

    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        logger.warning("gradient_ridge: find_peak_location returned None")
        return None

    n_rows, n_cols = grid_z.shape
    pr, pc = peak_rc
    if pr < 2 or pr >= n_rows - 2 or pc < 2 or pc >= n_cols - 2:
        logger.warning(
            "gradient_ridge: peak at border — peak_rc=(%d, %d), grid=%dx%d",
            pr, pc, n_rows, n_cols,
        )
        return None

    nan_mask = np.isnan(grid_z)
    grad_mag = compute_gradient_magnitude(smoothed, nan_mask)

    valid_grad = grad_mag[~np.isnan(grad_mag)]
    if valid_grad.size == 0 or float(np.max(valid_grad)) < 1e-12:
        logger.warning(
            "gradient_ridge: gradient magnitude flat — max=%.2e",
            float(np.max(valid_grad)) if valid_grad.size > 0 else 0.0,
        )
        return None

    contour_rc = _extract_ridge_via_radial_profiling(
        grad_mag, peak_rc, n_angles=n_angles, savgol_window=savgol_window,
    )
    if contour_rc is None:
        if snapshot_dir is not None:
            _save_gradient_snapshots(
                grid_z=grid_z, grad_mag=grad_mag,
                contour_rc=np.empty((0, 2)), contour_uv=np.empty((0, 2)),
                grid_u=grid_u, grid_v=grid_v, peak_rc=peak_rc,
                centroid_uv=(float("nan"), float("nan")),
                area_uv=float("nan"), circularity=float("nan"),
                snapshot_dir=snapshot_dir, snapshot_label=snapshot_label,
                contour_color=contour_color,
            )
        return None

    contour_uv = contour_pixels_to_uv(contour_rc, grid_u, grid_v)

    area_uv = compute_polygon_area(contour_uv)
    perimeter_uv = compute_polygon_perimeter(contour_uv)
    circularity = (
        4.0 * math.pi * area_uv / (perimeter_uv ** 2)
        if perimeter_uv > 0 else float("nan")
    )
    centroid_uv = compute_polygon_centroid(contour_uv)

    peak_rc_arr = np.array([[peak_rc[0], peak_rc[1]]], dtype=float)
    peak_uv_arr = contour_pixels_to_uv(peak_rc_arr, grid_u, grid_v)
    peak_uv = (float(peak_uv_arr[0, 0]), float(peak_uv_arr[0, 1]))

    pca_major, pca_minor, pca_orientation_deg = compute_contour_pca(contour_uv)
    mean_iff = sample_grid_along_contour(grid_z, contour_rc)
    iff_at_centroid = sample_grid_at_uv_point(grid_z, grid_u, grid_v, centroid_uv)

    if snapshot_dir is not None:
        _save_gradient_snapshots(
            grid_z=grid_z, grad_mag=grad_mag,
            contour_rc=contour_rc, contour_uv=contour_uv,
            grid_u=grid_u, grid_v=grid_v, peak_rc=peak_rc,
            centroid_uv=centroid_uv, area_uv=area_uv,
            circularity=circularity,
            snapshot_dir=snapshot_dir, snapshot_label=snapshot_label,
            contour_color=contour_color,
        )

    logger.info(
        "gradient_ridge: SUCCESS — contour_pts=%d, area_uv=%.4f, "
        "circularity=%.3f, centroid_uv=(%.3f, %.3f), iff_at_centroid=%.4f",
        len(contour_uv), area_uv, circularity,
        centroid_uv[0], centroid_uv[1], iff_at_centroid,
    )

    return GradientBoundary(
        contour_uv=contour_uv,
        area_uv=area_uv,
        perimeter_uv=perimeter_uv,
        circularity=circularity,
        centroid_uv=centroid_uv,
        peak_uv=peak_uv,
        pca_major_uv=pca_major,
        pca_minor_uv=pca_minor,
        pca_orientation_deg=pca_orientation_deg,
        mean_iff_on_contour=mean_iff,
        iff_at_centroid=iff_at_centroid,
        gradient_magnitude=grad_mag,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def gradient_boundary_to_dict(boundary: GradientBoundary) -> dict:
    """Serialize GradientBoundary to a JSON-safe dict.

    ``gradient_magnitude`` is excluded (too large for JSON); everything
    else follows the same format as ``inflection_boundary_to_dict``.
    """
    def _safe(val: float) -> float | None:
        if isinstance(val, float) and math.isnan(val):
            return None
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    return {
        "contour_uv": [[float(pt[0]), float(pt[1])] for pt in boundary.contour_uv],
        "area_uv": _safe(boundary.area_uv),
        "perimeter_uv": _safe(boundary.perimeter_uv),
        "circularity": _safe(boundary.circularity),
        "centroid_uv": [_safe(boundary.centroid_uv[0]), _safe(boundary.centroid_uv[1])],
        "peak_uv": [_safe(boundary.peak_uv[0]), _safe(boundary.peak_uv[1])],
        "pca_major_uv": _safe(boundary.pca_major_uv),
        "pca_minor_uv": _safe(boundary.pca_minor_uv),
        "pca_orientation_deg": _safe(boundary.pca_orientation_deg),
        "mean_iff_on_contour": _safe(boundary.mean_iff_on_contour),
        "iff_at_centroid": _safe(boundary.iff_at_centroid),
    }
