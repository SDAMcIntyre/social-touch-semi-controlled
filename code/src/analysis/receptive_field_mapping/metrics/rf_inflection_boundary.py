"""Inflection boundary detection for population RF heatmaps.

Detects the Laplacian zero-crossing on a 2D IFF heatmap, which marks
where the surface transitions from concave (near the peak) to convex
(on the flanks) — the ring of steepest descent around the peak.
"""

from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as label_components, laplace, map_coordinates
from skimage.measure import find_contours

logger = logging.getLogger(__name__)


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

    peak_uv: tuple[float, float]
    """UV coordinates of the global response maximum (grid-cell resolution)."""

    pca_major_uv: float
    """PCA major axis length (2σ) in UV space."""

    pca_minor_uv: float
    """PCA minor axis length (2σ) in UV space."""

    pca_orientation_deg: float
    """Major axis orientation in degrees."""

    mean_iff_on_contour: float
    """Mean IFF value sampled along the contour path."""

    iff_at_centroid: float
    """IFF value sampled at the polygon centroid via bilinear interpolation."""


# ---------------------------------------------------------------------------
# Public Laplacian computation
# ---------------------------------------------------------------------------


def compute_laplacian_arrays(
    grid_z: np.ndarray,
    gaussian_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Gaussian-smoothed field and its Laplacian for a 2D heatmap.

    Returns (smoothed, laplacian) — both (R, C) arrays with NaN where
    grid_z is NaN. ``smoothed`` is the NaN-aware Gaussian normalisation;
    ``laplacian`` is the discrete Laplacian of the extrapolated field,
    re-masked to the original NaN cells.
    """
    result = _compute_masked_laplacian(grid_z, gaussian_sigma)
    return result.normalized, result.laplacian


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _LaplacianResult:
    nan_mask: np.ndarray        # original NaN mask
    normalized: np.ndarray      # NaN-normalized Gaussian result (NaN where no data)
    extrapolated: np.ndarray    # normalized with NaN cells filled by nearest neighbor
    extrapolation_mask: np.ndarray  # True where extrapolation was applied (was NaN in normalized)
    laplacian: np.ndarray       # Laplacian of extrapolated, re-masked to original NaN


def _compute_masked_laplacian(
    grid_z: np.ndarray,
    gaussian_sigma: float,
) -> _LaplacianResult:
    """NaN-aware Laplacian of a 2D grid using nearest-neighbor extrapolation.

    Normalizes via Gaussian smoothing with a weight mask, then fills remaining
    NaN cells by nearest-neighbor extrapolation before computing the Laplacian.
    Re-masks only the original NaN cells in the result.

    Returns a _LaplacianResult exposing all intermediate arrays.
    """
    nan_mask = np.isnan(grid_z)

    values = np.where(nan_mask, 0.0, grid_z)
    weight = (~nan_mask).astype(float)

    smoothed_values = gaussian_filter(values, sigma=gaussian_sigma)
    smoothed_weight = gaussian_filter(weight, sigma=gaussian_sigma)

    normalized = np.where(smoothed_weight > 0, smoothed_values / smoothed_weight, np.nan)

    extrapolation_mask = np.isnan(normalized)

    if np.all(extrapolation_mask):
        extrapolated = np.zeros_like(normalized)
    else:
        _distances, indices = distance_transform_edt(extrapolation_mask, return_indices=True)
        extrapolated = normalized.copy()
        extrapolated[extrapolation_mask] = normalized[
            indices[0][extrapolation_mask], indices[1][extrapolation_mask]
        ]

    lap = laplace(extrapolated)
    lap = np.where(nan_mask, np.nan, lap)

    return _LaplacianResult(
        nan_mask=nan_mask,
        normalized=normalized,
        extrapolated=extrapolated,
        extrapolation_mask=extrapolation_mask,
        laplacian=lap,
    )


def _find_peak_location(grid_z: np.ndarray) -> tuple[int, int] | None:
    """Return (row, col) of the global maximum ignoring NaN.

    Returns None if grid_z is all NaN.
    """
    if np.all(np.isnan(grid_z)):
        return None
    flat_idx = np.nanargmax(grid_z)
    row, col = np.unravel_index(flat_idx, grid_z.shape)
    return int(row), int(col)


def _select_peak_basin_contour(
    laplacian: np.ndarray,
    peak_rc: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the boundary of the negative-Laplacian connected region containing the peak.

    The peak of grid_z has negative Laplacian (concave down). This function
    flood-fills from the peak through connected negative-Laplacian pixels and
    extracts the boundary contour of that region — the inflection ring.

    Returns (contour, component_mask) or (None, None) if the peak is not in a
    negative-Laplacian cell or the boundary has fewer than 4 points.
    """
    valid = ~np.isnan(laplacian)
    negative_mask = np.zeros_like(laplacian, dtype=bool)
    negative_mask[valid] = laplacian[valid] < 0

    pr, pc = peak_rc
    if not negative_mask[pr, pc]:
        lap_val = laplacian[pr, pc] if valid[pr, pc] else float("nan")
        logger.warning(
            "peak_basin: peak (%d,%d) not in negative Laplacian — "
            "lap_val=%.4e, is_nan=%s",
            pr, pc, lap_val, not valid[pr, pc],
        )
        return None, None

    labeled, n_components = label_components(negative_mask)
    peak_label = labeled[pr, pc]
    if peak_label == 0:
        logger.warning("peak_basin: peak label is 0 (should not happen)")
        return None, None

    component_mask = (labeled == peak_label).astype(float)
    component_size = int(component_mask.sum())
    contours = find_contours(component_mask, 0.5)

    if not contours:
        logger.warning(
            "peak_basin: find_contours empty — component_size=%d", component_size,
        )
        return None, None

    selected = max(contours, key=len)
    if len(selected) < 4:
        logger.warning(
            "peak_basin: largest contour too small — pts=%d, component_size=%d",
            len(selected), component_size,
        )
        return None, None

    logger.info(
        "peak_basin: OK — component_size=%d, n_components=%d, contour_pts=%d",
        component_size, n_components, len(selected),
    )
    return selected, component_mask


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
    """Polygon perimeter: sum of Euclidean segment lengths including closing segment."""
    diffs = np.diff(contour_uv, axis=0)
    segment_lengths = float(np.sum(np.linalg.norm(diffs, axis=1)))
    closing = float(np.linalg.norm(contour_uv[-1] - contour_uv[0]))
    return segment_lengths + closing


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


def _sample_grid_at_uv_point(
    grid_z: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    uv_point: tuple[float, float],
) -> float:
    """Bilinear interpolation of grid_z at a single UV-space point.

    Converts the UV coordinate to grid (row, col) using the inverse of
    ``_contour_pixels_to_uv``, then delegates to ``_sample_grid_along_contour``.
    """
    n_rows, n_cols = grid_u.shape
    u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
    v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])

    row = (uv_point[0] - u_min) / (u_max - u_min) * (n_rows - 1)
    col = (uv_point[1] - v_min) / (v_max - v_min) * (n_cols - 1)

    point_rc = np.array([[row, col]])
    return _sample_grid_along_contour(grid_z, point_rc)


def _sample_grid_along_contour(
    grid_z: np.ndarray,
    contour_rc: np.ndarray,
) -> float:
    """Bilinear interpolation of grid_z at contour (row, col) positions.

    Uses scipy.ndimage.map_coordinates with order=1. NaN cells in grid_z
    are replaced with the grid mean before sampling to avoid propagating NaN
    through the interpolation kernel.
    """
    nan_mask = np.isnan(grid_z)
    fill_value = float(np.nanmean(grid_z)) if not np.all(nan_mask) else 0.0
    filled = np.where(nan_mask, fill_value, grid_z)

    coords = np.array([contour_rc[:, 0], contour_rc[:, 1]])
    sampled = map_coordinates(filled, coords, order=1, mode="nearest")

    return float(np.mean(sampled))


# ---------------------------------------------------------------------------
# Snapshot renderer
# ---------------------------------------------------------------------------


def _render_step_gaussian(
    grid_z: np.ndarray,
    normalized: np.ndarray,
    nan_mask: np.ndarray,
    peak_rc: tuple[int, int],
) -> "plt.Figure":
    """Step 1: raw input alongside NaN-aware Gaussian smoothing result."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(12, 5))

    ax_in.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax_in.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax_in.set_title("Input grid_z")

    ax_out.imshow(np.ma.masked_invalid(normalized), cmap=jet, origin="upper")
    ax_out.set_title("Normalized Gaussian")

    fig.suptitle("Step 1 — NaN-aware Gaussian smoothing", fontsize=10)
    fig.tight_layout()
    return fig


def _render_step_laplacian(
    laplacian: np.ndarray,
    extrapolated: np.ndarray,
    extrapolation_mask: np.ndarray,
) -> "plt.Figure":
    """Step 2: extrapolated field alongside Laplacian with zero-crossing."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_ext, ax_lap) = plt.subplots(1, 2, figsize=(12, 5))

    ax_ext.imshow(np.ma.masked_invalid(extrapolated), cmap=jet, origin="upper")
    ext_contours = find_contours(extrapolation_mask.astype(float), 0.5)
    for c in ext_contours:
        ax_ext.plot(c[:, 1], c[:, 0], "--", color="white", linewidth=0.8, alpha=0.7)
    ax_ext.set_title("Extrapolated field")

    vabs = float(np.nanmax(np.abs(laplacian)))
    vabs = max(vabs, 1e-12)
    ax_lap.imshow(laplacian, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="upper")
    lap_filled = np.where(np.isnan(laplacian), 0.0, laplacian)
    ax_lap.contour(lap_filled, levels=[0.0], colors=["black"], linewidths=[0.5])
    ax_lap.set_title("Laplacian")

    fig.suptitle("Step 2 — Laplacian computation", fontsize=10)
    fig.tight_layout()
    return fig


def _render_step_basin(
    laplacian: np.ndarray,
    component_mask: np.ndarray | None,
    peak_rc: tuple[int, int],
) -> "plt.Figure":
    """Step 3: negative-Laplacian flood-fill basin mask."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))

    if component_mask is not None:
        ax.imshow(component_mask, cmap="Greys_r", vmin=0, vmax=1, origin="upper")
    else:
        ax.imshow(np.zeros_like(laplacian), cmap="Greys_r", vmin=0, vmax=1, origin="upper")
        ax.text(
            0.5, 0.5, "FAILED", transform=ax.transAxes,
            ha="center", va="center", fontsize=24, color="red", fontweight="bold",
        )
    ax.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax.set_title("Basin mask")

    fig.suptitle("Step 3 — Negative-Laplacian flood fill", fontsize=10)
    fig.tight_layout()
    return fig


def _render_step_contour(
    grid_z: np.ndarray,
    contour_rc: np.ndarray,
    peak_rc: tuple[int, int],
    contour_color: str = "red",
) -> "plt.Figure":
    """Step 4: marching-squares contour overlaid on input heatmap."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax.plot(contour_rc[:, 1], contour_rc[:, 0], "-", color=contour_color, linewidth=1.5)
    ax.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax.set_title("Contour extraction")

    fig.suptitle("Step 4 — Marching-squares contour", fontsize=10)
    fig.tight_layout()
    return fig


def _render_step_uv(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    contour_uv: np.ndarray,
    centroid_uv: tuple[float, float],
    pca_major_uv: float,
    pca_minor_uv: float,
    pca_orientation_deg: float,
    area_uv: float,
    circularity: float,
    contour_color: str = "red",
) -> "plt.Figure":
    """Step 5: contour in UV space with PCA axes and metric annotations."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(grid_u, grid_v, np.ma.masked_invalid(grid_z), cmap=jet, shading="auto")

    closed = np.vstack([contour_uv, contour_uv[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color=contour_color, linewidth=1.5)

    cu, cv = centroid_uv
    ax.plot(cu, cv, "+", color=contour_color, markersize=12, markeredgewidth=2)

    angle_rad = math.radians(pca_orientation_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    major_half = pca_major_uv / 2.0
    minor_half = pca_minor_uv / 2.0
    ax.plot(
        [cu - major_half * cos_a, cu + major_half * cos_a],
        [cv - major_half * sin_a, cv + major_half * sin_a],
        "-", color="black", linewidth=1.0,
    )
    ax.plot(
        [cu + minor_half * sin_a, cu - minor_half * sin_a],
        [cv - minor_half * cos_a, cv + minor_half * cos_a],
        "--", color="black", linewidth=1.0,
    )

    ax.text(
        0.02, 0.98,
        f"area={area_uv:.4f}  circ={circularity:.3f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_aspect("equal")
    ax.set_title("UV-space contour")

    fig.suptitle("Step 5 — Pixel-to-UV conversion", fontsize=10)
    fig.tight_layout()
    return fig


def _save_inflection_snapshots(
    grid_z: np.ndarray,
    peak_rc: tuple[int, int],
    lap_result: _LaplacianResult,
    component_mask: np.ndarray | None,
    contour_rc: np.ndarray | None,
    contour_uv: np.ndarray | None,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    centroid_uv: tuple[float, float] | None,
    pca_major_uv: float | None,
    pca_minor_uv: float | None,
    pca_orientation_deg: float | None,
    area_uv: float | None,
    circularity: float | None,
    snapshot_dir: pathlib.Path,
    snapshot_label: str,
    contour_color: str = "red",
) -> None:
    """Save per-step diagnostic PNGs for the inflection boundary pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = f"inflection_{snapshot_label}" if snapshot_label else "inflection"

    steps: list[tuple[str, object]] = [
        (f"{prefix}_step1_gaussian.png",
         lambda: _render_step_gaussian(grid_z, lap_result.normalized, lap_result.nan_mask, peak_rc)),
        (f"{prefix}_step2_laplacian.png",
         lambda: _render_step_laplacian(
             lap_result.laplacian, lap_result.extrapolated, lap_result.extrapolation_mask)),
        (f"{prefix}_step3_basin.png",
         lambda: _render_step_basin(lap_result.laplacian, component_mask, peak_rc)),
    ]

    if contour_rc is not None:
        cr = contour_rc
        cc = contour_color
        steps.append((f"{prefix}_step4_contour.png",
                       lambda: _render_step_contour(grid_z, cr, peak_rc, cc)))

    if contour_uv is not None and centroid_uv is not None:
        cuv, cen = contour_uv, centroid_uv
        maj, mino, ori = pca_major_uv, pca_minor_uv, pca_orientation_deg
        a, ci = area_uv, circularity
        cc = contour_color
        steps.append((f"{prefix}_step5_uv.png",
                       lambda: _render_step_uv(
                           grid_u, grid_v, grid_z, cuv, cen, maj, mino, ori, a, ci, cc)))

    for filename, build_fig in steps:
        fig = build_fig()
        path = pathlib.Path(snapshot_dir) / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.debug("inflection_snapshot: saved %s", path)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compute_inflection_boundary(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    gaussian_sigma: float = 4.0,
    snapshot_dir: pathlib.Path | None = None,
    snapshot_label: str = "",
    contour_color: str = "red",
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
    snapshot_dir:
        Directory in which to write per-step snapshot PNGs. No snapshot is
        written when None.
    snapshot_label:
        Label string appended to snapshot filenames.

    Returns
    -------
    InflectionBoundary or None if no valid boundary can be extracted
    (all-NaN input, flat surface, no closed contour enclosing the peak,
    or peak within 2 pixels of the grid border).
    """
    if grid_z is None:
        logger.warning("inflection_boundary: grid_z is None")
        return None

    if np.all(np.isnan(grid_z)):
        logger.debug("inflection_boundary: all-NaN grid, shape=%s", grid_z.shape)
        return None

    peak_rc = _find_peak_location(grid_z)
    if peak_rc is None:
        logger.warning("inflection_boundary: _find_peak_location returned None")
        return None

    n_rows, n_cols = grid_z.shape
    pr, pc = peak_rc
    if pr < 2 or pr >= n_rows - 2 or pc < 2 or pc >= n_cols - 2:
        nan_frac = float(np.isnan(grid_z).sum()) / grid_z.size
        logger.warning(
            "inflection_boundary: peak at border — peak_rc=(%d, %d), "
            "grid=%dx%d, nan_frac=%.3f, peak_val=%.4f",
            pr, pc, n_rows, n_cols, nan_frac, float(np.nanmax(grid_z)),
        )
        return None

    result = _compute_masked_laplacian(grid_z, gaussian_sigma)

    valid_lap = result.laplacian[~np.isnan(result.laplacian)]
    if valid_lap.size == 0 or float(np.ptp(valid_lap)) < 1e-12:
        nan_frac = float(np.isnan(grid_z).sum()) / grid_z.size
        lap_nan_frac = float(np.isnan(result.laplacian).sum()) / result.laplacian.size
        logger.warning(
            "inflection_boundary: Laplacian flat or empty — "
            "valid_lap.size=%d, ptp=%.2e, grid_nan_frac=%.3f, lap_nan_frac=%.3f",
            valid_lap.size,
            float(np.ptp(valid_lap)) if valid_lap.size > 0 else 0.0,
            nan_frac, lap_nan_frac,
        )
        return None

    selected, component_mask = _select_peak_basin_contour(result.laplacian, peak_rc)
    if selected is None:
        logger.warning(
            "inflection_boundary: peak basin contour failed — peak_rc=(%d, %d)",
            peak_rc[0], peak_rc[1],
        )
        if snapshot_dir is not None:
            _save_inflection_snapshots(
                grid_z=grid_z,
                peak_rc=peak_rc,
                lap_result=result,
                component_mask=None,
                contour_rc=None,
                contour_uv=None,
                grid_u=grid_u,
                grid_v=grid_v,
                centroid_uv=None,
                pca_major_uv=None,
                pca_minor_uv=None,
                pca_orientation_deg=None,
                area_uv=None,
                circularity=None,
                snapshot_dir=snapshot_dir,
                snapshot_label=snapshot_label,
                contour_color=contour_color,
            )
        return None

    contour_uv = _contour_pixels_to_uv(selected, grid_u, grid_v)

    area_uv = _compute_polygon_area(contour_uv)
    perimeter_uv = _compute_polygon_perimeter(contour_uv)

    if perimeter_uv > 0:
        circularity = 4.0 * math.pi * area_uv / (perimeter_uv ** 2)
    else:
        circularity = float("nan")

    centroid_uv = _compute_polygon_centroid(contour_uv)
    peak_rc_arr = np.array([[peak_rc[0], peak_rc[1]]], dtype=float)
    peak_uv_arr = _contour_pixels_to_uv(peak_rc_arr, grid_u, grid_v)
    peak_uv = (float(peak_uv_arr[0, 0]), float(peak_uv_arr[0, 1]))
    pca_major, pca_minor, pca_orientation_deg = _compute_contour_pca(contour_uv)
    mean_iff = _sample_grid_along_contour(grid_z, selected)
    iff_at_centroid = _sample_grid_at_uv_point(grid_z, grid_u, grid_v, centroid_uv)

    logger.info(
        "inflection_boundary: SUCCESS — contour_pts=%d, area_uv=%.4f, "
        "circularity=%.3f, centroid_uv=(%.3f, %.3f), iff_at_centroid=%.4f",
        len(contour_uv), area_uv, circularity, centroid_uv[0], centroid_uv[1],
        iff_at_centroid,
    )

    if snapshot_dir is not None:
        _save_inflection_snapshots(
            grid_z=grid_z,
            peak_rc=peak_rc,
            lap_result=result,
            component_mask=component_mask,
            contour_rc=selected,
            contour_uv=contour_uv,
            grid_u=grid_u,
            grid_v=grid_v,
            centroid_uv=centroid_uv,
            pca_major_uv=pca_major,
            pca_minor_uv=pca_minor,
            pca_orientation_deg=pca_orientation_deg,
            area_uv=area_uv,
            circularity=circularity,
            snapshot_dir=snapshot_dir,
            snapshot_label=snapshot_label,
            contour_color=contour_color,
        )

    return InflectionBoundary(
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
        "peak_uv": [_to_json_safe(boundary.peak_uv[0]), _to_json_safe(boundary.peak_uv[1])],
        "pca_major_uv": _to_json_safe(boundary.pca_major_uv),
        "pca_minor_uv": _to_json_safe(boundary.pca_minor_uv),
        "pca_orientation_deg": _to_json_safe(boundary.pca_orientation_deg),
        "mean_iff_on_contour": _to_json_safe(boundary.mean_iff_on_contour),
        "iff_at_centroid": _to_json_safe(boundary.iff_at_centroid),
    }
