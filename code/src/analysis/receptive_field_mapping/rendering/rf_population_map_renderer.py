"""2D population RF heatmap renderer — scatter + interpolated heatmap."""

import logging
from collections import defaultdict
from pathlib import Path

import igl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import generic_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


logger = logging.getLogger(__name__)


def _fill_interior_face_holes(
    faces: np.ndarray,
    contacted_face: np.ndarray,
) -> np.ndarray:
    """Include uncontacted face clusters that are fully enclosed by contacted faces.

    Builds face-face adjacency, finds connected components of uncontacted faces,
    and reclassifies components that don't touch the mesh boundary as contacted
    (interior holes where the harmonic solution already has valid values).
    """
    n_faces = len(faces)
    if contacted_face.all():
        return contacted_face

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n_faces):
        v0, v1, v2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        edge_to_faces[(min(v0, v1), max(v0, v1))].append(fi)
        edge_to_faces[(min(v1, v2), max(v1, v2))].append(fi)
        edge_to_faces[(min(v0, v2), max(v0, v2))].append(fi)

    rows, cols = [], []
    boundary_faces = set()
    for edge, flist in edge_to_faces.items():
        if len(flist) == 1:
            boundary_faces.add(flist[0])
        elif len(flist) == 2:
            rows.extend([flist[0], flist[1]])
            cols.extend([flist[1], flist[0]])

    adj = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_faces, n_faces),
    )

    uncontacted = ~contacted_face
    uncontacted_idx = np.where(uncontacted)[0]
    if len(uncontacted_idx) == 0:
        return contacted_face

    adj_sub = adj[np.ix_(uncontacted_idx, uncontacted_idx)]
    n_components, labels = connected_components(adj_sub, directed=False)

    filled = contacted_face.copy()
    for comp in range(n_components):
        comp_local = np.where(labels == comp)[0]
        comp_global = uncontacted_idx[comp_local]
        if not boundary_faces.intersection(comp_global):
            filled[comp_global] = True

    n_filled = int(filled.sum() - contacted_face.sum())
    if n_filled > 0:
        logger.info(
            "_fill_interior_face_holes: filled %d interior faces across %d hole(s).",
            n_filled,
            sum(
                1 for comp in range(n_components)
                if not boundary_faces.intersection(
                    uncontacted_idx[np.where(labels == comp)[0]]
                )
            ),
        )
    return filled


def _interpolate_on_mesh(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    median_filter_size: int | None = None,
) -> np.ndarray:
    above_threshold_mask = np.isfinite(heatmap_val) & (heatmap_val >= 0.0)
    b = np.where(above_threshold_mask)[0]

    if len(b) < 4:
        raise ValueError(
            f"_interpolate_on_mesh: harmonic interpolation requires at least 4 "
            f"contacted vertices, got {len(b)}. Gesture type may have too few contacts."
        )

    if median_filter_size is not None and (median_filter_size < 1 or median_filter_size % 2 == 0):
        raise ValueError(
            f"_interpolate_on_mesh: median_filter_size must be a positive odd integer, "
            f"got {median_filter_size}."
        )

    bc = heatmap_val[b].reshape(-1, 1)

    filled = igl.harmonic(forearm_V, forearm_faces, b, bc, 1).ravel()

    # Build the unmasked triangulation for CubicTriInterpolator.  Passing a
    # face_mask to mtri.Triangulation triggers matplotlib's
    # TrapezoidMapTriFinder which can reject valid SLIM meshes with many
    # hole-fill faces (RuntimeError: "Triangulation is invalid").  Instead,
    # interpolate on the full mesh and NaN-out grid points that fall in faces
    # with no contacted vertex.
    contacted_face = np.any(above_threshold_mask[forearm_faces], axis=1)
    contacted_face = _fill_interior_face_holes(forearm_faces, contacted_face)

    tri = mtri.Triangulation(
        forearm_uv[:, 0], forearm_uv[:, 1], triangles=forearm_faces,
    )
    interp = mtri.CubicTriInterpolator(tri, filled, kind='min_E')

    grid_z_masked = interp(grid_u, grid_v)
    grid_z = grid_z_masked.data.copy()
    grid_z[grid_z_masked.mask] = np.nan

    trifinder = tri.get_trifinder()
    face_idx = trifinder(grid_u.ravel(), grid_v.ravel()).reshape(grid_u.shape)
    outside = (face_idx < 0) | ~contacted_face[np.clip(face_idx, 0, None)]
    grid_z[outside] = np.nan

    if np.all(np.isnan(grid_z)):
        raise ValueError(
            "_interpolate_on_mesh: interpolation produced all-NaN output. "
            "All contacted vertices may fall outside the mesh."
        )

    grid_z = np.clip(grid_z, 0.0, None)

    if median_filter_size is not None:
        nan_mask = np.isnan(grid_z)
        grid_z = generic_filter(grid_z, np.nanmedian, size=median_filter_size)
        grid_z[nan_mask] = np.nan

    return grid_z


def compute_interpolated_grid(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    median_filter_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 150x150 interpolation grid over the forearm UV extent and interpolate heatmap values.

    Returns
    -------
    (grid_u, grid_v, grid_z):
        grid_u, grid_v — meshgrid coordinate arrays (150, 150).
        grid_z — interpolated heatmap values (150, 150), NaN outside the mesh.
    """
    u_all = forearm_uv[:, 0]
    v_all = forearm_uv[:, 1]
    margin_u = (u_all.max() - u_all.min()) * 0.05 or 1.0
    margin_v = (v_all.max() - v_all.min()) * 0.05 or 1.0
    grid_u, grid_v = np.mgrid[
        u_all.min() - margin_u : u_all.max() + margin_u : 150j,
        v_all.min() - margin_v : v_all.max() + margin_v : 150j,
    ]
    grid_z = _interpolate_on_mesh(
        forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v,
        median_filter_size=median_filter_size,
    )
    return grid_u, grid_v, grid_z


def _draw_forearm_mesh_background(
    ax,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Draw a triangulated forearm mesh as a PolyCollection background on ``ax``.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    forearm_uv:
        (V, 2) UV coordinates of forearm vertices.
    forearm_faces:
        (F, 3) triangle index array into ``forearm_uv``.
    vertex_colors:
        (V, 4) float64 RGBA per-vertex colors. When provided, each face color
        is the mean of its three vertex colors. When None, all faces are
        rendered in grey (#404040).
    """
    tri_verts = forearm_uv[forearm_faces]  # shape (F, 3, 2)
    if vertex_colors is not None:
        face_colors = vertex_colors[forearm_faces].mean(axis=1)  # shape (F, 4) RGBA
    else:
        face_colors = np.tile([0x40 / 255, 0x40 / 255, 0x40 / 255, 1.0], (len(forearm_faces), 1))
    mesh_coll = PolyCollection(tri_verts, facecolors=face_colors, edgecolors="none", zorder=0)
    ax.add_collection(mesh_coll)


def _draw_boundary(
    ax,
    contour_uv: np.ndarray,
    centroid_uv: tuple[float, float],
    contour_color: str = "red",
) -> None:
    closed = np.vstack([contour_uv, contour_uv[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=contour_color, linewidth=1.5, zorder=6)
    ax.plot(
        centroid_uv[0], centroid_uv[1],
        color=contour_color, marker='+', markersize=8, zorder=7,
    )


def render_population_rf_map(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    vmax: float,
    vmin: float,
    title: str,
    output_path: Path,
    median_filter_size: int | None = None,
    precomputed_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    boundary=None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    vertex_colors: np.ndarray | None = None,
    contour_color: str = "red",
) -> None:
    """Render a two-panel population RF heatmap (scatter + interpolated) and save as PNG.

    Parameters
    ----------
    forearm_uv:
        (V, 2) 2D UV coordinates for all forearm vertices.
    forearm_faces:
        (F, 3) integer triangle indices into forearm_uv / forearm_V.
    forearm_V:
        (V, 3) 3D vertex positions used for harmonic interpolation.
    heatmap_val:
        (V,) per-vertex RF values. NaN = uncontacted; negative finite = below-threshold.
    vmax:
        Colour scale upper bound (session-wide max).
    title:
        Figure title string.
    output_path:
        Destination PNG file path.
    precomputed_grid:
        If provided, skip internal grid computation and use (grid_u, grid_v, grid_z)
        directly. Must be the output of ``compute_interpolated_grid()``.
    boundary:
        If provided, draw the boundary contour on the interpolated heatmap panel.
    vertex_colors:
        (V, 4) float64 RGBA per-vertex skin colors. Passed to
        ``_draw_forearm_mesh_background()`` for both panels. When None, panels
        render with a grey mesh background.
    """
    matplotlib.use('Agg')

    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)

    valid_mask = np.isfinite(heatmap_val) & (heatmap_val > 0.0)
    valid_uv = forearm_uv[valid_mask]
    valid_vals = heatmap_val[valid_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle(title, color='white', fontsize=9, y=1.01)

    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_aspect('equal')

    # --- Panel 1: scatter ---
    ax_scatter = axes[0]

    _draw_forearm_mesh_background(ax_scatter, forearm_uv, forearm_faces, vertex_colors)

    if len(valid_uv) > 0:
        sc = ax_scatter.scatter(
            valid_uv[:, 0], valid_uv[:, 1],
            c=valid_vals, cmap=cmap, norm=norm,
            s=20, alpha=0.9, linewidths=0, zorder=3, rasterized=True,
        )
        cbar1 = plt.colorbar(sc, ax=ax_scatter, label='Mean IFF / spike', shrink=0.8)
        cbar1.ax.yaxis.set_tick_params(color='white')
        cbar1.ax.yaxis.label.set_color('white')
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    ax_scatter.set_xlabel('U')
    ax_scatter.set_ylabel('V')
    ax_scatter.set_title('Scatter', color='white', fontsize=10)

    # --- Panel 2: interpolated heatmap ---
    ax_hm = axes[1]

    _draw_forearm_mesh_background(ax_hm, forearm_uv, forearm_faces, vertex_colors)

    if precomputed_grid is not None:
        grid_u, grid_v, grid_z = precomputed_grid
    else:
        u_all = forearm_uv[:, 0]
        v_all = forearm_uv[:, 1]
        margin_u = (u_all.max() - u_all.min()) * 0.05 or 1.0
        margin_v = (v_all.max() - v_all.min()) * 0.05 or 1.0
        grid_u, grid_v = np.mgrid[
            u_all.min() - margin_u : u_all.max() + margin_u : 150j,
            v_all.min() - margin_v : v_all.max() + margin_v : 150j,
        ]
        grid_z = _interpolate_on_mesh(
            forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v,
            median_filter_size=median_filter_size,
        )

    display_z = np.where(grid_z > 0, grid_z, np.nan)
    im = ax_hm.pcolormesh(
        grid_u, grid_v, display_z,
        cmap=cmap, norm=norm, shading='auto',
    )

    if boundary is not None:
        _draw_boundary(ax_hm, boundary.contour_uv, boundary.centroid_uv, contour_color=contour_color)
    cbar2 = plt.colorbar(im, ax=ax_hm, label='Mean IFF / spike', shrink=0.8)
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.ax.yaxis.label.set_color('white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

    ax_hm.set_xlabel('U')
    ax_hm.set_ylabel('V')
    ax_hm.set_title('Interpolated heatmap', color='white', fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved population RF map: %s", output_path)
    plt.close(fig)


def compute_standalone_figwidth(
    longest_title: str,
    fontsize: float = 9.0,
    min_width: float = 6.0,
    dpi: float = 150.0,
) -> float:
    """Return the figure width (inches) needed to display ``longest_title`` without clipping.

    Creates a temporary figure to measure the rendered text extent, then adds a
    small margin. Pass the result as ``figwidth`` to every
    ``render_population_rf_standalone_interpolated`` call in the same batch so
    all output images share the same pixel width.

    Parameters
    ----------
    longest_title:
        The longest title string that will appear across all sessions.
    fontsize:
        Font size used by ``fig.suptitle`` in the renderer (default 9 pt).
    min_width:
        Minimum figure width regardless of title length (default 6 in).
    dpi:
        DPI used for rendering (default 150).
    """
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(50, 1), dpi=dpi)
    txt = fig.text(0.5, 0.5, longest_title, fontsize=fontsize, ha='center', va='center')
    fig.canvas.draw()
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    text_width_in = bb.width / dpi
    plt.close(fig)
    return max(min_width, text_width_in + 0.4)


def render_population_rf_standalone_interpolated(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    interp_grid: np.ndarray,
    forearm_uv: np.ndarray,
    boundary_u: np.ndarray | None,
    boundary_v: np.ndarray | None,
    output_path: Path,
    vmax: float,
    vmin: float,
    title: str,
    figwidth: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    vertex_colors: np.ndarray | None = None,
    forearm_faces: np.ndarray | None = None,
    contour_color: str = "red",
) -> None:
    """Render a single-panel interpolated population RF heatmap and save as PNG.

    ``figwidth`` must be the same value for every session in a batch (compute it
    once with ``compute_standalone_figwidth`` before rendering) so that all
    output images share identical pixel dimensions.

    Parameters
    ----------
    u_grid:
        (R, C) U-coordinate meshgrid (axis 0 = U dimension).
    v_grid:
        (R, C) V-coordinate meshgrid.
    interp_grid:
        (R, C) interpolated heatmap values; NaN outside the mesh.
    forearm_uv:
        (V, 2) UV coordinates of all forearm vertices, used to draw the
        background point cloud.
    boundary_u:
        U coordinates of the boundary polyline. If not None, boundary_v must
        also be not None.
    boundary_v:
        V coordinates of the boundary polyline. Must have the same length as
        boundary_u.
    output_path:
        Destination PNG file path.
    vmax:
        Colour scale upper bound (session-wide max).
    title:
        Figure title string. Use the same title format across sessions.
    figwidth:
        Figure width in inches — must be identical across all sessions in a
        batch. Compute via ``compute_standalone_figwidth`` before rendering.
    vmin:
        Colour scale lower bound.
    xlim:
        UV U-axis limits shared across sessions.
    ylim:
        UV V-axis limits shared across sessions.
    cmap:
        Matplotlib colormap name (default ``"inferno"``).
    vertex_colors:
        (V, 4) float64 RGBA per-vertex skin colors. When provided (together with
        ``forearm_faces``), the background is a triangulated mesh colored by
        skin texture. When either is None, falls back to a grey mesh.
    forearm_faces:
        (F, 3) integer triangle indices into ``forearm_uv``. Must be provided
        alongside ``vertex_colors`` for skin-colored mesh rendering. When None,
        ``_draw_forearm_mesh_background`` is not called and a plain grey scatter
        is used as fallback (backward-compat).
    """
    if (boundary_u is None) != (boundary_v is None):
        raise ValueError(
            "render_population_rf_standalone_interpolated: boundary_u and boundary_v "
            "must both be None or both be provided."
        )

    matplotlib.use('Agg')

    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(figwidth, 6), facecolor='black')
    fig.suptitle(title, color='white', fontsize=9)

    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.set_aspect('equal')

    if forearm_faces is not None:
        _draw_forearm_mesh_background(ax, forearm_uv, forearm_faces, vertex_colors)
    else:
        stride_bg = max(1, len(forearm_uv) // 5000)
        ax.scatter(
            forearm_uv[::stride_bg, 0], forearm_uv[::stride_bg, 1],
            c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
        )

    display_grid = np.where(interp_grid > 0, interp_grid, np.nan)
    im = ax.pcolormesh(
        u_grid, v_grid, display_grid,
        cmap=cmap, norm=norm, shading='auto',
    )

    if boundary_u is not None:
        closed_u = np.append(boundary_u, boundary_u[0])
        closed_v = np.append(boundary_v, boundary_v[0])
        ax.plot(closed_u, closed_v, color=contour_color, linewidth=1.5, zorder=6)

    ax.set_xlabel('U')
    ax.set_ylabel('V')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, facecolor='black')
    logger.info("Saved standalone interpolated RF heatmap: %s", output_path)
    plt.close(fig)


def render_population_rf_colorbar(
    output_path: Path,
    vmax: float,
    vmin: float,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
) -> None:
    """Render a standalone vertical colorbar PNG for population RF heatmaps.

    Parameters
    ----------
    output_path:
        Destination PNG file path.
    vmax:
        Colour scale upper bound (session-wide max).
    vmin:
        Colour scale lower bound.
    cmap:
        Matplotlib colormap name (default ``"inferno"``).
    """
    matplotlib.use('Agg')

    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(1.2, 4), facecolor='black')
    ax.set_visible(False)

    cbar = fig.colorbar(sm, ax=ax, fraction=1.0, label='Mean IFF / spike')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='black')
    logger.info("Saved population RF colorbar: %s", output_path)
    plt.close(fig)


def compute_uv_to_mm_scale(
    forearm_uv: np.ndarray,
    forearm_V: np.ndarray,
    forearm_faces: np.ndarray,
) -> float:
    """Return the mm-per-UV-unit scale factor estimated from mesh edge lengths.

    Computes the median ratio of 3D edge length (mm) to UV edge length across
    all unique edges in ``forearm_faces``. Edges with UV length < 1e-12 are
    filtered to avoid division by zero on degenerate UV seams.

    Parameters
    ----------
    forearm_uv:
        (N, 2) UV coordinates.
    forearm_V:
        (N, 3) 3D vertex positions in mm.
    forearm_faces:
        (F, 3) triangle index array.
    """
    i0 = forearm_faces[:, 0]
    i1 = forearm_faces[:, 1]
    i2 = forearm_faces[:, 2]

    edge_pairs = np.concatenate([
        np.stack([i0, i1], axis=1),
        np.stack([i1, i2], axis=1),
        np.stack([i2, i0], axis=1),
    ], axis=0)
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)

    a, b = edge_pairs[:, 0], edge_pairs[:, 1]

    len_3d = np.linalg.norm(forearm_V[a] - forearm_V[b], axis=1)
    len_uv = np.linalg.norm(forearm_uv[a] - forearm_uv[b], axis=1)

    valid = len_uv >= 1e-12
    if not np.any(valid):
        raise ValueError(
            "compute_uv_to_mm_scale: all edges have degenerate UV length (< 1e-12). "
            "The UV parameterisation may be collapsed."
        )

    return float(np.median(len_3d[valid] / len_uv[valid]))


def compute_highest_contour_center(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    interp_grid: np.ndarray,
    n_levels: int = 6,
) -> np.ndarray | None:
    """Return the mean position of all pixels at or above the highest contour level.

    Uses the same level formula as the contour rendering (linspace from min to
    max, excluding endpoints).  Starting from the highest level, selects all
    finite grid pixels whose value meets or exceeds the threshold and returns
    their unweighted centroid.  Falls back to lower levels only when no pixels
    qualify.

    Returns (2,) UV coordinate array, or None if no valid level exists.
    """
    finite_mask = np.isfinite(interp_grid)
    finite_vals = interp_grid[finite_mask]
    if finite_vals.size == 0 or finite_vals.min() == finite_vals.max():
        return None

    levels = np.linspace(finite_vals.min(), finite_vals.max(), n_levels + 2)[1:-1]

    for level in reversed(levels):
        above = finite_mask & (interp_grid >= level)
        if not np.any(above):
            continue
        return np.array([float(u_grid[above].mean()), float(v_grid[above].mean())])

    return None


def render_population_rf_circular_crop(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    interp_grid: np.ndarray,
    forearm_uv: np.ndarray,
    forearm_V: np.ndarray,
    forearm_faces: np.ndarray,
    center_uv: np.ndarray,
    radius_mm: float,
    vmax: float,
    vmin: float,
    output_path: Path,
    vertex_colors: np.ndarray | None = None,
    heatmap_space: str = "linear",
    dpi: int = 300,
    cmap: str = "inferno",
    contour_levels: int | None = None,
    contour_color: str = "white",
    contour_alpha: float = 0.7,
    centroid_uv: np.ndarray | None = None,
    centroid_color: str = "red",
    boundary_contour_uv: np.ndarray | None = None,
    boundary_color: str = "red",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    circular_crop_margin: float = 0.0,
) -> None:
    """Render a transparent circular crop of a population RF heatmap and save as PNG.

    The circular boundary is defined in UV space using the mm-to-UV scale
    derived from ``compute_uv_to_mm_scale``. The background shows a solid
    triangulated mesh surface with per-face skin colours (or grey if
    ``vertex_colors`` is None) for forearm faces that overlap the circle.
    The heatmap is drawn fully opaque (alpha=1.0).
    No axes, titles, spines, or decorations are included.

    Parameters
    ----------
    u_grid:
        (R, C) U-coordinate meshgrid.
    v_grid:
        (R, C) V-coordinate meshgrid.
    interp_grid:
        (R, C) interpolated heatmap values; NaN outside the mesh.
    forearm_uv:
        (N, 2) UV coordinates of all forearm vertices.
    forearm_V:
        (N, 3) 3D vertex positions in mm.
    forearm_faces:
        (F, 3) triangle index array.
    center_uv:
        (2,) UV coordinate of the circle centre.
    radius_mm:
        Circle radius in mm (world-space).
    vmax:
        Colour scale upper bound.
    vmin:
        Colour scale lower bound.
    output_path:
        Destination PNG file path.
    vertex_colors:
        (N, 4) RGBA float64 skin colors for forearm vertices. If None, grey
        (#606060) is used.
    heatmap_space:
        ``"log"`` for LogNorm; any other value uses linear Normalize.
    dpi:
        Output resolution (default 300).
    cmap:
        Matplotlib colormap name (default ``"inferno"``).
    contour_levels:
        Number of evenly-spaced contour levels to overlay. None disables contours.
    contour_color:
        Contour line color (default ``"white"``).
    contour_alpha:
        Contour line alpha (default 0.7).
    centroid_uv:
        (2,) UV coordinate for a centroid ``+`` marker. None disables the marker.
    centroid_color:
        Centroid marker color (default ``"red"``).
    boundary_contour_uv:
        (M, 2) UV coordinates of the receptive-field boundary. When provided,
        the closed boundary is drawn as a line overlay (clipped to the circle).
        None disables the boundary overlay.
    boundary_color:
        Boundary line color (default ``"red"``).
    xlim:
        Fixed (xmin, xmax) axis limits in UV space. If None, computed from
        ``center_uv ± radius_uv``.
    ylim:
        Fixed (ymin, ymax) axis limits in UV space. If None, computed from
        ``center_uv ± radius_uv``.
    """
    from matplotlib.patches import Circle

    matplotlib.use('Agg')

    scale = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)
    radius_uv = radius_mm / scale

    dist = np.linalg.norm(forearm_uv - center_uv, axis=1)
    vertex_mask = dist <= radius_uv

    face_mask = np.any(vertex_mask[forearm_faces], axis=1)
    selected_faces = forearm_faces[face_mask]

    use_fixed_viewport = xlim is not None and ylim is not None

    if use_fixed_viewport:
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        if data_width <= 0 or data_height <= 0:
            raise ValueError(
                f"render_population_rf_circular_crop: xlim/ylim define a non-positive "
                f"extent — xlim={xlim}, ylim={ylim}"
            )
        fig_height = 6.0
        fig_width = fig_height * data_width / data_height
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    else:
        fig, ax = plt.subplots(1, 1)

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_aspect('equal')
    ax.axis('off')

    _draw_forearm_mesh_background(ax, forearm_uv, selected_faces, vertex_colors)

    if heatmap_space == "log":
        norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    ax.pcolormesh(u_grid, v_grid, interp_grid, cmap=cmap, norm=norm, alpha=1.0, zorder=1, shading='auto')

    if contour_levels is not None:
        finite_vals = interp_grid[np.isfinite(interp_grid)]
        if finite_vals.size > 0 and finite_vals.min() != finite_vals.max():
            levels = np.linspace(finite_vals.min(), finite_vals.max(), contour_levels + 2)[1:-1]
            ax.contour(
                u_grid, v_grid, interp_grid,
                levels=levels, colors=contour_color, alpha=contour_alpha,
                linewidths=0.8, zorder=2,
            )

    if boundary_contour_uv is not None:
        closed = np.vstack([boundary_contour_uv, boundary_contour_uv[0]])
        ax.plot(
            closed[:, 0], closed[:, 1],
            color=boundary_color, linewidth=1.5, zorder=6,
        )

    if centroid_uv is not None:
        ax.plot(
            centroid_uv[0], centroid_uv[1],
            color=centroid_color, marker='+', markersize=8, zorder=7,
        )

    clip_circle = Circle(center_uv, radius_uv, transform=ax.transData, fill=False, edgecolor='none')
    ax.add_patch(clip_circle)
    for artist in list(ax.collections) + list(ax.lines):
        artist.set_clip_path(clip_circle)

    if use_fixed_viewport:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        margin = radius_uv * circular_crop_margin
        ax.set_xlim(center_uv[0] - radius_uv - margin, center_uv[0] + radius_uv + margin)
        ax.set_ylim(center_uv[1] - radius_uv - margin, center_uv[1] + radius_uv + margin)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_fixed_viewport:
        fig.savefig(output_path, dpi=dpi, pad_inches=0, transparent=True)
    else:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', transparent=True)
    logger.info("Saved circular RF crop: %s", output_path)
    plt.close(fig)


_PANEL_ORDER = ['all', 'tap', 'stroke_proximal', 'stroke_distal']


def render_population_rf_composite(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    results: dict[str, tuple[np.ndarray, int, int]],
    vmax: float,
    vmin: float,
    session_id: str,
    panel_type: str,
    output_path: Path,
    min_overlap_pct: float,
    uv_xlim: tuple[float, float],
    uv_ylim: tuple[float, float],
    median_filter_size: int | None = None,
    precomputed_grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    boundaries: dict | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    vertex_colors: np.ndarray | None = None,
    contour_color: str = "red",
) -> None:
    """Render a multi-panel composite (one panel per gesture type) and save as PNG.

    Parameters
    ----------
    forearm_uv:
        (V, 2) 2D UV coordinates for all forearm vertices.
    forearm_faces:
        (F, 3) integer triangle indices into forearm_uv / forearm_V.
    forearm_V:
        (V, 3) 3D vertex positions used for harmonic interpolation.
    results:
        Dict keyed by gesture type string. Each value is
        ``(thresholded_heatmap, n_touches, threshold)``.
    vmax:
        Colour scale upper bound (global across all sessions).
    session_id:
        Session identifier for the figure title.
    panel_type:
        ``"scatter"`` or ``"interpolated"``.
    output_path:
        Destination PNG file path.
    min_overlap_pct:
        Overlap percentage shown in panel subtitles.
    uv_xlim:
        (min, max) U axis limits (global across all sessions).
    uv_ylim:
        (min, max) V axis limits (global across all sessions).
    precomputed_grids:
        If provided, maps gesture type to (grid_u, grid_v, grid_z) — skips
        internal grid computation for those gesture types.
    boundaries:
        If provided, maps gesture type to a boundary object (or ``None``).
        Boundaries are drawn on the interpolated panel only.
    cmap:
        Matplotlib colormap name (default ``"inferno"``).
    vertex_colors:
        (V, 4) float64 RGBA per-vertex skin colors. Passed to
        ``_draw_forearm_mesh_background()`` for each panel. When None, panels
        render with a grey mesh background.
    """
    matplotlib.use('Agg')

    ordered_gtypes = [g for g in _PANEL_ORDER if g in results]
    if not ordered_gtypes:
        raise ValueError(
            f"render_population_rf_composite: no gesture types present in results "
            f"for session '{session_id}'."
        )

    n_panels = len(ordered_gtypes)
    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), facecolor='black')
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)
    fig.subplots_adjust(wspace=0.15, right=0.92)

    type_label = 'Scatter' if panel_type == 'scatter' else 'Interpolated heatmap'
    fig.suptitle(f"{session_id} | {type_label} composite", color='white', fontsize=11, y=1.01)

    if panel_type == 'interpolated':
        _shared_grid_u, _shared_grid_v = np.mgrid[
            uv_xlim[0] : uv_xlim[1] : 150j,
            uv_ylim[0] : uv_ylim[1] : 150j,
        ]

    for pi, gtype in enumerate(ordered_gtypes):
        ax = axes[pi]
        heatmap_val, n_touches, threshold = results[gtype]

        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_aspect('equal')

        ax.set_xlim(uv_xlim)
        ax.set_ylim(uv_ylim)

        valid_mask = np.isfinite(heatmap_val) & (heatmap_val > 0.0)
        valid_uv = forearm_uv[valid_mask]
        valid_vals = heatmap_val[valid_mask]

        _draw_forearm_mesh_background(ax, forearm_uv, forearm_faces, vertex_colors)

        if panel_type == 'scatter':
            if len(valid_uv) > 0:
                ax.scatter(
                    valid_uv[:, 0], valid_uv[:, 1],
                    c=valid_vals, cmap=cmap, norm=norm,
                    s=20, alpha=0.9, linewidths=0, zorder=3, rasterized=True,
                )
        else:
            if precomputed_grids is not None and gtype in precomputed_grids:
                grid_u, grid_v, grid_z = precomputed_grids[gtype]
            else:
                grid_u, grid_v = _shared_grid_u, _shared_grid_v
                grid_z = _interpolate_on_mesh(
                    forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v,
                    median_filter_size=median_filter_size,
                )
            display_z = np.where(grid_z > 0, grid_z, np.nan)
            ax.pcolormesh(
                grid_u, grid_v, display_z,
                cmap=cmap, norm=norm, shading='auto',
            )
            if (
                boundaries is not None
                and gtype in boundaries
                and boundaries[gtype] is not None
            ):
                _draw_boundary(
                    ax,
                    boundaries[gtype].contour_uv,
                    boundaries[gtype].centroid_uv,
                    contour_color=contour_color,
                )

        subtitle = (
            f"{gtype} | {n_touches} touches | "
            f"thr={threshold} ({min_overlap_pct:.0f}%)"
        )
        ax.set_title(subtitle, color='white', fontsize=9)
        if pi == 0:
            ax.set_xlabel('U')
            ax.set_ylabel('V')
        else:
            ax.set_xlabel('U')
            ax.set_ylabel('')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label='Mean IFF / spike', shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved population RF composite: %s", output_path)
    plt.close(fig)
