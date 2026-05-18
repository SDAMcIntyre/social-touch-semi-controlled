"""2D population RF heatmap renderer — scatter + interpolated heatmap."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def render_population_rf_map(
    forearm_uv: np.ndarray,
    heatmap_val: np.ndarray,
    vmax: float,
    title: str,
    output_path: Path,
    disjoint_mask_distance_mm: float = 10.0,
) -> None:
    """Render a two-panel population RF heatmap (scatter + interpolated) and save as PNG.

    Parameters
    ----------
    forearm_uv:
        (V, 2) 2D UV coordinates for all forearm vertices.
    heatmap_val:
        (V,) per-vertex RF values. NaN = uncontacted; -1.0 = below-threshold.
    vmax:
        Colour scale upper bound (session-wide max).
    title:
        Figure title string.
    output_path:
        Destination PNG file path.
    disjoint_mask_distance_mm:
        Grid cells farther than this distance (mm) from any real UV point are
        set to NaN in the interpolated panel to prevent spurious fill-in
        between disjoint hotspots.
    """
    matplotlib.use('Agg')

    norm = Normalize(vmin=0.0, vmax=vmax)

    # Vertices with a positive RF value (above-threshold contacts).
    valid_mask = np.isfinite(heatmap_val) & (heatmap_val >= 0.0)
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

    # --- Panel 1: scatter ---
    ax_scatter = axes[0]

    stride = max(1, len(forearm_uv) // 5000)
    ax_scatter.scatter(
        forearm_uv[::stride, 0], forearm_uv[::stride, 1],
        c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
    )

    if len(valid_uv) > 0:
        sc = ax_scatter.scatter(
            valid_uv[:, 0], valid_uv[:, 1],
            c=valid_vals, cmap='jet', norm=norm,
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

    stride_bg = max(1, len(forearm_uv) // 5000)
    ax_hm.scatter(
        forearm_uv[::stride_bg, 0], forearm_uv[::stride_bg, 1],
        c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
    )

    if len(valid_uv) < 4:
        raise ValueError(
            f"render_population_rf_map: griddata requires at least 4 non-NaN "
            f"input vertices, got {len(valid_uv)} for title='{title}'. "
            f"Gesture type may have too few contacts — check upstream data."
        )

    u = valid_uv[:, 0]
    v = valid_uv[:, 1]
    margin_u = (u.max() - u.min()) * 0.05 or 1.0
    margin_v = (v.max() - v.min()) * 0.05 or 1.0
    grid_u, grid_v = np.mgrid[
        u.min() - margin_u : u.max() + margin_u : 150j,
        v.min() - margin_v : v.max() + margin_v : 150j,
    ]

    grid_z = griddata(
        valid_uv, valid_vals.astype(float),
        (grid_u, grid_v), method='cubic',
    )

    if np.all(np.isnan(grid_z)):
        raise ValueError(
            f"render_population_rf_map: griddata produced all-NaN output for "
            f"title='{title}'. Too few non-NaN input vertices ({len(valid_uv)}) "
            f"or all points are collinear — check upstream RF data."
        )

    grid_points = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    tree = cKDTree(valid_uv)
    dist, _ = tree.query(grid_points)
    dist_grid = dist.reshape(grid_u.shape)
    grid_z[dist_grid > disjoint_mask_distance_mm] = np.nan

    grid_z = np.clip(grid_z, 0.0, None)

    im = ax_hm.pcolormesh(
        grid_u, grid_v, grid_z,
        cmap='jet', norm=norm, shading='auto',
    )
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
