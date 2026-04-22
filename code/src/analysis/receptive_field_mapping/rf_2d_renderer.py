"""2D scatter + interpolated heatmap renderer for RF mapping output.

Produces a two-panel figure:
  - Left: scatter plot of (u, v) contact points coloured by spike_count
  - Right: interpolated heatmap via scipy.interpolate.griddata (cubic)

Visual conventions match the 3D renderer: RdYlBu_r colormap, LogNorm,
black background, white axis labels and ticks.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AXIS_LABELS = {
    "cylindrical_unwrap": ("Circumferential (mm)", "Longitudinal (mm)"),
    "tangent_plane": ("u (mm)", "v (mm)"),
}


def render_2d_heatmap(
    uv_points: np.ndarray,
    counts: np.ndarray,
    output_path: Path,
    session_id: str,
    cluster_label: str,
    projection_method: str,
    forearm_uv: Optional[np.ndarray] = None,
    interactive: bool = False,
    cluster_description: str = '',
) -> None:
    """Render a 2D scatter + interpolated heatmap and save as PNG.

    Parameters
    ----------
    uv_points:
        (N, 2) projected contact points in mm.
    counts:
        (N,) spike count per contact point.
    output_path:
        Destination PNG path.
    session_id:
        Used in the figure title.
    cluster_label:
        Used in the figure title.
    projection_method:
        Key string from PROJECTION_METHODS — used for axis labels.
    forearm_uv:
        Optional (M, 2) projected forearm vertices for spatial context background.
    interactive:
        If True, display the figure window in addition to saving.
    """
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if len(uv_points) == 0:
        logger.warning(
            "render_2d_heatmap: no UV points for session %s, cluster %s — skipping.",
            session_id, cluster_label,
        )
        return

    xlabel, ylabel = _AXIS_LABELS.get(projection_method, ("u (mm)", "v (mm)"))

    vmin = max(1, float(counts.min()))
    vmax = float(counts.max())
    if vmin < vmax:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
    fig.subplots_adjust(wspace=0.35)
    if cluster_description:
        fig.suptitle(cluster_description, color='#aaaaaa', fontsize=9, y=1.02, style='italic')

    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    u = uv_points[:, 0]
    v = uv_points[:, 1]

    # --- Panel 1: scatter ---
    ax_scatter = axes[0]
    if forearm_uv is not None and len(forearm_uv) > 0:
        stride = max(1, len(forearm_uv) // 5000)
        ax_scatter.scatter(
            forearm_uv[::stride, 0], forearm_uv[::stride, 1],
            c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
        )
    sc = ax_scatter.scatter(
        u, v, c=counts, cmap='RdYlBu_r', s=30, alpha=0.9, norm=norm,
        linewidths=0, zorder=3,
    )
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_title(
        f'Scatter — {session_id} | cluster {cluster_label}',
        color='white', fontsize=10,
    )
    cbar1 = plt.colorbar(sc, ax=ax_scatter, label='Spike count', shrink=0.8)
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.ax.yaxis.label.set_color('white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    # --- Panel 2: interpolated heatmap ---
    ax_hm = axes[1]

    if len(uv_points) >= 4:
        from scipy.interpolate import griddata

        margin_u = (u.max() - u.min()) * 0.05 or 1.0
        margin_v = (v.max() - v.min()) * 0.05 or 1.0
        grid_u, grid_v = np.mgrid[
            u.min() - margin_u : u.max() + margin_u : 100j,
            v.min() - margin_v : v.max() + margin_v : 100j,
        ]
        try:
            grid_z = griddata(
                uv_points, counts.astype(float),
                (grid_u, grid_v), method='cubic',
            )
            grid_z = np.clip(grid_z, vmin if vmin else None, None)

            im = ax_hm.pcolormesh(
                grid_u, grid_v, grid_z,
                cmap='RdYlBu_r', norm=norm, shading='auto',
            )
            cbar2 = plt.colorbar(im, ax=ax_hm, label='Spike count', shrink=0.8)
            cbar2.ax.yaxis.set_tick_params(color='white')
            cbar2.ax.yaxis.label.set_color('white')
            plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
        except Exception:
            logger.warning("griddata interpolation failed, showing scatter only.", exc_info=True)
            ax_hm.scatter(u, v, c=counts, cmap='RdYlBu_r', s=30, norm=norm, linewidths=0)
    else:
        # Fewer than 4 points — scatter only
        ax_hm.scatter(u, v, c=counts, cmap='RdYlBu_r', s=50, norm=norm, linewidths=0)
        ax_hm.text(
            0.5, 0.97, 'Too few points for interpolation',
            color='white', ha='center', va='top', fontsize=8,
            transform=ax_hm.transAxes,
        )

    ax_hm.set_xlabel(xlabel)
    ax_hm.set_ylabel(ylabel)
    ax_hm.set_title(
        f'Heatmap ({projection_method}) — {session_id} | cluster {cluster_label}',
        color='white', fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved 2D heatmap: %s", output_path)

    if interactive:
        plt.show()

    plt.close(fig)
