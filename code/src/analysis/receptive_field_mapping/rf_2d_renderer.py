"""2D scatter + interpolated heatmap renderer for RF mapping output.

Produces a two-panel figure:
  - Left: scatter plot of (u, v) contact points coloured by spike_count
  - Right: interpolated heatmap via scipy.interpolate.griddata (cubic)

Visual conventions match the 3D renderer: RdYlBu_r colormap, LogNorm,
black background, white axis labels and ticks.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .rf_cluster_visualizer import RFRenderContext

logger = logging.getLogger(__name__)

_AXIS_LABELS = {
    "cylindrical_unwrap": ("Circumferential (mm)", "Longitudinal (mm)"),
    "tangent_plane": ("u (mm)", "v (mm)"),
}


def _draw_hull_2d(ax, points_2d: np.ndarray, color: str, label: str) -> None:
    """Draw a closed convex-hull perimeter on a 2D axes.

    Skips gracefully (with a log info) if fewer than 3 points are provided or
    if scipy raises QhullError. This is a legitimate data shape for sparse
    neurons — not a pipeline contract violation.

    Parameters
    ----------
    ax:
        Matplotlib 2D axes object.
    points_2d:
        (N, 2) array of 2D points.
    color:
        Line colour string.
    label:
        Legend label for this hull.
    """
    from scipy.spatial import ConvexHull, QhullError

    if points_2d is None or len(points_2d) < 3:
        logger.info(
            "_draw_hull_2d: too few points (%d) for hull '%s' — skipping.",
            len(points_2d) if points_2d is not None else 0,
            label,
        )
        return

    try:
        hull = ConvexHull(points_2d)
    except QhullError:
        logger.info("_draw_hull_2d: QhullError for hull '%s' — skipping.", label)
        return

    # Close the polygon by appending the first vertex
    vertices = hull.vertices
    closed = np.append(vertices, vertices[0])
    ax.plot(
        points_2d[closed, 0],
        points_2d[closed, 1],
        color=color,
        linewidth=1.2,
        linestyle='--',
        label=label,
        zorder=5,
    )


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
    display_metric: str = "spike_count",
    render_context: Optional['RFRenderContext'] = None,
    ratio_counts: Optional[np.ndarray] = None,
    neuron_contacts_uv: Optional[np.ndarray] = None,
    neuron_cluster_contacts_uv: Optional[np.ndarray] = None,
    disjoint_mask_distance_mm: float = 8.0,
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
    cluster_description:
        Short description string shown as figure suptitle.
    display_metric:
        ``"spike_count"`` (default) uses raw spike counts as the colour driver.
        ``"spike_ratio"`` uses ``ratio_counts`` as the colour driver (range [0, 1]).
        Any other value raises ``ValueError``.
    render_context:
        Optional :class:`RFRenderContext` carrying neuron-scoped touch counts
        and feature ranges. Used for metadata overlay text.
    ratio_counts:
        (N,) pre-computed ratio values (unique_touch_spike_count /
        neuron_cluster_touches). Required when ``display_metric="spike_ratio"``.
    neuron_contacts_uv:
        (N, 2) projected 2D positions of all neuron contacts (across all
        clusters). When provided, a convex hull is drawn on each panel.
    neuron_cluster_contacts_uv:
        (M, 2) projected 2D positions of neuron contacts in the current
        cluster. When provided, a second convex hull is drawn on each panel.
    disjoint_mask_distance_mm:
        Grid cells further than this distance (mm) from the nearest sample
        point are masked to NaN in the interpolated heatmap panel.
        Default 8.0 mm.
    """
    _VALID_METRICS = {"spike_count", "spike_ratio"}
    if display_metric not in _VALID_METRICS:
        raise ValueError(
            f"display_metric must be one of {_VALID_METRICS!r}, got {display_metric!r}."
        )

    if display_metric == "spike_ratio":
        if ratio_counts is None:
            raise ValueError(
                "display_metric='spike_ratio' requires ratio_counts to be provided, "
                "but got None."
            )

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

    # --- Determine colour driver and normalisation ---
    if display_metric == "spike_ratio":
        color_values = ratio_counts
        vmin = 0.0
        vmax = 1.0
        norm = None
        cbar_label = "Spike ratio"
    else:
        color_values = counts
        vmin = max(1, float(counts.min()))
        vmax = float(counts.max())
        norm = LogNorm(vmin=vmin, vmax=vmax) if vmin < vmax else None
        cbar_label = "Spike count"

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
        u, v, c=color_values, cmap='RdYlBu_r', s=30, alpha=0.9, norm=norm,
        linewidths=0, zorder=3,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
    )
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_scatter.set_title(
        f'Scatter — {session_id} | cluster {cluster_label}',
        color='white', fontsize=10,
    )
    cbar1 = plt.colorbar(sc, ax=ax_scatter, label=cbar_label, shrink=0.8)
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.ax.yaxis.label.set_color('white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    # --- Panel 2: interpolated heatmap ---
    ax_hm = axes[1]

    if len(uv_points) >= 4:
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree

        margin_u = (u.max() - u.min()) * 0.05 or 1.0
        margin_v = (v.max() - v.min()) * 0.05 or 1.0
        grid_u, grid_v = np.mgrid[
            u.min() - margin_u : u.max() + margin_u : 100j,
            v.min() - margin_v : v.max() + margin_v : 100j,
        ]
        try:
            grid_z = griddata(
                uv_points, color_values.astype(float),
                (grid_u, grid_v), method='cubic',
            )

            # --- NaN-mask cells that are too far from any sample point (Task 2.7) ---
            # This prevents spurious fill-in between disjoint hotspots.
            grid_points = np.column_stack([grid_u.ravel(), grid_v.ravel()])
            tree = cKDTree(uv_points)
            dist, _ = tree.query(grid_points)
            dist_grid = dist.reshape(grid_u.shape)
            grid_z[dist_grid > disjoint_mask_distance_mm] = np.nan

            grid_z = np.clip(grid_z, vmin if vmin else None, None)

            im = ax_hm.pcolormesh(
                grid_u, grid_v, grid_z,
                cmap='RdYlBu_r', norm=norm, shading='auto',
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
            )
            cbar2 = plt.colorbar(im, ax=ax_hm, label=cbar_label, shrink=0.8)
            cbar2.ax.yaxis.set_tick_params(color='white')
            cbar2.ax.yaxis.label.set_color('white')
            plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
        except Exception:
            logger.warning("griddata interpolation failed, showing scatter only.", exc_info=True)
            ax_hm.scatter(
                u, v, c=color_values, cmap='RdYlBu_r', s=30, norm=norm,
                linewidths=0,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
            )
    else:
        # Fewer than 4 points — scatter only
        ax_hm.scatter(
            u, v, c=color_values, cmap='RdYlBu_r', s=50, norm=norm,
            linewidths=0,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
        )
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

    # --- Draw convex hull perimeters (Task 2.5) ---
    if neuron_contacts_uv is not None:
        _draw_hull_2d(ax_scatter, neuron_contacts_uv, '#00aaff', 'neuron (all clusters)')
        _draw_hull_2d(ax_hm, neuron_contacts_uv, '#00aaff', 'neuron (all clusters)')
    if neuron_cluster_contacts_uv is not None:
        _draw_hull_2d(ax_scatter, neuron_cluster_contacts_uv, '#ffaa00', 'neuron ∩ cluster')
        _draw_hull_2d(ax_hm, neuron_cluster_contacts_uv, '#ffaa00', 'neuron ∩ cluster')

    # --- Metadata overlay (Task 2.5) ---
    if render_context is not None:
        from .rf_cluster_visualizer import _format_metadata_overlay
        metadata_text = _format_metadata_overlay(render_context)
        for ax in axes:
            ax.text(
                0.02, 0.98,
                metadata_text,
                transform=ax.transAxes,
                va='top',
                ha='left',
                color='#aaaaaa',
                fontsize=7,
                fontfamily='monospace',
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved 2D heatmap: %s", output_path)

    if interactive:
        plt.show()

    plt.close(fig)
