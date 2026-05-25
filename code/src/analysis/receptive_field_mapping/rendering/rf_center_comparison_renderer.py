"""Renderer for RF center proximal-distal comparison plots."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

logger = logging.getLogger(__name__)


def render_center_marked_heatmap(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    interp_grid: np.ndarray,
    forearm_uv: np.ndarray,
    centroid_uv: np.ndarray,
    output_path: Path,
    vmax: float,
    vmin: float,
    title: str,
    figwidth: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    heatmap_space: str = "linear",
    cmap: str = "jet",
) -> None:
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

    stride_bg = max(1, len(forearm_uv) // 5000)
    ax.scatter(
        forearm_uv[::stride_bg, 0], forearm_uv[::stride_bg, 1],
        c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
    )

    display_grid = np.where(interp_grid > 0, interp_grid, np.nan)
    ax.pcolormesh(u_grid, v_grid, display_grid, cmap=cmap, norm=norm, shading='auto')

    ax.plot(
        centroid_uv[0], centroid_uv[1],
        color='violet', marker='+', markersize=10, markeredgewidth=2, zorder=7,
    )

    ax.set_xlabel('U')
    ax.set_ylabel('V')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, facecolor='black')
    logger.info("Saved center-marked heatmap: %s", output_path)
    plt.close(fig)


def render_proximal_distal_aggregate(
    session_centroids: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    uv_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    if not session_centroids:
        raise ValueError("render_proximal_distal_aggregate: session_centroids is empty")

    try:
        cmap_tab20 = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap_tab20 = matplotlib.cm.get_cmap('tab20')

    session_ids = sorted(session_centroids.keys())
    n_sessions = len(session_ids)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

    for i, session_id in enumerate(session_ids):
        color = cmap_tab20(i / max(n_sessions, 1))
        offsets = session_centroids[session_id]
        prox = offsets['stroke_proximal']
        dist = offsets['stroke_distal']

        ax.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
        ax.plot(prox[0], prox[1], 'o', ms=6, color=color, label=session_id, zorder=3)
        ax.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)

    ax.set_xlabel('ΔU from all-gesture center')
    ax.set_ylabel('ΔV from all-gesture center')
    ax.set_title('Proximal vs Distal RF Center Offsets')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')

    if uv_limits is not None:
        ax.set_xlim(*uv_limits[0])
        ax.set_ylim(*uv_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    logger.info("Saved proximal-distal aggregate: %s", output_path)
    plt.close(fig)
