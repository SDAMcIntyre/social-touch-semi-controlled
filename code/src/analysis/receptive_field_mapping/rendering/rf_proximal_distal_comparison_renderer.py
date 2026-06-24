"""Renderer for proximal-distal RF comparison plots."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.patches
from matplotlib.lines import Line2D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    _draw_forearm_mesh_background,
)

logger = logging.getLogger(__name__)


def _fmt_p(p: float) -> str:
    """Format a p-value for figure annotation; 'n/a' when undefined."""
    return f"{p:.3f}" if np.isfinite(p) else "n/a"


def _compute_axis_significance(values: list[float]) -> dict:
    """Wilcoxon signed-rank + exact sign test for a 1D sample against zero.

    Returns NaN p-values when fewer than 5 finite observations are available
    (matching the population-strip gating) or when a test is undefined. No
    silent fallback — undefined tests stay NaN and the figure omits them.
    """
    finite = [v for v in values if np.isfinite(v)]
    n = len(finite)
    result = {'n': n, 'wilcoxon_p': float('nan'), 'sign_p': float('nan')}
    if n < 5:
        return result
    from scipy.stats import binomtest, wilcoxon
    try:
        _stat, result['wilcoxon_p'] = wilcoxon(finite)
    except ValueError:
        pass  # wilcoxon raises when all values are zero or only one unique value
    n_positive = sum(1 for v in finite if v > 0)
    n_nonzero = sum(1 for v in finite if v != 0)
    if n_nonzero > 0:
        result['sign_p'] = float(binomtest(n_positive, n_nonzero, 0.5).pvalue)
    return result


def _compute_hotelling_t2(along: list[float], across: list[float]) -> dict:
    """One-sample Hotelling's T² test that the mean 2D shift vector is the origin.

    Tests H0: E[(ΔU, ΔV)] = (0, 0) for the population of per-session shift
    vectors. Returns NaN statistics when there are too few sessions (N <= 2)
    or the covariance is singular — no silent fallback, the figure simply
    omits the 2D annotation in that case.
    """
    pts = np.array(
        [(u, v) for u, v in zip(along, across) if np.isfinite(u) and np.isfinite(v)],
        dtype=np.float64,
    )
    p = 2
    n = int(pts.shape[0]) if pts.ndim == 2 else 0
    result = {'n': n, 'T2': float('nan'), 'F': float('nan'), 'p': float('nan')}
    if n <= p:
        return result
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        logger.warning("Hotelling T²: singular covariance (N=%d) — skipping 2D test.", n)
        return result
    t2 = float(n * mean @ inv @ mean)
    f_stat = (n - p) / (p * (n - 1)) * t2
    from scipy.stats import f as f_dist
    result['T2'] = t2
    result['F'] = float(f_stat)
    result['p'] = float(f_dist.sf(f_stat, p, n - p))
    return result


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
    cmap: str = "inferno",
    peak_uv: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    forearm_faces: np.ndarray | None = None,
    contour_color: str = "red",
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

    if forearm_faces is not None:
        _draw_forearm_mesh_background(ax, forearm_uv, forearm_faces, vertex_colors=vertex_colors)
    else:
        stride_bg = max(1, len(forearm_uv) // 5000)
        ax.scatter(
            forearm_uv[::stride_bg, 0], forearm_uv[::stride_bg, 1],
            c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
        )

    display_grid = np.where(interp_grid > 0, interp_grid, np.nan)
    ax.pcolormesh(u_grid, v_grid, display_grid, cmap=cmap, norm=norm, shading='auto')

    ax.plot(
        centroid_uv[0], centroid_uv[1],
        color=contour_color, marker='+', markersize=10, markeredgewidth=2, zorder=7,
    )

    if peak_uv is not None:
        ax.plot(
            peak_uv[0], peak_uv[1],
            color='red', marker='*', markersize=10, markeredgewidth=1.5, zorder=8,
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
    mm_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    session_colors: dict[str, str] | None = None,
    session_neuron_types: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
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

    ax.set_xlabel('ΔU from all-gesture center (mm)')
    ax.set_ylabel('ΔV from all-gesture center (mm)')
    ax.set_title('Proximal vs Distal RF Center Offsets')
    ax.set_aspect('equal')
    marker_handles = [
        Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
        Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marker_handles, fontsize=7, loc='best')

    if mm_limits is not None:
        ax.set_xlim(*mm_limits[0])
        ax.set_ylim(*mm_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved proximal-distal aggregate: %s", output_path)
    plt.close(fig)

    # --- by-type variant ---
    if session_colors is not None:
        fig_bt, ax_bt = plt.subplots(figsize=(8, 7), facecolor='white')
        ax_bt.set_facecolor('white')
        ax_bt.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax_bt.axvline(0, color='lightgray', lw=0.8, zorder=0)

        for session_id in session_ids:
            color = session_colors.get(session_id, 'steelblue')
            offsets = session_centroids[session_id]
            prox = offsets['stroke_proximal']
            dist = offsets['stroke_distal']
            ax_bt.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
            ax_bt.plot(prox[0], prox[1], 'o', ms=6, color=color, zorder=3)
            ax_bt.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)
            ax_bt.text(prox[0], prox[1], f' {session_id}', fontsize=6, color=color,
                       ha='left', va='bottom', zorder=4)

        ax_bt.set_xlabel('ΔU from all-gesture center (mm)')
        ax_bt.set_ylabel('ΔV from all-gesture center (mm)')
        ax_bt.set_title('Proximal vs Distal RF Center Offsets (by type)')
        ax_bt.set_aspect('equal')

        all_legend_handles = []
        if neuron_type_legend:
            all_legend_handles.extend([
                matplotlib.patches.Patch(facecolor=c, label=nt)
                for nt, c in neuron_type_legend.items()
            ])
        all_legend_handles.extend([
            Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
            Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
        ])
        ax_bt.legend(handles=all_legend_handles, fontsize=7, loc='best')

        if mm_limits is not None:
            ax_bt.set_xlim(*mm_limits[0])
            ax_bt.set_ylim(*mm_limits[1])

        fig_bt.tight_layout()
        by_type_path = output_path.with_stem(output_path.stem + '_by_type')
        by_type_path.parent.mkdir(parents=True, exist_ok=True)
        fig_bt.savefig(str(by_type_path), dpi=120)
        fig_bt.savefig(by_type_path.with_suffix('.svg'), bbox_inches='tight')
        logger.info("Saved proximal-distal aggregate (by type): %s", by_type_path)
        plt.close(fig_bt)


def render_proximal_distal_contour_overlay(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    contour_proximal_uv: np.ndarray,
    contour_distal_uv: np.ndarray,
    centroid_proximal_uv: np.ndarray,
    centroid_distal_uv: np.ndarray,
    output_path: Path,
    contour_stroke_uv: np.ndarray | None = None,
    hotspot_proximal_uv: np.ndarray | None = None,
    hotspot_distal_uv: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    iou: float | None = None,
    area_ratio: float | None = None,
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
    if title:
        fig.suptitle(title, color='black', fontsize=9)

    ax.set_facecolor('white')
    ax.tick_params(colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.set_aspect('equal')

    _draw_forearm_mesh_background(ax, forearm_uv, forearm_faces, vertex_colors=vertex_colors)

    # Stroke contour (reference) as dashed grey
    if contour_stroke_uv is not None:
        _closed = np.vstack([contour_stroke_uv, contour_stroke_uv[0]])
        ax.plot(_closed[:, 0], _closed[:, 1], color='grey', linestyle='--', lw=1.2, zorder=4, label='stroke')

    # Proximal contour — cyan solid
    _prox_closed = np.vstack([contour_proximal_uv, contour_proximal_uv[0]])
    ax.plot(_prox_closed[:, 0], _prox_closed[:, 1], color='cyan', linestyle='-', lw=1.5, zorder=5, label='proximal')

    # Distal contour — magenta solid
    _dist_closed = np.vstack([contour_distal_uv, contour_distal_uv[0]])
    ax.plot(_dist_closed[:, 0], _dist_closed[:, 1], color='magenta', linestyle='-', lw=1.5, zorder=5, label='distal')

    # Centroids
    ax.plot(
        centroid_proximal_uv[0], centroid_proximal_uv[1],
        color='cyan', marker='+', markersize=10, markeredgewidth=2, linestyle='none', zorder=7,
    )
    ax.plot(
        centroid_distal_uv[0], centroid_distal_uv[1],
        color='magenta', marker='+', markersize=10, markeredgewidth=2, linestyle='none', zorder=7,
    )

    # Hotspots (optional)
    if hotspot_proximal_uv is not None:
        ax.plot(
            hotspot_proximal_uv[0], hotspot_proximal_uv[1],
            color='cyan', marker='*', markersize=10, markeredgewidth=1.5, linestyle='none', zorder=8,
        )
    if hotspot_distal_uv is not None:
        ax.plot(
            hotspot_distal_uv[0], hotspot_distal_uv[1],
            color='magenta', marker='*', markersize=10, markeredgewidth=1.5, linestyle='none', zorder=8,
        )

    # Annotation in top-left corner
    annotation_parts = []
    if iou is not None and np.isfinite(iou):
        annotation_parts.append(f"IoU={iou:.2f}")
    if area_ratio is not None and np.isfinite(area_ratio):
        annotation_parts.append(f"area ratio={area_ratio:.2f}")
    if annotation_parts:
        ax.text(
            0.02, 0.98, "\n".join(annotation_parts),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
        )

    ax.set_xlabel('U')
    ax.set_ylabel('V')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    logger.info("Saved contour overlay: %s", output_path)
    plt.close(fig)


def render_proximal_distal_heatmap_triptych(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z_proximal: np.ndarray,
    grid_z_distal: np.ndarray,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    output_path: Path,
    vmax: float,
    vmin: float,
    vertex_colors: np.ndarray | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    if title:
        fig.suptitle(title, color='white', fontsize=10)

    panel_titles = ["Proximal", "Distal", "Difference (P − D)"]

    # Compute difference grid
    diff = grid_z_proximal - grid_z_distal
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size > 0:
        max_abs = float(np.max(np.abs(finite_diff)))
    else:
        max_abs = 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    diff_norm = Normalize(vmin=-max_abs, vmax=max_abs)

    for panel_idx, ax in enumerate(axes):
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_aspect('equal')
        ax.set_title(panel_titles[panel_idx], color='white', fontsize=9)

        _draw_forearm_mesh_background(ax, forearm_uv, forearm_faces, vertex_colors=vertex_colors)

        if panel_idx == 0:
            display_grid = np.where(grid_z_proximal > 0, grid_z_proximal, np.nan)
            ax.pcolormesh(grid_u, grid_v, display_grid, cmap=cmap, norm=norm, shading='auto')
        elif panel_idx == 1:
            display_grid = np.where(grid_z_distal > 0, grid_z_distal, np.nan)
            ax.pcolormesh(grid_u, grid_v, display_grid, cmap=cmap, norm=norm, shading='auto')
        else:
            ax.pcolormesh(grid_u, grid_v, diff, cmap='RdBu_r', norm=diff_norm, shading='auto')

        ax.set_xlabel('U', color='white')
        ax.set_ylabel('V', color='white')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, facecolor='black')
    logger.info("Saved heatmap triptych: %s", output_path)
    plt.close(fig)


def render_proximal_distal_hotspot_aggregate(
    session_hotspots: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    mm_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    session_colors: dict[str, str] | None = None,
    session_neuron_types: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    if not session_hotspots:
        raise ValueError("render_proximal_distal_hotspot_aggregate: session_hotspots is empty")

    try:
        cmap_tab20 = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap_tab20 = matplotlib.cm.get_cmap('tab20')

    session_ids = sorted(session_hotspots.keys())
    n_sessions = len(session_ids)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

    for i, session_id in enumerate(session_ids):
        color = cmap_tab20(i / max(n_sessions, 1))
        offsets = session_hotspots[session_id]
        prox = offsets['stroke_proximal']
        dist = offsets['stroke_distal']

        ax.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
        ax.plot(prox[0], prox[1], 'o', ms=6, color=color, label=session_id, zorder=3)
        ax.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)

    ax.set_xlabel('ΔU from stroke hotspot (mm)')
    ax.set_ylabel('ΔV from stroke hotspot (mm)')
    ax.set_title('Proximal vs Distal RF Hotspot Offsets')
    ax.set_aspect('equal')

    marker_handles = [
        Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
        Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marker_handles, fontsize=7, loc='best')

    if mm_limits is not None:
        ax.set_xlim(*mm_limits[0])
        ax.set_ylim(*mm_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved proximal-distal hotspot aggregate: %s", output_path)
    plt.close(fig)

    # --- by-type variant ---
    if session_colors is not None:
        fig_bt, ax_bt = plt.subplots(figsize=(8, 7), facecolor='white')
        ax_bt.set_facecolor('white')
        ax_bt.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax_bt.axvline(0, color='lightgray', lw=0.8, zorder=0)

        for session_id in session_ids:
            color = session_colors.get(session_id, 'steelblue')
            offsets = session_hotspots[session_id]
            prox = offsets['stroke_proximal']
            dist = offsets['stroke_distal']
            ax_bt.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
            ax_bt.plot(prox[0], prox[1], 'o', ms=6, color=color, zorder=3)
            ax_bt.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)
            ax_bt.text(prox[0], prox[1], f' {session_id}', fontsize=6, color=color,
                       ha='left', va='bottom', zorder=4)

        ax_bt.set_xlabel('ΔU from stroke hotspot (mm)')
        ax_bt.set_ylabel('ΔV from stroke hotspot (mm)')
        ax_bt.set_title('Proximal vs Distal RF Hotspot Offsets (by type)')
        ax_bt.set_aspect('equal')

        all_legend_handles = []
        if neuron_type_legend:
            all_legend_handles.extend([
                matplotlib.patches.Patch(facecolor=c, label=nt)
                for nt, c in neuron_type_legend.items()
            ])
        all_legend_handles.extend([
            Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
            Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
        ])
        ax_bt.legend(handles=all_legend_handles, fontsize=7, loc='best')

        if mm_limits is not None:
            ax_bt.set_xlim(*mm_limits[0])
            ax_bt.set_ylim(*mm_limits[1])

        fig_bt.tight_layout()
        by_type_path = output_path.with_stem(output_path.stem + '_by_type')
        by_type_path.parent.mkdir(parents=True, exist_ok=True)
        fig_bt.savefig(str(by_type_path), dpi=120)
        fig_bt.savefig(by_type_path.with_suffix('.svg'), bbox_inches='tight')
        logger.info("Saved proximal-distal hotspot aggregate (by type): %s", by_type_path)
        plt.close(fig_bt)


def render_proximal_distal_contour_center_aggregate(
    session_contour_centers: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    mm_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    session_colors: dict[str, str] | None = None,
    session_neuron_types: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    if not session_contour_centers:
        raise ValueError("render_proximal_distal_contour_center_aggregate: session_contour_centers is empty")

    try:
        cmap_tab20 = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap_tab20 = matplotlib.cm.get_cmap('tab20')

    session_ids = sorted(session_contour_centers.keys())
    n_sessions = len(session_ids)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

    for i, session_id in enumerate(session_ids):
        color = cmap_tab20(i / max(n_sessions, 1))
        offsets = session_contour_centers[session_id]
        prox = offsets['stroke_proximal']
        dist = offsets['stroke_distal']

        ax.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
        ax.plot(prox[0], prox[1], 'o', ms=6, color=color, label=session_id, zorder=3)
        ax.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)

    ax.set_xlabel('ΔU from all-gesture contour center (mm)')
    ax.set_ylabel('ΔV from all-gesture contour center (mm)')
    ax.set_title('Proximal vs Distal RF Contour Center Offsets')
    ax.set_aspect('equal')

    marker_handles = [
        Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
        Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + marker_handles, fontsize=7, loc='best')

    if mm_limits is not None:
        ax.set_xlim(*mm_limits[0])
        ax.set_ylim(*mm_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved proximal-distal contour-center aggregate: %s", output_path)
    plt.close(fig)

    # --- by-type variant ---
    if session_colors is not None:
        fig_bt, ax_bt = plt.subplots(figsize=(8, 7), facecolor='white')
        ax_bt.set_facecolor('white')
        ax_bt.axhline(0, color='lightgray', lw=0.8, zorder=0)
        ax_bt.axvline(0, color='lightgray', lw=0.8, zorder=0)

        for session_id in session_ids:
            color = session_colors.get(session_id, 'steelblue')
            offsets = session_contour_centers[session_id]
            prox = offsets['stroke_proximal']
            dist = offsets['stroke_distal']
            ax_bt.plot([prox[0], dist[0]], [prox[1], dist[1]], color=color, lw=1.0, zorder=2)
            ax_bt.plot(prox[0], prox[1], 'o', ms=6, color=color, zorder=3)
            ax_bt.plot(dist[0], dist[1], 'D', ms=6, color=color, zorder=3)
            ax_bt.text(prox[0], prox[1], f' {session_id}', fontsize=6, color=color,
                       ha='left', va='bottom', zorder=4)

        ax_bt.set_xlabel('ΔU from all-gesture contour center (mm)')
        ax_bt.set_ylabel('ΔV from all-gesture contour center (mm)')
        ax_bt.set_title('Proximal vs Distal RF Contour Center Offsets (by type)')
        ax_bt.set_aspect('equal')

        all_legend_handles = []
        if neuron_type_legend:
            all_legend_handles.extend([
                matplotlib.patches.Patch(facecolor=c, label=nt)
                for nt, c in neuron_type_legend.items()
            ])
        all_legend_handles.extend([
            Line2D([], [], marker='o', color='gray', linestyle='None', ms=6, label='Proximal'),
            Line2D([], [], marker='D', color='gray', linestyle='None', ms=6, label='Distal'),
        ])
        ax_bt.legend(handles=all_legend_handles, fontsize=7, loc='best')

        if mm_limits is not None:
            ax_bt.set_xlim(*mm_limits[0])
            ax_bt.set_ylim(*mm_limits[1])

        fig_bt.tight_layout()
        by_type_path = output_path.with_stem(output_path.stem + '_by_type')
        by_type_path.parent.mkdir(parents=True, exist_ok=True)
        fig_bt.savefig(str(by_type_path), dpi=120)
        fig_bt.savefig(by_type_path.with_suffix('.svg'), bbox_inches='tight')
        logger.info("Saved proximal-distal contour-center aggregate (by type): %s", by_type_path)
        plt.close(fig_bt)


_DELTA_METRICS = [
    'delta_area_mm2',
    'delta_area_pct',
    'delta_perimeter_mm',
    'delta_circularity',
    'delta_pca_aspect_ratio',
    'delta_pca_orientation_deg',
    'delta_mean_iff_on_contour',
    'delta_peak_to_centroid_mm',
    'contour_overlap_iou',
    'delta_peak_iff',
    'delta_equivalent_diameter_mm',
    'delta_rf_sharpness',
    'delta_iff_at_centroid',
]

_STRIP_METRICS = [
    'delta_area_mm2',
    'delta_perimeter_mm',
    'delta_circularity',
    'delta_pca_aspect_ratio',
    'delta_mean_iff_on_contour',
    'contour_overlap_iou',
    'heatmap_pearson_r',
    'centroid_shift_along_arm_mm',
    'centroid_shift_across_arm_mm',
    'contour_center_shift_along_arm_mm',
    'contour_center_shift_across_arm_mm',
    'peak_shift_along_arm_mm',
    'peak_shift_across_arm_mm',
    'delta_peak_iff',
    'delta_equivalent_diameter_mm',
    'delta_rf_sharpness',
    'delta_iff_at_centroid',
    'containment_proximal_in_distal',
    'containment_distal_in_proximal',
]


def render_proximal_distal_metric_deltas(
    df: pd.DataFrame,
    output_path: Path,
    session_colors: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    """Multi-panel bar chart showing delta metrics per session, colored by neuron type."""
    n_cols = 3
    n_rows = -(-len(_DELTA_METRICS) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), facecolor='white')
    axes_flat = axes.ravel()

    for panel_idx, metric in enumerate(_DELTA_METRICS):
        ax = axes_flat[panel_idx]
        ax.set_facecolor('white')

        if metric not in df.columns:
            ax.set_visible(False)
            continue

        session_ids = df['session_id'].tolist()
        values = df[metric].tolist()

        bar_colors = [
            (session_colors[sid] if session_colors and sid in session_colors else 'steelblue')
            for sid in session_ids
        ]

        x_positions = range(len(session_ids))
        ax.bar(x_positions, values, color=bar_colors)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(session_ids, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric.replace('_', ' '), fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused panels (guard for when grid has more slots than metrics)
    for panel_idx in range(len(_DELTA_METRICS), len(axes_flat)):
        axes_flat[panel_idx].set_visible(False)

    # Neuron-type legend
    if neuron_type_legend:
        legend_handles = [
            matplotlib.patches.Patch(facecolor=color, label=ntype)
            for ntype, color in neuron_type_legend.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(len(neuron_type_legend), 5),
            fontsize=8,
            title='Neuron type',
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.05 if neuron_type_legend else 0, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved metric deltas bar chart: %s", output_path)
    plt.close(fig)


def render_proximal_distal_population_strips(
    df: pd.DataFrame,
    output_path: Path,
    session_colors: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    """Strip chart with one column per delta metric and Wilcoxon p-value annotation."""
    n_metrics = len(_STRIP_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 5), facecolor='white')
    if n_metrics == 1:
        axes = [axes]

    for col_idx, metric in enumerate(_STRIP_METRICS):
        ax = axes[col_idx]
        ax.set_facecolor('white')

        if metric not in df.columns:
            ax.set_title(metric.replace('_', ' '), fontsize=8)
            continue

        values_series = df[['session_id', metric]].dropna(subset=[metric])
        session_ids = values_series['session_id'].tolist()
        values = values_series[metric].tolist()

        for i, (sid, val) in enumerate(zip(session_ids, values)):
            color = session_colors[sid] if session_colors and sid in session_colors else 'steelblue'
            ax.plot(0, val, 'o', color=color, alpha=0.7, markersize=6)

        # Horizontal reference line at y=0 for delta metrics; skip for bounded metrics
        ax.axhline(0, color='gray', lw=0.8, linestyle='--')

        annotations = []

        # Wilcoxon signed-rank test when enough observations
        n_valid = len(values)
        if n_valid >= 5:
            from scipy.stats import wilcoxon
            try:
                _stat, p_val = wilcoxon(values)
                annotations.append(f"W p={p_val:.3f}")
            except Exception:
                pass  # wilcoxon raises when all values are zero or only one unique value

            # Exact sign test
            from scipy.stats import binomtest
            n_positive = sum(1 for v in values if v > 0)
            n_nonzero = sum(1 for v in values if v != 0)
            if n_nonzero > 0:
                sign_result = binomtest(n_positive, n_nonzero, 0.5)
                sign_p = sign_result.pvalue
                annotations.append(f"S p={sign_p:.3f}")

                # Consistency count with Clopper-Pearson CI
                from scipy.stats import beta as beta_dist
                n_negative = n_nonzero - n_positive
                majority = n_positive if n_positive >= n_negative else n_negative
                ci_lo = beta_dist.ppf(0.025, majority, n_nonzero - majority + 1) if majority > 0 else 0.0
                ci_hi = beta_dist.ppf(0.975, majority + 1, n_nonzero - majority) if majority < n_nonzero else 1.0
                annotations.append(f"{majority}/{n_nonzero} [{ci_lo:.0%}-{ci_hi:.0%}]")

        for i, txt in enumerate(annotations):
            ax.text(0, 1.0 + i * 0.06, txt, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7)

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(metric.replace('_', ' '), fontsize=8, wrap=True)
        ax.tick_params(axis='y', labelsize=7)

    # Neuron-type legend
    if neuron_type_legend:
        legend_handles = [
            matplotlib.patches.Patch(facecolor=color, label=ntype)
            for ntype, color in neuron_type_legend.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(len(neuron_type_legend), 5),
            fontsize=8,
            title='Neuron type',
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.08 if neuron_type_legend else 0, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved population strip chart: %s", output_path)
    plt.close(fig)


def render_paired_metric_violins(
    df: pd.DataFrame,
    metric_pairs: list[tuple[str, str, str]],
    condition_labels: tuple[str, str],
    output_path: Path,
    session_colors: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    """Violin + paired-dot distribution figure for paired-condition metrics.

    Produces a 1xN subplot figure (N = len(metric_pairs)). Each subplot draws a
    violin for condition A at x=0 and condition B at x=1 (alpha-filled bodies),
    overlays one paired dot per session at each x connected by a thin line
    (colored by ``session_colors``), and annotates above the subplot with the
    Wilcoxon signed-rank / sign-test p-values and a consistency count with
    Clopper-Pearson CI computed on the paired deltas ``(col_a - col_b)``.

    Sessions with NaN in either condition for a given metric are dropped from
    that subplot (paired values must be finite for both columns). Statistical
    annotations show "n/a" when fewer than 5 valid paired sessions remain
    (matching the population-strip gating via ``_compute_axis_significance``).
    """
    if not metric_pairs:
        raise ValueError("render_paired_metric_violins: metric_pairs is empty")
    if 'session_id' not in df.columns:
        raise ValueError("render_paired_metric_violins: df missing 'session_id' column")

    label_a, label_b = condition_labels

    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 5), facecolor='white')
    if n_metrics == 1:
        axes = [axes]

    from scipy.stats import beta as beta_dist

    for col_idx, (col_a, col_b, label) in enumerate(metric_pairs):
        ax = axes[col_idx]
        ax.set_facecolor('white')
        ax.set_title(label, fontsize=9, wrap=True)

        if col_a not in df.columns or col_b not in df.columns:
            ax.set_xticks([0, 1])
            ax.set_xticklabels([label_a, label_b], fontsize=8)
            continue

        # Paired rows: drop any session with NaN in either condition.
        paired = df[['session_id', col_a, col_b]].dropna(subset=[col_a, col_b])
        session_ids = paired['session_id'].tolist()
        values_a = paired[col_a].to_numpy(dtype=float)
        values_b = paired[col_b].to_numpy(dtype=float)
        deltas = (values_a - values_b).tolist()

        if len(session_ids) > 0:
            parts = ax.violinplot(
                [values_a, values_b],
                positions=[0, 1],
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body in parts['bodies']:
                body.set_facecolor('lightgray')
                body.set_edgecolor('gray')
                body.set_alpha(0.4)

            # Per-session paired dots connected by a thin line.
            for sid, va, vb in zip(session_ids, values_a, values_b):
                color = session_colors[sid] if session_colors and sid in session_colors else 'steelblue'
                ax.plot([0, 1], [va, vb], color=color, lw=0.8, alpha=0.6, zorder=2)
                ax.plot(0, va, 'o', color=color, alpha=0.8, markersize=5, zorder=3)
                ax.plot(1, vb, 'o', color=color, alpha=0.8, markersize=5, zorder=3)

        # Statistical annotation on paired deltas (col_a - col_b).
        axis_sig = _compute_axis_significance(deltas)
        annotations = [
            f"W p={_fmt_p(axis_sig['wilcoxon_p'])}",
            f"S p={_fmt_p(axis_sig['sign_p'])}",
        ]
        finite_deltas = [d for d in deltas if np.isfinite(d)]
        if len(finite_deltas) >= 5:
            n_positive = sum(1 for d in finite_deltas if d > 0)
            n_nonzero = sum(1 for d in finite_deltas if d != 0)
            if n_nonzero > 0:
                n_negative = n_nonzero - n_positive
                majority = n_positive if n_positive >= n_negative else n_negative
                ci_lo = beta_dist.ppf(0.025, majority, n_nonzero - majority + 1) if majority > 0 else 0.0
                ci_hi = beta_dist.ppf(0.975, majority + 1, n_nonzero - majority) if majority < n_nonzero else 1.0
                annotations.append(f"{majority}/{n_nonzero} [{ci_lo:.0%}-{ci_hi:.0%}]")
        else:
            annotations.append("n/a")

        for i, txt in enumerate(annotations):
            ax.text(0.5, 1.0 + i * 0.06, txt, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7)

        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_a, label_b], fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    # Neuron-type legend at the figure bottom.
    if neuron_type_legend:
        legend_handles = [
            matplotlib.patches.Patch(facecolor=color, label=ntype)
            for ntype, color in neuron_type_legend.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=min(len(neuron_type_legend), 5),
            fontsize=8,
            title='Neuron type',
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout(rect=[0, 0.08 if neuron_type_legend else 0, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved paired metric violins: %s", output_path)
    plt.close(fig)


def render_shift_decomposition(
    df: pd.DataFrame,
    output_path: Path,
    along_col: str,
    across_col: str,
    title: str,
    session_colors: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> dict:
    """Arrow/quiver plot showing per-session shift vectors for a given center type.

    Annotates the figure with a one-sample Hotelling's T² test (is the mean 2D
    shift vector different from the origin?) plus per-axis Wilcoxon signed-rank
    and exact sign tests on the along-arm (ΔU) and across-arm (ΔV) components.

    Returns a stats row (center type, N, mean shift, all p-values) so the caller
    can persist a per-comparison significance table.
    """
    center_type = along_col.replace('_shift_along_arm_mm', '')
    along_vals = df[along_col].tolist() if along_col in df.columns else []
    across_vals = df[across_col].tolist() if across_col in df.columns else []
    axis_along = _compute_axis_significance(along_vals)
    axis_across = _compute_axis_significance(across_vals)
    hotelling = _compute_hotelling_t2(along_vals, across_vals)

    try:
        cmap_tab20 = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap_tab20 = matplotlib.cm.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.axhline(0, color='lightgray', lw=0.8, zorder=0)
    ax.axvline(0, color='lightgray', lw=0.8, zorder=0)

    session_ids_all = df['session_id'].tolist()
    n_sessions = len(session_ids_all)

    for i, row in enumerate(df.itertuples(index=False)):
        session_id = row.session_id
        along = getattr(row, along_col, float('nan'))
        across = getattr(row, across_col, float('nan'))

        if not (np.isfinite(along) and np.isfinite(across)):
            continue

        if session_colors and session_id in session_colors:
            color = session_colors[session_id]
        else:
            color = cmap_tab20(i / max(n_sessions, 1))

        ax.quiver(
            0, 0, along, across,
            color=color,
            angles='xy', scale_units='xy', scale=1,
            width=0.004, headwidth=4, headlength=5,
            zorder=3,
        )
        ax.text(along, across, session_id, fontsize=6, color=color, zorder=4)

    ax.set_xlabel('Along arm (ΔU) mm')
    ax.set_ylabel('Across arm (ΔV) mm')
    ax.set_title(title)
    ax.set_aspect('equal')

    # Significance annotation: 2D Hotelling T² + per-axis Wilcoxon/sign tests
    stat_lines = [f"N = {hotelling['n']}"]
    if np.isfinite(hotelling['p']):
        stat_lines.append(f"Hotelling T²: p = {hotelling['p']:.3f}")
    else:
        stat_lines.append("Hotelling T²: n/a")
    stat_lines.append(
        f"ΔU along: W p={_fmt_p(axis_along['wilcoxon_p'])}, "
        f"sign p={_fmt_p(axis_along['sign_p'])}"
    )
    stat_lines.append(
        f"ΔV across: W p={_fmt_p(axis_across['wilcoxon_p'])}, "
        f"sign p={_fmt_p(axis_across['sign_p'])}"
    )
    ax.text(
        0.02, 0.98, "\n".join(stat_lines),
        transform=ax.transAxes, ha='left', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85),
        zorder=5,
    )

    if neuron_type_legend:
        legend_handles = [
            matplotlib.patches.Patch(facecolor=color, label=ntype)
            for ntype, color in neuron_type_legend.items()
        ]
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            title='Neuron type',
            loc='best',
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    logger.info("Saved centroid shift decomposition: %s", output_path)
    plt.close(fig)

    def _nanmean(vals: list[float]) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float('nan')

    return {
        'center_type': center_type,
        'n': hotelling['n'],
        'mean_along_mm': _nanmean(along_vals),
        'mean_across_mm': _nanmean(across_vals),
        'hotelling_T2': hotelling['T2'],
        'hotelling_F': hotelling['F'],
        'hotelling_p': hotelling['p'],
        'along_wilcoxon_p': axis_along['wilcoxon_p'],
        'along_sign_p': axis_along['sign_p'],
        'across_wilcoxon_p': axis_across['wilcoxon_p'],
        'across_sign_p': axis_across['sign_p'],
    }
