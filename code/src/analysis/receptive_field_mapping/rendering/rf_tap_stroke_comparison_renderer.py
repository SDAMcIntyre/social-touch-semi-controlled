"""Renderer for tap-vs-stroke RF comparison plots."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    _draw_forearm_mesh_background,
)

logger = logging.getLogger(__name__)

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
    'peak_shift_along_arm_mm',
    'peak_shift_across_arm_mm',
    'delta_peak_iff',
    'delta_equivalent_diameter_mm',
    'delta_rf_sharpness',
    'delta_iff_at_centroid',
    'containment_tap_in_stroke',
    'containment_stroke_in_tap',
    'contour_center_shift_along_arm_mm',
    'contour_center_shift_across_arm_mm',
]


def render_tap_stroke_contour_overlay(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    contour_tap_uv: np.ndarray,
    contour_stroke_uv: np.ndarray,
    centroid_tap_uv: np.ndarray,
    centroid_stroke_uv: np.ndarray,
    output_path: Path,
    contour_all_uv: np.ndarray | None = None,
    hotspot_tap_uv: np.ndarray | None = None,
    hotspot_stroke_uv: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    iou: float | None = None,
    area_ratio: float | None = None,
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Contour overlay: tap (green) vs stroke (red), optional all (dashed grey)."""
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

    # All-gesture contour (reference) as dashed grey
    if contour_all_uv is not None:
        _closed = np.vstack([contour_all_uv, contour_all_uv[0]])
        ax.plot(_closed[:, 0], _closed[:, 1], color='grey', linestyle='--', lw=1.2, zorder=4, label='all')

    # Tap contour — green solid
    _tap_closed = np.vstack([contour_tap_uv, contour_tap_uv[0]])
    ax.plot(_tap_closed[:, 0], _tap_closed[:, 1], color='#2ca02c', linestyle='-', lw=1.5, zorder=5, label='tap')

    # Stroke contour — red solid
    _stroke_closed = np.vstack([contour_stroke_uv, contour_stroke_uv[0]])
    ax.plot(_stroke_closed[:, 0], _stroke_closed[:, 1], color='#d62728', linestyle='-', lw=1.5, zorder=5, label='stroke')

    # Centroids
    ax.plot(
        centroid_tap_uv[0], centroid_tap_uv[1],
        color='#2ca02c', marker='+', markersize=10, markeredgewidth=2, linestyle='none', zorder=7,
    )
    ax.plot(
        centroid_stroke_uv[0], centroid_stroke_uv[1],
        color='#d62728', marker='+', markersize=10, markeredgewidth=2, linestyle='none', zorder=7,
    )

    # Hotspots (optional)
    if hotspot_tap_uv is not None:
        ax.plot(
            hotspot_tap_uv[0], hotspot_tap_uv[1],
            color='#2ca02c', marker='*', markersize=10, markeredgewidth=1.5, linestyle='none', zorder=8,
        )
    if hotspot_stroke_uv is not None:
        ax.plot(
            hotspot_stroke_uv[0], hotspot_stroke_uv[1],
            color='#d62728', marker='*', markersize=10, markeredgewidth=1.5, linestyle='none', zorder=8,
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
    logger.info("Saved tap-stroke contour overlay: %s", output_path)
    plt.close(fig)


def render_tap_stroke_heatmap_triptych(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z_tap: np.ndarray,
    grid_z_stroke: np.ndarray,
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
    """Three-panel heatmap: Tap | Stroke | Difference (T - S)."""
    norm = LogNorm(vmin=vmin, vmax=vmax) if heatmap_space == "log" else Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    if title:
        fig.suptitle(title, color='white', fontsize=10)

    panel_titles = ["Tap", "Stroke", "Difference (T − S)"]

    # Compute difference grid
    diff = grid_z_tap - grid_z_stroke
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
            display_grid = np.where(grid_z_tap > 0, grid_z_tap, np.nan)
            ax.pcolormesh(grid_u, grid_v, display_grid, cmap=cmap, norm=norm, shading='auto')
        elif panel_idx == 1:
            display_grid = np.where(grid_z_stroke > 0, grid_z_stroke, np.nan)
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
    logger.info("Saved tap-stroke heatmap triptych: %s", output_path)
    plt.close(fig)


def render_tap_stroke_aggregate(
    session_centroids: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    mm_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    """Scatter plot of tap and stroke centroid offsets relative to all-gesture center."""
    if not session_centroids:
        raise ValueError("render_tap_stroke_aggregate: session_centroids is empty")

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
        tap = offsets['tap']
        stroke = offsets['stroke']

        ax.plot([tap[0], stroke[0]], [tap[1], stroke[1]], color=color, lw=1.0, zorder=2)
        ax.plot(tap[0], tap[1], 'o', ms=6, color=color, label=session_id, zorder=3)
        ax.plot(stroke[0], stroke[1], 'D', ms=6, color=color, zorder=3)

    ax.set_xlabel('ΔU from all-gesture center (mm)')
    ax.set_ylabel('ΔV from all-gesture center (mm)')
    ax.set_title('Tap vs Stroke RF Center Offsets')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')

    if mm_limits is not None:
        ax.set_xlim(*mm_limits[0])
        ax.set_ylim(*mm_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    logger.info("Saved tap-stroke aggregate: %s", output_path)
    plt.close(fig)


def render_tap_stroke_hotspot_aggregate(
    session_hotspots: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    mm_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    """Scatter of peak offsets relative to all-gesture peak."""
    if not session_hotspots:
        raise ValueError("render_tap_stroke_hotspot_aggregate: session_hotspots is empty")

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
        tap = offsets['tap']
        stroke = offsets['stroke']

        ax.plot([tap[0], stroke[0]], [tap[1], stroke[1]], color=color, lw=1.0, zorder=2)
        ax.plot(tap[0], tap[1], 'o', ms=6, color=color, label=session_id, zorder=3)
        ax.plot(stroke[0], stroke[1], 'D', ms=6, color=color, zorder=3)

    ax.set_xlabel('ΔU from all-gesture hotspot (mm)')
    ax.set_ylabel('ΔV from all-gesture hotspot (mm)')
    ax.set_title('Tap vs Stroke RF Hotspot Offsets')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='best')

    if mm_limits is not None:
        ax.set_xlim(*mm_limits[0])
        ax.set_ylim(*mm_limits[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    logger.info("Saved tap-stroke hotspot aggregate: %s", output_path)
    plt.close(fig)


def render_tap_stroke_metric_deltas(
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
    logger.info("Saved tap-stroke metric deltas bar chart: %s", output_path)
    plt.close(fig)


def render_tap_stroke_population_strips(
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
    logger.info("Saved tap-stroke population strip chart: %s", output_path)
    plt.close(fig)
