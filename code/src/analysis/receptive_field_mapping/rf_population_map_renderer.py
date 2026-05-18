"""2D population RF heatmap renderer — scatter + interpolated heatmap."""

import logging
from pathlib import Path

import igl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)


def _interpolate_on_mesh(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
) -> np.ndarray:
    contacted_mask = np.isfinite(heatmap_val)
    b = np.where(contacted_mask)[0]

    if len(b) < 4:
        raise ValueError(
            f"_interpolate_on_mesh: harmonic interpolation requires at least 4 "
            f"contacted vertices, got {len(b)}. Gesture type may have too few contacts."
        )

    z = np.where(heatmap_val >= 0.0, heatmap_val, 0.0)
    bc = z[b].reshape(-1, 1)

    filled = igl.harmonic(forearm_V, forearm_faces, b, bc, 1).ravel()

    face_mask = ~np.any(contacted_mask[forearm_faces], axis=1)

    tri = mtri.Triangulation(forearm_uv[:, 0], forearm_uv[:, 1], triangles=forearm_faces, mask=face_mask)
    interp = mtri.CubicTriInterpolator(tri, filled, kind='min_E')

    grid_z_masked = interp(grid_u, grid_v)
    grid_z = grid_z_masked.data.copy()
    grid_z[grid_z_masked.mask] = np.nan

    if np.all(np.isnan(grid_z)):
        raise ValueError(
            "_interpolate_on_mesh: interpolation produced all-NaN output. "
            "All contacted vertices may fall outside unmasked triangles."
        )

    grid_z = np.clip(grid_z, 0.0, None)
    return grid_z


def render_population_rf_map(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    vmax: float,
    title: str,
    output_path: Path,
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
    """
    matplotlib.use('Agg')

    norm = Normalize(vmin=0.0, vmax=vmax)

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

    u_all = forearm_uv[:, 0]
    v_all = forearm_uv[:, 1]
    margin_u = (u_all.max() - u_all.min()) * 0.05 or 1.0
    margin_v = (v_all.max() - v_all.min()) * 0.05 or 1.0
    grid_u, grid_v = np.mgrid[
        u_all.min() - margin_u : u_all.max() + margin_u : 150j,
        v_all.min() - margin_v : v_all.max() + margin_v : 150j,
    ]

    grid_z = _interpolate_on_mesh(forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v)

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


_PANEL_ORDER = ['all', 'tap', 'stroke_proximal', 'stroke_distal']


def render_population_rf_composite(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    results: dict[str, tuple[np.ndarray, int, int]],
    vmax: float,
    session_id: str,
    panel_type: str,
    output_path: Path,
    min_overlap_pct: float,
    uv_xlim: tuple[float, float],
    uv_ylim: tuple[float, float],
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
    """
    matplotlib.use('Agg')

    ordered_gtypes = [g for g in _PANEL_ORDER if g in results]
    if not ordered_gtypes:
        raise ValueError(
            f"render_population_rf_composite: no gesture types present in results "
            f"for session '{session_id}'."
        )

    n_panels = len(ordered_gtypes)
    norm = Normalize(vmin=0.0, vmax=vmax)
    stride = max(1, len(forearm_uv) // 5000)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), facecolor='black')
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)
    fig.subplots_adjust(wspace=0.15, right=0.92)

    type_label = 'Scatter' if panel_type == 'scatter' else 'Interpolated heatmap'
    fig.suptitle(f"{session_id} | {type_label} composite", color='white', fontsize=11, y=1.01)

    if panel_type == 'interpolated':
        grid_u, grid_v = np.mgrid[
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

        ax.set_xlim(uv_xlim)
        ax.set_ylim(uv_ylim)

        valid_mask = np.isfinite(heatmap_val) & (heatmap_val >= 0.0)
        valid_uv = forearm_uv[valid_mask]
        valid_vals = heatmap_val[valid_mask]

        ax.scatter(
            forearm_uv[::stride, 0], forearm_uv[::stride, 1],
            c='#404040', s=4, alpha=0.5, linewidths=0, rasterized=True,
        )

        if panel_type == 'scatter':
            if len(valid_uv) > 0:
                ax.scatter(
                    valid_uv[:, 0], valid_uv[:, 1],
                    c=valid_vals, cmap='jet', norm=norm,
                    s=20, alpha=0.9, linewidths=0, zorder=3, rasterized=True,
                )
        else:
            grid_z = _interpolate_on_mesh(forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v)
            ax.pcolormesh(
                grid_u, grid_v, grid_z,
                cmap='jet', norm=norm, shading='auto',
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

    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label='Mean IFF / spike', shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved population RF composite: %s", output_path)
    plt.close(fig)
