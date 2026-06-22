"""Rendering functions for 1D RF cross-section profiles and boundary crossings."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_BG = '#1e1e1e'
_AX_BG = '#2d2d2d'
_SPINE_COLOR = '#555555'


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')


def render_profile_strip_with_boundary(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    contour_uv: np.ndarray | None,
    crossings_per_v: dict[int, np.ndarray],
    output_path: Path,
    title: str,
    cmap: str = "inferno",
    vmax: float | None = None,
    contour_color: str = "red",
) -> None:
    """Render a 2D pcolormesh heatmap strip with boundary contour and crossing markers."""
    v_coords = grid_v[0, :]
    u_coords = grid_u[:, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    mesh = ax.pcolormesh(
        v_coords, u_coords, grid_z,
        cmap=cmap, shading='auto', vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    if contour_uv is not None and len(contour_uv) > 0:
        ax.plot(
            contour_uv[:, 1], contour_uv[:, 0],
            color=contour_color, linewidth=1.5, zorder=3,
        )

    for v_idx, u_crossings in crossings_per_v.items():
        if len(u_crossings) == 0:
            continue
        v_val = float(v_coords[v_idx])
        ax.plot(
            np.full(len(u_crossings), v_val), u_crossings,
            marker='x', color='cyan', markersize=4, linestyle='none', zorder=4,
        )

    ax.set_xlabel("V (mm)")
    ax.set_ylabel("U (mm)")
    ax.set_title(title, color='white')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_representative_profiles(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    contour_uv: np.ndarray | None,
    output_path: Path,
    title: str,
    n_profiles: int = 15,
) -> None:
    """Render ~15 representative 1D line profiles as stacked subplots with boundary markers."""
    u_coords = grid_u[:, 0]
    n_cols = grid_v.shape[1]

    candidate_indices = np.linspace(0, n_cols - 1, n_profiles, dtype=int)

    from analysis.receptive_field_mapping.pipelines.rf_profile_extraction_pipeline import (
        find_boundary_u_crossings,
    )

    valid_entries: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for j in candidate_indices:
        iff_values = grid_z[:, j]
        if np.all(np.isnan(iff_values)):
            continue
        v_value = float(grid_v[0, j])
        crossings = find_boundary_u_crossings(contour_uv, v_value)
        valid_entries.append((j, v_value, iff_values, crossings))

    if not valid_entries:
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_AX_BG)
        ax.text(0.5, 0.5, "No valid profiles", transform=ax.transAxes,
                ha='center', va='center', color='white')
        ax.set_title(title, color='white')
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    n_valid = len(valid_entries)
    fig, axes = plt.subplots(n_valid, 1, figsize=(8, 2 * n_valid), sharex=True)
    fig.patch.set_facecolor(_BG)

    if n_valid == 1:
        axes = [axes]

    for idx, (j, v_value, iff_values, crossings) in enumerate(valid_entries):
        ax = axes[idx]
        _style_ax(ax)
        ax.plot(u_coords, iff_values, color='#4ec9b0', linewidth=1.0)
        for u_cross in crossings:
            ax.axvline(u_cross, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f"V = {v_value:.1f} mm", fontsize=8)

    axes[-1].set_xlabel("U (mm)")
    fig.suptitle(title, color='white')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
