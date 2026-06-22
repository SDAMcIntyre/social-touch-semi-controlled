"""Rendering functions for 1D RF cross-section profiles and boundary crossings."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

_BG = 'white'
_AX_BG = 'white'
_SPINE_COLOR = '#333333'


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors='black', which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')


def _tangent_half_length(coords: np.ndarray) -> float:
    """Return a moderate tangent half-length scaled to ~8% of the coordinate range."""
    span = float(np.nanmax(coords) - np.nanmin(coords))
    return span * 0.02 if span > 0 else 1.0


def render_center_axis_profile(
    coords: np.ndarray,
    iff_values: np.ndarray,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "IFF (Hz)",
    contour_color: str = "red",
) -> None:
    """Render a 1D line plot of IFF along one axis at a given center point."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(iff_values)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
    else:
        ax.plot(coords, iff_values, color='#0077b6', linewidth=1.2)

        valid = ~np.isnan(iff_values)
        c_valid = coords[valid]
        v_valid = iff_values[valid]
        half = _tangent_half_length(coords)

        for xc in boundary_crossings:
            yc = float(np.interp(xc, c_valid, v_valid))
            ax.plot(xc, yc, 'o', color=contour_color, markersize=6, zorder=5)

            dx = np.gradient(c_valid)
            dy = np.gradient(v_valid)
            slope = float(np.interp(xc, c_valid, dy / dx))
            x0, x1 = xc - half, xc + half
            y0, y1 = yc - half * slope, yc + half * slope
            ax.plot([x0, x1], [y0, y1], color=contour_color, linewidth=1.2,
                    alpha=0.9, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_laplacian_context(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    laplacian: np.ndarray,
    smoothed: np.ndarray,
    contour_uv: np.ndarray | None,
    scanline_coord: float,
    scanline_axis: str,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    contour_color: str = "red",
) -> None:
    """Render a 2D view of the smoothed field and its Laplacian with the profile scanline overlaid.

    Parameters
    ----------
    grid_u, grid_v:
        (R, C) coordinate grids (axis 0 = U, axis 1 = V).
    laplacian:
        (R, C) Laplacian array (NaN outside mesh).
    smoothed:
        (R, C) Gaussian-smoothed IFF field (NaN outside mesh).
    contour_uv:
        (N, 2) inflection contour in UV space, or None.
    scanline_coord:
        The fixed coordinate value of the 1D profile scanline.
    scanline_axis:
        ``"U"`` for a constant-V scanline (profile along U),
        ``"V"`` for a constant-U scanline (profile along V).
    boundary_crossings:
        Sorted crossing coordinates along the profile axis.
    output_path:
        Path for the PNG output. SVG is written alongside.
    title:
        Figure suptitle.
    contour_color:
        Color for the inflection contour and crossing markers.
    """
    fig, (ax_smooth, ax_lap) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax_smooth)
    _style_ax(ax_lap)

    u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
    v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
    extent = [u_min, u_max, v_min, v_max]

    jet = plt.cm.jet.copy()
    jet.set_bad('white')
    smoothed_t = np.ma.masked_invalid(smoothed.T)
    ax_smooth.imshow(smoothed_t, cmap=jet, origin='lower', extent=extent, aspect='auto')
    ax_smooth.set_title("Smoothed IFF", color='black')
    ax_smooth.set_xlabel("U (mm)")
    ax_smooth.set_ylabel("V (mm)")

    vabs = float(np.nanmax(np.abs(laplacian)))
    vabs = max(vabs, 1e-12)
    laplacian_t = np.ma.masked_invalid(laplacian.T)
    ax_lap.imshow(laplacian_t, cmap='RdBu_r', origin='lower', extent=extent,
                  vmin=-vabs, vmax=vabs, aspect='auto')
    ax_lap.set_title("Laplacian", color='black')
    ax_lap.set_xlabel("U (mm)")
    ax_lap.set_ylabel("V (mm)")

    for ax in (ax_smooth, ax_lap):
        if contour_uv is not None and len(contour_uv) > 0:
            closed = np.vstack([contour_uv, contour_uv[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=contour_color,
                    linewidth=1.2, zorder=5)

        if scanline_axis == "U":
            ax.axhline(scanline_coord, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.7, zorder=4)
            for xc in boundary_crossings:
                ax.plot(xc, scanline_coord, 'o', color=contour_color,
                        markersize=5, zorder=6)
        else:
            ax.axvline(scanline_coord, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.7, zorder=4)
            for yc in boundary_crossings:
                ax.plot(scanline_coord, yc, 'o', color=contour_color,
                        markersize=5, zorder=6)

    fig.suptitle(title, color='black', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_laplacian_profile(
    coords: np.ndarray,
    lap_values: np.ndarray,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    contour_color: str = "red",
) -> None:
    """Render a 1D Laplacian cross-section with zero-crossing reference line.

    The Laplacian curve is colored by sign: blue (negative, concave basin)
    below zero, red-orange (positive, convex flank) above zero. Boundary
    crossings mark the negative-to-positive transitions where the inflection
    contour intersects the scanline.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(lap_values)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
    else:
        valid = ~np.isnan(lap_values)
        c_valid = coords[valid]
        v_valid = lap_values[valid]

        ax.fill_between(
            c_valid, v_valid, 0,
            where=v_valid <= 0, interpolate=True,
            color='#0077b6', alpha=0.25, label='negative (concave)',
        )
        ax.fill_between(
            c_valid, v_valid, 0,
            where=v_valid >= 0, interpolate=True,
            color='#e63946', alpha=0.25, label='positive (convex)',
        )
        ax.plot(coords, lap_values, color='#333333', linewidth=1.2)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5, zorder=3)

        for xc in boundary_crossings:
            ax.plot(xc, 0, 'o', color=contour_color, markersize=6, zorder=5)
            ax.axvline(xc, color=contour_color, linewidth=0.6,
                       linestyle=':', alpha=0.6, zorder=4)

        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Laplacian")
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
