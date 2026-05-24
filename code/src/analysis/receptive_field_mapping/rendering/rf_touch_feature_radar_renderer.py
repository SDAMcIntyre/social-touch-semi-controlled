"""Radar/spider plot renderer for touch gesture feature distributions.

Produces dark-themed polar-axes figures showing median + IQR per gesture type.
All functions are pure renderers — no I/O except writing the output PNG.
The Agg backend is forced at import time so this module is safe to use in
headless pipeline contexts.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

_BG = "#1a1a1a"
_AXIS_COLOR = "#444444"
_TICK_COLOR = "white"
_TICK_FONTSIZE = 24

# ---------------------------------------------------------------------------
# Gesture color palette
# ---------------------------------------------------------------------------

GESTURE_COLORS = {
    "tap": "#00bcd4",
    "stroke_proximal": "#ff9800",
    "stroke_distal": "#e040fb",
    "all": "#78909c",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_closed(arr: np.ndarray) -> np.ndarray:
    """Append first element to end to close the radar polygon."""
    return np.concatenate([arr, arr[:1]])


def _build_angles(n: int) -> np.ndarray:
    """Return N+1 angles (radians) evenly spaced around the circle, closed."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([angles, angles[:1]])


def _format_value(v: float) -> str:
    """Format a numeric value compactly for annotation labels."""
    abs_v = abs(v)
    if abs_v == 0:
        return "0"
    if abs_v >= 100:
        return f"{v:.0f}"
    if abs_v >= 1:
        return f"{v:.1f}"
    if abs_v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def _apply_polar_dark_style(fig: plt.Figure, ax) -> None:
    """Apply dark theme to a single polar Axes instance."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Gridlines and spines
    ax.grid(color=_AXIS_COLOR, linewidth=0.8)
    ax.spines["polar"].set_color(_AXIS_COLOR)

    # Tick colours
    ax.tick_params(colors=_TICK_COLOR, labelsize=_TICK_FONTSIZE)

    # Radial axis range — no numeric radial tick labels (per-axis max shown instead)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])

    # Title colour
    ax.title.set_color("white")


def _draw_radar_trace(
    ax,
    angles: np.ndarray,
    medians: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    color: str,
) -> None:
    """Draw one gesture trace (median polygon + IQR band) onto *ax*.

    Parameters
    ----------
    ax:
        Polar Axes to draw on.
    angles:
        Closed angle array of length N+1.
    medians, q25, q75:
        Already-closed arrays of length N+1.
    color:
        Hex colour string.
    """
    # IQR shaded band
    ax.fill_between(angles, q25, q75, color=color, alpha=0.15)

    # Median polygon — filled + edge line
    ax.fill(angles, medians, color=color, alpha=0.4)
    ax.plot(angles, medians, color=color, linewidth=1.8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_gesture_radar(
    labels: list[str],
    medians: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    title: str,
    color: str,
    out_path: Path,
    axis_max_values: np.ndarray,
    raw_medians: np.ndarray,
) -> None:
    """Render a single-gesture radar plot and save to *out_path*.

    Parameters
    ----------
    labels:
        List of N axis label strings.
    medians:
        1-D array of length N, values in [0, 1] (normalised for plotting).
    q25:
        1-D array of length N, lower IQR bound in [0, 1].
    q75:
        1-D array of length N, upper IQR bound in [0, 1].
    title:
        Figure title string.
    color:
        Hex colour string for this gesture type.
    out_path:
        ``pathlib.Path`` — save figure here as PNG.  Parent directory must
        already exist; this function does NOT create directories.
    axis_max_values:
        1-D array of length N — raw (un-normalised) maximum value per axis,
        displayed at the outer ring of each spoke.
    raw_medians:
        1-D array of length N — raw (un-normalised) median values, annotated
        next to each data point on the median polygon.
    """
    n = len(labels)
    if n < 3:
        raise ValueError(f"Need at least 3 feature axes for a radar plot, got {n}.")
    if len(medians) != n or len(q25) != n or len(q75) != n:
        raise ValueError(
            f"labels length ({n}) must match medians ({len(medians)}), "
            f"q25 ({len(q25)}), and q75 ({len(q75)}) lengths."
        )

    angles = _build_angles(n)
    med_c = _make_closed(medians)
    q25_c = _make_closed(q25)
    q75_c = _make_closed(q75)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    _apply_polar_dark_style(fig, ax)

    # Axis labels with max value on the outer ring
    ax.set_xticks(angles[:-1])
    outer_labels = [
        f"{lbl}\n{_format_value(mx)}"
        for lbl, mx in zip(labels, axis_max_values)
    ]
    ax.set_xticklabels(outer_labels, color=_TICK_COLOR, fontsize=_TICK_FONTSIZE)

    _draw_radar_trace(ax, angles, med_c, q25_c, q75_c, color)

    # Annotate raw median values near each data point
    _ANNOTATION_FS = _TICK_FONTSIZE - 1
    for i in range(n):
        r = medians[i]
        offset_r = r + 0.07 if r < 0.85 else r - 0.07
        ax.text(
            angles[i], offset_r, _format_value(raw_medians[i]),
            ha="center", va="center",
            color="white", fontsize=_ANNOTATION_FS,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=_BG, ec="none", alpha=0.7),
        )

    # Legend
    median_patch = mpatches.Patch(color=color, alpha=0.4, label="Median")
    iqr_patch = mpatches.Patch(color=color, alpha=0.15, label="IQR (Q25–Q75)")
    ax.legend(
        handles=[median_patch, iqr_patch],
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        facecolor="#2a2a2a",
        labelcolor="white",
        fontsize=_TICK_FONTSIZE - 2,
        framealpha=0.85,
    )

    ax.set_title(title, color="white", pad=20, fontsize=_TICK_FONTSIZE + 2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


