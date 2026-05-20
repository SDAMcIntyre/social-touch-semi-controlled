"""Radar/spider plot renderer for touch gesture feature distributions.

Produces dark-themed polar-axes figures showing median + IQR per gesture type.
All functions are pure renderers — no I/O except writing the output PNG.
The Agg backend is forced at import time so this module is safe to use in
headless pipeline contexts.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

_BG = "#1a1a1a"
_AXIS_COLOR = "#444444"
_TICK_COLOR = "white"
_TICK_FONTSIZE = 8

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


def _apply_polar_dark_style(fig: plt.Figure, ax) -> None:
    """Apply dark theme to a single polar Axes instance."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Gridlines and spines
    ax.grid(color=_AXIS_COLOR, linewidth=0.8)
    ax.spines["polar"].set_color(_AXIS_COLOR)

    # Tick colours
    ax.tick_params(colors=_TICK_COLOR, labelsize=_TICK_FONTSIZE)

    # Radial axis range
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       color=_TICK_COLOR, fontsize=_TICK_FONTSIZE - 1)

    # Title colour
    ax.title.set_color("white")


def _draw_radar_trace(
    ax,
    angles: np.ndarray,
    medians: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    color: str,
    label: str | None = None,
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
    label:
        Legend label, or None.
    """
    # IQR shaded band
    ax.fill_between(angles, q25, q75, color=color, alpha=0.15)

    # Median polygon — filled + edge line
    ax.fill(angles, medians, color=color, alpha=0.4)
    ax.plot(angles, medians, color=color, linewidth=1.8, label=label)


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
) -> None:
    """Render a single-gesture radar plot and save to *out_path*.

    Parameters
    ----------
    labels:
        List of N axis label strings.
    medians:
        1-D array of length N, values in [0, 1] (pre-normalised).
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

    # Set axis labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=_TICK_COLOR, fontsize=_TICK_FONTSIZE)

    _draw_radar_trace(ax, angles, med_c, q25_c, q75_c, color)

    ax.set_title(title, color="white", pad=20, fontsize=12)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


def render_gesture_radar_composite(
    labels: list[str],
    gesture_stats: dict[str, dict],
    title: str,
    out_path: Path,
) -> None:
    """Render a composite radar plot overlaying multiple gesture types.

    Parameters
    ----------
    labels:
        List of N axis label strings (shared across all gesture types).
    gesture_stats:
        Dict mapping gesture name (``str``) to a dict with keys:

        * ``"medians"`` — 1-D array of length N in [0, 1]
        * ``"q25"``     — 1-D array of length N, lower IQR bound in [0, 1]
        * ``"q75"``     — 1-D array of length N, upper IQR bound in [0, 1]
        * ``"color"``   — hex colour string

    title:
        Figure title string.
    out_path:
        ``pathlib.Path`` — save figure here as PNG.  Parent directory must
        already exist; this function does NOT create directories.
    """
    n = len(labels)
    if n < 3:
        raise ValueError(f"Need at least 3 feature axes for a radar plot, got {n}.")
    if not gesture_stats:
        raise ValueError("gesture_stats must contain at least one gesture entry.")

    angles = _build_angles(n)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    _apply_polar_dark_style(fig, ax)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=_TICK_COLOR, fontsize=_TICK_FONTSIZE)

    for gesture_name, stats in gesture_stats.items():
        medians = stats["medians"]
        q25 = stats["q25"]
        q75 = stats["q75"]
        color = stats["color"]

        if len(medians) != n or len(q25) != n or len(q75) != n:
            raise ValueError(
                f"Gesture '{gesture_name}': array lengths must match labels length {n}. "
                f"Got medians={len(medians)}, q25={len(q25)}, q75={len(q75)}."
            )

        med_c = _make_closed(medians)
        q25_c = _make_closed(q25)
        q75_c = _make_closed(q75)

        _draw_radar_trace(ax, angles, med_c, q25_c, q75_c, color, label=gesture_name)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.15),
        facecolor="#2a2a2a",
        labelcolor="white",
        fontsize=9,
        framealpha=0.85,
    )

    ax.set_title(title, color="white", pad=20, fontsize=12)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
