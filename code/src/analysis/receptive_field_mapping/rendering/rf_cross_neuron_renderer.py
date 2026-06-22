"""Neuron-type-grouped strip charts with non-parametric statistical annotations.

Two public entry points:
- ``render_cross_neuron_comparison`` — single-condition: one bar per neuron type.
- ``render_cross_neuron_comparison_multicondition`` — multi-condition: grouped
  bars per neuron type, one group per type, one bar per condition.

Statistical annotations:
- Kruskal-Wallis H-test across all groups (displayed as text in figure corner).
- Mann-Whitney U pairwise between all pairs with n>=2 (bracket annotations
  with significance stars; pairs with p>=0.05 are not annotated).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy import stats

from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    NEURON_TYPE_COLORS,
    NEURON_TYPE_ORDER,
)


_BG = '#1a1a1a'
_SPINE_COLOR = '#555555'
_GRID_COLOR = '#444444'

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _draw_bracket(ax, x1: float, x2: float, y: float, label: str, color: str = "white") -> None:
    """Draw a significance bracket between x1 and x2 at height y."""
    tick_h = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y], lw=1.0, color=color)
    ax.text(
        (x1 + x2) / 2,
        y + tick_h * 1.2,
        label,
        ha='center',
        va='bottom',
        fontsize=9,
        color=color,
    )


def _pairwise_mannwhitney_brackets(
    ax,
    positions: list[float],
    groups: list[list[float]],
    y_top: float,
    y_step_frac: float = 0.08,
) -> float:
    """Draw Mann-Whitney brackets for all significant pairs.

    Returns the updated y_top after all brackets are drawn.
    """
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_step = y_step_frac * y_range

    bracket_level: dict[int, int] = {}
    pair_results: list[tuple[int, int, str]] = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            if len(a) < 2 or len(b) < 2:
                continue
            _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            stars = _significance_stars(p)
            if not stars:
                continue
            pair_results.append((i, j, stars))

    for i, j, stars in pair_results:
        level = 0
        for k in range(i, j + 1):
            level = max(level, bracket_level.get(k, 0))
        level += 1
        for k in range(i, j + 1):
            bracket_level[k] = level
        y = y_top + (level - 1) * y_step
        _draw_bracket(ax, positions[i], positions[j], y, stars)
        new_y = y + y_step
        if new_y > y_top:
            y_top = new_y

    return y_top


def _kruskal_wallis_text(ax, groups: list[list[float]]) -> None:
    """Display Kruskal-Wallis result in the upper-left corner of *ax*."""
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return
    h, p = stats.kruskal(*valid)
    text = f"KW H={h:.2f}, p={p:.3f}"
    ax.text(
        0.02, 0.97, text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=8, color='#cccccc',
    )


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------


def _style_cross_neuron_ax(
    ax,
    x_labels: list[str],
    x_positions: list[float],
    ylabel: str,
) -> None:
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10, color='white')
    ax.set_ylabel(ylabel or "", color='white', fontsize=11)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for s in ax.spines:
        ax.spines[s].set_edgecolor(_SPINE_COLOR)
    ax.grid(axis='y', color=_GRID_COLOR, linestyle='--', alpha=0.5)
    ax.set_facecolor(_BG)


# ---------------------------------------------------------------------------
# Single-condition renderer
# ---------------------------------------------------------------------------


def render_cross_neuron_comparison(
    neuron_data: dict[str, list[float]],
    metric_name: str,
    output_path: Path,
    *,
    ylabel: str = "",
    title: str = "",
) -> None:
    """Render a neuron-type-grouped bar+strip chart for a single condition.

    Parameters
    ----------
    neuron_data:
        ``{neuron_type: [value, ...]}`` — one float per neuron for this type.
        Types absent from this dict are skipped.
    metric_name:
        Metric identifier used in the figure title when *title* is empty.
    output_path:
        Absolute path where the PNG will be saved.
    ylabel:
        Y-axis label (defaults to *metric_name* when empty).
    title:
        Figure title (defaults to *metric_name* when empty).
    """
    ordered_types = [nt for nt in NEURON_TYPE_ORDER if nt in neuron_data]
    if not ordered_types:
        raise ValueError(
            f"render_cross_neuron_comparison: neuron_data is empty or contains no "
            f"known neuron types. Known types: {NEURON_TYPE_ORDER}."
        )

    n_groups = len(ordered_types)
    positions = list(range(n_groups))
    groups = [neuron_data[nt] for nt in ordered_types]
    colors = [NEURON_TYPE_COLORS[nt] for nt in ordered_types]

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * n_groups), 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    for x, values, color in zip(positions, groups, colors):
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        mean = float(np.mean(arr))
        sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        ax.bar(
            x, mean, width=0.6,
            color=(*_hex_to_rgb(color), 0.55),
            edgecolor=color,
            linewidth=1.2,
        )
        if sem > 0:
            ax.errorbar(
                x, mean, yerr=sem,
                fmt='none', ecolor='white', elinewidth=1.5, capsize=4,
            )
        jitter = _RNG.uniform(-0.18, 0.18, size=len(arr))
        ax.scatter(
            x + jitter, arr,
            s=28, color=color, alpha=0.85, zorder=3, edgecolors='none',
        )

    _style_cross_neuron_ax(ax, ordered_types, positions, ylabel or metric_name)
    ax.set_title(title or metric_name, color='white', fontsize=12)

    y_top_data = _data_y_top(groups, ax)
    ax.set_ylim(ax.get_ylim()[0], y_top_data * 1.05)
    _kruskal_wallis_text(ax, [list(g) for g in groups])
    _pairwise_mannwhitney_brackets(ax, positions, [list(g) for g in groups], y_top=y_top_data)

    ax.set_xlim(-0.6, n_groups - 0.4)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.88, bottom=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-condition renderer
# ---------------------------------------------------------------------------


def render_cross_neuron_comparison_multicondition(
    neuron_data: dict[str, dict[str, list[float]]],
    metric_name: str,
    condition_labels: list[str],
    output_path: Path,
    *,
    ylabel: str = "",
    title: str = "",
) -> None:
    """Render a grouped bar+strip chart: one group per neuron type, one bar per condition.

    Parameters
    ----------
    neuron_data:
        ``{neuron_type: {condition: [value, ...]}}`` — outer key = neuron type,
        inner key = condition label (must match *condition_labels*).
    metric_name:
        Metric identifier used in the figure title when *title* is empty.
    condition_labels:
        Ordered list of condition names.  Must be non-empty.
    output_path:
        Absolute path where the PNG will be saved.
    ylabel:
        Y-axis label (defaults to *metric_name* when empty).
    title:
        Figure title (defaults to *metric_name* when empty).
    """
    if not condition_labels:
        raise ValueError(
            "render_cross_neuron_comparison_multicondition: condition_labels must not be empty."
        )

    ordered_types = [nt for nt in NEURON_TYPE_ORDER if nt in neuron_data]
    if not ordered_types:
        raise ValueError(
            f"render_cross_neuron_comparison_multicondition: neuron_data is empty or "
            f"contains no known neuron types. Known types: {NEURON_TYPE_ORDER}."
        )

    n_types = len(ordered_types)
    n_conds = len(condition_labels)
    group_width = 0.8
    bar_width = group_width / n_conds
    offsets = np.linspace(
        -(group_width / 2) + bar_width / 2,
        (group_width / 2) - bar_width / 2,
        n_conds,
    )
    group_centers = list(range(n_types))

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_types), 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    legend_handles: list[mpatches.Patch] = []

    for ci, cond in enumerate(condition_labels):
        alpha = 0.8 - 0.3 * (ci / max(n_conds - 1, 1))

        for ti, nt in enumerate(ordered_types):
            base_color = NEURON_TYPE_COLORS[nt]
            cond_data = neuron_data[nt].get(cond, [])
            arr = np.array(cond_data, dtype=float)
            arr = arr[np.isfinite(arr)]

            x = group_centers[ti] + offsets[ci]
            if len(arr) == 0:
                continue
            mean = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            r, g, b = _hex_to_rgb(base_color)
            bar_color = (r * alpha + (1 - alpha), g * alpha + (1 - alpha), b * alpha + (1 - alpha))
            ax.bar(
                x, mean, width=bar_width * 0.9,
                color=(*bar_color, 0.7),
                edgecolor=base_color,
                linewidth=1.0,
            )
            if sem > 0:
                ax.errorbar(
                    x, mean, yerr=sem,
                    fmt='none', ecolor='white', elinewidth=1.2, capsize=3,
                )
            jitter = _RNG.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(arr))
            ax.scatter(
                x + jitter, arr,
                s=20, color=base_color, alpha=0.85, zorder=3, edgecolors='none',
            )

        legend_handles.append(
            mpatches.Patch(
                facecolor=(alpha, alpha, alpha),
                edgecolor='white',
                linewidth=0.8,
                label=cond,
            )
        )

    _style_cross_neuron_ax(ax, ordered_types, group_centers, ylabel or metric_name)
    ax.set_title(title or metric_name, color='white', fontsize=12)

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        framealpha=0.3,
        facecolor=_BG,
        edgecolor=_SPINE_COLOR,
        labelcolor='white',
        fontsize=8,
    )

    all_groups: list[list[float]] = []
    for nt in ordered_types:
        combined: list[float] = []
        for cond in condition_labels:
            combined.extend(neuron_data[nt].get(cond, []))
        all_groups.append(combined)

    y_top_data = _data_y_top(all_groups, ax)
    ax.set_ylim(ax.get_ylim()[0], y_top_data * 1.05)
    _kruskal_wallis_text(ax, all_groups)
    _pairwise_mannwhitney_brackets(ax, group_centers, all_groups, y_top=y_top_data)

    ax.set_xlim(-0.6, n_types - 0.4)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.88, bottom=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"_hex_to_rgb: expected '#RRGGBB', got '{hex_color}'")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _data_y_top(groups: list[list[float]], ax) -> float:
    """Return a y-top value just above the tallest data point."""
    all_vals: list[float] = []
    for g in groups:
        arr = np.array(g, dtype=float)
        all_vals.extend(arr[np.isfinite(arr)].tolist())
    if not all_vals:
        return ax.get_ylim()[1]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    return max(all_vals) + 0.05 * y_range
