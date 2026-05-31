"""Pure rendering functions for IFF instruction tuning bar charts."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import NamedTuple

from analysis.receptive_field_mapping.rendering.rf_iff_tuning_renderer import (
    _cat_display,
    _style_dark_ax,
    _BG,
    _AX_BG,
    _SPINE_COLOR,
    _IFF_COLOR,
)


class CategoryResult(NamedTuple):
    category_labels: list[str]   # ordered list of all global category labels
    mean_iff: np.ndarray          # shape (n_categories,) — NaN for empty categories
    std_iff: np.ndarray           # shape (n_categories,) — NaN if count < 2
    counts: np.ndarray            # shape (n_categories,) int


def _group_by_category(df, category_col: str, iff_col: str, global_levels: list[str]) -> CategoryResult:
    """Group df by a categorical metadata column and compute per-level IFF statistics.

    Parameters
    ----------
    df:
        DataFrame containing at least ``category_col`` and ``iff_col``.
    category_col:
        Column name of the instruction-level categorical column
        (e.g. ``"contact_area_metadata"``).
    iff_col:
        Column name for the IFF response values (e.g. ``"Nerve_freq_mean"``).
    global_levels:
        Ordered list of ALL category values across ALL sessions.  Entries absent
        from this session are returned with NaN mean/std and zero count.

    Returns
    -------
    CategoryResult
        Entries are in the same order as ``global_levels``.

    Raises
    ------
    ValueError
        If ``category_col`` is not in ``df.columns``, ``iff_col`` is not in
        ``df.columns``, or ``global_levels`` is empty.
    """
    if category_col not in df.columns:
        raise ValueError(
            f"_group_by_category: '{category_col}' not in df.columns. "
            f"Available: {sorted(df.columns)}"
        )
    if iff_col not in df.columns:
        raise ValueError(
            f"_group_by_category: '{iff_col}' not in df.columns. "
            f"Available: {sorted(df.columns)}"
        )
    if not global_levels:
        raise ValueError("_group_by_category: global_levels must not be empty.")

    n = len(global_levels)
    mean_iff = np.full(n, np.nan)
    std_iff = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)

    for i, label in enumerate(global_levels):
        mask = df[category_col] == label
        subset = df.loc[mask, iff_col].dropna()
        count = len(subset)
        counts[i] = count
        if count == 0:
            pass  # mean and std remain NaN
        elif count == 1:
            mean_iff[i] = float(subset.iloc[0])
            # std remains NaN
        else:
            mean_iff[i] = float(subset.mean())
            std_iff[i] = float(subset.std(ddof=1))

    return CategoryResult(
        category_labels=list(global_levels),
        mean_iff=mean_iff,
        std_iff=std_iff,
        counts=counts,
    )


def render_session_instruction_tuning(
    category_result: CategoryResult,
    category_col: str,
    iff_col_name: str,
    session_id: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    iff_ylabel: str = "Mean IFF (Hz)",
) -> None:
    """Render a single-axis bar chart of mean IFF per instruction level for one session.

    Parameters
    ----------
    category_result:
        Grouped statistics from ``_group_by_category``.
    category_col:
        Column name used for grouping — drives the X-axis label.
    iff_col_name:
        IFF column name (unused visually; kept for forward compatibility).
    session_id:
        Session identifier — used in the plot title.
    gesture_subset:
        Gesture subset label — used in the plot title.
    out_path:
        Absolute path where the PNG will be saved.  Parent directories are
        created automatically.
    iff_ylim:
        ``(ymin, ymax)`` for the IFF Y-axis.
    iff_ylabel:
        Y-axis label text.
    """
    labels = category_result.category_labels
    mean_iff = category_result.mean_iff
    std_iff = category_result.std_iff
    counts = category_result.counts
    n_cats = len(labels)
    x_pos = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(6, n_cats * 1.2), 5), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    # Draw bars (height = mean IFF; zero height for empty categories).
    bar_heights = np.where(np.isnan(mean_iff), 0.0, mean_iff)
    ax.bar(x_pos, bar_heights, color=_IFF_COLOR, alpha=0.75, zorder=2)

    # Overlay error bars — use nan_to_num so positions with count < 2 get yerr=0.
    yerr = np.nan_to_num(std_iff, nan=0.0)
    # Only draw error bars where there is a valid mean.
    valid_mask = ~np.isnan(mean_iff)
    if valid_mask.any():
        ax.errorbar(
            x_pos[valid_mask],
            mean_iff[valid_mask],
            yerr=yerr[valid_mask],
            fmt='none',
            color='white',
            alpha=0.7,
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )

    # Count annotations above each bar.
    for i, (cnt, bar_h) in enumerate(zip(counts, bar_heights)):
        if cnt == 0:
            continue
        annotation_y = bar_h + yerr[i] + (iff_ylim[1] - iff_ylim[0]) * 0.02
        ax.text(
            x_pos[i],
            annotation_y,
            f"n={cnt}",
            ha='center',
            va='bottom',
            color='white',
            fontsize=8,
            zorder=4,
        )

    # Axes formatting.
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    if n_cats > 4:
        ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_cat_display(category_col), color='white', fontsize=10)
    ax.set_title(f"{session_id} | {gesture_subset}", color='white', fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overlay_instruction_tuning(
    session_category_data: dict[str, CategoryResult],
    category_col: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    session_colors: dict[str, tuple],
    iff_ylabel: str = "Mean IFF (Hz)",
) -> None:
    """Render a cross-session jittered dot plot for instruction-level IFF tuning.

    Each session is plotted as jittered dots (horizontal jitter, ±0.15) at each
    categorical X position, with vertical error bars showing ±1 STD.  Sessions
    are color-coded and a legend is included.

    Parameters
    ----------
    session_category_data:
        Mapping from ``session_id`` to its ``CategoryResult``.  All results must
        share the same ``category_labels`` ordering (i.e. derived from the same
        ``global_levels`` list).
    category_col:
        Column name used for grouping — drives the X-axis label.
    gesture_subset:
        Gesture subset label — used in the plot title.
    out_path:
        Absolute path where the PNG will be saved.  Parent directories are
        created automatically.
    iff_ylim:
        ``(ymin, ymax)`` for the shared IFF Y-axis.
    session_colors:
        Mapping from ``session_id`` to an RGBA tuple (from
        ``assign_session_colors``).
    iff_ylabel:
        Y-axis label text.
    """
    if not session_category_data:
        raise ValueError("render_overlay_instruction_tuning: session_category_data must not be empty.")

    # Derive global category labels from the first session (all share the same ordering).
    first_result = next(iter(session_category_data.values()))
    labels = first_result.category_labels
    n_cats = len(labels)
    x_pos = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(6, n_cats * 1.2), 5), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    rng = np.random.default_rng(42)

    for session_id, cat_result in session_category_data.items():
        color = session_colors[session_id]
        mean_iff = cat_result.mean_iff
        std_iff = cat_result.std_iff

        jitter = rng.uniform(-0.15, 0.15, size=n_cats)
        x_jittered = x_pos.astype(float) + jitter

        valid_mask = ~np.isnan(mean_iff)
        if not valid_mask.any():
            continue

        yerr = np.nan_to_num(std_iff, nan=0.0)

        # Error bars (vertical) with low alpha.
        ax.errorbar(
            x_jittered[valid_mask],
            mean_iff[valid_mask],
            yerr=yerr[valid_mask],
            fmt='none',
            color=color,
            alpha=0.3,
            capsize=0,
            linewidth=0.8,
            zorder=2,
        )

        # Dots with session label (only label the first valid dot to avoid legend duplication).
        ax.scatter(
            x_jittered[valid_mask],
            mean_iff[valid_mask],
            color=color,
            alpha=0.85,
            s=30,
            zorder=3,
            label=session_id,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    if n_cats > 4:
        ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_cat_display(category_col), color='white', fontsize=10)
    ax.set_title(f"All sessions | {gesture_subset}", color='white', fontsize=11)

    legend = ax.legend(fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
