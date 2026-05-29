"""Pure rendering functions for cross-session touch feature comparison plots."""

import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


_BG = '#1a1a1a'
_SPINE_COLOR = '#555555'
_GRID_COLOR = '#444444'


_DATE_PREFIX_RE = re.compile(r'^\d{4}-\d{2}-\d{2}_')


def _display_id(session_id: str) -> str:
    return _DATE_PREFIX_RE.sub('', session_id)


def assign_session_colors(session_ids: list[str]) -> dict[str, tuple]:
    """Map each session_id to a distinct colour from tab20."""
    if not session_ids:
        raise ValueError("assign_session_colors: session_ids must not be empty")
    cmap = cm.get_cmap('tab20')
    n = max(len(session_ids), 1)
    return {sid: cmap(i / n) for i, sid in enumerate(session_ids)}


def _draw_feature_on_ax(
    ax,
    df_gesture: pd.DataFrame,
    feature_col: str,
    session_ids: list[str],
    colors: dict[str, tuple],
    plot_type: Literal['box_strip', 'violin', 'bar_error'],
) -> None:
    rng = np.random.default_rng(42)

    for i, sid in enumerate(session_ids):
        subset = df_gesture[df_gesture['session_id'] == sid][feature_col]
        values = subset.dropna().to_numpy()

        if plot_type == 'box_strip':
            if len(values) == 0:
                continue
            ax.boxplot(
                values,
                positions=[i],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=(*colors[sid][:3], 0.4), edgecolor=colors[sid]),
                medianprops=dict(color=colors[sid], linewidth=2),
                whiskerprops=dict(color='#aaaaaa'),
                capprops=dict(color='#aaaaaa'),
                flierprops=dict(marker='', linestyle='none'),
            )
            jitter = rng.uniform(-0.2, 0.2, size=len(values))
            ax.scatter(
                i + jitter,
                values,
                alpha=0.5,
                s=15,
                color=colors[sid],
                zorder=3,
            )

        elif plot_type == 'violin':
            if len(values) < 2:
                continue
            vparts = ax.violinplot(
                [values],
                positions=[i],
                widths=0.6,
                showmedians=True,
                showextrema=False,
            )
            vparts['bodies'][0].set_facecolor(colors[sid])
            vparts['bodies'][0].set_alpha(0.5)
            vparts['cmedians'].set_color(colors[sid])

        elif plot_type == 'bar_error':
            if len(values) == 0:
                continue
            mean = np.nanmean(values)
            std = np.nanstd(values)
            ax.bar(
                i,
                mean,
                width=0.6,
                color=(*colors[sid][:3], 0.6),
                edgecolor=colors[sid],
                linewidth=1.2,
            )
            ax.errorbar(
                i,
                mean,
                yerr=std,
                fmt='none',
                ecolor='white',
                elinewidth=1.5,
                capsize=4,
            )

        else:
            raise ValueError(f"Unknown plot_type: {plot_type!r}")


def _annotate_session_counts(
    ax,
    df_gesture: pd.DataFrame,
    feature_col: str,
    session_ids: list[str],
) -> None:
    ylo, yhi = ax.get_ylim()
    offset = 0.03 * (yhi - ylo)
    for i, sid in enumerate(session_ids):
        values = df_gesture[df_gesture['session_id'] == sid][feature_col].dropna()
        if values.empty:
            continue
        ax.text(
            i, float(values.max()) + offset, str(len(values)),
            ha='center', va='bottom',
            fontsize=11, color='#cccccc',
            clip_on=False,
        )


def _style_ax(ax, session_ids: list[str], display_label: str, ylim: tuple[float, float] | None) -> None:
    ax.set_xticks(range(len(session_ids)))
    ax.set_xticklabels([_display_id(s) for s in session_ids], rotation=45, ha='right', fontsize=8, color='white')
    ax.set_xlim(-0.5, len(session_ids) - 0.5)
    ax.set_ylabel(display_label, color='white', fontsize=10)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    for s in ax.spines:
        ax.spines[s].set_edgecolor(_SPINE_COLOR)
    ax.grid(axis='y', color=_GRID_COLOR, linestyle='--', alpha=0.5)
    ax.set_facecolor(_BG)
    if ylim is not None:
        ax.set_ylim(ylim)


def render_feature_session_comparison(
    df: pd.DataFrame,
    feature_col: str,
    display_label: str,
    session_ids: list[str],
    colors: dict[str, tuple],
    plot_type: Literal['box_strip', 'violin', 'bar_error'],
    output_path: Path,
    gesture_type: str = 'all',
    ylim: tuple[float, float] | None = None,
) -> None:
    """Render a single feature comparison figure across all sessions."""
    if feature_col not in df.columns:
        raise ValueError(
            f"render_feature_session_comparison: '{feature_col}' not in df.columns"
        )

    if gesture_type == 'all':
        df_gesture = df
    else:
        df_gesture = df[df['gesture_type'] == gesture_type]

    fig_w = max(6, 0.8 * len(session_ids))
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=150)
    fig.patch.set_facecolor(_BG)

    _draw_feature_on_ax(ax, df_gesture, feature_col, session_ids, colors, plot_type)
    _style_ax(ax, session_ids, display_label, ylim)
    _annotate_session_counts(ax, df_gesture, feature_col, session_ids)
    ax.set_title(f"{gesture_type} | {display_label}", color='white', fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_feature_summary_grid(
    df: pd.DataFrame,
    feature_cols: list[str],
    display_labels: list[str],
    session_ids: list[str],
    colors: dict[str, tuple],
    plot_type: Literal['box_strip', 'violin', 'bar_error'],
    output_path: Path,
    gesture_type: str = 'all',
    ylims: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Render a multi-panel grid: one axes per feature, 3 columns."""
    if not feature_cols:
        raise ValueError("render_feature_summary_grid: feature_cols must not be empty")
    if len(feature_cols) != len(display_labels):
        raise ValueError(
            f"render_feature_summary_grid: feature_cols ({len(feature_cols)}) "
            f"and display_labels ({len(display_labels)}) must have the same length"
        )

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(
                f"render_feature_summary_grid: '{col}' not in df.columns"
            )

    if gesture_type == 'all':
        df_gesture = df
    else:
        df_gesture = df[df['gesture_type'] == gesture_type]

    ncols = 3
    nrows = math.ceil(len(feature_cols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=150)
    fig.patch.set_facecolor(_BG)

    axes_flat = np.array(axes).flatten() if nrows > 1 or ncols > 1 else np.array([axes])

    for idx, (col, label) in enumerate(zip(feature_cols, display_labels)):
        ax = axes_flat[idx]
        _draw_feature_on_ax(ax, df_gesture, col, session_ids, colors, plot_type)
        _style_ax(ax, session_ids, label, ylim=ylims.get(col) if ylims is not None else None)
        _annotate_session_counts(ax, df_gesture, col, session_ids)
        ax.set_title(label, color='white', fontsize=9)

    for idx in range(len(feature_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Cross-session summary | {gesture_type}",
        color='white',
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
