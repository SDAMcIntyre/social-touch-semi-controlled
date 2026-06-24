"""Pure rendering functions for RF spatial tuning scatter plots.

Renders scatter plots of RF boundary metrics (area_mm2, circularity,
pca_aspect_ratio, pca_orientation_deg) as a function of a binned stimulus
parameter (velocity, depth, contact_area).

One figure per (session_id, gesture_subset, tuning_feature) with 4 subplots.
Cross-session overlay renders one figure per (gesture_subset, tuning_feature)
combining all sessions coloured by neuron type.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

from analysis.receptive_field_mapping.rendering.fit_models import fit_model as _fit_model
from analysis.receptive_field_mapping.rendering.rf_response_tuning_renderer import (
    _BG,
    _AX_BG,
    _SPINE_COLOR,
    _FIT_COLORS,
    _FIT_LINESTYLES,
    _display_id,
)


# ---------------------------------------------------------------------------
# RF metric registry
# ---------------------------------------------------------------------------

_RF_METRICS = [
    ("area_mm2",            "RF Area (mm²)"),
    ("circularity",         "Circularity"),
    ("pca_aspect_ratio",    "PCA Aspect Ratio"),
    ("pca_orientation_deg", "PCA Orientation (°)"),
]

_RF_METRIC_KEYS = [m[0] for m in _RF_METRICS]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _style_dark_ax(ax) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)


def _spearman_annotation(x: np.ndarray, y: np.ndarray) -> str:
    """Return a Spearman r / p annotation string for valid (non-NaN) pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return f"r=n/a, p=n/a (N={n})"
    r, p = stats.spearmanr(x[mask], y[mask])
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return f"r={r:.2f}, p={p_str} (N={n})"


def _draw_scatter_subplot(
    ax,
    bin_centers: np.ndarray,
    metric_vals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    fit_models: "list[str] | None" = None,
    fit_degrees: "list[int] | None" = None,
    dot_color: str = '#4ec9b0',
    dot_alpha: float = 0.7,
    multi_session: bool = False,
    metric_stds: "np.ndarray | None" = None,
) -> None:
    """Draw a single RF-metric vs bin-center scatter subplot with fit line(s).

    Parameters
    ----------
    ax:
        Target matplotlib axes.
    bin_centers:
        X values (one per bin).
    metric_vals:
        Y values for this metric (NaN for skipped bins).
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.
    title:
        Axes title.
    fit_models:
        List of named model strings to overlay (preferred).
    fit_degrees:
        Legacy list of polynomial degrees (converted to model names).
    dot_color:
        Colour for scatter dots and (single-model) fit line.
    dot_alpha:
        Scatter dot opacity.
    multi_session:
        When True, draw fit lines with distinct colors from _FIT_COLORS
        regardless of how many models are requested.
    metric_stds:
        Per-bin standard deviations for error bars (same length as
        *bin_centers*).  NaN entries are silently skipped.
    """
    if fit_models is None:
        if fit_degrees is not None:
            fit_models = [f"poly{d}" for d in fit_degrees]
        else:
            fit_models = ["poly1"]

    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    valid = np.isfinite(bin_centers) & np.isfinite(metric_vals)
    x = bin_centers[valid]
    y = metric_vals[valid]

    ax.scatter(x, y, s=30, color=dot_color, alpha=dot_alpha, edgecolors='none', zorder=3)

    if metric_stds is not None:
        valid_stds = metric_stds[valid]
        std_mask = np.isfinite(valid_stds)
        if std_mask.any():
            ax.errorbar(
                x[std_mask], y[std_mask], yerr=valid_stds[std_mask],
                fmt='none', ecolor=dot_color, alpha=dot_alpha * 0.7,
                capsize=3, capthick=1, elinewidth=1, zorder=2,
            )

    multi = len(fit_models) > 1 or multi_session
    r2_lines: list[str] = []
    x_line = np.linspace(float(x.min()), float(x.max()), 200) if len(x) > 0 else None

    for i, model_name in enumerate(fit_models):
        fit_color = _FIT_COLORS[i % len(_FIT_COLORS)] if multi else dot_color
        fit_ls = _FIT_LINESTYLES[i % len(_FIT_LINESTYLES)] if multi else '-'

        result = _fit_model(x, y, model_name)
        if result.params is not None and x_line is not None:
            r2_lines.append(f"{result.display_label}: R²={result.r_squared:.2f}")
            ax.plot(x_line, result.evaluate(x_line),
                    color=fit_color, linewidth=2.0, linestyle=fit_ls, zorder=5)
        else:
            r2_lines.append(f"{result.display_label}: R²=n/a")

    spearman_str = _spearman_annotation(x, y)

    annotation_parts = r2_lines if multi else []
    annotation_parts.append(spearman_str)
    annotation = "\n".join(annotation_parts)

    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes,
        fontsize=7, verticalalignment='top',
        color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.7,
                  edgecolor='#555555'),
    )

    ax.set_xlabel(xlabel, color='white', fontsize=9)
    ax.set_ylabel(ylabel, color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=9)


# ---------------------------------------------------------------------------
# Per-session renderer
# ---------------------------------------------------------------------------


def render_session_spatial_tuning(
    session_id: str,
    bins_data: list[dict],
    tuning_feature: str,
    gesture_subset: str,
    out_path: Path,
    fit_models: "list[str] | None" = None,
    fit_degrees: "list[int] | None" = None,
    dot_alpha: float = 0.7,
    line_color: str = '#4ec9b0',
) -> None:
    """Render a 4-subplot figure of RF metrics vs binned stimulus parameter for one session.

    Parameters
    ----------
    session_id:
        Session identifier for the figure title.
    bins_data:
        List of dicts, one per valid bin, each with keys:
            ``bin_center`` (float),
            ``area_mm2`` (float),
            ``circularity`` (float),
            ``pca_aspect_ratio`` (float),
            ``pca_orientation_deg`` (float).
        Bins with no inflection boundary should be omitted from the list.
    tuning_feature:
        Column name of the stimulus feature used for binning (X axis).
    gesture_subset:
        Gesture subset label (used in the figure title).
    out_path:
        Destination PNG file path.
    fit_models:
        Named model strings to overlay on each subplot (preferred).
    fit_degrees:
        Legacy polynomial degrees (converted to model names).
    dot_alpha:
        Scatter dot opacity (default 0.7).
    line_color:
        Hex color string for scatter dots and single-model fit lines.

    Raises
    ------
    ValueError
        If ``bins_data`` is empty (caller should skip rendering for sessions
        with no valid bins).
    """
    if not bins_data:
        raise ValueError(
            f"render_session_spatial_tuning: bins_data is empty for session "
            f"'{session_id}', feature '{tuning_feature}', subset '{gesture_subset}'. "
            f"Caller must skip rendering when no bins yielded an inflection boundary."
        )

    bin_centers = np.array([b["bin_center"] for b in bins_data], dtype=float)
    metric_arrays: dict[str, np.ndarray] = {
        key: np.array([b[key] for b in bins_data], dtype=float)
        for key in _RF_METRIC_KEYS
    }
    std_arrays: dict[str, np.ndarray] = {
        key: np.array([b.get(f"{key}_std", float("nan")) for b in bins_data], dtype=float)
        for key in _RF_METRIC_KEYS
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        f"{session_id} | {gesture_subset} | RF metrics vs {_display_id(tuning_feature)}",
        color='white', fontsize=11, y=1.01,
    )

    for ax, (key, ylabel) in zip(axes.ravel(), _RF_METRICS):
        _draw_scatter_subplot(
            ax=ax,
            bin_centers=bin_centers,
            metric_vals=metric_arrays[key],
            xlabel=_display_id(tuning_feature),
            ylabel=ylabel,
            title=ylabel,
            fit_models=fit_models,
            fit_degrees=fit_degrees,
            dot_color=line_color,
            dot_alpha=dot_alpha,
            metric_stds=std_arrays[key],
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-session overlay renderer
# ---------------------------------------------------------------------------


def render_overlay_spatial_tuning(
    all_sessions_data: dict[str, list[dict]],
    tuning_feature: str,
    gesture_subset: str,
    out_path: Path,
    fit_models: "list[str] | None" = None,
    fit_degrees: "list[int] | None" = None,
    session_colors: dict[str, str] = None,
    dot_alpha: float = 0.4,
    legend_mode: str = "by_session",
    session_neuron_types: dict[str, str] | None = None,
    type_colors: dict[str, str] | None = None,
) -> None:
    """Render a 4-subplot cross-session overlay of RF metrics vs binned stimulus parameter.

    Each session contributes a scatter cloud plus fit line(s) in its neuron-type
    colour. Spearman annotation is computed per session and placed as a small
    text block using the session color.

    Parameters
    ----------
    all_sessions_data:
        Dict mapping session_id -> list of per-bin dicts (same format as
        ``render_session_spatial_tuning``). Sessions with empty lists are skipped.
    tuning_feature:
        Column name of the stimulus feature (X axis).
    gesture_subset:
        Gesture subset label.
    out_path:
        Destination PNG file path.
    fit_models:
        Named model strings to overlay per session (preferred).
    fit_degrees:
        Legacy polynomial degrees (converted to model names).
    session_colors:
        session_id -> hex color string.
    dot_alpha:
        Scatter dot opacity (default 0.4).
    legend_mode:
        ``"by_session"`` or ``"by_type"`` -- controls the legend.
    session_neuron_types:
        session_id -> neuron type string; required when ``legend_mode="by_type"``.
    type_colors:
        neuron_type -> hex color string; required when ``legend_mode="by_type"``.

    Raises
    ------
    ValueError
        If ``all_sessions_data`` is entirely empty.
    """
    if fit_models is None:
        if fit_degrees is not None:
            fit_models = [f"poly{d}" for d in fit_degrees]
        else:
            fit_models = ["poly1"]

    non_empty = {sid: data for sid, data in all_sessions_data.items() if data}
    if not non_empty:
        raise ValueError(
            f"render_overlay_spatial_tuning: all_sessions_data is empty for "
            f"feature '{tuning_feature}', subset '{gesture_subset}'."
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        f"All sessions | {gesture_subset} | RF metrics vs {_display_id(tuning_feature)}",
        color='white', fontsize=11, y=1.01,
    )

    multi = len(fit_models) > 1
    legend_labels_seen: set[str] = set()

    for ax_idx, (ax, (key, ylabel)) in enumerate(zip(axes.ravel(), _RF_METRICS)):
        _style_dark_ax(ax)
        ax.grid(alpha=0.15, color='white', linestyle='--')
        ax.set_xlabel(_display_id(tuning_feature), color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.set_title(ylabel, color='white', fontsize=9)

        for session_id, bins_data in non_empty.items():
            color = session_colors[session_id]
            if legend_mode == "by_type" and session_neuron_types is not None:
                line_label = session_neuron_types[session_id]
            else:
                line_label = session_id

            bin_centers = np.array([b["bin_center"] for b in bins_data], dtype=float)
            metric_vals = np.array([b[key] for b in bins_data], dtype=float)
            metric_stds = np.array(
                [b.get(f"{key}_std", float("nan")) for b in bins_data], dtype=float,
            )

            valid = np.isfinite(bin_centers) & np.isfinite(metric_vals)
            x = bin_centers[valid]
            y = metric_vals[valid]

            if len(x) == 0:
                continue

            ax.scatter(x, y, s=20, color=color, alpha=dot_alpha,
                       edgecolors='none', zorder=2)

            valid_stds = metric_stds[valid]
            std_mask = np.isfinite(valid_stds)
            if std_mask.any():
                ax.errorbar(
                    x[std_mask], y[std_mask], yerr=valid_stds[std_mask],
                    fmt='none', ecolor=color, alpha=dot_alpha * 0.7,
                    capsize=2, capthick=0.8, elinewidth=0.8, zorder=1,
                )

            x_line = np.linspace(float(x.min()), float(x.max()), 200)
            r2_lines: list[str] = []

            for i, model_name in enumerate(fit_models):
                fit_ls = _FIT_LINESTYLES[i % len(_FIT_LINESTYLES)] if multi else '-'
                result = _fit_model(x, y, model_name)
                if result.params is not None:
                    y_line = result.evaluate(x_line)
                    label_key = f"{line_label} ({result.display_label})" if multi else line_label
                    plot_label = label_key if (ax_idx == 0 and label_key not in legend_labels_seen) else "_nolegend_"
                    ax.plot(x_line, y_line, color=color, linewidth=2.0,
                            linestyle=fit_ls, label=plot_label, zorder=3)
                    if plot_label != "_nolegend_":
                        legend_labels_seen.add(label_key)
                    r2_lines.append(f"{result.display_label}: R²={result.r_squared:.2f}")
                else:
                    r2_lines.append(f"{result.display_label}: R²=n/a")

            spearman_str = _spearman_annotation(x, y)
            annotation = f"{session_id}\n" + "\n".join(r2_lines) + f"\n{spearman_str}"
            ax.text(
                0.03, 0.97, annotation,
                transform=ax.transAxes,
                fontsize=5.5, verticalalignment='top',
                color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222222',
                          alpha=0.55, edgecolor=color),
            )

    if legend_mode == "by_type" and type_colors is not None:
        handles = [
            Line2D([0], [0], color=c, linewidth=2, label=nt)
            for nt, c in type_colors.items()
        ]
        for ax in axes.ravel():
            legend = ax.legend(handles=handles, title="Neuron Type", fontsize=7,
                               framealpha=0.3, facecolor=_AX_BG, labelcolor='white',
                               loc='upper right')
            legend.get_title().set_color('white')
    else:
        for ax in axes.ravel():
            legend = ax.legend(title="Session", fontsize=7, framealpha=0.3,
                               facecolor=_AX_BG, labelcolor='white', loc='upper right')
            legend.get_title().set_color('white')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
