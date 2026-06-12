"""Pure rendering functions for IFF tuning curve plots."""

import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import NamedTuple


class BinResult(NamedTuple):
    bin_centers: np.ndarray      # shape (n_bins,)
    bin_low: np.ndarray          # shape (n_bins,)
    bin_high: np.ndarray         # shape (n_bins,)
    mean_iff: np.ndarray         # shape (n_bins,) — NaN for empty bins; sum when bin_agg="sum"
    std_iff: np.ndarray          # shape (n_bins,) — NaN if count < 2; always NaN when bin_agg="sum"
    counts: np.ndarray           # shape (n_bins,) int
    level_counts: dict[str, np.ndarray]   # level -> shape (n_bins,) int
    level_props: dict[str, np.ndarray]    # level -> shape (n_bins,) float in [0,1]


_BG = '#1e1e1e'
_AX_BG = '#2d2d2d'
_SPINE_COLOR = '#555555'
_IFF_COLOR = '#4ec9b0'
_COUNT_COLOR = '#569cd6'

_CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "contact_area_metadata": "Contact Area Level",
    "speed_metadata":        "Speed Level",
    "force_metadata":        "Force Level",
    "type_metadata":         "Touch Type",
}


def _cat_display(col_name: str) -> str:
    return _CATEGORY_DISPLAY_NAMES.get(col_name, col_name)


_DISPLAY_NAMES: dict[str, str] = {
    "contact_area_mean":            "Contact Area (mm²)",
    "contact_depth_mean":           "Depth (mm)",
    "pressure_mean":                "Pressure (N/mm²)",
    "hand_velocity_amplitude_mean": "Hand Velocity (mm/s)",
    "hand_velocity_signed_mean":    "Signed Velocity (mm/s)",
    "hand_velocity_x_mean":         "Velocity X (mm/s)",
    "hand_velocity_y_mean":         "Velocity Y (mm/s)",
    "hand_velocity_z_mean":         "Velocity Z (mm/s)",
    "hand_acceleration_x_mean":     "Accel. X (mm/s²)",
    "hand_acceleration_y_mean":     "Accel. Y (mm/s²)",
    "hand_acceleration_z_mean":     "Accel. Z (mm/s²)",
    "mos_strain_mean":              "Strain",
    "mos_stress_kpa_mean":          "Stress (kPa)",
    "mos_strain_rate_mean":         "Strain Rate (1/s)",
    "mos_elastic_energy_mj_mean":   "Elastic E. (mJ)",
    "mos_impulse_mns_mean":         "Impulse (mN·s)",
}


def _display_id(feature_name: str) -> str:
    return _DISPLAY_NAMES.get(feature_name, feature_name)


def _style_dark_ax(ax, bg_color: str = '#2d2d2d') -> None:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)


_BIN_AGG_VALUES = ("mean", "sum")


def _bin_data(
    df: pd.DataFrame,
    feature_col: str,
    iff_col: str,
    bin_low: np.ndarray,
    bin_high: np.ndarray,
    category_col: str | None = None,
    category_levels: list | None = None,
    bin_agg: str = "mean",
) -> BinResult:
    """Bin *df* rows by *feature_col* and aggregate *iff_col* per bin.

    Parameters
    ----------
    bin_agg:
        Aggregation to apply to *iff_col* within each bin.
        ``"mean"`` → per-bin mean ± STD (existing behaviour).
        ``"sum"``  → per-bin sum; STD is left as NaN (no error bars).

    Raises
    ------
    ValueError
        If *feature_col* or *iff_col* is missing, or *bin_agg* is unknown.
    """
    if bin_agg not in _BIN_AGG_VALUES:
        raise ValueError(
            f"_bin_data: unknown bin_agg={bin_agg!r}. "
            f"Expected one of {_BIN_AGG_VALUES}."
        )
    if feature_col not in df.columns:
        raise ValueError(
            f"_bin_data: '{feature_col}' not in df.columns. "
            f"Available: {sorted(df.columns)}"
        )
    if iff_col not in df.columns:
        raise ValueError(
            f"_bin_data: '{iff_col}' not in df.columns. "
            f"Available: {sorted(df.columns)}"
        )

    valid = df[[feature_col, iff_col]].dropna()
    feature_vals = valid[feature_col].to_numpy(dtype=float)
    iff_vals = valid[iff_col].to_numpy(dtype=float)

    n_bins = len(bin_low)
    bin_centers = 0.5 * (bin_low + bin_high)
    response_per_bin = np.full(n_bins, np.nan)
    std_per_bin = np.full(n_bins, np.nan)
    count_per_bin = np.zeros(n_bins, dtype=int)

    use_categories = (
        category_col is not None
        and category_levels is not None
        and len(category_levels) > 0
    )

    if use_categories:
        level_counts: dict[str, np.ndarray] = {
            level: np.zeros(n_bins, dtype=int) for level in category_levels
        }
        level_counts["unassigned"] = np.zeros(n_bins, dtype=int)
        if category_col in df.columns:
            cat_series = df[category_col]
        else:
            cat_series = pd.Series(
                np.full(len(df), None, dtype=object), index=df.index
            )
        cat_vals = cat_series.loc[valid.index].to_numpy()
    else:
        level_counts = {}

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (feature_vals >= bin_low[i]) & (feature_vals < bin_high[i])
        else:
            mask = (feature_vals >= bin_low[i]) & (feature_vals <= bin_high[i])

        count_per_bin[i] = int(mask.sum())
        if count_per_bin[i] > 0:
            if bin_agg == "sum":
                response_per_bin[i] = float(np.sum(iff_vals[mask]))
            else:
                response_per_bin[i] = float(np.mean(iff_vals[mask]))
                if count_per_bin[i] >= 2:
                    std_per_bin[i] = float(np.std(iff_vals[mask], ddof=1))

        if use_categories and count_per_bin[i] > 0:
            bin_cat_vals = cat_vals[mask]
            for level in category_levels:
                level_counts[level][i] = int(np.sum(bin_cat_vals == level))
            assigned = sum(level_counts[level][i] for level in category_levels)
            level_counts["unassigned"][i] = count_per_bin[i] - assigned

    if use_categories:
        level_props: dict[str, np.ndarray] = {}
        nonzero = count_per_bin > 0
        for level in category_levels:
            props = np.zeros(n_bins, dtype=float)
            props[nonzero] = level_counts[level][nonzero] / count_per_bin[nonzero]
            level_props[level] = props
        level_counts_out = level_counts
    else:
        level_props = {}
        level_counts_out = {}

    return BinResult(
        bin_centers=bin_centers,
        bin_low=bin_low,
        bin_high=bin_high,
        mean_iff=response_per_bin,
        std_iff=std_per_bin,
        counts=count_per_bin,
        level_counts=level_counts_out,
        level_props=level_props,
    )


def _smooth_nan_aware(values: np.ndarray, sigma: float) -> np.ndarray:
    """Nadaraya-Watson Gaussian smoothing that ignores NaN bins."""
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return values.copy()
    filled = np.where(valid, values, 0.0)
    weights = valid.astype(float)
    smoothed_vals = gaussian_filter1d(filled, sigma=sigma)
    smoothed_weights = gaussian_filter1d(weights, sigma=sigma)
    result = np.full_like(values, np.nan)
    nonzero = smoothed_weights > 0
    result[nonzero] = smoothed_vals[nonzero] / smoothed_weights[nonzero]
    return result


def _make_category_palette(n: int) -> list[str]:
    """Return n hex colours sampled from 'tab10' (n<=10) else 'tab20'."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10' if n <= 10 else 'tab20')
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in (cmap(i / max(n-1, 1)) for i in range(n))]


def render_session_tuning_curve(
    bin_centers: np.ndarray,
    mean_iff: np.ndarray,
    counts: np.ndarray,
    feature_name: str,
    session_id: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    count_ymax: float,
    iff_ylabel: str = "Mean IFF (Hz)",
    smoothing_sigma: float = 0.0,
    std_iff: np.ndarray | None = None,
    level_counts: dict | None = None,
    level_props: dict | None = None,
    category_levels: list | None = None,
    category_display_name: str = "",
    line_color: str = _IFF_COLOR,
) -> None:
    if std_iff is None:
        std_iff = np.full_like(mean_iff, np.nan)
    if level_counts is None:
        level_counts = {}
    if level_props is None:
        level_props = {}

    fig, ax_iff = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    _style_dark_ax(ax_iff)
    ax_iff.grid(alpha=0.15, color='white', linestyle='--')

    ax_count = ax_iff.twinx()
    _style_dark_ax(ax_count)

    bar_width = (bin_centers[1] - bin_centers[0]) * 0.8 if len(bin_centers) > 1 else 1.0

    if category_levels and level_props:
        palette = _make_category_palette(len(category_levels))
        bottoms = np.zeros(len(bin_centers))
        for lv, color in zip(category_levels, palette):
            props = level_props.get(lv, np.zeros(len(bin_centers)))
            heights = counts * props
            ax_count.bar(
                bin_centers, heights,
                bottom=bottoms,
                width=bar_width,
                color=color, alpha=0.65, zorder=2,
                label=str(lv),
            )
            bottoms += heights
        leg_title = category_display_name if category_display_name else "Level"
        leg = ax_count.legend(
            fontsize=7, framealpha=0.3, facecolor=_AX_BG,
            labelcolor='white', title=leg_title,
            title_fontsize=7, loc='upper right',
        )
        leg.get_title().set_color('white')
        current_title = f"{session_id} | {gesture_subset}"
        ax_iff.set_title(f"{current_title} | by {leg_title}", color='white', fontsize=11)
    else:
        ax_count.bar(
            bin_centers, counts,
            width=bar_width,
            color=_COUNT_COLOR, alpha=0.5, zorder=2,
        )
        ax_iff.set_title(f"{session_id} | {gesture_subset}", color='white', fontsize=11)

    ax_count.set_ylim(0, count_ymax)
    ax_count.set_ylabel('Touch count', color='white', fontsize=10)

    valid_mask = ~np.isnan(mean_iff)
    if valid_mask.any():
        yerr = np.where(np.isnan(std_iff[valid_mask]), 0.0, std_iff[valid_mask])
        ax_iff.errorbar(
            bin_centers[valid_mask], mean_iff[valid_mask],
            yerr=yerr,
            fmt='o', capsize=2, color=line_color, alpha=0.85,
            markersize=4, linewidth=1, zorder=4,
        )
        if smoothing_sigma > 0:
            smoothed = _smooth_nan_aware(mean_iff, smoothing_sigma)
            smooth_valid = ~np.isnan(smoothed)
            if smooth_valid.any():
                ax_iff.plot(
                    bin_centers[smooth_valid], smoothed[smooth_valid],
                    color=line_color, linewidth=2, zorder=3, alpha=0.6,
                )

    ax_iff.set_ylim(iff_ylim)
    ax_iff.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax_iff.set_xlabel(_display_id(feature_name), color='white', fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def _fit_polynomial(
    feature_vals: np.ndarray,
    response_vals: np.ndarray,
    fit_degree: int,
) -> tuple[np.ndarray | None, float | None]:
    """Fit a degree-``fit_degree`` polynomial and compute its R².

    Returns ``(coeffs, r2)`` where *coeffs* are the ``np.polyfit`` coefficients
    (highest power first) and *r2* the coefficient of determination.

    Graceful degradation (explicitly requested in plan):
      - Returns ``(None, None)`` when fewer than ``fit_degree + 1`` valid rows
        exist — caller renders dots only with ``R²=n/a``.
      - Returns ``r2 = 0.0`` when ``ss_tot == 0`` (e.g. all-constant response).
    """
    n = len(feature_vals)
    if n < fit_degree + 1:
        # Too few points to determine the polynomial — skip the fit (dots only).
        return None, None

    coeffs = np.polyfit(feature_vals, response_vals, fit_degree)
    predicted = np.polyval(coeffs, feature_vals)
    ss_res = float(np.sum((response_vals - predicted) ** 2))
    ss_tot = float(np.sum((response_vals - np.mean(response_vals)) ** 2))
    if ss_tot == 0.0:
        # Constant response: R² is undefined; report 0.0 by convention.
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return coeffs, r2


def render_session_raw_dots(
    feature_vals: np.ndarray,
    response_vals: np.ndarray,
    feature_name: str,
    session_id: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    iff_ylabel: str = "IFF (Hz)",
    fit_degree: int = 1,
    dot_alpha: float = 0.35,
    show_fit_ci: bool = False,
    line_color: str = _IFF_COLOR,
    secondary_vals: "np.ndarray | None" = None,
    secondary_label: str = "",
    secondary_cmap: str = "viridis",
) -> None:
    """Per-session raw-touch scatter + polynomial fit line.

    Every touch is plotted as a semi-transparent dot at its exact
    ``(feature_value, response_value)`` coordinate. A degree-``fit_degree``
    polynomial regression line is drawn through the points. The total touch
    count N and the fit R² are reported in the title (there are no bins, so no
    right-axis count bars are drawn).
    """
    feature_vals = np.asarray(feature_vals, dtype=float)
    response_vals = np.asarray(response_vals, dtype=float)

    # Drop NaN pairs so the fit only sees complete observations.
    valid_mask = ~(np.isnan(feature_vals) | np.isnan(response_vals))
    feat = feature_vals[valid_mask]
    resp = response_vals[valid_mask]
    n = int(feat.size)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    if secondary_vals is not None and len(secondary_vals) == len(feat):
        sc = ax.scatter(
            feat, resp,
            s=20, c=secondary_vals, cmap=secondary_cmap,
            alpha=dot_alpha, edgecolors='none', zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        if secondary_label:
            cbar.set_label(_display_id(secondary_label), color='white')
    else:
        ax.scatter(
            feat, resp,
            s=20, color=line_color, alpha=dot_alpha,
            edgecolors='none', zorder=3,
        )

    coeffs, r2 = _fit_polynomial(feat, resp, fit_degree)
    if coeffs is not None:
        r2_text = f"{r2:.2f}"
        x_line = np.linspace(float(np.min(feat)), float(np.max(feat)), 200)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color=line_color, linewidth=2.5, zorder=5)

        # CI band: closed-form, degree-1 only. For higher degrees the simple
        # standard-error formula does not apply, so we warn and disable it
        # gracefully (explicitly requested in plan).
        if show_fit_ci and fit_degree > 1:
            warnings.warn(
                "render_session_raw_dots: show_fit_ci is only supported for "
                f"fit_degree=1; got fit_degree={fit_degree}. Disabling the CI "
                "band.",
                stacklevel=2,
            )
        elif show_fit_ci and fit_degree == 1 and n >= 3:
            predicted = np.polyval(coeffs, feat)
            residuals = resp - predicted
            dof = n - 2
            s_err = np.sqrt(np.sum(residuals ** 2) / dof)
            x_mean = float(np.mean(feat))
            ss_xx = float(np.sum((feat - x_mean) ** 2))
            if ss_xx > 0:
                # 95% CI for the mean response (t≈1.96 large-sample approx).
                se_line = s_err * np.sqrt(
                    1.0 / n + (x_line - x_mean) ** 2 / ss_xx
                )
                ci = 1.96 * se_line
                ax.fill_between(
                    x_line, y_line - ci, y_line + ci,
                    color=line_color, alpha=0.2, zorder=4, linewidth=0,
                )
    else:
        r2_text = "n/a"

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_display_id(feature_name), color='white', fontsize=10)
    ax.set_title(
        f"{session_id} | {gesture_subset} | N={n}, R²={r2_text}",
        color='white', fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overlay_raw_dots(
    session_data: "dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]",
    feature_name: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    session_colors: dict[str, str],
    iff_ylabel: str = "IFF (Hz)",
    fit_degree: int = 1,
    dot_alpha: float = 0.15,
    show_fit_ci: bool = False,
    legend_mode: str = "by_session",
    session_neuron_types: dict[str, str] | None = None,
    type_colors: dict[str, str] | None = None,
    secondary_label: str = "",
    secondary_cmap: str = "viridis",
) -> None:
    """Multi-session raw-touch scatter + per-session fit lines.

    Each session contributes a faint scatter cloud (``alpha=dot_alpha``,
    ``s=20``) and a bold polynomial fit line (``linewidth=2.5``) in its
    neuron-type / session colour. The legend block mirrors
    ``render_overlay_tuning_curve`` (``by_type`` / ``by_session`` modes).
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)
    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    # Compute shared colorbar range from pooled secondary values across sessions.
    sec_arrays = [tup[2] for tup in session_data.values() if len(tup) > 2 and tup[2] is not None]
    shared_sec_norm = None
    if sec_arrays and secondary_label:
        all_sec = np.concatenate([s for s in sec_arrays if len(s) > 0])
        all_sec_finite = all_sec[np.isfinite(all_sec)]
        if len(all_sec_finite) > 0:
            sec_vmin = float(np.min(all_sec_finite))
            sec_vmax = float(np.max(all_sec_finite))
            if sec_vmax > sec_vmin:
                shared_sec_norm = plt.Normalize(vmin=sec_vmin, vmax=sec_vmax)

    for session_id, session_tup in session_data.items():
        feature_vals, response_vals = session_tup[0], session_tup[1]
        sec_vals = session_tup[2] if len(session_tup) > 2 else None
        color = session_colors[session_id]
        if legend_mode == "by_type" and session_neuron_types is not None:
            line_label = session_neuron_types[session_id]
        else:
            line_label = session_id

        feat = np.asarray(feature_vals, dtype=float)
        resp = np.asarray(response_vals, dtype=float)
        valid_mask = ~(np.isnan(feat) | np.isnan(resp))
        feat = feat[valid_mask]
        resp = resp[valid_mask]
        if feat.size == 0:
            continue

        use_secondary = (
            shared_sec_norm is not None
            and sec_vals is not None
            and len(sec_vals) == len(feat)
        )
        if use_secondary:
            ax.scatter(
                feat, resp,
                s=20, c=sec_vals, cmap=secondary_cmap, norm=shared_sec_norm,
                alpha=dot_alpha, edgecolors='none', zorder=2,
            )
        else:
            ax.scatter(
                feat, resp,
                s=20, color=color, alpha=dot_alpha,
                edgecolors='none', zorder=2,
            )

        coeffs, _ = _fit_polynomial(feat, resp, fit_degree)
        if coeffs is not None:
            x_line = np.linspace(float(np.min(feat)), float(np.max(feat)), 200)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(
                x_line, y_line,
                color=color, linewidth=2.5, label=line_label, zorder=3,
            )

    if shared_sec_norm is not None:
        sm = plt.cm.ScalarMappable(cmap=secondary_cmap, norm=shared_sec_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        if secondary_label:
            cbar.set_label(_display_id(secondary_label), color='white')

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_display_id(feature_name), color='white', fontsize=10)
    ax.set_title(f"All sessions | {gesture_subset}", color='white', fontsize=11)

    if legend_mode == "by_type" and type_colors is not None:
        handles = [
            Line2D([0], [0], color=color, linewidth=2, label=ntype)
            for ntype, color in type_colors.items()
        ]
        legend = ax.legend(handles=handles, title="Neuron Type", fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')
        legend.get_title().set_color('white')
    else:
        legend = ax.legend(title="Session", fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')
        legend.get_title().set_color('white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overlay_tuning_curve(
    session_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_name: str,
    gesture_subset: str,
    out_path: Path,
    iff_ylim: tuple[float, float],
    session_colors: dict[str, str],
    iff_ylabel: str = "Mean IFF (Hz)",
    smoothing_sigma: float = 0.0,
    legend_mode: str = "by_session",
    session_neuron_types: dict[str, str] | None = None,
    type_colors: dict[str, str] | None = None,
) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    for session_id, (bin_centers, mean_iff, std_iff) in session_data.items():
        color = session_colors[session_id]
        if legend_mode == "by_type" and session_neuron_types is not None:
            line_label = session_neuron_types[session_id]
        else:
            line_label = session_id
        valid_mask = ~np.isnan(mean_iff)
        if valid_mask.any():
            yerr = np.where(np.isnan(std_iff[valid_mask]), 0.0, std_iff[valid_mask])
            ax.errorbar(
                bin_centers[valid_mask], mean_iff[valid_mask],
                yerr=yerr,
                fmt='none', color=color, alpha=0.2, capsize=0, linewidth=0.8, zorder=2,
            )
            if smoothing_sigma > 0:
                ax.scatter(
                    bin_centers[valid_mask], mean_iff[valid_mask],
                    color=color, alpha=0.3, s=12, zorder=2,
                )
                smoothed = _smooth_nan_aware(mean_iff, smoothing_sigma)
                smooth_valid = ~np.isnan(smoothed)
                if smooth_valid.any():
                    ax.plot(
                        bin_centers[smooth_valid], smoothed[smooth_valid],
                        color=color, linewidth=2, label=line_label, zorder=3,
                    )
            else:
                ax.plot(
                    bin_centers[valid_mask], mean_iff[valid_mask],
                    color=color, linewidth=2, label=line_label,
                )

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_display_id(feature_name), color='white', fontsize=10)
    ax.set_title(f"All sessions | {gesture_subset}", color='white', fontsize=11)

    if legend_mode == "by_type" and type_colors is not None:
        handles = [
            Line2D([0], [0], color=color, linewidth=2, label=ntype)
            for ntype, color in type_colors.items()
        ]
        legend = ax.legend(handles=handles, title="Neuron Type", fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')
        legend.get_title().set_color('white')
    else:
        legend = ax.legend(title="Session", fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')
        legend.get_title().set_color('white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
