"""Pure rendering functions for IFF tuning curve plots."""

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
    mean_iff: np.ndarray         # shape (n_bins,) — NaN for empty bins
    std_iff: np.ndarray          # shape (n_bins,) — NaN if count < 2
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


def _bin_data(
    df: pd.DataFrame,
    feature_col: str,
    iff_col: str,
    bin_low: np.ndarray,
    bin_high: np.ndarray,
    category_col: str | None = None,
    category_levels: list | None = None,
) -> BinResult:
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
    mean_iff_per_bin = np.full(n_bins, np.nan)
    std_iff_per_bin = np.full(n_bins, np.nan)
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
            mean_iff_per_bin[i] = float(np.mean(iff_vals[mask]))
            if count_per_bin[i] >= 2:
                std_iff_per_bin[i] = float(np.std(iff_vals[mask], ddof=1))

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
        mean_iff=mean_iff_per_bin,
        std_iff=std_iff_per_bin,
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
            fmt='o', capsize=2, color=_IFF_COLOR, alpha=0.85,
            markersize=4, linewidth=1, zorder=4,
        )
        if smoothing_sigma > 0:
            smoothed = _smooth_nan_aware(mean_iff, smoothing_sigma)
            smooth_valid = ~np.isnan(smoothed)
            if smooth_valid.any():
                ax_iff.plot(
                    bin_centers[smooth_valid], smoothed[smooth_valid],
                    color=_IFF_COLOR, linewidth=2, zorder=3, alpha=0.6,
                )

    ax_iff.set_ylim(iff_ylim)
    ax_iff.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax_iff.set_xlabel(_display_id(feature_name), color='white', fontsize=10)

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
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    _style_dark_ax(ax)
    ax.grid(alpha=0.15, color='white', linestyle='--')

    for session_id, (bin_centers, mean_iff, std_iff) in session_data.items():
        color = session_colors[session_id]
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
                        color=color, linewidth=2, label=session_id, zorder=3,
                    )
            else:
                ax.plot(
                    bin_centers[valid_mask], mean_iff[valid_mask],
                    color=color, linewidth=2, label=session_id,
                )

    ax.set_ylim(iff_ylim)
    ax.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax.set_xlabel(_display_id(feature_name), color='white', fontsize=10)
    ax.set_title(f"All sessions | {gesture_subset}", color='white', fontsize=11)

    legend = ax.legend(fontsize=8, framealpha=0.3, facecolor=_AX_BG, labelcolor='white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
