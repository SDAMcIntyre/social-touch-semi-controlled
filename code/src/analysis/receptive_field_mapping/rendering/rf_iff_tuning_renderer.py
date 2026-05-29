"""Pure rendering functions for IFF tuning curve plots."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


_BG = '#1e1e1e'
_AX_BG = '#2d2d2d'
_SPINE_COLOR = '#555555'
_IFF_COLOR = '#4ec9b0'
_COUNT_COLOR = '#569cd6'

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
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_iff_per_bin = np.full(n_bins, np.nan)
    count_per_bin = np.zeros(n_bins, dtype=int)

    bin_indices = np.digitize(feature_vals, bin_edges) - 1

    for i in range(n_bins):
        mask = bin_indices == i
        count_per_bin[i] = int(mask.sum())
        if count_per_bin[i] > 0:
            mean_iff_per_bin[i] = float(np.mean(iff_vals[mask]))

    return bin_centers, mean_iff_per_bin, count_per_bin


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
) -> None:
    fig, ax_iff = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_BG)

    _style_dark_ax(ax_iff)
    ax_iff.grid(alpha=0.15, color='white', linestyle='--')

    ax_count = ax_iff.twinx()
    _style_dark_ax(ax_count)

    ax_count.bar(
        bin_centers,
        counts,
        width=(bin_centers[1] - bin_centers[0]) * 0.8 if len(bin_centers) > 1 else 1.0,
        color=_COUNT_COLOR,
        alpha=0.5,
        zorder=2,
    )
    ax_count.set_ylim(0, count_ymax)
    ax_count.set_ylabel('Touch count', color='white', fontsize=10)

    valid_mask = ~np.isnan(mean_iff)
    if valid_mask.any():
        if smoothing_sigma > 0:
            ax_iff.scatter(
                bin_centers[valid_mask], mean_iff[valid_mask],
                color=_IFF_COLOR, alpha=0.4, s=18, zorder=3,
            )
            smoothed = _smooth_nan_aware(mean_iff, smoothing_sigma)
            smooth_valid = ~np.isnan(smoothed)
            if smooth_valid.any():
                ax_iff.plot(
                    bin_centers[smooth_valid], smoothed[smooth_valid],
                    color=_IFF_COLOR, linewidth=2, zorder=4,
                )
        else:
            ax_iff.plot(
                bin_centers[valid_mask], mean_iff[valid_mask],
                color=_IFF_COLOR, linewidth=2, zorder=3,
            )
    ax_iff.set_ylim(iff_ylim)
    ax_iff.set_ylabel(iff_ylabel, color='white', fontsize=10)
    ax_iff.set_xlabel(_display_id(feature_name), color='white', fontsize=10)
    ax_iff.set_title(f"{session_id} | {gesture_subset}", color='white', fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_overlay_tuning_curve(
    session_data: dict[str, tuple[np.ndarray, np.ndarray]],
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

    for session_id, (bin_centers, mean_iff) in session_data.items():
        color = session_colors[session_id]
        valid_mask = ~np.isnan(mean_iff)
        if valid_mask.any():
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
