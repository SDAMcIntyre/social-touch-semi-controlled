# clustering/cartesian_binning_renderer.py
"""
Cartesian-binning 2-D feature-space renderer.

Produces a scatter plot of touches coloured by cluster_label, overlaid with
the bin-edge grid for the two chosen features.  Occupied cells stand out from
the grid; empty cells remain dark, making sparsity immediately visible.

Usage (standalone, e.g. from a notebook)::

    from analysis.touch_analytics.clustering.cartesian_binning_renderer import (
        render_cartesian_bin_partition,
    )
    render_cartesian_bin_partition(result_df, metadata, "pressure_mean",
                                   "hand_velocity_y_mean", Path("feature_space.png"))
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd

_CMAP = plt.colormaps['tab10']
_LOG_SKEWNESS_THRESHOLD = 2.0


def _should_use_log_scale(data: np.ndarray) -> bool:
    finite = data[np.isfinite(data)]
    if len(finite) < 3:
        raise ValueError(
            "_should_use_log_scale: need >= 3 finite values to compute skewness, "
            f"got {len(finite)}."
        )
    if np.any(finite <= 0):
        return False
    mean = finite.mean()
    m2 = np.mean((finite - mean) ** 2)
    if m2 < 1e-12:
        return False
    m3 = np.mean((finite - mean) ** 3)
    skewness = m3 / (m2 ** 1.5)
    return bool(skewness > _LOG_SKEWNESS_THRESHOLD)


def render_cartesian_bin_partition(
    result_df: pd.DataFrame,
    metadata: dict,
    x_feature: str,
    y_feature: str,
    output_path: Path,
) -> None:
    """
    Write a 2-D feature-space PNG for a cartesian-binning clustering result.

    The scatter is overlaid with the bin-edge grid for *x_feature* and
    *y_feature* so the partitioning is directly visible.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain columns ``cluster_label``, *x_feature*, *y_feature*.
    metadata : dict
        Must contain ``'bin_edges'``, ``'binned_features'``, ``'n_clusters'``,
        ``'bin_method'``, ``'n_bins_per_feature'``.
    x_feature : str
        Column name for the X axis.
    y_feature : str
        Column name for the Y axis.
    output_path : Path
        Destination PNG file.

    Raises
    ------
    ValueError
        If ``metadata['binned_features']`` is empty.
    KeyError
        If *x_feature* or *y_feature* is not present in ``metadata['bin_edges']``.
    """
    binned_features: list = metadata.get('binned_features', [])
    if not binned_features:
        raise ValueError(
            "render_cartesian_bin_partition: metadata['binned_features'] is empty — "
            "no columns were binned, nothing to render."
        )

    bin_edges: dict = metadata['bin_edges']
    for feat in (x_feature, y_feature):
        if feat not in bin_edges:
            raise KeyError(
                f"render_cartesian_bin_partition: feature '{feat}' not found in "
                f"bin_edges. Available: {list(bin_edges)}."
            )

    n_clusters: int = metadata['n_clusters']
    bin_method: str = metadata.get('bin_method', 'equal_width')
    n_bins_per_feature: dict = metadata.get('n_bins_per_feature', {})
    n_x = len(bin_edges[x_feature]) - 1
    n_y = len(bin_edges[y_feature]) - 1

    x_data = result_df[x_feature].to_numpy(dtype=float)
    y_data = result_df[y_feature].to_numpy(dtype=float)
    cluster_labels = result_df['cluster_label'].to_numpy()

    x_log = _should_use_log_scale(x_data)
    y_log = _should_use_log_scale(y_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    # Draw bin-edge grid before scatter so points render on top
    grid_style = dict(color='gray', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
    for edge in bin_edges[x_feature]:
        ax.axvline(x=edge, **grid_style)
    for edge in bin_edges[y_feature]:
        ax.axhline(y=edge, **grid_style)

    # Draw outlier boundary lines
    outlier_info: dict = metadata.get('outlier_info', {})
    outlier_style = dict(color='red', linewidth=1.2, linestyle=':', alpha=0.7, zorder=1.5)
    if x_feature in outlier_info:
        oi = outlier_info[x_feature]
        if oi['n_low_outliers'] > 0:
            ax.axvline(x=oi['lower_bound'], **outlier_style)
        if oi['n_high_outliers'] > 0:
            ax.axvline(x=oi['upper_bound'], **outlier_style)
    if y_feature in outlier_info:
        oi = outlier_info[y_feature]
        if oi['n_low_outliers'] > 0:
            ax.axhline(y=oi['lower_bound'], **outlier_style)
        if oi['n_high_outliers'] > 0:
            ax.axhline(y=oi['upper_bound'], **outlier_style)

    legend_patches = []
    for ci in range(n_clusters):
        mask = cluster_labels == ci
        color = _CMAP(ci % 10)
        n_pts = int(mask.sum())
        ax.scatter(
            x_data[mask], y_data[mask],
            color=color, alpha=0.5, s=12, linewidths=0, zorder=2,
        )
        legend_patches.append(
            mpatches.Patch(color=color, label=f'cluster {ci} (n={n_pts})')
        )

    ax.set_xlabel(x_feature, fontsize=10)
    ax.set_ylabel(y_feature, fontsize=10)
    scale_parts = []
    if x_log:
        scale_parts.append('x=log')
    if y_log:
        scale_parts.append('y=log')
    scale_suffix = f' — scale: {", ".join(scale_parts)}' if scale_parts else ''

    outlier_method_str = metadata.get('outlier_method')
    outlier_suffix = f' — outliers: {outlier_method_str}' if outlier_method_str else ''

    ax.set_title(
        f'cartesian_binning — {n_clusters} clusters — {n_x}×{n_y} bins — {bin_method}{scale_suffix}{outlier_suffix}',
        fontsize=11,
    )

    _LEGEND_MAX_CLUSTERS = 10
    if n_clusters <= _LEGEND_MAX_CLUSTERS:
        ax.legend(
            handles=legend_patches,
            fontsize=8,
            framealpha=0.3,
            labelcolor='white',
            facecolor='black',
            edgecolor='gray',
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    logging.info(f"render_cartesian_bin_partition: wrote {output_path}")
