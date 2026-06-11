import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from pathlib import Path


def _nan_safe_correlation_distance(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            row_i = matrix[i]
            row_j = matrix[j]
            mask = ~(np.isnan(row_i) | np.isnan(row_j))
            if mask.sum() < 2:
                d = np.nansum((row_i - row_j) ** 2) ** 0.5
            else:
                r_i = row_i[mask]
                r_j = row_j[mask]
                std_i = r_i.std()
                std_j = r_j.std()
                if std_i == 0 or std_j == 0:
                    d = 0.0
                else:
                    corr = np.corrcoef(r_i, r_j)[0, 1]
                    d = 1.0 - corr
            dist[i, j] = dist[j, i] = d
    from scipy.spatial.distance import squareform
    return squareform(dist)


def render_boundary_contour_overlay(
    contours: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    gesture_type: str,
    output_path: Path,
    uv_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    if not contours:
        raise ValueError("render_boundary_contour_overlay: contours dict is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmap = matplotlib.colormaps['tab20']
    except AttributeError:
        cmap = matplotlib.cm.get_cmap('tab20')

    session_ids = sorted(contours.keys())
    n = len(session_ids)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, session_id in enumerate(session_ids):
        color = cmap(i / max(n, 1))
        contour = contours[session_id]
        closed_contour = np.vstack([contour, contour[:1]])
        ax.plot(closed_contour[:, 0], closed_contour[:, 1], color=color, lw=1.5, label=session_id)
        if session_id in centroids:
            centroid = centroids[session_id]
            ax.plot(centroid[0], centroid[1], marker='o', ms=5, color=color)

    ax.set_xlabel('UV u')
    ax.set_ylabel('UV v')
    ax.set_title(f'RF Boundary Contour Overlay — {gesture_type}')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    if uv_limits is not None:
        ax.set_xlim(*uv_limits[0])
        ax.set_ylim(*uv_limits[1])
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def render_boundary_metric_panels(
    df: pd.DataFrame,
    gesture_type: str,
    metrics: list[str],
    output_path: Path,
    metric_limits: dict[str, tuple[float, float]] | None = None,
    session_colors: dict[str, str] | None = None,
    neuron_type_legend: dict[str, str] | None = None,
) -> None:
    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("render_boundary_metric_panels: metrics list is empty")

    n_cols = 3
    n_rows = ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        values = df[metric_name].to_numpy(dtype=float)
        session_ids = df['session_id'].tolist()
        if session_colors is not None:
            bar_colors = [session_colors.get(sid, 'steelblue') for sid in session_ids]
        else:
            bar_colors = 'steelblue'
        ax.bar(range(len(session_ids)), values, color=bar_colors)
        ax.set_xticks(range(len(session_ids)))
        ax.set_xticklabels(session_ids, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric_name, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=8)
        if metric_limits is not None and metric_name in metric_limits:
            ax.set_ylim(*metric_limits[metric_name])
        if neuron_type_legend is not None and idx == 0:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=color, label=label)
                for label, color in neuron_type_legend.items()
            ]
            ax.legend(handles=legend_handles, fontsize=7, loc='upper right')

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Boundary Metrics — {gesture_type}', fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def render_session_gesture_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    cluster_sessions: bool = True,
) -> None:
    matrix = df.pivot_table(
        index='session_id',
        columns='gesture_type',
        values=metric_name,
        aggfunc='first',
    )

    fig_w = max(6, 1.5 * matrix.shape[1])
    fig_h = max(5, 0.6 * matrix.shape[0])

    use_dendrogram = cluster_sessions and matrix.shape[0] >= 3

    if use_dendrogram:
        import scipy.cluster.hierarchy
        import scipy.spatial.distance

        distance_matrix = _nan_safe_correlation_distance(matrix.to_numpy())
        linkage = scipy.cluster.hierarchy.linkage(distance_matrix, method='average')
        row_order = scipy.cluster.hierarchy.leaves_list(linkage)
        matrix = matrix.iloc[row_order]

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.05)
        ax_dendro = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        scipy.cluster.hierarchy.dendrogram(
            linkage,
            orientation='left',
            ax=ax_dendro,
            color_threshold=0,
            above_threshold_color='gray',
            labels=None,
            no_labels=True,
        )
        ax_dendro.set_axis_off()
    else:
        fig, ax_heat = plt.subplots(figsize=(fig_w, fig_h))

    data = matrix.to_numpy(dtype=float)
    im = ax_heat.pcolormesh(data, cmap='viridis')
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    ax_heat.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax_heat.set_xticklabels(matrix.columns.tolist(), rotation=30, ha='right', fontsize=8)

    if use_dendrogram:
        ax_heat.set_yticks([])
    else:
        ax_heat.set_yticks(np.arange(matrix.shape[0]) + 0.5)
        ax_heat.set_yticklabels(matrix.index.tolist(), fontsize=7)

    ax_heat.set_title(f'{metric_name}', fontsize=11)

    fig.suptitle(f'Session × Gesture Heatmap: {metric_name}', fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
