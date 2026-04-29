# clustering/feature_space_renderer.py
"""
GMM 2-D feature-space renderer.

Produces a scatter plot of touches colored by cluster_label, overlaid with
each GMM component's 1σ and 2σ covariance ellipses. Axes are in physical
units: the renderer marginalises the K-component GMM from the full scaled
feature space onto the chosen 2-D subspace, then inverse-transforms to
original units via the StandardScaler affine.

Usage (standalone, e.g. from a notebook)::

    from analysis.touch_analytics.clustering.feature_space_renderer import (
        render_gmm_feature_space,
    )
    render_gmm_feature_space(result_df, metadata, "pressure_mean",
                             "hand_velocity_y_mean", Path("feature_space.png"))
"""

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

import numpy as np
import pandas as pd

# Use a colormap that works well on a dark background
_CMAP = plt.colormaps['tab10']


def render_gmm_feature_space(
    result_df: pd.DataFrame,
    metadata: dict,
    x_feature: str,
    y_feature: str,
    output_path: Path,
) -> None:
    """
    Write a 2-D feature-space PNG for a GMM clustering result.

    The 2-D Gaussian parameters are obtained by analytically marginalising the
    K-component GMM onto the chosen subspace, then inverse-transforming via the
    per-axis affine of the StandardScaler::

        indices = [retained_columns.index(x_feature),
                   retained_columns.index(y_feature)]
        μ_2d_scaled = means[k][indices]
        Σ_2d_scaled = covariances[k][np.ix_(indices, indices)]

        s = scaler_scale[indices]
        m = scaler_mean[indices]
        μ_2d_orig = μ_2d_scaled * s + m
        Σ_2d_orig = diag(s) @ Σ_2d_scaled @ diag(s)

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain columns ``cluster_label``, *x_feature*, *y_feature*.
    metadata : dict
        Must contain ``'means'``, ``'covariances'``, ``'scaler_mean'``,
        ``'scaler_scale'``, ``'retained_columns'``, ``'k'``,
        ``'covariance_type'``, ``'bic_scores'``.
    x_feature : str
        Column name for the X axis (physical units).
    y_feature : str
        Column name for the Y axis (physical units).
    output_path : Path
        Destination PNG file.
    """
    means = np.array(metadata['means'], dtype=float)        # K × D
    covariances = np.array(metadata['covariances'], dtype=float)  # K × D × D
    scaler_mean = np.array(metadata['scaler_mean'], dtype=float)  # D
    scaler_scale = np.array(metadata['scaler_scale'], dtype=float)  # D
    retained_columns: List[str] = metadata['retained_columns']
    k: int = metadata['k']
    covariance_type: str = metadata.get('covariance_type', 'unknown')
    bic_scores: dict = metadata.get('bic_scores', {})
    best_bic = bic_scores.get(str(k), float('nan'))
    if isinstance(best_bic, str):
        try:
            best_bic = float(best_bic)
        except ValueError:
            best_bic = float('nan')

    mus_2d, sigmas_2d = _marginalise_2d(
        means, covariances, scaler_mean, scaler_scale,
        retained_columns, x_feature, y_feature,
    )

    # ------------------------------------------------------------------ #
    # Data points
    # ------------------------------------------------------------------ #
    x_data = result_df[x_feature].to_numpy(dtype=float)
    y_data = result_df[y_feature].to_numpy(dtype=float)
    cluster_labels = result_df['cluster_label'].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    legend_patches = []
    for ci in range(k):
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

    # ------------------------------------------------------------------ #
    # Ellipses and component means
    # ------------------------------------------------------------------ #
    for ci in range(k):
        color = _CMAP(ci % 10)
        mu = mus_2d[ci]
        sigma = sigmas_2d[ci]

        # Component mean marker
        ax.scatter(
            [mu[0]], [mu[1]],
            marker='x', s=80, linewidths=1.5, color=color,
            zorder=5,
        )

        for n_sigma, ls in ((1, 'solid'), (2, 'dashed')):
            try:
                ellipse = _ellipse_from_cov(mu, sigma, n_sigma)
                ellipse.set_edgecolor(color)
                ellipse.set_facecolor('none')
                ellipse.set_linewidth(1.0)
                ellipse.set_linestyle(ls)
                ellipse.set_zorder(4)
                ax.add_patch(ellipse)
            except ValueError as exc:
                logging.warning(
                    f"render_gmm_feature_space: skipping {n_sigma}σ ellipse "
                    f"for component {ci}: {exc}"
                )

    ax.set_xlabel(x_feature, fontsize=10)
    ax.set_ylabel(y_feature, fontsize=10)
    bic_str = f'{best_bic:.1f}' if np.isfinite(best_bic) else 'N/A'
    ax.set_title(
        f'gmm — k={k} — cov={covariance_type} — BIC={bic_str}',
        fontsize=11,
    )

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
    logging.info(f"render_gmm_feature_space: wrote {output_path}")


def _marginalise_2d(
    means: np.ndarray,
    covariances: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    retained_columns: List[str],
    x_feature: str,
    y_feature: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalise the K-component GMM onto (x_feature, y_feature) in physical units.

    The marginal of a multivariate Gaussian onto a subset of axes is exact:
    take the sub-vector of the mean and the corresponding sub-matrix of the
    covariance.  Then inverse-transform via the StandardScaler affine::

        μ_orig = μ_scaled * s + m
        Σ_orig = diag(s) @ Σ_scaled @ diag(s)

    Parameters
    ----------
    means : np.ndarray, shape (K, D)
    covariances : np.ndarray, shape (K, D, D)
    scaler_mean : np.ndarray, shape (D,)
    scaler_scale : np.ndarray, shape (D,)
    retained_columns : list of str, length D
    x_feature, y_feature : str

    Returns
    -------
    mus_2d : np.ndarray, shape (K, 2)  — component means in physical units
    sigmas_2d : np.ndarray, shape (K, 2, 2)  — covariance sub-matrices in physical units
    """
    ix = retained_columns.index(x_feature)
    iy = retained_columns.index(y_feature)
    indices = [ix, iy]

    K = means.shape[0]
    mus_2d = np.empty((K, 2))
    sigmas_2d = np.empty((K, 2, 2))

    s = scaler_scale[indices]  # shape (2,)
    m = scaler_mean[indices]   # shape (2,)
    S = np.diag(s)              # 2×2

    for k in range(K):
        mu_scaled = means[k][indices]            # shape (2,)
        cov_scaled = covariances[k][np.ix_(indices, indices)]  # 2×2

        mus_2d[k] = mu_scaled * s + m
        sigmas_2d[k] = S @ cov_scaled @ S

    return mus_2d, sigmas_2d


def _ellipse_from_cov(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sigma: float,
) -> Ellipse:
    """
    Build a matplotlib Ellipse patch for a 2-D Gaussian at n_sigma standard deviations.

    Uses eigendecomposition of *cov*: semi-axes = n_sigma * sqrt(eigenvalues),
    rotation angle from the dominant eigenvector.

    Parameters
    ----------
    mu : array-like, shape (2,)
        Centre of the ellipse in data coordinates.
    cov : np.ndarray, shape (2, 2)
        2-D covariance matrix (must be positive semi-definite).
    n_sigma : float
        Number of standard deviations for the ellipse boundary.

    Returns
    -------
    matplotlib.patches.Ellipse

    Raises
    ------
    ValueError
        If any eigenvalue is negative (non-PSD matrix) or nearly zero
        (< 1e-12), indicating a degenerate or numerically singular covariance.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if np.any(eigenvalues < -1e-10):
        raise ValueError(
            f"_ellipse_from_cov: covariance matrix is not positive semi-definite "
            f"(min eigenvalue={eigenvalues.min():.3e})."
        )
    if np.any(eigenvalues < 1e-12):
        raise ValueError(
            f"_ellipse_from_cov: covariance matrix has near-zero eigenvalue "
            f"({eigenvalues.min():.3e}), ellipse would be degenerate."
        )

    # Eigenvectors are columns; sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Angle of the dominant eigenvector (degrees, measured from +x axis)
    angle_deg = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    width = 2.0 * n_sigma * np.sqrt(eigenvalues[0])
    height = 2.0 * n_sigma * np.sqrt(eigenvalues[1])

    return Ellipse(
        xy=(float(mu[0]), float(mu[1])),
        width=width,
        height=height,
        angle=angle_deg,
    )
