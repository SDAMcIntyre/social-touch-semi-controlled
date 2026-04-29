"""Unit tests for the GMM 2-D feature-space renderer."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Stub analysis.touch_analytics to avoid heavy __init__ imports
for _pkg in ('analysis', 'analysis.touch_analytics', 'analysis.touch_analytics.clustering'):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _parts = _pkg.split('.')
        _pkg_dir = _SRC / Path(*_parts)
        if _pkg_dir.exists():
            _mod.__path__ = [str(_pkg_dir)]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod
    if '.' in _pkg:
        _parent_name, _child_name = _pkg.rsplit('.', 1)
        setattr(sys.modules[_parent_name], _child_name, sys.modules[_pkg])

from analysis.touch_analytics.clustering.feature_space_renderer import (  # noqa: E402
    _ellipse_from_cov,
    _marginalise_2d,
    render_gmm_feature_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    K: int = 3,
    D: int = 4,
    seed: int = 0,
) -> dict:
    """Build a minimal valid metadata dict for K components in D dimensions."""
    rng = np.random.default_rng(seed)
    # K means, each D-dimensional, in roughly unit range
    means = rng.uniform(-1, 1, size=(K, D))
    # K positive-definite D×D covariances built from A @ A.T
    covs = []
    for _ in range(K):
        A = rng.standard_normal((D, D)) * 0.3
        covs.append((A @ A.T + np.eye(D) * 0.1).tolist())

    retained = ['pressure_mean', 'hand_velocity_x_mean', 'hand_velocity_y_mean', 'hand_velocity_z_mean']
    scaler_mean = rng.uniform(0, 2, size=D)
    scaler_scale = rng.uniform(0.5, 2.0, size=D)

    return {
        'k': K,
        'covariance_type': 'full',
        'bic_scores': {str(K): -1234.5},
        'means': means.tolist(),
        'covariances': covs,
        'scaler_mean': scaler_mean.tolist(),
        'scaler_scale': scaler_scale.tolist(),
        'retained_columns': retained,
    }


def _make_result_df(n: int = 120, K: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'cluster_label': rng.integers(0, K, size=n),
        'pressure_mean': rng.uniform(0, 5, size=n),
        'hand_velocity_x_mean': rng.uniform(-2, 2, size=n),
        'hand_velocity_y_mean': rng.uniform(-2, 2, size=n),
        'hand_velocity_z_mean': rng.uniform(-2, 2, size=n),
    })


# ---------------------------------------------------------------------------
# _marginalise_2d
# ---------------------------------------------------------------------------

def test_marginalise_2d_recovers_submatrix():
    """_marginalise_2d must extract the exact 2×2 sub-covariance and transform means."""
    K, D = 2, 4
    rng = np.random.default_rng(1)
    means = rng.standard_normal((K, D))
    # build PD covariances
    covs = np.array([
        (lambda A: A @ A.T + np.eye(D) * 0.1)(rng.standard_normal((D, D)) * 0.5)
        for _ in range(K)
    ])
    scaler_mean = rng.uniform(0, 2, size=D)
    scaler_scale = rng.uniform(0.5, 2.0, size=D)
    retained = ['a', 'b', 'c', 'd']

    mus_2d, sigmas_2d = _marginalise_2d(
        means, covs, scaler_mean, scaler_scale, retained, 'a', 'c'
    )

    # Check shape
    assert mus_2d.shape == (K, 2)
    assert sigmas_2d.shape == (K, 2, 2)

    # Verify inverse-transform of mean: mu_orig = mu_scaled * s + m  (for axes 0 and 2)
    for k in range(K):
        expected_mu = means[k, [0, 2]] * scaler_scale[[0, 2]] + scaler_mean[[0, 2]]
        np.testing.assert_allclose(mus_2d[k], expected_mu, atol=1e-10)

    # Verify covariance sub-matrix: Sigma_orig = diag(s) @ Sigma_scaled[ix,iy] @ diag(s)
    for k in range(K):
        s = scaler_scale[[0, 2]]
        expected_cov = np.diag(s) @ covs[k][np.ix_([0, 2], [0, 2])] @ np.diag(s)
        np.testing.assert_allclose(sigmas_2d[k], expected_cov, atol=1e-10)


def test_marginalise_2d_round_trip_identity_scaler():
    """With mean=0 and scale=1 (identity scaler), 2D cov must equal the raw sub-matrix."""
    K, D = 2, 4
    rng = np.random.default_rng(2)
    means = rng.standard_normal((K, D))
    covs = np.array([
        (lambda A: A @ A.T + np.eye(D) * 0.1)(rng.standard_normal((D, D)) * 0.5)
        for _ in range(K)
    ])
    scaler_mean = np.zeros(D)
    scaler_scale = np.ones(D)
    retained = ['p', 'vx', 'vy', 'vz']

    mus_2d, sigmas_2d = _marginalise_2d(
        means, covs, scaler_mean, scaler_scale, retained, 'vx', 'vy'
    )

    for k in range(K):
        np.testing.assert_allclose(mus_2d[k], means[k, [1, 2]], atol=1e-10)
        np.testing.assert_allclose(sigmas_2d[k], covs[k][np.ix_([1, 2], [1, 2])], atol=1e-10)


# ---------------------------------------------------------------------------
# _ellipse_from_cov
# ---------------------------------------------------------------------------

def test_ellipse_axis_aligned_diagonal():
    """Axis-aligned diagonal cov → angle≈0, semi-axes match n_sigma * sqrt(diag)."""
    cov = np.diag([4.0, 1.0])  # larger variance on x
    mu = np.array([1.0, 2.0])
    ellipse = _ellipse_from_cov(mu, cov, n_sigma=1)

    # Semi-axes (half-widths): sqrt(4) * 1 = 2, sqrt(1) * 1 = 1
    np.testing.assert_allclose(ellipse.width, 2 * 2.0, atol=1e-8)
    np.testing.assert_allclose(ellipse.height, 2 * 1.0, atol=1e-8)
    # Angle: dominant eigenvector points along x (eigenvalue 4 > 1)
    np.testing.assert_allclose(abs(ellipse.angle) % 180, 0.0, atol=1e-6)


def test_ellipse_n_sigma_scaling():
    """Semi-axis length scales linearly with n_sigma."""
    cov = np.diag([1.0, 1.0])
    mu = np.array([0.0, 0.0])
    e1 = _ellipse_from_cov(mu, cov, n_sigma=1)
    e2 = _ellipse_from_cov(mu, cov, n_sigma=2)
    np.testing.assert_allclose(e2.width, 2 * e1.width, atol=1e-8)
    np.testing.assert_allclose(e2.height, 2 * e1.height, atol=1e-8)


def test_ellipse_negative_eigenvalue_raises():
    """Non-PSD covariance must raise ValueError."""
    cov = np.array([[1.0, 5.0], [5.0, 1.0]])  # off-diagonal larger → not PSD
    with pytest.raises(ValueError, match="positive semi-definite"):
        _ellipse_from_cov(np.array([0.0, 0.0]), cov, n_sigma=1)


def test_ellipse_near_zero_eigenvalue_raises():
    """Near-zero eigenvalue (degenerate) must raise ValueError."""
    cov = np.zeros((2, 2))  # all-zero covariance
    with pytest.raises(ValueError, match="near-zero eigenvalue"):
        _ellipse_from_cov(np.array([0.0, 0.0]), cov, n_sigma=1)


# ---------------------------------------------------------------------------
# render_gmm_feature_space — integration
# ---------------------------------------------------------------------------

def test_render_writes_non_empty_png(tmp_path):
    """Renderer must write a non-empty PNG for valid synthetic 3-cluster data."""
    meta = _make_metadata(K=3, D=4)
    df = _make_result_df(n=120, K=3)
    out = tmp_path / 'feature_space.png'

    render_gmm_feature_space(
        result_df=df,
        metadata=meta,
        x_feature='pressure_mean',
        y_feature='hand_velocity_y_mean',
        output_path=out,
    )

    assert out.exists(), "PNG was not written"
    assert out.stat().st_size > 1000, "PNG is suspiciously small"


def test_render_missing_feature_key_raises(tmp_path):
    """KeyError from _marginalise_2d when a requested feature is not in retained_columns."""
    meta = _make_metadata(K=2, D=4)
    df = _make_result_df(n=60, K=2)
    # retained_columns doesn't include 'nonexistent_feature'
    with pytest.raises((KeyError, ValueError)):
        render_gmm_feature_space(
            result_df=df,
            metadata=meta,
            x_feature='pressure_mean',
            y_feature='nonexistent_feature',
            output_path=tmp_path / 'out.png',
        )


def test_render_single_cluster(tmp_path):
    """Renderer handles K=1 (early-exit placeholder case) without error."""
    meta = _make_metadata(K=1, D=4)
    df = _make_result_df(n=30, K=1)
    out = tmp_path / 'single_cluster.png'

    render_gmm_feature_space(
        result_df=df,
        metadata=meta,
        x_feature='pressure_mean',
        y_feature='hand_velocity_y_mean',
        output_path=out,
    )

    assert out.exists()
