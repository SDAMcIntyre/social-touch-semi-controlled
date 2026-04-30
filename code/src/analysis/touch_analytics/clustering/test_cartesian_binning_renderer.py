# clustering/test_cartesian_binning_renderer.py
import numpy as np
import pandas as pd
import pytest

from .cartesian_binning_renderer import _should_use_log_scale, render_cartesian_bin_partition


def _make_result_df(n=30, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'pressure_mean': rng.uniform(0, 10, n),
        'velocity_mean': rng.uniform(0, 5, n),
        'cluster_label': rng.integers(0, 4, n),
    })


def _make_metadata(result_df):
    p_edges = np.linspace(0, 10, 4).tolist()   # 3 bins
    v_edges = np.linspace(0, 5, 4).tolist()    # 3 bins
    return {
        'algorithm': 'cartesian_binning',
        'binned_features': ['pressure_mean', 'velocity_mean'],
        'bin_edges': {
            'pressure_mean': p_edges,
            'velocity_mean': v_edges,
        },
        'n_clusters': 4,
        'bin_method': 'equal_width',
        'n_bins_per_feature': {'pressure_mean': 3, 'velocity_mean': 3},
    }


def test_render_writes_png(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    out = tmp_path / 'feature_space.png'
    render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_raises_on_empty_binned_features(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    metadata['binned_features'] = []
    with pytest.raises(ValueError, match="binned_features"):
        render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', tmp_path / 'out.png')


def test_render_raises_on_missing_feature_in_bin_edges(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    with pytest.raises(KeyError, match="nonexistent"):
        render_cartesian_bin_partition(result_df, metadata, 'nonexistent', 'velocity_mean', tmp_path / 'out.png')


def test_render_creates_parent_directory(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    out = tmp_path / 'nested' / 'dir' / 'out.png'
    render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', out)
    assert out.exists()


# --- _should_use_log_scale unit tests ----------------------------------------

def test_should_use_log_scale_high_skewness():
    rng = np.random.default_rng(42)
    data = rng.lognormal(0, 2, 500)
    assert _should_use_log_scale(data) is True


def test_should_use_log_scale_symmetric_data():
    rng = np.random.default_rng(42)
    data = rng.normal(5, 1, 500)
    assert _should_use_log_scale(data) is False


def test_should_use_log_scale_with_zeros():
    rng = np.random.default_rng(42)
    data = rng.lognormal(0, 2, 500)
    data[0] = 0.0
    assert _should_use_log_scale(data) is False


def test_should_use_log_scale_with_negatives():
    rng = np.random.default_rng(42)
    data = rng.lognormal(0, 2, 500)
    data[0] = -1.0
    assert _should_use_log_scale(data) is False


def test_should_use_log_scale_constant_data():
    data = np.full(100, 5.0)
    assert _should_use_log_scale(data) is False


def test_should_use_log_scale_too_few_values():
    data = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="3 finite values"):
        _should_use_log_scale(data)


# --- Renderer with log-scale data --------------------------------------------

def test_render_with_log_scale_writes_png(tmp_path):
    rng = np.random.default_rng(42)
    n = 50
    pressure = rng.lognormal(0, 2, n)
    velocity = rng.uniform(1, 10, n)
    result_df = pd.DataFrame({
        'pressure_mean': pressure,
        'velocity_mean': velocity,
        'cluster_label': rng.integers(0, 4, n),
    })
    p_edges = np.geomspace(pressure.min(), pressure.max(), 4).tolist()
    v_edges = np.linspace(velocity.min(), velocity.max(), 4).tolist()
    metadata = {
        'algorithm': 'cartesian_binning',
        'binned_features': ['pressure_mean', 'velocity_mean'],
        'bin_edges': {'pressure_mean': p_edges, 'velocity_mean': v_edges},
        'n_clusters': 4,
        'bin_method': 'equal_frequency',
        'n_bins_per_feature': {'pressure_mean': 3, 'velocity_mean': 3},
    }
    out = tmp_path / 'log_scale.png'
    render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', out)
    assert out.exists()
    assert out.stat().st_size > 0


# --- Renderer with outlier metadata -------------------------------------------

def test_render_with_outlier_info_writes_png(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    metadata['outlier_method'] = 'iqr'
    metadata['outlier_info'] = {
        'pressure_mean': {
            'method': 'iqr',
            'params': {},
            'lower_bound': 1.0,
            'upper_bound': 9.0,
            'n_low_outliers': 2,
            'n_high_outliers': 3,
        },
    }
    out = tmp_path / 'outlier_feature_space.png'
    render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_without_outlier_info_still_works(tmp_path):
    result_df = _make_result_df()
    metadata = _make_metadata(result_df)
    out = tmp_path / 'no_outlier.png'
    render_cartesian_bin_partition(result_df, metadata, 'pressure_mean', 'velocity_mean', out)
    assert out.exists()
