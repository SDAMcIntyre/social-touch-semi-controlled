"""Unit tests for GMMClusterer."""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Stub analysis.touch_analytics to avoid running its heavy __init__.py
# (which imports clustering_pipeline, preparation_pipeline, etc.) when we only
# need the clustering sub-package.
for _pkg in ('analysis', 'analysis.touch_analytics'):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _parts = _pkg.split('.')
        _pkg_dir = _SRC / Path(*_parts)
        if _pkg_dir.exists():
            _mod.__path__ = [str(_pkg_dir)]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod
    # Python normally sets child modules as attributes on the parent so that
    # unittest.mock.patch can resolve dotted paths like
    # 'analysis.touch_analytics.clustering.gmm_clusterer.GaussianMixture'.
    if '.' in _pkg:
        _parent_name, _child_name = _pkg.rsplit('.', 1)
        setattr(sys.modules[_parent_name], _child_name, sys.modules[_pkg])

from analysis.touch_analytics.clustering import CLUSTERER_REGISTRY, get_clusterer  # noqa: E402
from analysis.touch_analytics.clustering.base import ClusteringContext  # noqa: E402
from analysis.touch_analytics.clustering.gmm_clusterer import GMMClusterer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> ClusteringContext:
    return ClusteringContext()


def _multimodal_df(n_per_cluster: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = [(-4.0, -4.0), (0.0, 0.0), (4.0, 4.0)]
    parts = [rng.normal(loc=c, scale=0.4, size=(n_per_cluster, 2)) for c in centers]
    X = np.vstack(parts)
    return pd.DataFrame(X, columns=['feature_a', 'feature_b'])


def _unimodal_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=(0.0, 0.0), scale=0.4, size=(n, 2))
    return pd.DataFrame(X, columns=['feature_a', 'feature_b'])


def _cfg(**kw) -> dict:
    return {'min_touches_per_component': 30, **kw}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gmm_basic_multimodal():
    df = _multimodal_df(n_per_cluster=100)
    labels, meta = GMMClusterer().fit_predict(df, _cfg(max_components=8, n_init=5), _ctx())

    assert 2 <= meta['k'] <= 5
    assert meta['convergence'] is True
    assert 'bic_scores' in meta
    assert 'component_weights' in meta
    assert 'extra_columns' in meta
    assert len(labels) == len(df)

    # New: ellipse metadata
    k = meta['k']
    D = len(df.columns)
    assert 'means' in meta
    assert 'covariances' in meta
    assert 'feature_columns' in meta
    assert len(meta['means']) == k
    assert len(meta['means'][0]) == D
    assert len(meta['covariances']) == k
    assert len(meta['covariances'][0]) == D
    assert len(meta['covariances'][0][0]) == D
    assert meta['feature_columns'] == df.columns.tolist()


def test_gmm_single_component():
    df = _unimodal_df(n=100)
    labels, meta = GMMClusterer().fit_predict(df, _cfg(max_components=5, n_init=5), _ctx())

    assert meta['k'] == 1
    assert np.all(labels == 0)


def test_gmm_min_touches_enforcement():
    rng = np.random.default_rng(0)
    # 200 pts in a dense cluster, 10 pts in a well-separated cluster
    cluster_a = rng.normal(loc=(-6.0, 0.0), scale=0.3, size=(200, 2))
    cluster_b = rng.normal(loc=(6.0, 0.0), scale=0.3, size=(10, 2))
    df = pd.DataFrame(np.vstack([cluster_a, cluster_b]), columns=['f1', 'f2'])

    _, meta = GMMClusterer().fit_predict(
        df, _cfg(max_components=5, min_touches_per_component=30, n_init=3), _ctx()
    )

    # K=2 would have a 10-pt component → invalid; algorithm must pick K=1
    assert meta['k'] == 1 or np.bincount(meta['cluster_sizes']).min() >= 30  # type: ignore[arg-type]


def test_gmm_too_few_samples(caplog):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 2)), columns=['f1', 'f2'])

    with caplog.at_level(logging.WARNING):
        labels, meta = GMMClusterer().fit_predict(df, _cfg(min_touches_per_component=30), _ctx())

    assert np.all(labels == 0)
    assert meta['k'] == 1
    assert any('20' in msg for msg in caplog.messages)


@pytest.mark.parametrize('cov_type', ['full', 'tied', 'diag', 'spherical'])
def test_gmm_covariance_types(cov_type):
    df = _multimodal_df(n_per_cluster=60)
    _, meta = GMMClusterer().fit_predict(
        df, _cfg(covariance_type=cov_type, max_components=4, n_init=3), _ctx()
    )

    assert meta['convergence'] is True
    assert meta['covariance_type'] == cov_type


def test_gmm_convergence_failure():
    df = _multimodal_df(n_per_cluster=60)

    mock_gm = MagicMock()
    mock_gm.converged_ = False

    with patch(
        'analysis.touch_analytics.clustering.gmm_clusterer.GaussianMixture',
        return_value=mock_gm,
    ):
        with pytest.raises(RuntimeError, match="did not converge"):
            GMMClusterer().fit_predict(df, _cfg(max_components=3, n_init=1), _ctx())


def test_gmm_registry():
    assert 'gmm' in CLUSTERER_REGISTRY
    assert isinstance(get_clusterer('gmm'), GMMClusterer)


def test_gmm_soft_assignments_shape():
    df = _multimodal_df(n_per_cluster=60)
    _, meta = GMMClusterer().fit_predict(df, _cfg(max_components=5, n_init=3), _ctx())

    k = meta['k']
    extra = meta['extra_columns']
    assert len(extra) == k
    for i in range(k):
        assert f'gmm_prob_{i}' in extra

    prob_matrix = np.stack([extra[f'gmm_prob_{i}'] for i in range(k)], axis=1)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_gmm_metadata_json_serializable():
    df = _multimodal_df(n_per_cluster=60)
    _, meta = GMMClusterer().fit_predict(df, _cfg(max_components=5, n_init=3), _ctx())

    meta.pop('extra_columns', None)
    serialized = json.dumps(meta)  # must not raise
    restored = json.loads(serialized)
    # new keys must round-trip
    assert 'means' in restored
    assert 'covariances' in restored
    assert 'feature_columns' in restored


def test_gmm_deterministic():
    df = _multimodal_df(n_per_cluster=60)
    cfg = _cfg(max_components=5, n_init=3)

    labels1, _ = GMMClusterer().fit_predict(df, cfg, _ctx())
    labels2, _ = GMMClusterer().fit_predict(df, cfg, _ctx())

    np.testing.assert_array_equal(labels1, labels2)


def test_gmm_early_exit_has_ellipse_metadata():
    """Early-exit branch (too few samples) must still return means/covariances/feature_columns."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(20, 4)), columns=['p', 'vx', 'vy', 'vz'])

    _, meta = GMMClusterer().fit_predict(df, _cfg(min_touches_per_component=30), _ctx())

    assert meta['k'] == 1
    assert 'means' in meta and len(meta['means']) == 1
    assert len(meta['means'][0]) == 4
    assert 'covariances' in meta and len(meta['covariances']) == 1
    assert len(meta['covariances'][0]) == 4
    assert 'feature_columns' in meta
    assert meta['feature_columns'] == ['p', 'vx', 'vy', 'vz']
    # must be JSON-serializable
    import json
    meta.pop('extra_columns', None)
    json.dumps(meta)
