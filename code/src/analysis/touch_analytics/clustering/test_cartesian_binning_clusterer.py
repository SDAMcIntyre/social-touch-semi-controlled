# clustering/test_cartesian_binning_clusterer.py
import logging

import numpy as np
import pandas as pd
import pytest

from .cartesian_binning_clusterer import CartesianBinningClusterer
from .base import ClusteringContext


def _ctx() -> ClusteringContext:
    return ClusteringContext()


def _clusterer() -> CartesianBinningClusterer:
    return CartesianBinningClusterer()


class TestDenseIdsOverOccupiedCombos:
    def test_dense_ids_over_occupied_combos(self):
        # 2 features, values deliberately span exactly 4 of the 9 possible
        # (3×3) bin combinations.
        rng = np.random.default_rng(0)
        # feature A: values in [0, 1), [1, 2), [2, 3) — 3 bands
        # feature B: values in [0, 1), [1, 2), [2, 3) — 3 bands
        # Occupied combos: (0,0), (0,1), (1,2), (2,0) → 4 unique rows
        fa = np.array([0.1, 0.2, 0.5, 0.6, 1.5, 2.5, 2.6, 2.7])
        fb = np.array([0.1, 0.2, 1.1, 1.2, 2.1, 0.3, 0.4, 0.5])
        df = pd.DataFrame({'A': fa, 'B': fb})

        config = {'n_bins': 3, 'bin_method': 'equal_width'}
        labels, meta = _clusterer().fit_predict(df, config, _ctx())

        assert meta['n_clusters'] == 4
        assert set(labels.tolist()) == {0, 1, 2, 3}
        assert len(labels) == len(df)
        assert labels.dtype == np.intp or np.issubdtype(labels.dtype, np.integer)


class TestNBinsPerFeatureOverride:
    def test_n_bins_per_feature_override(self):
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({
            'x': rng.uniform(0, 10, n),
            'y': rng.uniform(0, 10, n),
        })
        config = {
            'n_bins': 2,
            'n_bins_per_feature': {'y': 5},
            'bin_method': 'equal_width',
        }
        labels, meta = _clusterer().fit_predict(df, config, _ctx())

        # bin_x should have at most 2 distinct values (n_bins=2 default)
        bin_x = meta['extra_columns']['bin_x']
        assert len(np.unique(bin_x)) <= 2, "bin_x should respect default n_bins=2"

        # bin_y should have at most 5 distinct values (n_bins_per_feature override)
        bin_y = meta['extra_columns']['bin_y']
        assert len(np.unique(bin_y)) <= 5, "bin_y should respect per-feature n_bins=5"
        assert len(np.unique(bin_y)) > len(np.unique(bin_x)), (
            "bin_y (5 bins) should have more distinct values than bin_x (2 bins)"
        )

        assert meta['n_bins_per_feature']['x'] == 2
        assert meta['n_bins_per_feature']['y'] == 5


class TestUnknownOverrideRaises:
    def test_unknown_override_raises(self):
        df = pd.DataFrame({'real_col': [1.0, 2.0, 3.0, 4.0, 5.0]})
        config = {
            'n_bins': 3,
            'n_bins_per_feature': {'nonexistent_col': 3},
        }
        with pytest.raises(ValueError, match='nonexistent_col'):
            _clusterer().fit_predict(df, config, _ctx())


class TestConstantColumnSkipped:
    def test_constant_column_skipped(self, caplog):
        df = pd.DataFrame({
            'const': [7.0] * 20,
            'varying': list(range(20)),
        })
        config = {'n_bins': 4, 'bin_method': 'equal_width'}

        import logging
        with caplog.at_level(logging.WARNING):
            labels, meta = _clusterer().fit_predict(df, config, _ctx())

        assert any("'const'" in rec.message for rec in caplog.records), (
            "Expected a log warning mentioning the constant column 'const'"
        )

        assert 'const' not in meta['binned_features']
        assert 'varying' in meta['binned_features']
        # Only 'varying' drives the combos, so bin_const must not appear
        assert 'bin_const' not in meta['extra_columns']
        assert 'bin_varying' in meta['extra_columns']
        # Labels are dense ids from the single remaining feature
        assert len(labels) == len(df)


class TestNBinsTooSmallRaises:
    def test_n_bins_too_small_raises(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]})
        config = {'n_bins': 1}
        with pytest.raises(ValueError, match='n_bins must be >= 2'):
            _clusterer().fit_predict(df, config, _ctx())
