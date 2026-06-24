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


# ---------------------------------------------------------------------------
# Outlier detection integration
# ---------------------------------------------------------------------------

def _make_df_with_outliers() -> pd.DataFrame:
    """Single feature with clear high-end outliers for IQR detection."""
    normal = list(range(1, 21))
    outliers = [200, 300]
    return pd.DataFrame({'x': normal + outliers})


class TestOutlierDisabledByDefault:
    def test_no_outlier_method_in_config(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'bin_method': 'equal_width'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert meta['outlier_method'] is None
        assert meta['outlier_info'] == {}

    def test_explicit_none(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'outlier_method': None}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert meta['outlier_method'] is None


class TestOutlierIQR:
    def test_high_outliers_get_dedicated_bin(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'bin_method': 'equal_width', 'outlier_method': 'iqr'}
        labels, meta = _clusterer().fit_predict(df, config, _ctx())

        bin_x = meta['extra_columns']['bin_x']
        # Rows at index 20 and 21 are the outlier values (200, 300)
        normal_bins = bin_x[:20]
        outlier_indices = bin_x[20:]
        normal_max_bin = max(b for b in normal_bins if b >= 0)
        assert all(b > normal_max_bin for b in outlier_indices)

        assert 'x' in meta['outlier_info']
        info = meta['outlier_info']['x']
        assert info['method'] == 'iqr'
        assert info['n_high_outliers'] == 2
        assert info['n_low_outliers'] == 0


class TestOutlierMAD:
    def test_detects_outliers(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'outlier_method': 'mad'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert 'x' in meta['outlier_info']
        assert meta['outlier_info']['x']['n_high_outliers'] >= 1


class TestOutlierPercentile:
    def test_detects_outliers(self):
        normal = list(range(1, 101))
        outliers = [1000, 2000]
        df = pd.DataFrame({'x': normal + outliers})
        config = {'n_bins': 5, 'outlier_method': 'percentile', 'outlier_params': {'p': 1.0}}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert meta['outlier_method'] == 'percentile'


class TestOutlierTukey:
    def test_detects_outliers(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'outlier_method': 'tukey'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert meta['outlier_method'] == 'tukey'
        assert 'x' in meta['outlier_info']


class TestOutlierOneSided:
    def test_low_only(self):
        values = [-500, -400] + list(range(1, 21))
        df = pd.DataFrame({'x': values})
        config = {'n_bins': 4, 'outlier_method': 'iqr'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())

        bin_x = meta['extra_columns']['bin_x']
        assert -2 in bin_x
        info = meta['outlier_info']['x']
        assert info['n_low_outliers'] == 2
        assert info['n_high_outliers'] == 0

    def test_high_only(self):
        values = list(range(1, 21)) + [500, 600]
        df = pd.DataFrame({'x': values})
        config = {'n_bins': 4, 'outlier_method': 'iqr'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())

        info = meta['outlier_info']['x']
        assert info['n_low_outliers'] == 0
        assert info['n_high_outliers'] == 2


class TestOutlierNoOutliersDetected:
    def test_clean_data(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({'x': rng.uniform(0, 1, 100)})
        config = {'n_bins': 5, 'outlier_method': 'iqr'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())
        assert meta['outlier_info'] == {}


class TestOutlierInvalidConfig:
    def test_invalid_method_raises(self):
        df = pd.DataFrame({'x': list(range(20))})
        config = {'n_bins': 3, 'outlier_method': 'zscore'}
        with pytest.raises(ValueError, match='outlier_method must be one of'):
            _clusterer().fit_predict(df, config, _ctx())

    def test_invalid_params_raises(self):
        df = pd.DataFrame({'x': list(range(20))})
        config = {'n_bins': 3, 'outlier_method': 'iqr', 'outlier_params': {'bogus': 5}}
        with pytest.raises(ValueError):
            _clusterer().fit_predict(df, config, _ctx())


class TestOutlierCartesianProduct:
    def test_outlier_bins_participate_in_combinations(self):
        normal_a = list(range(1, 21))
        normal_b = list(range(1, 21))
        df = pd.DataFrame({
            'A': normal_a + [500],
            'B': normal_b + [1],
        })
        config = {'n_bins': 3, 'outlier_method': 'iqr', 'bin_method': 'equal_width'}
        labels, meta = _clusterer().fit_predict(df, config, _ctx())

        assert len(labels) == len(df)
        combos = meta['cluster_combinations']
        flat = [v for row in combos for v in row]
        # The high outlier in A should produce a bin index outside [0, 2]
        assert any(v < 0 or v > 2 for v in flat)


class TestOutlierMetadataStructure:
    def test_keys_present(self):
        df = _make_df_with_outliers()
        config = {'n_bins': 4, 'outlier_method': 'iqr'}
        _labels, meta = _clusterer().fit_predict(df, config, _ctx())

        assert 'outlier_method' in meta
        assert 'outlier_info' in meta
        info = meta['outlier_info']['x']
        for key in ('method', 'params', 'lower_bound', 'upper_bound',
                     'n_low_outliers', 'n_high_outliers'):
            assert key in info, f"Missing key '{key}' in outlier_info"
