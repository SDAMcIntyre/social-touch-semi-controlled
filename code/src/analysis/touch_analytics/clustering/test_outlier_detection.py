# clustering/test_outlier_detection.py
import numpy as np
import pandas as pd
import pytest

from .outlier_detection import detect_outlier_bounds, OUTLIER_BIN_LOW


class TestIQRBounds:
    def test_known_outliers(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        lo, hi = detect_outlier_bounds(data, "iqr", {})
        assert 100 > hi
        assert lo < 1

    def test_custom_k(self):
        data = pd.Series(list(range(1, 101)))
        lo_default, hi_default = detect_outlier_bounds(data, "iqr", {})
        lo_tight, hi_tight = detect_outlier_bounds(data, "iqr", {"k": 0.5})
        assert (hi_tight - lo_tight) < (hi_default - lo_default)


class TestMADBounds:
    def test_known_outliers(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        lo, hi = detect_outlier_bounds(data, "mad", {})
        assert 100 > hi

    def test_constant_data_returns_inf(self):
        data = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        lo, hi = detect_outlier_bounds(data, "mad", {})
        assert lo == -np.inf
        assert hi == np.inf

    def test_custom_threshold(self):
        data = pd.Series(list(range(1, 51)))
        lo_default, hi_default = detect_outlier_bounds(data, "mad", {})
        lo_tight, hi_tight = detect_outlier_bounds(data, "mad", {"threshold": 1.5})
        assert (hi_tight - lo_tight) < (hi_default - lo_default)


class TestPercentileBounds:
    def test_trims_extremes(self):
        data = pd.Series(list(range(1, 101)))
        lo, hi = detect_outlier_bounds(data, "percentile", {"p": 5})
        assert lo >= 5
        assert hi <= 96

    def test_default_p(self):
        data = pd.Series(list(range(1, 1001)))
        lo, hi = detect_outlier_bounds(data, "percentile", {})
        assert lo > data.min()
        assert hi < data.max()


class TestTukeyBounds:
    def test_outer_fences(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 200])
        lo, hi = detect_outlier_bounds(data, "tukey", {})
        assert 200 > hi

    def test_wider_than_iqr_default(self):
        data = pd.Series(list(range(1, 101)))
        lo_iqr, hi_iqr = detect_outlier_bounds(data, "iqr", {})
        lo_tukey, hi_tukey = detect_outlier_bounds(data, "tukey", {})
        assert (hi_tukey - lo_tukey) >= (hi_iqr - lo_iqr)


class TestValidation:
    def test_unknown_method_raises(self):
        data = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Unknown outlier method"):
            detect_outlier_bounds(data, "zscore", {})

    def test_unknown_param_key_raises(self):
        data = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="unknown keys"):
            detect_outlier_bounds(data, "iqr", {"bogus": 42})

    def test_wrong_param_for_method_raises(self):
        data = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="unknown keys"):
            detect_outlier_bounds(data, "mad", {"k": 1.5})


class TestEdgeCases:
    def test_too_few_samples_returns_inf(self):
        data = pd.Series([1.0, 2.0, 3.0])
        lo, hi = detect_outlier_bounds(data, "iqr", {})
        assert lo == -np.inf
        assert hi == np.inf

    def test_nan_values_ignored(self):
        data = pd.Series([1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 100])
        lo, hi = detect_outlier_bounds(data, "iqr", {})
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_no_outliers_in_uniform_data(self):
        rng = np.random.default_rng(42)
        data = pd.Series(rng.uniform(0, 1, 200))
        lo, hi = detect_outlier_bounds(data, "iqr", {})
        assert lo < data.min()
        assert hi > data.max()


class TestOutlierBinLowConstant:
    def test_value(self):
        assert OUTLIER_BIN_LOW == -2
