"""Tests for analysis.receptive_field_mapping.rendering.fit_models."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub the heavy __init__.py chain so we can import the leaf module directly.
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stub(dotted: str) -> None:
    if dotted in sys.modules:
        return
    mod = types.ModuleType(dotted)
    pkg_dir = _SRC / Path(*dotted.split("."))
    if pkg_dir.exists():
        mod.__path__ = [str(pkg_dir)]
    mod.__package__ = dotted
    sys.modules[dotted] = mod


_stub("analysis")
_stub("analysis.receptive_field_mapping")
_stub("analysis.receptive_field_mapping.rendering")

from analysis.receptive_field_mapping.rendering.fit_models import (  # noqa: E402
    MODEL_REGISTRY,
    FitResult,
    fit_model,
    parse_fit_config,
)


# ---------------------------------------------------------------------------
# Polynomial backward compatibility
# ---------------------------------------------------------------------------

class TestPolynomialBackwardCompat:

    def test_poly2_matches_numpy(self):
        rng = np.random.default_rng(42)
        x = np.linspace(1, 10, 50)
        y = 3.0 * x ** 2 - 2.0 * x + 1.0 + rng.normal(0, 0.5, len(x))

        result = fit_model(x, y, "poly2")
        coeffs_np = np.polyfit(x, y, 2)

        assert result.params is not None
        np.testing.assert_allclose(result.params, coeffs_np, rtol=1e-10)
        assert result.r_squared > 0.99

    def test_poly1_single_degree(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = fit_model(x, y, "poly1")
        assert result.params is not None
        assert result.display_label == "deg 1"
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_poly_evaluate_matches_polyval(self):
        rng = np.random.default_rng(7)
        x = np.linspace(0, 5, 30)
        y = x ** 3 + rng.normal(0, 0.1, len(x))

        result = fit_model(x, y, "poly3")
        x_eval = np.linspace(0, 5, 100)
        np.testing.assert_allclose(
            result.evaluate(x_eval),
            np.polyval(result.params, x_eval),
            rtol=1e-12,
        )

    def test_constant_response_returns_r2_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])

        result = fit_model(x, y, "poly1")
        assert result.params is not None
        assert result.r_squared == 0.0


# ---------------------------------------------------------------------------
# Nonlinear models on synthetic data
# ---------------------------------------------------------------------------

class TestPowerLaw:

    def test_recovery(self):
        x = np.linspace(1, 20, 80)
        y = 3.0 * np.power(x, 0.5) + 1.0

        result = fit_model(x, y, "power")
        assert result.params is not None
        assert result.r_squared > 0.99
        a, b, c = result.params
        assert b == pytest.approx(0.5, abs=0.05)

    def test_noisy_data(self):
        rng = np.random.default_rng(123)
        x = np.linspace(1, 50, 100)
        y = 5.0 * np.power(x, 0.3) + 2.0 + rng.normal(0, 0.5, len(x))

        result = fit_model(x, y, "power")
        assert result.params is not None
        assert result.r_squared > 0.90


class TestLogarithmic:

    def test_recovery(self):
        x = np.linspace(1, 100, 80)
        y = 2.0 * np.log(x) + 5.0

        result = fit_model(x, y, "log")
        assert result.params is not None
        assert result.r_squared > 0.99
        a, b = result.params
        assert a == pytest.approx(2.0, abs=0.1)
        assert b == pytest.approx(5.0, abs=0.5)


class TestNakaRushton:

    def test_recovery(self):
        x = np.linspace(1, 200, 100)
        y = 100.0 * x ** 2 / (x ** 2 + 50.0 ** 2) + 5.0

        result = fit_model(x, y, "naka_rushton")
        assert result.params is not None
        assert result.r_squared > 0.95


class TestSigmoid:

    def test_recovery(self):
        x = np.linspace(0, 40, 100)
        y = 10.0 / (1.0 + np.exp(-0.5 * (x - 20.0))) + 2.0

        result = fit_model(x, y, "sigmoid")
        assert result.params is not None
        assert result.r_squared > 0.99


class TestExpSat:

    def test_recovery(self):
        x = np.linspace(0, 50, 100)
        y = 5.0 * (1.0 - np.exp(-0.1 * x)) + 1.0

        result = fit_model(x, y, "exp_sat")
        assert result.params is not None
        assert result.r_squared > 0.99
        a, b, c = result.params
        assert a == pytest.approx(5.0, abs=0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_insufficient_data_returns_none(self):
        x = np.array([1.0])
        y = np.array([2.0])

        result = fit_model(x, y, "poly2")
        assert result.params is None
        assert result.r_squared is None
        assert result.evaluate is None

    def test_nonlinear_insufficient_data(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        result = fit_model(x, y, "naka_rushton")
        assert result.params is None

    def test_unknown_model_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown fit model"):
            fit_model(x, y, "nonexistent_model")

    def test_all_registered_models_have_names(self):
        for name, spec in MODEL_REGISTRY.items():
            assert spec.name == name
            assert spec.display_label
            assert spec.min_points >= 1

    def test_evaluate_is_none_on_failure(self):
        x = np.array([1.0])
        y = np.array([2.0])
        result = fit_model(x, y, "power")
        assert result.evaluate is None

    def test_fit_result_fields(self):
        x = np.linspace(1, 10, 20)
        y = 2.0 * x + 1.0

        result = fit_model(x, y, "poly1")
        assert isinstance(result, FitResult)
        assert result.model_name == "poly1"
        assert result.display_label == "deg 1"
        assert result.params is not None
        assert result.r_squared is not None
        assert callable(result.evaluate)


# ---------------------------------------------------------------------------
# parse_fit_config
# ---------------------------------------------------------------------------

class TestParseFitConfig:

    def test_legacy_single_int(self):
        assert parse_fit_config({"fit_degree": 4}) == ["poly4"]

    def test_legacy_list_int(self):
        assert parse_fit_config({"fit_degrees": [1, 2]}, degree_key="fit_degrees") == [
            "poly1", "poly2",
        ]

    def test_new_format_list(self):
        assert parse_fit_config({"fit_models": ["power", "log", "poly2"]}) == [
            "power", "log", "poly2",
        ]

    def test_new_format_single_string(self):
        assert parse_fit_config({"fit_models": "power"}) == ["power"]

    def test_fit_models_overrides_fit_degree(self):
        opts = {"fit_degree": 4, "fit_models": ["power"]}
        assert parse_fit_config(opts) == ["power"]

    def test_default_when_no_key(self):
        assert parse_fit_config({}) == ["poly1"]

    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError, match="Unknown fit model"):
            parse_fit_config({"fit_models": ["bogus"]})

    def test_invalid_degree_type_raises(self):
        with pytest.raises(ValueError):
            parse_fit_config({"fit_degree": "four"})

    def test_all_keyword(self):
        result = parse_fit_config({"fit_models": "all"})
        assert result == sorted(MODEL_REGISTRY.keys())
        assert len(result) == 14

    def test_invalid_models_type_raises(self):
        with pytest.raises(ValueError):
            parse_fit_config({"fit_models": 42})
