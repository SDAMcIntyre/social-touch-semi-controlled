"""Named fit model registry for tuning-curve rendering.

Provides polynomial and nonlinear curve-fitting models used by the
multi-fit overlay system in ``rf_response_tuning_renderer`` and
``rf_spatial_tuning_renderer``.  Each model is registered as a
:class:`FitModelSpec` in :data:`MODEL_REGISTRY`; the main entry point
is :func:`fit_model` which returns a :class:`FitResult`.

Nonlinear models use ``scipy.optimize.curve_fit``; polynomial models
delegate to ``numpy.polyfit`` / ``numpy.polyval`` so that existing
polynomial behaviour is byte-identical.

Available models
----------------
``poly1`` … ``poly9``
    Polynomial of degree 1–9.
``power``
    Stevens' power law: ``y = a * |x|^b + c``.
``log``
    Logarithmic (Weber-Fechner): ``y = a * ln(x) + b``.
``naka_rushton``
    Naka-Rushton saturation: ``y = R_max * |x|^n / (|x|^n + σ^n) + b``.
``sigmoid``
    Logistic sigmoid: ``y = L / (1 + exp(-k·(x − x₀))) + b``.
``exp_sat``
    Exponential saturation: ``y = a * (1 − exp(−b·x)) + c``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FitResult:
    """Outcome of fitting a single model to data."""

    model_name: str
    display_label: str
    params: np.ndarray | None
    r_squared: float | None
    evaluate: Callable[[np.ndarray], np.ndarray] | None


@dataclass(frozen=True)
class FitModelSpec:
    """Specification for a named fit model."""

    name: str
    display_label: str
    func: Callable | None
    guess: Callable[[np.ndarray, np.ndarray], tuple] | None
    min_points: int
    param_bounds: Callable[[np.ndarray, np.ndarray], tuple] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, FitModelSpec] = {}


def register_model(spec: FitModelSpec) -> None:
    """Add *spec* to the global model registry."""
    MODEL_REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Polynomial models (poly1 … poly9)
# ---------------------------------------------------------------------------

for _deg in range(1, 10):
    register_model(FitModelSpec(
        name=f"poly{_deg}",
        display_label=f"deg {_deg}",
        func=None,
        guess=None,
        min_points=_deg + 1,
    ))


# ---------------------------------------------------------------------------
# Nonlinear model definitions
# ---------------------------------------------------------------------------

# ---- Power law: y = a * |x|^b + c ----------------------------------------

def _power_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(np.abs(x), b) + c


def _power_guess(x: np.ndarray, y: np.ndarray) -> tuple:
    c0 = float(np.min(y))
    a0 = float(np.max(y) - c0) or 1.0
    b0 = 0.5
    return (a0, b0, c0)


def _power_bounds(x: np.ndarray, y: np.ndarray) -> tuple:
    return ((-np.inf, 0.01, -np.inf), (np.inf, 10.0, np.inf))


register_model(FitModelSpec(
    name="power",
    display_label="power",
    func=_power_func,
    guess=_power_guess,
    min_points=3,
    param_bounds=_power_bounds,
))

# ---- Logarithmic: y = a * ln(x) + b --------------------------------------

def _log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log(np.maximum(x, 1e-12)) + b


def _log_guess(x: np.ndarray, y: np.ndarray) -> tuple:
    x_pos = x[x > 0]
    if len(x_pos) < 2:
        return (1.0, float(np.mean(y)))
    log_x = np.log(x_pos)
    denom = float(log_x.max() - log_x.min())
    a0 = float(y.max() - y.min()) / max(denom, 1e-12)
    b0 = float(np.mean(y) - a0 * np.mean(log_x))
    return (a0, b0)


register_model(FitModelSpec(
    name="log",
    display_label="log",
    func=_log_func,
    guess=_log_guess,
    min_points=2,
))

# ---- Naka-Rushton: y = R_max * |x|^n / (|x|^n + sigma^n) + b ------------

def _naka_rushton_func(
    x: np.ndarray, r_max: float, n: float, sigma: float, b: float,
) -> np.ndarray:
    xn = np.power(np.abs(x), n)
    return r_max * xn / (xn + sigma ** n + 1e-12) + b


def _naka_rushton_guess(x: np.ndarray, y: np.ndarray) -> tuple:
    r_max0 = float(np.max(y) - np.min(y)) or 1.0
    n0 = 1.0
    sigma0 = float(np.median(np.abs(x))) or 1.0
    b0 = float(np.min(y))
    return (r_max0, n0, sigma0, b0)


def _naka_rushton_bounds(x: np.ndarray, y: np.ndarray) -> tuple:
    return ((0, 0.1, 1e-6, -np.inf), (np.inf, 10.0, np.inf, np.inf))


register_model(FitModelSpec(
    name="naka_rushton",
    display_label="naka-rushton",
    func=_naka_rushton_func,
    guess=_naka_rushton_guess,
    min_points=4,
    param_bounds=_naka_rushton_bounds,
))

# ---- Sigmoid / logistic: y = L / (1 + exp(-k*(x - x0))) + b -------------

def _sigmoid_func(
    x: np.ndarray, L: float, k: float, x0: float, b: float,
) -> np.ndarray:
    return L / (1.0 + np.exp(np.clip(-k * (x - x0), -500, 500))) + b


def _sigmoid_guess(x: np.ndarray, y: np.ndarray) -> tuple:
    L0 = float(np.max(y) - np.min(y)) or 1.0
    k0 = 1.0
    x0_0 = float(np.median(x))
    b0 = float(np.min(y))
    return (L0, k0, x0_0, b0)


def _sigmoid_bounds(x: np.ndarray, y: np.ndarray) -> tuple:
    return ((0, 0, -np.inf, -np.inf), (np.inf, 100.0, np.inf, np.inf))


register_model(FitModelSpec(
    name="sigmoid",
    display_label="sigmoid",
    func=_sigmoid_func,
    guess=_sigmoid_guess,
    min_points=4,
    param_bounds=_sigmoid_bounds,
))

# ---- Exponential saturation: y = a * (1 - exp(-b*x)) + c -----------------

def _exp_sat_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (1.0 - np.exp(np.clip(-b * x, -500, 500))) + c


def _exp_sat_guess(x: np.ndarray, y: np.ndarray) -> tuple:
    a0 = float(np.max(y) - np.min(y)) or 1.0
    x_range = float(np.max(np.abs(x))) or 1.0
    b0 = 2.3 / x_range
    c0 = float(np.min(y))
    return (a0, b0, c0)


def _exp_sat_bounds(x: np.ndarray, y: np.ndarray) -> tuple:
    return ((0, 0, -np.inf), (np.inf, np.inf, np.inf))


register_model(FitModelSpec(
    name="exp_sat",
    display_label="exp-sat",
    func=_exp_sat_func,
    guess=_exp_sat_guess,
    min_points=3,
    param_bounds=_exp_sat_bounds,
))


# ---------------------------------------------------------------------------
# R-squared helper
# ---------------------------------------------------------------------------

def _compute_r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def fit_model(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> FitResult:
    """Fit a named model to *(x, y)* data.

    Returns a :class:`FitResult` with ``params=None`` and
    ``r_squared=None`` when too few points exist or when
    ``curve_fit`` fails to converge.

    Raises
    ------
    ValueError
        If *model_name* is not in :data:`MODEL_REGISTRY`.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown fit model {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )

    spec = MODEL_REGISTRY[model_name]
    n = len(x)

    fail = FitResult(
        model_name=spec.name,
        display_label=spec.display_label,
        params=None,
        r_squared=None,
        evaluate=None,
    )

    if n < spec.min_points:
        return fail

    # -- Polynomial path (byte-identical to legacy fit_polynomial) ----------
    if spec.func is None:
        deg = int(spec.name.removeprefix("poly"))
        coeffs = np.polyfit(x, y, deg)
        predicted = np.polyval(coeffs, x)
        r2 = _compute_r_squared(y, predicted)
        return FitResult(
            model_name=spec.name,
            display_label=spec.display_label,
            params=coeffs,
            r_squared=r2,
            evaluate=lambda xs, _c=coeffs: np.polyval(_c, xs),
        )

    # -- Nonlinear path via scipy.optimize.curve_fit ------------------------
    p0 = spec.guess(x, y)
    bounds = spec.param_bounds(x, y) if spec.param_bounds is not None else (-np.inf, np.inf)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                spec.func,
                x, y,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        return fail

    predicted = spec.func(x, *popt)
    r2 = _compute_r_squared(y, predicted)
    func = spec.func

    return FitResult(
        model_name=spec.name,
        display_label=spec.display_label,
        params=popt,
        r_squared=r2,
        evaluate=lambda xs, _f=func, _p=popt: _f(xs, *_p),
    )


# ---------------------------------------------------------------------------
# Config parsing helper
# ---------------------------------------------------------------------------

def parse_fit_config(
    options: dict,
    degree_key: str = "fit_degree",
    models_key: str = "fit_models",
) -> list[str]:
    """Parse YAML options into a list of model-name strings.

    Priority: *models_key* > *degree_key*.

    Backward compatible::

        fit_degree: 4          ->  ["poly4"]
        fit_degrees: [1, 2]    ->  ["poly1", "poly2"]
        fit_models: ["power"]  ->  ["power"]
    """
    if models_key in options:
        raw = options[models_key]
        if isinstance(raw, str):
            if raw == "all":
                return sorted(MODEL_REGISTRY.keys())
            raw = [raw]
        if not isinstance(raw, list) or not all(isinstance(m, str) for m in raw):
            raise ValueError(
                f"{models_key} must be a string or list of strings, got {raw!r}"
            )
        for name in raw:
            if name not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown fit model {name!r} in {models_key}. "
                    f"Available: {sorted(MODEL_REGISTRY)}"
                )
        return list(raw)

    raw_deg = options.get(degree_key, 1)
    if isinstance(raw_deg, int):
        return [f"poly{raw_deg}"]
    if isinstance(raw_deg, list):
        if not all(isinstance(d, int) for d in raw_deg):
            raise ValueError(
                f"{degree_key} list must contain only integers, got {raw_deg!r}"
            )
        return [f"poly{d}" for d in raw_deg]
    raise ValueError(
        f"{degree_key} must be int or list[int], got {type(raw_deg).__name__}"
    )
