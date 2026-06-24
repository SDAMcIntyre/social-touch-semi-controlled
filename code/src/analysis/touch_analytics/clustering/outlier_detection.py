# clustering/outlier_detection.py
"""Per-feature univariate outlier detection for cartesian binning."""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTLIER_BIN_LOW: int = -2

_VALID_METHODS = {"iqr", "mad", "percentile", "tukey"}

_VALID_PARAMS: dict[str, set[str]] = {
    "iqr": {"k"},
    "mad": {"threshold"},
    "percentile": {"p"},
    "tukey": {"k_outer"},
}

_MIN_SAMPLES = 4


def detect_outlier_bounds(
    series: pd.Series,
    method: str,
    params: dict,
) -> tuple[float, float]:
    """Compute (lower_bound, upper_bound) outside which values are outliers.

    Parameters
    ----------
    series
        Numeric feature column.  NaN values are ignored.
    method
        One of ``"iqr"``, ``"mad"``, ``"percentile"``, ``"tukey"``.
    params
        Technique-specific parameter overrides.

    Returns
    -------
    (lower_bound, upper_bound)
        Values strictly below *lower_bound* or strictly above *upper_bound*
        are considered outliers.  Returns ``(-inf, +inf)`` when the technique
        cannot determine meaningful bounds (e.g. too few samples or MAD = 0).

    Raises
    ------
    ValueError
        Unknown *method* or unrecognised keys in *params*.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown outlier method '{method}'. "
            f"Valid methods: {sorted(_VALID_METHODS)}"
        )

    bad_keys = set(params) - _VALID_PARAMS[method]
    if bad_keys:
        raise ValueError(
            f"outlier_params contains unknown keys {sorted(bad_keys)} "
            f"for method '{method}'. "
            f"Valid keys: {sorted(_VALID_PARAMS[method])}"
        )

    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) < _MIN_SAMPLES:
        return (-np.inf, np.inf)

    if method == "iqr":
        return _iqr_bounds(clean, k=params.get("k", 1.5))
    if method == "mad":
        return _mad_bounds(clean, threshold=params.get("threshold", 3.5))
    if method == "percentile":
        return _percentile_bounds(clean, p=params.get("p", 1.0))
    # method == "tukey"
    return _tukey_bounds(clean, k_outer=params.get("k_outer", 3.0))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _iqr_bounds(data: np.ndarray, k: float) -> tuple[float, float]:
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def _mad_bounds(data: np.ndarray, threshold: float) -> tuple[float, float]:
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad < 1e-12:
        return (-np.inf, np.inf)
    cutoff = threshold * mad / 0.6745
    return (median - cutoff, median + cutoff)


def _percentile_bounds(data: np.ndarray, p: float) -> tuple[float, float]:
    return (
        float(np.percentile(data, p)),
        float(np.percentile(data, 100 - p)),
    )


def _tukey_bounds(data: np.ndarray, k_outer: float) -> tuple[float, float]:
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return (q1 - k_outer * iqr, q3 + k_outer * iqr)
