# comparing/synthesis.py
"""
Dispersion-weighted synthesis of per-stratum comparison results.
"""

import numpy as np
import pandas as pd

from .base import ComparisonResult


def synthesize_across_strata(
    results: list[ComparisonResult],
    weighting: str = 'inverse_dispersion',
) -> pd.DataFrame:
    """
    Aggregate per-stratum ComparisonResults into a global summary.

    Tight strata (low dispersion) carry more evidential weight because
    "similar process" is more tightly guaranteed within them.

    Parameters
    ----------
    results
        All ComparisonResult objects from one or more strategies.
    weighting
        ``'inverse_dispersion'`` (default): weight = 1 / (dispersion + epsilon).
        Other values fall back to uniform weighting.

    Returns
    -------
    DataFrame with columns:
        strategy, stratum_label, stratum_dispersion, n_sensors, n_observations,
        test_statistic, p_value, effect_size, weight, weighted_p
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            'strategy': r.strategy_name,
            'stratum_label': r.stratum_label,
            'stratum_dispersion': r.stratum_dispersion,
            'n_sensors': r.n_sensors,
            'n_observations': r.n_observations,
            'test_statistic': r.test_statistic,
            'p_value': r.p_value,
            'effect_size': r.effect_size,
        })

    df = pd.DataFrame(rows)

    eps = 1e-9
    if weighting == 'inverse_dispersion':
        df['weight'] = 1.0 / (df['stratum_dispersion'] + eps)
    else:
        df['weight'] = 1.0

    # Normalize weights per strategy so they sum to 1
    for strategy in df['strategy'].unique():
        mask = df['strategy'] == strategy
        total = df.loc[mask, 'weight'].sum()
        if total > 0:
            df.loc[mask, 'weight'] = df.loc[mask, 'weight'] / total

    # Weighted p-value (Fisher-style weighted combination is complex;
    # we report weighted mean p as a descriptive summary metric)
    df['weighted_p'] = df['weight'] * df['p_value'].fillna(1.0)

    return df
