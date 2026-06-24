# comparing/distribution_comparator.py
"""
Distribution comparator: pairwise two-sample KS tests across sensors.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .base import ComparisonResult, ComparisonStrategy


class DistributionComparator(ComparisonStrategy):
    """
    Detects distributional differences across sensors via pairwise KS tests.

    Test: two-sample Kolmogorov-Smirnov (scipy.stats.ks_2samp).
    Summary statistic: maximum KS statistic across all sensor pairs.
    Minimum p-value across all pairs is reported as the overall p_value.
    Effect size: max KS statistic (already in [0, 1]).
    """

    def compare(
        self,
        stratum_df: pd.DataFrame,
        sensor_col: str,
        measurement_col: str,
        config: dict,
    ) -> ComparisonResult:
        sensors = stratum_df[sensor_col].unique()
        sensor_data = {
            s: stratum_df.loc[stratum_df[sensor_col] == s, measurement_col].dropna().values
            for s in sensors
        }
        sensor_data = {s: g for s, g in sensor_data.items() if len(g) > 0}
        valid_sensors = list(sensor_data.keys())
        n_sensors = len(valid_sensors)
        n_obs = sum(len(g) for g in sensor_data.values())

        if n_sensors < 2:
            return ComparisonResult(
                strategy_name='distribution',
                stratum_label=int(stratum_df['cluster_label'].iloc[0]),
                stratum_dispersion=0.0,
                n_sensors=n_sensors,
                n_observations=n_obs,
                test_statistic=float('nan'),
                p_value=float('nan'),
                effect_size=None,
                detail={'error': 'insufficient data'},
            )

        pairwise = {}
        ks_stats = []
        p_values = []

        for s1, s2 in combinations(valid_sensors, 2):
            pair = f"{s1} vs {s2}"
            try:
                ks_stat, p_val = stats.ks_2samp(sensor_data[s1], sensor_data[s2])
                pairwise[pair] = {'ks_statistic': float(ks_stat), 'p_value': float(p_val)}
                ks_stats.append(ks_stat)
                p_values.append(p_val)
            except Exception as exc:
                logging.warning(f"DistributionComparator KS failed for {pair}: {exc}")

        max_ks = float(max(ks_stats)) if ks_stats else float('nan')
        min_p = float(min(p_values)) if p_values else float('nan')

        return ComparisonResult(
            strategy_name='distribution',
            stratum_label=int(stratum_df['cluster_label'].iloc[0]),
            stratum_dispersion=0.0,
            n_sensors=n_sensors,
            n_observations=n_obs,
            test_statistic=max_ks,
            p_value=min_p,
            effect_size=max_ks if ks_stats else None,
            detail={'pairwise': pairwise},
        )
