# comparing/precision_comparator.py
"""
Precision comparator: Levene's test on measurement variance across sensors.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from .base import ComparisonResult, ComparisonStrategy


class PrecisionComparator(ComparisonStrategy):
    """
    Detects differences in measurement variance (precision) across sensors.

    Test: Levene's test (scipy.stats.levene, center='median').
    Effect size: ratio of max-to-min per-sensor variance (variance ratio).
    """

    def compare(
        self,
        stratum_df: pd.DataFrame,
        sensor_col: str,
        measurement_col: str,
        config: dict,
    ) -> ComparisonResult:
        sensors = stratum_df[sensor_col].unique()
        groups = [
            stratum_df.loc[stratum_df[sensor_col] == s, measurement_col].dropna().values
            for s in sensors
        ]
        groups = [g for g in groups if len(g) > 1]
        n_sensors = len(groups)
        n_obs = sum(len(g) for g in groups)

        if n_sensors < 2:
            return ComparisonResult(
                strategy_name='precision',
                stratum_label=int(stratum_df['cluster_label'].iloc[0]),
                stratum_dispersion=0.0,
                n_sensors=n_sensors,
                n_observations=n_obs,
                test_statistic=float('nan'),
                p_value=float('nan'),
                effect_size=None,
                detail={'error': 'insufficient data'},
            )

        try:
            stat, p_value = stats.levene(*groups, center='median')
        except Exception as exc:
            logging.warning(f"PrecisionComparator Levene failed: {exc}")
            stat, p_value = float('nan'), float('nan')

        variances = [float(np.var(g, ddof=1)) for g in groups]
        non_zero = [v for v in variances if v > 0]
        variance_ratio = (max(non_zero) / min(non_zero)) if len(non_zero) >= 2 else None

        return ComparisonResult(
            strategy_name='precision',
            stratum_label=int(stratum_df['cluster_label'].iloc[0]),
            stratum_dispersion=0.0,
            n_sensors=n_sensors,
            n_observations=n_obs,
            test_statistic=float(stat) if not np.isnan(stat) else float('nan'),
            p_value=float(p_value) if not np.isnan(p_value) else float('nan'),
            effect_size=float(variance_ratio) if variance_ratio is not None else None,
            detail={
                'per_sensor_variance': {
                    str(s): float(np.var(g, ddof=1)) for s, g in zip(sensors, groups)
                },
            },
        )
