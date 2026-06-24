# comparing/bias_comparator.py
"""
Bias comparator: one-way ANOVA + Tukey HSD post-hoc.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .base import ComparisonResult, ComparisonStrategy


class BiasComparator(ComparisonStrategy):
    """
    Detects systematic differences in central tendency across sensors.

    Test: one-way ANOVA (scipy.stats.f_oneway).
    Post-hoc: Tukey HSD (scipy.stats.tukey_hsd) for pairwise comparisons.
    Effect size: eta-squared (SS_between / SS_total).
    """

    def compare(
        self,
        stratum_df: pd.DataFrame,
        sensor_col: str,
        measurement_col: str,
        config: dict,
    ) -> ComparisonResult:
        alpha = config.get('alpha', 0.05)
        sensors = stratum_df[sensor_col].unique()
        groups = [
            stratum_df.loc[stratum_df[sensor_col] == s, measurement_col].dropna().values
            for s in sensors
        ]
        groups = [g for g in groups if len(g) > 0]
        n_sensors = len(groups)
        n_obs = sum(len(g) for g in groups)

        if n_sensors < 2 or n_obs < 3:
            return ComparisonResult(
                strategy_name='bias',
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
            f_stat, p_value = stats.f_oneway(*groups)
        except Exception as exc:
            logging.warning(f"BiasComparator ANOVA failed: {exc}")
            f_stat, p_value = float('nan'), float('nan')

        # eta-squared effect size
        all_vals = np.concatenate(groups)
        grand_mean = all_vals.mean()
        ss_total = float(np.sum((all_vals - grand_mean) ** 2))
        ss_between = float(sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
        ))
        eta_sq = (ss_between / ss_total) if ss_total > 0 else None

        # Tukey HSD post-hoc
        pairwise = {}
        try:
            tukey = stats.tukey_hsd(*groups)
            sensor_labels = list(sensors[:n_sensors])
            for i in range(n_sensors):
                for j in range(i + 1, n_sensors):
                    pair = f"{sensor_labels[i]} vs {sensor_labels[j]}"
                    pairwise[pair] = {
                        'statistic': float(tukey.statistic[i, j]),
                        'p_value': float(tukey.pvalue[i, j]),
                        'mean_diff': float(groups[i].mean() - groups[j].mean()),
                    }
        except Exception as exc:
            logging.warning(f"BiasComparator Tukey HSD failed: {exc}")

        return ComparisonResult(
            strategy_name='bias',
            stratum_label=int(stratum_df['cluster_label'].iloc[0]),
            stratum_dispersion=0.0,
            n_sensors=n_sensors,
            n_observations=n_obs,
            test_statistic=float(f_stat) if not np.isnan(f_stat) else float('nan'),
            p_value=float(p_value) if not np.isnan(p_value) else float('nan'),
            effect_size=float(eta_sq) if eta_sq is not None else None,
            detail={
                'per_sensor_mean': {
                    str(s): float(g.mean()) for s, g in zip(sensors, groups)
                },
                'pairwise': pairwise,
                'alpha': alpha,
            },
        )
