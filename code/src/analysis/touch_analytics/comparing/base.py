# comparing/base.py
"""
Base classes for the comparing stage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ComparisonResult:
    strategy_name: str
    stratum_label: int
    stratum_dispersion: float
    n_sensors: int
    n_observations: int
    test_statistic: float
    p_value: float
    effect_size: Optional[float]
    detail: dict = field(default_factory=dict)


class ComparisonStrategy(ABC):
    @abstractmethod
    def compare(
        self,
        stratum_df: pd.DataFrame,
        sensor_col: str,
        measurement_col: str,
        config: dict,
    ) -> ComparisonResult:
        """
        Compare *measurement_col* across sensors within one stratum.

        Parameters
        ----------
        stratum_df
            Rows belonging to a single stratum (cluster_label == k).
        sensor_col
            Column identifying the sensor/session (e.g. 'session_id').
        measurement_col
            Column containing the measurement to compare (e.g. 'spike_elicited').
        config
            Strategy-specific options from the YAML comparing profile.

        Returns
        -------
        ComparisonResult
        """
