# representation/feature_characterization/statistical.py
import pandas as pd
import numpy as np
from .base import FeatureExtractor
from ..series_level.kinematics import compute_velocity_magnitudes, compute_acceleration_magnitudes

_DEFAULT_AGGREGATIONS = ['mean', 'median', 'std', 'min', 'max', 'range', 'skewness']


class StatisticalExtractor(FeatureExtractor):
    """
    Summarizes each kinematic variable (depth, area, velocity, acceleration)
    with configurable statistical aggregations.
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        aggregations = config.get('aggregations', _DEFAULT_AGGREGATIONS)
        fps = config.get('fps', 30.0)

        vel = compute_velocity_magnitudes(group, fps=fps)
        accel = compute_acceleration_magnitudes(vel, fps=fps)

        variables = {
            'depth': group['contact_depth'],
            'area': group['contact_area'],
            'velocity': vel,
            'acceleration': accel,
        }

        row = {}
        for var_name, series in variables.items():
            for agg in aggregations:
                if agg == 'mean':
                    row[f'{var_name}_mean'] = series.mean()
                elif agg == 'median':
                    row[f'{var_name}_median'] = series.median()
                elif agg == 'std':
                    row[f'{var_name}_std'] = series.std(ddof=1)
                elif agg == 'min':
                    row[f'{var_name}_min'] = series.min()
                elif agg == 'max':
                    row[f'{var_name}_max'] = series.max()
                elif agg == 'range':
                    row[f'{var_name}_range'] = series.max() - series.min()
                elif agg == 'skewness':
                    try:
                        from scipy.stats import skew
                        row[f'{var_name}_skewness'] = float(skew(series.dropna()))
                    except ImportError:
                        row[f'{var_name}_skewness'] = np.nan
        return row
