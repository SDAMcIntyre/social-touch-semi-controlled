# representation/feature_characterization/statistical.py
import pandas as pd
import numpy as np
from .base import FeatureExtractor

_DEFAULT_AGGREGATIONS = ['mean', 'median', 'std', 'min', 'max', 'range', 'skewness']

EXCLUDED_QUANTITATIVE_COLUMNS: frozenset[str] = frozenset({
    # Identifiers / indices
    'block_order_id', 'trial_id', 'single_touch_id', 'frame_id',
    # Time base
    'time', 't', 'timestamp',
    # Event channel — binary; spike_elicited is the aggregate
    'Nerve_spike',
    # Raw sticker positions — subsumed by velocity_magnitude / acceleration_magnitude
    'sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z',
    'sticker_green_position_x', 'sticker_green_position_y', 'sticker_green_position_z',
    'sticker_red_position_x',  'sticker_red_position_y',  'sticker_red_position_z',
})


def _discover_quantitative_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.select_dtypes(include='number').columns
        if col not in EXCLUDED_QUANTITATIVE_COLUMNS
    ]


class StatisticalExtractor(FeatureExtractor):
    """
    Summarizes every quantitative column (minus the exclude list) with
    configurable statistical aggregations.
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        aggregations = config.get('aggregations', _DEFAULT_AGGREGATIONS)
        columns = _discover_quantitative_columns(group)

        row = {}
        for col in columns:
            series = group[col]
            for agg in aggregations:
                if agg == 'mean':
                    row[f'{col}_mean'] = series.mean()
                elif agg == 'median':
                    row[f'{col}_median'] = series.median()
                elif agg == 'std':
                    row[f'{col}_std'] = series.std(ddof=1)
                elif agg == 'min':
                    row[f'{col}_min'] = series.min()
                elif agg == 'max':
                    row[f'{col}_max'] = series.max()
                elif agg == 'range':
                    row[f'{col}_range'] = series.max() - series.min()
                elif agg == 'skewness':
                    try:
                        from scipy.stats import skew
                        row[f'{col}_skewness'] = float(skew(series.dropna()))
                    except ImportError:
                        row[f'{col}_skewness'] = np.nan
        return row
