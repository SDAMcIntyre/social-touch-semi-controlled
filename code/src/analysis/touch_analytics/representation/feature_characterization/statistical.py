# representation/feature_characterization/statistical.py
import pandas as pd
import numpy as np
from .base import FeatureExtractor

try:
    from scipy.stats import skew as _scipy_skew
except ImportError:
    _scipy_skew = None

_DEFAULT_AGGREGATIONS = ['mean', 'median', 'std', 'min', 'max', 'range', 'skewness']

_EXCLUDE_FROM_AGGREGATION = frozenset({
    # Orchestrator / touch-ID columns
    'block_order_id', 'trial_id', 'single_touch_id',
    'type_metadata', 'gesture_type',
    'mean_contact_x', 'mean_contact_y', 'mean_contact_z',
    'spike_elicited', 'session_id',
    # Frame indexing
    'frame', 'frame_id', 'Unnamed: 0',
    # Source metadata
    'source_block_file', 'contact_area_metadata',
    # Nerve event marker (binary, not a continuous signal)
    'Nerve_spike',
    # Raw sticker positions (consumed by hand_position transform)
    'sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z',
    'sticker_green_position_x', 'sticker_green_position_y', 'sticker_green_position_z',
    'sticker_yellow_position_x', 'sticker_yellow_position_y', 'sticker_yellow_position_z',
})


class StatisticalExtractor(FeatureExtractor):
    """
    Generic aggregator: discovers all numeric columns in the input and
    summarizes each with configurable statistical aggregations.
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        aggregations = config.get('aggregations', _DEFAULT_AGGREGATIONS)

        numeric_cols = [
            col for col in group.columns
            if col not in _EXCLUDE_FROM_AGGREGATION
            and pd.api.types.is_numeric_dtype(group[col])
        ]

        row = {}
        for col in numeric_cols:
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
                    if _scipy_skew is not None:
                        row[f'{col}_skewness'] = float(_scipy_skew(series.dropna()))
                    else:
                        row[f'{col}_skewness'] = np.nan
        return row
