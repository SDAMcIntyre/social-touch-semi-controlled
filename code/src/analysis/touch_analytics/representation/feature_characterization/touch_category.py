# representation/feature_characterization/touch_category.py
"""
Touch category feature extractor.

Encodes touch type (tap / stroke) and stroke direction (proximal / distal)
as one-hot binary integer columns.

When the pre-computed ``gesture_type`` column is present (prepared CSV),
it is used directly.  When only ``type_metadata`` is available (raw CSV),
``is_tap`` / ``is_stroke`` are filled but ``dir_*`` columns remain 0.
"""

import pandas as pd
from .base import FeatureExtractor


class TouchCategoryExtractor(FeatureExtractor):
    """
    Per-touch binary one-hot encoding of touch type and stroke direction.

    | Column       | Value | Condition                                    |
    |--------------|-------|----------------------------------------------|
    | is_tap       | 1     | gesture_type == 'tap'                        |
    | is_stroke    | 1     | gesture_type in ('stroke_proximal', 'stroke_distal') |
    | dir_proximal | 1     | gesture_type == 'stroke_proximal'            |
    | dir_distal   | 1     | gesture_type == 'stroke_distal'              |
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        result = {'is_tap': 0, 'is_stroke': 0, 'dir_proximal': 0, 'dir_distal': 0}
        if 'gesture_type' in group.columns:
            gesture_type = group['gesture_type'].iloc[0]
            if gesture_type == 'tap':
                result['is_tap'] = 1
            elif gesture_type == 'stroke_proximal':
                result['is_stroke'] = 1
                result['dir_proximal'] = 1
            elif gesture_type == 'stroke_distal':
                result['is_stroke'] = 1
                result['dir_distal'] = 1
            return result
        if 'type_metadata' not in group.columns:
            return result
        touch_type = group['type_metadata'].iloc[0]
        if touch_type == 'tap':
            result['is_tap'] = 1
        elif touch_type == 'stroke':
            result['is_stroke'] = 1
        return result
