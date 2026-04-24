# representation/feature_characterization/touch_category.py
"""
Touch category feature extractor.

Encodes touch type (tap / stroke) and stroke direction (proximal / distal)
as one-hot binary integer columns.

Direction inference is delegated to representation.series_level.direction.infer_direction,
which is the single source of truth for the proximal/distal rule:
    proximal if end_y > start_y, else distal.
Single-frame strokes (start_y == end_y) resolve to distal (end_y <= start_y).

If 'type_metadata' or 'sticker_blue_position_y' is absent, all four columns
return 0 without raising.
"""

import pandas as pd
from .base import FeatureExtractor
from ..series_level.direction import infer_direction


class TouchCategoryExtractor(FeatureExtractor):
    """
    Per-touch binary one-hot encoding of touch type and stroke direction.

    | Column       | Value | Condition                                    |
    |--------------|-------|----------------------------------------------|
    | is_tap       | 1     | type_metadata == 'tap'                       |
    | is_stroke    | 1     | type_metadata == 'stroke'                    |
    | dir_proximal | 1     | stroke and sticker_blue_position_y increases |
    | dir_distal   | 1     | stroke and sticker_blue_position_y decreases |
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        result = {'is_tap': 0, 'is_stroke': 0, 'dir_proximal': 0, 'dir_distal': 0}

        if 'type_metadata' not in group.columns:
            return result

        touch_type = group['type_metadata'].iloc[0]

        if touch_type == 'tap':
            result['is_tap'] = 1
        elif touch_type == 'stroke':
            result['is_stroke'] = 1
            direction = infer_direction(group)
            if direction == 'proximal':
                result['dir_proximal'] = 1
            else:
                result['dir_distal'] = 1

        return result
