# representation/series_level/direction.py
"""
Stroke direction inference for semi-controlled touch data.

Rule: a stroke is 'proximal' if ``sticker_blue_position_y`` increases
(end_y > start_y), 'distal' otherwise (end_y <= start_y, including the
single-frame edge case).  Non-stroke touches are 'static'.

.. deprecated::
    ``infer_direction`` is superseded by ``classify_gesture_type`` in
    ``preparation.gesture_type``.  All pipeline code now reads the pre-computed
    ``gesture_type`` column.  This module will be removed in a future cleanup.
"""

import warnings
import pandas as pd


def infer_direction(group: pd.DataFrame) -> str:
    """
    Infer the direction of a single touch group.

    Parameters
    ----------
    group : pd.DataFrame
        All frames for a single (block_order_id, trial_id, single_touch_id) touch.
        Must contain ``type_metadata`` and, for strokes, ``sticker_blue_position_y``.

    Returns
    -------
    str
        ``'proximal'`` — stroke moving in the +y direction (end_y > start_y).
        ``'distal'``   — stroke moving in the -y direction (end_y <= start_y).
        ``'static'``   — touch type is not 'stroke', or columns are absent.
    """
    warnings.warn(
        "infer_direction() is deprecated. Use classify_gesture_type() from "
        "preparation.gesture_type instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    touch_type = (
        group['type_metadata'].iloc[0]
        if 'type_metadata' in group.columns
        else None
    )

    if touch_type != 'stroke':
        return 'static'

    if 'sticker_blue_position_y' not in group.columns:
        return 'static'

    valid_y = group['sticker_blue_position_y'].dropna()
    if valid_y.empty:
        return 'static'
    start_y = valid_y.iloc[0]
    end_y = valid_y.iloc[-1]
    return 'proximal' if end_y > start_y else 'distal'
