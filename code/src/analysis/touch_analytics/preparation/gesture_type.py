# preparation/gesture_type.py
import logging
import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import TOUCH_ID_COLS

_log = logging.getLogger(__name__)

_KNOWN_TYPES = {'tap', 'stroke'}
_GROUP_COLS = list(TOUCH_ID_COLS)  # ['block_order_id', 'trial_id', 'single_touch_id']
_CONTACT_LOCATION_COLS = ['contact_location_x', 'contact_location_y', 'contact_location_z']


def classify_gesture_type(group: pd.DataFrame) -> str:
    """
    Classify a single touch group as 'tap', 'stroke_proximal', or 'stroke_distal'.

    Stroke direction is determined by fitting a degree-1 polynomial (affine fit /
    linear regression) to the non-NaN ``contact_location_x`` values over frame
    index using ``numpy.polyfit``.  The sign of the slope decides direction: a
    positive slope (x increasing over time) → ``'stroke_proximal'``; a negative
    or zero slope → ``'stroke_distal'``.  Using the overall trend rather than
    only the endpoint delta makes the classification robust to noisy leading/
    trailing frames.

    Returns ``'stroke_unknown'`` when fewer than two valid (non-NaN)
    ``contact_location_x`` values are available, as a linear fit requires at
    least two points.

    Parameters
    ----------
    group : pd.DataFrame
        All frames for a single (block_order_id, trial_id, single_touch_id) touch.

    Returns
    -------
    str
        ``'tap'``, ``'stroke_proximal'``, ``'stroke_distal'``, or
        ``'stroke_unknown'`` (when fewer than two valid ``contact_location_x``
        values exist).

    Raises
    ------
    ValueError
        If ``type_metadata`` column is absent.
        If ``type_metadata`` value is not in ``{'tap', 'stroke'}``.
        If touch type is 'stroke' and any of ``contact_location_x/y/z`` is absent.
    """
    if 'type_metadata' not in group.columns:
        raise ValueError("classify_gesture_type: 'type_metadata' column is absent from group")

    touch_type = group['type_metadata'].iloc[0]
    if touch_type not in _KNOWN_TYPES:
        raise ValueError(
            f"classify_gesture_type: unknown type_metadata value {touch_type!r}; "
            f"expected one of {sorted(_KNOWN_TYPES)}"
        )

    if touch_type == 'tap':
        return 'tap'

    missing = [c for c in _CONTACT_LOCATION_COLS if c not in group.columns]
    if missing:
        raise ValueError(
            f"classify_gesture_type: contact location columns absent for a stroke group: "
            f"{missing}"
        )

    valid_x = group['contact_location_x'].dropna()
    if valid_x.empty:
        # No valid contact location — Kinect tracking was entirely absent for this touch.
        # Explicitly requested exception to fail-fast: return sentinel rather than raise.
        _log.warning(
            "classify_gesture_type: all contact_location_x values are NaN for a stroke "
            "group (block=%s, trial=%s, touch=%s) — labelled 'stroke_unknown'",
            group['block_order_id'].iloc[0],
            group['trial_id'].iloc[0],
            group['single_touch_id'].iloc[0],
        )
        return 'stroke_unknown'
    if len(valid_x) < 2:
        # A linear fit requires at least two points.
        # Explicitly requested exception to fail-fast: return sentinel rather than raise.
        _log.warning(
            "classify_gesture_type: only one valid contact_location_x value for a stroke "
            "group (block=%s, trial=%s, touch=%s) — labelled 'stroke_unknown'",
            group['block_order_id'].iloc[0],
            group['trial_id'].iloc[0],
            group['single_touch_id'].iloc[0],
        )
        return 'stroke_unknown'
    slope, _ = np.polyfit(np.arange(len(valid_x)), valid_x.values, 1)
    return 'stroke_proximal' if slope > 0 else 'stroke_distal'


def assign_gesture_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``gesture_type`` column to *df* by classifying each touch group.

    Groups where ``single_touch_id == 0`` are skipped; their rows receive NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame with at minimum ``block_order_id``, ``trial_id``,
        ``single_touch_id``, ``type_metadata``, and (for stroke rows)
        ``contact_location_x``, ``contact_location_y``, ``contact_location_z``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new ``gesture_type`` column appended.
    """
    df = df.copy()
    df['gesture_type'] = None

    touch_mask = df['single_touch_id'] != 0
    touch_df = df[touch_mask]

    gesture_map: dict[tuple, str] = {}
    for key, group in touch_df.groupby(_GROUP_COLS, sort=False):
        gesture_map[key] = classify_gesture_type(group)

    for key, label in gesture_map.items():
        block_id, trial_id, touch_id = key
        mask = (
            (df['block_order_id'] == block_id)
            & (df['trial_id'] == trial_id)
            & (df['single_touch_id'] == touch_id)
        )
        df.loc[mask, 'gesture_type'] = label

    return df
