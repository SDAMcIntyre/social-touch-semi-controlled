# preparation/gesture_type.py
import pandas as pd

_KNOWN_TYPES = {'tap', 'stroke'}
_GROUP_COLS = ['block_order_id', 'trial_id', 'single_touch_id']
_CONTACT_LOCATION_COLS = ['contact_location_x', 'contact_location_y', 'contact_location_z']


def classify_gesture_type(group: pd.DataFrame) -> str:
    """
    Classify a single touch group as 'tap', 'stroke_proximal', or 'stroke_distal'.

    Stroke direction is determined from the 3D contact location: a stroke is
    proximal when ``contact_location_x`` increases from first to last frame
    (Δx > 0), distal otherwise.

    Parameters
    ----------
    group : pd.DataFrame
        All frames for a single (block_order_id, trial_id, single_touch_id) touch.

    Returns
    -------
    str
        ``'tap'``, ``'stroke_proximal'``, or ``'stroke_distal'``.

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

    start_x = group['contact_location_x'].iloc[0]
    end_x = group['contact_location_x'].iloc[-1]
    return 'stroke_proximal' if end_x > start_x else 'stroke_distal'


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
