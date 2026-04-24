# preparation/interpolation.py
"""
Per-touch-group NaN-gap interpolation for session touch columns.

The aggregated CSVs are already at 1 kHz, but touch columns contain real values
only at every ~33rd row (30 Hz Kinect samples) with NaN between them. This module
fills those gaps with cubic or linear interpolation per touch group, producing
continuous signals suitable for downstream kinematics and feature extraction.
"""

import pandas as pd


CONTINUOUS_COLUMNS = [
    'contact_depth',
    'contact_area',
    'sticker_blue_position_x',
    'sticker_blue_position_y',
    'sticker_blue_position_z',
    'contact_location_x',
    'contact_location_y',
    'contact_location_z',
]

ALREADY_1KHZ = ['Nerve_spike', 'Nerve_freq', 'Nerve_TTL', 'time']

BINARY_FFILL = ['led_on', 'contact_detected']

IDENTITY = ['block_order_id', 'trial_id', 'single_touch_id', 'type_metadata']

_KNOWN_COLUMNS = set(CONTINUOUS_COLUMNS) | set(ALREADY_1KHZ) | set(BINARY_FFILL) | set(IDENTITY)


def _interpolate_group(group: pd.DataFrame, cols: list, method: str) -> pd.DataFrame:
    """
    Fill NaN gaps in *group* for each column in *cols*.

    The interpolation method degrades gracefully based on the number of
    non-NaN values available in the group:
    - ≥ 4 non-NaN: use *method* (cubic by default)
    - 2 or 3 non-NaN: fall back to linear
    - 1 non-NaN: forward-fill then back-fill (constant)
    - 0 non-NaN: leave as NaN

    Unknown columns (not in any classification list) are also filled:
    numeric ones are interpolated, non-numeric ones are forward-filled.

    Parameters
    ----------
    group : pd.DataFrame
        Slice of a session DataFrame for a single touch group.
    cols : list
        Column names to interpolate (pre-filtered to those present in group).
    method : str
        pandas interpolation method to use when ≥ 4 non-NaN values exist.

    Returns
    -------
    pd.DataFrame
        Copy of *group* with NaN gaps filled.
    """
    group = group.copy()

    for col in cols:
        n_valid = group[col].notna().sum()
        if n_valid == 0:
            continue
        elif n_valid == 1:
            group[col] = group[col].ffill().bfill()
        elif n_valid < 4:
            group[col] = group[col].interpolate(method='linear', limit_direction='both')
        else:
            group[col] = group[col].interpolate(method=method, limit_direction='both')

    # Handle unknown columns (not in any classification list)
    for col in group.columns:
        if col in _KNOWN_COLUMNS or col in cols:
            continue
        if pd.api.types.is_numeric_dtype(group[col]):
            n_valid = group[col].notna().sum()
            if n_valid == 0:
                continue
            elif n_valid == 1:
                group[col] = group[col].ffill().bfill()
            elif n_valid < 4:
                group[col] = group[col].interpolate(method='linear', limit_direction='both')
            else:
                group[col] = group[col].interpolate(method=method, limit_direction='both')
        else:
            group[col] = group[col].ffill()

    return group


def interpolate_touch_columns(df: pd.DataFrame, method: str = 'cubic') -> pd.DataFrame:
    """
    Fill NaN gaps in touch columns for every touch group in *df*.

    Groups the DataFrame by ``['block_order_id', 'trial_id', 'single_touch_id']``
    and applies :func:`_interpolate_group` to each group. Groups with
    ``single_touch_id == 0`` are included (interpolated for continuity).

    After interpolation, ``contact_depth`` and ``contact_area`` are clamped
    to ≥ 0 to guard against cubic spline overshoot.

    Parameters
    ----------
    df : pd.DataFrame
        Session touch DataFrame at 1 kHz with NaN gaps in touch columns.
    method : str
        Interpolation method passed to ``pd.Series.interpolate`` when a group
        has ≥ 4 non-NaN values. Default is ``'cubic'``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the same structure and index, NaN gaps filled.
    """
    df = df.copy()

    cols = [c for c in CONTINUOUS_COLUMNS if c in df.columns]

    group_keys = ['block_order_id', 'trial_id', 'single_touch_id']
    present_keys = [k for k in group_keys if k in df.columns]

    groups = df.groupby(present_keys, sort=False, group_keys=False)
    df = groups.apply(lambda g: _interpolate_group(g, cols, method))

    if 'contact_depth' in df.columns:
        df['contact_depth'] = df['contact_depth'].clip(lower=0)
    if 'contact_area' in df.columns:
        df['contact_area'] = df['contact_area'].clip(lower=0)

    return df
