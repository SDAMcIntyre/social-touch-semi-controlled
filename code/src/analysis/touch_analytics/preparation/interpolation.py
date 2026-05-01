# preparation/interpolation.py
"""
Per-touch-group NaN-gap interpolation for session touch columns.

The aggregated CSVs are already at 1 kHz, but touch columns contain real values
only at every ~33rd row (30 Hz Kinect samples) with NaN between them. This module
fills those gaps with cubic or linear interpolation per touch group, producing
continuous signals suitable for downstream kinematics and feature extraction.
"""

import pandas as pd

from .grouping import group_touches


CONTINUOUS_COLUMNS = [
    'contact_depth',
    'contact_area',
    'sticker_blue_position_x',
    'sticker_blue_position_y',
    'sticker_blue_position_z',
    'sticker_green_position_x',
    'sticker_green_position_y',
    'sticker_green_position_z',
    'sticker_yellow_position_x',
    'sticker_yellow_position_y',
    'sticker_yellow_position_z',
    'contact_location_x',
    'contact_location_y',
    'contact_location_z',
]

ALREADY_1KHZ = ['Nerve_spike', 'Nerve_freq', 'Nerve_TTL', 'time']

BINARY_FFILL = ['led_on', 'contact_detected', 'contact_points']

# Kinect-derived columns that are present only at ~30 Hz sample rows.
# Forward-filled to 1 kHz before grouping so that every intermediate row
# is assigned to the correct touch group.
GROUP_FFILL_COLUMNS = [
    'trial_id',
    'single_touch_id',
    'type_metadata',
    'speed_metadata',
    'contact_area_metadata',
    'force_metadata',
]


def _interpolate_group(
    group: pd.DataFrame,
    cols: list,
    ffill_cols: list,
    method: str,
) -> pd.DataFrame:
    """
    Fill NaN gaps in *group* for each column in *cols* (interpolated) and
    *ffill_cols* (forward-filled).

    Interpolation degrades gracefully based on the number of non-NaN values:
    - ≥ 4 non-NaN: use *method* (cubic by default)
    - 2 or 3 non-NaN: fall back to linear
    - 1 non-NaN: forward-fill then back-fill (constant)
    - 0 non-NaN: leave as NaN

    Only columns explicitly listed in *cols* or *ffill_cols* are processed;
    all other columns are left untouched.

    Parameters
    ----------
    group : pd.DataFrame
        Slice of a session DataFrame for a single touch group.
    cols : list
        Column names to interpolate (pre-filtered to those present in group).
    ffill_cols : list
        Column names to forward-fill (pre-filtered to those present in group).
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
            group[col] = group[col].interpolate(method='linear', limit_direction='both', limit_area='inside')
        else:
            group[col] = group[col].interpolate(method=method, limit_direction='both', limit_area='inside')

    for col in ffill_cols:
        group[col] = group[col].ffill()

    return group


def interpolate_touch_columns(df: pd.DataFrame, method: str = 'cubic') -> pd.DataFrame:
    """
    Fill NaN gaps in touch columns for every touch group in *df*.

    Before grouping, forward-fills :data:`GROUP_FFILL_COLUMNS` (``trial_id``,
    ``single_touch_id``, and the four ``*_metadata`` columns) so that every
    intermediate 1 kHz row inherits the label of its surrounding 30 Hz Kinect
    sample.  This makes ``group_touches`` include all 1 kHz rows within each
    touch span, not just the ~30 Hz Kinect sample rows.

    Groups the DataFrame by ``['block_order_id', 'trial_id', 'single_touch_id']``
    via :func:`group_touches` and applies :func:`_interpolate_group` to each
    group. Groups with ``single_touch_id == 0`` are skipped (non-touch
    inter-trial frames).

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

    for col in GROUP_FFILL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].ffill()

    cols = [c for c in CONTINUOUS_COLUMNS if c in df.columns]
    ffill_cols = [c for c in BINARY_FFILL if c in df.columns]

    groups = group_touches(df)

    for (_, _, touch_id), group in groups:
        if group.empty or touch_id == 0:
            continue
        interpolated = _interpolate_group(group, cols, ffill_cols, method)
        df.loc[interpolated.index] = interpolated


    cols = ['contact_depth', 'contact_area']
    existing = [c for c in cols if c in df.columns]

    if existing:
        # Interpolation between two consecutive zero-contact regions can introduce
        # small numerical noise: one variable may go slightly negative while the other
        # stays positive. Set both to 0 whenever either is negative to keep them
        # physically consistent.
        mask = (df[existing] < 0).any(axis=1)
        df.loc[mask, existing] = 0

    return df
