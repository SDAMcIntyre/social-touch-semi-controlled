# preparation/interpolation.py
"""
Per-touch-group NaN-gap interpolation for session touch columns.

The aggregated CSVs are already at 1 kHz, but touch columns contain real values
only at every ~33rd row (30 Hz Kinect samples) with NaN between them. This module
fills those gaps with linear interpolation per touch group, producing continuous
signals suitable for downstream kinematics and feature extraction.

All Kinect-sampled columns use linear interpolation. Cubic splines were removed
because they overshoot near zero-crossings (contact_depth, contact_area) and
introduce unnecessary oscillation for position columns; the post-interpolation
clamp guards against any residual linear undershoot.
"""

import pandas as pd

from .grouping import group_touches
from analysis.pipeline.shared_constants import (
    NERVE_SPIKE_COL,
    CONTACT_POINTS_COL,
    NERVE_FREQ_COL,
)


# All Kinect-sampled columns that receive linear NaN-gap interpolation.
# 14 columns: 2 scalar (depth, area) + 9 sticker positions + 3 contact location.
INTERPOLATED_COLUMNS = [
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

ALREADY_1KHZ = [NERVE_SPIKE_COL, NERVE_FREQ_COL, 'Nerve_TTL', 'time']

BINARY_FFILL = ['led_on', 'contact_detected', CONTACT_POINTS_COL]

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
    interp_cols: list,
    ffill_cols: list,
) -> pd.DataFrame:
    """
    Fill NaN gaps in *group* for interpolated and ffill column sets.

    All interpolated columns (depth, area, sticker positions, contact location)
    use linear interpolation:
    - ≥ 2 non-NaN: linear interpolation (inside only)
    - 1 non-NaN: forward-fill then back-fill (constant; pandas linear needs ≥2 endpoints)
    - 0 non-NaN: leave as NaN

    Only columns explicitly listed are processed; all others are left untouched.
    """
    group = group.copy()

    for col in interp_cols:
        n_valid = group[col].notna().sum()
        if n_valid == 0:
            continue
        elif n_valid == 1:
            group[col] = group[col].ffill().bfill()
        else:
            group[col] = group[col].interpolate(method='linear', limit_direction='both', limit_area='inside')

    for col in ffill_cols:
        group[col] = group[col].ffill()

    return group


def interpolate_touch_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    All Kinect-sampled columns in :data:`INTERPOLATED_COLUMNS` are filled with
    linear interpolation. After interpolation, ``contact_depth`` and
    ``contact_area`` are clamped to ≥ 0 to guard against linear undershoot.

    Parameters
    ----------
    df : pd.DataFrame
        Session touch DataFrame at 1 kHz with NaN gaps in touch columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the same structure and index, NaN gaps filled.
    """
    df = df.copy()

    for col in GROUP_FFILL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].ffill()

    interp_cols = [c for c in INTERPOLATED_COLUMNS if c in df.columns]
    ffill_cols = [c for c in BINARY_FFILL if c in df.columns]

    groups = group_touches(df)

    for (_, _, touch_id), group in groups:
        if group.empty or touch_id == 0:
            continue
        interpolated = _interpolate_group(group, interp_cols, ffill_cols)
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
