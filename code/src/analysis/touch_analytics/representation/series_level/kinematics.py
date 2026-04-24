# representation/series_level/kinematics.py
from typing import List, Tuple

import numpy as np
import pandas as pd

DATA_RATE_HZ: float = 1000.0

STICKER_INPUT_COLUMNS: List[str] = [
    'sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z',
    'sticker_green_position_x', 'sticker_green_position_y', 'sticker_green_position_z',
    'sticker_yellow_position_x', 'sticker_yellow_position_y', 'sticker_yellow_position_z',
]

HAND_POSITION_COLUMNS: List[str] = [
    'hand_position_x', 'hand_position_y', 'hand_position_z',
]

HAND_VELOCITY_COLUMNS: List[str] = [
    'hand_velocity_x', 'hand_velocity_y', 'hand_velocity_z',
]

HAND_ACCELERATION_COLUMNS: List[str] = [
    'hand_acceleration_x', 'hand_acceleration_y', 'hand_acceleration_z',
]


def resolve_hand_position(group: pd.DataFrame) -> pd.DataFrame:
    """Select sticker based on contact_area_metadata and return hand_position_{x,y,z} DataFrame."""
    if 'contact_area_metadata' not in group.columns:
        raise KeyError(
            "resolve_hand_position: 'contact_area_metadata' column not found. "
            "This column is required for sticker selection."
        )
    metadata = str(group['contact_area_metadata'].iloc[0]).lower()
    if 'hand' in metadata:
        required = [
            'sticker_green_position_x', 'sticker_green_position_y', 'sticker_green_position_z',
            'sticker_yellow_position_x', 'sticker_yellow_position_y', 'sticker_yellow_position_z',
        ]
        missing = [c for c in required if c not in group.columns]
        if missing:
            raise KeyError(
                f"resolve_hand_position: columns {missing} not found. "
                f"Required for 'hand' contact type (contact_area_metadata='{metadata}')."
            )
        return pd.DataFrame({
            'hand_position_x': (group['sticker_green_position_x'] + group['sticker_yellow_position_x']) / 2,
            'hand_position_y': (group['sticker_green_position_y'] + group['sticker_yellow_position_y']) / 2,
            'hand_position_z': (group['sticker_green_position_z'] + group['sticker_yellow_position_z']) / 2,
        }, index=group.index)
    else:
        required = [
            'sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z',
        ]
        missing = [c for c in required if c not in group.columns]
        if missing:
            raise KeyError(
                f"resolve_hand_position: columns {missing} not found. "
                f"Required for fingertip contact type (contact_area_metadata='{metadata}')."
            )
        return pd.DataFrame({
            'hand_position_x': group['sticker_blue_position_x'],
            'hand_position_y': group['sticker_blue_position_y'],
            'hand_position_z': group['sticker_blue_position_z'],
        }, index=group.index)


def compute_velocity(
    group: pd.DataFrame,
    position_cols: List[str] | None = None,
) -> pd.DataFrame:
    """3D velocity components in mm/s — one row per frame, columns hand_velocity_x/y/z."""
    if position_cols is None:
        position_cols = HAND_POSITION_COLUMNS
    vectors = group[position_cols].diff().fillna(0) * DATA_RATE_HZ
    return vectors.rename(columns=dict(zip(position_cols, HAND_VELOCITY_COLUMNS)))


def compute_acceleration(vel_df: pd.DataFrame) -> pd.DataFrame:
    """3D acceleration components in mm/s² — one row per frame, columns hand_acceleration_x/y/z."""
    vectors = vel_df[HAND_VELOCITY_COLUMNS].diff().fillna(0) * DATA_RATE_HZ
    return vectors.rename(columns=dict(zip(HAND_VELOCITY_COLUMNS, HAND_ACCELERATION_COLUMNS)))


def get_kinematics(group: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (velocity_df, acceleration_df) with x/y/z component columns each.

    Fast-path 1: both pre-computed component columns present (augmented CSV already written).
    Fast-path 2: hand_position_* columns present (Stage 2a intermediate result).
    Slow-path: resolve sticker selection from contact_area_metadata and compute.
    """
    if all(c in group.columns for c in HAND_VELOCITY_COLUMNS) and \
            all(c in group.columns for c in HAND_ACCELERATION_COLUMNS):
        return group[HAND_VELOCITY_COLUMNS], group[HAND_ACCELERATION_COLUMNS]
    if all(c in group.columns for c in HAND_POSITION_COLUMNS):
        vel = compute_velocity(group)
        accel = compute_acceleration(vel)
        return vel, accel
    hand_pos = resolve_hand_position(group)
    vel = compute_velocity(hand_pos)
    accel = compute_acceleration(vel)
    return vel, accel
