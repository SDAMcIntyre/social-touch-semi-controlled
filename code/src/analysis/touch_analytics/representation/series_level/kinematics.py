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


def compute_velocity_magnitudes(
    group: pd.DataFrame,
    position_cols: List[str] | None = None,
) -> pd.Series:
    """3D position velocity magnitudes in mm/s (one value per frame)."""
    if position_cols is None:
        position_cols = HAND_POSITION_COLUMNS
    vectors = group[position_cols].diff().fillna(0)
    return np.sqrt(vectors.pow(2).sum(axis=1)) * DATA_RATE_HZ


def compute_acceleration_magnitudes(velocity_magnitudes: pd.Series) -> pd.Series:
    """Velocity magnitude first-difference magnitudes in mm/s² (one value per frame)."""
    velocity_vectors = velocity_magnitudes.diff().fillna(0)
    return velocity_vectors.abs() * DATA_RATE_HZ


def get_kinematics(group: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (velocity_magnitude, acceleration_magnitude) series.

    Fast-path 1: both pre-computed columns present (augmented CSV already written).
    Fast-path 2: hand_position_* columns present (Stage 2a intermediate result).
    Slow-path: resolve sticker selection from contact_area_metadata and compute.
    """
    if 'velocity_magnitude' in group.columns and 'acceleration_magnitude' in group.columns:
        return group['velocity_magnitude'], group['acceleration_magnitude']
    if all(c in group.columns for c in HAND_POSITION_COLUMNS):
        vel = compute_velocity_magnitudes(group)
        accel = compute_acceleration_magnitudes(vel)
        return vel, accel
    hand_pos = resolve_hand_position(group)
    vel = compute_velocity_magnitudes(hand_pos)
    accel = compute_acceleration_magnitudes(vel)
    return vel, accel
