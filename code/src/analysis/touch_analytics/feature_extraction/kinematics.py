# feature_extraction/kinematics.py
import numpy as np
import pandas as pd


def compute_velocity_magnitudes(group: pd.DataFrame) -> pd.Series:
    """3D sticker position first-difference magnitudes (one value per frame)."""
    vectors = group[
        ['sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z']
    ].diff().fillna(0)
    return np.sqrt(vectors.pow(2).sum(axis=1))


def compute_acceleration_magnitudes(velocity_magnitudes: pd.Series) -> pd.Series:
    """Velocity magnitude first-difference magnitudes (one value per frame)."""
    velocity_vectors = velocity_magnitudes.diff().fillna(0)
    return velocity_vectors.abs()
