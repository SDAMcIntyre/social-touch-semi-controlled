# representation/series_level/velocity_scalar.py
import numpy as np
import pandas as pd

from .kinematics import HAND_VELOCITY_COLUMNS

HAND_VELOCITY_AMPLITUDE_COLUMN = 'hand_velocity_amplitude'
HAND_VELOCITY_SIGNED_COLUMN = 'hand_velocity_signed'


def compute_velocity_amplitude(vel_df: pd.DataFrame) -> pd.Series:
    """Return ‖(vx, vy, vz)‖₂ per frame — always ≥ 0."""
    V = vel_df[HAND_VELOCITY_COLUMNS].to_numpy()
    return pd.Series(np.sqrt((V ** 2).sum(axis=1)), index=vel_df.index, name=HAND_VELOCITY_AMPLITUDE_COLUMN)


def compute_velocity_signed(vel_df: pd.DataFrame) -> pd.Series:
    """Return the signed projection of velocity onto the per-touch principal motion axis.

    The principal axis u is the first right-singular vector of the centred velocity
    matrix V_c = V − mean(V).  u is flipped when mean(V) · u < 0 so that + consistently
    means motion in the dominant direction.  Projection uses raw (uncentred) velocity so
    the value is comparable to components and amplitude.

    Edge case: when the first singular value < 1e-12 (effectively zero motion), returns
    zeros — this is mathematically correct, not an error condition.
    """
    V = vel_df[HAND_VELOCITY_COLUMNS].to_numpy()
    V_mean = V.mean(axis=0)
    V_centred = V - V_mean
    _, s, Vt = np.linalg.svd(V_centred, full_matrices=False)
    if s[0] < 1e-12:
        return pd.Series(np.zeros(len(vel_df)), index=vel_df.index, name=HAND_VELOCITY_SIGNED_COLUMN)
    u = Vt[0]
    if np.dot(V_mean, u) < 0:
        u = -u
    return pd.Series(V @ u, index=vel_df.index, name=HAND_VELOCITY_SIGNED_COLUMN)
