# representation/feature_characterization/pressure.py
"""
Geometric pressure proxy extractor.

Computes ``contact_depth / contact_area`` per frame as a parameter-free proxy
for contact pressure concentration.  Unlike the model-based stress in
``MechanicsOfSolidsExtractor``, this metric carries no tissue assumptions and
is robust for cross-session / cross-subject comparisons.

Also computes contact velocity from sticker 3-D position using
``compute_velocity_magnitudes()`` from ``series_level.kinematics``.

Units: geo_pressure in mm⁻¹  (depth in mm, area in mm²  → ratio in mm⁻¹)
       geo_velocity  in mm/s

Output columns (depend on ``aggregation`` config param)
-------------------------------------------------------
aggregation='mean'  →  geo_pressure_mean, geo_velocity_mean
aggregation='max'   →  geo_pressure_max,  geo_velocity_max

Division safety: frames where ``contact_area <= 0`` yield NaN for pressure.
If all pressure frames are NaN, the pressure column is NaN; the velocity
column is still computed normally.
"""

import numpy as np
import pandas as pd
from .base import FeatureExtractor
from ..series_level.kinematics import get_kinematics
from ..series_level.pressure import get_geo_pressure


class PressureExtractor(FeatureExtractor):
    """
    Per-touch geometric pressure and velocity features.

    Config params
    -------------
    aggregation : {'mean', 'max'}
        Aggregation function applied to both signals.  Default: 'mean'.

    | Feature             | Formula                  | Units   |
    |---------------------|--------------------------|---------|
    | geo_pressure_mean   | nanmean(depth / area)    | mm⁻¹    |
    | geo_pressure_max    | nanmax(depth / area)     | mm⁻¹    |
    | geo_velocity_mean   | mean(velocity magnitudes)| mm/s    |
    | geo_velocity_max    | max(velocity magnitudes) | mm/s    |
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        aggregation = config.get('aggregation', 'mean')
        fps = config.get('fps', 30.0)
        if aggregation not in ('mean', 'max'):
            raise ValueError(
                f"PressureExtractor: unknown aggregation '{aggregation}'. "
                "Expected 'mean' or 'max'."
            )

        pressure = get_geo_pressure(group).values.astype(float)

        # Velocity magnitudes (mm/s) — always computed, NaN-free
        vel, _ = get_kinematics(group, fps=fps)
        velocity = vel.values.astype(float)

        if aggregation == 'mean':
            pressure_val = float(np.nan) if np.all(np.isnan(pressure)) else float(np.nanmean(pressure))
            velocity_val = float(np.mean(velocity)) if len(velocity) > 0 else float(np.nan)
            return {
                'geo_pressure_mean': pressure_val,
                'geo_velocity_mean': velocity_val,
            }
        else:  # 'max'
            pressure_val = float(np.nan) if np.all(np.isnan(pressure)) else float(np.nanmax(pressure))
            velocity_val = float(np.max(velocity)) if len(velocity) > 0 else float(np.nan)
            return {
                'geo_pressure_max': pressure_val,
                'geo_velocity_max': velocity_val,
            }
