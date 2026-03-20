# feature_extraction/max_extractor.py
import pandas as pd
from .base import FeatureExtractor
from .kinematics import compute_velocity_magnitudes, compute_acceleration_magnitudes


class MaxExtractor(FeatureExtractor):
    """
    Replicates the original touch_analysis._process_touch_analysis() behavior:
    max depth, max contact area, max velocity, max acceleration.
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        fps = config.get('fps', 30.0)
        max_depth = group['contact_depth'].max()
        max_contact_area = group['contact_area'].max()

        vel_magnitudes = compute_velocity_magnitudes(group, fps=fps)
        accel_magnitudes = compute_acceleration_magnitudes(vel_magnitudes, fps=fps)

        return {
            'max_depth': max_depth,
            'max_contact_area': max_contact_area,
            'max_velocity': vel_magnitudes.max(),
            'max_acceleration': accel_magnitudes.max(),
        }
