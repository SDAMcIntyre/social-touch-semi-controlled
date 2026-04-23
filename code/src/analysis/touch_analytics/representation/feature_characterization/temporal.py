# representation/feature_characterization/temporal.py
import numpy as np
import pandas as pd
from .base import FeatureExtractor

_SLOPE_FRAMES = 5  # number of frames used for onset/offset slope linear fit


class TemporalExtractor(FeatureExtractor):
    """
    Extracts time-domain features of each touch:
      - duration (frame count)
      - time_to_peak_depth / time_to_peak_area (frame index of maximum)
      - auc_depth / auc_area (trapezoid integration over frame indices)
      - onset_slope_depth / offset_slope_depth (linear fit on first/last N frames)
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        slope_frames = config.get('slope_frames', _SLOPE_FRAMES)
        depth = group['contact_depth'].values
        area = group['contact_area'].values
        n = len(depth)
        t = np.arange(n, dtype=float)

        row = {
            'duration_frames': n,
            'time_to_peak_depth': int(np.argmax(depth)),
            'time_to_peak_area': int(np.argmax(area)),
            'auc_depth': float(np.trapz(depth, t)),
            'auc_area': float(np.trapz(area, t)),
        }

        # Onset slope: linear fit on first slope_frames frames
        row['onset_slope_depth'] = _linear_slope(t[:slope_frames], depth[:slope_frames])
        row['offset_slope_depth'] = _linear_slope(t[-slope_frames:], depth[-slope_frames:])

        return row


def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return float('nan')
    coeffs = np.polyfit(t, y, 1)
    return float(coeffs[0])
