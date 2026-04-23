# representation/feature_characterization/temporal.py
import numpy as np
import pandas as pd
from .base import FeatureExtractor

_SLOPE_MS = 167  # onset/offset slope window in milliseconds (~5 frames at 30 Hz)


class TemporalExtractor(FeatureExtractor):
    """
    Extracts time-domain features of each touch.

    All time values are in physical seconds (duration_s, time_to_peak_*_s)
    or physical units (auc_depth in mm·s, auc_area in mm²·s).
    Frame-count and frame-index aliases are kept for backwards compatibility.
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        fps = config.get('fps', 1000.0)
        slope_ms = config.get('slope_ms', _SLOPE_MS)
        slope_frames = max(2, int(round(slope_ms * fps / 1000.0)))

        depth = group['contact_depth'].values
        area = group['contact_area'].values
        n = len(depth)
        t = np.arange(n, dtype=float)
        t_s = t / fps  # time axis in seconds

        row = {
            'duration_frames': n,
            'duration_s': float(n / fps),
            'time_to_peak_depth': int(np.argmax(depth)),
            'time_to_peak_area': int(np.argmax(area)),
            'time_to_peak_depth_s': float(np.argmax(depth) / fps),
            'time_to_peak_area_s': float(np.argmax(area) / fps),
            'auc_depth': float(np.trapz(depth, t_s)),
            'auc_area': float(np.trapz(area, t_s)),
        }

        row['onset_slope_depth'] = _linear_slope(t_s[:slope_frames], depth[:slope_frames])
        row['offset_slope_depth'] = _linear_slope(t_s[-slope_frames:], depth[-slope_frames:])

        return row


def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return float('nan')
    coeffs = np.polyfit(t, y, 1)
    return float(coeffs[0])
