# file: outlier_detector.py
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OutlierConfig:
    """Configuration for the outlier detector.

    Both detection methods are independently toggleable.  At least one must be
    enabled for the detector to flag any frames.

    Hard-threshold detection compares instantaneous velocity/acceleration
    magnitudes to physical limits; statistical detection flags points whose
    deviation exceeds a MAD-based envelope.
    """

    enable_hard_threshold: bool = True
    max_velocity_mm_per_s: float = 2000.0
    max_acceleration_mm_per_s2: float = 50000.0

    enable_statistical: bool = True
    mad_multiplier: float = 5.0


class OutlierDetector:
    """Detects and interpolates physically impossible motion spikes in a 1-D
    position signal.

    Detection methods:
    - **Hard threshold:** flags frames where the velocity (first finite
      difference × frame rate) or acceleration (second finite difference ×
      frame rate²) exceeds a configurable physical limit.
    - **Statistical (MAD-based):** flags frames where the velocity magnitude
      deviates from the median by more than ``mad_multiplier × MAD``.

    The two methods are OR-combined when both are enabled.
    """

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self._cfg = config if config is not None else OutlierConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, position_series: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        """Return a boolean mask of outlier frames.

        Args:
            position_series: 1-D position array in mm.  NaN values are
                             treated as non-outlier (they are already gaps;
                             callers handle them separately).
            sampling_rate_hz: Recording frame rate.

        Returns:
            Boolean array of the same length as *position_series*; ``True``
            where a frame is considered an outlier.
        """
        n = len(position_series)
        mask = np.zeros(n, dtype=bool)

        if n < 3:
            return mask

        # Velocity: mm per second (central-difference approximation for interior)
        velocity = np.gradient(position_series, 1.0 / sampling_rate_hz)
        # Acceleration: mm per second squared
        acceleration = np.gradient(velocity, 1.0 / sampling_rate_hz)

        cfg = self._cfg

        if cfg.enable_hard_threshold:
            mask |= np.abs(velocity) > cfg.max_velocity_mm_per_s
            mask |= np.abs(acceleration) > cfg.max_acceleration_mm_per_s2

        if cfg.enable_statistical:
            speed = np.abs(velocity)
            median_speed = np.nanmedian(speed)
            mad = np.nanmedian(np.abs(speed - median_speed))
            threshold = median_speed + cfg.mad_multiplier * mad
            mask |= speed > threshold

        # Do not flag existing NaN positions as outliers — they are already
        # handled by the NaN-interpolation step upstream.
        mask[np.isnan(position_series)] = False

        return mask

    def interpolate_outliers(
        self,
        position_series: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Replace flagged frames with linear interpolation from neighbours.

        Frames at the very start or end of the signal that are flagged are
        replaced by the nearest non-outlier value (edge-clamping) so the
        signal length is always preserved.

        Args:
            position_series: 1-D position array in mm.
            mask: Boolean outlier mask (same length as *position_series*).

        Returns:
            A copy of *position_series* with outlier frames replaced.
        """
        if not np.any(mask):
            return position_series.copy()

        cleaned = position_series.copy()
        n = len(cleaned)
        valid_indices = np.flatnonzero(~mask)

        if len(valid_indices) == 0:
            # All frames are outliers — nothing to interpolate from.
            return cleaned

        # Linear interpolation: numpy interp clamps outside the valid range.
        outlier_indices = np.flatnonzero(mask)
        cleaned[outlier_indices] = np.interp(
            outlier_indices, valid_indices, cleaned[valid_indices]
        )
        return cleaned
