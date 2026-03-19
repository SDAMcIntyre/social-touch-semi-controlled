# file: motion_filter_interface.py
from abc import ABC, abstractmethod

import numpy as np


class MotionFilterInterface(ABC):
    """Abstract base class for 1-D motion smoothing filters.

    Each implementation receives a 1-D position signal (one axis of one sticker)
    and the recording sampling rate, and returns a smoothed signal of the same
    length.
    """

    @abstractmethod
    def filter(self, signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        """Apply the filter to *signal* and return the smoothed result.

        Args:
            signal: 1-D array of position values (mm).  Must not contain NaN —
                    callers are responsible for interpolating gaps beforehand.
            sampling_rate_hz: Acquisition frame rate (e.g. 30.0 for Azure Kinect).

        Returns:
            Smoothed 1-D array of the same shape as *signal*.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable label used in plot legends (e.g. ``"Butterworth 6 Hz"``)."""
        ...
