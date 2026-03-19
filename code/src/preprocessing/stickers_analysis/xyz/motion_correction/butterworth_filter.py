# file: butterworth_filter.py
import numpy as np
from scipy.signal import butter, filtfilt

from .motion_filter_interface import MotionFilterInterface


class ButterworthFilter(MotionFilterInterface):
    """Zero-phase low-pass Butterworth filter using ``scipy.signal.filtfilt``.

    ``filtfilt`` applies the filter twice (forward + backward) to eliminate
    temporal lag — important for accurately computing derivatives later.

    Args:
        order: Filter order (default 2).
        cutoff_hz: Cut-off frequency in Hz (default 6.0).  At 30 fps the
                   Nyquist limit is 15 Hz; 6 Hz retains voluntary hand motion
                   while attenuating depth-sensor noise.
    """

    def __init__(self, order: int = 2, cutoff_hz: float = 6.0) -> None:
        self._order = order
        self._cutoff_hz = cutoff_hz

    def filter(self, signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        nyquist = sampling_rate_hz / 2.0
        normalized_cutoff = self._cutoff_hz / nyquist
        b, a = butter(self._order, normalized_cutoff, btype="low", analog=False)
        return filtfilt(b, a, signal)

    def name(self) -> str:
        return f"Butterworth {self._cutoff_hz} Hz (order {self._order})"
