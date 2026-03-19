# file: savgol_filter.py
import numpy as np
from scipy.signal import savgol_filter

from .motion_filter_interface import MotionFilterInterface


class SavgolFilter(MotionFilterInterface):
    """Savitzky-Golay polynomial smoothing filter.

    Fits successive sub-windows of the signal with a polynomial of the given
    order, reducing noise while approximately preserving peak positions.

    Args:
        window_length: Number of frames in each smoothing window (must be odd,
                       default 11 ≈ 0.37 s at 30 fps).
        polyorder: Order of the fitting polynomial (default 3).  Must be less
                   than *window_length*.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3) -> None:
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")
        self._window_length = window_length
        self._polyorder = polyorder

    def filter(self, signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:  # noqa: ARG002
        return savgol_filter(signal, self._window_length, self._polyorder)

    def name(self) -> str:
        return f"Savitzky-Golay (win={self._window_length}, poly={self._polyorder})"
