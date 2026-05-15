# file: one_euro_filter.py
# One-Euro adaptive filter — Casiez, Roussel & Vogel, CHI 2012.
# The public `filter()` applies a forward-backward (zero-phase) variant:
# one causal pass forward, then a second causal pass on the reversed result,
# which is then reversed again.  This mirrors the `filtfilt` principle used by
# `ButterworthFilter` and eliminates phase lag for offline batch processing.
import math

import numpy as np

from .motion_filter_interface import MotionFilterInterface


class OneEuroFilter(MotionFilterInterface):
    """Adaptive low-pass filter whose cutoff rises with signal speed.

    At rest the cutoff drops to *min_cutoff*, producing aggressive smoothing.
    During fast motion the cutoff rises via the speed coefficient *beta*,
    preserving transient fidelity.  This avoids the fixed-cutoff trade-off
    between over-smoothing strokes and under-smoothing stationary periods.

    The public :meth:`filter` method applies the filter in a forward-backward
    (zero-phase) pass — identical in spirit to ``scipy.signal.filtfilt``.

    Args:
        min_cutoff: Minimum cutoff frequency in Hz (default 1.0).  Controls
                    the maximum smoothing applied at rest.
        beta: Speed coefficient in Hz·s/unit (default 0.007).  Controls how
              quickly the cutoff rises with instantaneous speed.
        d_cutoff: Fixed cutoff frequency (Hz) for the derivative smoother
                  (default 1.0).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be > 0; got {min_cutoff}.")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff must be > 0; got {d_cutoff}.")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0; got {beta}.")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def _smoothing_factor(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _one_euro_pass(self, signal: np.ndarray, dt: float) -> np.ndarray:
        n = len(signal)
        result = np.empty(n, dtype=float)

        alpha_d = self._smoothing_factor(self.d_cutoff, dt)

        dx_smooth = 0.0
        result[0] = signal[0]

        for i in range(1, n):
            dx_raw = (signal[i] - result[i - 1]) / dt
            dx_smooth = alpha_d * dx_raw + (1.0 - alpha_d) * dx_smooth

            fc = self.min_cutoff + self.beta * abs(dx_smooth)
            alpha = self._smoothing_factor(fc, dt)
            result[i] = alpha * signal[i] + (1.0 - alpha) * result[i - 1]

        return result

    def filter(self, signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
        if sampling_rate_hz <= 0.0:
            raise ValueError(
                f"sampling_rate_hz must be > 0; got {sampling_rate_hz}."
            )
        if signal.ndim != 1:
            raise ValueError(
                f"signal must be 1-D; got shape {signal.shape}."
            )
        dt = 1.0 / sampling_rate_hz
        forward = self._one_euro_pass(signal, dt)
        backward_reversed = self._one_euro_pass(forward[::-1], dt)
        return backward_reversed[::-1]

    def name(self) -> str:
        return f"One-Euro (min_fc={self.min_cutoff}, β={self.beta})"
