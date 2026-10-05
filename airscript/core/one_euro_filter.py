"""One Euro Filter implementation for low-lag jitter-free real-time trajectory filtering.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12).
"""
import math
import time
from typing import Optional, Tuple


class LowPassFilter:
    """First-order low-pass exponential smoothing filter."""

    def __init__(self, alpha: float = 1.0, initval: float = 0.0):
        self.alpha = alpha
        self.s = initval
        self.initialized = False

    def reset(self, initval: float = 0.0):
        self.s = initval
        self.initialized = False

    def filter(self, value: float) -> float:
        if self.initialized:
            result = self.alpha * value + (1.0 - self.alpha) * self.s
        else:
            result = value
            self.initialized = True
        self.s = result
        return result

    def filter_with_alpha(self, value: float, alpha: float) -> float:
        self.alpha = alpha
        return self.filter(value)

    def last_value(self) -> float:
        return self.s


class OneEuroFilter:
    """Adaptive 1D speed-based low-pass filter."""

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def __call__(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is not None and timestamp > self.last_time:
            dt = timestamp - self.last_time
            if dt > 1e-5:
                self.freq = 1.0 / dt
        self.last_time = timestamp

        # Estimate derivative of the signal
        prev_x = self.x_filter.last_value()
        dx = (x - prev_x) * self.freq if self.x_filter.initialized else 0.0

        # Filter the derivative
        a_d = self._alpha(self.d_cutoff)
        edx = self.dx_filter.filter_with_alpha(dx, a_d)

        # Compute adaptive cutoff for signal
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff)

        return self.x_filter.filter_with_alpha(x, a)


class PointFilter:
    """Multi-dimensional 1€ filter for (x, y, z) fingertip coordinates."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_z = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()

    def filter(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        fx = self.filter_x(x, timestamp)
        fy = self.filter_y(y, timestamp)
        fz = self.filter_z(z, timestamp)
        return fx, fy, fz
