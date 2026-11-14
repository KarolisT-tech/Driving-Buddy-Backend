"""Jaw-clench threshold calibration and double-clench detection."""

from __future__ import annotations

from typing import Optional

import numpy as np


# Keep the existing jaw-clench detection parameters unchanged.
JAW_MAD_MULTIPLIER = 4.5
JAW_P95_MULTIPLIER = 1.55
JAW_BASELINE_MULTIPLIER = 1.85
JAW_MIN_THRESHOLD_UV = 1.5

MIN_CLENCH_SECONDS = 0.04
MAX_CLENCH_SECONDS = 0.55
JAW_RELEASE_RATIO = 0.65
MIN_GAP_SECONDS = 0.15


def calculate_jaw_threshold(values: list[float], scale: float):
    """Calculate the adaptive jaw-clench threshold in microvolts."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 20:
        raise RuntimeError(f"Not enough jaw calibration values: {x.size}")

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_sigma = 1.4826 * mad
    p95 = float(np.percentile(x, 95))

    base = max(
        median + JAW_MAD_MULTIPLIER * robust_sigma,
        p95 * JAW_P95_MULTIPLIER,
        median * JAW_BASELINE_MULTIPLIER,
        JAW_MIN_THRESHOLD_UV,
    )

    return base * scale, {
        "median": median,
        "robust_noise": robust_sigma,
        "p95": p95,
        "base_threshold": base,
    }


class DoubleClenchDetector:
    """Recognize two valid jaw-clench bursts within a time window."""

    def __init__(self, threshold: float, double_window: float):
        self.threshold = threshold
        self.double_window = double_window
        self.in_burst = False
        self.burst_start: Optional[float] = None
        self.first_clench_time: Optional[float] = None
        self.peak = 0.0

    def reset(self):
        self.in_burst = False
        self.burst_start = None
        self.first_clench_time = None
        self.peak = 0.0

    def update(self, rms_uv: float, timestamp: float):
        if self.first_clench_time is not None:
            if timestamp - self.first_clench_time > self.double_window:
                self.first_clench_time = None

        if not self.in_burst:
            if rms_uv >= self.threshold:
                self.in_burst = True
                self.burst_start = timestamp
                self.peak = rms_uv
            return "", None, None

        self.peak = max(self.peak, rms_uv)

        if rms_uv <= self.threshold * JAW_RELEASE_RATIO:
            duration = timestamp - self.burst_start
            peak = self.peak

            self.in_burst = False
            self.burst_start = None
            self.peak = 0.0

            if not MIN_CLENCH_SECONDS <= duration <= MAX_CLENCH_SECONDS:
                return "", duration, peak

            if self.first_clench_time is None:
                self.first_clench_time = timestamp
                return "first", duration, peak

            interval = timestamp - self.first_clench_time
            if MIN_GAP_SECONDS <= interval <= self.double_window:
                self.first_clench_time = None
                return "double", duration, peak

            self.first_clench_time = timestamp
            return "first", duration, peak

        if timestamp - self.burst_start > MAX_CLENCH_SECONDS:
            self.in_burst = False
            self.burst_start = None
            self.peak = 0.0

        return "", None, None
