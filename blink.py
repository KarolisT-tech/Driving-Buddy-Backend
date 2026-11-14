"""Individual blink threshold calibration and EEG peak detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


# Keep the existing blink detector parameters unchanged.
BLINK_MIN_WIDTH_SECONDS = 0.05
BLINK_MAX_WIDTH_SECONDS = 0.40
BLINK_MIN_DISTANCE_SECONDS = 0.25
BLINK_PEAK_CONFIRM_SECONDS = 0.25
BLINK_BUFFER_SECONDS = 4.0


@dataclass(frozen=True)
class BlinkEvent:
    """One confirmed blink, identified by its original EEG sample number."""

    sample_id: int



def calculate_blink_threshold(
    values: list[float],
    blink_k: float,
    min_excess_uv: float,
):
    """Calculate the positive blink peak threshold in microvolts."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 100:
        raise RuntimeError(f"Not enough blink calibration samples: {x.size}")

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))  # unscaled MAD, as in notebook
    excess = max(blink_k * mad, min_excess_uv)
    threshold = median + excess

    return threshold, {
        "median": median,
        "mad": mad,
        "excess": excess,
        "threshold": threshold,
    }


class BlinkDetector:
    """Detect individual blink peaks in the filtered frontal-occipital signal.

    The detector owns its rolling signal/sample buffers and remembers the last
    blink it emitted, so the main program does not have to manage that state.
    """

    def __init__(
        self,
        threshold: float,
        prominence: float,
        sample_rate: int,
        initial_last_sample: Optional[int] = None,
    ):
        self.threshold = threshold
        self.prominence = prominence
        self.sample_rate = sample_rate
        self.last_blink_sample = initial_last_sample

        buffer_size = int(BLINK_BUFFER_SECONDS * sample_rate)
        self.signal_buffer = deque(maxlen=buffer_size)
        self.sample_id_buffer = deque(maxlen=buffer_size)

    def add_samples(self, signal_values, sample_ids):
        """Add newly filtered samples; call detect() after each input chunk."""
        for value, sample_id in zip(signal_values, sample_ids):
            if np.isfinite(value):
                self.signal_buffer.append(float(value))
                self.sample_id_buffer.append(int(sample_id))

    def detect(self) -> list[BlinkEvent]:
        """Return newly confirmed blink events from the rolling buffer."""
        if len(self.signal_buffer) < int(1.0 * self.sample_rate):
            return []

        signal = np.asarray(self.signal_buffer, dtype=float)
        sample_ids = np.asarray(self.sample_id_buffer, dtype=np.int64)

        min_distance = max(
            1,
            int(BLINK_MIN_DISTANCE_SECONDS * self.sample_rate),
        )
        min_width = max(
            1,
            int(BLINK_MIN_WIDTH_SECONDS * self.sample_rate),
        )
        max_width = max(
            min_width + 1,
            int(BLINK_MAX_WIDTH_SECONDS * self.sample_rate),
        )

        peaks, _ = find_peaks(
            signal,
            height=self.threshold,
            prominence=self.prominence,
            distance=min_distance,
            width=(min_width, max_width),
        )

        # A peak near the right edge may still be rising or changing width.
        stable_end = (
            len(signal)
            - int(BLINK_PEAK_CONFIRM_SECONDS * self.sample_rate)
        )

        events: list[BlinkEvent] = []

        for peak_index in peaks:
            if peak_index >= stable_end:
                continue

            peak_sample = int(sample_ids[peak_index])

            # Prevent the same peak from being emitted repeatedly.
            if (
                self.last_blink_sample is not None
                and peak_sample <= self.last_blink_sample
            ):
                continue

            self.last_blink_sample = peak_sample
            events.append(BlinkEvent(sample_id=peak_sample))

        return events

    def clear_buffer(self):
        """Clear buffered signal after an EEG sample gap."""
        self.signal_buffer.clear()
        self.sample_id_buffer.clear()

    def reset_sample_tracking(self):
        """Reset peak de-duplication if the HALO sample counter restarts."""
        self.last_blink_sample = None
        self.clear_buffer()
