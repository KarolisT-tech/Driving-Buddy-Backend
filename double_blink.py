"""Recognize two individual blink events within a configured time window."""

from __future__ import annotations

from typing import Optional


class DoubleBlinkDetector:
    """Track blink timestamps and recognize a pair of blinks.

    update() returns the time between the two blinks when a pair is found;
    otherwise it returns None. After a pair, the next blink starts a new pair.
    """

    def __init__(self, double_window: float):
        self.double_window = double_window
        self.last_blink_time: Optional[float] = None

    def reset(self):
        self.last_blink_time = None

    def update(self, blink_time: float) -> Optional[float]:
        if self.last_blink_time is not None:
            gap = blink_time - self.last_blink_time

            if 0.0 < gap <= self.double_window:
                # Reset the pair so a third blink starts a new pair.
                self.last_blink_time = None
                return gap

            self.last_blink_time = blink_time
            return None

        self.last_blink_time = blink_time
        return None
