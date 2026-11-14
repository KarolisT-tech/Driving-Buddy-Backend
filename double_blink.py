"""Recognize single and double blink events within a configured time window."""

from __future__ import annotations

from typing import Optional


class DoubleBlinkDetector:
    """Track blink timestamps and recognize single blinks and blink pairs.

    update() returns the time between the two blinks when a pair is found;
    otherwise it returns None. After a pair, the next blink starts a new pair.

    poll_single() returns True once, when a blink was NOT followed by a second
    blink within the double window (i.e. it was a single blink).
    """

    def __init__(self, double_window: float, confirm_delay: float = 0.0):
        self.double_window = double_window
        # Extra wait so a second blink that is still being confirmed by the
        # peak detector is not mistaken for "no second blink".
        self.confirm_delay = confirm_delay
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

    def poll_single(self, now: float) -> bool:
        """Call regularly with the current signal time (seconds)."""
        if self.last_blink_time is None:
            return False

        if now - self.last_blink_time > self.double_window + self.confirm_delay:
            self.last_blink_time = None
            return True

        return False
