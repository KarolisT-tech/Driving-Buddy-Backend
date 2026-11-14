"""Occipital EEG calibration and smooth score-based drowsiness detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, welch


# EEG frequency bands (Hz)
EEG_BANDPASS_LOW_HZ = 2.0
EEG_BANDPASS_HIGH_HZ = 35.0
EEG_FILTER_ORDER = 4

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)

# Window & artifact bounds
WINDOW_SECONDS = 2.0
EVAL_INTERVAL_SECONDS = 0.25
WARMUP_SECONDS = 1.0
MIN_CLEAN_STD_UV = 0.8
MAX_ARTIFACT_STD_UV = 90.0

# Combined log2-gain score weights and leaky-integrator meter
WEIGHT_ALPHA = 0.45
WEIGHT_RATIO = 0.35
WEIGHT_THETA = 0.20

SCORE_THRESHOLD = 0.92        # slightly above 0.88 so eyes-open noise doesn't cross
MIN_HIGH_STREAK = 4           # must stay above threshold for 1.0s (4 x 0.25s) before % starts rising
CHARGE_PER_STEP = 0.05        # +5% every 0.25s once streak is reached (~6s total to 100%)
DRAIN_PER_STEP = 0.10         # -10% every 0.25s immediately when awake
DEFAULT_COOLDOWN_SECONDS = 12.0


@dataclass(frozen=True)
class DrowsinessBaseline:
    """Trimmed awake baseline ratios from calibration."""

    theta_alpha_to_beta: float
    relative_alpha: float
    relative_theta: float


@dataclass(frozen=True)
class DrowsinessEvent:
    """Emitted when the sleepiness meter reaches 100%."""

    timestamp: float
    ratio_factor: float
    alpha_factor: float
    sustained_seconds: float


def _trimmed_median(values: list[float]) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    trim = len(x) // 7  # drop ~14% lowest and highest outliers
    if len(x) - 2 * trim >= 3:
        x = x[trim : len(x) - trim]
    return max(float(np.median(x)), 1e-6)


def _band_powers(signal_o1: np.ndarray, signal_o2: np.ndarray, sample_rate: int) -> Optional[dict[str, float]]:
    """Compute band metrics using whichever of O1/O2 has clean contact."""
    n = min(len(signal_o1), len(signal_o2))
    s1 = signal_o1[-n:]
    s2 = signal_o2[-n:]

    std1 = float(np.std(s1))
    std2 = float(np.std(s2))
    ok1 = MIN_CLEAN_STD_UV <= std1 <= MAX_ARTIFACT_STD_UV
    ok2 = MIN_CLEAN_STD_UV <= std2 <= MAX_ARTIFACT_STD_UV

    if not ok1 and not ok2:
        return None

    psds = []
    freqs = None
    for seg, ok in ((s1, ok1), (s2, ok2)):
        if ok:
            freqs, psd = welch(
                seg,
                fs=sample_rate,
                nperseg=n,
                window="hann",
                detrend="constant",
            )
            psds.append(psd)

    psd = np.mean(psds, axis=0)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    def band(low: float, high: float, inclusive_high: bool = False) -> float:
        mask = (freqs >= low) & (freqs <= high) if inclusive_high else (freqs >= low) & (freqs < high)
        return float(np.sum(psd[mask]) * df)

    theta = max(band(*THETA_BAND), 1e-6)
    alpha = max(band(*ALPHA_BAND, inclusive_high=True), 1e-6)
    beta = max(band(*BETA_BAND, inclusive_high=True), 1e-6)

    total = max(theta + alpha + beta, 1e-6)
    return {
        "ratio": (theta + alpha) / beta,
        "rel_alpha": alpha / total,
        "rel_theta": theta / total,
    }


def calculate_drowsiness_baseline(
    o1_values: list[float],
    o2_values: list[float],
    sample_rate: int,
) -> tuple[DrowsinessBaseline, dict[str, float]]:
    """Calculate a stable awake baseline from O1/O2 calibration."""
    o1 = np.asarray(o1_values, dtype=float)
    o2 = np.asarray(o2_values, dtype=float)
    valid = np.isfinite(o1) & np.isfinite(o2)
    o1, o2 = o1[valid], o2[valid]

    win_samples = int(WINDOW_SECONDS * sample_rate)
    step_samples = max(1, int(EVAL_INTERVAL_SECONDS * sample_rate))
    warmup_samples = int(WARMUP_SECONDS * sample_rate)

    if len(o1) < warmup_samples + win_samples:
        raise RuntimeError(
            f"Not enough drowsiness calibration samples: {len(o1)} "
            f"(need at least {warmup_samples + win_samples})"
        )

    sos = butter(
        EEG_FILTER_ORDER,
        [EEG_BANDPASS_LOW_HZ, EEG_BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    pair = np.column_stack([o1, o2])
    filtered = sosfilt(sos, pair, axis=0)[warmup_samples:]

    ratios: list[float] = []
    rel_alphas: list[float] = []
    rel_thetas: list[float] = []

    for start in range(0, len(filtered) - win_samples + 1, step_samples):
        seg1 = filtered[start : start + win_samples, 0]
        seg2 = filtered[start : start + win_samples, 1]
        metrics = _band_powers(seg1, seg2, sample_rate)
        if metrics is None:
            continue
        ratios.append(metrics["ratio"])
        rel_alphas.append(metrics["rel_alpha"])
        rel_thetas.append(metrics["rel_theta"])

    if not ratios:
        raise RuntimeError("No clean O1/O2 calibration windows found (check headset fit).")

    base_ratio = _trimmed_median(ratios)
    base_alpha = _trimmed_median(rel_alphas)
    base_theta = _trimmed_median(rel_thetas)

    baseline = DrowsinessBaseline(
        theta_alpha_to_beta=base_ratio,
        relative_alpha=base_alpha,
        relative_theta=base_theta,
    )
    return baseline, {
        "baseline_ratio": base_ratio,
        "baseline_rel_alpha": base_alpha,
        "baseline_rel_theta": base_theta,
        "windows_used": float(len(ratios)),
    }


class DrowsinessDetector:
    """Detect sustained drowsiness using a smooth score + leaky-integrator meter."""

    def __init__(
        self,
        baseline: DrowsinessBaseline,
        sample_rate: int,
        sensitivity: float = 1.0,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        ema_alpha: float = 0.30,
    ):
        """
        sensitivity = 1.0 (default)
        sensitivity = 1.25 (triggers easier)
        sensitivity = 0.80 (stricter)
        """
        self.baseline = baseline
        self.sample_rate = sample_rate
        self.score_threshold = SCORE_THRESHOLD / max(sensitivity, 0.25)
        self.cooldown_seconds = cooldown_seconds
        self.ema_alpha = ema_alpha

        self.sos = butter(
            EEG_FILTER_ORDER,
            [EEG_BANDPASS_LOW_HZ, EEG_BANDPASS_HIGH_HZ],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        self.filter_state = np.zeros((self.sos.shape[0], 2, 2), dtype=float)

        self.win_samples = int(WINDOW_SECONDS * sample_rate)
        self.eval_step_samples = max(1, int(EVAL_INTERVAL_SECONDS * sample_rate))
        self.warmup_left = int(WARMUP_SECONDS * sample_rate)

        self.o1_buffer: deque[float] = deque(maxlen=self.win_samples)
        self.o2_buffer: deque[float] = deque(maxlen=self.win_samples)
        self.samples_since_eval = 0

        self.ema_ratio: Optional[float] = None
        self.ema_rel_alpha: Optional[float] = None
        self.ema_rel_theta: Optional[float] = None

        self.last_ratio_factor: float = 1.0
        self.last_alpha_factor: float = 1.0
        self.last_score: float = 0.0
        self.drowsy_level: float = 0.0
        self.high_streak: int = 0
        self.is_drowsy: bool = False
        self.last_alert_time: Optional[float] = None

    def clear_buffer(self):
        """Reset filter and buffers after an EEG sample gap or reconnect."""
        self.filter_state.fill(0.0)
        self.o1_buffer.clear()
        self.o2_buffer.clear()
        self.samples_since_eval = 0
        self.warmup_left = int(WARMUP_SECONDS * self.sample_rate)
        self.drowsy_level = 0.0
        self.high_streak = 0
        self.is_drowsy = False

    def reset(self):
        """Reset buffers plus smoothing and alert cooldown state."""
        self.clear_buffer()
        self.ema_ratio = None
        self.ema_rel_alpha = None
        self.ema_rel_theta = None
        self.last_ratio_factor = 1.0
        self.last_alpha_factor = 1.0
        self.last_score = 0.0
        self.last_alert_time = None

    def update(
        self,
        o1_uv: np.ndarray,
        o2_uv: np.ndarray,
        timestamp: float,
    ) -> Optional[DrowsinessEvent]:
        """Add new raw O1/O2 microvolt samples and return an alert if meter hits 100%."""
        if len(o1_uv) == 0 or len(o2_uv) == 0:
            return None

        o1 = np.asarray(o1_uv, dtype=float)
        o2 = np.asarray(o2_uv, dtype=float)
        valid = np.isfinite(o1) & np.isfinite(o2)
        if not np.any(valid):
            return None

        pair = np.column_stack([o1[valid], o2[valid]])
        filtered, self.filter_state = sosfilt(self.sos, pair, axis=0, zi=self.filter_state)

        for s1, s2 in filtered:
            if self.warmup_left > 0:
                self.warmup_left -= 1
                continue
            self.o1_buffer.append(float(s1))
            self.o2_buffer.append(float(s2))
            self.samples_since_eval += 1

        if len(self.o1_buffer) < self.win_samples or self.samples_since_eval < self.eval_step_samples:
            return None

        self.samples_since_eval = 0
        seg1 = np.asarray(self.o1_buffer, dtype=float)
        seg2 = np.asarray(self.o2_buffer, dtype=float)

        metrics = _band_powers(seg1, seg2, self.sample_rate)
        if metrics is None:
            self.high_streak = 0
            self.drowsy_level = max(0.0, self.drowsy_level - DRAIN_PER_STEP)
            self.is_drowsy = False
            return None

        if self.ema_ratio is None:
            self.ema_ratio = metrics["ratio"]
            self.ema_rel_alpha = metrics["rel_alpha"]
            self.ema_rel_theta = metrics["rel_theta"]
        else:
            a = self.ema_alpha
            self.ema_ratio = a * metrics["ratio"] + (1.0 - a) * self.ema_ratio
            self.ema_rel_alpha = a * metrics["rel_alpha"] + (1.0 - a) * self.ema_rel_alpha
            self.ema_rel_theta = a * metrics["rel_theta"] + (1.0 - a) * self.ema_rel_theta

        self.last_ratio_factor = self.ema_ratio / self.baseline.theta_alpha_to_beta
        self.last_alpha_factor = self.ema_rel_alpha / self.baseline.relative_alpha
        theta_factor = self.ema_rel_theta / self.baseline.relative_theta

        # Smooth log2 gain score (0.0 at baseline, ~1.0 when ~2x drowsy shift)
        alpha_gain = max(0.0, float(np.log2(max(self.last_alpha_factor, 1e-6))))
        ratio_gain = max(0.0, float(np.log2(max(self.last_ratio_factor, 1e-6))))
        theta_gain = max(0.0, float(np.log2(max(theta_factor, 1e-6))))

        self.last_score = (
            WEIGHT_ALPHA * alpha_gain
            + WEIGHT_RATIO * ratio_gain
            + WEIGHT_THETA * theta_gain
        )

        # Leaky-integrator meter (0.0 -> 1.0), no dead-zone + 1.0s streak guard
        if self.last_score >= self.score_threshold:
            self.high_streak += 1
            if self.high_streak >= MIN_HIGH_STREAK:
                self.drowsy_level = min(1.0, self.drowsy_level + CHARGE_PER_STEP)
        else:
            self.high_streak = 0
            self.drowsy_level = max(0.0, self.drowsy_level - DRAIN_PER_STEP)

        self.is_drowsy = self.drowsy_level >= 1.0

        if self.is_drowsy:
            if self.last_alert_time is None or (timestamp - self.last_alert_time) >= self.cooldown_seconds:
                self.last_alert_time = timestamp
                self.drowsy_level = 0.0
                self.high_streak = 0
                print(
                    f"\n[Drowsy] score={self.last_score:.2f} (thr={self.score_threshold:.2f}) | "
                    f"alpha={self.last_alpha_factor:.2f}x ratio={self.last_ratio_factor:.2f}x theta={theta_factor:.2f}x",
                    flush=True,
                )
                return DrowsinessEvent(
                    timestamp=timestamp,
                    ratio_factor=self.last_ratio_factor,
                    alpha_factor=self.last_alpha_factor,
                    sustained_seconds=(MIN_HIGH_STREAK + 1.0 / CHARGE_PER_STEP) * EVAL_INTERVAL_SECONDS,
                )

        return None