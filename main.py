#!/usr/bin/env python3
"""Run jaw-clench, blink and double-blink detection with BrainAccess HALO."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
import traceback
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt

from brainaccess.core.eeg_manager import EEGManager
from brainaccess.utils import acquisition

from blink import BlinkDetector, calculate_blink_threshold
from clench import DoubleClenchDetector, calculate_jaw_threshold
from double_blink import DoubleBlinkDetector


# ---------------------------------------------------------------------
# HALO AND SIGNAL CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_DEVICE = "BA HALO 206"
CAP = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
CHANNELS = list(CAP.values())
FS = 250

READ_SECONDS = 4.0
SDK_WINDOW_SECONDS = 5.0
POLL_SECONDS = 0.05
FILTER_WARMUP_SECONDS = 0.75

# Jaw signal filter (same settings as the previous working script).
JAW_LOW_HZ = 25.0
JAW_HIGH_HZ = 90.0
JAW_FILTER_ORDER = 4
JAW_RMS_WINDOW_SECONDS = 0.10
DEFAULT_JAW_DOUBLE_WINDOW = 1.50

# Blink signal filter (same settings as the previous working script).
BLINK_LOW_HZ = 1.0
BLINK_HIGH_HZ = 10.0
BLINK_FILTER_ORDER = 4
DEFAULT_BLINK_K = 4.0
DEFAULT_BLINK_MIN_EXCESS_UV = 8.0
DEFAULT_BLINK_DOUBLE_WINDOW = 0.70


# ---------------------------------------------------------------------
# BRAINACCESS DATA READING
# ---------------------------------------------------------------------

def read_new_samples(eeg: acquisition.EEG, last_sample: Optional[int]):
    """Read a rolling EEG window and return only samples not yet processed."""
    raw = eeg.get_mne(tim=READ_SECONDS, annotations=False)

    required = CHANNELS + ["Sample"]
    missing = [name for name in required if name not in raw.ch_names]
    if missing:
        raise RuntimeError(
            "Missing SDK channel(s): " + ", ".join(missing)
            + ". Check the CAP mapping and SDK version."
        )

    data = raw.get_data(picks=list(range(len(raw.ch_names))))
    eeg_rows = [raw.ch_names.index(name) for name in CHANNELS]
    sample_row = raw.ch_names.index("Sample")

    eeg_volts = data[eeg_rows, :].T
    sample_values = data[sample_row, :]

    finite = np.isfinite(sample_values) & np.all(np.isfinite(eeg_volts), axis=1)
    sample_ids = np.rint(sample_values[finite]).astype(np.int64)
    eeg_volts = eeg_volts[finite, :]

    if sample_ids.size == 0:
        return (
            np.empty((0, len(CHANNELS))),
            np.empty(0, dtype=np.int64),
            last_sample,
            False,
        )

    # Remove padding/repeated sample counter values while preserving order.
    increasing = np.ones(sample_ids.size, dtype=bool)
    if sample_ids.size > 1:
        increasing[1:] = np.diff(sample_ids) > 0
    sample_ids = sample_ids[increasing]
    eeg_volts = eeg_volts[increasing, :]

    if sample_ids.size == 0:
        return (
            np.empty((0, len(CHANNELS))),
            np.empty(0, dtype=np.int64),
            last_sample,
            False,
        )

    reset = False
    if last_sample is not None and int(sample_ids[-1]) < last_sample:
        # HALO sample counter restarted (for example, after reconnect).
        last_sample = None
        reset = True

    if last_sample is not None:
        new = sample_ids > last_sample
        sample_ids = sample_ids[new]
        eeg_volts = eeg_volts[new, :]

    if sample_ids.size == 0:
        return np.empty((0, len(CHANNELS))), sample_ids, last_sample, reset

    # MNE electrode data from BrainAccess is in volts.
    eeg_microvolts = eeg_volts * 1_000_000.0
    return eeg_microvolts, sample_ids, int(sample_ids[-1]), reset


# ---------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------

def run(args: argparse.Namespace):
    if FS / 2 <= BLINK_HIGH_HZ or FS / 2 <= JAW_HIGH_HZ:
        raise RuntimeError("Configured sample rate is too low for the filters.")

    jaw_sos = butter(
        JAW_FILTER_ORDER,
        [JAW_LOW_HZ, JAW_HIGH_HZ],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    blink_sos = butter(
        BLINK_FILTER_ORDER,
        [BLINK_LOW_HZ, BLINK_HIGH_HZ],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    jaw_filter_state = np.zeros((jaw_sos.shape[0], 2, len(CHANNELS)), dtype=float)
    blink_filter_state = np.zeros((blink_sos.shape[0], 2), dtype=float)

    jaw_rms_samples = max(5, int(round(JAW_RMS_WINDOW_SECONDS * FS)))
    jaw_power_window = deque(maxlen=jaw_rms_samples)

    warmup_left = int(round(FILTER_WARMUP_SECONDS * FS))

    jaw_calibration_values: list[float] = []
    blink_calibration_values: list[float] = []
    calibration_start_sample: Optional[int] = None

    jaw_detector: Optional[DoubleClenchDetector] = None
    blink_detector: Optional[BlinkDetector] = None
    double_blink_detector: Optional[DoubleBlinkDetector] = None
    blink_threshold: Optional[float] = None

    last_sample: Optional[int] = None
    latest_jaw_rms: Optional[float] = None

    status = "CONNECTING"
    listening_until: Optional[float] = None

    last_status_print = 0.0
    last_new_data_wall = time.monotonic()
    last_warning_wall = 0.0

    eeg = acquisition.EEG(mode="roll")
    stream_started = False

    print(f"Connecting directly to BrainAccess device: {args.device_name}", flush=True)
    print("Disconnect HALO from BrainAccess Board before running this script.", flush=True)

    try:
        with EEGManager() as manager:
            setup_args = {
                "device_name": args.device_name,
                "cap": CAP,
                "sfreq": FS,
            }

            # Support SDK builds that expose a rolling-buffer size argument.
            try:
                if "window_seconds" in inspect.signature(eeg.setup).parameters:
                    setup_args["window_seconds"] = SDK_WINDOW_SECONDS
            except (TypeError, ValueError):
                pass

            eeg.setup(manager, **setup_args)
            print("Connected to BrainAccess HALO.", flush=True)
            print(f"Channels: {', '.join(CHANNELS)} | Sample rate: {FS} Hz", flush=True)

            eeg.start_acquisition()
            stream_started = True
            print("Acquisition started. Waiting for samples...", flush=True)
            time.sleep(1.0)
            print(
                f"Keep your jaw relaxed and look forward for "
                f"{args.calibration_seconds:.0f} seconds. Natural blinking is okay.",
                flush=True,
            )
            status = "CALIBRATING"

            try:
                while True:
                    poll_start = time.monotonic()
                    previous_sample = last_sample

                    try:
                        eeg_uv, sample_ids, newest, counter_reset = read_new_samples(
                            eeg, last_sample
                        )
                    except Exception as exc:
                        if time.monotonic() - last_new_data_wall < 4.0:
                            time.sleep(POLL_SECONDS)
                            continue
                        raise RuntimeError(
                            "Could not read EEG samples from BrainAccess SDK"
                        ) from exc

                    if newest is not None:
                        last_sample = newest
                        last_new_data_wall = time.monotonic()

                    if eeg_uv.size:
                        gap_limit = int(0.20 * FS)
                        gap_between_reads = (
                            previous_sample is not None
                            and sample_ids.size > 0
                            and sample_ids[0] - previous_sample > gap_limit
                        )
                        gap_inside_read = (
                            sample_ids.size > 1
                            and np.any(np.diff(sample_ids) > gap_limit)
                        )

                        if counter_reset or gap_between_reads or gap_inside_read:
                            if args.debug:
                                print(
                                    "\nEEG sample gap/reset: resetting filters and detector state.",
                                    flush=True,
                                )

                            jaw_filter_state.fill(0.0)
                            blink_filter_state.fill(0.0)
                            jaw_power_window.clear()
                            warmup_left = int(round(FILTER_WARMUP_SECONDS * FS))

                            if jaw_detector is not None:
                                jaw_detector.reset()

                            if blink_detector is not None:
                                blink_detector.clear_buffer()

                            if counter_reset:
                                if blink_detector is not None:
                                    blink_detector.reset_sample_tracking()
                                if double_blink_detector is not None:
                                    double_blink_detector.reset()

                        # Filter new samples only once (no reprocessing of rolling buffer).
                        jaw_filtered, jaw_filter_state = sosfilt(
                            jaw_sos,
                            eeg_uv,
                            axis=0,
                            zi=jaw_filter_state,
                        )

                        # BrainAccess example's derived blink signal:
                        # mean(Fp1, Fp2) - mean(O1, O2).
                        blink_derived_uv = (
                            (eeg_uv[:, 0] + eeg_uv[:, 1]) / 2.0
                            - (eeg_uv[:, 2] + eeg_uv[:, 3]) / 2.0
                        )
                        blink_filtered, blink_filter_state = sosfilt(
                            blink_sos,
                            blink_derived_uv,
                            zi=blink_filter_state,
                        )

                        for jaw_sample, blink_sample, sample_id in zip(
                            jaw_filtered,
                            blink_filtered,
                            sample_ids,
                        ):
                            sample_id = int(sample_id)

                            if not np.all(np.isfinite(jaw_sample)) or not np.isfinite(blink_sample):
                                continue

                            if warmup_left > 0:
                                warmup_left -= 1
                                continue

                            # Jaw-clench RMS across the four channels.
                            jaw_power_window.append(float(np.mean(np.square(jaw_sample))))
                            if len(jaw_power_window) < jaw_rms_samples:
                                continue

                            latest_jaw_rms = float(np.sqrt(np.mean(jaw_power_window)))
                            if not np.isfinite(latest_jaw_rms):
                                continue

                            # Collect both calibration signals simultaneously.
                            if jaw_detector is None:
                                if calibration_start_sample is None:
                                    calibration_start_sample = sample_id

                                jaw_calibration_values.append(latest_jaw_rms)
                                blink_calibration_values.append(float(blink_sample))

                                elapsed = (sample_id - calibration_start_sample) / FS
                                if elapsed >= args.calibration_seconds:
                                    jaw_thr, jaw_stats = calculate_jaw_threshold(
                                        jaw_calibration_values,
                                        args.threshold_scale,
                                    )
                                    blink_thr, blink_stats = calculate_blink_threshold(
                                        blink_calibration_values,
                                        args.blink_k,
                                        args.blink_min_excess_uv,
                                    )

                                    jaw_detector = DoubleClenchDetector(
                                        jaw_thr,
                                        args.jaw_double_window,
                                    )
                                    blink_threshold = blink_thr
                                    blink_detector = BlinkDetector(
                                        threshold=blink_thr,
                                        prominence=blink_stats["excess"],
                                        sample_rate=FS,
                                        initial_last_sample=sample_id,
                                    )
                                    double_blink_detector = DoubleBlinkDetector(
                                        args.blink_double_window
                                    )

                                    status = "NO REQUEST"
                                    print("\nCalibration complete:", flush=True)
                                    print(
                                        f"  jaw baseline median={jaw_stats['median']:.2f} µV, "
                                        f"robust noise={jaw_stats['robust_noise']:.2f} µV, "
                                        f"95th percentile={jaw_stats['p95']:.2f} µV, "
                                        f"threshold={jaw_thr:.2f} µV",
                                        flush=True,
                                    )
                                    print(
                                        f"  blink median={blink_stats['median']:.2f} µV, "
                                        f"MAD={blink_stats['mad']:.2f} µV, "
                                        f"k={args.blink_k:.2f}, "
                                        f"threshold={blink_thr:.2f} µV",
                                        flush=True,
                                    )
                                    print("Status: NO REQUEST", flush=True)
                                continue

                            # Add each post-calibration blink sample to the detector.
                            blink_detector.add_samples([blink_sample], [sample_id])

                            # Preserve existing jaw-clench functionality.
                            now = time.monotonic()
                            if listening_until is not None:
                                if now < listening_until:
                                    status = "LISTENING . . ."
                                else:
                                    listening_until = None
                                    jaw_detector.reset()
                                    status = "NO REQUEST"

                            if listening_until is None:
                                event, duration, peak = jaw_detector.update(
                                    latest_jaw_rms,
                                    sample_id / FS,
                                )

                                if event == "first":
                                    status = "NO REQUEST"
                                    if args.debug:
                                        print(
                                            f"\nFirst jaw clench: {duration:.2f}s, "
                                            f"peak {peak:.2f} µV; waiting for second...",
                                            flush=True,
                                        )

                                elif event == "double":
                                    status = "LISTENING . . ."
                                    listening_until = time.monotonic() + args.listening_seconds
                                    jaw_detector.reset()
                                    print(
                                        "\nDOUBLE JAW CLENCH DETECTED -> LISTENING . . .",
                                        flush=True,
                                    )
                                    last_status_print = time.monotonic()

                        # Detect newly confirmed blink peaks after calibration.
                        if blink_detector is not None and double_blink_detector is not None:
                            blink_events = blink_detector.detect()
                            for blink_event in blink_events:
                                blink_time = blink_event.sample_id / FS
                                print("\nBLINK", flush=True)
                                gap = double_blink_detector.update(blink_time)
                                if gap is not None:
                                    print(f"TWO BLINKS (gap: {gap:.2f} s)", flush=True)

                    now = time.monotonic()

                    if listening_until is not None and now >= listening_until:
                        listening_until = None
                        if jaw_detector is not None:
                            jaw_detector.reset()
                        status = "NO REQUEST"

                    if now - last_status_print >= args.status_interval:
                        if jaw_detector is None:
                            line = "Status: CALIBRATING"
                        else:
                            label = (
                                "NO REQUEST (1/2 clenches)"
                                if jaw_detector.first_clench_time is not None
                                else status
                            )
                            if latest_jaw_rms is None:
                                line = f"Status: {label}"
                            else:
                                line = (
                                    f"Status: {label} | "
                                    f"jaw RMS: {latest_jaw_rms:6.2f} µV | "
                                    f"jaw threshold: {jaw_detector.threshold:6.2f} µV | "
                                    f"blink threshold: {blink_threshold:6.2f} µV"
                                )

                        if now - last_new_data_wall > 3.0:
                            line += " | WARNING: no new HALO samples"
                            if now - last_warning_wall > 3.0:
                                print(
                                    "\nWARNING: no new HALO samples. Check headset power and Bluetooth.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                last_warning_wall = now

                        print("\r" + line.ljust(150), end="", flush=True)
                        last_status_print = now

                    delay = POLL_SECONDS - (time.monotonic() - poll_start)
                    if delay > 0:
                        time.sleep(delay)

            except KeyboardInterrupt:
                print("\nStopping acquisition...", flush=True)

            finally:
                if stream_started:
                    try:
                        eeg.stop_acquisition()
                        stream_started = False
                        print("Acquisition stopped.", flush=True)
                    except Exception as exc:
                        print(f"Warning stopping acquisition: {exc}", file=sys.stderr)

                try:
                    manager.disconnect()
                except Exception:
                    pass

    finally:
        try:
            # Must be called after the EEGManager context has exited.
            eeg.close()
        except Exception as exc:
            print(f"Warning closing BrainAccess library: {exc}", file=sys.stderr)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect jaw clenches and single/double blinks using BrainAccess HALO."
    )
    parser.add_argument("--device-name", default=os.environ.get("BA_DEVICE", DEFAULT_DEVICE))
    parser.add_argument("--calibration-seconds", type=float, default=8.0)
    parser.add_argument("--jaw-double-window", type=float, default=DEFAULT_JAW_DOUBLE_WINDOW)
    parser.add_argument("--listening-seconds", type=float, default=5.0)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--blink-k", type=float, default=DEFAULT_BLINK_K)
    parser.add_argument("--blink-min-excess-uv", type=float, default=DEFAULT_BLINK_MIN_EXCESS_UV)
    parser.add_argument("--blink-double-window", type=float, default=DEFAULT_BLINK_DOUBLE_WINDOW)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.calibration_seconds < 2:
        parser.error("--calibration-seconds must be at least 2")
    if args.jaw_double_window <= 0 or args.listening_seconds <= 0:
        parser.error("Jaw timing values must be positive")
    if args.blink_double_window <= 0:
        parser.error("--blink-double-window must be positive")
    if args.status_interval <= 0:
        parser.error("--status-interval must be positive")
    if not 0.3 <= args.threshold_scale <= 2.0:
        parser.error("--threshold-scale must be between 0.3 and 2.0")
    if not 2.0 <= args.blink_k <= 8.0:
        parser.error("--blink-k must be between 2.0 and 8.0")
    if args.blink_min_excess_uv <= 0:
        parser.error("--blink-min-excess-uv must be positive")
    if not args.device_name.strip():
        parser.error("--device-name cannot be empty")

    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nFATAL ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SystemExit(1)
