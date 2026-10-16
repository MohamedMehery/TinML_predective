"""Protocol-neutral loader for actual Paderborn (KAt) MAT recordings.

This module reuses the existing isolated reader. It does NOT select the
speed reference for R3_Order and does not modify the research protocol.
It returns a validated data-loading interface only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .reader import MatReaderError, read_mat, inspect_mat

NOMINAL_RPM_MAP = {
    "N15": 1500,
    "N09": 900,
}

FS_NOMINAL_HZ = 64000


class PuRecordingError(MatReaderError):
    """Raised when a Paderborn recording cannot be loaded/validated."""


@dataclass
class PuRecording:
    vibration: np.ndarray
    vibration_time: np.ndarray
    speed_hz: np.ndarray
    fs_nominal_hz: int
    fs_endpoint_hz: float
    metadata: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _parse_filename(path: str):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    if len(parts) < 4:
        raise PuRecordingError(
            f"filename {base!r} does not contain condition code and bearing id"
        )
    condition_code = "_".join(parts[:3])
    bearing_id = parts[3]
    speed_token = parts[0]
    if speed_token not in NOMINAL_RPM_MAP:
        raise PuRecordingError(
            f"unsupported condition code prefix {speed_token!r} in {base!r}; "
            f"supported: {sorted(NOMINAL_RPM_MAP)}"
        )
    nominal_rpm = NOMINAL_RPM_MAP[speed_token]
    return bearing_id, condition_code, nominal_rpm


def load_pu_recording(path: str, *, speed_unit: str) -> PuRecording:
    """Load and validate a Paderborn MAT recording.

    ``speed_unit`` must be explicitly ``"rpm_assumed"``.
    """
    if speed_unit != "rpm_assumed":
        raise PuRecordingError(
            f"unsupported speed_unit {speed_unit!r}; only 'rpm_assumed' is supported"
        )

    bearing_id, condition_code, nominal_rpm = _parse_filename(path)

    data = read_mat(path)
    channels = data["channels"]
    if "vibration_1" not in channels:
        raise PuRecordingError(f"required channel 'vibration_1' missing in {path!r}")
    if "speed" not in channels:
        raise PuRecordingError(f"required channel 'speed' missing in {path!r}")

    vib = channels["vibration_1"]
    spd = channels["speed"]

    vibration = np.asarray(vib.values, dtype=np.float64).ravel()
    vibration_time = np.asarray(vib.timestamps, dtype=np.float64).ravel().copy()
    speed_values = np.asarray(spd.values, dtype=np.float64).ravel()
    speed_time = np.asarray(spd.timestamps, dtype=np.float64).ravel().copy()

    # Validation gate 3: timestamp validity (before truncation)
    if vibration.shape != vibration_time.shape:
        raise PuRecordingError(
            f"vibration values/timestamps length mismatch: "
            f"{vibration.shape} vs {vibration_time.shape}"
        )
    if speed_values.shape != speed_time.shape:
        raise PuRecordingError(
            f"speed values/timestamps length mismatch: "
            f"{speed_values.shape} vs {speed_time.shape}"
        )
    if not np.all(np.isfinite(vibration_time)):
        raise PuRecordingError("vibration timestamps contain non-finite values")
    if not np.all(np.isfinite(speed_time)):
        raise PuRecordingError("speed timestamps contain non-finite values")
    if not np.all(np.isfinite(vibration)):
        raise PuRecordingError("vibration values contain non-finite values")
    if not np.all(np.isfinite(speed_values)):
        raise PuRecordingError("speed values contain non-finite values")
    if vibration_time.size > 1 and np.any(np.diff(vibration_time) <= 0):
        raise PuRecordingError("vibration timestamps are not strictly increasing")
    if speed_time.size > 1 and np.any(np.diff(speed_time) <= 0):
        raise PuRecordingError("speed timestamps are not strictly increasing")

    # Gates 1 & 2 from existing reader (original axis)
    rep = inspect_mat(path)
    vrep = rep["channels"]["vibration_1"]
    fs_endpoint_hz = float(vrep["endpoint_rate_hz"])
    max_abs_residual_s = float(vrep["max_abs_residual_s"])

    if abs(fs_endpoint_hz - FS_NOMINAL_HZ) / FS_NOMINAL_HZ >= 0.001:
        raise PuRecordingError(
            f"endpoint rate gate failed: {fs_endpoint_hz} Hz vs nominal "
            f"{FS_NOMINAL_HZ} Hz"
        )
    if not (max_abs_residual_s < 1.0 / FS_NOMINAL_HZ):
        raise PuRecordingError(
            f"timestamp residual gate failed: {max_abs_residual_s} s >= "
            f"{1.0 / FS_NOMINAL_HZ} s"
        )

    # Common time support
    t_start = max(float(vibration_time[0]), float(speed_time[0]))
    t_end = min(float(vibration_time[-1]), float(speed_time[-1]))
    if not (t_end > t_start):
        raise PuRecordingError("no valid common time support between vibration and speed")

    # Retain vibration samples within common support
    mask = (vibration_time >= t_start) & (vibration_time <= t_end)
    dropped_begin = int(np.sum(vibration_time < t_start))
    dropped_end = int(np.sum(vibration_time > t_end))
    retained_vibration = vibration[mask]
    retained_time = vibration_time[mask].copy()

    if retained_time.size == 0:
        raise PuRecordingError("no vibration samples retained in common support")

    # No extrapolation
    if np.any(retained_time < speed_time[0]) or np.any(retained_time > speed_time[-1]):
        raise PuRecordingError(
            "interpolation query outside original speed timestamp range"
        )

    # Gate 4: speed consistency
    speed_mask = (speed_time >= t_start) & (speed_time <= t_end)
    speed_in_support = speed_values[speed_mask]
    if speed_in_support.size == 0:
        raise PuRecordingError("no speed samples in common time support")
    mean_speed_rpm = float(np.mean(speed_in_support))
    nominal_hz = nominal_rpm / 60.0
    mean_speed_hz = mean_speed_rpm / 60.0
    if nominal_hz <= 0 or abs(mean_speed_hz - nominal_hz) / nominal_hz > 0.05:
        raise PuRecordingError(
            f"speed consistency gate failed: mean {mean_speed_hz} Hz vs "
            f"nominal {nominal_hz} Hz"
        )

    # Interpolate speed (RPM) at retained vibration timestamps, then convert
    speed_rpm_interp = np.interp(retained_time, speed_time, speed_values)
    speed_hz = (speed_rpm_interp / 60.0).astype(np.float64)

    metadata = {
        "bearing_id": bearing_id,
        "condition_code": condition_code,
        "nominal_rpm": nominal_rpm,
        "nominal_hz": nominal_hz,
        "fs_nominal_hz": FS_NOMINAL_HZ,
        "fs_endpoint_hz": fs_endpoint_hz,
        "max_abs_residual_s": max_abs_residual_s,
        "dropped_begin": dropped_begin,
        "dropped_end": dropped_end,
        "dropped_total": int(dropped_begin + dropped_end),
        "original_vibration_count": int(vibration_time.size),
        "retained_count": int(retained_time.size),
        "speed_unit_assumption": "rpm_assumed",
        "mean_speed_rpm_in_support": mean_speed_rpm,
        "mean_speed_hz_in_support": mean_speed_hz,
        "t_start": t_start,
        "t_end": t_end,
        "assumptions": [
            "speed values are assumed to be in RPM and converted to Hz by /60",
            "fs_nominal_hz=64000 is a nominal engineering assumption",
            "common time support from original timestamps",
            "speed linearly interpolated at retained vibration timestamps",
            "no extrapolation; queries verified within speed timestamp range",
        ],
    }

    return PuRecording(
        vibration=retained_vibration.astype(np.float64),
        vibration_time=retained_time.astype(np.float64),
        speed_hz=speed_hz,
        fs_nominal_hz=FS_NOMINAL_HZ,
        fs_endpoint_hz=fs_endpoint_hz,
        metadata=metadata,
    )
