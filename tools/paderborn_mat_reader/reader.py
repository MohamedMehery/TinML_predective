"""Isolated reader for Paderborn (KAt) MATLAB 5.0 recordings.

Scope: read local .mat files, discover channels by name, resolve the
time axis through the stored (1-based) XIndex, validate timestamps,
and produce a small inspection report. No resampling, filtering,
alignment, or unit conversion is performed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "paderborn_mat_reader requires numpy. Install with: pip install numpy"
    ) from exc

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "paderborn_mat_reader requires scipy. Install with: pip install scipy"
    ) from exc


class MatReaderError(Exception):
    """Raised when a MAT file cannot be parsed or validated."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _unwrap(obj: Any) -> Any:
    """Unwrap 0-d / singleton numpy arrays to their scalar/element.

    Structured arrays (dtype.names) are never unwrapped.
    """
    if isinstance(obj, np.ndarray):
        if getattr(obj.dtype, "names", None):
            return obj
        if obj.shape == ():
            try:
                return obj.item()
            except (ValueError, TypeError):
                return obj.reshape(-1)[0] if obj.size else obj
        if obj.size == 1 and obj.ndim > 0:
            return obj.reshape(-1)[0]
    return obj


def _to_1d_array(value: Any, label: str) -> "np.ndarray":
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        try:
            arr = np.asarray(value)
        except Exception as exc:
            raise MatReaderError(f"{label}: cannot convert to array ({exc})") from exc
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return arr


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (np.str_,)):
        return str(value).strip()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S"):
            try:
                if value.dtype.kind == "S":
                    return value.tobytes().decode("utf-8", errors="replace").strip("\x00 ").strip()
                return "".join(str(c) for c in value.flatten()).strip()
            except Exception:
                pass
        if value.size == 1:
            return str(_unwrap(value)).strip()
    try:
        v = _unwrap(value)
        if isinstance(v, str):
            return v.strip()
        return str(v).strip()
    except Exception:
        return str(value).strip()


def _to_int(value: Any, label: str) -> int:
    v = _unwrap(value)
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError("not an integer")
            return int(v)
        return int(v)
    except (TypeError, ValueError) as exc:
        raise MatReaderError(f"{label}: expected an integer XIndex, got {value!r}") from exc


def _find_field(d: Dict[str, Any], candidates: List[str]) -> Any:
    # case-insensitive lookup
    lower = {str(k).lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in lower:
            return d[lower[c.lower()]]
    return None


def _unit_field(d: Dict[str, Any]) -> str:
    v = _find_field(d, ["Unit", "unit", "Units", "units", "unit_name"])
    if v is None:
        return "UNKNOWN"
    s = _to_str(v)
    return s if s else "UNKNOWN"


def _entry_to_dict(entry: Any) -> Dict[str, Any]:
    arr = None
    if isinstance(entry, np.ndarray) and getattr(entry.dtype, "names", None):
        arr = entry
    elif isinstance(entry, np.void) and getattr(entry.dtype, "names", None):
        arr = entry
    if arr is not None:
        out = {}
        for n in arr.dtype.names:
            v = arr[n]
            if isinstance(v, np.ndarray) and v.shape == () and not getattr(v.dtype, "names", None):
                try:
                    v = v.item()
                except (ValueError, TypeError):
                    pass
            out[n] = v
        return out
    if isinstance(entry, np.ndarray) and entry.shape == ():
        entry = _unwrap(entry)
    if hasattr(entry, "_fieldnames"):
        return {n: getattr(entry, n) for n in entry._fieldnames}
    if isinstance(entry, dict):
        return dict(entry)
    if hasattr(entry, "__dict__"):
        return {k: v for k, v in vars(entry).items() if not k.startswith("_")}
    raise MatReaderError(f"cannot interpret entry of type {type(entry).__name__}")


def _to_object_array(value: Any, label: str) -> "np.ndarray":
    if isinstance(value, np.ndarray):
        # 0-d structured array -> single entry
        if value.shape == () and getattr(value.dtype, "names", None):
            return np.array([value], dtype=object)
        # 0-d object array -> unwrap
        if value.shape == () and value.dtype == object:
            try:
                inner = value.item()
            except (ValueError, TypeError):
                inner = value
            if isinstance(inner, np.ndarray) and getattr(inner.dtype, "names", None):
                return np.array([inner], dtype=object)
            if isinstance(inner, np.ndarray) and inner.ndim > 0:
                return inner
            return np.array([inner], dtype=object)
        if value.ndim == 0:
            return np.array([value], dtype=object)
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=object)
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        return np.array([value], dtype=object)
    raise MatReaderError(f"field '{label}' is not an array of structs")


def _load_root(path: str) -> Tuple[str, Any]:
    if not os.path.isfile(path):
        raise MatReaderError(f"file not found: {path}")
    try:
        raw = loadmat(path, squeeze_me=True, struct_as_record=True)
    except Exception as exc:
        raise MatReaderError(f"failed to load MAT file {path!r}: {exc}") from exc
    data_keys = [k for k in raw.keys() if not str(k).startswith("__")]
    if not data_keys:
        raise MatReaderError(f"no data variables found in {path!r}")
    if len(data_keys) > 1:
        raise MatReaderError(
            f"multiple top-level data variables in {path!r}: {data_keys}; "
            "expected a single root structure"
        )
    root_key = data_keys[0]
    root = raw[root_key]
    if isinstance(root, np.ndarray) and root.shape != ():
        if root.size == 1:
            root = root.reshape(-1)[0]
        else:
            raise MatReaderError(
                f"root variable {root_key!r} is not a scalar struct "
                f"(shape {tuple(root.shape)})"
            )
    if isinstance(root, np.ndarray):
        if root.shape != () or not getattr(root.dtype, "names", None):
            # maybe still a struct with fields
            if not getattr(root.dtype, "names", None):
                raise MatReaderError(
                    f"root variable {root_key!r} is not a struct "
                    f"(type {type(root).__name__})"
                )
    elif not hasattr(root, "_fieldnames") and not isinstance(root, dict):
        raise MatReaderError(
            f"root variable {root_key!r} is not a struct "
            f"(type {type(root).__name__})"
        )
    return root_key, root


def _get_field(root: Any, name: str, required: bool = True) -> Any:
    if isinstance(root, np.ndarray) and getattr(root.dtype, "names", None) and name in root.dtype.names:
        return _unwrap(root[name])
    # case-insensitive
    if isinstance(root, np.ndarray) and getattr(root.dtype, "names", None):
        for n in root.dtype.names:
            if str(n).lower() == name.lower():
                return _unwrap(root[n])
    if hasattr(root, "_fieldnames"):
        for n in root._fieldnames:
            if str(n).lower() == name.lower():
                return getattr(root, n)
    if isinstance(root, dict):
        for k in root:
            if str(k).lower() == name.lower():
                return root[k]
    if required:
        raise MatReaderError(f"root structure missing required field '{name}'")
    return None


def _validate_timestamps(t: "np.ndarray", label: str) -> None:
    if t.size == 0:
        raise MatReaderError(f"{label}: empty timestamp array")
    if not np.all(np.isfinite(t)):
        raise MatReaderError(f"{label}: timestamps contain NaN or infinity")
    if t.size > 1:
        diffs = np.diff(t)
        bad = int(np.sum(diffs <= 0))
        if bad > 0:
            raise MatReaderError(
                f"{label}: timestamps are not strictly increasing "
                f"({bad} non-positive difference(s))"
            )


@dataclass
class ChannelData:
    name: str
    values: "np.ndarray"
    timestamps: "np.ndarray"
    unit: str
    x_index: int
    x_name: str
    x_unit: str


def read_mat(path: str) -> Dict[str, Any]:
    """Read a Paderborn MAT file and return channel data.

    Returns a dict with keys: 'root_name', 'channels' (name -> ChannelData),
    'x_axes' (1-based index -> dict).
    """
    root_name, root = _load_root(path)

    # Required fields X and Y (Info/Description optional)
    x_raw = _get_field(root, "X", required=True)
    y_raw = _get_field(root, "Y", required=True)

    x_arr = _to_object_array(x_raw, "X")
    y_arr = _to_object_array(y_raw, "Y")

    # Parse time axes (1-based indexing)
    x_axes: Dict[int, Dict[str, Any]] = {}
    for i, entry in enumerate(x_arr, start=1):
        d = _entry_to_dict(entry)
        name = _to_str(_find_field(d, ["Name", "name", "name_"]))
        if not name:
            name = f"X[{i}]"
        data_field = _find_field(d, ["Data", "data", "Values", "values", "Time", "time"])
        if data_field is None:
            raise MatReaderError(f"X[{i}] ({name!r}) missing time Data field")
        data = _to_1d_array(data_field, f"X[{i}].Data")
        unit = _unit_field(d)
        x_axes[i] = {"name": name, "data": data, "unit": unit}

    if not x_axes:
        raise MatReaderError("no time axes (X) found")

    # Parse channels
    channels: Dict[str, ChannelData] = {}
    for entry in y_arr:
        d = _entry_to_dict(entry)
        name = _to_str(_find_field(d, ["Name", "name", "Channel", "channel"]))
        if not name:
            raise MatReaderError("a Y entry has no Name")
        if name in channels:
            raise MatReaderError(f"duplicate channel name {name!r}")
        data_field = _find_field(d, ["Data", "data", "Values", "values"])
        if data_field is None:
            raise MatReaderError(f"channel {name!r} missing Data field")
        data = _to_1d_array(data_field, f"{name}.Data")
        xi_raw = _find_field(d, ["XIndex", "xindex", "X_index", "x_index", "XIdx"])
        if xi_raw is None:
            raise MatReaderError(f"channel {name!r} missing XIndex field")
        xi = _to_int(xi_raw, f"{name}.XIndex")
        if xi < 1 or xi > len(x_axes):
            raise MatReaderError(
                f"channel {name!r}: XIndex {xi} out of range (valid 1..{len(x_axes)})"
            )
        t = x_axes[xi]["data"]
        if data.shape != t.shape:
            raise MatReaderError(
                f"channel {name!r}: data length {data.shape[0]} != "
                f"time length {t.shape[0]} (X[{xi}])"
            )
        _validate_timestamps(t, f"channel {name!r} time axis X[{xi}]")
        unit = _unit_field(d)
        channels[name] = ChannelData(
            name=name,
            values=data,
            timestamps=t,
            unit=unit,
            x_index=xi,
            x_name=x_axes[xi]["name"],
            x_unit=x_axes[xi]["unit"],
        )

    # Target channels required for this reader's scope
    for required_name in ("vibration_1", "speed"):
        if required_name not in channels:
            raise MatReaderError(
                f"required channel {required_name!r} not found in {path!r}"
            )

    return {
        "root_name": root_name,
        "channels": channels,
        "x_axes": x_axes,
    }


def _intervals(t: "np.ndarray") -> Tuple[float, float, float, int]:
    if t.size < 2:
        return float("nan"), float("nan"), float("nan"), 0
    diffs = np.diff(t)
    pos = diffs[diffs > 0]
    nonpos = int(np.sum(diffs <= 0))
    if pos.size == 0:
        return float("nan"), float("nan"), float("nan"), nonpos
    return float(np.min(pos)), float(np.median(pos)), float(np.max(pos)), nonpos


def inspect_mat(path: str) -> Dict[str, Any]:
    """Produce a small inspection report for a MAT file."""
    result = read_mat(path)
    channels_report: Dict[str, Any] = {}
    for name, ch in result["channels"].items():
        t = ch.timestamps
        vmin, vmed, vmax, nonpos = _intervals(t)
        if t.size >= 2 and vmed and not math.isnan(vmed) and vmed > 0:
            rate = 1.0 / vmed
        else:
            rate = float("nan")
        channels_report[name] = {
            "name": name,
            "source_metadata": f"Y entry Name={name!r}, XIndex={ch.x_index} "
            f"(X name {ch.x_name!r})",
            "unit": ch.unit,
            "n_samples": int(t.size),
            "value_shape": list(ch.values.shape),
            "value_dtype": str(ch.values.dtype),
            "timestamp_shape": list(t.shape),
            "timestamp_dtype": str(t.dtype),
            "first_timestamp": float(t[0]) if t.size else None,
            "last_timestamp": float(t[-1]) if t.size else None,
            "min_interval": vmin,
            "median_interval": vmed,
            "max_interval": vmax,
            "nonpositive_diff_count": nonpos,
            "conditional_sample_rate_hz": rate,
            "sample_rate_note": "valid only if timestamps are in seconds",
            "validation_status": "OK",
            "warnings": [],
        }

    # common time support
    support = {"valid": False}
    if "vibration_1" in result["channels"] and "speed" in result["channels"]:
        vt = result["channels"]["vibration_1"].timestamps
        st = result["channels"]["speed"].timestamps
        ss = max(float(vt[0]), float(st[0]))
        se = min(float(vt[-1]), float(st[-1]))
        if se > ss:
            support = {
                "valid": True,
                "support_start": ss,
                "support_end": se,
                "duration": se - ss,
            }
        else:
            support = {
                "valid": False,
                "support_start": ss,
                "support_end": se,
                "duration": None,
                "warning": "no temporal overlap between vibration and speed",
            }
    else:
        support = {"valid": False, "warning": "vibration_1 or speed channel missing"}

    return {
        "file": os.path.abspath(path),
        "root_name": result["root_name"],
        "channels": channels_report,
        "common_time_support": support,
    }
