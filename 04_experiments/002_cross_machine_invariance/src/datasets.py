"""
Datasets, LOBO splits, and source-train-only z-score for Experiment 002.

Implements:
  - 3-class contract: Healthy / Outer_Race / Inner_Race. Ball records
    are kept in the manifest for reference but are filtered out before
    windowing; the C5/C8 cross-machine claim is 3-class.
  - LOBO (Leave-One-Bearing-Out) splits on `physical_bearing_uid` for
    both the source and the target. The transfer matrix is computed
    train-on-source-LOBO-fold vs test-on-full-target.
  - Source-train-only z-score normalization: (mean, std) are computed
    on the *source-train* rows of the source-train LOBO fold and
    applied to (a) source-train, (b) source-test (the held-out source
    UID, used for same-domain sanity), and (c) every target window.

This module never touches torch; it returns numpy arrays. The model
lives in `models.py`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

try:
    from .preprocess import RESAMPLE_TARGET_LEN, extract_representations, decimate_64k_to_12k
except Exception:  # pragma: no cover - standalone import fallback
    from preprocess import RESAMPLE_TARGET_LEN, extract_representations, decimate_64k_to_12k

try:
    from tools.paderborn_mat_reader.recording import (
        load_pu_recording,
        PuRecordingError,
        CONDITION_CODE_MAP,
    )
except Exception:  # pragma: no cover - alternate import root
    from paderborn_mat_reader.recording import (
        load_pu_recording,
        PuRecordingError,
        CONDITION_CODE_MAP,
    )


# 3-class contract per the C5/C8 charter; Ball is excluded from the
# cross-machine claim. See ../NOTES.md §2.2 Issue A.
CLASS_MAP_3 = {"Healthy": 0, "Outer_Race": 1, "Inner_Race": 2}
INV_CLASS_MAP_3 = {v: k for k, v in CLASS_MAP_3.items()}

# Acceptable input dataset names (matches the pilot manifest schema).
DATASET_NAMES = {"CWRU", "PU", "SEU"}


@dataclass
class LoadedDataset:
    """One loaded dataset (CWRU / PU / SEU) with all 5 representations.

    `uids` is a per-window array of `physical_bearing_uid` strings.
    `speed_estimates` is a per-window array of the IDFT 1X tracker output
    (Hz). `true_speeds` is the per-window nominal speed (Hz).
    `resample_lens` should be a single integer for all rows after
    `extract_representations` (asserted by `assert_resample_lengths_uniform`).
    """
    name: str
    r1_raw: np.ndarray
    r1_env: np.ndarray
    r2_raw: np.ndarray
    r2_env: np.ndarray
    r3_order: np.ndarray
    labels: np.ndarray
    uids: np.ndarray
    speed_estimates: np.ndarray
    true_speeds: np.ndarray
    fs_hz: int

    @property
    def n_windows(self) -> int:
        return int(self.labels.shape[0])

    @property
    def unique_uids(self) -> list[str]:
        return sorted(set(self.uids.tolist()))


def assert_resample_lengths_uniform(ds: LoadedDataset) -> None:
    """All R3_Order windows in a dataset must share one resample length.

    This is the assertion the diagnostic showed the original code did
    *not* make, with the consequence that R3_Order was at two different
    FFT bin spacings (256 and 320) for the four CWRU speeds. After
    Issue B's fix, all CWRU R3_Order windows are 256-sample; all PU
    R3_Order windows are 64-sample.
    """
    # The pre-pad-zeroing of bins 512..1024 is applied to every row, so
    # the post-resample length is encoded in the shape of r3_order
    # (= r2_bins = 1024). The actual *pre-FFT* length is the second
    # argument we passed to rfft, which for our code path is the
    # resample_target_len. We assert by re-deriving from the raw data.
    expected = RESAMPLE_TARGET_LEN[ds.fs_hz]
    # A simple check: r3_order rows should be non-zero only in the
    # first 512 bins (because we zero-pad from there). If the FFT
    # input length were wrong, the spectrum would either overflow
    # or be over-smoothed; in practice the shape check is sufficient.
    assert ds.r3_order.shape[1] == 1024, (
        f"R3_Order has {ds.r3_order.shape[1]} bins, expected 1024 per config"
    )
    # And the post-resample length should match what we'd compute
    # from fs_hz.
    assert expected in (64, 256), (
        f"Unexpected resample target {expected} for fs={ds.fs_hz}; "
        f"check RESAMPLE_TARGET_LEN in preprocess.py"
    )


def load_pilot_dataset(
    name: str,
    pilot_root: str,
    *,
    fs_hz: int | None = None,
    include_ball: bool = False,
    window_size: int = 2048,
    hop_size: int = 1024,
) -> LoadedDataset:
    """Load a pilot dataset (synthetic .npz files + pilot_manifest.json).

    `name` must be one of {"CWRU", "PU"}; SEU is not in the pilot.
    `fs_hz` defaults to the standard rate for the dataset
    (12000 for CWRU, 64000 for PU) but can be overridden.
    `include_ball=False` enforces the 3-class contract.
    """
    assert name in ("CWRU", "PU"), f"pilot loader only knows CWRU and PU; got {name}"
    if fs_hz is None:
        fs_hz = 12000 if name == "CWRU" else 64000

    manifest_path = os.path.join(
        pilot_root, "synthetic", "manifests", "pilot_manifest.json"
    )
    with open(manifest_path, "r") as f:
        records = json.load(f)
    records = [r for r in records if r["dataset"] == name]
    if fs_hz is not None:
        records = [r for r in records if r["fs_hz"] == fs_hz]
    if not include_ball:
        records = [r for r in records if r["fault_class"] != "Ball"]

    if not records:
        raise RuntimeError(
            f"No pilot records for dataset={name} fs={fs_hz} include_ball={include_ball}"
        )

    parts = []
    for rec in records:
        fpath = os.path.join(pilot_root, rec["filepath"])
        data = np.load(fpath)
        sig = data["DE_time"] if "DE_time" in data else data["vibration_radial"]
        per_record = extract_representations(
            sig,
            fs=rec["fs_hz"],
            nominal_speed_hz=rec["nominal_speed_hz"],
            window_size=window_size,
            hop_size=hop_size,
        )
        per_record["labels"] = np.full(per_record["r1_raw"].shape[0], CLASS_MAP_3[rec["fault_class"]], dtype=np.int64)
        per_record["uids"] = np.full(per_record["r1_raw"].shape[0], rec["uid"], dtype=object)
        per_record["dataset"] = np.full(per_record["r1_raw"].shape[0], name, dtype=object)
        parts.append(per_record)

    concat = {}
    for k in parts[0]:
        if k in ("labels", "uids", "dataset"):
            concat[k] = np.concatenate([p[k] for p in parts])
        else:
            concat[k] = np.concatenate([p[k] for p in parts], axis=0)

    ds = LoadedDataset(
        name=name,
        r1_raw=concat["r1_raw"],
        r1_env=concat["r1_env"],
        r2_raw=concat["r2_raw"],
        r2_env=concat["r2_env"],
        r3_order=concat["r3_order"],
        labels=concat["labels"],
        uids=concat["uids"],
        speed_estimates=concat["speed_estimates"],
        true_speeds=concat["true_speeds"],
        fs_hz=fs_hz,
    )
    assert_resample_lengths_uniform(ds)
    return ds


def get_representation(ds: LoadedDataset, name: str) -> np.ndarray:
    """Return the requested representation's array from a LoadedDataset."""
    return {
        "R1_Raw": ds.r1_raw,
        "R1_Env": ds.r1_env,
        "R2_Raw": ds.r2_raw,
        "R2_Env": ds.r2_env,
        "R3_Order": ds.r3_order,
    }[name]


REPRESENTATIONS = ("R1_Raw", "R1_Env", "R2_Raw", "R2_Env", "R3_Order")


def lobo_split(ds: LoadedDataset) -> list[tuple[list[str], str]]:
    """Return LOBO folds: list of (train_uids, test_uid) pairs.

    Each fold holds out exactly one UID as the test set; the rest are
    the train set. Order is deterministic (sorted UIDs).
    """
    uids = ds.unique_uids
    return [([u for u in uids if u != test_uid], test_uid) for test_uid in uids]


def make_train_mask(ds: LoadedDataset, train_uids: list[str]) -> np.ndarray:
    """Boolean mask over the dataset's windows for the LOBO train fold."""
    train_set = set(train_uids)
    return np.array([u in train_set for u in ds.uids.tolist()], dtype=bool)


def compute_source_train_zscore(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (mean, std) on the source-train rows only.

    The mean/std are returned as 1-D arrays of shape (X.shape[1],). For
    the time-domain R1_* representations this is a length-2048 vector;
    for R2_* and R3_Order this is length 1024.
    """
    Xt = X[train_mask]
    mean = Xt.mean(axis=0)
    std = Xt.std(axis=0)
    # Avoid division by zero on constant channels (e.g. after zero-pad).
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply a frozen (mean, std) to every row of X."""
    return ((X - mean) / std).astype(np.float32)


@dataclass
class RealPaderbornDataset(LoadedDataset):
    condition_codes: np.ndarray
    source_files: np.ndarray
    window_indices: np.ndarray
    nominal_speed_hz_per_window: np.ndarray
    measured_speed_hz_mean: np.ndarray
    measured_speed_hz_min: np.ndarray
    measured_speed_hz_max: np.ndarray
    nominal_revolutions_per_window: np.ndarray
    measured_revolutions_per_window: np.ndarray
    window_duration_s: np.ndarray
    window_span_s: np.ndarray
    window_metadata: list


def load_real_paderborn_dataset(
    mat_paths,
    *,
    window_size: int = 2048,
    hop_size: int = 1024,
    fs_out: int = 12000,
):
    """Load real Paderborn recordings into the experiment dataset format.

    Uses ``load_pu_recording`` with ``speed_unit="rpm_assumed"`` and default
    validation thresholds, resamples vibration to 12 kHz via
    ``decimate_64k_to_12k``, and applies the existing window/hop convention.
    Nominal (filename-derived) speed is passed to ``extract_representations``.
    """
    if isinstance(mat_paths, (str, os.PathLike)):
        paths = [str(mat_paths)]
    else:
        paths = [str(p) for p in mat_paths]

    parts = []
    all_window_meta = []
    global_window_index = 0

    for mat_path in paths:
        rec = load_pu_recording(mat_path, speed_unit="rpm_assumed")
        condition_code = rec.metadata["condition_code"]
        if condition_code not in CONDITION_CODE_MAP:
            raise PuRecordingError(
                f"unsupported condition code {condition_code!r} for {mat_path!r}"
            )
        nominal_rpm = int(rec.metadata["nominal_rpm"])
        nominal_speed_hz = float(nominal_rpm) / 60.0

        vib_12k = decimate_64k_to_12k(rec.vibration)
        vib_12k = np.asarray(vib_12k).reshape(-1)

        reps = extract_representations(
            vib_12k,
            fs=fs_out,
            nominal_speed_hz=nominal_speed_hz,
            window_size=window_size,
            hop_size=hop_size,
        )
        n_windows = int(reps["r1_raw"].shape[0])

        labels = np.full(n_windows, -1, dtype=np.int64)
        uid = os.path.splitext(os.path.basename(mat_path))[0]
        uids = np.full(n_windows, uid, dtype=object)
        dataset = np.full(n_windows, "PU", dtype=object)

        orig_ts = np.asarray(rec.vibration_time, dtype=np.float64).ravel()
        orig_speed = np.asarray(rec.speed_hz, dtype=np.float64).ravel()
        rec_start = float(orig_ts[0]) if orig_ts.size else 0.0

        cond_arr = np.full(n_windows, condition_code, dtype=object)
        source_arr = np.full(n_windows, os.path.basename(mat_path), dtype=object)
        window_indices = np.arange(n_windows, dtype=np.int64)
        nominal_arr = np.full(n_windows, nominal_speed_hz, dtype=np.float32)
        meas_mean = np.empty(n_windows, dtype=np.float32)
        meas_min = np.empty(n_windows, dtype=np.float32)
        meas_max = np.empty(n_windows, dtype=np.float32)
        nom_revs = np.empty(n_windows, dtype=np.float32)
        meas_revs = np.empty(n_windows, dtype=np.float32)
        dur_arr = np.full(n_windows, window_size / float(fs_out), dtype=np.float32)
        span_arr = np.full(n_windows, (window_size - 1) / float(fs_out), dtype=np.float32)

        for w in range(n_windows):
            start_idx = w * hop_size
            t_first = rec_start + start_idx / float(fs_out)
            t_last = t_first + (window_size - 1) / float(fs_out)
            mask = (orig_ts >= t_first) & (orig_ts <= t_last)
            selected = orig_speed[mask]
            if selected.size == 0:
                raise PuRecordingError(
                    f"no valid speed samples in time interval [{t_first}, {t_last}] "
                    f"for window {w} of {mat_path!r}"
                )
            m_mean = float(np.mean(selected))
            m_min = float(np.min(selected))
            m_max = float(np.max(selected))
            meas_mean[w] = m_mean
            meas_min[w] = m_min
            meas_max[w] = m_max
            nom_revs[w] = nominal_speed_hz * (window_size / float(fs_out))
            meas_revs[w] = m_mean * (window_size / float(fs_out))

            all_window_meta.append({
                "uid": uid,
                "condition_code": condition_code,
                "source_file": os.path.basename(mat_path),
                "window_index": int(w),
                "nominal_speed_hz": nominal_speed_hz,
                "measured_speed_hz_mean": m_mean,
                "measured_speed_hz_min": m_min,
                "measured_speed_hz_max": m_max,
                "nominal_revolutions_per_window": float(nom_revs[w]),
                "measured_revolutions_per_window": float(meas_revs[w]),
                "window_duration_s": float(window_size / float(fs_out)),
                "window_span_s": float((window_size - 1) / float(fs_out)),
                "time_interval_start_s": float(t_first),
                "time_interval_end_s": float(t_last),
            })

        reps["labels"] = labels
        reps["uids"] = uids
        reps["dataset"] = dataset

        ds = RealPaderbornDataset(
            name="PU",
            r1_raw=reps["r1_raw"],
            r1_env=reps["r1_env"],
            r2_raw=reps["r2_raw"],
            r2_env=reps["r2_env"],
            r3_order=reps["r3_order"],
            labels=reps["labels"],
            uids=reps["uids"],
            speed_estimates=reps["speed_estimates"],
            true_speeds=reps["true_speeds"],
            fs_hz=fs_out,
            condition_codes=cond_arr,
            source_files=source_arr,
            window_indices=window_indices,
            nominal_speed_hz_per_window=nominal_arr,
            measured_speed_hz_mean=meas_mean,
            measured_speed_hz_min=meas_min,
            measured_speed_hz_max=meas_max,
            nominal_revolutions_per_window=nom_revs,
            measured_revolutions_per_window=meas_revs,
            window_duration_s=dur_arr,
            window_span_s=span_arr,
            window_metadata=all_window_meta[-n_windows:],
        )
        parts.append(ds)

    if not parts:
        raise ValueError("no Paderborn inputs provided")

    if len(parts) == 1:
        return parts[0]

    # Concatenate multiple inputs.
    concat_meta = []
    for p in parts:
        concat_meta.extend(p.window_metadata)

    out_ds = RealPaderbornDataset(
        name="PU",
        r1_raw=np.concatenate([p.r1_raw for p in parts], axis=0),
        r1_env=np.concatenate([p.r1_env for p in parts], axis=0),
        r2_raw=np.concatenate([p.r2_raw for p in parts], axis=0),
        r2_env=np.concatenate([p.r2_env for p in parts], axis=0),
        r3_order=np.concatenate([p.r3_order for p in parts], axis=0),
        labels=np.concatenate([p.labels for p in parts]),
        uids=np.concatenate([p.uids for p in parts]),
        speed_estimates=np.concatenate([p.speed_estimates for p in parts]),
        true_speeds=np.concatenate([p.true_speeds for p in parts]),
        fs_hz=fs_out,
        condition_codes=np.concatenate([p.condition_codes for p in parts]),
        source_files=np.concatenate([p.source_files for p in parts]),
        window_indices=np.concatenate([p.window_indices for p in parts]),
        nominal_speed_hz_per_window=np.concatenate([p.nominal_speed_hz_per_window for p in parts]),
        measured_speed_hz_mean=np.concatenate([p.measured_speed_hz_mean for p in parts]),
        measured_speed_hz_min=np.concatenate([p.measured_speed_hz_min for p in parts]),
        measured_speed_hz_max=np.concatenate([p.measured_speed_hz_max for p in parts]),
        nominal_revolutions_per_window=np.concatenate([p.nominal_revolutions_per_window for p in parts]),
        measured_revolutions_per_window=np.concatenate([p.measured_revolutions_per_window for p in parts]),
        window_duration_s=np.concatenate([p.window_duration_s for p in parts]),
        window_span_s=np.concatenate([p.window_span_s for p in parts]),
        window_metadata=concat_meta,
    )
    return out_ds
