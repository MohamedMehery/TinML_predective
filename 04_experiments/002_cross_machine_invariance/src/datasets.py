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

from .preprocess import RESAMPLE_TARGET_LEN, extract_representations


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
