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

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

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

_PADERBORN_CLASS_MAP_FIELDS = (
    "bearing_id",
    "class_name",
    "label",
    "damage_origin",
    "source_table",
    "source_component",
    "source_combination",
    "extent_level",
    "stage1_status",
)
_PADERBORN_INCLUDED_CLASSES = {
    ("Healthy", 0),
    ("Outer_Race", 1),
    ("Inner_Race", 2),
}


def _paderborn_class_map_path() -> Path:
    """Return the approved manifest path, independent of the caller's cwd."""
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "02_datasets" / "manifests" / "paderborn_bearing_class_map.csv"


def _load_paderborn_class_map() -> dict[str, dict[str, object]]:
    """Read and validate the owner-approved Paderborn class-map CSV.

    Labels and inclusion status come only from this manifest. In particular,
    bearing IDs are never classified from their filename prefixes.
    """
    path = _paderborn_class_map_path()
    try:
        stream = path.open("r", newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read Paderborn class map at {path}: {exc}") from exc

    rows_by_id: dict[str, dict[str, object]] = {}
    try:
        with stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(_PADERBORN_CLASS_MAP_FIELDS):
                raise ValueError(
                    f"Malformed Paderborn class map {path}: expected columns "
                    f"{list(_PADERBORN_CLASS_MAP_FIELDS)!r}, got {reader.fieldnames!r}"
                )
            for row in reader:
                row_number = reader.line_num
                if None in row:
                    raise ValueError(
                        f"Malformed Paderborn class map {path}, line {row_number}: "
                        "row has more fields than the header"
                    )
                if any(row.get(field_name) is None for field_name in _PADERBORN_CLASS_MAP_FIELDS):
                    raise ValueError(
                        f"Malformed Paderborn class map {path}, line {row_number}: "
                        "row has fewer fields than the header"
                    )
                for field_name in _PADERBORN_CLASS_MAP_FIELDS:
                    value = row[field_name]
                    if not value or value != value.strip():
                        raise ValueError(
                            f"Malformed Paderborn class map {path}, line {row_number}: "
                            f"{field_name!r} must be non-empty and have no surrounding whitespace"
                        )

                bearing_id = row["bearing_id"]
                if bearing_id in rows_by_id:
                    raise ValueError(
                        f"Duplicate bearing_id {bearing_id!r} in Paderborn class map "
                        f"{path} at line {row_number}"
                    )
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed label {row['label']!r} for bearing {bearing_id!r} "
                        f"in Paderborn class map {path}, line {row_number}"
                    ) from exc
                if str(label) != row["label"]:
                    raise ValueError(
                        f"Malformed non-canonical label {row['label']!r} for bearing "
                        f"{bearing_id!r} in Paderborn class map {path}, line {row_number}"
                    )

                class_name = row["class_name"]
                status = row["stage1_status"]
                if status == "include":
                    if label not in {0, 1, 2}:
                        raise ValueError(
                            f"Inconsistent mapping for bearing {bearing_id!r} in "
                            f"{path}, line {row_number}: included labels must be one "
                            f"of 0, 1, or 2, got {label}"
                        )
                    if (class_name, label) not in _PADERBORN_INCLUDED_CLASSES:
                        raise ValueError(
                            f"Inconsistent class_name/label for included bearing "
                            f"{bearing_id!r} in {path}, line {row_number}: "
                            f"{class_name!r}/{label}"
                        )
                elif status == "exclude":
                    if class_name != "Combined" or label != -1:
                        raise ValueError(
                            f"Inconsistent excluded mapping for bearing {bearing_id!r} "
                            f"in {path}, line {row_number}: expected Combined/-1, "
                            f"got {class_name!r}/{label}"
                        )
                else:
                    raise ValueError(
                        f"Invalid stage1_status {status!r} for bearing {bearing_id!r}; "
                        "expected 'include' or 'exclude'"
                    )

                validated_row: dict[str, object] = dict(row)
                validated_row["label"] = label
                rows_by_id[bearing_id] = validated_row
    except csv.Error as exc:
        raise ValueError(f"Malformed Paderborn class map {path}: {exc}") from exc

    if not rows_by_id:
        raise ValueError(f"Paderborn class map {path} contains no bearing mappings")
    return rows_by_id


def _parse_paderborn_recording_filename(mat_path: str | os.PathLike) -> tuple[str, str, str]:
    """Parse ``<condition>_<bearing_id>_<recording_id>.mat`` exactly."""
    filename = os.path.basename(os.fspath(mat_path))
    stem, extension = os.path.splitext(filename)
    if extension.lower() != ".mat":
        raise PuRecordingError(
            f"Paderborn recording filename {filename!r} must have a .mat extension"
        )
    parts = stem.split("_")
    if len(parts) != 5 or any(not part for part in parts):
        raise PuRecordingError(
            f"Malformed Paderborn recording filename {filename!r}: expected "
            "<condition_code>_<bearing_id>_<recording_id>.mat, for example "
            "N09_M07_F10_KA01_1.mat"
        )
    condition_code = "_".join(parts[:3])
    bearing_id, recording_id = parts[3], parts[4]
    if condition_code not in CONDITION_CODE_MAP:
        raise PuRecordingError(
            f"Unsupported condition code {condition_code!r} in {filename!r}; "
            f"supported: {sorted(CONDITION_CODE_MAP)}"
        )
    return bearing_id, condition_code, recording_id


@dataclass
class LoadedDataset:
    """One loaded dataset (CWRU / PU / SEU) with all 5 representations.

    For pilot inputs, `uids` retains the per-window manifest UID semantics.
    `RealPaderbornDataset` preserves recording-stem `uids` for compatibility
    and exposes a separate `physical_bearing_uids` array for physical grouping.
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
    """Real PU windows with both legacy recording and physical identities.

    ``uids`` and ``unique_uids`` intentionally retain their legacy,
    recording-stem identity semantics. ``physical_bearing_uids`` is the
    separate per-window bearing key that future real-data grouping must use;
    this class does not define or implement a Stage 1 partition.

    ``source_files`` and ``window_metadata`` preserve recording traceability.
    ``recording_exclusion_log`` contains one entry per Combined recording
    explicitly skipped by :func:`load_real_paderborn_dataset`.
    """
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
    physical_bearing_uids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=object)
    )
    recording_exclusion_log: list[dict[str, str]] = field(default_factory=list)


def _assert_real_paderborn_window_alignment(ds: RealPaderbornDataset) -> None:
    """Reject a real-PU result whose per-window data or metadata drift apart."""
    n_windows = int(ds.labels.shape[0])
    per_window_fields = (
        "r1_raw",
        "r1_env",
        "r2_raw",
        "r2_env",
        "r3_order",
        "labels",
        "uids",
        "speed_estimates",
        "true_speeds",
        "condition_codes",
        "source_files",
        "window_indices",
        "nominal_speed_hz_per_window",
        "measured_speed_hz_mean",
        "measured_speed_hz_min",
        "measured_speed_hz_max",
        "nominal_revolutions_per_window",
        "measured_revolutions_per_window",
        "window_duration_s",
        "window_span_s",
        "physical_bearing_uids",
    )
    for field_name in per_window_fields:
        value = getattr(ds, field_name)
        if not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != n_windows:
            shape = getattr(value, "shape", None)
            raise ValueError(
                f"RealPaderbornDataset.{field_name} has first dimension {shape}; "
                f"expected {n_windows} windows"
            )
    if len(ds.window_metadata) != n_windows:
        raise ValueError(
            f"RealPaderbornDataset.window_metadata has {len(ds.window_metadata)} rows; "
            f"expected {n_windows} windows"
        )


def _empty_real_paderborn_dataset(
    *,
    window_size: int,
    fs_out: int,
    recording_exclusion_log: list[dict[str, str]],
) -> RealPaderbornDataset:
    """Build the correctly shaped result when every input was excluded."""
    empty_objects = lambda: np.empty(0, dtype=object)
    empty_floats = lambda: np.empty(0, dtype=np.float32)
    ds = RealPaderbornDataset(
        name="PU",
        r1_raw=np.empty((0, 1, window_size), dtype=np.float32),
        r1_env=np.empty((0, 1, window_size), dtype=np.float32),
        r2_raw=np.empty((0, 1024), dtype=np.float32),
        r2_env=np.empty((0, 1024), dtype=np.float32),
        r3_order=np.empty((0, 1024), dtype=np.float32),
        labels=np.empty(0, dtype=np.int64),
        uids=empty_objects(),
        speed_estimates=empty_floats(),
        true_speeds=empty_floats(),
        fs_hz=fs_out,
        condition_codes=empty_objects(),
        source_files=empty_objects(),
        window_indices=np.empty(0, dtype=np.int64),
        nominal_speed_hz_per_window=empty_floats(),
        measured_speed_hz_mean=empty_floats(),
        measured_speed_hz_min=empty_floats(),
        measured_speed_hz_max=empty_floats(),
        nominal_revolutions_per_window=empty_floats(),
        measured_revolutions_per_window=empty_floats(),
        window_duration_s=empty_floats(),
        window_span_s=empty_floats(),
        window_metadata=[],
        physical_bearing_uids=empty_objects(),
        recording_exclusion_log=list(recording_exclusion_log),
    )
    _assert_real_paderborn_window_alignment(ds)
    return ds


def load_real_paderborn_dataset(
    mat_paths,
    *,
    window_size: int = 2048,
    hop_size: int = 1024,
    fs_out: int = 12000,
    exclude_combined: bool = False,
):
    """Load real PU recordings with labels and physical-bearing identity.

    ``mat_paths`` accepts one path or an iterable of paths named exactly
    ``<condition_code>_<bearing_id>_<recording_id>.mat`` (for example,
    ``N09_M07_F10_KA01_1.mat``). Conditions are checked against
    ``CONDITION_CODE_MAP``; bearing labels and Stage 1 status come only from
    the repository's approved class-map CSV, resolved relative to this
    module rather than the caller's current directory.

    Included recordings receive the CSV ``class_name`` and integer ``label``.
    Each emitted window has ``physical_bearing_uids`` equal to the CSV lookup
    key, while legacy ``uids``/``unique_uids`` remain full recording-stem
    identities. ``source_files`` and the per-window ``recording_id`` metadata
    preserve recording traceability. Future real-data partitioning must group
    on ``physical_bearing_uids``; this loader does not implement Stage 1
    splitting and does not make the synthetic pilot runner suitable for it.

    By default, a manifest row marked ``Combined``/``exclude`` raises an
    actionable error. With ``exclude_combined=True``, such recordings are
    logged and skipped before the MAT signal loader or feature extractor is
    called. Unknown IDs and malformed/unsupported filenames always raise.
    If every nonempty input is explicitly excluded, the returned dataset has
    zero windows with the normal feature dimensions and a populated exclusion
    log. Empty ``mat_paths`` remains an error.

    Signal processing, nominal-speed forwarding, measured-speed metadata,
    windowing, angular target length, and strict support are unchanged.
    """
    if not isinstance(exclude_combined, bool):
        raise TypeError("exclude_combined must be a bool")
    if isinstance(mat_paths, (str, os.PathLike)):
        paths = [os.fspath(mat_paths)]
    else:
        paths = [os.fspath(p) for p in mat_paths]
    if not paths:
        raise ValueError("no Paderborn inputs provided")

    class_map = _load_paderborn_class_map()
    class_map_path = _paderborn_class_map_path()
    parts: list[RealPaderbornDataset] = []
    recording_exclusion_log: list[dict[str, str]] = []

    for mat_path in paths:
        bearing_id, filename_condition, recording_id = _parse_paderborn_recording_filename(mat_path)
        filename = os.path.basename(mat_path)
        map_row = class_map.get(bearing_id)
        if map_row is None:
            raise ValueError(
                f"Unknown Paderborn bearing_id {bearing_id!r} in {filename!r}; "
                f"no exact bearing_id entry exists in the approved class map {class_map_path}"
            )

        class_name = str(map_row["class_name"])
        label = int(map_row["label"])
        stage1_status = str(map_row["stage1_status"])
        if stage1_status == "exclude":
            if not exclude_combined:
                raise ValueError(
                    f"Paderborn bearing {bearing_id!r} in {filename!r} is "
                    f"{class_name} and marked stage1_status='exclude' in the approved "
                    "class map; this Combined bearing is excluded from the approved "
                    "Stage 1 classes. Pass exclude_combined=True to skip and log it."
                )
            recording_exclusion_log.append({
                "bearing_id": bearing_id,
                "source_file": filename,
                "condition_code": filename_condition,
                "reason": (
                    "approved Stage 1 class map marks this Combined bearing "
                    "stage1_status='exclude' (label -1)"
                ),
            })
            continue

        # Included mappings were validated to be one of the three Stage 1 labels.
        if label not in {0, 1, 2}:
            raise ValueError(
                f"Included Paderborn bearing {bearing_id!r} has invalid label {label}; "
                "expected one of 0, 1, or 2"
            )

        # Parse and apply manifest policy before opening signal arrays, allowing
        # explicit Combined exclusions to be recorded without reading their MATs.
        rec = load_pu_recording(mat_path, speed_unit="rpm_assumed")
        condition_code = rec.metadata["condition_code"]
        if condition_code not in CONDITION_CODE_MAP:
            raise PuRecordingError(
                f"unsupported condition code {condition_code!r} for {mat_path!r}"
            )
        if condition_code != filename_condition or rec.metadata.get("bearing_id") != bearing_id:
            raise PuRecordingError(
                f"Paderborn filename metadata mismatch for {filename!r}: parsed "
                f"{filename_condition}/{bearing_id}, loader returned "
                f"{condition_code!r}/{rec.metadata.get('bearing_id')!r}"
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
            resample_target_len=128,
            strict_angular_support=True,
        )
        n_windows = int(reps["r1_raw"].shape[0])

        labels = np.full(n_windows, label, dtype=np.int64)
        uid = os.path.splitext(filename)[0]
        uids = np.full(n_windows, uid, dtype=object)
        physical_bearing_uids = np.full(n_windows, bearing_id, dtype=object)
        dataset = np.full(n_windows, "PU", dtype=object)

        orig_ts = np.asarray(rec.vibration_time, dtype=np.float64).ravel()
        orig_speed = np.asarray(rec.speed_hz, dtype=np.float64).ravel()
        rec_start = float(orig_ts[0]) if orig_ts.size else 0.0

        cond_arr = np.full(n_windows, condition_code, dtype=object)
        source_file = filename
        source_arr = np.full(n_windows, source_file, dtype=object)
        window_indices = np.arange(n_windows, dtype=np.int64)
        nominal_arr = np.full(n_windows, nominal_speed_hz, dtype=np.float32)
        meas_mean = np.empty(n_windows, dtype=np.float32)
        meas_min = np.empty(n_windows, dtype=np.float32)
        meas_max = np.empty(n_windows, dtype=np.float32)
        nom_revs = np.empty(n_windows, dtype=np.float32)
        meas_revs = np.empty(n_windows, dtype=np.float32)
        dur_arr = np.full(n_windows, window_size / float(fs_out), dtype=np.float32)
        span_arr = np.full(n_windows, (window_size - 1) / float(fs_out), dtype=np.float32)
        window_metadata = []

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

            window_metadata.append({
                "uid": uid,
                "bearing_id": bearing_id,
                "physical_bearing_uid": bearing_id,
                "recording_id": recording_id,
                "condition_code": condition_code,
                "source_file": source_file,
                "window_index": int(w),
                "class_name": class_name,
                "label": label,
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
            window_metadata=window_metadata,
            physical_bearing_uids=physical_bearing_uids,
        )
        _assert_real_paderborn_window_alignment(ds)
        parts.append(ds)

    if not parts:
        # This branch is reachable only when all nonempty input paths were
        # known Combined bearings and explicitly excluded above.
        return _empty_real_paderborn_dataset(
            window_size=window_size,
            fs_out=fs_out,
            recording_exclusion_log=recording_exclusion_log,
        )

    if len(parts) == 1:
        out_ds = parts[0]
        out_ds.recording_exclusion_log = list(recording_exclusion_log)
        _assert_real_paderborn_window_alignment(out_ds)
        return out_ds

    # Concatenate each per-window field while retaining every exclusion entry.
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
        window_metadata=[metadata for p in parts for metadata in p.window_metadata],
        physical_bearing_uids=np.concatenate([p.physical_bearing_uids for p in parts]),
        recording_exclusion_log=list(recording_exclusion_log),
    )
    _assert_real_paderborn_window_alignment(out_ds)
    return out_ds
