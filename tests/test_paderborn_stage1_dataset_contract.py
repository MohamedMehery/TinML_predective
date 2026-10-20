"""Identity/label contract tests for the real Paderborn dataset loader.

Most cases use an in-memory fake recording boundary so that labels,
identity, exclusion, and metadata behavior do not depend on local data.
They are not evidence of real-bearing recording availability.
"""

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPO_ROOT), str(EXP_ROOT), str(EXP_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.datasets as D
except Exception:
    import datasets as D


def _path(tmp_path, condition, bearing_id, recording_id="1"):
    return tmp_path / f"{condition}_{bearing_id}_{recording_id}.mat"


def _fake_pipeline(monkeypatch):
    """Install a fast, deterministic recording/feature boundary for contract tests."""
    calls = {"load": [], "extract": []}

    def fake_load(path, *, speed_unit):
        assert speed_unit == "rpm_assumed"
        filename = os.path.basename(os.fspath(path))
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        condition_code = "_".join(parts[:3])
        bearing_id = parts[3]
        nominal_rpm = D.CONDITION_CODE_MAP[condition_code]["nominal_rpm"]
        timestamps = np.arange(64000, dtype=np.float64) / 64000.0
        calls["load"].append(filename)
        return SimpleNamespace(
            vibration=np.zeros(64000, dtype=np.float64),
            vibration_time=timestamps,
            speed_hz=np.full(timestamps.shape, nominal_rpm / 60.0, dtype=np.float64),
            metadata={
                "bearing_id": bearing_id,
                "condition_code": condition_code,
                "nominal_rpm": nominal_rpm,
            },
        )

    def fake_decimate(signal):
        return np.zeros(4096, dtype=np.float32)

    def fake_extract(
        signal,
        fs,
        nominal_speed_hz,
        window_size=2048,
        hop_size=1024,
        **kwargs,
    ):
        n_windows = (signal.size - window_size) // hop_size + 1
        calls["extract"].append({
            "fs": fs,
            "nominal_speed_hz": nominal_speed_hz,
            "window_size": window_size,
            "hop_size": hop_size,
            **kwargs,
        })
        return {
            "r1_raw": np.zeros((n_windows, 1, window_size), dtype=np.float32),
            "r1_env": np.zeros((n_windows, 1, window_size), dtype=np.float32),
            "r2_raw": np.zeros((n_windows, 1024), dtype=np.float32),
            "r2_env": np.zeros((n_windows, 1024), dtype=np.float32),
            "r3_order": np.zeros((n_windows, 1024), dtype=np.float32),
            "speed_estimates": np.full(n_windows, nominal_speed_hz, dtype=np.float32),
            "true_speeds": np.full(n_windows, nominal_speed_hz, dtype=np.float32),
        }

    monkeypatch.setattr(D, "load_pu_recording", fake_load)
    monkeypatch.setattr(D, "decimate_64k_to_12k", fake_decimate)
    monkeypatch.setattr(D, "extract_representations", fake_extract)
    return calls


def _assert_window_alignment(ds):
    n = ds.n_windows
    fields = (
        "r1_raw", "r1_env", "r2_raw", "r2_env", "r3_order", "labels", "uids",
        "speed_estimates", "true_speeds", "condition_codes", "source_files",
        "window_indices", "nominal_speed_hz_per_window", "measured_speed_hz_mean",
        "measured_speed_hz_min", "measured_speed_hz_max",
        "nominal_revolutions_per_window", "measured_revolutions_per_window",
        "window_duration_s", "window_span_s", "physical_bearing_uids",
    )
    for field in fields:
        assert getattr(ds, field).shape[0] == n, field
    assert len(ds.window_metadata) == n


def test_manifest_labels_populate_all_three_classes_and_physical_ids(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    paths = [
        _path(tmp_path, "N15_M07_F10", "KA01"),
        _path(tmp_path, "N15_M07_F10", "K001"),
        _path(tmp_path, "N15_M07_F10", "KI01"),
    ]

    ds = D.load_real_paderborn_dataset(paths)

    assert ds.labels.tolist() == [1] * 3 + [0] * 3 + [2] * 3
    assert ds.physical_bearing_uids.tolist() == ["KA01"] * 3 + ["K001"] * 3 + ["KI01"] * 3
    assert set(ds.physical_bearing_uids.tolist()) == {"KA01", "K001", "KI01"}
    assert -1 not in ds.labels
    assert [m["class_name"] for m in (ds.window_metadata[0], ds.window_metadata[3], ds.window_metadata[6])] == [
        "Outer_Race", "Healthy", "Inner_Race"
    ]
    _assert_window_alignment(ds)


def test_same_bearing_across_conditions_and_recordings_has_distinct_recording_identity(
    tmp_path, monkeypatch
):
    _fake_pipeline(monkeypatch)
    paths = [
        _path(tmp_path, "N09_M07_F10", "KA01", "1"),
        _path(tmp_path, "N15_M07_F10", "KA01", "2"),
    ]

    ds = D.load_real_paderborn_dataset(paths)

    assert set(ds.physical_bearing_uids.tolist()) == {"KA01"}
    assert set(ds.labels.tolist()) == {1}
    assert ds.uids[0] == "N09_M07_F10_KA01_1"
    assert ds.uids[3] == "N15_M07_F10_KA01_2"
    assert len(ds.unique_uids) == 2
    assert {m["recording_id"] for m in ds.window_metadata} == {"1", "2"}
    assert {m["condition_code"] for m in ds.window_metadata} == {
        "N09_M07_F10", "N15_M07_F10"
    }
    for index, metadata in enumerate(ds.window_metadata):
        expected_condition = "N09_M07_F10" if index < 3 else "N15_M07_F10"
        expected_recording = "1" if index < 3 else "2"
        expected_source = f"{expected_condition}_KA01_{expected_recording}.mat"
        assert metadata["bearing_id"] == metadata["physical_bearing_uid"] == "KA01"
        assert metadata["recording_id"] == expected_recording
        assert metadata["condition_code"] == expected_condition
        assert metadata["source_file"] == expected_source
        assert metadata["window_index"] == index % 3
        assert metadata["label"] == 1
        assert ds.uids[index] == os.path.splitext(expected_source)[0]
    _assert_window_alignment(ds)


def test_unknown_bearing_fails_even_when_combined_exclusion_is_enabled(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    path = _path(tmp_path, "N15_M07_F10", "K999")

    with pytest.raises(ValueError, match="Unknown Paderborn bearing_id 'K999'"):
        D.load_real_paderborn_dataset(path, exclude_combined=True)


def test_combined_bearings_fail_by_default_before_signal_loading(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("excluded recording must not be opened or feature-extracted")

    monkeypatch.setattr(D, "load_pu_recording", unexpected)
    monkeypatch.setattr(D, "extract_representations", unexpected)
    for bearing_id in ("KB23", "KB24", "KB27"):
        path = _path(tmp_path, "N15_M07_F10", bearing_id)
        with pytest.raises(ValueError, match=rf"{bearing_id}.*Combined.*Stage 1"):
            D.load_real_paderborn_dataset(path)


def test_explicit_exclusion_skips_reader_and_extractor_and_logs_recording(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("excluded recording must be skipped before this call")

    monkeypatch.setattr(D, "load_pu_recording", unexpected)
    monkeypatch.setattr(D, "extract_representations", unexpected)
    path = _path(tmp_path, "N09_M07_F10", "KB23", "2")

    ds = D.load_real_paderborn_dataset(path, exclude_combined=True)

    assert ds.n_windows == 0
    assert ds.recording_exclusion_log == [{
        "bearing_id": "KB23",
        "source_file": "N09_M07_F10_KB23_2.mat",
        "condition_code": "N09_M07_F10",
        "reason": "approved Stage 1 class map marks this Combined bearing stage1_status='exclude' (label -1)",
    }]
    _assert_window_alignment(ds)


def test_mixed_included_and_excluded_inputs_preserve_alignment_and_log(tmp_path, monkeypatch):
    calls = _fake_pipeline(monkeypatch)
    paths = [
        _path(tmp_path, "N15_M07_F10", "KA01"),
        _path(tmp_path, "N09_M07_F10", "KB24", "2"),
        _path(tmp_path, "N15_M07_F10", "K001", "3"),
    ]

    ds = D.load_real_paderborn_dataset(paths, exclude_combined=True)

    assert ds.n_windows == 6
    assert ds.labels.tolist() == [1] * 3 + [0] * 3
    assert ds.physical_bearing_uids.tolist() == ["KA01"] * 3 + ["K001"] * 3
    assert len(calls["load"]) == len(calls["extract"]) == 2
    assert calls["load"] == [paths[0].name, paths[2].name]
    assert [entry["source_file"] for entry in ds.recording_exclusion_log] == [paths[1].name]
    assert all(m["source_file"] != paths[1].name for m in ds.window_metadata)
    assert -1 not in ds.labels
    _assert_window_alignment(ds)


def test_all_combined_inputs_return_correctly_shaped_empty_dataset(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("all-excluded inputs must not reach signal or feature processing")

    monkeypatch.setattr(D, "load_pu_recording", unexpected)
    monkeypatch.setattr(D, "extract_representations", unexpected)
    paths = [
        _path(tmp_path, "N15_M07_F10", "KB23"),
        _path(tmp_path, "N09_M07_F10", "KB24", "2"),
        _path(tmp_path, "N15_M07_F10", "KB27", "3"),
    ]

    ds = D.load_real_paderborn_dataset(paths, exclude_combined=True)

    assert ds.r1_raw.shape == (0, 1, 2048)
    assert ds.r1_env.shape == (0, 1, 2048)
    assert ds.r2_raw.shape == ds.r2_env.shape == ds.r3_order.shape == (0, 1024)
    assert ds.labels.dtype == np.int64
    assert ds.physical_bearing_uids.shape == ds.uids.shape == (0,)
    assert ds.window_metadata == []
    assert [entry["bearing_id"] for entry in ds.recording_exclusion_log] == [
        "KB23", "KB24", "KB27"
    ]
    _assert_window_alignment(ds)


@pytest.mark.parametrize(
    "filename",
    [
        "N09_M07_F10_KA01.mat",
        "N09_M07_F10_KA01_1_extra.mat",
        "N99_M07_F10_KA01_1.mat",
        "N09_M07_F10_KA01_1.txt",
    ],
)
def test_malformed_or_unsupported_recording_names_are_rejected(tmp_path, monkeypatch, filename):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid names must be rejected before loading signal data")

    monkeypatch.setattr(D, "load_pu_recording", unexpected)
    with pytest.raises(D.PuRecordingError):
        D.load_real_paderborn_dataset(tmp_path / filename)


def test_manifest_resolution_is_independent_of_current_directory(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    caller_cwd = tmp_path / "unrelated-cwd"
    caller_cwd.mkdir()
    path = _path(tmp_path, "N09_M07_F10", "KA01")
    monkeypatch.chdir(caller_cwd)

    ds = D.load_real_paderborn_dataset(path)

    assert set(ds.labels.tolist()) == {1}
    assert set(ds.physical_bearing_uids.tolist()) == {"KA01"}


def _write_manifest(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_rows():
    path = D._paderborn_class_map_path()
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_class_map_validation_rejects_missing_duplicate_malformed_and_inconsistent_rows(
    tmp_path, monkeypatch
):
    real_fields = list(D._PADERBORN_CLASS_MAP_FIELDS)
    rows = _manifest_rows()
    custom_path = tmp_path / "class_map.csv"
    monkeypatch.setattr(D, "_paderborn_class_map_path", lambda: custom_path)

    with pytest.raises(ValueError, match="Cannot read Paderborn class map"):
        D._load_paderborn_class_map()

    _write_manifest(custom_path, real_fields, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="Duplicate bearing_id 'K001'"):
        D._load_paderborn_class_map()

    _write_manifest(custom_path, real_fields[:-1], [{key: rows[0][key] for key in real_fields[:-1]}])
    with pytest.raises(ValueError, match="Malformed Paderborn class map.*expected columns"):
        D._load_paderborn_class_map()

    inconsistent = dict(next(row for row in rows if row["bearing_id"] == "KA01"))
    inconsistent["label"] = "2"
    _write_manifest(custom_path, real_fields, [inconsistent])
    with pytest.raises(ValueError, match="Inconsistent class_name/label.*KA01"):
        D._load_paderborn_class_map()

    blank_id = dict(rows[0])
    blank_id["bearing_id"] = ""
    _write_manifest(custom_path, real_fields, [blank_id])
    with pytest.raises(ValueError, match="bearing_id.*non-empty"):
        D._load_paderborn_class_map()


def test_synthetic_pilot_loader_keeps_legacy_labels_and_uids(tmp_path, monkeypatch):
    calls = _fake_pipeline(monkeypatch)
    pilot_root = tmp_path / "pilot"
    raw_dir = pilot_root / "synthetic" / "raw_data"
    manifest_dir = pilot_root / "synthetic" / "manifests"
    raw_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    np.savez(raw_dir / "cwru_healthy.npz", DE_time=np.zeros(4096, dtype=np.float32))
    manifest = [{
        "dataset": "CWRU",
        "uid": "CWRU_N",
        "filepath": "synthetic/raw_data/cwru_healthy.npz",
        "fault_class": "Healthy",
        "fs_hz": 12000,
        "nominal_speed_hz": 29.95,
    }]
    (manifest_dir / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ds = D.load_pilot_dataset("CWRU", str(pilot_root))

    assert ds.labels.tolist() == [0, 0, 0]
    assert ds.uids.tolist() == ["CWRU_N"] * 3
    assert not hasattr(ds, "physical_bearing_uids")
    assert len(calls["extract"]) == 1
