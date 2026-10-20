"""Synthetic, empty-file tests for the Stage 1 filename/coverage CLI only."""

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPO_ROOT), str(EXP_ROOT), str(EXP_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.datasets as D
    import src.stage1_coverage_cli as CLI
    import src.stage1_split as S
except Exception:
    import datasets as D
    import stage1_coverage_cli as CLI
    import stage1_split as S


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_required_coverage(root: Path) -> list[Path]:
    """Build empty fixtures from check_coverage's authoritative pair list."""
    required_pairs = S.check_coverage([])["required_pairs"]
    paths = []
    for pair in required_pairs:
        path = root / (
            f"{pair['condition_code']}_{pair['bearing_id']}_1.mat"
        )
        paths.append(_touch(path))
    return paths


def _run_cli(paths, capsys):
    exit_code = CLI.main([str(path) for path in paths])
    report = json.loads(capsys.readouterr().out)
    return exit_code, report


def test_empty_mat_fixtures_pass_manifest_coverage_without_reading_payloads(
    tmp_path, monkeypatch, capsys
):
    paths = _make_required_coverage(tmp_path)

    def payload_must_not_be_opened(*args, **kwargs):
        pytest.fail("coverage CLI must not load or inspect MAT payloads")

    monkeypatch.setattr(D, "load_pu_recording", payload_must_not_be_opened)

    exit_code, report = _run_cli([tmp_path], capsys)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["discovered_path_count"] == len(paths) == 40
    assert report["coverage_input_recording_count"] == 40
    assert report["coverage"]["input_file_count"] == 40
    assert report["coverage"]["missing_required"] == []
    # The manifest explicitly treats this view as optional/non-fatal.
    assert len(report["coverage"]["secondary_condition_gaps"]) == 11
    assert report["mat_payload_opened"] is False
    assert report["signal_channels_verified"] is False
    assert report["sampling_rate_verified"] is False
    assert report["signal_length_verified"] is False
    assert report["usable_windows_verified"] is False
    assert report["authoritative_sources"]["stage1_split"].endswith(
        "config/stage1_split.json"
    )
    assert report["authoritative_sources"]["paderborn_class_map"].endswith(
        "paderborn_bearing_class_map.csv"
    )
    assert next(row for row in report["files"] if row["bearing_id"] == "KA01")["label"] == 1


def test_duplicate_identity_is_detected_across_paths_by_recording_uid(
    tmp_path, capsys
):
    first = _touch(tmp_path / "copy-a" / "N15_M07_F10_KA01_1.mat")
    # The extension differs only in case: the repository parser accepts both,
    # and the established recording UID remains the same stem.
    second = _touch(tmp_path / "copy-b" / "N15_M07_F10_KA01_1.MAT")

    exit_code, report = _run_cli([tmp_path], capsys)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert len(report["duplicate_recording_identities"]) == 1
    duplicate = report["duplicate_recording_identities"][0]
    assert duplicate["recording_uid"] == "N15_M07_F10_KA01_1"
    assert set(duplicate["paths"]) == {str(first), str(second)}
    assert all(
        entry["identity_status"] == "DUPLICATE_RECORDING_IDENTITY"
        for entry in report["files"]
    )
    assert any(
        error["code"] == "DUPLICATE_RECORDING_IDENTITY"
        for error in report["errors"]
    )


def test_approved_condition_views_of_one_test_bearing_are_not_duplicates(
    tmp_path, capsys
):
    split = S.load_split()
    bearing_id = split["allocations"]["test"][0]
    views = ("test_in_condition", "test_primary", "test_secondary")
    paths = [
        _touch(
            tmp_path
            / f"{split['conditions'][view]}_{bearing_id}_1.mat"
        )
        for view in views
    ]

    exit_code, report = _run_cli([tmp_path], capsys)

    assert exit_code == 1  # other required bearings are intentionally absent
    assert report["duplicate_recording_identities"] == []
    assert report["unresolved_identities"] == []
    assert report["coverage_input_recording_count"] == 3
    assert {entry["physical_bearing_uid"] for entry in report["files"]} == {
        bearing_id
    }
    assert {entry["condition_code"] for entry in report["files"]} == {
        split["conditions"][view] for view in views
    }
    assert {entry["recording_uid"] for entry in report["files"]} == {
        path.stem for path in paths
    }


def test_distinct_recording_ids_for_same_bearing_and_condition_are_not_duplicates(
    tmp_path, capsys
):
    paths = [
        _touch(tmp_path / "N15_M07_F10_KA01_1.mat"),
        _touch(tmp_path / "N15_M07_F10_KA01_2.mat"),
    ]

    exit_code, report = _run_cli([tmp_path], capsys)

    assert exit_code == 1  # this fixture intentionally lacks full coverage
    assert report["duplicate_recording_identities"] == []
    assert report["coverage"]["recording_counts"]["KA01"]["N15_M07_F10"] == 2
    assert {entry["recording_uid"] for entry in report["files"]} == {
        path.stem for path in paths
    }


@pytest.mark.parametrize(
    "filename",
    [
        "not_a_recording.mat",
        "N15_M07_F10_K999_1.mat",
    ],
)
def test_unresolvable_filename_identity_is_reported_and_fails(
    tmp_path, capsys, filename
):
    path = _touch(tmp_path / filename)

    exit_code, report = _run_cli([path], capsys)

    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert report["files"][0]["identity_status"] == "UNRESOLVED_IDENTITY"
    assert report["unresolved_identities"][0]["status"] == "UNRESOLVED_IDENTITY"
    assert any(error["code"] == "UNRESOLVED_IDENTITY" for error in report["errors"])
    assert report["coverage_input_recording_count"] == 0


def test_missing_explicit_file_is_reported_without_claiming_it_was_covered(
    tmp_path, capsys
):
    missing = tmp_path / "N15_M07_F10_KA01_1.mat"

    exit_code, report = _run_cli([missing], capsys)

    assert exit_code == 1
    assert report["files"][0]["file_status"] == "MISSING_OR_NOT_FILE"
    assert report["files"][0]["identity_status"] == "RESOLVED"
    assert report["coverage_input_recording_count"] == 0
    assert any(error["code"] == "FILE_NOT_PRESENT" for error in report["errors"])
