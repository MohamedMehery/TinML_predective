"""Stage 1 bearing-level split, view-mask, and filename coverage contracts."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPO_ROOT), str(EXP_ROOT), str(EXP_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.datasets as D
    import src.stage1_split as S
except Exception:
    import datasets as D
    import stage1_split as S


N15_M07_F10 = "N15_M07_F10"
N09_M07_F10 = "N09_M07_F10"
N15_M07_F04 = "N15_M07_F04"

EXPECTED_ALLOCATIONS = {
    "train": [
        "K002", "K004", "K006",
        "KA01", "KA05", "KA07", "KA04", "KA15",
        "KI01", "KI07", "KI04", "KI14", "KI18",
    ],
    "validation": ["K005", "KA09", "KA16", "KI03", "KI17"],
    "test": [
        "K001", "K003",
        "KA03", "KA06", "KA08", "KA22", "KA30",
        "KI05", "KI08", "KI16", "KI21",
    ],
    "excluded": ["KB23", "KB24", "KB27"],
}
EXPECTED_CONDITIONS = {
    "train": N15_M07_F10,
    "validation": N15_M07_F10,
    "test_in_condition": N15_M07_F10,
    "test_primary": N09_M07_F10,
    "test_secondary": N15_M07_F04,
}


def _split_json():
    path = EXP_ROOT / "config" / "stage1_split.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(definition):
    return S._validate_split_definition(definition, D._load_paderborn_class_map())


def _recording(condition_code, bearing_id, recording_id="1"):
    # These intentionally do not exist: coverage must never open MAT files.
    return Path("/filename-only-dry-run") / (
        f"{condition_code}_{bearing_id}_{recording_id}.mat"
    )


def _required_filenames(excluded_pairs=()):
    excluded_pairs = set(excluded_pairs)
    split = S.load_split()
    paths = []
    for allocation_name in ("train", "validation"):
        condition = split["conditions"][allocation_name]
        for bearing_id in split["allocations"][allocation_name]:
            pair = (allocation_name, allocation_name, bearing_id, condition)
            if pair not in excluded_pairs:
                paths.append(_recording(condition, bearing_id))
    for view in ("test_in_condition", "test_primary"):
        condition = split["conditions"][view]
        for bearing_id in split["allocations"]["test"]:
            pair = ("test", view, bearing_id, condition)
            if pair not in excluded_pairs:
                paths.append(_recording(condition, bearing_id))
    return paths


def test_load_split_validates_approved_csv_and_exact_owner_allocations():
    split = S.load_split()
    class_map = D._load_paderborn_class_map()

    assert split["allocations"] == EXPECTED_ALLOCATIONS
    assert split["conditions"] == EXPECTED_CONDITIONS
    assert {name: len(ids) for name, ids in split["allocations"].items()} == {
        "train": 13,
        "validation": 5,
        "test": 11,
        "excluded": 3,
    }
    assert len(set().union(*map(set, split["allocations"].values()))) == 32
    class_map_included = {
        bearing_id
        for bearing_id, row in class_map.items()
        if row["stage1_status"] == "include"
    }
    split_included = set(
        split["allocations"]["train"]
        + split["allocations"]["validation"]
        + split["allocations"]["test"]
    )
    assert class_map_included == split_included
    assert all(
        class_map[bearing_id]["stage1_status"] == "exclude"
        for bearing_id in split["allocations"]["excluded"]
    )


def test_duplicate_bearing_within_one_allocation_is_actionable():
    definition = copy.deepcopy(_split_json())
    definition["allocations"]["train"].append("K002")
    with pytest.raises(ValueError, match="Duplicate bearing ID 'K002'.*train"):
        _validate(definition)


def test_duplicate_across_allocations_is_rejected_as_non_disjoint():
    definition = copy.deepcopy(_split_json())
    definition["allocations"]["validation"].append("K002")
    with pytest.raises(ValueError, match="pairwise disjoint.*K002"):
        _validate(definition)


def test_unknown_bearing_is_reported_against_approved_class_map():
    definition = copy.deepcopy(_split_json())
    definition["allocations"]["train"].append("K999")
    with pytest.raises(ValueError, match="Unknown bearing ID 'K999'.*class-map CSV"):
        _validate(definition)


def test_kb_bearing_cannot_be_placed_in_an_included_allocation():
    definition = copy.deepcopy(_split_json())
    definition["allocations"]["train"].append("KB23")
    with pytest.raises(ValueError, match="KB bearing 'KB23'.*only in the excluded"):
        _validate(definition)


def test_missing_class_in_an_allocation_is_rejected():
    definition = copy.deepcopy(_split_json())
    definition["allocations"]["validation"].remove("K005")
    definition["allocations"]["train"].append("K005")
    with pytest.raises(ValueError, match=r"allocation 'validation'.*missing labels=\[0\]"):
        _validate(definition)


def test_missing_artificial_or_real_origin_for_each_fault_class_is_rejected():
    definition = copy.deepcopy(_split_json())
    artificial_outer_race = ("KA01", "KA05", "KA07")
    for bearing_id in artificial_outer_race:
        definition["allocations"]["train"].remove(bearing_id)
        definition["allocations"]["validation"].append(bearing_id)
    with pytest.raises(
        ValueError,
        match=r"allocation 'train'.*class 'Outer_Race'.*missing=\['artificial'\]",
    ):
        _validate(definition)


def test_repeated_recordings_of_a_physical_bearing_stay_in_one_allocation():
    paths = [
        _recording(N15_M07_F10, "KA01", "1"),
        _recording(N15_M07_F10, "KA01", "2"),
    ]
    report = S.check_coverage(paths)
    assert report["recording_counts"]["KA01"][N15_M07_F10] == 2

    # Multiple window rows from those recordings share the physical UID. The
    # bearing-level mask cannot send one recording to validation or test.
    physical_uids = np.array(["KA01"] * 4, dtype=object)
    conditions = np.array(
        [N15_M07_F10, N15_M07_F10, N09_M07_F10, N15_M07_F04],
        dtype=object,
    )
    assert S.partition_mask(physical_uids, conditions, "train").tolist() == [
        True, True, False, False
    ]
    assert not S.partition_mask(physical_uids, conditions, "validation").any()
    assert not S.partition_mask(physical_uids, conditions, "test_primary").any()


def test_test_bearing_is_reused_across_three_views_not_train_or_validation():
    # KA03 is one physical test UID observed in three condition windows.
    physical_uids = np.array(
        ["KA03"] * 3 + ["KA01"] * 3 + ["KA09"] * 3,
        dtype=object,
    )
    conditions = np.array(
        [N15_M07_F10, N09_M07_F10, N15_M07_F04]
        + [N15_M07_F10, N09_M07_F10, N15_M07_F04]
        + [N15_M07_F10, N09_M07_F10, N15_M07_F04],
        dtype=object,
    )

    in_condition = S.partition_mask(physical_uids, conditions, "test_in_condition")
    primary = S.partition_mask(physical_uids, conditions, "test_primary")
    secondary = S.partition_mask(physical_uids, conditions, "test_secondary")
    assert in_condition[:3].tolist() == [True, False, False]
    assert primary[:3].tolist() == [False, True, False]
    assert secondary[:3].tolist() == [False, False, True]

    train = S.partition_mask(physical_uids, conditions, "train")
    validation = S.partition_mask(physical_uids, conditions, "validation")
    assert not train[:3].any() and not validation[:3].any()
    assert train[3:6].tolist() == [True, False, False]
    assert validation[6:9].tolist() == [True, False, False]
    assert not train[6:9].any() and not validation[3:6].any()


def test_complete_filename_coverage_passes_and_secondary_gaps_are_nonfatal():
    paths = _required_filenames()
    # One optional secondary recording is enough to show that its remaining
    # gaps are reported separately and do not make strict primary coverage fail.
    paths.append(_recording(N15_M07_F04, "KA03"))

    report = S.check_coverage(paths, strict=True)
    assert report["input_file_count"] == 41
    assert report["missing_required"] == []
    assert len(report["secondary_condition_gaps"]) == 10
    assert report["recording_counts"]["K002"][N15_M07_F10] == 1
    assert report["recording_counts"]["KA03"][N09_M07_F10] == 1
    assert report["recording_counts"]["KA03"][N15_M07_F04] == 1
    assert set(report["recording_counts"]) == set(
        EXPECTED_ALLOCATIONS["train"]
        + EXPECTED_ALLOCATIONS["validation"]
        + EXPECTED_ALLOCATIONS["test"]
        + EXPECTED_ALLOCATIONS["excluded"]
    )


def test_strict_coverage_lists_every_missing_required_pair_at_once():
    omitted = {
        ("train", "train", "K002", N15_M07_F10),
        ("validation", "validation", "K005", N15_M07_F10),
        ("test", "test_in_condition", "KA03", N15_M07_F10),
        ("test", "test_primary", "KI21", N09_M07_F10),
    }
    paths = _required_filenames(excluded_pairs=omitted)

    with pytest.raises(ValueError) as exc_info:
        S.check_coverage(paths, strict=True)

    message = str(exc_info.value)
    assert "Missing required Stage 1 recording pairs (4 total)" in message
    for allocation, view, bearing_id, condition in omitted:
        assert f"{allocation}/{view}: {bearing_id} @ {condition} (0 recordings)" in message
    assert message.count("recordings)") == 4


def test_malformed_or_unassigned_input_is_rejected_by_shared_filename_parser():
    with pytest.raises(ValueError, match="Malformed Paderborn recording filename"):
        S.check_coverage([Path("/filename-only-dry-run/not_a_recording.mat")])
    with pytest.raises(ValueError, match="not assigned in the validated Stage 1 split"):
        S.check_coverage([_recording(N15_M07_F10, "K999")])
