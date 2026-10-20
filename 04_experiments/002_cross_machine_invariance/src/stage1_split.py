"""Stage 1 physical-bearing split and filename-only coverage checks.

This module defines allocations of physical Paderborn bearings. Evaluation
condition names are views of the same held-out ``test`` allocation, not new
bearing partitions. Coverage checks inspect only recording filenames; they do
not open or load MAT files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .datasets import (
        CONDITION_CODE_MAP,
        _load_paderborn_class_map,
        _parse_paderborn_recording_filename,
    )
except ImportError:  # pragma: no cover - standalone import from src/
    from datasets import (
        CONDITION_CODE_MAP,
        _load_paderborn_class_map,
        _parse_paderborn_recording_filename,
    )


_SPLIT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "stage1_split.json"
)
_ALLOCATION_NAMES = ("train", "validation", "test", "excluded")
_EXPECTED_CONDITION_ASSIGNMENT = {
    "train": "N15_M07_F10",
    "validation": "N15_M07_F10",
    "test_in_condition": "N15_M07_F10",
    "test_primary": "N09_M07_F10",
    "test_secondary": "N15_M07_F04",
}
_EXPECTED_LABELS = {0, 1, 2}
_EXPECTED_DAMAGE_ORIGINS = {"artificial", "real"}
_DAMAGE_CLASSES = ("Outer_Race", "Inner_Race")


def _validate_split_definition(
    definition: Any,
    class_map: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Validate a decoded split against the already validated class-map CSV."""
    if not isinstance(definition, dict):
        raise ValueError("Stage 1 split JSON must contain a top-level object")

    expected_top_level = {"schema_version", "allocations", "conditions"}
    if set(definition) != expected_top_level:
        missing = sorted(expected_top_level - set(definition))
        extra = sorted(set(definition) - expected_top_level)
        raise ValueError(
            "Stage 1 split JSON must have exactly schema_version, allocations, "
            f"and conditions (missing={missing}, unexpected={extra})"
        )
    schema_version = definition["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(
            f"Unsupported Stage 1 split schema_version {schema_version!r}; expected 1"
        )

    allocations = definition["allocations"]
    if not isinstance(allocations, dict) or set(allocations) != set(_ALLOCATION_NAMES):
        actual = (
            sorted(allocations)
            if isinstance(allocations, dict)
            else type(allocations).__name__
        )
        raise ValueError(
            "Stage 1 allocations must define exactly train, validation, test, and "
            f"excluded (got {actual!r})"
        )

    normalized_allocations: dict[str, list[str]] = {}
    owner_by_bearing: dict[str, str] = {}
    for allocation_name in _ALLOCATION_NAMES:
        bearing_ids = allocations[allocation_name]
        if not isinstance(bearing_ids, list):
            raise ValueError(
                f"Stage 1 allocation {allocation_name!r} must be a JSON array of bearing IDs"
            )
        if any(
            not isinstance(bearing_id, str)
            or not bearing_id
            or bearing_id != bearing_id.strip()
            for bearing_id in bearing_ids
        ):
            raise ValueError(
                f"Stage 1 allocation {allocation_name!r} contains an empty, non-string, "
                "or whitespace-padded bearing ID"
            )

        local_seen: set[str] = set()
        for bearing_id in bearing_ids:
            if bearing_id not in class_map:
                raise ValueError(
                    f"Unknown bearing ID {bearing_id!r} in Stage 1 allocation "
                    f"{allocation_name!r}; check the approved class-map CSV"
                )
            if bearing_id in local_seen:
                raise ValueError(
                    f"Duplicate bearing ID {bearing_id!r} within Stage 1 allocation "
                    f"{allocation_name!r}"
                )
            local_seen.add(bearing_id)

            row = class_map[bearing_id]
            status = row["stage1_status"]
            if bearing_id.startswith("KB") and allocation_name != "excluded":
                raise ValueError(
                    f"KB bearing {bearing_id!r} may appear only in the excluded allocation"
                )
            if allocation_name == "excluded":
                if status != "exclude":
                    raise ValueError(
                        f"Bearing {bearing_id!r} is marked stage1_status={status!r} in the "
                        "class map and cannot be placed in excluded"
                    )
            elif status != "include":
                raise ValueError(
                    f"Bearing {bearing_id!r} in allocation {allocation_name!r} has "
                    f"stage1_status={status!r}; only class-map include bearings may be "
                    "assigned to train, validation, or test"
                )

            previous_owner = owner_by_bearing.get(bearing_id)
            if previous_owner is not None:
                raise ValueError(
                    f"Stage 1 allocations must be pairwise disjoint: bearing "
                    f"{bearing_id!r} appears in both {previous_owner!r} and "
                    f"{allocation_name!r}"
                )
            owner_by_bearing[bearing_id] = allocation_name

        normalized_allocations[allocation_name] = list(bearing_ids)

    included_ids = {
        bearing_id
        for bearing_id, row in class_map.items()
        if row["stage1_status"] == "include"
    }
    excluded_ids = {
        bearing_id
        for bearing_id, row in class_map.items()
        if row["stage1_status"] == "exclude"
    }
    assigned_included = {
        bearing_id
        for allocation_name in ("train", "validation", "test")
        for bearing_id in normalized_allocations[allocation_name]
    }
    if assigned_included != included_ids:
        missing = sorted(included_ids - assigned_included)
        unexpected = sorted(assigned_included - included_ids)
        raise ValueError(
            "Every class-map stage1_status=include bearing must occur exactly once "
            f"in train, validation, or test (missing={missing}, unexpected={unexpected})"
        )

    assigned_excluded = set(normalized_allocations["excluded"])
    if assigned_excluded != excluded_ids:
        missing = sorted(excluded_ids - assigned_excluded)
        unexpected = sorted(assigned_excluded - excluded_ids)
        raise ValueError(
            "The excluded allocation must contain exactly the class-map "
            f"stage1_status=exclude bearings (missing={missing}, unexpected={unexpected})"
        )

    for allocation_name in ("train", "validation", "test"):
        labels = {
            int(class_map[bearing_id]["label"])
            for bearing_id in normalized_allocations[allocation_name]
        }
        if labels != _EXPECTED_LABELS:
            missing = sorted(_EXPECTED_LABELS - labels)
            raise ValueError(
                f"Stage 1 allocation {allocation_name!r} must contain all three labels "
                f"0, 1, and 2 (missing labels={missing})"
            )

    for allocation_name in ("train", "test"):
        for class_name in _DAMAGE_CLASSES:
            origins = {
                str(class_map[bearing_id]["damage_origin"])
                for bearing_id in normalized_allocations[allocation_name]
                if class_map[bearing_id]["class_name"] == class_name
            }
            missing = sorted(_EXPECTED_DAMAGE_ORIGINS - origins)
            if missing:
                raise ValueError(
                    f"Stage 1 allocation {allocation_name!r}, class {class_name!r}, "
                    "must include both artificial and real damage_origin values "
                    f"(missing={missing})"
                )

    conditions = definition["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != set(
        _EXPECTED_CONDITION_ASSIGNMENT
    ):
        actual = sorted(conditions) if isinstance(conditions, dict) else type(conditions).__name__
        raise ValueError(
            "Stage 1 conditions must define train, validation, test_in_condition, "
            f"test_primary, and test_secondary (got {actual!r})"
        )
    for view_name, expected_code in _EXPECTED_CONDITION_ASSIGNMENT.items():
        condition_code = conditions[view_name]
        if not isinstance(condition_code, str) or condition_code not in CONDITION_CODE_MAP:
            raise ValueError(
                f"Unsupported condition code {condition_code!r} for Stage 1 view "
                f"{view_name!r}; supported: {sorted(CONDITION_CODE_MAP)}"
            )
        if condition_code != expected_code:
            raise ValueError(
                f"Stage 1 view {view_name!r} must use condition {expected_code!r}, "
                f"not {condition_code!r}"
            )

    return {
        "schema_version": 1,
        "allocations": normalized_allocations,
        "conditions": dict(conditions),
    }


def load_split() -> dict[str, Any]:
    """Read and validate the Stage 1 JSON split against the approved class map."""
    try:
        with _SPLIT_PATH.open("r", encoding="utf-8") as stream:
            definition = json.load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read Stage 1 split at {_SPLIT_PATH}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed Stage 1 split JSON at {_SPLIT_PATH}, line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    class_map = _load_paderborn_class_map()
    return _validate_split_definition(definition, class_map)


def partition_mask(
    physical_bearing_uids: Any,
    condition_codes: Any,
    view: str,
) -> np.ndarray:
    """Return a row mask using physical bearing IDs and the requested condition.

    ``physical_bearing_uids`` must be the per-window physical bearing identity;
    legacy recording-stem ``uids`` are intentionally not used for partitioning.
    ``train`` and ``validation`` use their allocated bearings at N15_M07_F10.
    The three ``test_*`` views reuse the same held-out test bearings at their
    respective evaluation conditions.
    """
    split = load_split()
    uids = np.asarray(physical_bearing_uids, dtype=object)
    conditions = np.asarray(condition_codes, dtype=object)
    if uids.ndim != 1 or conditions.ndim != 1 or uids.shape != conditions.shape:
        raise ValueError(
            "physical_bearing_uids and condition_codes must be one-dimensional arrays "
            "with the same length"
        )

    if view in ("train", "validation"):
        allocation_name = view
    elif view in ("test_in_condition", "test_primary", "test_secondary"):
        allocation_name = "test"
    else:
        valid_views = ("train", "validation", "test_in_condition", "test_primary", "test_secondary")
        raise ValueError(f"Unknown Stage 1 view {view!r}; choose one of {valid_views}")

    condition_code = split["conditions"][view]
    allocated_ids = split["allocations"][allocation_name]
    return np.asarray(
        np.isin(uids, np.asarray(allocated_ids, dtype=object))
        & (conditions == condition_code),
        dtype=bool,
    )


def check_coverage(mat_paths: Any, *, strict: bool = False) -> dict[str, Any]:
    """Count recording filenames by physical bearing and condition.

    This is a filename-only dry run: each path is passed to the validated
    filename parser reused from ``datasets.py``; no MAT file is opened. Missing
    train/validation and primary test-condition pairs are returned in the
    report. Missing ``test_secondary`` pairs are reported separately but are
    non-fatal, including when ``strict=True``.
    """
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool")

    split = load_split()
    all_bearing_ids = [
        bearing_id
        for allocation_name in _ALLOCATION_NAMES
        for bearing_id in split["allocations"][allocation_name]
    ]
    condition_order = list(CONDITION_CODE_MAP)
    recording_counts = {
        bearing_id: {condition_code: 0 for condition_code in condition_order}
        for bearing_id in all_bearing_ids
    }

    if isinstance(mat_paths, (str, os.PathLike)):
        paths = [mat_paths]
    else:
        paths = list(mat_paths)

    for mat_path in paths:
        bearing_id, condition_code, _recording_id = _parse_paderborn_recording_filename(
            mat_path
        )
        if bearing_id not in recording_counts:
            raise ValueError(
                f"Filename {os.path.basename(os.fspath(mat_path))!r} contains bearing "
                f"ID {bearing_id!r}, which is not assigned in the validated Stage 1 split"
            )
        recording_counts[bearing_id][condition_code] += 1

    required_pairs: list[dict[str, str]] = []
    for allocation_name in ("train", "validation"):
        condition_code = split["conditions"][allocation_name]
        view_name = allocation_name
        for bearing_id in split["allocations"][allocation_name]:
            required_pairs.append(
                {
                    "allocation": allocation_name,
                    "view": view_name,
                    "bearing_id": bearing_id,
                    "condition_code": condition_code,
                }
            )

    for view_name in ("test_in_condition", "test_primary"):
        condition_code = split["conditions"][view_name]
        for bearing_id in split["allocations"]["test"]:
            required_pairs.append(
                {
                    "allocation": "test",
                    "view": view_name,
                    "bearing_id": bearing_id,
                    "condition_code": condition_code,
                }
            )

    missing_required = [
        {**pair, "recording_count": 0}
        for pair in required_pairs
        if recording_counts[pair["bearing_id"]][pair["condition_code"]] == 0
    ]
    secondary_condition = split["conditions"]["test_secondary"]
    secondary_gaps = [
        {
            "allocation": "test",
            "view": "test_secondary",
            "bearing_id": bearing_id,
            "condition_code": secondary_condition,
            "recording_count": 0,
        }
        for bearing_id in split["allocations"]["test"]
        if recording_counts[bearing_id][secondary_condition] == 0
    ]

    report: dict[str, Any] = {
        "recording_counts": recording_counts,
        "required_pairs": required_pairs,
        "missing_required": missing_required,
        "secondary_condition_gaps": secondary_gaps,
        "input_file_count": len(paths),
    }
    if strict and missing_required:
        lines = [
            "Missing required Stage 1 recording pairs "
            f"({len(missing_required)} total):"
        ]
        lines.extend(
            "- {allocation}/{view}: {bearing_id} @ {condition_code} "
            "({recording_count} recordings)".format(**pair)
            for pair in missing_required
        )
        if secondary_gaps:
            lines.append(
                "Non-fatal test_secondary coverage gaps "
                f"({len(secondary_gaps)} total):"
            )
            lines.extend(
                "- {bearing_id} @ {condition_code}".format(**pair)
                for pair in secondary_gaps
            )
        raise ValueError("\n".join(lines))

    return report
