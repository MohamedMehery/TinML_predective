"""Filename-only CLI for Stage 1 Paderborn coverage preflight.

The Stage 1 required bearing-condition pairs come exclusively from
``load_split()`` and ``check_coverage()``. Files are never opened as MAT data:
this command checks path presence, filename metadata, recording identity, and
manifest-defined coverage only. A passing report is not evidence that any MAT
payload, channel, sampling rate, signal length, or window is valid.

Run from the experiment directory with::

    python src/stage1_coverage_cli.py /path/to/paderborn/files

Paths may be files or directories; directories are searched recursively for
``.mat`` files. Malformed filenames and bearing IDs absent from the approved
class map are reported as ``UNRESOLVED_IDENTITY`` and fail the preflight,
consistent with the existing loader/parser hard-failure policy. Duplicate
recording UIDs and missing required pairs also fail; optional secondary-view
gaps retain ``check_coverage()``'s non-fatal policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Support both ``python -m src.stage1_coverage_cli`` from the experiment
# directory and direct script invocation from any working directory.
_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _import_root in (_REPOSITORY_ROOT, _EXPERIMENT_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

if __package__ in (None, ""):
    from src.datasets import (
        _load_paderborn_class_map,
        _parse_paderborn_recording_filename,
    )
    from src.stage1_split import check_coverage, load_split
else:
    from .datasets import (
        _load_paderborn_class_map,
        _parse_paderborn_recording_filename,
    )
    from .stage1_split import check_coverage, load_split


_SCOPE_NOTE = (
    "Filename-derived metadata, path presence, and Stage 1 coverage only. "
    "No MAT payload is opened or validated."
)


def _expand_inputs(inputs: Sequence[str]) -> tuple[list[Path], list[dict[str, str]]]:
    """Expand directory inputs without inspecting file contents."""
    candidates: list[Path] = []
    discovery_errors: list[dict[str, str]] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            try:
                candidates.extend(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.suffix.lower() == ".mat" and not candidate.is_dir()
                )
            except OSError as exc:
                discovery_errors.append(
                    {
                        "code": "DIRECTORY_SCAN_FAILED",
                        "path": os.fspath(path),
                        "message": str(exc),
                    }
                )
        else:
            # Include missing and wrongly suffixed explicit paths in the
            # report, so presence/parser failures are visible rather than
            # silently dropped from the preflight.
            candidates.append(path)

    candidates.sort(key=os.fspath)
    return candidates, discovery_errors


def _base_report(inputs: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FAIL",
        "evidence_scope": _SCOPE_NOTE,
        "mat_payload_opened": False,
        "signal_channels_verified": False,
        "sampling_rate_verified": False,
        "signal_length_verified": False,
        "usable_windows_verified": False,
        "input_paths": [os.fspath(path) for path in inputs],
        "authoritative_sources": {
            "stage1_split": "04_experiments/002_cross_machine_invariance/config/stage1_split.json",
            "paderborn_class_map": "02_datasets/manifests/paderborn_bearing_class_map.csv",
        },
        "discovered_path_count": 0,
        "coverage_input_recording_count": 0,
        "files": [],
        "duplicate_recording_identities": [],
        "unresolved_identities": [],
        "coverage": None,
        "errors": [],
    }


def build_report(inputs: Sequence[str]) -> dict[str, Any]:
    """Validate inputs and return a JSON-serializable filename-only report."""
    report = _base_report(inputs)
    try:
        split = load_split()
        class_map = _load_paderborn_class_map()
    except Exception as exc:
        report["errors"].append(
            {
                "code": "AUTHORITATIVE_SPLIT_UNAVAILABLE",
                "message": str(exc),
            }
        )
        return report

    candidates, discovery_errors = _expand_inputs(inputs)
    report["discovered_path_count"] = len(candidates)
    report["errors"].extend(discovery_errors)

    allocation_by_bearing = {
        bearing_id: allocation_name
        for allocation_name, bearing_ids in split["allocations"].items()
        for bearing_id in bearing_ids
    }
    identity_to_file_indices: dict[tuple[str, str, str], list[int]] = {}
    coverage_path_by_identity: dict[tuple[str, str, str], Path] = {}

    for path in candidates:
        file_present = path.is_file()
        entry: dict[str, Any] = {
            "path": os.fspath(path),
            "file_status": "PRESENT" if file_present else "MISSING_OR_NOT_FILE",
            "identity_status": "UNRESOLVED_IDENTITY",
        }
        report["files"].append(entry)
        entry_index = len(report["files"]) - 1

        if not file_present:
            report["errors"].append(
                {
                    "code": "FILE_NOT_PRESENT",
                    "path": os.fspath(path),
                    "message": "Input path is not a present regular file",
                }
            )

        try:
            bearing_id, condition_code, recording_id = (
                _parse_paderborn_recording_filename(path)
            )
        except (TypeError, ValueError) as exc:
            entry["identity_status"] = "UNRESOLVED_IDENTITY"
            entry["identity_error"] = str(exc)
            report["unresolved_identities"].append(
                {"path": os.fspath(path), "status": "UNRESOLVED_IDENTITY", "reason": str(exc)}
            )
            report["errors"].append(
                {
                    "code": "UNRESOLVED_IDENTITY",
                    "path": os.fspath(path),
                    "message": str(exc),
                }
            )
            continue

        entry.update(
            {
                "bearing_id": bearing_id,
                "physical_bearing_uid": bearing_id,
                "condition_code": condition_code,
                "recording_id": recording_id,
            }
        )
        class_row = class_map.get(bearing_id)
        allocation_name = allocation_by_bearing.get(bearing_id)
        if class_row is None or allocation_name is None:
            reason = (
                f"Parsed bearing ID {bearing_id!r} has no identity in the "
                "approved class map and validated Stage 1 split"
            )
            entry["identity_status"] = "UNRESOLVED_IDENTITY"
            entry["identity_error"] = reason
            report["unresolved_identities"].append(
                {"path": os.fspath(path), "status": "UNRESOLVED_IDENTITY", "reason": reason}
            )
            report["errors"].append(
                {
                    "code": "UNRESOLVED_IDENTITY",
                    "path": os.fspath(path),
                    "message": reason,
                }
            )
            continue

        # This matches RealPaderbornDataset.uids: the recording stem contains
        # condition, physical bearing, and recording ID. The condition is part
        # of recording identity, so approved condition views of the same held-
        # out bearing remain distinct observations rather than false duplicates.
        recording_uid = f"{condition_code}_{bearing_id}_{recording_id}"
        entry.update(
            {
                "recording_uid": recording_uid,
                "class_name": str(class_row["class_name"]),
                "label": int(class_row["label"]),
                "stage1_status": str(class_row["stage1_status"]),
                "allocation": allocation_name,
                "identity_status": "RESOLVED",
            }
        )

        if not file_present:
            continue

        identity = (condition_code, bearing_id, recording_id)
        identity_to_file_indices.setdefault(identity, []).append(entry_index)
        coverage_path_by_identity.setdefault(identity, path)

    for identity, file_indices in identity_to_file_indices.items():
        if len(file_indices) < 2:
            continue
        condition_code, bearing_id, recording_id = identity
        paths = [report["files"][index]["path"] for index in file_indices]
        duplicate = {
            "recording_uid": f"{condition_code}_{bearing_id}_{recording_id}",
            "condition_code": condition_code,
            "physical_bearing_uid": bearing_id,
            "recording_id": recording_id,
            "paths": paths,
        }
        report["duplicate_recording_identities"].append(duplicate)
        for index in file_indices:
            report["files"][index]["identity_status"] = (
                "DUPLICATE_RECORDING_IDENTITY"
            )
        report["errors"].append(
            {
                "code": "DUPLICATE_RECORDING_IDENTITY",
                **duplicate,
                "message": (
                    "Multiple present paths resolve to the same repository "
                    "recording UID"
                ),
            }
        )

    coverage_paths = list(coverage_path_by_identity.values())
    report["coverage_input_recording_count"] = len(coverage_paths)
    try:
        # The required pairs and secondary-view policy are defined only by
        # check_coverage() over load_split(); this CLI does not rebuild them.
        coverage = check_coverage(coverage_paths, strict=False)
    except Exception as exc:
        report["errors"].append(
            {"code": "COVERAGE_CHECK_FAILED", "message": str(exc)}
        )
        return report

    report["coverage"] = coverage
    if coverage["missing_required"]:
        report["errors"].append(
            {
                "code": "MISSING_REQUIRED_COVERAGE",
                "missing_pair_count": len(coverage["missing_required"]),
                "pairs": coverage["missing_required"],
            }
        )

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check Paderborn Stage 1 filename metadata, file presence, and "
            "manifest-defined coverage. MAT contents are never opened."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="MAT file path(s) or directory path(s) to scan recursively",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the preflight report as JSON; return nonzero on hard failures."""
    args = _make_argparser().parse_args(argv)
    report = build_report(args.paths)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    raise SystemExit(main())
