"""Stage 1 real-Paderborn training and bearing-disjoint evaluation runner.

This runner uses the approved Stage 1 split and filename coverage interfaces.
It never changes the split or preprocessing protocol. Run from the experiment
folder, or invoke the script by path from any directory::

    python src/run_stage1.py --data-root /path/to/paderborn \
        --out-dir /path/to/stage1-results

Filename coverage is a preflight only; it does not verify MAT payloads,
channels, sampling rate, signal length, or usable windows. ``--dry-run`` writes
a plan and performs no MAT loading, feature extraction, model construction, or
training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Keep both package and direct-script invocation independent of cwd.
_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for _import_root in (_REPOSITORY_ROOT, _EXPERIMENT_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

if __package__ in (None, ""):
    from src import datasets as ds_mod
    from src import stage1_split as split_mod
    from src.stage1_coverage_cli import build_report as build_coverage_report
else:
    from . import datasets as ds_mod
    from . import stage1_split as split_mod
    from .stage1_coverage_cli import build_report as build_coverage_report


DEFAULT_SEEDS = (42, 1337, 2026, 7, 99)
DEFAULT_REPRESENTATIONS = tuple(ds_mod.REPRESENTATIONS)
EVALUATION_VIEWS = ("test_in_condition", "test_primary", "test_secondary")
FIXED_LABELS = (0, 1, 2)
RESULT_COLUMNS = (
    "representation",
    "seed",
    "view",
    "condition",
    "n_windows",
    "n_bearings",
    "macro_f1",
    "f1_healthy",
    "f1_outer",
    "f1_inner",
    "ci_lo",
    "ci_hi",
    "class_weights_used",
    "timestamp",
)


class Stage1RunError(RuntimeError):
    """Raised for a Stage 1 precondition or partition-contract failure."""


def _load_training_api():
    """Import torch-backed training code only when training is requested."""
    if __package__ in (None, ""):
        from src import train as train_mod
    else:
        from . import train as train_mod
    return train_mod


def _load_evaluation_api():
    if __package__ in (None, ""):
        from src import evaluate as evaluate_mod
    else:
        from . import evaluate as evaluate_mod
    return evaluate_mod


def _parse_representations(value: str) -> list[str]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    allowed = set(ds_mod.REPRESENTATIONS)
    if not requested:
        raise argparse.ArgumentTypeError("at least one representation is required")
    if len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("representations must not be repeated")
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown representations {unknown}; choose from {sorted(allowed)}"
        )
    return requested


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must not be repeated")
    return seeds


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 1 real-Paderborn training with the approved bearing split. "
            "Coverage is filename-only; no MAT content validation is claimed."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root directory recursively scanned for .mat recordings")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Required output directory; outputs are never placed by default in the repository")
    parser.add_argument("--reps", type=_parse_representations,
                        default=list(DEFAULT_REPRESENTATIONS),
                        help="Comma-separated representations (default: all five Stage 1 representations)")
    parser.add_argument("--seeds", type=_parse_seeds,
                        default=list(DEFAULT_SEEDS),
                        help="Comma-separated seeds (default: 42,1337,2026,7,99)")
    parser.add_argument("--max-recordings-per-pair", type=_positive_int, default=None,
                        help="Optional positive cap per (physical bearing, condition) pair; default uses all")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable training-partition inverse-frequency CrossEntropy weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight and record the plan without loading MAT data or training")
    return parser


def _jsonable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_file_list(relative_paths: Sequence[str]) -> str:
    # Newline-delimited UTF-8 is unambiguous here because path strings may not
    # contain newline characters in the filesystem inventory contract.
    return _sha256_bytes("\n".join(relative_paths).encode("utf-8"))


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for distribution in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _code_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        text=True,
    ).strip()


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _summary_text(manifest: Mapping[str, Any]) -> str:
    coverage = manifest.get("coverage") or {}
    memory = manifest.get("memory_guard") or {}
    lines = [
        "Stage 1 training run summary",
        f"status: {manifest.get('status')}",
        f"current_phase: {manifest.get('current_phase')}",
        f"run_id: {manifest.get('run_id')}",
        f"code_commit_sha: {manifest.get('code_commit_sha')}",
        f"data_root: {manifest.get('cli_arguments', {}).get('data_root')}",
        f"representations: {', '.join(manifest.get('representations', []))}",
        f"seeds: {', '.join(str(seed) for seed in manifest.get('seeds', []))}",
        f"class_weights_enabled: {manifest.get('class_weights_enabled')}",
        f"normalization_source: {manifest.get('normalization_source')}",
        f"training_batch_policy: {manifest.get('training_batch_policy')}",
        f"discovered_mat_files: {manifest.get('discovered_mat_file_count', 0)}",
        f"loaded_file_count: {manifest.get('loaded_file_count', 0)}",
        f"loaded_file_list_sha256: {manifest.get('loaded_file_list_sha256')}",
        f"selected_file_list_sha256: {manifest.get('selected_file_list_sha256')}",
        f"excluded_file_count: {manifest.get('excluded_file_count', 0)}",
        f"required_coverage_gaps: {len(coverage.get('missing_required', []))}",
        f"secondary_condition_gaps: {len(coverage.get('secondary_condition_gaps', []))}",
        f"completed_result_rows: {len(manifest.get('completed_runs', []))}",
        f"failed_combinations: {len(manifest.get('failed_runs', []))}",
        f"skipped_combinations: {len(manifest.get('skipped_runs', []))}",
        "",
        "Uncertainty intervals are descriptive cluster-bootstrap intervals on physical_bearing_uids.",
        "Filename coverage does not verify MAT signal content, channels, sampling rate, length, or windows.",
    ]
    if coverage.get("secondary_condition_gaps"):
        lines.extend(["", "non-fatal secondary-condition coverage gaps:"])
        lines.extend(
            f"- {gap.get('bearing_id')} @ {gap.get('condition_code')}"
            for gap in coverage["secondary_condition_gaps"]
        )
    if memory:
        lines.extend(
            [
                f"required_pair_candidate_count: {memory.get('required_pair_candidate_count', 0)}",
        f"optional_secondary_pair_candidate_count: {memory.get('optional_secondary_pair_candidate_count', 0)}",
        f"selected_file_count: {memory.get('selected_file_count', 0)}",
                f"excluded_outside_approved_pairs: {memory.get('excluded_outside_approved_pairs_count', 0)}",
                f"excluded_by_recording_cap: {memory.get('excluded_by_recording_cap_count', 0)}",
            ]
        )
    if manifest.get("partition_counts"):
        lines.extend(["", "loaded partition counts (windows / physical bearings; classes 0,1,2):"])
        for view, counts in manifest["partition_counts"].items():
            class_counts = counts["class_counts"]
            lines.append(
                f"- {view}: {counts['n_windows']} / {counts['n_bearings']}; "
                f"[{class_counts['0']}, {class_counts['1']}, {class_counts['2']}]"
            )
    if manifest.get("error"):
        lines.extend(["", "error:", str(manifest["error"])])
    if manifest.get("recording_exclusion_log"):
        lines.extend(["", "loader recording exclusion log:"])
        lines.extend(
            f"- {entry.get('source_file')} ({entry.get('reason')})"
            for entry in manifest["recording_exclusion_log"]
        )
    if manifest.get("completed_runs"):
        lines.extend(["", "completed evaluation rows (pooled-window F1; descriptive cluster-bootstrap CI):"])
        lines.extend(
            "- {representation} seed={seed} view={view}: Macro-F1={macro_f1:.4f}; "
            "F1[0,1,2]=[{f1_healthy:.4f}, {f1_outer:.4f}, {f1_inner:.4f}]; "
            "CI=[{ci_lo:.4f}, {ci_hi:.4f}]".format(**row)
            for row in manifest["completed_runs"]
        )
    if manifest.get("failed_runs"):
        lines.extend(["", "failed combinations:"])
        lines.extend(
            f"- {entry.get('representation')} seed={entry.get('seed')} view={entry.get('view')}: {entry.get('error')}"
            for entry in manifest["failed_runs"]
        )
    if manifest.get("skipped_runs"):
        lines.extend(["", "skipped combinations:"])
        lines.extend(
            f"- {entry.get('representation')} seed={entry.get('seed')} view={entry.get('view')}: {entry.get('reason')}"
            for entry in manifest["skipped_runs"]
        )
    return "\n".join(lines) + "\n"


def _persist_run_files(
    manifest: dict[str, Any], manifest_path: Path, summary_path: Path
) -> None:
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_text(summary_path, _summary_text(manifest))


def discover_mat_files(data_root: Path) -> list[Path]:
    """Return present .mat files recursively in deterministic relative-path order."""
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise Stage1RunError(f"--data-root must be an existing directory: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mat"),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def required_bearing_condition_pairs(split: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Return required train/validation and primary-test pairs from the split."""
    allocations = split["allocations"]
    conditions = split["conditions"]
    pairs: set[tuple[str, str]] = set()
    for allocation_name in ("train", "validation"):
        pairs.update(
            (bearing_id, conditions[allocation_name])
            for bearing_id in allocations[allocation_name]
        )
    for view_name in ("test_in_condition", "test_primary"):
        pairs.update(
            (bearing_id, conditions[view_name])
            for bearing_id in allocations["test"]
        )
    return pairs


def optional_secondary_bearing_condition_pairs(
    split: Mapping[str, Any],
) -> set[tuple[str, str]]:
    """Return approved optional secondary-view pairs without making them fatal."""
    condition = split["conditions"]["test_secondary"]
    return {
        (bearing_id, condition)
        for bearing_id in split["allocations"]["test"]
    }


def select_approved_recordings(
    paths: Sequence[Path],
    data_root: Path,
    split: Mapping[str, Any],
    max_recordings_per_pair: int | None = None,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    """Filter to approved pairs, then deterministically cap per pair if requested."""
    required_pairs = required_bearing_condition_pairs(split)
    optional_pairs = optional_secondary_bearing_condition_pairs(split)
    approved_pairs = required_pairs | optional_pairs
    grouped: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    excluded: list[dict[str, Any]] = []

    for path in paths:
        bearing_id, condition_code, recording_id = (
            ds_mod._parse_paderborn_recording_filename(path)
        )
        pair = (bearing_id, condition_code)
        relative_path = path.relative_to(data_root).as_posix()
        if pair not in approved_pairs:
            excluded.append(
                {
                    "path": relative_path,
                    "bearing_id": bearing_id,
                    "condition_code": condition_code,
                    "recording_id": recording_id,
                    "reason": "outside_stage1_approved_bearing_condition_pairs",
                }
            )
            continue
        grouped.setdefault(pair, []).append((recording_id, path))

    selected: list[Path] = []
    selection_groups: list[dict[str, Any]] = []
    excluded_by_cap = 0
    for bearing_id, condition_code in sorted(approved_pairs):
        pair_files = sorted(
            grouped.get((bearing_id, condition_code), []),
            key=lambda item: (item[0], item[1].relative_to(data_root).as_posix()),
        )
        if max_recordings_per_pair is None:
            kept = pair_files
            dropped = []
        else:
            kept = pair_files[:max_recordings_per_pair]
            dropped = pair_files[max_recordings_per_pair:]
        selected.extend(path for _, path in kept)
        excluded_by_cap += len(dropped)
        for recording_id, path in dropped:
            excluded.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "bearing_id": bearing_id,
                    "condition_code": condition_code,
                    "recording_id": recording_id,
                    "reason": "max_recordings_per_pair",
                }
            )
        selection_groups.append(
            {
                "bearing_id": bearing_id,
                "condition_code": condition_code,
                "coverage_requirement": (
                    "required"
                    if (bearing_id, condition_code) in required_pairs
                    else "optional_secondary"
                ),
                "candidate_recording_ids": [recording_id for recording_id, _ in pair_files],
                "selected_recording_ids": [recording_id for recording_id, _ in kept],
                "selected_files": [path.relative_to(data_root).as_posix() for _, path in kept],
                "excluded_recording_ids_by_cap": [recording_id for recording_id, _ in dropped],
            }
        )

    selected.sort(key=lambda path: path.relative_to(data_root).as_posix())
    required_candidate_count = sum(len(grouped.get(pair, [])) for pair in required_pairs)
    optional_candidate_count = sum(len(grouped.get(pair, [])) for pair in optional_pairs)
    excluded_outside_count = sum(
        1 for row in excluded
        if row["reason"] == "outside_stage1_approved_bearing_condition_pairs"
    )
    selection = {
        "max_recordings_per_pair": max_recordings_per_pair,
        "required_pair_count": len(required_pairs),
        "optional_secondary_pair_count": len(optional_pairs),
        "approved_pair_count": len(approved_pairs),
        "required_pair_candidate_count": required_candidate_count,
        "optional_secondary_pair_candidate_count": optional_candidate_count,
        "approved_pair_candidate_count": required_candidate_count + optional_candidate_count,
        "selected_file_count": len(selected),
        "excluded_outside_approved_pairs_count": excluded_outside_count,
        "excluded_by_recording_cap_count": excluded_by_cap,
        "selected_files": [path.relative_to(data_root).as_posix() for path in selected],
        "groups": selection_groups,
    }
    return selected, excluded, selection


def normalize_partitions(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test_views: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Fit source-train z-score constants on X_train only and apply them unchanged."""
    X_train = np.asarray(X_train)
    X_validation = np.asarray(X_validation)
    if X_train.ndim < 2 or X_train.shape[0] == 0:
        raise Stage1RunError("X_train must be a non-empty feature matrix")
    if X_validation.ndim != X_train.ndim or X_validation.shape[1:] != X_train.shape[1:]:
        raise Stage1RunError("X_validation feature shape does not match X_train")

    train_only_mask = np.ones(X_train.shape[0], dtype=bool)
    mean, std = ds_mod.compute_source_train_zscore(X_train, train_only_mask)
    X_train_z = ds_mod.apply_zscore(X_train, mean, std)
    X_validation_z = ds_mod.apply_zscore(X_validation, mean, std)
    X_tests_z: dict[str, np.ndarray] = {}
    for view, X_view in X_test_views.items():
        X_view = np.asarray(X_view)
        if X_view.ndim != X_train.ndim or X_view.shape[1:] != X_train.shape[1:]:
            raise Stage1RunError(f"{view} feature shape does not match X_train")
        X_tests_z[view] = ds_mod.apply_zscore(X_view, mean, std)
    return X_train_z, X_validation_z, X_tests_z, mean, std


def _labels_for_partition(
    labels: np.ndarray,
    partition_name: str,
    *,
    require_all_classes: bool,
) -> None:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise Stage1RunError(f"{partition_name} partition is empty or has invalid label shape")
    observed = set(labels.tolist())
    unexpected = observed - set(FIXED_LABELS)
    if unexpected:
        raise Stage1RunError(f"{partition_name} partition has unexpected labels {sorted(unexpected)}")
    if require_all_classes:
        missing = sorted(set(FIXED_LABELS) - observed)
        if missing:
            raise Stage1RunError(
                f"{partition_name} partition is missing required Stage 1 classes {missing}"
            )


def _assert_no_physical_bearing_leakage(
    physical_bearing_uids: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> None:
    train_bearings = set(physical_bearing_uids[masks["train"]].tolist())
    validation_bearings = set(physical_bearing_uids[masks["validation"]].tolist())
    train_validation = train_bearings | validation_bearings
    if train_bearings & validation_bearings:
        raise Stage1RunError(
            f"Physical bearing leakage between train and validation: "
            f"{sorted(train_bearings & validation_bearings)}"
        )
    for view in EVALUATION_VIEWS:
        test_bearings = set(physical_bearing_uids[masks[view]].tolist())
        overlap = train_validation & test_bearings
        if overlap:
            raise Stage1RunError(
                f"Physical bearing leakage between train/validation and {view}: "
                f"{sorted(overlap)}"
            )


def _partition_masks(ds: Any) -> dict[str, np.ndarray]:
    physical = np.asarray(ds.physical_bearing_uids, dtype=object)
    conditions = np.asarray(ds.condition_codes, dtype=object)
    if physical.ndim != 1 or conditions.ndim != 1 or physical.shape != conditions.shape:
        raise Stage1RunError(
            "loaded physical_bearing_uids and condition_codes must be aligned 1-D arrays"
        )
    masks = {
        view: np.asarray(
            split_mod.partition_mask(physical, conditions, view), dtype=bool
        )
        for view in ("train", "validation", *EVALUATION_VIEWS)
    }
    if any(mask.shape != physical.shape for mask in masks.values()):
        raise Stage1RunError("partition_mask returned a mask with an invalid shape")
    _assert_no_physical_bearing_leakage(physical, masks)
    labels = np.asarray(ds.labels)
    if labels.ndim != 1 or labels.shape != physical.shape:
        raise Stage1RunError("loaded labels do not align with physical bearing IDs")
    for name in ("train", "validation", "test_in_condition", "test_primary"):
        _labels_for_partition(
            labels[masks[name]], name, require_all_classes=name in ("train", "validation")
        )
    if masks["test_secondary"].any():
        _labels_for_partition(
            labels[masks["test_secondary"]], "test_secondary", require_all_classes=False
        )

    split = split_mod.load_split()
    expected_bearings = {
        "train": set(split["allocations"]["train"]),
        "validation": set(split["allocations"]["validation"]),
        "test_in_condition": set(split["allocations"]["test"]),
        "test_primary": set(split["allocations"]["test"]),
    }
    for view, expected in expected_bearings.items():
        observed = set(physical[masks[view]].tolist())
        missing = sorted(expected - observed)
        if missing:
            raise Stage1RunError(
                f"{view} partition is missing allocated physical bearings with no usable windows: {missing}"
            )
    return masks


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]


def _new_manifest(args: argparse.Namespace, run_id: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    split_path = split_mod._SPLIT_PATH
    class_map_path = ds_mod._paderborn_class_map_path()
    experiment_config_path = _EXPERIMENT_ROOT / "config" / "experiment_config.json"
    try:
        commit_sha: str | None = _code_commit_sha()
    except Exception:
        commit_sha = None
    try:
        split_hash: str | None = _sha256_file(split_path)
    except OSError:
        split_hash = None
    try:
        class_map_hash: str | None = _sha256_file(class_map_path)
    except OSError:
        class_map_hash = None
    try:
        experiment_config_hash: str | None = _sha256_file(experiment_config_path)
    except OSError:
        experiment_config_hash = None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "RUNNING",
        "current_phase": "INITIAL_PREFLIGHT",
        "started_at": _utc_now(),
        "completed_at": None,
        "code_commit_sha": commit_sha,
        "split_file": str(split_path),
        "split_file_sha256": split_hash,
        "class_map_file": str(class_map_path),
        "class_map_sha256": class_map_hash,
        "experiment_config_file": str(experiment_config_path),
        "experiment_config_sha256": experiment_config_hash,
        "data_root": str(args.data_root.resolve()),
        "cli_arguments": _jsonable_arguments(args),
        "seeds": list(args.seeds),
        "representations": list(args.reps),
        "class_weights_enabled": not args.no_class_weights,
        "normalization_source": "X_train only; frozen mean/std applied to validation and each test view",
        "training_batch_policy": (
            "drop_last_train=True with configured batch_size. Both model families use BatchNorm1d; "
            "the MLP BatchNorm1d path cannot train on a singleton batch. The trainer drops the "
            "entire final incomplete batch, and requires at least one complete batch."
        ),
        "uncertainty_note": "Cluster-bootstrap intervals are descriptive only and resample physical_bearing_uids.",
        "coverage_evidence_note": (
            "Filename coverage verifies names and required bearing-condition pairs only; "
            "it does not verify MAT signal content, channels, sampling rate, signal length, or usable windows."
        ),
        "library_versions": _library_versions(),
        "output_files": {key: str(value) for key, value in paths.items()},
        "discovered_mat_file_count": 0,
        "selected_file_count": 0,
        "selected_file_list": [],
        "selected_file_list_sha256": None,
        "loader_input_files": [],
        "loader_input_file_count": 0,
        "loaded_files": [],
        "loaded_file_count": 0,
        "loaded_file_list_sha256": None,
        "excluded_file_count": 0,
        "excluded_recordings": [],
        "recording_exclusion_log": [],
        "coverage": None,
        "coverage_preflight": None,
        "memory_guard": None,
        "partition_counts": None,
        "planned_runs": [],
        "completed_runs": [],
        "failed_runs": [],
        "skipped_runs": [],
        "error": None,
    }


def _initialise_output_files(out_dir: Path) -> tuple[str, dict[str, Path], Any, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    paths = {
        "results_csv": out_dir / f"stage1_results_{run_id}.csv",
        "run_manifest": out_dir / f"stage1_manifest_{run_id}.json",
        "summary": out_dir / f"stage1_summary_{run_id}.txt",
    }
    results_stream = paths["results_csv"].open("x", newline="", encoding="utf-8")
    writer = csv.DictWriter(results_stream, fieldnames=RESULT_COLUMNS, extrasaction="raise")
    writer.writeheader()
    results_stream.flush()
    os.fsync(results_stream.fileno())
    return run_id, paths, results_stream, writer


def _timestamp_for_row() -> str:
    return _utc_now()


def run_stage1(args: argparse.Namespace) -> int:
    """Execute or dry-run the Stage 1 protocol and persist an auditable bundle."""
    run_id, output_paths, results_stream, csv_writer = _initialise_output_files(args.out_dir)
    manifest = _new_manifest(args, run_id, output_paths)
    phase = "INITIAL_PREFLIGHT"
    secondary_inventory_present = False

    def persist() -> None:
        _persist_run_files(
            manifest,
            output_paths["run_manifest"],
            output_paths["summary"],
        )

    try:
        persist()
        split = split_mod.load_split()
        all_paths = discover_mat_files(args.data_root)
        manifest["discovered_mat_file_count"] = len(all_paths)

        # Run coverage against the exact deterministic discovery inventory,
        # before memory filtering or any --max-recordings-per-pair selection.
        phase = "COVERAGE_PREFLIGHT"
        manifest["current_phase"] = phase
        persist()
        coverage_report = build_coverage_report(all_paths)
        manifest["coverage_preflight"] = coverage_report
        try:
            coverage = split_mod.check_coverage(all_paths, strict=False)
        except Exception as coverage_exc:
            manifest["coverage"] = {
                "input_file_count": len(all_paths),
                "error": f"{type(coverage_exc).__name__}: {coverage_exc}",
            }
            raise Stage1RunError(
                f"authoritative filename coverage check failed: {coverage_exc}"
            ) from coverage_exc
        manifest["coverage"] = coverage
        persist()
        if coverage["missing_required"]:
            raise Stage1RunError(
                "required Stage 1 coverage is incomplete: "
                f"{len(coverage['missing_required'])} required bearing-condition pairs are missing"
            )
        if coverage_report["status"] != "PASS":
            error_codes = [entry.get("code") for entry in coverage_report.get("errors", [])]
            raise Stage1RunError(
                f"filename/identity preflight failed with errors: {error_codes}"
            )

        phase = "MEMORY_GUARD"
        manifest["current_phase"] = phase
        persist()
        selected_paths, excluded, selection = select_approved_recordings(
            all_paths,
            args.data_root,
            split,
            max_recordings_per_pair=args.max_recordings_per_pair,
        )
        selected_relative = [path.relative_to(args.data_root).as_posix() for path in selected_paths]
        manifest["memory_guard"] = selection
        manifest["selected_file_count"] = len(selected_relative)
        manifest["selected_file_list"] = selected_relative
        manifest["selected_file_list_sha256"] = _sha256_file_list(selected_relative)
        manifest["excluded_recordings"] = excluded
        manifest["excluded_file_count"] = len(excluded)
        manifest["secondary_condition_gaps"] = coverage["secondary_condition_gaps"]

        if not selected_paths:
            raise Stage1RunError("memory guard selected no recordings for the approved Stage 1 pairs")

        secondary_condition = split["conditions"]["test_secondary"]
        secondary_pairs = {
            (bearing, secondary_condition)
            for bearing in split["allocations"]["test"]
        }
        secondary_inventory_present = any(
            ds_mod._parse_paderborn_recording_filename(path)[:2] in secondary_pairs
            for path in selected_paths
        )
        manifest["secondary_inventory_present"] = secondary_inventory_present
        manifest["planned_runs"] = [
            {"representation": rep, "seed": seed, "view": view}
            for rep in args.reps
            for seed in args.seeds
            for view in EVALUATION_VIEWS
        ]
        if not secondary_inventory_present and args.dry_run:
            manifest["skipped_runs"].extend(
                {
                    "representation": rep,
                    "seed": seed,
                    "view": "test_secondary",
                    "reason": "optional test_secondary has no recording files in the discovered inventory",
                }
                for rep in args.reps
                for seed in args.seeds
            )
        persist()

        if args.dry_run:
            manifest["status"] = "DRY_RUN"
            manifest["current_phase"] = None
            manifest["completed_at"] = _utc_now()
            persist()
            return 0

        phase = "MAT_LOADING_AND_FEATURE_EXTRACTION"
        manifest["current_phase"] = phase
        persist()
        ds = ds_mod.load_real_paderborn_dataset(
            selected_paths,
            exclude_combined=True,
        )
        exclusion_log = list(getattr(ds, "recording_exclusion_log", []))
        manifest["recording_exclusion_log"] = exclusion_log
        manifest["loader_input_files"] = selected_relative
        manifest["loader_input_file_count"] = len(selected_relative)
        loader_excluded_names = {
            Path(str(entry.get("source_file", ""))).name
            for entry in exclusion_log
            if entry.get("source_file")
        }
        loaded_relative = [
            path.relative_to(args.data_root).as_posix()
            for path in selected_paths
            if path.name not in loader_excluded_names
        ]
        manifest["loaded_files"] = loaded_relative
        manifest["loaded_file_count"] = len(loaded_relative)
        manifest["loaded_file_list_sha256"] = _sha256_file_list(loaded_relative)
        manifest["excluded_file_count"] = len(excluded) + len(exclusion_log)
        manifest["excluded_recordings"].extend(
            {
                "path": entry.get("source_file"),
                "bearing_id": entry.get("bearing_id"),
                "condition_code": entry.get("condition_code"),
                "reason": entry.get("reason"),
            }
            for entry in exclusion_log
        )
        persist()

        phase = "PARTITION_VALIDATION"
        manifest["current_phase"] = phase
        persist()
        masks = _partition_masks(ds)
        dataset_physical = np.asarray(ds.physical_bearing_uids, dtype=object)
        dataset_labels = np.asarray(ds.labels)
        manifest["partition_counts"] = {
            view: {
                "n_windows": int(masks[view].sum()),
                "n_bearings": int(len(set(dataset_physical[masks[view]].tolist()))),
                "class_counts": {
                    str(label): int(np.count_nonzero(dataset_labels[masks[view]] == label))
                    for label in FIXED_LABELS
                },
            }
            for view in ("train", "validation", *EVALUATION_VIEWS)
        }
        persist()
        test_views = {
            view: np.flatnonzero(masks[view])
            for view in EVALUATION_VIEWS
            if masks[view].any()
        }
        if "test_secondary" not in test_views:
            skip_reason = (
                "optional test_secondary recording files produced no usable windows"
                if secondary_inventory_present
                else "optional test_secondary has no recording files in the discovered inventory"
            )
            already_skipped = {
                (entry.get("representation"), entry.get("seed"), entry.get("view"))
                for entry in manifest["skipped_runs"]
            }
            manifest["skipped_runs"].extend(
                {
                    "representation": rep,
                    "seed": seed,
                    "view": "test_secondary",
                    "reason": skip_reason,
                }
                for rep in args.reps
                for seed in args.seeds
                if (rep, seed, "test_secondary") not in already_skipped
            )
        if "test_in_condition" not in test_views or "test_primary" not in test_views:
            raise Stage1RunError("a required test view has no usable windows after loading")

        # The trainer requires a complete batch when drop_last_train=True.
        # Its MLP uses BatchNorm1d over (batch, features), which fails for a
        # singleton batch; the configured 32-row batching and explicit drop
        # policy prevent that architecture-specific failure.
        phase = "TRAINING_AND_EVALUATION"
        manifest["current_phase"] = phase
        persist()
        train_api = _load_training_api()
        evaluation_api = _load_evaluation_api()
        with (_EXPERIMENT_ROOT / "config" / "experiment_config.json").open(
            "r", encoding="utf-8"
        ) as stream:
            experiment_config = json.load(stream)
        training_config = experiment_config["training"]
        statistical_config = experiment_config["statistical_evaluation"]
        manifest["effective_training_config"] = training_config
        manifest["effective_statistical_evaluation_config"] = statistical_config
        persist()

        for representation in args.reps:
            features = np.asarray(ds_mod.get_representation(ds, representation))
            if features.shape[0] != ds.labels.shape[0]:
                raise Stage1RunError(
                    f"{representation} feature rows do not align with labels"
                )
            train_idx = np.flatnonzero(masks["train"])
            validation_idx = np.flatnonzero(masks["validation"])
            X_train = features[train_idx]
            y_train = np.asarray(ds.labels)[train_idx]
            X_validation = features[validation_idx]
            y_validation = np.asarray(ds.labels)[validation_idx]
            X_tests = {view: features[index] for view, index in test_views.items()}
            X_train_z, X_validation_z, X_tests_z, mean, std = normalize_partitions(
                X_train, X_validation, X_tests
            )
            manifest.setdefault("normalization", {})[representation] = {
                "source_partition": "train",
                "mean_shape": list(mean.shape),
                "std_shape": list(std.shape),
                "mean_sha256": _sha256_bytes(np.asarray(mean, dtype=np.float32).tobytes()),
                "std_sha256": _sha256_bytes(np.asarray(std, dtype=np.float32).tobytes()),
            }
            persist()

            for seed in args.seeds:
                combo_id = {"representation": representation, "seed": seed}
                try:
                    class_weights = None
                    if not args.no_class_weights:
                        class_weights = train_api.compute_class_weights(
                            y_train, num_classes=3
                        )
                    model = train_api.train_one_fold(
                        X_train_z,
                        y_train,
                        X_validation_z,
                        y_validation,
                        representation=representation,
                        seed=seed,
                        num_classes=3,
                        lr=float(training_config["learning_rate"]),
                        weight_decay=float(training_config["weight_decay"]),
                        batch_size=int(training_config["batch_size"]),
                        max_epochs=int(training_config["max_epochs"]),
                        early_stopping_patience=int(training_config["early_stopping_patience"]),
                        use_supplied_validation=True,
                        class_weights=class_weights,
                        drop_last_train=True,
                    )
                except Exception as exc:
                    for view in test_views:
                        manifest["failed_runs"].append(
                            {
                                **combo_id,
                                "view": view,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    persist()
                    continue

                for view, indices in test_views.items():
                    view_labels = np.asarray(ds.labels)[indices]
                    view_physical_uids = np.asarray(ds.physical_bearing_uids, dtype=object)[indices]
                    try:
                        prediction = np.asarray(
                            train_api.predict(model, X_tests_z[view]), dtype=np.int64
                        )
                        if prediction.shape != view_labels.shape:
                            raise Stage1RunError(
                                f"{view} prediction shape {prediction.shape} does not match labels {view_labels.shape}"
                            )
                        macro = evaluation_api.macro_f1(
                            view_labels,
                            prediction,
                            labels=list(FIXED_LABELS),
                        )
                        per_class = evaluation_api.per_class_f1(
                            view_labels,
                            prediction,
                            labels=FIXED_LABELS,
                        )
                        _, ci_lo, ci_hi, _ = evaluation_api.cluster_bootstrap_ci(
                            view_labels,
                            prediction,
                            view_physical_uids,
                            n_iterations=int(statistical_config["cluster_bootstrap_iterations"]),
                            ci=float(statistical_config["confidence_interval_level"]),
                            seed=seed,
                            labels=list(FIXED_LABELS),
                        )
                        row = {
                            "representation": representation,
                            "seed": seed,
                            "view": view,
                            "condition": split["conditions"][view],
                            "n_windows": int(view_labels.size),
                            "n_bearings": int(len(set(view_physical_uids.tolist()))),
                            "macro_f1": float(macro),
                            "f1_healthy": float(per_class[0]),
                            "f1_outer": float(per_class[1]),
                            "f1_inner": float(per_class[2]),
                            "ci_lo": float(ci_lo),
                            "ci_hi": float(ci_hi),
                            "class_weights_used": not args.no_class_weights,
                            "timestamp": _timestamp_for_row(),
                        }
                        csv_writer.writerow(row)
                        results_stream.flush()
                        os.fsync(results_stream.fileno())
                        manifest["completed_runs"].append(row)
                    except Exception as exc:
                        manifest["failed_runs"].append(
                            {
                                **combo_id,
                                "view": view,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    persist()

        planned_keys = {
            (entry["representation"], entry["seed"], entry["view"])
            for entry in manifest["planned_runs"]
        }
        accounted_keys = {
            (entry["representation"], entry["seed"], entry["view"])
            for key in ("completed_runs", "failed_runs", "skipped_runs")
            for entry in manifest[key]
        }
        unaccounted = planned_keys - accounted_keys
        if unaccounted:
            raise Stage1RunError(
                f"run ended with unaccounted planned combinations: {sorted(unaccounted)}"
            )
        manifest["completed_at"] = _utc_now()
        manifest["current_phase"] = None
        manifest["status"] = "PARTIAL_FAILURE" if manifest["failed_runs"] else "COMPLETE"
        persist()
        return 1 if manifest["failed_runs"] else 0

    except Exception as exc:
        if phase in ("INITIAL_PREFLIGHT", "COVERAGE_PREFLIGHT", "MEMORY_GUARD"):
            manifest["status"] = "FAILED_PREFLIGHT"
        elif phase == "MAT_LOADING_AND_FEATURE_EXTRACTION":
            manifest["status"] = "FAILED_LOADING"
        elif phase == "PARTITION_VALIDATION":
            manifest["status"] = "FAILED_PARTITION_VALIDATION"
        else:
            manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "phase": phase}
        manifest["current_phase"] = None
        manifest["completed_at"] = _utc_now()
        if phase in (
            "MAT_LOADING_AND_FEATURE_EXTRACTION",
            "PARTITION_VALIDATION",
            "TRAINING_AND_EVALUATION",
        ):
            if not secondary_inventory_present:
                existing_skips = {
                    (entry.get("representation"), entry.get("seed"), entry.get("view"))
                    for entry in manifest["skipped_runs"]
                }
                for rep in args.reps:
                    for seed in args.seeds:
                        key = (rep, seed, "test_secondary")
                        if key not in existing_skips:
                            manifest["skipped_runs"].append(
                                {
                                    "representation": rep,
                                    "seed": seed,
                                    "view": "test_secondary",
                                    "reason": "optional test_secondary has no recording files in the discovered inventory",
                                }
                            )
            represented = {
                (entry.get("representation"), entry.get("seed"), entry.get("view"))
                for key in ("completed_runs", "failed_runs", "skipped_runs")
                for entry in manifest.get(key, [])
            }
            for planned in manifest.get("planned_runs", []):
                key = (planned.get("representation"), planned.get("seed"), planned.get("view"))
                if key not in represented:
                    manifest["failed_runs"].append(
                        {
                            **planned,
                            "error": (
                                f"Run aborted during {phase}: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
        try:
            persist()
        except Exception as persist_exc:
            print(f"Failed to persist Stage 1 failure report: {persist_exc}", file=sys.stderr)
        print(f"Stage 1 run failed during {phase}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        results_stream.flush()
        results_stream.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_argparser().parse_args(argv)
    return run_stage1(args)


if __name__ == "__main__":  # pragma: no cover - exercised by command line
    raise SystemExit(main())
