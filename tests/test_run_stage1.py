"""Synthetic-only tests for the Stage 1 runner.

All inventories use empty .mat filename fixtures, and all training/evaluation
paths use mocked loaders and models. These tests never open real Paderborn MAT
payloads, use a GPU, or train a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPOSITORY_ROOT), str(EXPERIMENT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.datasets as D
    import src.run_stage1 as RUN
    import src.stage1_split as S
except Exception:  # pragma: no cover - standalone source-tree fallback
    import datasets as D
    import run_stage1 as RUN
    import stage1_split as S


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _make_inventory(
    root: Path,
    *,
    include_secondary: bool = True,
    extra_files: bool = False,
    special_pair: tuple[str, str] | None = None,
) -> list[Path]:
    """Create empty filename fixtures based on check_coverage's pair list."""
    split = S.load_split()
    required = S.check_coverage([])["required_pairs"]
    paths: list[Path] = []
    for index, pair in enumerate(required):
        bearing = pair["bearing_id"]
        condition = pair["condition_code"]
        ids = ("2", "10") if special_pair == (bearing, condition) else ("1",)
        for recording_id in ids:
            path = root / f"batch-{index % 3}" / (
                f"{condition}_{bearing}_{recording_id}.mat"
            )
            paths.append(_touch(path))

    if include_secondary:
        condition = split["conditions"]["test_secondary"]
        for index, bearing in enumerate(split["allocations"]["test"]):
            path = root / f"secondary-{index % 2}" / f"{condition}_{bearing}_1.mat"
            paths.append(_touch(path))

    if extra_files:
        # A valid, class-mapped but non-Stage-1 condition: preflight sees it,
        # while the memory guard must exclude it from the loader input.
        paths.append(_touch(root / "extra" / "N15_M01_F10_K002_9.mat"))
        _touch(root / "ignore.txt")
    return paths


def _make_fake_dataset(*, include_secondary: bool = True, secondary_single_class: bool = False):
    split = S.load_split()
    class_map = D._load_paderborn_class_map()
    physical: list[str] = []
    conditions: list[str] = []
    labels: list[int] = []
    recording_uids: list[str] = []

    for allocation_name in ("train", "validation"):
        condition = split["conditions"][allocation_name]
        for bearing in split["allocations"][allocation_name]:
            physical.append(bearing)
            conditions.append(condition)
            labels.append(int(class_map[bearing]["label"]))
            recording_uids.append(f"recording::{condition}::{bearing}::1")

    test_views = ("test_in_condition", "test_primary")
    if include_secondary:
        test_views += ("test_secondary",)
    for view in test_views:
        condition = split["conditions"][view]
        for bearing in split["allocations"]["test"]:
            physical.append(bearing)
            conditions.append(condition)
            label = int(class_map[bearing]["label"])
            if view == "test_secondary" and secondary_single_class:
                label = 0
            labels.append(label)
            recording_uids.append(f"recording::{condition}::{bearing}::1")

    n_rows = len(labels)
    row = np.arange(n_rows, dtype=np.float32)
    features = np.column_stack(
        [row, (row * row) % 23, (row % 7) + 2.0, 3.0 * row + 1.0]
    ).astype(np.float32)
    return SimpleNamespace(
        name="PU-synthetic",
        labels=np.asarray(labels, dtype=np.int64),
        uids=np.asarray(recording_uids, dtype=object),
        physical_bearing_uids=np.asarray(physical, dtype=object),
        condition_codes=np.asarray(conditions, dtype=object),
        r1_raw=features.copy(),
        r1_env=(features + 10.0).copy(),
        r2_raw=(features + 20.0).copy(),
        r2_env=(features + 30.0).copy(),
        r3_order=(features + 40.0).copy(),
        recording_exclusion_log=[],
    )


def _install_fakes(monkeypatch, dataset):
    traces = {
        "loader_calls": [],
        "weight_labels": [],
        "train_calls": [],
        "macro_labels": [],
        "per_class_labels": [],
        "bootstrap_calls": [],
    }

    def load_real(paths, *, exclude_combined):
        traces["loader_calls"].append(
            {"paths": [Path(path) for path in paths], "exclude_combined": exclude_combined}
        )
        return dataset

    def compute_class_weights(labels, num_classes=3):
        traces["weight_labels"].append(np.asarray(labels).copy())
        return np.ones(num_classes, dtype=np.float32)

    def train_one_fold(X_train, y_train, X_val, y_val, **kwargs):
        traces["train_calls"].append(
            {
                "X_train": np.asarray(X_train).copy(),
                "y_train": np.asarray(y_train).copy(),
                "X_val": np.asarray(X_val).copy(),
                "y_val": np.asarray(y_val).copy(),
                "kwargs": dict(kwargs),
            }
        )
        return object()

    def predict(_model, X):
        return np.zeros(np.asarray(X).shape[0], dtype=np.int64)

    def macro_f1(y_true, y_pred, *, labels=None):
        traces["macro_labels"].append(tuple(labels) if labels is not None else None)
        return 0.5

    def per_class_f1(y_true, y_pred, *, labels=(0, 1, 2)):
        traces["per_class_labels"].append(tuple(labels))
        return {0: 0.4, 1: 0.5, 2: 0.6}

    def cluster_bootstrap_ci(y_true, y_pred, uids, **kwargs):
        traces["bootstrap_calls"].append(
            {"uids": np.asarray(uids, dtype=object).copy(), "kwargs": dict(kwargs)}
        )
        return (0.5, 0.35, 0.65, 0.08)

    train_api = SimpleNamespace(
        compute_class_weights=compute_class_weights,
        train_one_fold=train_one_fold,
        predict=predict,
    )
    evaluation_api = SimpleNamespace(
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        cluster_bootstrap_ci=cluster_bootstrap_ci,
    )
    monkeypatch.setattr(RUN.ds_mod, "load_real_paderborn_dataset", load_real)
    monkeypatch.setattr(RUN, "_load_training_api", lambda: train_api)
    monkeypatch.setattr(RUN, "_load_evaluation_api", lambda: evaluation_api)
    return traces


def _run_args(root: Path, out: Path, *, reps="R1_Raw", seeds="42", extra=()):
    return [
        "--data-root", str(root),
        "--out-dir", str(out),
        "--reps", reps,
        "--seeds", seeds,
        *extra,
    ]


def _read_outputs(out: Path):
    manifest_path = next(out.glob("stage1_manifest_*.json"))
    results_path = next(out.glob("stage1_results_*.csv"))
    summary_path = next(out.glob("stage1_summary_*.txt"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with results_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = summary_path.read_text(encoding="utf-8")
    return manifest, rows, summary


def test_dry_run_scans_exact_inventory_and_never_loads_or_trains(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True, extra_files=True)
    out = tmp_path / "out"
    captured = {}
    original_check = RUN.split_mod.check_coverage

    def check_spy(paths, *, strict=False):
        captured["paths"] = list(paths)
        captured["strict"] = strict
        return original_check(paths, strict=strict)

    monkeypatch.setattr(RUN.split_mod, "check_coverage", check_spy)

    def forbidden(operation):
        def fail_if_called(*args, **kwargs):
            pytest.fail(f"dry-run must not perform {operation}")

        return fail_if_called

    monkeypatch.setattr(
        RUN.ds_mod,
        "load_real_paderborn_dataset",
        forbidden("MAT loading"),
    )
    monkeypatch.setattr(
        RUN.ds_mod,
        "extract_representations",
        forbidden("feature extraction"),
    )
    # Training/model construction/prediction are reachable only through the
    # lazy training API; forbidding that boundary proves dry-run never enters it.
    monkeypatch.setattr(
        RUN,
        "_load_training_api",
        forbidden("model construction, training, or prediction"),
    )
    monkeypatch.setattr(
        RUN,
        "_load_evaluation_api",
        forbidden("evaluation or prediction"),
    )

    expected = RUN.discover_mat_files(root)
    exit_code = RUN.main(_run_args(root, out, reps="R1_Raw", seeds="42", extra=("--dry-run",)))

    assert exit_code == 0
    assert captured["paths"] == expected
    assert captured["strict"] is False
    manifest, rows, summary = _read_outputs(out)
    assert manifest["status"] == "DRY_RUN"
    assert manifest["loaded_file_count"] == 0
    assert manifest["selected_file_count"] == 51
    assert manifest["memory_guard"]["required_pair_count"] == 40
    assert manifest["memory_guard"]["optional_secondary_pair_count"] == 11
    assert manifest["coverage"]["input_file_count"] == len(expected)
    assert rows == []
    assert manifest["coverage_evidence_note"].startswith("Filename coverage")
    effective = manifest["effective_preprocessing"]
    assert effective["record_kind"] == (
        "source-grounded descriptive record; not a processing configuration"
    )
    assert "not a complete effective processing contract" in effective[
        "legacy_declared_configuration"
    ]["status"]
    assert effective["legacy_declared_configuration"]["preprocessing"]["detrend"] == "linear"
    implementation = effective["effective_stage1_implementation"]
    assert implementation["sampling"] == {
        "nominal_input_sampling_rate_hz": 64000,
        "output_sampling_rate_hz": 12000,
        "polyphase_ratio_up": 3,
        "polyphase_ratio_down": 16,
        "output_length_convention": "ceil(n_input_samples * 3 / 16)",
    }
    assert implementation["classification_windowing"]["window_length_samples"] == 2048
    assert implementation["classification_windowing"]["hop_length_samples"] == 1024
    assert implementation["mean_subtraction_and_detrending"]["linear_detrending_applied"] is False
    assert "before bandpass filtering" in implementation["R1_Raw_source"]["array"]
    assert "not sig_bp" in implementation["R1_Raw_source"]["distinction"]
    assert implementation["bandpass_and_envelope"]["envelope"].startswith("Hilbert magnitude")
    speed = implementation["speed_estimation_and_fallback"]
    assert speed["tracker_returns_condition_nominal_when"]
    assert "<=0 or >200 Hz" in speed["angular_resampler_fallback"]
    assert "does not drive" in speed["measured_speed_role"]
    assert implementation["angular_resampling"]["target_samples"] == 128
    assert implementation["angular_resampling"]["samples_per_revolution"] == 64
    assert implementation["angular_resampling"]["nominal_revolutions"] == 2
    assert implementation["R3_Order_spectrum"]["natural_rfft_bins"] == 65
    assert implementation["R3_Order_spectrum"]["output_feature_bins"] == 1024
    limitations = effective["unresolved_scientific_validation"]
    assert "not established" in limitations["angular_antialiasing"]
    assert "not the legacy-configured linear detrending" in limitations["linear_detrending"]
    assert "65 natural bins" in limitations["R3_bin_contract"]
    assert "not measured" in limitations["embedded_resource_compliance"]
    assert "not proof" in limitations["diagnostic_accuracy"]
    assert "does not resolve" in limitations["metadata_scope"]
    assert "filename-only" in RUN._make_argparser().description
    assert "does not verify MAT" in summary


def test_required_coverage_gap_aborts_before_loader_and_reports_failure(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    paths = _make_inventory(root, include_secondary=False)
    paths[0].unlink()
    out = tmp_path / "out"

    def forbidden(*args, **kwargs):
        pytest.fail("required coverage failure must abort before loading/training")

    monkeypatch.setattr(RUN.ds_mod, "load_real_paderborn_dataset", forbidden)
    monkeypatch.setattr(RUN, "_load_training_api", forbidden)

    exit_code = RUN.main(_run_args(root, out, extra=("--dry-run",)))

    assert exit_code == 1
    manifest, rows, _ = _read_outputs(out)
    assert manifest["status"] == "FAILED_PREFLIGHT"
    assert manifest["coverage"]["missing_required"]
    assert manifest["error"]["phase"] == "COVERAGE_PREFLIGHT"
    assert rows == []
    assert "required Stage 1 coverage is incomplete" in capsys.readouterr().err


def test_secondary_coverage_gap_is_nonfatal_and_reported(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=False)
    out = tmp_path / "out"

    exit_code = RUN.main(_run_args(root, out, extra=("--dry-run",)))

    assert exit_code == 0
    manifest, rows, _ = _read_outputs(out)
    assert manifest["status"] == "DRY_RUN"
    assert manifest["coverage"]["missing_required"] == []
    assert len(manifest["coverage"]["secondary_condition_gaps"]) == 11
    assert rows == []


def test_memory_guard_filters_outside_pairs_before_loader(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True, extra_files=True)
    out = tmp_path / "out"
    dataset = _make_fake_dataset()
    traces = _install_fakes(monkeypatch, dataset)

    exit_code = RUN.main(_run_args(root, out))

    assert exit_code == 0
    assert len(traces["loader_calls"]) == 1
    call = traces["loader_calls"][0]
    assert call["exclude_combined"] is True
    assert len(call["paths"]) == 51
    assert all(path.name != "N15_M01_F10_K002_9.mat" for path in call["paths"])
    manifest, _, _ = _read_outputs(out)
    assert manifest["memory_guard"]["excluded_outside_approved_pairs_count"] == 1
    assert manifest["excluded_file_count"] == 1
    assert manifest["loaded_file_count"] == 51
    assert manifest["loaded_file_list_sha256"] == manifest["selected_file_list_sha256"]


def test_recording_cap_is_deterministic_and_manifest_lists_exact_selection(tmp_path, monkeypatch):
    root = tmp_path / "data"
    split = S.load_split()
    first_pair = S.check_coverage([])["required_pairs"][0]
    pair = (first_pair["bearing_id"], first_pair["condition_code"])
    _make_inventory(root, include_secondary=False, special_pair=pair)
    out = tmp_path / "out"

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not load")

    monkeypatch.setattr(RUN.ds_mod, "load_real_paderborn_dataset", forbidden)
    code = RUN.main(
        _run_args(root, out, extra=("--max-recordings-per-pair", "1", "--dry-run"))
    )

    assert code == 0
    manifest, _, _ = _read_outputs(out)
    group = next(
        row for row in manifest["memory_guard"]["groups"]
        if (row["bearing_id"], row["condition_code"]) == pair
    )
    assert group["candidate_recording_ids"] == ["10", "2"]
    assert group["selected_recording_ids"] == ["10"]
    assert manifest["memory_guard"]["excluded_by_recording_cap_count"] == 1
    assert manifest["selected_file_count"] == 40
    selected_bytes = "\n".join(manifest["selected_file_list"]).encode()
    assert manifest["selected_file_list_sha256"] == hashlib.sha256(selected_bytes).hexdigest()
    assert manifest["cli_arguments"]["max_recordings_per_pair"] == 1


def test_train_uses_supplied_validation_and_training_labels_only_for_weights(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True)
    out = tmp_path / "out"
    dataset = _make_fake_dataset()
    traces = _install_fakes(monkeypatch, dataset)

    code = RUN.main(_run_args(root, out))

    assert code == 0
    assert len(traces["train_calls"]) == 1
    train_call = traces["train_calls"][0]
    kwargs = train_call["kwargs"]
    assert kwargs["use_supplied_validation"] is True
    assert kwargs["drop_last_train"] is True
    assert kwargs["class_weights"] is not None
    assert kwargs["batch_size"] == 32
    assert kwargs["max_epochs"] == 60

    masks = {
        view: S.partition_mask(dataset.physical_bearing_uids, dataset.condition_codes, view)
        for view in ("train", "validation")
    }
    np.testing.assert_array_equal(train_call["y_train"], dataset.labels[masks["train"]])
    np.testing.assert_array_equal(train_call["y_val"], dataset.labels[masks["validation"]])
    np.testing.assert_array_equal(traces["weight_labels"][0], dataset.labels[masks["train"]])
    assert not np.array_equal(traces["weight_labels"][0], dataset.labels)


def test_class_weights_can_be_disabled(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=False)
    out = tmp_path / "out"
    dataset = _make_fake_dataset(include_secondary=False)
    traces = _install_fakes(monkeypatch, dataset)
    original_training_api = RUN._load_training_api

    def disabled_weights(*args, **kwargs):
        api = original_training_api()
        api.compute_class_weights = lambda *a, **k: pytest.fail("weights were disabled")
        return api

    monkeypatch.setattr(RUN, "_load_training_api", disabled_weights)
    code = RUN.main(_run_args(root, out, extra=("--no-class-weights",)))

    assert code == 0
    assert traces["weight_labels"] == []
    assert traces["train_calls"][0]["kwargs"]["class_weights"] is None
    manifest, rows, _ = _read_outputs(out)
    assert manifest["class_weights_enabled"] is False
    assert all(row["class_weights_used"] == "False" for row in rows)


def test_normalization_uses_train_only_constants_for_validation_and_all_tests():
    X_train = np.asarray([[1.0, 10.0], [3.0, 20.0]], dtype=np.float32)
    X_validation = np.asarray([[100.0, 200.0]], dtype=np.float32)
    X_test = np.asarray([[-100.0, -200.0]], dtype=np.float32)

    X_train_z, X_validation_z, tests_z, mean, std = RUN.normalize_partitions(
        X_train,
        X_validation,
        {"test_primary": X_test, "test_secondary": X_test + 1},
    )

    np.testing.assert_allclose(mean, [2.0, 15.0])
    np.testing.assert_allclose(std, [1.0, 5.0])
    np.testing.assert_allclose(X_train_z, [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(X_validation_z, [[98.0, 37.0]])
    np.testing.assert_allclose(tests_z["test_primary"], [[-102.0, -43.0]])
    np.testing.assert_allclose(tests_z["test_secondary"], [[-101.0, -42.8]])


def test_fixed_label_metrics_and_bootstrap_use_physical_bearing_uids(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True)
    out = tmp_path / "out"
    dataset = _make_fake_dataset()
    traces = _install_fakes(monkeypatch, dataset)

    code = RUN.main(_run_args(root, out))

    assert code == 0
    assert traces["macro_labels"] == [(0, 1, 2)] * 3
    assert traces["per_class_labels"] == [(0, 1, 2)] * 3
    assert len(traces["bootstrap_calls"]) == 3
    for view, call in zip(RUN.EVALUATION_VIEWS, traces["bootstrap_calls"]):
        mask = S.partition_mask(dataset.physical_bearing_uids, dataset.condition_codes, view)
        np.testing.assert_array_equal(call["uids"], dataset.physical_bearing_uids[mask])
        assert not np.array_equal(call["uids"], dataset.uids[mask])
        assert call["kwargs"]["labels"] == [0, 1, 2]
        assert call["kwargs"]["n_iterations"] == 10_000
        assert call["kwargs"]["ci"] == 0.95

    manifest, rows, _ = _read_outputs(out)
    assert {row["view"] for row in rows} == set(RUN.EVALUATION_VIEWS)
    assert all(row["macro_f1"] == "0.5" for row in rows)
    assert all(row["f1_healthy"] == "0.4" for row in rows)
    assert all(row["ci_lo"] == "0.35" and row["ci_hi"] == "0.65" for row in rows)
    assert len(manifest["completed_runs"]) == 3


def test_secondary_no_windows_is_skipped_nonfatally_and_single_class_test_is_allowed(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=False)
    out = tmp_path / "out"
    dataset = _make_fake_dataset(include_secondary=False)
    traces = _install_fakes(monkeypatch, dataset)

    code = RUN.main(_run_args(root, out))

    assert code == 0
    assert len(traces["train_calls"]) == 1
    manifest, rows, _ = _read_outputs(out)
    assert {row["view"] for row in rows} == {"test_in_condition", "test_primary"}
    assert len(manifest["skipped_runs"]) == 1
    assert manifest["skipped_runs"][0]["view"] == "test_secondary"

    one_class_secondary = _make_fake_dataset(
        include_secondary=True, secondary_single_class=True
    )
    masks = RUN._partition_masks(one_class_secondary)
    assert masks["test_secondary"].any()


def test_physical_bearing_leakage_is_rejected_but_test_views_may_repeat_held_out_bearings():
    physical = np.asarray(["train-A", "validation-B", "test-C", "test-C"], dtype=object)
    valid_masks = {
        "train": np.asarray([True, False, False, False]),
        "validation": np.asarray([False, True, False, False]),
        "test_in_condition": np.asarray([False, False, True, False]),
        "test_primary": np.asarray([False, False, False, True]),
        "test_secondary": np.asarray([False, False, True, True]),
    }
    RUN._assert_no_physical_bearing_leakage(physical, valid_masks)

    leaked = dict(valid_masks)
    leaked["test_primary"] = np.asarray([True, False, False, False])
    with pytest.raises(RUN.Stage1RunError, match="Physical bearing leakage"):
        RUN._assert_no_physical_bearing_leakage(physical, leaked)


def test_empty_or_single_class_required_partitions_fail_clearly_but_test_class_need_not():
    with pytest.raises(RUN.Stage1RunError, match="empty"):
        RUN._labels_for_partition(np.asarray([], dtype=np.int64), "train", require_all_classes=True)
    with pytest.raises(RUN.Stage1RunError, match="missing required Stage 1 classes"):
        RUN._labels_for_partition(np.asarray([0, 0]), "validation", require_all_classes=True)
    RUN._labels_for_partition(np.asarray([2, 2]), "test_primary", require_all_classes=False)


def test_result_and_manifest_outputs_are_incremental_and_truthful(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True)
    out = tmp_path / "out"
    traces = _install_fakes(monkeypatch, _make_fake_dataset())

    code = RUN.main(_run_args(root, out))

    assert code == 0
    manifest, rows, summary = _read_outputs(out)
    assert manifest["status"] == "COMPLETE"
    assert manifest["completed_at"]
    assert manifest["failed_runs"] == []
    assert len(rows) == len(manifest["completed_runs"]) == 3
    assert set(rows[0]) == set(RUN.RESULT_COLUMNS)
    assert all(row["timestamp"] for row in rows)
    assert "status: COMPLETE" in summary
    assert "descriptive cluster-bootstrap CI" in summary
    assert len(traces["train_calls"]) == 1


def test_partition_validator_does_not_require_all_three_classes_in_test_views():
    dataset = _make_fake_dataset(include_secondary=True, secondary_single_class=True)
    masks = RUN._partition_masks(dataset)
    test_labels = dataset.labels[masks["test_secondary"]]
    assert set(test_labels.tolist()) == {0}


def test_default_cli_configuration_and_repository_safe_output_argument():
    args = RUN._make_argparser().parse_args(
        ["--data-root", "/data/PU", "--out-dir", "/tmp/stage1"]
    )
    assert args.seeds == [42, 1337, 2026, 7, 99]
    assert args.reps == ["R1_Raw", "R3_Order"]
    all_representations = list(D.REPRESENTATIONS)
    assert RUN._parse_representations(",".join(all_representations)) == all_representations
    help_text = " ".join(RUN._make_argparser().format_help().split())
    assert "default: R1_Raw,R3_Order" in help_text
    assert "any supported representation may be selected explicitly" in help_text
    assert args.max_recordings_per_pair is None
    assert args.no_class_weights is False
    assert args.dry_run is False
    assert "out_dir" in RUN._jsonable_arguments(args)


def test_training_errors_mark_every_affected_view_failed_and_never_complete(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=True)
    out = tmp_path / "out"
    traces = _install_fakes(monkeypatch, _make_fake_dataset())
    api = RUN._load_training_api()

    def fail_training(*args, **kwargs):
        raise RuntimeError("synthetic trainer failure")

    api.train_one_fold = fail_training
    monkeypatch.setattr(RUN, "_load_training_api", lambda: api)

    code = RUN.main(_run_args(root, out))

    assert code == 1
    manifest, rows, summary = _read_outputs(out)
    assert manifest["status"] == "PARTIAL_FAILURE"
    assert rows == []
    assert len(manifest["failed_runs"]) == 3
    assert {entry["view"] for entry in manifest["failed_runs"]} == set(RUN.EVALUATION_VIEWS)
    assert all("synthetic trainer failure" in entry["error"] for entry in manifest["failed_runs"])
    assert "status: PARTIAL_FAILURE" in summary
    assert traces["train_calls"] == []


def test_partition_contract_failure_fails_all_planned_combinations(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _make_inventory(root, include_secondary=False)
    out = tmp_path / "out"
    dataset = _make_fake_dataset(include_secondary=False)
    train_mask = S.partition_mask(
        dataset.physical_bearing_uids, dataset.condition_codes, "train"
    )
    dataset.labels[train_mask] = 0
    traces = _install_fakes(monkeypatch, dataset)

    code = RUN.main(_run_args(root, out))

    assert code == 1
    manifest, rows, _ = _read_outputs(out)
    assert manifest["status"] == "FAILED_PARTITION_VALIDATION"
    assert "train partition is missing required Stage 1 classes" in manifest["error"]["message"]
    assert rows == []
    assert len(manifest["failed_runs"]) == 2
    assert {entry["view"] for entry in manifest["failed_runs"]} == {
        "test_in_condition", "test_primary"
    }
    assert traces["train_calls"] == []
