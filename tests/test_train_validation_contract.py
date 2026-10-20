"""Synthetic-only contracts for train_one_fold's validation data source."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPO_ROOT), str(EXP_ROOT), str(EXP_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.train as T
except Exception:
    import train as T


def test_supplied_validation_option_is_keyword_only_and_defaults_false():
    parameter = inspect.signature(T.train_one_fold).parameters["use_supplied_validation"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def _marker_rows(markers, width=1024):
    values = np.asarray(markers, dtype=np.float32)
    return np.repeat(values[:, None], width, axis=1)


def _install_data_path_instrumentation(monkeypatch):
    original_make_loader = T._make_loader
    original_build_model = T.build_model
    original_cross_entropy = T.nn.CrossEntropyLoss
    traces = {
        "loader_calls": [],
        "training_loader_batches": [],
        "validation_loader_batches": [],
        "training_model_inputs": [],
        "validation_model_inputs": [],
        "training_loss_targets": [],
        "validation_loss_targets": [],
        "validation_loss_logits": [],
    }
    current_model = {}

    def build_model_spy(representation, num_classes=3):
        model = original_build_model(representation, num_classes=num_classes)
        current_model["model"] = model

        def capture_model_input(module, args):
            phase = "training" if module.training else "validation"
            traces[f"{phase}_model_inputs"].append(
                args[0].detach().cpu().numpy().copy()
            )

        model.register_forward_pre_hook(capture_model_input)
        return model

    class RecordingCrossEntropy:
        def __init__(self):
            self.inner = original_cross_entropy()

        def __call__(self, logits, targets):
            phase = "training" if current_model["model"].training else "validation"
            if phase == "validation":
                traces["validation_loss_logits"].append(
                    logits.detach().cpu().numpy().copy()
                )
            traces[f"{phase}_loss_targets"].append(
                targets.detach().cpu().numpy().copy()
            )
            return self.inner(logits, targets)

    def make_loader_spy(X, y, batch_size, shuffle):
        phase = "training" if shuffle else "validation"
        traces["loader_calls"].append(
            {
                "X": np.asarray(X).copy(),
                "y": np.asarray(y).copy(),
                "shuffle": shuffle,
                "batch_size": batch_size,
            }
        )
        loader = original_make_loader(X, y, batch_size=batch_size, shuffle=shuffle)

        def record_batches():
            for xb, yb in loader:
                traces[f"{phase}_loader_batches"].append(
                    (xb.detach().cpu().numpy().copy(), yb.detach().cpu().numpy().copy())
                )
                yield xb, yb

        return record_batches()

    monkeypatch.setattr(T, "build_model", build_model_spy)
    monkeypatch.setattr(T, "_make_loader", make_loader_spy)
    monkeypatch.setattr(T.nn, "CrossEntropyLoss", RecordingCrossEntropy)
    return traces


def _reset_traces(traces):
    for values in traces.values():
        values.clear()


def _assert_batches_equal(first, second):
    assert len(first) == len(second)
    for first_batch, second_batch in zip(first, second):
        if isinstance(first_batch, dict):
            assert first_batch["shuffle"] == second_batch["shuffle"]
            assert first_batch["batch_size"] == second_batch["batch_size"]
            np.testing.assert_array_equal(first_batch["X"], second_batch["X"])
            np.testing.assert_array_equal(first_batch["y"], second_batch["y"])
        elif isinstance(first_batch, tuple):
            np.testing.assert_array_equal(first_batch[0], second_batch[0])
            np.testing.assert_array_equal(first_batch[1], second_batch[1])
        else:
            np.testing.assert_array_equal(first_batch, second_batch)


def _call_train(X_train, y_train, X_val, y_val, *, seed=17, **kwargs):
    return T.train_one_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        representation="R3_Order",
        seed=seed,
        max_epochs=1,
        batch_size=2,
        early_stopping_patience=2,
        **kwargs,
    )


def test_default_path_uses_internal_train_split_and_ignores_supplied_validation(monkeypatch):
    traces = _install_data_path_instrumentation(monkeypatch)
    X_train = _marker_rows(range(10, 18))
    y_train = np.asarray([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
    supplied_validation_a = _marker_rows([81, 82])
    supplied_labels_a = np.asarray([2, 2], dtype=np.int64)
    supplied_validation_b = _marker_rows([-81, -82])
    supplied_labels_b = np.asarray([0, 1], dtype=np.int64)
    seed = 23
    val_frac = 0.25

    _call_train(
        X_train,
        y_train,
        supplied_validation_a,
        supplied_labels_a,
        seed=seed,
        val_frac=val_frac,
    )
    first_run = {key: list(values) for key, values in traces.items()}

    _reset_traces(traces)
    _call_train(
        X_train,
        y_train,
        supplied_validation_b,
        supplied_labels_b,
        seed=seed,
        val_frac=val_frac,
    )
    second_run = {key: list(values) for key, values in traces.items()}

    permutation = np.random.RandomState(seed).permutation(len(X_train))
    n_val = max(1, int(round(val_frac * len(X_train))))
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]
    expected_train_X = X_train[train_indices]
    expected_train_y = y_train[train_indices]
    expected_val_X = X_train[val_indices]
    expected_val_y = y_train[val_indices]

    for run in (first_run, second_run):
        assert len(run["loader_calls"]) == 2
        assert run["loader_calls"][0]["shuffle"] is True
        assert run["loader_calls"][1]["shuffle"] is False
        np.testing.assert_array_equal(run["loader_calls"][0]["X"], expected_train_X)
        np.testing.assert_array_equal(run["loader_calls"][0]["y"], expected_train_y)
        np.testing.assert_array_equal(run["loader_calls"][1]["X"], expected_val_X)
        np.testing.assert_array_equal(run["loader_calls"][1]["y"], expected_val_y)

        seen_train_X = np.concatenate(run["training_model_inputs"], axis=0)
        seen_val_X = np.concatenate(run["validation_model_inputs"], axis=0)
        assert sorted(seen_train_X[:, 0].tolist()) == sorted(expected_train_X[:, 0].tolist())
        np.testing.assert_array_equal(seen_val_X, expected_val_X)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(run["training_loss_targets"])),
            np.sort(expected_train_y),
        )
        np.testing.assert_array_equal(
            np.concatenate(run["validation_loss_targets"]), expected_val_y
        )
        supplied_markers = np.concatenate(
            (supplied_validation_a[:, 0], supplied_validation_b[:, 0])
        )
        assert not np.isin(seen_train_X[:, 0], supplied_markers).any()
        assert not np.isin(seen_val_X[:, 0], supplied_markers).any()

    for key in first_run:
        _assert_batches_equal(first_run[key], second_run[key])


def test_opt_in_uses_exact_supplied_validation_arrays_and_keeps_them_out_of_training(monkeypatch):
    traces = _install_data_path_instrumentation(monkeypatch)
    X_train = _marker_rows([1, 2, 3, 4])
    y_train = np.asarray([0, 1, 2, 0], dtype=np.int64)
    X_val = _marker_rows([101, 102, 103])
    y_val = np.asarray([2, 0, 1], dtype=np.int64)

    _call_train(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=31,
        use_supplied_validation=True,
    )

    assert len(traces["loader_calls"]) == 2
    training_loader, validation_loader = traces["loader_calls"]
    assert training_loader["shuffle"] is True
    assert validation_loader["shuffle"] is False
    np.testing.assert_array_equal(training_loader["X"], X_train)
    np.testing.assert_array_equal(training_loader["y"], y_train)
    np.testing.assert_array_equal(validation_loader["X"], X_val)
    np.testing.assert_array_equal(validation_loader["y"], y_val)

    # Trace the actual batches and model inputs, using distinct sample markers
    # rather than comparing unrelated local row indices.
    actual_train_batches = np.concatenate(
        [X_batch for X_batch, _ in traces["training_loader_batches"]], axis=0
    )
    actual_train_inputs = np.concatenate(traces["training_model_inputs"], axis=0)
    actual_validation_batches = np.concatenate(
        [X_batch for X_batch, _ in traces["validation_loader_batches"]], axis=0
    )
    actual_validation_targets = np.concatenate(
        [y_batch for _, y_batch in traces["validation_loader_batches"]]
    )
    actual_validation_inputs = np.concatenate(
        traces["validation_model_inputs"], axis=0
    )
    actual_validation_loss_targets = np.concatenate(
        traces["validation_loss_targets"]
    )

    np.testing.assert_array_equal(
        np.sort(actual_train_batches[:, 0]), np.sort(X_train[:, 0])
    )
    np.testing.assert_array_equal(
        np.sort(actual_train_inputs[:, 0]), np.sort(X_train[:, 0])
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate(traces["training_loss_targets"])), np.sort(y_train)
    )
    np.testing.assert_array_equal(actual_validation_batches, X_val)
    np.testing.assert_array_equal(actual_validation_targets, y_val)
    np.testing.assert_array_equal(actual_validation_inputs, X_val)
    np.testing.assert_array_equal(actual_validation_loss_targets, y_val)
    assert len(traces["validation_loss_logits"]) == len(
        traces["validation_model_inputs"]
    )
    assert set(actual_train_batches[:, 0]).isdisjoint(set(X_val[:, 0]))
    assert set(actual_train_inputs[:, 0]).isdisjoint(set(X_val[:, 0]))
    assert set(actual_validation_inputs[:, 0]) == set(X_val[:, 0])


def test_invalid_supplied_validation_fails_before_training(monkeypatch):
    X_train = _marker_rows([1, 2])
    y_train = np.asarray([0, 1], dtype=np.int64)

    def training_must_not_start(*args, **kwargs):
        pytest.fail("validation contract failure must happen before training starts")

    monkeypatch.setattr(T, "_set_seed", training_must_not_start)
    monkeypatch.setattr(T, "build_model", training_must_not_start)
    monkeypatch.setattr(T, "_make_loader", training_must_not_start)

    invalid_cases = [
        (None, np.asarray([0, 1]), "non-None X_val"),
        (_marker_rows([10, 11]), None, "non-None y_val"),
        (_marker_rows([]), np.asarray([], dtype=np.int64), "X_val to contain at least one"),
        (_marker_rows([10]), np.asarray([], dtype=np.int64), "y_val to contain at least one"),
        (_marker_rows([10, 11]), np.asarray([1]), "matching X_val/y_val sample counts"),
    ]
    for X_val, y_val, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            T.train_one_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                representation="R3_Order",
                seed=5,
                use_supplied_validation=True,
            )


def test_omitted_required_validation_arguments_fail_at_call_boundary():
    X_train = _marker_rows([1, 2])
    y_train = np.asarray([0, 1], dtype=np.int64)
    common = {
        "X_train": X_train,
        "y_train": y_train,
        "representation": "R3_Order",
        "seed": 7,
        "use_supplied_validation": True,
    }
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'X_val'"):
        T.train_one_fold(**common, y_val=y_train)
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'y_val'"):
        T.train_one_fold(**common, X_val=X_train)


def test_default_and_opt_in_paths_return_trainable_three_class_models():
    X_train = _marker_rows([1, 2, 3, 4])
    y_train = np.asarray([0, 1, 2, 0], dtype=np.int64)
    X_val = _marker_rows([21, 22])
    y_val = np.asarray([1, 2], dtype=np.int64)

    default_model = T.train_one_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        representation="R3_Order",
        seed=41,
        max_epochs=1,
        batch_size=3,
        val_frac=0.25,
    )
    supplied_model = T.train_one_fold(
        X_train[:3],
        y_train[:3],
        X_val,
        y_val,
        representation="R3_Order",
        seed=43,
        max_epochs=1,
        batch_size=3,
        use_supplied_validation=True,
    )

    assert isinstance(default_model, torch.nn.Module)
    assert isinstance(supplied_model, torch.nn.Module)
    assert next(default_model.parameters()).device.type == "cpu"
    assert next(supplied_model.parameters()).device.type == "cpu"
    test_inputs = torch.from_numpy(X_train[:2]).float()
    with torch.no_grad():
        assert default_model(test_inputs).shape == (2, 3)
        assert supplied_model(test_inputs).shape == (2, 3)
