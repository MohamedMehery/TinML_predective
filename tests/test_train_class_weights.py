"""Synthetic tests for opt-in class weights and safe train batching."""

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


def _training_arrays(n_train=5, n_val=2, num_classes=2):
    X_train = np.arange(n_train * 2, dtype=np.float32).reshape(n_train, 2) / 10.0
    y_train = np.arange(n_train, dtype=np.int64) % num_classes
    X_val = np.arange(n_val * 2, dtype=np.float32).reshape(n_val, 2) / 7.0
    y_val = np.arange(n_val, dtype=np.int64) % num_classes
    return X_train, y_train, X_val, y_val


def _install_tiny_model(monkeypatch):
    monkeypatch.setattr(
        T,
        "build_model",
        lambda representation, num_classes=3: torch.nn.Linear(2, num_classes),
    )


def _prepare_tiny_fold(X_train, y_train, X_val, y_val, **kwargs):
    return T.train_one_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        representation="tiny-test-model",
        seed=41,
        num_classes=2,
        batch_size=2,
        max_epochs=0,
        early_stopping_patience=1,
        use_supplied_validation=True,
        **kwargs,
    )


def test_compute_class_weights_uses_inverse_frequency_formula_in_class_order():
    labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)

    weights = T.compute_class_weights(labels, num_classes=3)

    np.testing.assert_allclose(weights, np.asarray([1.0, 2.0, 2.0 / 3.0], dtype=np.float32))
    assert weights.dtype == np.float32


def test_compute_class_weights_rejects_empty_or_missing_classes():
    with pytest.raises(ValueError, match="at least one class label"):
        T.compute_class_weights(np.asarray([], dtype=np.int64), num_classes=3)
    with pytest.raises(ValueError, match="missing required classes"):
        T.compute_class_weights(np.asarray([0, 0, 1], dtype=np.int64), num_classes=3)


@pytest.mark.parametrize("num_classes", [0, -1, 2.5, True])
def test_compute_class_weights_rejects_invalid_num_classes(num_classes):
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        T.compute_class_weights(np.asarray([0, 1], dtype=np.int64), num_classes=num_classes)


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([-1, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 3], dtype=np.int64),
    ],
)
def test_compute_class_weights_rejects_out_of_range_labels(labels):
    with pytest.raises(ValueError, match="range 0 to 2"):
        T.compute_class_weights(labels, num_classes=3)


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0.0, 1.5, 2.0]),
        np.asarray([0.0, np.nan, 2.0]),
        np.asarray([0.0, np.inf, 2.0]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        np.asarray(["0", "1", "2"]),
    ],
)
def test_compute_class_weights_rejects_uninterpretable_training_labels(labels):
    with pytest.raises(ValueError):
        T.compute_class_weights(labels, num_classes=3)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([1.0, np.nan]), "finite"),
        (np.asarray([1.0, np.inf]), "finite"),
        (np.asarray([1.0, -np.inf]), "finite"),
        (np.asarray([1.0, 0.0]), "strictly positive"),
        (np.asarray([1.0, -0.5]), "strictly positive"),
        (np.asarray([1.0]), "shape"),
    ],
)
def test_train_rejects_invalid_class_weights_before_model_creation(
    monkeypatch, weights, message
):
    X_train, y_train, X_val, y_val = _training_arrays()
    monkeypatch.setattr(
        T,
        "build_model",
        lambda *args, **kwargs: pytest.fail("invalid weights reached model creation"),
    )

    with pytest.raises(ValueError, match=message):
        _prepare_tiny_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            class_weights=weights,
        )


def test_supplied_weights_reach_cross_entropy_on_model_device(monkeypatch):
    X_train, y_train, X_val, y_val = _training_arrays()
    _install_tiny_model(monkeypatch)
    original_loss = T.nn.CrossEntropyLoss
    captured = []

    def cross_entropy_spy(*args, **kwargs):
        criterion = original_loss(*args, **kwargs)
        captured.append(kwargs.copy())
        return criterion

    monkeypatch.setattr(T.nn, "CrossEntropyLoss", cross_entropy_spy)
    model = _prepare_tiny_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights=np.asarray([1.0, 2.5], dtype=np.float64),
    )

    assert len(captured) == 1
    weight = captured[0]["weight"]
    assert weight.dtype == torch.float32
    assert weight.device == next(model.parameters()).device
    torch.testing.assert_close(weight.cpu(), torch.tensor([1.0, 2.5]))


def test_none_class_weights_preserves_unweighted_loss_construction(monkeypatch):
    X_train, y_train, X_val, y_val = _training_arrays()
    _install_tiny_model(monkeypatch)
    original_loss = T.nn.CrossEntropyLoss
    captured = []

    def cross_entropy_spy(*args, **kwargs):
        captured.append(kwargs.copy())
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(T.nn, "CrossEntropyLoss", cross_entropy_spy)
    _prepare_tiny_fold(X_train, y_train, X_val, y_val, class_weights=None)

    assert captured == [{}]


def _capture_loaders(monkeypatch):
    original_make_loader = T._make_loader
    observed = []

    def loader_spy(X, y, batch_size, shuffle, *, drop_last=False):
        phase = "train" if shuffle else "validation"
        loader = original_make_loader(
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        observed.append((phase, loader, drop_last))
        return loader

    monkeypatch.setattr(T, "_make_loader", loader_spy)
    return observed


@pytest.mark.parametrize(
    ("drop_last_train", "expected_train_sizes"),
    [(True, [2, 2]), (False, [2, 2, 1])],
)
def test_drop_last_train_only_changes_training_partial_batch(
    monkeypatch, drop_last_train, expected_train_sizes
):
    X_train, y_train, X_val, y_val = _training_arrays(n_train=5, n_val=2)
    _install_tiny_model(monkeypatch)
    observed = _capture_loaders(monkeypatch)

    _prepare_tiny_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        drop_last_train=drop_last_train,
    )

    train_entries = [entry for entry in observed if entry[0] == "train"]
    val_entries = [entry for entry in observed if entry[0] == "validation"]
    train_sizes = [len(y_batch) for _, loader, _ in train_entries for _, y_batch in loader]
    val_sizes = [len(y_batch) for _, loader, _ in val_entries for _, y_batch in loader]
    assert train_sizes == expected_train_sizes
    assert val_sizes == [2]
    assert all(drop_last is False for _, _, drop_last in val_entries)
    if drop_last_train:
        assert all(size > 1 for size in train_sizes)
        assert all(drop_last is True for _, _, drop_last in train_entries)
    else:
        assert all(drop_last is False for _, _, drop_last in observed)


def test_drop_last_train_rejects_fewer_examples_than_batch_before_model_creation(monkeypatch):
    X_train, y_train, X_val, y_val = _training_arrays(n_train=3, n_val=2)
    monkeypatch.setattr(
        T,
        "build_model",
        lambda *args, **kwargs: pytest.fail("empty-loader guard ran too late"),
    )

    with pytest.raises(ValueError, match="drop_last_train=True would produce no complete training batches"):
        T.train_one_fold(
            X_train,
            y_train,
            X_val,
            y_val,
            representation="tiny-test-model",
            seed=41,
            num_classes=2,
            batch_size=4,
            max_epochs=1,
            use_supplied_validation=True,
            drop_last_train=True,
        )
