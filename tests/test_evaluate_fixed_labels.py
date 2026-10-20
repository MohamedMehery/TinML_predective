"""Regression tests for fixed-label evaluation utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "04_experiments" / "002_cross_machine_invariance"
for candidate in (str(REPO_ROOT), str(EXP_ROOT), str(EXP_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import src.evaluate as E
except Exception:
    import evaluate as E


def test_macro_f1_without_labels_matches_existing_sklearn_inference():
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1], dtype=np.int64)

    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)

    assert E.macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_fixed_label_macro_f1_includes_absent_classes_and_uses_zero_division(monkeypatch):
    y_true = np.asarray([0, 0], dtype=np.int64)
    y_pred = np.asarray([0, 0], dtype=np.int64)
    original_f1_score = E.f1_score
    observed = []

    def f1_score_spy(*args, **kwargs):
        observed.append(kwargs.copy())
        return original_f1_score(*args, **kwargs)

    monkeypatch.setattr(E, "f1_score", f1_score_spy)
    result = E.macro_f1(y_true, y_pred, labels=[0, 1, 2])

    assert result == pytest.approx(1.0 / 3.0)
    assert observed == [
        {"average": "macro", "labels": [0, 1, 2], "zero_division": 0}
    ]


def test_per_class_f1_returns_all_labels_in_requested_order(monkeypatch):
    y_true = np.asarray([0, 0], dtype=np.int64)
    y_pred = np.asarray([0, 0], dtype=np.int64)
    original_f1_score = E.f1_score
    observed = []

    def f1_score_spy(*args, **kwargs):
        observed.append(kwargs.copy())
        return original_f1_score(*args, **kwargs)

    monkeypatch.setattr(E, "f1_score", f1_score_spy)
    scores = E.per_class_f1(y_true, y_pred, labels=(2, 0, 1))

    assert list(scores) == [2, 0, 1]
    assert scores == {2: 0.0, 0: 1.0, 1: 0.0}
    assert all(isinstance(value, float) for value in scores.values())
    assert observed == [
        {"average": None, "labels": (2, 0, 1), "zero_division": 0}
    ]


def test_per_uid_macro_f1_default_behavior_is_unchanged():
    y_true = np.asarray([0, 1, 0, 0, 1, 1], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
    uids = np.asarray(["rec-b", "rec-a", "rec-b", "rec-c", "rec-a", "rec-c"])

    expected = {}
    for uid in sorted(set(uids.tolist())):
        mask = uids == uid
        expected[uid] = float(
            f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        )

    assert E.per_uid_macro_f1(y_true, y_pred, uids) == expected


def test_cluster_bootstrap_default_output_matches_baseline_fixed_seed():
    y_true = np.asarray([0, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 2], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1, 2, 0, 0, 1, 0, 0, 2, 2], dtype=np.int64)
    uids = np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4)

    result = E.cluster_bootstrap_ci(
        y_true,
        y_pred,
        uids,
        n_iterations=31,
        ci=0.90,
        seed=13,
    )

    np.testing.assert_allclose(
        result,
        (
            0.7579365079365079,
            0.6237373737373737,
            0.7878787878787878,
            0.05178332902772799,
        ),
        rtol=0.0,
        atol=1e-14,
    )


def test_cluster_bootstrap_passes_fixed_labels_to_each_macro_f1_call(monkeypatch):
    y_true = np.asarray([0, 1, 0, 1, 2, 0], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1, 2, 0], dtype=np.int64)
    uids = np.asarray(["bearing-a"] * 2 + ["bearing-b"] * 2 + ["bearing-c"] * 2)
    original_macro_f1 = E.macro_f1
    observed_labels = []

    def macro_f1_spy(y_true, y_pred, *, labels=None):
        observed_labels.append(labels)
        return original_macro_f1(y_true, y_pred, labels=labels)

    monkeypatch.setattr(E, "macro_f1", macro_f1_spy)
    E.cluster_bootstrap_ci(
        y_true,
        y_pred,
        uids,
        n_iterations=7,
        ci=0.90,
        seed=5,
        labels=[0, 1, 2],
    )

    assert len(observed_labels) == 8  # point estimate plus seven resamples
    assert all(labels == (0, 1, 2) for labels in observed_labels)
