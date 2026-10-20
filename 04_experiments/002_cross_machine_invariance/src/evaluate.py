"""
Evaluation utilities for Experiment 002.

Implements:
  - Macro-F1 on the target's test set (the standard multi-class
    summary metric for bearing-fault classification; matches the
    C5/C8 charter's "Δ Macro-F1" notation).
  - Cluster bootstrap on caller-supplied UIDs for descriptive 95 %
    intervals. Pass physical-bearing IDs when bearing-level resampling
    is intended; the statistic remains pooled-window Macro-F1.

For the T12-only pilot, the bootstrap is over the 13 PU UIDs
(LOBO-style, 13 folds of the target).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[Sequence[int]] = None,
) -> float:
    """Return macro-averaged F1, optionally over an explicit label set.

    When ``labels`` is omitted, scikit-learn infers the labels from the
    observed inputs, preserving the original behavior. Supplying labels
    keeps absent expected classes in the average; their zero-division
    value is zero.
    """
    if labels is None:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=labels,
            zero_division=0,
        )
    )


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int] = (0, 1, 2),
) -> dict[int, float]:
    """Return F1 for every requested label, preserving the given order."""
    ordered_labels = tuple(labels)
    scores = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=ordered_labels,
        zero_division=0,
    )
    return {
        label: float(score)
        for label, score in zip(ordered_labels, scores)
    }


def per_uid_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, uids: np.ndarray) -> dict[str, float]:
    """Compute default Macro-F1 within each caller-supplied UID.

    This groups by the values in ``uids`` without determining whether
    they identify recordings or physical bearings. It preserves the
    existing per-UID statistic and does not imply a physical-bearing
    metric.
    """
    out: dict[str, float] = {}
    for uid in sorted(set(uids.tolist())):
        m = uids == uid
        if m.sum() == 0:
            continue
        out[uid] = macro_f1(y_true[m], y_pred[m])
    return out


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uids: np.ndarray,
    *,
    n_iterations: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    labels: Optional[Sequence[int]] = None,
) -> tuple[float, float, float, float]:
    """Return a descriptive cluster-bootstrap interval for Macro-F1.

    Each iteration samples the caller-supplied UIDs with replacement,
    preserving every window belonging to each sampled UID. The statistic
    is pooled-window Macro-F1 over the concatenated resampled clusters,
    not an equal-UID average. Supply physical-bearing IDs to obtain
    physical-bearing-level resampling; recording IDs instead resample
    recordings. The resulting interval is descriptive and does not by
    itself address dependence across conditions or transfer-pair contrasts.

    Returns ``(point_estimate, ci_lo, ci_hi, std)``.
    """
    rng = np.random.RandomState(seed)
    uid_list = sorted(set(uids.tolist()))
    uid_index = {u: i for i, u in enumerate(uid_list)}
    uid_to_rows = {u: np.where(uids == u)[0] for u in uid_list}
    n_uids = len(uid_list)
    metric_labels = None if labels is None else tuple(labels)

    if metric_labels is None:
        point = macro_f1(y_true, y_pred)
    else:
        point = macro_f1(y_true, y_pred, labels=metric_labels)
    boots = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        sampled = rng.randint(0, n_uids, size=n_uids)
        rows = np.concatenate([uid_to_rows[uid_list[u]] for u in sampled])
        if rows.size == 0:
            boots[i] = point
            continue
        if metric_labels is None:
            boots[i] = macro_f1(y_true[rows], y_pred[rows])
        else:
            boots[i] = macro_f1(
                y_true[rows], y_pred[rows], labels=metric_labels
            )
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(point), float(lo), float(hi), float(boots.std())
