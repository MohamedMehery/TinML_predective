"""
Evaluation utilities for Experiment 002.

Implements:
  - Macro-F1 on the target's test set (the standard multi-class
    summary metric for bearing-fault classification; matches the
    C5/C8 charter's "Δ Macro-F1" notation).
  - Cluster bootstrap on `physical_bearing_uid` for 95 % CIs, with
    `n_iterations` controlled by experiment_config.json's
    `statistical_evaluation.cluster_bootstrap_iterations = 10000`.

For the T12-only pilot, the bootstrap is over the 13 PU UIDs
(LOBO-style, 13 folds of the target).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over the unique labels in y_true.

    Labels not present in y_true contribute 0 to the average.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_uid_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, uids: np.ndarray) -> dict[str, float]:
    """Compute Macro-F1 within each UID; return uid -> f1."""
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
) -> tuple[float, float, float, float]:
    """Cluster bootstrap on `uids` for a 95 % CI of Macro-F1.

    Each iteration samples UIDs with replacement (preserving all
    windows from each sampled UID), computes Macro-F1 on the
    concatenated sample, and accumulates the distribution. Returns
    (point_estimate, ci_lo, ci_hi, std).
    """
    rng = np.random.RandomState(seed)
    uid_list = sorted(set(uids.tolist()))
    uid_index = {u: i for i, u in enumerate(uid_list)}
    uid_to_rows = {u: np.where(uids == u)[0] for u in uid_list}
    n_uids = len(uid_list)

    point = macro_f1(y_true, y_pred)
    boots = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        sampled = rng.randint(0, n_uids, size=n_uids)
        rows = np.concatenate([uid_to_rows[uid_list[u]] for u in sampled])
        if rows.size == 0:
            boots[i] = point
            continue
        boots[i] = macro_f1(y_true[rows], y_pred[rows])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(point), float(lo), float(hi), float(boots.std())
