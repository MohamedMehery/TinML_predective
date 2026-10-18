"""
Per-seed training for Experiment 002.

The training loop is intentionally small: AdamW, early stopping on
source-train-val (a 15 % slice of the source-train fold), CrossEntropy
loss. CPU-deterministic per the config (`device: "cpu_deterministic"`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model


def _set_seed(seed: int) -> None:
    """Set seeds for numpy, torch CPU. Mirrors the config's
    `device: "cpu_deterministic"`."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # `torch.use_deterministic_algorithms(True)` is intentionally not set
    # globally because the AdaptiveAvgPool1d path uses a kernel-size-1
    # pooling which is deterministic, but enabling the global flag can
    # surface in unrelated ways during debugging. The seed reset is
    # sufficient for the per-fold training.


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a torch DataLoader from numpy arrays.

    The first axis of X is the window index; subsequent axes are
    representation-specific (1 x 2048 for R1, 1024 for R2/R3).
    """
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    representation: str,
    seed: int,
    *,
    num_classes: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 60,
    early_stopping_patience: int = 12,
    val_frac: float = 0.15,
    use_supplied_validation: bool = False,
) -> nn.Module:
    """Train one model on a source-train fold with optional supplied validation.

    With ``use_supplied_validation=False`` (the default), the existing
    internal window-level random split of ``X_train/y_train`` is preserved.
    That split is unsuitable for bearing-level generalization claims because
    windows from one physical bearing may occur in both the fit and validation
    subsets.

    For Stage 1 bearing-level work, callers must explicitly set
    ``use_supplied_validation=True`` and pass verified, bearing-disjoint,
    source-domain training and validation arrays. This function does not
    establish or verify bearing-disjointness; constructing and checking that
    split is the caller's responsibility. In a zero-shot transfer evaluation,
    target-domain final-evaluation data must never be supplied for early
    stopping.

    When enabled, the supplied ``X_val/y_val`` arrays are required to be
    non-empty and have matching sample counts. The function returns the
    best-validation model state (re-loaded into a fresh model object so the
    caller can ``state_dict()`` it for serialization).
    """
    if use_supplied_validation:
        if X_val is None:
            raise ValueError(
                "use_supplied_validation=True requires a non-None X_val array"
            )
        if y_val is None:
            raise ValueError(
                "use_supplied_validation=True requires a non-None y_val array"
            )
        try:
            n_x_val = len(X_val)
        except TypeError as exc:
            raise ValueError(
                "use_supplied_validation=True requires X_val to be a non-empty "
                "sample array"
            ) from exc
        try:
            n_y_val = len(y_val)
        except TypeError as exc:
            raise ValueError(
                "use_supplied_validation=True requires y_val to be a non-empty "
                "sample array"
            ) from exc
        if n_x_val == 0:
            raise ValueError(
                "use_supplied_validation=True requires X_val to contain at least "
                "one validation sample"
            )
        if n_y_val == 0:
            raise ValueError(
                "use_supplied_validation=True requires y_val to contain at least "
                "one validation label"
            )
        if n_x_val != n_y_val:
            raise ValueError(
                "use_supplied_validation=True requires matching X_val/y_val "
                f"sample counts; got {n_x_val} and {n_y_val}"
            )

    _set_seed(seed)

    if use_supplied_validation:
        Xtr, ytr = X_train, y_train
        Xv, yv = X_val, y_val
    else:
        # Carve out a validation slice from the source-train fold.
        n = X_train.shape[0]
        perm = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(round(val_frac * n)))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xv, yv = X_train[val_idx], y_train[val_idx]

    model = build_model(representation, num_classes=num_classes)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    train_loader = _make_loader(Xtr, ytr, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(Xv, yv, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += float(crit(logits, yb).item()) * xb.size(0)
        val_loss /= max(1, Xv.shape[0])

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Return class predictions (argmax) as an int64 numpy array."""
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        logits = model(xb)
        out.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Return softmax probabilities as a float32 numpy array."""
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).float()
        logits = model(xb)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
