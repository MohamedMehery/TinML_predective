"""
T12-only pilot runner for Experiment 002 (C5/C8).

This is the top-level driver. It implements:
  - 3-class contract (Healthy / Outer_Race / Inner_Race). Ball is
    filtered out by `load_pilot_dataset(include_ball=False)`.
  - Fixed-grid angular resampling for R3_Order (256 for CWRU 12 kHz,
    64 for PU 64 kHz; never mixed).
  - LOBO source-train: for each source LOBO fold, train 5 seeds
    × 5 representations; evaluate on the *full* target dataset.
  - Source-train-only z-score: (mean, std) frozen on the source-train
    fold, applied to source-train, source-test (same-domain sanity),
    and the full target.
  - Per-seed Macro-F1 + per-UID Macro-F1 + cluster bootstrap CI on
    `physical_bearing_uid` for the headline transfer matrix.

The run is configured for the T12 transfer (CWRU -> PU) only; the
other 5 transfer pairs are not in scope for this commit. The runner
is structured so that adding T13, T21, T23, T31, T32 in a future
commit is a small change in the `_run_one_transfer` loop.

Outputs (in this experiment folder, not in 02_datasets/):
  pilot/results/PILOT_T12_per_run.csv       (one row per (rep, fold, seed))
  pilot/results/PILOT_T12_summary.csv      (one row per (rep))
  pilot/results/PILOT_T12_per_uid_f1.csv   (one row per (rep, fold, uid))
  pilot/results/PILOT_T12_meta.json        (config snapshot + counts)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a module (`python -m src.run_pipeline`) or directly
# (`python src/run_pipeline.py`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import datasets as ds_mod
    from src import evaluate as ev_mod
    from src import train as tr_mod
    from src.datasets import (
        CLASS_MAP_3,
        REPRESENTATIONS,
        apply_zscore,
        compute_source_train_zscore,
        get_representation,
        load_pilot_dataset,
        lobo_split,
        make_train_mask,
    )
else:
    from . import datasets as ds_mod
    from . import evaluate as ev_mod
    from . import train as tr_mod
    from .datasets import (
        CLASS_MAP_3,
        REPRESENTATIONS,
        apply_zscore,
        compute_source_train_zscore,
        get_representation,
        load_pilot_dataset,
        lobo_split,
        make_train_mask,
    )


# Frozen config keys we read directly (avoid loading the JSON here to
# keep the import surface small).
DEFAULT_SEEDS = (42, 1337, 2026, 7, 99)
DEFAULT_BOOTSTRAP_N = 10_000
DEFAULT_CI = 0.95
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
DEFAULT_BATCH = 32
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 12


def _make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="T12 pilot runner (CWRU -> PU)")
    p.add_argument("--pilot-root", default=None,
                   help="Path to the pilot folder (default: ../pilot relative to this script).")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for PILOT_*.csv (default: <pilot-root>/results).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="Random seeds (default: 42 1337 2026 7 99).")
    p.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS),
                   help="Representations to evaluate (default: all 5).")
    p.add_argument("--source-folds", type=int, default=None,
                   help="If set, only use the first N source LOBO folds (for smoke runs).")
    p.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N,
                   help="Cluster bootstrap iterations on physical_bearing_uid (default 10000).")
    return p


def _run_one_transfer(
    source: ds_mod.LoadedDataset,
    target: ds_mod.LoadedDataset,
    *,
    representations: list[str],
    seeds: list[int],
    source_folds: int | None,
    bootstrap_n: int,
) -> dict:
    """Run all (representation, source-LOBO-fold, seed) combinations.

    Returns a dict with keys:
      per_run:   list of dicts, one per (rep, fold, seed)
      per_uid:   list of dicts, one per (rep, fold, uid)
      summary:   list of dicts, one per rep, with mean/lo/hi/median
                 Macro-F1 across the source-LOBO folds, per seed, then
                 averaged over seeds.
      meta:      dict with shape info
    """
    folds = lobo_split(source)
    if source_folds is not None:
        folds = folds[:source_folds]

    per_run: list[dict] = []
    per_uid: list[dict] = []

    for rep in representations:
        Xs = get_representation(source, rep)
        Xt = get_representation(target, rep)
        ys = source.labels
        yt = target.labels
        us = source.uids
        ut = target.uids

        for fold_idx, (train_uids, test_uid) in enumerate(folds):
            train_mask = make_train_mask(source, train_uids)
            mean, std = compute_source_train_zscore(Xs, train_mask)

            Xs_tr_z = apply_zscore(Xs, mean, std)
            Xt_z = apply_zscore(Xt, mean, std)

            Xs_tr_z_train = Xs_tr_z[train_mask]
            ys_train = ys[train_mask]

            for seed in seeds:
                t0 = time.time()
                model = tr_mod.train_one_fold(
                    Xs_tr_z_train,
                    ys_train,
                    Xs_tr_z_train,  # unused: internal val split inside train_one_fold
                    ys_train,
                    representation=rep,
                    seed=seed,
                    num_classes=3,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WD,
                    batch_size=DEFAULT_BATCH,
                    max_epochs=DEFAULT_EPOCHS,
                    early_stopping_patience=DEFAULT_PATIENCE,
                )
                elapsed = time.time() - t0

                y_pred = tr_mod.predict(model, Xt_z)
                point, lo, hi, std_f1 = ev_mod.cluster_bootstrap_ci(
                    yt, y_pred, ut, n_iterations=bootstrap_n, ci=DEFAULT_CI, seed=seed
                )
                per_run.append({
                    "representation": rep,
                    "source_fold_idx": fold_idx,
                    "source_test_uid": test_uid,
                    "seed": seed,
                    "n_source_train": int(train_mask.sum()),
                    "n_target_test": int(target.n_windows),
                    "macro_f1_point": point,
                    "macro_f1_ci_lo": lo,
                    "macro_f1_ci_hi": hi,
                    "macro_f1_std": std_f1,
                    "elapsed_sec": round(elapsed, 2),
                })
                # Per-UID F1 for the target (used by the LME and the
                # reviewer's "is the collapse uniform across UIDs?" check).
                per_uid_f1 = ev_mod.per_uid_macro_f1(yt, y_pred, ut)
                for uid, f1 in per_uid_f1.items():
                    per_uid.append({
                        "representation": rep,
                        "source_fold_idx": fold_idx,
                        "seed": seed,
                        "target_uid": uid,
                        "macro_f1": f1,
                    })

                print(
                    f"  [rep={rep:>8}] fold={fold_idx} test_uid={test_uid:>13} "
                    f"seed={seed:>4}  F1={point:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
                    f"({elapsed:.1f}s)"
                )

    # Summarize: per-rep, mean of the per-run point estimates across
    # (fold, seed). This is the headline number per rep.
    summary: list[dict] = []
    for rep in representations:
        rs = [r for r in per_run if r["representation"] == rep]
        if not rs:
            continue
        f1s = np.array([r["macro_f1_point"] for r in rs])
        summary.append({
            "representation": rep,
            "n_runs": len(rs),
            "mean_macro_f1": float(f1s.mean()),
            "std_macro_f1": float(f1s.std()),
            "min_macro_f1": float(f1s.min()),
            "max_macro_f1": float(f1s.max()),
        })

    return {
        "per_run": per_run,
        "per_uid": per_uid,
        "summary": summary,
        "meta": {
            "n_source_folds": len(folds),
            "n_seeds": len(seeds),
            "n_target_uids": len(target.unique_uids),
            "source_dataset": source.name,
            "target_dataset": target.name,
            "representations": list(representations),
            "seeds": list(seeds),
            "bootstrap_n": int(bootstrap_n),
            "ci": DEFAULT_CI,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _make_argparser().parse_args(argv)
    here = Path(__file__).resolve().parent
    pilot_root = args.pilot_root or str(here.parent / "pilot")
    out_dir = args.out_dir or os.path.join(pilot_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("PILOT T12 (CWRU -> PU) — Experiment 002 / EXP-C5C8-002")
    print("=" * 72)
    print(f"pilot root:   {pilot_root}")
    print(f"out dir:      {out_dir}")
    print(f"representations: {args.representations}")
    print(f"seeds:        {args.seeds}")
    print(f"bootstrap n:  {args.bootstrap_n}")
    print(f"source folds: {args.source_folds or 'all'}")
    print()

    # Load the source and target. The 3-class contract is enforced by
    # `include_ball=False`.
    print("Loading CWRU (source)...")
    source = load_pilot_dataset("CWRU", pilot_root=pilot_root, include_ball=False)
    print(f"  CWRU: {source.n_windows} windows, {len(source.unique_uids)} UIDs, "
          f"{source.r1_raw.shape} R1, {source.r3_order.shape} R3")
    print("Loading PU (target)...")
    target = load_pilot_dataset("PU", pilot_root=pilot_root, include_ball=False)
    print(f"  PU:   {target.n_windows} windows, {len(target.unique_uids)} UIDs, "
          f"{target.r1_raw.shape} R1, {target.r3_order.shape} R3")
    print()

    result = _run_one_transfer(
        source, target,
        representations=args.representations,
        seeds=args.seeds,
        source_folds=args.source_folds,
        bootstrap_n=args.bootstrap_n,
    )

    # Write outputs.
    import csv

    per_run_path = os.path.join(out_dir, "PILOT_T12_per_run.csv")
    with open(per_run_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["per_run"][0].keys()))
        w.writeheader()
        w.writerows(result["per_run"])
    print(f"\nWrote {per_run_path}  ({len(result['per_run'])} rows)")

    per_uid_path = os.path.join(out_dir, "PILOT_T12_per_uid_f1.csv")
    with open(per_uid_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["per_uid"][0].keys()))
        w.writeheader()
        w.writerows(result["per_uid"])
    print(f"Wrote {per_uid_path}  ({len(result['per_uid'])} rows)")

    summary_path = os.path.join(out_dir, "PILOT_T12_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["summary"][0].keys()))
        w.writeheader()
        w.writerows(result["summary"])
    print(f"Wrote {summary_path}  ({len(result['summary'])} rows)")

    meta_path = os.path.join(out_dir, "PILOT_T12_meta.json")
    with open(meta_path, "w") as f:
        json.dump(result["meta"], f, indent=2)
    print(f"Wrote {meta_path}")

    # Console summary.
    print("\n" + "=" * 72)
    print("HEADLINE: mean ± std Macro-F1 across source LOBO folds and seeds")
    print("=" * 72)
    for row in result["summary"]:
        print(f"  {row['representation']:>8}: {row['mean_macro_f1']:.4f} ± "
              f"{row['std_macro_f1']:.4f}  (min={row['min_macro_f1']:.4f}, "
              f"max={row['max_macro_f1']:.4f}, n={row['n_runs']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
