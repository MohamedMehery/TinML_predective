# Pilot T12 Results — CWRU → PU (synthetic data)

> **This file is a placeholder; the real numbers are filled in once the
> pilot run completes.** The current state of the headline numbers is
> below; the CSV outputs in `pilot/results/` are the source of truth.

## What this is

The T12 (CWRU → PU) cross-machine transfer pilot, run on the synthetic
data in `pilot/synthetic/raw_data/`. The pilot implements the four
diagnostic-surfaced issues from `../NOTES.md`:

  - 3-class contract (Healthy / Outer_Race / Inner_Race; Ball filtered).
  - Fixed-grid angular resampling for R3_Order.
  - LOBO on `physical_bearing_uid` for the source (7 folds).
  - Source-train-only z-score normalization.

And the full statistical protocol from `config/experiment_config.json`:

  - 5 representations (R1_Raw, R1_Env, R2_Raw, R2_Env, R3_Order).
  - 5 random seeds (42, 1337, 2026, 7, 99).
  - Cluster bootstrap on `physical_bearing_uid` (n=10,000) for 95 % CIs.
  - 3-class Macro-F1 on the full PU target.

Total: 5 × 7 × 5 = **175 training runs**.

## How to read the results

| File | What it contains |
|---|---|
| `PILOT_T12_per_run.csv` | One row per (representation, source LOBO fold, seed). Columns: `representation`, `source_fold_idx`, `source_test_uid`, `seed`, `n_source_train`, `n_target_test`, `macro_f1_point`, `macro_f1_ci_lo`, `macro_f1_ci_hi`, `macro_f1_std`, `elapsed_sec`. |
| `PILOT_T12_per_uid_f1.csv` | One row per (representation, source LOBO fold, seed, target UID). For the cluster bootstrap and the per-UID variance check. |
| `PILOT_T12_summary.csv` | One row per representation, with `mean_macro_f1`, `std_macro_f1`, `min`, `max`, `n_runs` aggregated over the 35 (fold × seed) runs. |
| `PILOT_T12_meta.json` | The config snapshot (representations, seeds, bootstrap n, dataset names) used for the run. |

## Headline (filled by the run)

| Representation | Mean Macro-F1 | Std | n |
|---|---|---|---|
| R1_Raw | _TBD_ | _TBD_ | 35 |
| R1_Env | _TBD_ | _TBD_ | 35 |
| R2_Raw | _TBD_ | _TBD_ | 35 |
| R2_Env | _TBD_ | _TBD_ | 35 |
| R3_Order | _TBD_ | _TBD_ | 35 |

## What the headline does and does NOT mean

**What it does mean:**

- Whether the 5-representation pipeline (R1-Raw through R3-Order)
  can be trained end-to-end on the pilot data without numerical
  errors, NaNs, or shape mismatches. If any of those happen, the
  pilot fails its smoke-test role.
- The relative ordering of the 5 representations on the synthetic
  CWRU → PU transfer. If R3_Order outperforms the data-driven reps
  (R1_Raw, R1_Env, R2_Raw, R2_Env) on this transfer, the C5/C8
  hypothesis is *consistent* with the synthetic data. If it
  underperforms, the synthetic generator is not a faithful
  testbed for the hypothesis and the real-data experiment is
  needed.

**What it does NOT mean:**

- The C5/C8 hypothesis is not tested by the pilot. The pilot tests
  the *pipeline*. The hypothesis is about real cross-machine transfer
  on the public CWRU / Paderborn / SEU benchmarks. The pilot's
  synthetic data is constructed from a deterministic kinematic +
  structural-resonance model; the casing transfer functions are
  *designed in* by the generator, not learned. The C5/C8 hypothesis
  is precisely about *learning* the casing transfer function being
  the failure mode, so a synthetic dataset where the casing is
  built in cannot confirm or refute it.
- These Macro-F1 numbers are not comparable to any real-data
  number. They are only comparable to each other and to the
  per-UID breakdown.
- A drop in Macro-F1 on the `CWRU_N` (Healthy) holdout fold is
  *expected*, not a bug: when the only Healthy UID is held out
  of the source train fold, the model has no positive examples of
  the Healthy class, and the macro-F1 drops because the model
  over-predicts fault. The per-UID F1 CSV is the place to
  verify this is what is happening.

## How to reproduce

```bash
cd 04_experiments/002_cross_machine_invariance
python3 -m src.run_pipeline --bootstrap-n 10000
```

The run is deterministic per seed (`numpy.random.RandomState(seed)`
for splits, `torch.manual_seed(seed)` for training). Re-running
should reproduce the headline numbers to within the bootstrap
quantization.

## What is NOT in this pilot

- The other 5 transfer pairs (T13 CWRU→SEU, T21 PU→CWRU, T23
  PU→SEU, T31 SEU→CWRU, T32 SEU→PU). T12 is the headline
  cross-machine pair; the other 5 are TBD.
- The LME regression and Hedges' g effect size. The CSV outputs
  contain the per-(rep, fold, seed) F1s and the per-UID F1s; a
  separate `evaluate_lme.py` (TBD) will fit the mixed-effects
  model and the effect sizes.
- Real CWRU / PU / SEU data. The `[TO VERIFY FROM SOURCE DOC]`
  markers in `02_datasets/manifests/cwru.md` and `paderborn.md`
  are still open; the real-data experiment is a separate
  future commit.
