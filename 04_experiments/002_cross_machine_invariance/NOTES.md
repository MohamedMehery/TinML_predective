# NOTES — Experiment 002 (EXP-C5C8-002)

> **Purpose.** This file is the lab notebook for Experiment 002. It records
> (a) the diagnostic findings that motivated the implementation roadmap,
> (b) the implementation gaps still open in the codebase, and (c) the
> decisions taken to resolve them, in chronological order.
>
> Per `04_experiments/README.md` rule 2, every result must pass the
> Critical Reviewer (`08_ai_agents/reviewer.md`) before being committed.
> This NOTES.md is the place to record the reviewer's findings.

## 1. Frozen protocol (as committed in `548098f`)

The frozen protocol is in `config/experiment_config.json` and the
ratified C5/C8 charter in `03_research_questions/candidate-questions.md`.
The pre-execution validation log is in `validation_log.txt`. All three
must be read together; the config is the machine-readable contract, the
charter is the scientific contract, the validation log is the
provenance.

### 1.1 Documented deviation from charter

The charter (§2) specifies *"Squared Hilbert Envelope"*. The frozen
config uses *Hilbert magnitude* (`envelope_extraction.squared = false`).
This deviation is recorded in the experiment README §6 and in
`validation_log.txt`. It is **not yet resolved**; see §3.1 below.

## 2. Diagnostic findings (2026-10-09, sandbox run)

A diagnostic copy of the original pipeline-extraction logic was run on
the pilot synthetic data (no commit, no push; diagnostic script lived
in `/tmp/diagnostic/`, never in the repo). The findings are the
motivating evidence for the implementation roadmap in §4.

### 2.1 What the diagnostic confirmed

| Check | Result | Implication |
|---|---|---|
| All 5 representation shapes match `experiment_config.json` | (11334, 1, 2048) for R1_*, (11334, 1024) for R2_* and R3_Order | Shape contract is correct. |
| NaN/inf count across all representations | 0 / 11334 windows in each of the 5 reps | Preprocessing does not silently produce non-finite values. |
| IDFT 1X tracker convergence on clean synthetic signals | 100 % of CWRU and PU windows, 0 % relative error | Tracker math is correct on clean signals. Does not test low-SNR (see §3.2). |
| LOBO prerequisite: unique physical-bearing-UIDs per dataset | CWRU: 7 (3-class) / 10 (4-class); PU: 13 (3-class) / 16 (4-class) | LOBO is feasible; ≥ 5 folds in both datasets. |

### 2.2 What the diagnostic surfaced (the open issues)

#### Issue A — 3-class vs 4-class contract

The original pipeline code used `CLASS_MAP = {Healthy: 0, Outer_Race: 1, Inner_Race: 2, Ball: 3}` with `num_classes=4`. The C5/C8 cross-machine claim is 3-class (Healthy / IR / OR; Ball is excluded from the cross-machine comparison because it is not present in SEU and is not aligned with the CWRU 3-class reporting in Vieira 2026; see `02_datasets/manifests/cwru.md` and the `[TO VERIFY FROM SOURCE DOC]` marker on the fault-collapse line).

The diagnostic showed 11,334 windows (3-class, Ball dropped) vs 14,592 windows (4-class). A 28.7 % delta. A 4-class F1 score is not comparable to a 3-class F1 score, so the contract has to be aligned before any number is published.

**Resolution:** `run_pipeline.py` must be 3-class by default. The 4-class path stays in the code as a `--include-ball` flag for internal ablation, but the headline `Δ Macro-F1` numbers are 3-class. See §4.1.

#### Issue B — Angular resample bin misalignment

The angular resample step uses `revs = max(1, int(window_size * f_speed_est / fs))` and produces `revs * 64` output samples. At CWRU's 12 kHz / 2048-sample window with the four recorded speeds (1797 / 1772 / 1750 / 1730 RPM), the rounded-down revolutions are 5 / 5 / 4 / 4, producing two distinct R3_Order output lengths: 320 (faster speeds) and 256 (slower speeds). PU's 64 kHz / 5 s records all produce 64 samples consistently because at 1500 / 900 RPM the round-down gives exactly 1 revolution per window.

Two adjacent R3_Order windows at different lengths are at different FFT bin spacings, so the *i-th* R3_Order bin at speed 1797 RPM does not correspond to the *i-th* R3_Order bin at speed 1730 RPM. The MLP can still learn per-window features, but the R3_Order representation is no longer pixel-aligned across speeds, which silently breaks the "same representation across the cross-machine transfer" assumption.

**Resolution:** Use a fixed-grid angular resample. The resample output length is fixed at 256 samples (4 revolutions × 64 samples/rev) for the CWRU 12 kHz case, with `revs` computed by rounding to the nearest integer and the time grid extending to the full window. See §4.2.

#### Issue C — No LOBO / no leakage prevention

The original diagnostic did not implement LOBO splitting on `physical_bearing_uid`. The C5/C8 charter pins `primary_unit = "physical_bearing_uid"` and the decision log (2026-10-07) explicitly rejects *"Random window splitting across same bearings"* as a leakage path. The 11,334 windows from 7 CWRU bearings are not 7 independent samples; they are 7 × ~464 correlated samples. Random splitting across these windows would inflate the F1 by an unknown amount.

**Resolution:** Implement strict LOBO on `physical_bearing_uid`. The test fold is the entire UID; the train fold is the remaining UIDs in the source dataset. The CWRU↔PU transfer is then: train on CWRU LOBO folds, test on the full PU target. See §4.3.

#### Issue D — Normalization was unspecified in the original code

The original diagnostic did not compute or apply any normalization. The charter pins `normalization = "source_train_zscore_only"`, which means: compute the per-channel mean and standard deviation on the **source-train fold only** (not on the union of source-train + source-test, not on the target's data), and apply that to every window the model sees. This is the standard cross-domain normalization rule; deviations silently leak target statistics into training.

**Resolution:** Implement source-train-only z-score. The mean and std are computed once per source-train fold, frozen, and applied to (a) source-train, (b) source-test, (c) all target windows. See §4.4.

#### Issue E — No multi-seed statistical evaluation

The original diagnostic did not train a model. The charter pins `random_seeds = [42, 1337, 2026, 7, 99]` (5 seeds), `cluster_bootstrap_iterations = 10000` on `physical_bearing_uid`, `confidence_interval_level = 0.95`, `effect_size_metric = "hedges_g"`, and `regression_model = "linear_mixed_effects"`. The "result" the experiment claims is a *distribution* of Macro-F1 across the 5 seeds × the 6 transfer pairs, with bootstrap CIs and Hedges' g, not a single number.

**Resolution:** The headline report is a CSV per (representation × transfer pair) of the form: `(mean_macro_f1, ci_lo, ci_hi, hedges_g_vs_R1_Raw, p_value_lme)` over the 5-seed distribution. The bootstrap is on `physical_bearing_uid`; the LME fits `macro_f1 ~ representation * transfer_pair` with random effect on `seed`. See §4.5.

#### Issue F — Pilot data is synthetic; results on pilot are not C5/C8 evidence

The pilot generates signals from a deterministic kinematic + structural-resonance model with the real CWRU/PU bearing geometry constants. The signals are **not** measurements. Any "result" on the pilot data tests whether the pipeline respects the kinematic-order decoupling **as designed into the generator**, not whether the C5/C8 hypothesis holds on real machinery. The pilot result is a pipeline smoke test, not an evidence point.

**Resolution:** Pilot results are written to `04_experiments/002_cross_machine_invariance/pilot/results/` with the `PILOT_` prefix. They are never merged with the real-experiment `results/`. The Critical Reviewer must reject any commit that cites pilot numbers as experiment results. (See `08_ai_agents/reviewer.md` line 1 — the "data leakage" question generalizes to "synthetic / real confusion".)

## 3. Implementation gaps still open

These are the gaps in the codebase that the diagnostic did not address. Each is tracked with a TBD status; the plan to close them is in §4.

### 3.1 Squared-vs-magnitude envelope (charter deviation)

- **Status**: open.
- **Owner**: tbd (proposed: a new entry in `00_lab/decision-log.md` ratifying the change, with the technical justification in §6 of `README.md`).
- **Why it's open**: the charter says Squared. The config says magnitude. The pre-execution validation flagged the deviation. The deviation is *not yet resolved* — only documented.
- **Plan**: when `src/preprocess.py` is written, implement the squared envelope in parallel and run a CWRU-subset ablation. If the squared form does not change transfer accuracy beyond noise, propose a 2026-10-XX entry in `00_lab/decision-log.md` formally amending the charter and updating `experiment_config.json` to `squared = true`. If it does change accuracy, change the config to `squared = true` and re-validate.

### 3.2 Pre-execution validation under low SNR

- **Status**: TBD.
- **Owner**: tbd.
- **Why it's open**: the IDFT 1X tracker check (`pre_execution_validation.py`) uses a clean synthetic signal. The real-data failure mode (per the decision log) is PU outer-race at SNR ≈ 3 dB. The current check does not exercise that case.
- **Plan**: add a `validate_speed_tracking_low_snr` check that adds Gaussian noise to the carrier-modulated test signal at SNR = 3 dB and asserts the tracker still converges within 5 % (relaxed from the 2 % clean threshold).

### 3.3 `Compact1DCNN_3Layer` vs the original `Compact1DCNN` architecture

- **Status**: TBD.
- **Owner**: tbd.
- **Why it's open**: the original `Compact1DCNN` in the diagnostic code has 3 conv layers with 16 / 32 / 64 channels (channels grow). The config's `backbone: "Compact1DCNN_3Layer"` does not pin channel widths. The `StandardMLP_2Layer` similarly is not pinned (2 hidden layers? 3? widths?).
- **Plan**: the config should grow a `models` section with explicit per-backbone shape contracts (channels, kernel sizes, activations, dropout, parameter count, RAM estimate on Cortex-M4F). Until then, the implementation must default to the simplest interpretation (3 conv layers, 2 MLP hidden layers) and document the choice in this NOTES.md.

### 3.4 Real-data download and `[TO VERIFY FROM SOURCE DOC]` markers

- **Status**: open.
- **Owner**: tbd (likely the Dataset Agent per `08_ai_agents/dataset-agent.md`).
- **Why it's open**: the per-dataset manifests (CWRU / PU / SEU) have 4 / 4 / 4 `[TO VERIFY FROM SOURCE DOC]` markers respectively. The 12 fields must be verified against the providers' own pages before `src/preprocess.py` is run on the real data.
- **Plan**: this is a manual step; not in scope for `run_pipeline.py`.

## 4. Implementation roadmap (the `src/` files, in dependency order)

The roadmap below is what `run_pipeline.py` and the supporting `src/` files must implement to close Issues A–F. Each item is small enough to be one commit; I will not bundle them.

### 4.1 3-class contract (closes Issue A)

`src/models.py`:
- `Compact1DCNN(num_classes=3)` as the default.
- `StandardMLP(in_features=1024, num_classes=3)` as the default.
- Both classes accept `num_classes` as a constructor arg for ablation; the pipeline always passes 3.

`src/datasets.py`:
- `CLASS_MAP_3 = {"Healthy": 0, "Outer_Race": 1, "Inner_Race": 2}`.
- A `drop_ball(records)` filter applied before windowing; Ball records are kept in the raw manifest for reference but never enter the model.

### 4.2 Fixed-grid angular resample (closes Issue B)

`src/preprocess.py`:
- The angular resample output length is **fixed at 256** for the 12 kHz CWRU case. For the 64 kHz PU case, the resample length is **fixed at 64** (1 revolution × 64 samples).
- `revs` is computed by rounding to the *nearest* integer, with the time grid extended to the full window. If the rounded `revs * samples_per_rev` would under-cover the window, the angular resample is computed on a slightly extended time grid (≤ 1 sample of `np.interp` extrapolation at the right edge) and the FFT input is trimmed back to `samples_per_rev` samples.
- The implementation is verified by an additional pre-execution check that asserts *all* R3_Order windows for a given dataset have the same output length (no 256-vs-320 split).

### 4.3 LOBO splitting on `physical_bearing_uid` (closes Issue C)

`src/datasets.py`:
- `lobo_split(records, dataset_name)` returns a list of `(train_uids, test_uid)` pairs, one per unique UID in the dataset. For CWRU (7 UIDs in 3-class) this is 7 folds; for PU (13 UIDs) this is 13 folds.
- The cross-machine transfer is then: for each source LOBO fold, train on the source-train UIDs, test on the **full target dataset** (all target UIDs). This is the standard "leave-one-source-bearing-out, evaluate on full target" protocol that the C5/C8 charter implies.

### 4.4 Source-train-only normalization (closes Issue D)

`src/datasets.py` or `src/preprocess.py`:
- `compute_zscore(features, train_mask)` returns `(mean, std)` over the train-fold rows only.
- `apply_zscore(features, mean, std)` is applied to source-train, source-test, and every target window using the **source-train** mean/std (frozen, never recomputed).
- This is the standard cross-domain rule. A unit test asserts that recomputing the mean/std on the union of source-train + target changes the numbers, so the leakage is detectable.

### 4.5 Multi-seed statistical evaluation (closes Issue E)

`src/train.py` and `src/evaluate.py`:
- For each (representation, transfer pair, seed) tuple, train one model and evaluate on the target. This is 5 representations × 6 transfer pairs × 5 seeds = 150 training runs per pilot.
- `evaluate.py` aggregates the per-run Macro-F1 into:
  - Per (representation, transfer pair): mean ± 95 % CI from a cluster bootstrap on `physical_bearing_uid` (n=10,000 per the config).
  - Per representation: Hedges' g vs R1_Raw (the data-driven spatial baseline) for each transfer pair.
  - LME: `macro_f1 ~ representation * transfer_pair + (1|seed)` fit on the 5-seed × 5-representation × 6-pair grid.
- The headline result CSV is at `results/transfer_matrix_summary.csv` (one row per (representation, transfer pair)). Pilot-only outputs go to `pilot/results/PILOT_*.csv`.

### 4.6 Pilot smoke test integration

`run_pipeline.py`:
- Top-level driver. Accepts `--source`, `--target`, `--representations`, `--seeds`, and a `--pilot` flag.
- In pilot mode, loads the synthetic data from `pilot/synthetic/raw_data/` (the `.npz` files are git-ignored, regenerated by the user) and writes results to `pilot/results/` with the `PILOT_` prefix.
- In real-data mode, loads the CWRU / PU / SEU `.npz` files from `02_datasets/raw_data/` (a path that does not yet exist; the user must populate it after clearing the `[TO VERIFY FROM SOURCE DOC]` markers) and writes to `results/`.
- The `--include-ball` flag enables 4-class mode for ablation; the default is 3-class.

## 5. Decision log

- **2026-10-09**: Diagnostic run confirms representation shapes, no NaN/inf, IDFT tracker convergence on clean signals, LOBO feasibility. Angular resample bin misalignment, 4-class contract bug, and missing LOBO/normalization/multi-seed evaluation surfaced. The implementation roadmap (§4) is the plan to close these in subsequent commits.
- **2026-10-09 (open)**: Squared-vs-magnitude envelope deviation. Recorded in `README.md` §6 and `validation_log.txt`. Resolution: see §3.1.
