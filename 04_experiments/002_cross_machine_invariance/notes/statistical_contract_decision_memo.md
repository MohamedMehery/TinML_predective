# Decision Memo — Statistical Contract for EXP-002 (DRAFT, awaiting research-lead approval)

**Status:** DRAFT — Not yet approved. Recorded for traceability, not as a decision.

**Date drafted:** 2026-10-12
**Phase:** 20 (evidence-driven gap closure, research audit, controlled repair)
**Author:** audit / pending research-lead approval

---

## 1. Question being decided

What is the precise statistical contract that supports the C5/C8 hypothesis test
("R3_Order maintains zero-shot cross-machine diagnostic accuracy with
Δ Macro-F1 ≤ 15%, whereas raw convolutional representations collapse by ≥ 40%")
in EXP-C5C8-002?

Sub-decisions:
- 1A. Is `R1_Raw` the formal baseline for the H0/H1 contrast?
- 1B. Is Hedges' g the appropriate effect-size metric, and if so, on what unit
  of analysis?
- 1C. Is a linear mixed-effects (LME) regression justified, and if so, with
  what specification?
- 1D. What are the exact acceptance criteria for the headline result?

## 2. Evidence currently in the repository

- `experiment_config.json` (`statistical_evaluation`):
  - `primary_unit = "physical_bearing_uid"` ✓
  - `cluster_bootstrap_iterations = 10000` ✓
  - `confidence_interval_level = 0.95` ✓
  - `effect_size_metric = "hedges_g"` (declared, **not yet implemented in evaluate.py**)
  - `regression_model = "linear_mixed_effects"` (declared, **not yet implemented**)

- `decision-log.md` 2026-10-11 (Phase 15 row): "(v) baseline NOT DECLARED".
- `NOTES.md` §4.5 (roadmap, not a ratified decision): "Per representation:
  Hedges' g vs R1_Raw ... LME: `macro_f1 ~ representation * transfer_pair + (1|seed)`".
- `README.md` §2 H0/H1: "Δ Macro-F1 ≤ 15%" (R3_Order) vs "≥ 40%" (raw conv).
- `src/evaluate.py`: implements only `macro_f1`, `per_uid_macro_f1`,
  `cluster_bootstrap_ci`. Does NOT implement Hedges' g or LME.
- `src/run_pipeline.py`: writes per-run CSV (5 reps × 7 folds × 5 seeds = 175
  rows) and per-uid CSV (2275 rows); does NOT compute Hedges' g or LME.

## 3. Data structure (verified empirically from per_run.csv)

- 5 representations × 7 source LOBO folds × 5 seeds = 175 runs per transfer pair.
- 7 distinct source test UIDs (CWRU: 3 IR + 1 Healthy + 3 OR).
- 13 distinct target UIDs (PU in pilot: 3 Healthy + 5 OR + 5 IR).
- Replication: 5 seeds nested within each of 7 folds.
- The natural unit of independent observation is `source_fold × seed` (n=35
  per rep per transfer pair) or `source_fold` alone (n=7 per rep per
  transfer pair).

## 4. Available alternatives and their consequences

### 4A. Baseline (R1_Raw vs alternatives)

- **Option A: R1_Raw (data-driven 1D-CNN time-domain)**
  - Pro: matches H0/H1 framing ("raw convolutional representations").
  - Pro: matches the 1D-CNN family in P16, P20 (the published 99.1% / 98.7%
    CWRU baselines, both single-domain).
  - Con: no published CWRU↔PU↔SEU cross-machine transfer number exists to
    use as a literature baseline.
  - Con: under the lab rule "no experiment without a pre-defined Baseline"
    (`04_experiments/README.md` line 14), R1_Raw is currently only an
    implicit comparison, not a ratified baseline.
- **Option B: Add a second baseline (chance, majority class, classical-ML)**
  - Pro: enriches comparison.
  - Con: expands scope; the C5/C8 charter pins 5 representations and
    compares them pairwise; adding a 6th element changes the design.
- **Option C: Use R1_Raw as primary baseline + report R1_Env and R2_Raw as
  secondary data-driven baselines (in the same 1D-CNN / MLP family)**
  - Pro: matches the actual 5-rep design (R1_Raw, R1_Env, R2_Raw, R2_Env,
    R3_Order) and treats data-driven reps as the reference family.
  - Pro: provides three independent contrasts for R3_Order instead of one.
  - Con: requires a decision about how to aggregate multi-baseline evidence.

**Recommendation:** Option C — R1_Raw is the primary baseline (the
1D-CNN time-domain rep; the strongest data-driven challenger per the H0
framing), with R1_Env and R2_Raw as secondary data-driven baselines for
robustness. **Requires research-lead approval** because the baseline
selection is a scientific decision.

### 4B. Effect-size metric (Hedges' g)

- **Option A: Hedges' g on the full 35 (fold, seed) runs per rep**
  - Pro: matches the existing `experiment_config.json` declaration.
  - Con: violates the assumption of independent observations — the 5 seeds
    within a fold are correlated (same train/test data, different model
    initialization only). The standard error of g will be underestimated.
  - Con: pseudo-replication inflates the apparent precision of the effect
    size.
- **Option B: Hedges' g on fold means (n=7 per rep per transfer pair)**
  - Pro: respects the hierarchical structure (5 seeds nested in 1 fold).
  - Con: n=7 is small; the 95 % CI of g will be wide; the effect-size power
    will be low.
  - Con: only 7 independent observations per (rep, transfer pair) — the
    headline claim would rely on very few independent points.
- **Option C: Paired Hedges' g (g_av or g_z) on within-block differences**
  - Pro: respects the (fold, seed) block structure; treats each block as a
    paired comparison between R3_Order and R1_Raw.
  - Pro: standard in cross-over and within-subject designs; recommended in
    Cumming (2012) "Understanding the New Statistics".
  - Pro: n_blocks = 35 (5 seeds × 7 folds), giving reasonable power.
  - Con: not what `experiment_config.json` literally says. Requires a
    protocol amendment to the JSON.

**Recommendation:** Option C — paired Hedges' g (g_z) on within-block
differences for the R3_Order vs R1_Raw contrast. **Requires research-lead
approval** to amend the statistical contract in `experiment_config.json`.

### 4C. Linear mixed-effects (LME)

- **Option A: LME `macro_f1 ~ representation * transfer_pair + (1|seed)`**
  - Pro: matches NOTES §4.5.
  - Con: with 5 seeds, the random intercept on `seed` has only 5 levels.
    Convergence is fragile; the variance component may be poorly identified
    (boundary issues).
  - Con: n=175 total observations; the LME has 5×6=30 fixed-effect cells
    (representation × transfer pair) — each cell has only ~6 observations.
    The interaction effect (5 levels × 6 levels = 30 cells) is grossly
    underpowered.
  - Con: residuals of Macro-F1 (bounded [0, 1]) likely violate the
    normality assumption; a beta regression or quasi-binomial GLMM would be
    more defensible.
- **Option B: LME with `(1|fold)` or `(1|fold) + (1|seed)`**
  - Pro: more defensible hierarchical structure.
  - Con: still underpowered (n_folds=7, n_seeds=5); the interaction
    (5×6=30 cells) cannot be reliably estimated.
- **Option C: No LME; report bootstrap CI + paired g_z + descriptive
  per-(rep, transfer-pair) table**
  - Pro: matches the actual data structure; honest about power.
  - Pro: simpler to interpret; matches the C5/C8 headline framing
    (Δ Macro-F1 with 95 % CI).
  - Con: drops the LME that `experiment_config.json` declares.

**Recommendation:** Option C for the headline result, with paired g_z as
the secondary effect-size statistic. The LME declaration in
`experiment_config.json` should be amended to reflect this. **Requires
research-lead approval** to amend the contract.

### 4D. Acceptance criteria for the H1 claim

- The charter (README §2) says R3_Order should achieve Δ Macro-F1 ≤ 15% on
  cross-machine transfer, vs ≥ 40% for raw conv.
- The numerical Δ in the pilot summary is the difference between the source
  domain and the target domain Macro-F1 — but the pilot does NOT include a
  source-domain F1 number (the metric is computed only on the target).
  **The Δ formulation is not directly computable from the current outputs.**
- To make the Δ computable, the pipeline would need a source-domain F1
  per (rep, fold, seed) — the same 175-run structure as the target.
  Currently `run_pipeline.py` does not emit source-domain F1.

**Recommendation:** add a source-domain F1 to the per-run CSV. This is
a **technical / unambiguous implementation addition** (it does not change
the scientific question) and can be done as a small extension to
`run_pipeline.py`. **Not a research-lead decision; just a code
completion.** Recorded in §6 of this memo.

## 5. Evidence supporting each recommendation

- Option C for baseline: P16 (El Boughardini 2026) 1D-CNN 99.1 % on
  CWRU/PU/MFPT single-domain; P20 (Cotrino 2026) 1D-CNN 98.7 % on CWRU.
  These are the strongest published single-domain baselines; neither is a
  cross-machine transfer number, so the experiment's R1_Raw-in-the-loop is
  the only defensible cross-machine baseline.
- Option C for effect size: Cumming (2012) "Understanding the New
  Statistics" — within-block paired designs are the standard for
  pre-registered replication studies. (Paper not in repo's paper matrix;
  this is a methodological reference, not a domain one.)
- Option C for LME: Gelman & Hill (2007) "Data Analysis Using Regression
  and Multilevel/Hierarchical Models" — LME with 5 random-effect levels
  is widely considered underpowered for variance-component estimation.
  (Same status as Cumming 2012 — methodological, not domain.)
- Option for source-domain F1: NOTES.md §2.2 Issue E + the existing
  per-run CSV schema trivially accommodates an additional column.

## 6. Specific changes proposed (subject to research-lead approval)

1. Amend `experiment_config.json` `statistical_evaluation` block:
   - Keep `primary_unit`, `cluster_bootstrap_iterations`, `confidence_interval_level`.
   - Change `effect_size_metric` from `"hedges_g"` to
     `"paired_hedges_g_on_block_differences"` (or equivalent; exact wording
     to be ratified).
   - Change `regression_model` from `"linear_mixed_effects"` to
     `"none_for_headline_use_bootstrap_ci_and_paired_g_z"` (or equivalent).
2. Add a source-domain F1 to the per-run CSV in `run_pipeline.py`. This is
   a small extension (one more metric per run) and does not change the
   scientific question. **Can be done in this phase as a category-2 repair
   if the research lead approves the statistical-contract amendment, OR
   as a category-3 (provenance) repair documenting that the source-domain
   F1 was not collected in the pilot and must be added before real-data.**
3. The current `evaluate.py` does not implement paired g_z. Adding it
   requires either:
   - A new `paired_hedges_g` function in `evaluate.py` (pure-Python,
     ~30 lines, no new dependencies).
   - Or an R script using `effsize` or `compute.es` package.
   The Python implementation is preferred for reproducibility. **Can be
   done in this phase as a category-2 repair** (it implements the amended
   contract; it does not silently change the design).

## 7. Required follow-up and closure evidence

- Research-lead decision on §1A, §1B, §1C, §1D.
- If approved: implement the paired g_z function in `evaluate.py`, add
  unit tests with known inputs/outputs, add source-domain F1 to per-run
  CSV, update `experiment_config.json`.
- Re-run the pilot aggregation with the new statistics; confirm that the
  paired g_z for R3_Order vs R1_Raw has reasonable precision and that the
  headline CIs are consistent with the per-fold bootstrap CIs already
  reported.

## 8. Limitations

- With 7 folds and 5 seeds, the pilot's effect-size precision is limited
  even with the corrected paired analysis. The real-data experiment, with
  more folds (PU has 13, SEU has more), will have higher precision.
- The pilot is **synthetic**. The C5/C8 hypothesis is about real cross-
  machine transfer. Statistical findings on the pilot only validate the
  pipeline shape, not the physics.
- The C5/C8 H0/H1 framing (Δ ≤ 15% vs ≥ 40%) is operationalized as the
  per-(rep, transfer pair) mean F1 on the target, but the Δ formulation
  in the charter assumes a source-domain reference number. The source-
  domain F1 must be added before the Δ can be computed as stated.

---

**End of decision memo. Status: DRAFT, awaiting research-lead approval.**
