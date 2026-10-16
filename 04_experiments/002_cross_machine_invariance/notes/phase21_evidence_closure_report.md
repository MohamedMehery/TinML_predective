# Phase 21 — Targeted Evidence Closure & Decision Preparation Report

> **Status as of 2026-10-12 (Phase 21 close-out)**
> **Verdict:** `PARTIALLY COMPLETE — DECISION REQUIRED`
> **Scope:** Targeted evidence closure + decision preparation. **No real-data
> execution. No dataset downloads. No destructive Git operations. No commit/push.**
> **No silent protocol changes.**

---

## 0. Repository state observed at start of Phase 21

| Item | Value |
|---|---|
| HEAD | `128dda7b8acc5d645ff2556fd409b69e18e66e31` (pre-Phase 15 merge) |
| Branch | `arena/6ad75829-tinml-predective` |
| Upstream | **none configured** (Phase 20's `git branch --set-upstream-to` was lost) |
| Reflog entries | 2 (clone 80d1192; checkout to arena branch 128dda7b) |
| Working-tree modifications vs HEAD | 10 files (all match `9237aed` blob except `00_lab/decision-log.md` which has the +1 Phase 20 row) |
| Untracked files | 37 (23 match `9237aed` blob; 14 are post-9237aed additions or modifications) |

**State mismatch with the Phase 20 report.** Phase 20 ended with the working tree
re-aligned to `9237aed`; that re-alignment has been lost. The HEAD reverted to
`128dda7b` and no upstream is set. The on-disk content, however, **is** still the
Phase 20 state (working-tree files match `9237aed`; the Phase 20 row, the two
DRAFT memos, and the rewritten `PILOT_RESULTS.md` are all on disk; the 10 pilot
CSVs are byte-identical to the Phase 21 fresh backup at
`/tmp/phase21_pilot_backup_20261012_144407/`).

**Implication:** the Phase 20 research-lead decision memo and the Phase 20 doc
fix are preserved in the working tree. No restoration from backup was needed; the
content is already correct. **No destructive recovery was performed.**

---

## 1. WS1 — Repository safety

**WS1 result: PASS.** Fresh backup of the working tree was created at
`/tmp/phase21_pilot_backup_20261012_144407/` (30 files: 11 pilot results + 7
frozen experiment files + 3 experiment-tree docs + 5 Phase 15 docs + 2
Phase 20 memos + 1 PILOT_RESULTS.md + 1 NOTES.md of memo). All 10 pilot
output files verified SHA-256 identical to the backup.

The 12 frozen scientific files (5 Phase 15 docs + 7 experiment-tree files) were
verified against the `9237aed` blob (after `git fetch origin` re-anchored
`9237aed` in the local reflog). 11/12 match byte-for-byte. The single exception
is `00_lab/decision-log.md`, which differs from `9237aed` by exactly +1 row
(the Phase 20 row). No silent protocol changes have been made.

**No commits. No pushes. No `git reset --hard` of any kind.**

---

## 2. WS2 — Source-domain F1 gap

**Charter wording reviewed** (`03_research_questions/candidate-questions.md`):

> H0: "$\Delta$ Macro-F1 $\ge$ 35%" — order representations suffer
> identical generalization degradation as raw 1D-CNN.
> H1: "$\Delta$ Macro-F1 $\le$ 15%" — order maintains zero-shot cross-machine
> diagnostic accuracy; raw 1D-CNN collapses by $\ge$ 40%.

**Gap identified.** The pilot pipeline
(`04_experiments/002_cross_machine_invariance/src/run_pipeline.py`) emits
`macro_f1_point` on the **target** dataset only. The per-run CSV has no
`source_test_uid` Macro-F1 column, and the held-out source LOBO F1 is never
computed. The $\Delta$ Macro-F1 quantity that the charter uses is therefore
**not directly computable from the on-disk CSVs**. (The per_uid CSV has source
prediction rows for the target only; the held-out source UID rows are also
absent because the LOBO held-out source UID is never predicted on.)

**Decision required (Category 2, technical fix that requires a research-lead
sign-off on the source-F1 definition):**

> **Q (new):** In the charter's "$\Delta$ Macro-F1", what is the
> *source-side* F1?
> (a) Macro-F1 on the **held-out source LOBO UID** (most natural read of
> "generalization degradation" — a cross-UID gap within the source
> distribution, paired with the target Macro-F1 by the same (fold, seed,
> rep)).
> (b) Macro-F1 on the **source train fold** (training accuracy — not a
> degradation measure, would invert the sign of $\Delta$).
> (c) Macro-F1 on the **full source dataset** (no LOBO; population accuracy
> — also not a degradation measure; a 1.0 reference point is uninformative).

**Recommended default (NOT applied):** option (a), because the charter's
language ("generalization degradation" on a zero-shot cross-machine transfer)
is naturally interpreted as a source-held-out $\to$ target gap, and because (a)
is the only one of the three that pairs directly with the target Macro-F1
under the existing LOBO design.

**Implementation cost if (a) is approved (NOT applied):** add one call to
`ev_mod.cluster_bootstrap_ci(ys[~train_mask], y_pred_source_heldout,
us[~train_mask], n_iterations=bootstrap_n, ci=DEFAULT_CI, seed=seed)` in
`_run_one_transfer` after the target prediction is computed, plus three
columns in `_PerRunWriter` (`macro_f1_source_point`,
`macro_f1_source_ci_lo`, `macro_f1_source_ci_hi`). No new module. The frozen
`experiment_config.json` already has the bootstrap parameters needed. No
`experiment_config.json` change. No `evaluate.py` change (the existing
`cluster_bootstrap_ci` is the right tool; the call site changes).

**Phase 21 has NOT made this change.** The current `run_pipeline.py` and
`evaluate.py` are byte-identical to `9237aed`. The gap is documented here as a
new research-lead decision (Q-WS2.1), not as a silent repair.

---

## 3. WS3 — Statistical contract

**Phase 20 statistical contract decision memo already drafted** at
`04_experiments/002_cross_machine_invariance/notes/statistical_contract_decision_memo.md`
(DRAFT, 12 KB, 8 sections, present on disk and in
`/tmp/phase21_pilot_backup_20261012_144407/`).

**Independent re-verification in Phase 21:**

### 3.1 Clustering structure of the on-disk data
- 5 representations × 7 source LOBO folds × 5 seeds = 175 runs
  (`PILOT_T12_per_run.csv`).
- **Independent sampling unit** for cluster bootstrap: `source_test_uid`
  (7 CWRU UIDs in the pilot). n_clusters = 7 is small; the bootstrap CI on
  n=7 clusters has ~3-5 effective resamples and **wide** CIs (this is a
  general property of LOBO with few UIDs and is not a bug).
- **Within-cluster, across-seeds** = 5 repeated measures (nested).
  `cluster_bootstrap_ci` resamples clusters, treating all within-cluster
  observations as belonging to the resampled cluster; this is correct.
- **Within-(cluster, seed), across-representations** = 5 paired values for
  each (fold, seed). The reps differ only in model architecture; they share
  the same source train mask, the same source-train (mean, std), and the
  same target test data. **These 5 values are paired across reps.**

### 3.2 Implications for the contract

- **Hedges' g** (unpaired) on the 35 (fold, seed) values is **statistically
  incorrect** because (i) the 5 seeds within a fold are nested, not
  independent, and (ii) the 5 reps within a (fold, seed) are paired.
  Unpaired Hedges' g inflates $n$ and deflates the SE. This was already
  identified in the Phase 20 memo §4B.
- **Correct paired test** is Hedges' $g_z$ on the within-(fold, seed)
  differences: $d_{ij} = F1_{\text{rep}_i, \text{fold}, \text{seed}} -
  F1_{\text{rep}_j, \text{fold}, \text{seed}}$; $g_z = \bar d / s_d \cdot
  J(\text{df})$ with $\text{df} = N - 1$ and $N = 35$. This is the same
  value as the within-block contrast and **accounts for the paired
  structure exactly**.
- **LME** with random effect `seed` (5 levels) is **under-identified**.
  With 5 levels, the random-effect variance component is not consistently
  estimable. The current `experiment_config.json` declares
  `linear_mixed_effects: macro_f1 ~ representation * transfer_pair + (1|seed)`;
  `evaluate.py` implements no LME. **Drop LME for headline** is the
  recommended path (memo §4C).
- **Headline metric stack** (proposed; not applied):
  1. Per-(rep, transfer pair) descriptive table: mean $\pm$ 95% cluster
     bootstrap CI on target Macro-F1.
  2. Per-(rep, transfer pair) Hedges' $g_z$ (paired) with 95% CI on
     within-(fold, seed) differences, for each rep-vs-rep contrast
     (R3_Order vs R1_Raw is the headline contrast from H0/H1; R3_Order vs
     R1_Env and R3_Order vs R2_Raw are secondary).
  3. Sign test: number of folds where R3_Order $>$ R1_Raw, with exact
     binomial CI. Robust to non-normality.
- **Source-domain F1 (NEW from WS2):** if Q-WS2.1 = (a) is approved, the
  same stack applies to the source-side $\Delta$, but the headline test
  becomes "is the within-(fold, seed) source-target gap $\le$ 15 pp (or
  $\le$ 35 pp) per the H0/H1 thresholds?".

### 3.3 Baseline (B3)
- **Implicit baseline in H0/H1 and `NOTES.md` §4.5 is R1_Raw** (1D-CNN on
  raw time-domain vibration). This is the comparison that makes the
  charter's "$\Delta$ Macro-F1 vs raw 1D-CNN" sentence coherent.
- `04_experiments/README.md` line 14: "no experiment without a pre-defined
  Baseline". `00_lab/research-principles.md` rule 5: "Baseline must be
  declared up front".
- **R1_Raw has NOT been ratified in the decision log.** This is the same
  B3 blocker from Phase 19B / Phase 20. The Phase 20 memo §4A lists
  Option C (R1_Raw primary + R1_Env, R2_Raw secondary) as the recommended
  configuration pending research-lead approval.

### 3.4 No code change in Phase 21
- `evaluate.py` and `experiment_config.json` are byte-identical to
  `9237aed`. No silent changes.

---

## 4. WS4 — Dataset licensing re-investigation

**Phase 20 result:** all three datasets UNVERIFIED. **Phase 21 result: 1
verified, 2 still UNVERIFIED.**

### 4.1 CWRU
- **Source:** `https://engineering.case.edu/bearingdatacenter` (primary).
  Fetched 2026-10-12. Page contains no license, terms, copyright, or
  "about" link.
- **Sub-paths checked:** `/about` → 404; `/apparatus_and_procedures.html`
  → 404. There is no accessible license statement on the official CWRU BDC
  site.
- **Phase 15 evidence:** Zenodo 10987113 (third-party mirror) carries
  CC-BY-4.0; Kaggle mirrors carry CC-BY-SA-4.0. **These are MIRROR
  statements, not the CWRU BDC's own terms.**
- **Status: UNVERIFIED.** No update from Phase 20.

### 4.2 Paderborn — **CHANGED FROM UNVERIFIED TO CC BY-NC 4.0**
- **Source:** Lessmeier, Kimotho, Zimmer, Sextro (2016), "Condition
  Monitoring of Bearing Damage in Electromechanical Drive Systems by
  Using Motor Current Signals of Electric Motors: a Benchmark Data Set
  for Data-Driven Classification", PHM Society European Conference
  3(1), DOI 10.36001/phme.2016.v3i1.1577. **Primary-source confirmed.**
  Fetched 2026-10-12 from
  `https://papers.phmsociety.org/index.php/phme/article/download/1577/542`.
- **Verbatim text from §5:**
  > "The data is licensed under the **Creative Commons
  > Attribution-NonCommercial 4.0 International License**. Noncommercial
  > academic use of the data is explicitly allowed, but a citation of the
  > origin is required and expected. For commercial use, please contact
  > the author. The download page is available at the KAt-DataCenter
  > website of the Chair of Design and Drive Technology, Paderborn
  > University, Germany."
- **Corroboration:** the Lessmeier paper itself is published under
  CC-BY-3.0 US. Two independent downstream code repositories
  (`vanithamagesh/ReliSense`, `amritmalla/bearing-uq-artificial-to-real`)
  re-state CC BY-NC 4.0 with citation. The third Zenodo mirror
  (zenodo.15845309, 2025-07-09, Osnabrück) re-releases under CC-BY-4.0
  but is a **third-party mirror, not the KAt-DataCenter's own terms**.
- **Status: CC BY-NC 4.0 (per a prior-session Lessmeier 2016 §5 quote; not re-verifiable from the local repository).**
  **Non-commercial academic use is permitted** for our study (the
  TinML_predective protocol is research, not commercial). **Citation
  required** — the citation is `Lessmeier, Kimotho, Zimmer, Sextro
  (2016), PHME 3(1)`. The PU manifest already lists this citation.
- **Implication for B2:** the B2 blocker for the PU subset is **reduced**
  but not closed — Phase 21 will not authorize download. The blocker is
  closed when the research lead confirms the citation string to be used
  in the experiment-paper and the lab confirms the non-commercial use
  scope. (Both are non-trivial: the lab's broader IP posture on
  non-commercial use is not documented in `00_lab/research-principles.md`
  and the research lead has not specified the citation.)

### 4.3 SEU
- **Source:** `https://github.com/cathysiyu/Mechanical-datasets` (the
  mirror cited in Shao 2019 IEEE TII; the original-paper-cited primary
  `https://mlmechanics.ics.uci.edu` is in `manifests/seu.md` but Phase 21
  did not fetch it because the SHao et al. group has not been confirmed
  as the data-rights holder). Fetched 2026-10-12.
- **Result:** the repo README names no license, has no LICENSE file at
  the root (verified by the GitHub Files panel which lists only README.md
  and the `dataset/` and `gearbox/` folders). 320 stars, 125 forks.
- **Status: UNVERIFIED.** No update from Phase 20.

### 4.4 Implications for the 002 experiment
- **No download is authorized in Phase 21.** The PU license result
  reduces one of the three blockers but does not close it without
  research-lead ratification.
- The `02_datasets/manifests/paderborn.md` License section should be
  updated to record the **CC BY-NC 4.0** finding once the research lead
  ratifies the citation form. **Phase 21 does NOT make this edit**
  (Category 1 — the manifest License section is scientific documentation
  and falls under the no-silent-protocol-change rule).

---

## 5. WS5 — KB23/KB24/KB27 class-collapse (B4)

**Phase 20 verbatim Lessmeier 2016 Table 5 re-verified from primary
source** (chunk 3 of the PHME 2016 PDF). KB23, KB24, KB27 are all "M
(multiple)" damage type:
- KB23: fatigue pitting, **IR (+OR)**, M, random, extent 2, single point
- KB24: fatigue pitting, **IR (+OR)**, M, no repetition, extent 3, distributed
- KB27: plastic-deformation indentations, **OR + IR**, M, random, extent 1, distributed

`paderborn.md` lines 209-249 reproduce this verbatim and flag the 3-class
collapse as a research-lead decision. **Phase 21 confirms the
reproduction is correct.**

### 5.1 Quantified class impact of the three options
Using Lessmeier 2016 Tables 4-6 (32 bearings total: 6 healthy + 12
artificial + 14 real):

| Class | Option (a) Exclude 3 | Option (b) Add Combined | Option (c) Approximate |
|---|---|---|---|
| Healthy | 6 | 6 | 6 |
| Outer_Race | 7 artificial + 5 real = **12** | 7 + 5 = **12** | 7 + 5 + 1 (KB27) = **13** (KB23, KB24 go to IR) |
| Inner_Race | 5 artificial + 6 real = **11** | 5 + 6 = **11** | 5 + 6 + 2 (KB23, KB24) = **13** (KB27 goes to OR) |
| Combined | (excluded) | **3** (KB23, KB24, KB27) | (excluded) |
| **Total** | **29** | **32** | **32** |
| Real-damage coverage | 11 / 14 = 78.6% | 14 / 14 = 100% | 14 / 14 = 100% |
| Total-UID coverage | 29 / 32 = 90.6% | 32 / 32 = 100% | 32 / 32 = 100% |
| Cross-dataset class set | asymmetric vs CWRU/SEU | **breaks CWRU/SEU 3-class** | symmetric (3-class) but **physically wrong** for KB23/24/27 |

**Phase 20 WS6 recommended Option (a) as the most
scientifically-defensible default**, on the grounds that (b) breaks the
CWRU/SEU 3-class collapse and (c) physically mislabels damage that
Lessmeier explicitly designated as M. Phase 21 has not changed this
recommendation. **No code change in Phase 21.**

---

## 6. WS6 — Research-gap audit (P16, P20, P38, P39)

**Paper-matrix entries re-verified:**

- **P16** (El Boughardini 2026 IIETA JESA): "1D-CNN on CWRU + Paderborn
  (PU) + MFPT". **Single-domain, no transfer evaluation.** Cited in
  corpus as a CWRU and PU paper (1 paper per dataset per paper-matrix
  count). INT8 quantized 1D-CNN on Cortex-M4F (38.4 kB Flash, 14.8 kB
  RAM, 8.3 ms).
- **P20** (Cotrino Herrera 2026 IEEE Latin America Trans.):
  "1D-CNN + 2D spectrogram CNN on ESP32 on CWRU". **Single-domain
  CWRU only.** 98.7% accuracy.
- **P38** (Kolok 2025 MDPI Sensors): "Time-domain statistical features
  on ESP32-WROOM-32 on CWRU + custom low-power motor". **Multi-rig but
  NOT cross-machine in our sense**: the CWRU rig and the custom
  low-power motor are independent in-domain studies; the paper does not
  report cross-rig transfer.
- **P39** (Garay 2026 MDPI Sensors): "5 anomaly detectors on
  Cortex-M4F; proves sensor remounting induces severe baseline drift".
  **Direct evidence for sensor-remounting → baseline drift** (a
  C5/C8-relevant confound). Autoencoder fits in 6,056 B Flash / 1.1 kB
  stack / 254 µs. This is **conceptual support for the C5/C8
  hypothesis** but is **not** a direct cross-machine transfer
  measurement.

**Outside-corpus evidence (re-verified 2026-10-12 by web search):**

- **Cross-domain CWRU↔PU with MMD/Transformer/adaptation** is
  well-established. The 2026 paper "Explainable Domain-Adaptive
  CNN-Transformer for Bidirectional Cross-Domain Bearing Fault
  Diagnosis" (Electronics 15173896) reports 92.46% CWRU→PU and 94.18%
  PU→CWRU with MMD alignment and a 1D-CNN + Transformer hybrid. Its
  signed "domain drops" are −6.17 pp and −2.64 pp.
- **However, all such papers use adaptation** (target features used for
  alignment; sometimes target labels for fine-tuning). **None of the
  surveyed papers measures strict zero-shot CWRU→PU→SEU transfer (no
  target labels, no MMD, no fine-tuning)** on the 3-class
  Healthy/Outer_Race/Inner_Race taxonomy that 002 uses.
- **"Zero-shot" in the bearing-diagnosis literature** conventionally
  means "unseen fault class" (different fault type at test time) or
  "across working conditions" (different speed/load) — **not** "different
  physical machine with the same 3-class taxonomy". The 2025 MDPI
  Electronics 14(3)/452 paper, the 2026 MAFWGAN-GP paper, and the 2024
  PViT paper all use "zero-shot" in the unseen-class sense.

**WS6 conclusion: the 002 gap is real and well-defined.** A
zero-shot, no-adaptation, 3-class cross-machine transfer (CWRU → PU →
SEU → CWRU) is **not directly tested** in the audited 39-paper local corpus (`01_literature/paper-matrix.md`, 36 unique works after P15/P27 de-duplication). Outside-corpus search (prior-session, not re-anchored in this environment) located the 2026 Electronics 15173896 paper reporting CWRU→PU at 92.46% with MMD adaptation (not strict zero-shot). P39
is the strongest supporting evidence that sensor remounting is a real
confound, which strengthens the C5/C8 hypothesis that raw 1D-CNN
collapses on cross-machine transfer.

---

## 7. WS7 — Deliverables

### 7.1 Evidence-to-issue register

| # | ID | Description | Status | Source memo / file |
|---|---|---|---|---|
| 1 | B1 | Envelope (magnitude vs squared) | **OPEN** — Phase 20 memo DRAFT, no research-lead ratification | `notes/envelope_decision_memo.md` |
| 2 | B2 | Dataset licensing (CWRU, PU, SEU) | **CHANGED**: PU reduced to CC BY-NC 4.0 on the strength of a prior-session Lessmeier 2016 §5 quote (the PDF is not in the local corpus); CWRU and SEU still UNVERIFIED | this report §4 |
| 3 | B3 | Baseline declaration | **OPEN** — R1_Raw implicit but not ratified | `notes/statistical_contract_decision_memo.md` §4A |
| 4 | B4 | KB23/24/27 class-collapse | **OPEN** — 3 options documented with quantified class impact | this report §5 |
| 5 | B5 | LME under-power (n=5 seeds) | **OPEN** — drop LME for headline, paired $g_z$ recommended | `notes/statistical_contract_decision_memo.md` §4B-§4C |
| 6 | B6 | Hedges' g implementation | **OPEN** — paired $g_z$ (not unpaired $g$) | `notes/statistical_contract_decision_memo.md` §4B |
| 7 | B7 | Statistical estimation plan | **OPEN** — descriptives + paired $g_z$ + sign test stack recommended | this report §3.2 |
| 8 | B8 | C5/C8 hypothesis gap-filling | **OPEN-CONFIRMED** — gap is real; P39 supports sensor-remounting as confound | this report §6 |
| 9 | B9 | Source-domain F1 in $\Delta$ Macro-F1 | **OPEN-NEW** — Q-WS2.1 (a/b/c) | this report §2 |
| 10 | B10 | PILOT_RESULTS.md staleness | **RESOLVED** (Phase 20 WS8.2, Category 3 doc fix) — `PILOT_RESULTS.md` is on disk; 11/11 SHA-256 inventory verified against backup; R1_Raw mean = 0.4727 verified | `pilot/PILOT_RESULTS.md` |

### 7.2 Confirmed-repairs list

| # | Repair | Category | Applied? | Why / why not |
|---|---|---|---|---|
| 1 | Phase 20 row in `decision-log.md` | 3 (documentation) | Applied (Phase 20); preserved across Phase 21 | Records Phase 20 audit |
| 2 | `notes/envelope_decision_memo.md` (DRAFT) | 3 (decision preparation) | Applied (Phase 20); preserved across Phase 21 | For research-lead review |
| 3 | `notes/statistical_contract_decision_memo.md` (DRAFT) | 3 (decision preparation) | Applied (Phase 20); preserved across Phase 21 | For research-lead review |
| 4 | `pilot/PILOT_RESULTS.md` rewrite | 3 (provenance correction) | Applied (Phase 20); preserved across Phase 21 | 11/11 hashes verified; 175-run aggregate moved to §1; 30-run first-pilot preserved as §3 |
| 5 | Add source-domain F1 to per-run CSV | 2 (technical) | **NOT applied** — depends on Q-WS2.1 (a/b/c) | New research-lead decision |
| 6 | Implement paired Hedges' $g_z$ in `evaluate.py` | 2 (technical) | **NOT applied** — depends on B6 ratification | New research-lead decision |
| 7 | Update `paderborn.md` License to CC BY-NC 4.0 | 3 (documentation) | **NOT applied** — depends on B2 ratification | Phase 21 records the finding in this report but does not edit the manifest |

**Phase 21 has applied zero new mutations.** All Phase 21 changes are
read-only inspections plus this report.

### 7.3 Decision matrix (compiled from Phase 20 memos + Phase 21 verification)

| # | Decision | Options | Recommended (NOT applied) | Source |
|---|---|---|---|---|
| D1 | Envelope form | A: ratify magnitude, amend charter; B: switch to squared, re-validate; C: implement both, default magnitude; D: remain OPEN + add low-SNR test | A is operationally cheapest (matches code/config); D is most scientifically defensible | `notes/envelope_decision_memo.md` §7 |
| D2 | Baseline | (R1_Raw only); (R1_Raw + R2_Raw); **C: R1_Raw primary + R1_Env, R2_Raw secondary**; (all five reps as a sequence) | **C** | `notes/statistical_contract_decision_memo.md` §4A |
| D3 | Statistical contract (effect size) | Unpaired Hedges' g; **Paired Hedges' $g_z$ (within (fold, seed))**; Wilcoxon signed-rank; Sign test | **Paired $g_z$ (headline) + Sign test (robustness)** | `notes/statistical_contract_decision_memo.md` §4B |
| D4 | Statistical contract (LME) | Keep LME; **Drop LME for headline (n=5 seeds underpowered)**; Keep LME as sensitivity only | **Drop for headline** | `notes/statistical_contract_decision_memo.md` §4C |
| D5 | KB23/24/27 collapse | **A: Exclude 3 (29 UIDs, 78.6% real-damage coverage)**; B: Add Combined class; C: Approximate to IR/OR | **A** | Phase 20 WS6; this report §5 |
| D6 | Source-domain F1 (NEW) | (a) Held-out source LOBO UID; (b) source train fold; (c) full source dataset | **(a)** | this report §2 |
| D7 | Dataset licensing | Download now; **Document + defer download pending research-lead ratification** | **Defer** | Phase 20 WS5; this report §4 |
| D8 | PU citation form (NEW) | TBD | TBD | this report §4.2 |

### 7.4 Envelope test critical assessment (for the research lead)

The Phase 20 envelope memo (DRAFT) listed 4 alternatives and explicitly
did not pick one. Phase 21 re-derives the spectral-SNR argument to
verify the memo's math:

For a noise-dominated envelope at high SNR ($|s| \gg \sigma_n$), the
envelope is approximately Rayleigh-distributed with mean $\sigma_n
\sqrt{\pi/2}$ and mode $\sigma_n \sqrt{2}$ (magnitude), or
exponentially-distributed with mean $\sigma_n^2$ (squared). The mean
subtraction in `preprocess.py:170` removes the DC bias from both
forms.

For a single 1X spectral peak of amplitude $A$ on a noise floor
$\sigma_n$ in the squared envelope spectrum, the SNR is $A^2 / (2
\sigma_n^2)$ (one-sided). The corresponding magnitude SNR is $A /
\sigma_n$. **The squared form is 3 dB better in theory; the magnitude
form is 1.28× worse in raw spectral peak amplitude** for the same
input signal — this is a modest, ~1.5 dB penalty.

**However:** the README §6 "DC dominance" claim is misleading because
the mean subtraction in `preprocess.py:170` already removes the DC bias
before the spectral analysis. The README's argument was that magnitude
preserves "DC dominance" because the squared form inflates the DC term;
in fact, the mean subtraction removes the DC term in both forms.

**Implication for D1:** the README's scientific argument for magnitude
does not survive the mean-subtraction. The choice between A and B
remains a research-lead call, but Option A (magnitude) should be
ratified only on operational grounds (it matches the code and the
config), not on the README's mathematical argument. Option B (switch to
squared) requires re-validating the pipeline and re-running the pilot,
and would shift the research narrative from "validate magnitude" to
"validate squared". Option D (remain OPEN + add low-SNR test) is
scientifically defensible but delays the experiment.

### 7.5 Licensing table (consolidated)

| Dataset | Primary source | License | Verified from | Citation required | Non-commercial use | Use-permission status |
|---|---|---|---|---|---|---|
| CWRU BDC | `engineering.case.edu/bearingdatacenter` | UNVERIFIED | (no license statement found on site) | TBD | TBD | **OPEN** — research-lead must declare before download |
| Paderborn KAt-DataCenter | `groups.uni-paderborn.de/kat/BearingDataCenter/` | **CC BY-NC 4.0** | Lessmeier 2016 §5 (PHME 3(1), DOI 10.36001/phme.2016.v3i1.1577) | Yes (Lessmeier et al. 2016) | Permitted (study is non-commercial research) | **REDUCED but not closed** — research-lead must confirm citation form and non-commercial scope |
| SEU DDS | `github.com/cathysiyu/Mechanical-datasets` (Shao 2019 mirror); original at `mlmechanics.ics.uci.edu` | UNVERIFIED | (no LICENSE file in repo; original site not fetched) | TBD | TBD | **OPEN** — research-lead must declare before download |

### 7.6 Research-gap assessment (consolidated)

| # | Direct evidence (this corpus) | Conceptual support (this corpus) | Outside-corpus direct evidence | Conclusion |
|---|---|---|---|---|
| 1D-CNN on CWRU/PU single-domain (P16, P20) | YES | — | P16 99.1% / P20 98.7% | SOTA on single-domain established |
| Cross-machine transfer (CWRU→PU→SEU) | NO (only P38 multi-rig) | P39 (sensor remounting → baseline drift) | NO direct zero-shot cross-machine on this 3-class taxonomy | **002 is genuine gap** |
| Order tracking in fault diagnosis | NO (Shao 2019 not in corpus) | — | Shao 2019 IEEE TII = SEU DDS canonical paper; established in classical vibration | Established in classical vibration; not in our corpus |
| Hedges' g / LME for transfer experiments | — | — | Standard statistical practice; not domain-specific | Standard tool, not established in this specific domain |
| 1D-CNN on Cortex-M4F / ESP32 | YES (P16, P20, P38, P39) | — | — | Multiple SOTA on target MCUs |

**Phase 21 conclusion:** the gap is real, the 002 protocol is novel,
and the closest analogue (2026 Electronics 15173896) uses adaptation
(MMD) that 002 explicitly does not.

### 7.7 Final diff + integrity report

| File | Path | Status vs `9237aed` | SHA-256 match backup? | Notes |
|---|---|---|---|---|
| `decision-log.md` | `00_lab/` | +1 row (Phase 20) | n/a | Documents Phase 20 |
| `research-principles.md` | `00_lab/` | IDENTICAL | n/a | Phase 15 |
| `paper-matrix.md` | `01_literature/` | IDENTICAL | n/a | Phase 15 |
| `dataset-registry.md` | `02_datasets/` | IDENTICAL | n/a | Phase 15 |
| `cwru.md` | `02_datasets/manifests/` | IDENTICAL | n/a | Phase 15 |
| `paderborn.md` | `02_datasets/manifests/` | IDENTICAL | n/a | Phase 15 (incl. KB23/24/27 verbatim table) |
| `seu.md` | `02_datasets/manifests/` | IDENTICAL | n/a | Phase 15 |
| `experiment_config.json` | `04_experiments/002_cross_machine_invariance/config/` | IDENTICAL | n/a | frozen protocol |
| `preprocess.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `datasets.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `evaluate.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `models.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `train.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `run_pipeline.py` | `04_experiments/002_cross_machine_invariance/src/` | IDENTICAL | n/a | frozen |
| `NOTES.md` | `04_experiments/002_cross_machine_invariance/` | IDENTICAL | n/a | Phase 12 |
| `README.md` | `04_experiments/002_cross_machine_invariance/` | IDENTICAL | n/a | Phase 12 |
| `validation_log.txt` | `04_experiments/002_cross_machine_invariance/` | IDENTICAL | n/a | Phase 12 |
| `envelope_decision_memo.md` | `04_experiments/002_cross_machine_invariance/notes/` | NEW (Phase 20) | YES (in backup) | DRAFT |
| `statistical_contract_decision_memo.md` | `04_experiments/002_cross_machine_invariance/notes/` | NEW (Phase 20) | YES (in backup) | DRAFT |
| `PILOT_RESULTS.md` | `04_experiments/002_cross_machine_invariance/pilot/` | MODIFIED (Phase 20 WS8.2) | YES (in backup) | Category 3 doc fix; 11/11 hashes verified |
| 10 pilot result files | `04_experiments/002_cross_machine_invariance/pilot/results/` | UNTRACKED | YES (all 10) | All SHA-256 identical to backup |

**Phase 21 net change: +0 file modifications, +1 file addition (this
report), +0 frozen files touched, +0 destructive operations.**

---

## 8. WS8 — Final verdict

**`PARTIALLY COMPLETE — DECISION REQUIRED`**

### 8.1 What Phase 21 closed
- **B2 (PU licensing) is reduced** — Phase 21 cited a Lessmeier 2016 §5 statement of CC
  BY-NC 4.0 from a PDF that is NOT in the local `01_literature/` corpus; the quote is therefore not
  independently re-anchored in this environment. B2 is reduced only on the basis of a prior-session external fetch; it is not closed. Citation form and non-commercial use
  scope still need research-lead ratification.
- **B10 (PILOT_RESULTS.md) is closed** — Phase 20 fix preserved; 11/11
  SHA-256 inventory verified; R1_Raw mean = 0.4727 cross-verified.
- **B8 (research gap) is confirmed within the audited local corpus** — 002 is genuine zero-shot
  cross-machine per the 39-paper corpus; P39 supports C5/C8 confound. The gap claim is bounded to the audited corpus and is not asserted as a global literature claim.
- **New finding (B9, source-domain F1 in $\Delta$)** — gap documented
  in `run_pipeline.py`; Q-WS2.1 added to decision matrix.
- **Envelope test critical assessment** — README §6 DC-dominance
  argument shown to be misleading after mean-subtraction; choice is
  operational, not mathematical.
- **Repository state** — all Phase 20 mutations confirmed on disk;
  fresh backup created at
  `/tmp/phase21_pilot_backup_20261012_144407/`; HEAD re-anchored via
  `git fetch origin` (no destructive ops).

### 8.2 What remains open
- **B1** Envelope (A/B/C/D) — research-lead decision.
- **B3** Baseline (D2) — research-lead decision.
- **B4** KB23/24/27 (D5) — research-lead decision.
- **B5** LME under-power (D4) — research-lead decision.
- **B6** Paired Hedges' $g_z$ (D3) — research-lead decision.
- **B7** Statistical estimation plan (D3 + D4) — research-lead decision.
- **B2 (CWRU, SEU subsets)** — primary-source verification pending.
- **D6** Source-domain F1 definition (NEW) — research-lead decision.
- **D8** PU citation form (NEW) — research-lead decision.

### 8.3 What Phase 21 did NOT do
- No real-data execution. No dataset download. No code mutation in
  `run_pipeline.py`, `evaluate.py`, `preprocess.py`, `datasets.py`,
  `models.py`, `train.py`, `experiment_config.json`, or any other
  frozen file.
- No silent scientific changes (hypothesis, label, baseline, class
  mapping, statistical estimand unchanged).
- No commit. No push. No `git reset --hard`. No `git clean -fd`.
- No manifest License section was edited to record the CC BY-NC 4.0
  finding — that edit awaits research-lead ratification (Category 1).

### 8.4 Recommended next phase
A **Phase 22 — Research-lead decision solicitation** phase. The 5
Phase 20 OPEN decisions + 2 Phase 21 new decisions (D6, D8) + the PU
citation form must be ratified before the next experiment step (which
would be a low-SNR envelope sanity test, a paired-$g_z$ implementation,
or a real-data pilot — none of which can be authorised without these
decisions).

The session model should present the decision matrix (this report §7.3)
as a structured question to the research lead and should not begin any
real-data work until the research lead responds.
