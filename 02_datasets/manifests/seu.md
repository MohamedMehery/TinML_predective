# Dataset Manifest — D3: Southeast University (SEU) Gearbox

> **Authoritative source for this manifest**: `01_literature/paper-matrix.md`
> row P27 (Łuczak 2024) and the ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md` plus the
> `00_lab/decision-log.md` "Dataset Harmonization & Feasibility Audit"
> row (2026-10-07), which ratifies CWRU ↔ PU ↔ SEU as the cross-machine
> triad. SEU is the *gearbox* member of the triad and is the principal
> test of "different testbed family". Anything not directly supported
> by these documents is marked `[TO VERIFY FROM SOURCE DOC]`.

## Card (per repo template)

```
Dataset:               Southeast University (SEU) Gearbox Dataset
URL:                   [TO VERIFY FROM SOURCE DOC] (P27 cites the dataset
                       in MDPI Electronics data-availability statement;
                       the canonical URL is not preserved in the
                       paper-matrix row and must be looked up from
                       the P27 PDF directly)
Institution / Year:    Southeast University, School of Mechanical
                       Engineering. Year of release [TO VERIFY FROM
                       SOURCE DOC] (P27 is the 2024 publication, so
                       release year is on or before 2024).
Machine:               SEU drivetrain dynamometer; the rig is a
                       gearbox + bearings combination, which is
                       structurally different from CWRU's induction
                       motor and PU's modular test rig. The gear-mesh
                       excitation is a major additional spectral
                       component that the C5/C8 protocol explicitly
                       addresses via "SEU gear-mesh envelope isolation"
                       in the decision log "Protocol Design Gate Freeze"
                       row.
Sensor:                Vibration accelerometer (model [TO VERIFY FROM
                       SOURCE DOC] — P27 §2 describes the channel but
                       the sensor model is not in the paper-matrix
                       row)
Sampling Rate:         [TO VERIFY FROM SOURCE DOC] (P27 used 2D CWT
                       scalograms on the raw signal; the native rate
                       must be confirmed before the 12 kHz
                       harmonization step in
                       `experiment_config.json` is applied)
Raw signal:            Yes (raw vibration; P27 converts to CWT
                       scalograms as its first step, but the raw
                       time series is available for Experiment 002's
                       R1/R2/R3 pipeline)
Number of machines:    [TO VERIFY FROM SOURCE DOC] (one drivetrain
                       testbed; multiple operating conditions and
                       fault states)
Operating conditions:  [TO VERIFY FROM SOURCE DOC] (P27 reports
                       multiple fault classes; the exact operating
                       condition enumeration is not in the
                       paper-matrix row)
Fault types:           Gearbox fault classes; P27 reports multiple
                       fault categories. For Experiment 002 the
                       cross-machine collapse is:
                         Healthy / Gear-fault / Bearing-fault
                       (any finer granularity is not present in
                       CWRU / PU at the same level and so cannot be
                       used in the cross-machine comparison).
                       The decision log explicitly notes "SEU
                       gear-mesh envelope isolation" as one of the
                       three technical closures of the C5/C8 protocol
                       freeze.
Healthy data:          Yes
Fault data:            Yes
Labels:                Per-file metadata [TO VERIFY FROM SOURCE DOC]
Standard train/test protocol:
                       P27 used random k-fold cross-validation on the
                       CWT-scalogram representations; we reject this
                       in Experiment 002 because it leaks physical
                       bearing / gear identity across folds. Experiment
                       002 uses physical-bearing-UID + run-level
                       separation per charter §2.
Publicly available:    Yes (per P27's MDPI data availability statement)
License:               CC-BY 4.0 (assumed per MDPI Electronics
                       open-data policy; [TO VERIFY FROM SOURCE DOC]
                       from the SEU release page directly)
Download size:         [TO VERIFY FROM SOURCE DOC]
Documentation:         electronics-13-00452.pdf (P27) §2 (Experimental
                       Setup) + the SEU release README once the URL
                       is confirmed
Known limitations:     (a) SEU is a *gearbox* rig, not just a
                       bearing rig. The gear-mesh excitation is at a
                       much higher order than the bearing-fault
                       characteristic orders. The C5/C8 protocol
                       freeze explicitly addresses this via "SEU
                       gear-mesh envelope isolation" (decision log
                       2026-10-07), which must be implemented in
                       `src/preprocess.py` before SEU enters the
                       transfer matrix. (b) P27 used CWT scalograms
                       (a 2D transform) which exceed the 8.2 kB RAM
                       budget of the charter, so the 2D path is NOT
                       used in Experiment 002 — the protocol uses
                       R3_Order (1D, MLP) on the SEU channel
                       instead. (c) The casing transfer function of
                       the SEU drivetrain is the third distinct
                       h_casing(t) in the triad; it is the most
                       dissimilar from CWRU's motor casing and
                       therefore the hardest test of the
                       cross-machine hypothesis.
Suitable research questions:
                       EXP-C5C8-002 — as the "different testbed
                       family" target of T_CWRU→SEU and T_PU→SEU and
                       as the source of T_SEU→CWRU, T_SEU→PU,
                       T_SEU→SEU (closed-domain control).
```

## Evidence grounding (which paper said what)

| Claim | Paper | Tag |
|---|---|---|
| SEU gearbox used with CWT-Morlet 2D ResNet | P27 | `[DIRECT]` |
| CWRU ↔ PU ↔ SEU triad ratified as cross-machine benchmark | Decision log 2026-10-07 | `[DIRECT]` |
| "SEU gear-mesh envelope isolation" is one of the three C5/C8 protocol closures | Decision log 2026-10-07, Protocol Freeze row | `[DIRECT]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU release URL from
   P27 §Data Availability and pull the canonical license text.
2. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU native sampling rate
   and the recommended resampling factor for 12 kHz harmonization.
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU fault class labels
   and align with the Healthy / Gear-fault / Bearing-fault collapse
   used in Experiment 002.
4. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU operating-condition
   enumeration (rpm, load) for the IDFT 1X speed tracker ±15 %
   search window.

## Configuration binding

This dataset feeds Experiment 002 with `source=seu` and `target=seu`
and as the source/target of the 2 cross-machine pairs in the 6-pair
transfer matrix that touch SEU:

| Transfer pair | Source | Target | Use |
|---|---|---|---|
| $T_{33}$ | SEU | SEU | Closed-domain control (different runs, same rig) |
| $T_{31}$ | SEU | CWRU | Cross-machine, different testbed family |
| $T_{32}$ | SEU | PU | Cross-machine, different testbed family |
| $T_{13}$ | CWRU | SEU | Reverse of $T_{31}$ |
| $T_{23}$ | PU | SEU | Reverse of $T_{32}$ |
