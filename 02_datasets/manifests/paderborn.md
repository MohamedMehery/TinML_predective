# Dataset Manifest — D2: Paderborn University Bearing (PU / KAt-Datacenter)

> **Authoritative source for this manifest**: `01_literature/paper-matrix.md`
> row P16 (El Boughardini 2026) and the ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md` plus the
> `00_lab/decision-log.md` "Dataset Harmonization & Feasibility Audit"
> row (2026-10-07), which verifies the CWRU ↔ PU ↔ SEU triad. The
> 2024 Zenodo refresh is the snapshot currently intended for
> Experiment 002; details not directly supported by the repo documents
> are marked `[TO VERIFY FROM SOURCE DOC]`.

## Card (per repo template)

```
Dataset:               Paderborn University Bearing Dataset (KAt-Datacenter,
                       also "PU" in the literature; the 2024 Zenodo
                       release supersedes the 2016 original)
URL:                   https://groups.uni-paderborn.de/kat/BearingDataCenter/
                       (2024 Zenodo release URL
                       [TO VERIFY FROM SOURCE DOC])
Institution / Year:    Paderborn University, Chair of Design and Drive
                       Technology. Original release 2016; the 2024
                       Zenodo snapshot is the one referenced in the
                       C5/C8 decision-log row.
Machine:               Modular test rig; 32 bearings total in the public
                       2016 release; per-bearing damage seeded by
                       pitting, plastic deformation, and cracks
                       (real accelerated lifetime damage, not EDM).
Sensor:                IEPE vibration sensor + motor current + temperature
                       (multi-modal; vibration is the primary channel
                       for Experiment 002)
Sampling Rate:         64 kHz nominal; Experiment 002 will resample to
                       12 kHz using the polyphase anti-alias filter
                       specified in `experiment_config.json`
                       (filter_order=64). [TO VERIFY FROM SOURCE DOC]
                       for the exact native rate of the 2024 snapshot.
Raw signal:            Yes (raw vibration; current + temperature
                       available as auxiliary channels for future
                       non-claims checks but not used in R1/R2/R3 of
                       Experiment 002)
Number of machines:    1 modular testbed; multiple bearing UIDs (32 in
                       the 2016 release; the 2024 snapshot may differ,
                       [TO VERIFY FROM SOURCE DOC])
Operating conditions:  4 operating conditions per bearing in the 2016
                       release (rotational speed, load torque, radial
                       force on outer race); 2024 snapshot conditions
                       [TO VERIFY FROM SOURCE DOC]
Fault types:           Real damage (artificial pitting, plastic
                       deformation, cracks). For Experiment 002 the
                       fault class collapse is: Healthy vs. Damaged
                       (binary or 2-class, depending on what the 2024
                       release's label granularity allows). Multi-class
                       damage labels are NOT used in the cross-machine
                       comparison because they are not present in CWRU
                       and SEU at the same granularity.
Healthy data:          Yes (one healthy bearing per test series)
Fault data:            Yes (multiple damage modes per bearing)
Labels:                Per-file metadata; the canonical PU labels
                       (healthy / OR / IR / combined) are documented in
                       the dataset README [TO VERIFY FROM SOURCE DOC]
Standard train/test protocol:
                       P16 (El Boughardini 2026) uses 70 / 30 random
                       split on PU; we reject this in Experiment 002
                       because it leaks physical bearing identity.
                       Experiment 002 uses physical-bearing-UID +
                       run-level separation per charter §2.
Publicly available:    Yes
License:               CC-BY 4.0 (per the 2024 Zenodo release terms;
                       [TO VERIFY FROM SOURCE DOC])
Download size:         [TO VERIFY FROM SOURCE DOC] (the 2016 release
                       was on the order of 5–10 GB across all 32
                       bearings; the 2024 snapshot may differ)
Documentation:         groups.uni-paderborn.de/kat/BearingDataCenter/
                       + the P16 paper §3 (Bearing Dataset) reproduces
                       the 2016 labeling convention
Known limitations:     (a) Fault taxonomy is real-damage, not seeded,
                       so there is no fault-diameter sweep like CWRU.
                       This means Experiment 002's fault class
                       granularity must collapse to Healthy vs.
                       Damaged, or Normal vs. Outer-race / Inner-race
                       using PU's own labels. (b) The decision log
                       (2026-10-07) explicitly verified that PU is
                       the appropriate "real-damage" complement to
                       CWRU's "single-point seeded" rig. (c) The PU
                       test rig has a different casing transfer
                       function from CWRU; this is the cross-machine
                       structural resonance that the C5/C8 hypothesis
                       predicts will collapse 1D-CNNs but be
                       preserved by R3_Order.
Suitable research questions:
                       EXP-C5C8-002 — as the "real-damage" target of
                       T_CWRU→PU and as the source of T_PU→CWRU,
                       T_PU→SEU, T_PU→PU (closed-domain control).
```

## Evidence grounding (which paper said what)

| Claim | Paper | Tag |
|---|---|---|
| PU + CWRU + MFPT used together; INT8 1D-CNN on Cortex-M4F | P16 | `[DIRECT]` |
| CWRU ↔ PU ↔ SEU triad ratified as cross-machine benchmark | Decision log 2026-10-07 | `[DIRECT]` |
| Random-window split (P16 baseline) rejected for leakage | Charter §2 invariants | `[DIRECT]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Confirm the 2024 Zenodo snapshot URL,
   license text, and the file inventory.
2. `[TO VERIFY FROM SOURCE DOC]` Confirm the 2024 snapshot's native
   sampling rate and the recommended resampling factor.
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the per-bearing damage-mode
   labels available in the 2024 release and align with the
   Healthy / Damaged collapse used in Experiment 002.
4. `[TO VERIFY FROM SOURCE DOC]` Confirm the per-bearing rpm at each
   operating condition (needed for the IDFT 1X speed tracker ±15 %
   search window).

## Configuration binding

This dataset feeds Experiment 002 with `source=paderborn` and
`target=paderborn` and as the source/target of the 2 cross-machine pairs
in the 6-pair transfer matrix that touch PU:

| Transfer pair | Source | Target | Use |
|---|---|---|---|
| $T_{22}$ | PU | PU | Closed-domain control (different physical bearings, same rig) |
| $T_{21}$ | PU | CWRU | Cross-machine, dissimilar rig |
| $T_{23}$ | PU | SEU | Cross-machine, different testbed family |
| $T_{12}$ | CWRU | PU | Reverse of $T_{21}$ |
| $T_{32}$ | SEU | PU | Reverse of $T_{23}$ |
