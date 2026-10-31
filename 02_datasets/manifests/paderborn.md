# Dataset Manifest — D2: Paderborn University Bearing (PU / KAt-Datacenter)

> **Authoritative source for this manifest**: the KAt-Datacenter
> official page at
> `https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download`
> (the verified working English URL, 2026-10-11; the German-path
> `https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter`
> previously recorded in this manifest was 404 as of 2026-10-11 and
> has been replaced), the Lessmeier et al. 2016 PHM Society paper
> (the canonical citation for the dataset — see below), the
> `01_literature/paper-matrix.md` row P16 (El Boughardini 2026), the
> ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md`, and the
> `00_lab/decision-log.md` "Dataset Harmonization & Feasibility Audit"
> row (2026-10-07), which verifies the CWRU ↔ PU ↔ SEU triad.
>
> **Important correction (2026-10-11, Phase 11 audit)**: The earlier
> draft of this manifest asserted that "a 2024 Zenodo snapshot is the
> one referenced in the C5/C8 decision-log row" and used the URL
> `https://groups.uni-paderborn.de/kat/BearingDataCenter/`. Phase 11
> web verification could NOT locate a distinct 2024 Zenodo snapshot
> of the Paderborn bearing dataset; the canonical official page is on
> `mb.uni-paderborn.de`, not on `groups.uni-paderborn.de`. The license
> text also proved inconsistent across secondary sources (CC-BY
> with commercial-use contact vs CC-BY-NC-4.0 vs unspecified) and
> therefore remains explicitly UNRESOLVED below.
>
> **Phase 15 corrections (2026-10-11, evidence-based documentation
> patch)**: the Phase 14 audit re-verified Lessmeier et al. (2016)
> Section 4.2 + Table 5 verbatim and added a per-bearing damage-mode
> evidence table for the combined-damage bearings KB23, KB24, and
> KB27. The canonical KAt-Datacenter URL was updated to the verified
> working English path. The bearing-level "OR vs IR" simple
> classification used in the cross-machine collapse is documented
> as an approximation for these three combined-damage bearings (see
> new "KB23 / KB24 / KB27 damage-mode evidence table" below).
> License, sensor model, and per-bearing rpm/load-torque values
> remain UNVERIFIED or `[TO VERIFY FROM SOURCE DOC]` because no
> authoritative source could be located for them in the Phase 14
> audit.

## Card (per repo template)

```
Dataset:               Paderborn University Bearing Dataset
                       (KAt-Datacenter, also "PU" in the literature;
                       the 2016 release is the canonical public
                       version)
URL:                   https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
                       (canonical official KAt data-sets-and-download
                       page, verified working 2026-10-11).
                       The English-language KAt research landing
                       page is at
                       https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/
                       (verified 2026-10-11).
                       The previously-recorded German-path URL
                       (mb.uni-paderborn.de/kat/forschung/...)
                       returned 404 as of 2026-10-11 and has been
                       superseded.
                       The older hostname
                       `https://groups.uni-paderborn.de/kat/BearingDataCenter/`
                       appears in some 2025 secondary papers but is
                       no longer the canonical host.
Institution / Year:    Paderborn University, Chair of Design and Drive
                       Technology (KAt). Original release 2016,
                       accompanying the Lessmeier, Kimotho, Zimmer,
                       Sextro paper, "Condition Monitoring of Bearing
                       Damage in Electromechanical Drive Systems by
                       Using Motor Current Signals of Electric Motors:
                       A Benchmark Data Set for Data-Driven
                       Classification", PHM Society European
                       Conference 3(1), 2016,
                       DOI 10.36001/phme.2016.v3i1.1577
                       (citation verified 2026-10-11). A distinct
                       "2024 Zenodo snapshot" of this dataset was
                       NOT located by the Phase 11 web audit; the
                       2016 release is the one referenced in the
                       KAt-DataCenter citation string and remains
                       the canonical source.
Machine:               Modular test rig; 32 bearings total in the public
                       2016 release; per-bearing damage seeded by
                       pitting, plastic deformation, and cracks
                       (real accelerated lifetime damage, not EDM).
Sensor:                Vibration sensor + motor current + temperature
                       (multi-modal; vibration is the primary channel
                       for Experiment 002). Specific sensor model
                       (e.g., "IEPE") [TO VERIFY FROM SOURCE DOC] —
                       the KAt-DataCenter site does not name a
                       sensor model on the page verified 2026-10-11.
Sampling Rate:         64 kHz nominal, confirmed by the KAt-DataCenter
                       page (verified 2026-10-11) and corroborated by
                       multiple secondary sources (e.g., Lessmeier
                       2016 abstract). Experiment 002 will resample
                       to 12 kHz using the polyphase anti-alias filter
                       specified in `experiment_config.json`
                       (filter_order=64).
Raw signal:            Yes (raw vibration; current + temperature
                       available as auxiliary channels for future
                       non-claims checks but not used in R1/R2/R3 of
                       Experiment 002)
Number of machines:    1 modular testbed; 32 bearing UIDs total in
                       the 2016 release (6 healthy + 12 artificially
                       damaged + 14 with real damage from accelerated
                       lifetime tests; 6203 ball bearings). The
                       KAt-DataCenter page (verified 2026-10-11)
                       confirms "32 different bearings" and the
                       6/12/14 split; the Lessmeier 2016 paper
                       confirms the same. A distinct 2024 snapshot
                       with a different bearing inventory was NOT
                       located by the Phase 11 audit.
Operating conditions:  4 operating conditions per bearing (rotational
                       speed, load torque, radial force, temperature);
                       20 measurements × 4 sec each per
                       operating-condition / bearing combination.
                       This matches the KAt-DataCenter page
                       description (verified 2026-10-11) and the
                       Lessmeier 2016 paper.
Fault types:           Real damage (artificial pitting, plastic
                       deformation, cracks). For Experiment 002 the
                       fault class collapse is: Healthy vs. Damaged
                       (binary or 2-class, depending on what the 2024
                       release's label granularity allows). Multi-class
                       damage labels are NOT used in the cross-machine
                       comparison because they are not present in CWRU
                       and SEU at the same granularity.
                       [Phase 15 note] The three combined-damage
                       bearings KB23, KB24, and KB27 documented in
                       Lessmeier 2016 Table 5 (see evidence table
                       below) are NOT a simple single-element OR/IR
                       classification; mapping them to a single
                       "OR" or "IR" class is an APPROXIMATION
                       requiring a research-lead decision.
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
Publicly available:    Yes (per the KAt-DataCenter page, verified
                       2026-10-11).
License:               [TO VERIFY FROM SOURCE DOC] — EVIDENCE
                       CONFLICTING across secondary sources:
                       (a) jonathanwvd/awesome-industrial-datasets
                           states "CC BY with a request to contact
                           authors for commercial use";
                       (b) zenn.dev/logicia32 (2026-07-14) states
                           "CC BY-NC 4.0";
                       (c) the 2025 Springer paper "Optimal feature
                           complexity..." Data Availability states
                           the dataset is "publicly available at
                           https://groups.uni-paderborn.de/kat/
                           BearingDataCenter/" without specifying
                           a license;
                       (d) the KAt-DataCenter page itself (verified
                           2026-10-11) requests citation of the
                           Lessmeier 2016 paper but does not
                           explicitly state a license on the page.
                       Therefore NO single license variant is
                       confirmed; the manifest retains the
                       uncertainty rather than asserting "CC-BY 4.0"
                       (the earlier draft wording).
Download size:         [TO VERIFY FROM SOURCE DOC] (the 2016 release
                       is plausibly on the order of 5–15 GB across
                       32 bearings × 4 conditions × 20 measurements
                       × 64 kHz × 4 sec ≈ 1 GB raw, but this is
                       unverified).
Documentation:         https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
                       (canonical KAt data-sets-and-download page,
                       verified working 2026-10-11) + the Lessmeier,
                       Kimotho, Zimmer, Sextro (2016) PHM Society
                       European Conference paper
                       (DOI 10.36001/phme.2016.v3i1.1577) which
                       defines the 32-bearing benchmark. Section 4.2
                       and Table 5 of the Lessmeier 2016 paper are
                       the authoritative source for the per-bearing
                       damage-mode labels (see evidence table
                       below).
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
                       preserved by R3_Order. (d) The three
                       combined-damage bearings KB23, KB24, KB27
                       documented in Lessmeier 2016 Table 5 are
                       multi-element (IR+OR or OR+IR) and cannot be
                       cleanly mapped to a single OR or IR class
                       without losing fidelity (see evidence
                       table).
Suitable research questions:
                       EXP-C5C8-002 — as the "real-damage" target of
                       T_CWRU→PU and as the source of T_PU→CWRU,
                       T_PU→SEU, T_PU→PU (closed-domain control).
```

## KB23 / KB24 / KB27 damage-mode evidence table (Lessmeier 2016, Table 5)

The following table reproduces Lessmeier, Kimotho, Zimmer, Sextro
(2016), PHM Society European Conference 3(1), Section 4.2 + Table 5
(DOI 10.36001/phme.2016.v3i1.1577) verbatim for the three
combined-damage real-damage bearings used in the cross-machine
collapse. Access: `mb.uni-paderborn.de/fileadmin-mb/kat/PDF/
Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf` (PHM Society
European Conference 2016, open access), re-verified 2026-10-11.

Legend (Lessmeier 2016 §2): **IR** = inner ring; **OR** = outer
ring; **S** = single damage (one symptom on one component);
**R** = repetitive damage (identical symptoms repeated on the same
component); **M** = multiple damage (different symptoms on the
bearing, or identical symptoms on different components; may include
repetitive damages). Extent of damage uses the 5-level scale from
Table 2 (1 = ≤2 mm, 2 = 2–4.5 mm, 3 = 4.5–13.5 mm, 4 = 13.5–31.5 mm,
5 = >31.5 mm for the 6203 bearing). Characteristic of damage:
"single point" (damage length ≤ rolling element diameter) or
"distributed" (damage length > rolling element diameter).

| Bearing code | Damage (main mode and symptom) | Bearing element | Combination | Arrangement | Extent of damage | Characteristic of damage |
|---|---|---|---|---|---|---|
| **KB23** | fatigue: pitting | IR (+OR) | M (multiple) | random | 2 | single point |
| **KB24** | fatigue: pitting | IR (+OR) | M (multiple) | no repetition | 3 | distributed |
| **KB27** | Plastic deform.: indentations | OR + IR | M (multiple) | random | 1 | distributed |

> **Implication for EXP-002 class mapping** (research-lead decision
> pending, NOT resolved by this patch): All three of KB23, KB24, and
> KB27 are classified in Lessmeier 2016 as **multiple damage (M)**,
> meaning damage is present on more than one bearing element.
> KB23 and KB24 are fatigue pitting with the **inner ring as the
> primary element** and the outer ring as a secondary element
> ("IR (+OR)"). KB27 is plastic-deformation indentations on **both
> outer and inner ring** ("OR + IR"). The simple "Healthy / OR / IR"
> 3-class collapse used in `experiment_config.json` therefore
> cannot represent these three bearings exactly:
>  - mapping KB23 and KB24 to a single "IR" class drops the OR
>    component of the damage;
>  - mapping KB27 to either "OR" or "IR" drops the complementary
>    element's contribution.
> A research-lead decision is required to either (a) accept the
> approximation and document it as a class-collapse limitation, or
> (b) carry an extra "Combined" class through EXP-002 (which would
> not align with the CWRU/SEU class collapse). This manifest
> records the verbatim Lessmeier 2016 labels and FLAGS the
> approximation; it does NOT change the EXP-002 class mapping.

## Evidence grounding (which source said what)

| Claim | Source | Tag |
|---|---|---|
| PU dataset is 32 bearings (6 healthy + 12 artificial + 14 real), 4 operating conditions, 20 measurements × 4 sec each, 64 kHz, .mat format | `mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download` (verified working 2026-10-11) | `[DIRECT]` |
| PU dataset benchmark paper citation: Lessmeier, Kimotho, Zimmer, Sextro (2016), PHM Society European Conference 3(1), DOI 10.36001/phme.2016.v3i1.1577 | `mb.uni-paderborn.de/en/kat/research/bearing-datacenter/` (citation string on the page) and PHM Society open-access PDF | `[DIRECT]` |
| KB23, KB24, KB27 per-bearing damage mode + symptom + element + combination + arrangement + extent + characteristic, as above | Lessmeier 2016 §2 (categorization criteria) and Table 5 (test bearings with real damages) — re-verified 2026-10-11 from the PHM Society open-access PDF | `[DIRECT]` |
| PU + CWRU + MFPT used together; INT8 1D-CNN on Cortex-M4F | P16 (El Boughardini 2026) | `[DIRECT]` |
| CWRU ↔ PU ↔ SEU triad ratified as cross-machine benchmark | Decision log 2026-10-07 | `[DIRECT]` |
| Random-window split (P16 baseline) rejected for leakage | Charter §2 invariants | `[DIRECT]` |
| ~~Distinct 2024 Zenodo snapshot of PU~~ — REMOVED: Phase 11 audit could not locate a distinct 2024 Zenodo snapshot; the canonical source is the KAt-DataCenter page and the 2016 Lessmeier paper. | Phase 11 audit, 2026-10-11 | `[CONFLICT RESOLVED]` |
| ~~PU license is CC-BY 4.0~~ — DOWNGRADED: license is genuinely unresolved across secondary sources (CC-BY with commercial-use contact vs CC-BY-NC-4.0 vs unspecified). | Phase 11 audit, 2026-10-11 | `[CONFLICTING EVIDENCE]` |
| **Phase 15 supersession (2026-10-11)**: the KAt-Datacenter URL `mb.uni-paderborn.de/kat/forschung/...` (German "forschung" path) previously recorded in this manifest returned 404 as of 2026-10-11. The verified working URL is the English "research" path: `mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download`. | Phase 14/15 audit, 2026-10-11 | `[URL CORRECTED]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Confirm the canonical license text
   for the Paderborn KAt-Datacenter dataset. The Phase 11 audit
   found conflicting secondary-source claims (CC-BY with
   commercial-use contact vs CC-BY-NC-4.0 vs unspecified); the
   KAt-DataCenter page (verified 2026-10-11) does not state a
   license on the public page. The 2016 Lessmeier et al. paper
   text should be checked for the license statement.
2. `[TO VERIFY FROM SOURCE DOC]` Confirm the per-bearing damage-mode
   labels available in the 2016 release and align with the
   Healthy / Damaged collapse used in Experiment 002. The Lessmeier
   2016 paper defines the labeling convention; the README on the
   KAt-DataCenter page (verified 2026-10-11) lists the 32-bearing
   inventory.
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the per-bearing rpm at each
   of the 4 operating conditions (needed for the IDFT 1X speed
   tracker ±15 % search window).
4. `[RESEARCH-LEAD DECISION REQUIRED]` Decide whether the three
   combined-damage bearings KB23 / KB24 / KB27 should be (a)
   approximated into a single IR or OR class (current 3-class
   collapse), with the loss-of-fidelity documented; or (b) carried
   as a separate "Combined" class. See the KB23/KB24/KB27
   damage-mode evidence table above for the verbatim Lessmeier
   2016 Table 5 labels.

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
