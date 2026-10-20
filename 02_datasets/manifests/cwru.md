# Dataset Manifest — D1: CWRU Bearing Data Center

> **Authoritative source for this manifest**: the CWRU Bearing Data
> Center official site at
> `https://engineering.case.edu/bearingdatacenter` (verified
> 2026-10-11), the `01_literature/paper-matrix.md` rows P15 (Hakam
> 2026), P16 (El Boughardini 2026), P20 (Cotrino 2026), and P38 (Kolok
> 2025), plus the ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md` and the
> `00_lab/decision-log.md` Protocol Freeze (2026-10-07). Anything not
> directly supported by these documents is marked
> `[TO VERIFY FROM SOURCE DOC]`.
>
> **Important corrections (2026-10-11, Phase 11 audit)**:
> (1) the previously cited "Vieira 2026" 3-class collapse excluding
> Ball could not be located in the CWRU paper rows P15/P16/P20/P38
> nor in a Phase 11 web search; the reference has been removed;
> (2) the CWRU license is not an explicit CC-BY 4.0; the dataset is
> released "publicly for academic use" without a formal CC-BY
> statement; (3) a distinct "2024 refresh" of the CWRU dataset
> could not be located — the live site is the 2003/2010 release.
>
> **Phase 15 corrections (2026-10-11, evidence-based documentation
> patch)**: the Phase 14 audit confirmed that (a) the public CWRU
> BDC file tables list only 0.007", 0.014", and 0.021" fault
> diameters; the 0.028" and 0.040" diameters are fault-documented
> on the apparatus/specifications page but do NOT have corresponding
> `.mat` file links in the 12k DE, 12k FE, 48k DE, or normal-baseline
> download tables; and (b) the CWRU BDC sensor model is NOT
> officially named on the public engineering.case.edu site, and the
> PCB 352C33 attribution that previously linked CWRU to P15 (Hakam
> 2026) is incorrect because P15 does not use CWRU data, does not use
> a PCB 352C33 sensor, does not use an STM32F4 microcontroller, and
> does not report 97.2% accuracy on CWRU (it is a Kaggle IEPE
> vibration study on a custom testbed, see corrected P15 row in
> `01_literature/paper-matrix.md`). The CWRU BDC license/usage
> terms remain UNVERIFIED from accessible official sources; only
> third-party Zenodo and Kaggle mirror licenses exist and these are
> NOT the original CWRU BDC's own terms. See the
> "Evidence grounding" table below for new `[CONFLICTING EVIDENCE]`
> / `[MISATTRIBUTION CORRECTED]` / `[UNVERIFIED]` rows.

## Card (per repo template)

```
Dataset:               Case Western Reserve University Bearing Data Center
URL:                   https://engineering.case.edu/bearingdatacenter
                       (canonical, verified 2026-10-11). Sub-page
                       `https://engineering.case.edu/bearingdatacenter/
                       48k-drive-end-bearing-fault-data` enumerates
                       the 48 kHz DE files (verified 2026-10-11).
                       An older mirror at
                       `https://csegroups.case.edu/bearingdatacenter/
                       pages/download-data-file` is also still
                       referenced by some secondary sources (e.g.,
                       the `cathysiyu/Mechanical-datasets` README).
                       The csegroups URL currently returns HTTP 000
                       (offline) and is therefore not a working
                       download path (verified 2026-10-11).
Institution / Year:    Case Western Reserve University, original release
                       2003 (last public revision per the
                       engineering.case.edu site is the 2010 batch;
                       a distinct "2024 refresh" of the dataset
                       itself was NOT located by Phase 11 audit —
                       the 2024-04 Zenodo 10987113 record is a
                       third-party mirror, not an official CWRU
                       release).
Machine:               2-hp (1.49 kW) 4-pole induction motor, relubricated
                       drive-end and fan-end bearings, single-point
                       single-line-to-line fault seeding via EDM
Sensor:                Accelerometer model NOT officially named on
                       the public engineering.case.edu site. The
                       download page (verified 2026-10-11) only
                       states the channel roles "DE - drive end
                       accelerometer data; FE - fan end accelerometer
                       data; BA - base accelerometer data" without
                       a vendor or model number. The previous
                       "PCB 352C33" attribution to CWRU via P15
                       (Hakam 2026) has been removed because P15
                       uses an AS-062 IEPE sensor on a different
                       custom rig, NOT a PCB 352C33 and NOT CWRU
                       data (see the corrected P15 row in
                       `01_literature/paper-matrix.md`). The CWRU
                       BDC accelerometer model is therefore
                       UNVERIFIED. [TO VERIFY FROM SOURCE DOC
                       against the readme.pdf bundled with the .mat
                       files — not accessible without a direct
                       download.] 16-channel DAQ and shaft encoder
                       for 1X reference [TO VERIFY FROM SOURCE DOC].
Sampling Rate:         12 kHz and 48 kHz files available; P16 confirms 12 kHz
                       used for the 1D-CNN baseline; current Experiment 002
                       targets 12 kHz (matches `experiment_config.json`
                       `sampling_rate_harmonization.target_fs_hz = 12000`)
Raw signal:            Yes (raw vibration time series; no preprocessing
                       required other than resampling if 48 kHz is used)
Number of machines:    1 physical testbed, 1 motor, 2 bearing housings
                       (DE, FE) ⇒ multiple physical bearing UIDs
Operating conditions:  0 / 1 / 2 / 3 hp loads (≈ 0 / 1.49 / 2.98 / 4.48 Nm
                       shaft torque), 4 motor speeds documented
                       (~ 1797 / 1772 / 1750 / 1720 rpm in the
                       public release, confirmed by
                       engineering.case.edu/bearingdatacenter/48k-
                       drive-end-bearing-fault-data verified
                       2026-10-11).
Fault types:           Inner-race (IR), Outer-race (OR), Ball (BA)
                       faults. Per the audited public file tables
                       (verified 2026-10-11):
                         • 12k DE: 0.007", 0.014", 0.021" only
                           (IR, BA, OR@6/OR@3/OR@12)
                         • 12k FE: 0.007", 0.014", 0.021" only
                           (IR, BA, OR@6/OR@3/OR@12; some cells
                           are "Data not available" marked with \*)
                         • 48k DE: 0.007", 0.014", 0.021" only
                           (IR, BA, OR@6/OR@3/OR@12)
                         • Normal baseline: 4 files 97/98/99/100.mat
                           at loads 0/1/2/3 HP
                       The `engineering.case.edu/bearingdatacenter/
                       fault-specifications` page additionally
                       documents the seeded damage apparatus as
                       having 0.028" DE (IR+OR NTN, 0.050"/0.150"
                       depth) and 0.040" DE OR (NTN, 0.050" depth),
                       BUT these diameters do NOT have corresponding
                       `.mat` file links in any of the 4 audited
                       download tables (verified 2026-10-11). The
                       0.028" and 0.040" diameters are therefore
                       apparatus-documented but NOT publicly
                       downloadable. The previous Phase 11 wording
                       that asserted "0.028" is in the public
                       set" is corrected here.
                       For Experiment 002 the cross-machine collapse
                       is 3-class (Normal / IR / OR) per the triad
                       protocol — Ball (BA) is excluded from the
                       cross-machine collapse. The earlier draft
                       cited "Vieira 2026" as the source for the
                       BA exclusion; that citation was not located
                       in `01_literature/paper-matrix.md` rows
                       P15/P16/P20/P38 and could not be confirmed
                       by Phase 11 audit, so the BA-exclusion is
                       now recorded as a project-internal choice
                       without external attribution.
Healthy data:          Yes (baseline runs at each load/speed combination)
Fault data:            Yes (single-point seeded faults, fault diameter
                       graded, single bearing per test)
Labels:                Filename-encoded; canonical mapping is
                       (DE/FE)_(fault location)_(fault size)_(load/rpm)
Standard train/test protocol:
                       1D-CNN literature uses random-window splitting
                       (P15, P16, P20), which we explicitly reject for
                       Experiment 002 because it leaks physical bearing
                       identity across folds. Experiment 002 uses
                       physical-bearing-UID + run-level separation per
                       charter §2 invariants.
Publicly available:    Yes (per the CWRU Bearing Data Center site,
                       verified 2026-10-11). Public file inventory
                       audited 2026-10-11 covers the 0.007/0.014/0.021"
                       diameter set only (see Fault types above).
License:               UNVERIFIED from accessible official sources.
                       The CWRU BDC public site
                       (engineering.case.edu/bearingdatacenter/*) does
                       NOT contain a license, terms, copyright, or
                       about page that is accessible to this audit
                       (verified 2026-10-11; the
                       `about-this-site`, `faq`, `acknowledgements`,
                       `project-history` sub-paths return 404 or
                       are absent). The dataset is described on the
                       site as "publicly available" for academic use
                       without a formal CC-BY or other license
                       statement. The Zenodo 10987113 (2024-04)
                       third-party mirror carries a CC-BY-4.0 license
                       badge and the Kaggle mirror carries
                       CC-BY-SA-4.0, but these are MIRROR
                       statements, not the CWRU BDC's own terms, and
                       should not be cited as the original dataset's
                       license. The manifest does NOT assert
                       "CC-BY 4.0" and explicitly records the
                       license as UNVERIFIED.
Download size:         ~120 MB (P16 reports this; [TO VERIFY FROM
                       SOURCE DOC] for the full current release
                       inventory — the 48 kHz DE subset alone is
                       approximately 50 MB of .mat files).
Documentation:         engineering.case.edu/bearingdatacenter
                       (canonical CWRU BDC site, verified
                       2026-10-11) + the 48 kHz DE sub-page
                       (engineering.case.edu/bearingdatacenter/
                       48k-drive-end-bearing-fault-data, verified
                       2026-10-11) + the readme.pdf bundled with
                       the .mat files (NOT verified by Phase 11
                       audit; would need a direct download to
                       confirm). The earlier claim that "P15 §3
                       (Hakam 2026) reproduces the canonical file
                       list" is incorrect — P15 does not use CWRU
                       data (it uses the Kaggle
                       `medlamane/small-vibration-dataset-of-ccsiepe-
                       sensor-as-062` set on a custom testbed).
Known limitations:     (a) Concatenation across drive-end and fan-end
                       files creates artificial phase jumps — this is why
                       the decision log ("Dataset Harmonization & Feasibility
                       Audit" row, 2026-10-07) explicitly rejects
                       "concatenating static CWRU/SEU files for streaming
                       claims" and "random window splitting across same
                       bearings". (b) The casing transfer function
                       h_casing(t) of the CWRU rig is one of the specific
                       structural resonances the C5/C8 hypothesis predicts
                       1D-CNNs will overfit to. (c) Single motor, so
                       "cross-machine" claims are about transfer to PU
                       and SEU, not within CWRU itself. (d) The CWRU
                       BDC accelerometer model is not officially named
                       on the public site; any sensor-model claim
                       (e.g. PCB 352C33) used downstream must be
                       separately verified.
Suitable research questions:
                       EXP-C5C8-002 (Cross-Machine Invariance Transfer)
                       — specifically as the source for the 3 transfer
                       pairs T_CWRU→PU, T_CWRU→SEU, and as the
                       single-machine closed-domain control in T_CWRU→CWRU
                       (same-bearings different-run sanity check).
```

## Evidence grounding (which source said what)

| Claim | Source | Tag |
|---|---|---|
| CWRU canonical URL `engineering.case.edu/bearingdatacenter` is live | `engineering.case.edu/bearingdatacenter` (verified 2026-10-11) | `[DIRECT]` |
| Public CWRU file tables (12k DE, 12k FE, 48k DE, normal baseline) list only 0.007", 0.014", 0.021" fault diameters. The 0.028" and 0.040" diameters are apparatus-documented on `fault-specifications` but have NO `.mat` link in any of the 4 audited download tables. | `engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data`, `/12k-fan-end-bearing-fault-data`, `/48k-drive-end-bearing-fault-data`, `/normal-baseline-data`, `/fault-specifications` (all verified 2026-10-11) | `[DIRECT]` |
| CWRU motor loads 0/1/2/3 HP at 1797/1772/1750/1720 RPM | `engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data` (verified 2026-10-11) | `[DIRECT]` |
| CWRU BDC accelerometer model is NOT officially named on the public engineering.case.edu site. The download page only states channel roles (DE/FE/BA). | `engineering.case.edu/bearingdatacenter/download-data-file` (verified 2026-10-11) | `[DIRECT]` |
| P15 (Hakam 2026) uses an AS-062 IEPE accelerometer on an ESP32 + NanoPi Neo rig, evaluates SVM/RF/k-NN/MLP/CNN/ANN/RNN on the Kaggle `medlamane/small-vibration-dataset-of-ccsiepe-sensor-as-062` dataset, and does NOT use CWRU, does NOT use a PCB 352C33, and does NOT use an STM32F4. RF F1=99.17±0.15%, SVM F1=98.99±0.20%, CNN F1=93.16±0.86% (5-fold CV on the Kaggle IEPE set). | ScienceDirect `S2468227626001924` (P15 paper, re-verified 2026-10-11) and ResearchGate publication 403928635 | `[DIRECT]` |
| PCB 352C33 is a real PCB Piezotronics accelerometer (100 mV/g, 0.5–10 kHz, ceramic shear) used in the XJTU-SY bearing dataset and in custom rigs (e.g., Vishwakarma Institute PMC article); it is NOT named in the CWRU BDC public sources. | `pcb.com/products?m=352C33`; XJTU-SY official `biaowang.tech/xjtu-sy-bearing-datasets/` (verified 2026-10-11) | `[DIRECT]` (PCB sensor datasheet); `[INDEPENDENT DATASET — NOT CWRU]` |
| 1D-CNN on CWRU, 12 kHz, 99.1% accuracy on CWRU/PU/MFPT | P16 (El Boughardini 2026) | `[DIRECT]` |
| Quantized 1D-CNN on ESP32, 98.7% on CWRU | P20 (Cotrino 2026) | `[DIRECT]` |
| RMS / kurtosis / crest features on CWRU + custom motor | P38 (Kolok 2025) | `[DIRECT]` |
| CWRU ↔ PU ↔ SEU is the ratified cross-machine triad | Decision log 2026-10-07 | `[DIRECT]` |
| ~~CWRU fault diameters are 0.007", 0.014", 0.021" only~~ — Phase 11 CORRECTION (now also superseded): per engineering.case.edu/.../48k-drive-end-bearing-fault-data (verified 2026-10-11), the public set includes 0.028" as well. | Phase 11 audit, 2026-10-11 | `[MISATTRIBUTION CORRECTED by Phase 15]` |
| **Phase 15 supersession (2026-10-11)**: the Phase 11 "0.028" in public set" claim is itself factually wrong. The audited 48 kHz DE table contains only 0.007/0.014/0.021" rows; the 0.028" diameter is apparatus-documented on `fault-specifications` (DE IR + DE Ball NTN, 0.050"/0.150" depth) but has NO `.mat` link in the 48 kHz DE table, the 12 kHz DE table, or the 12 kHz FE table. The public file inventory is therefore 0.007"/0.014"/0.021" only. | Phase 14/15 audit, 2026-10-11 | `[CONFLICTING EVIDENCE RESOLVED]` |
| ~~CWRU sensor is PCB 352C33~~ — REMOVED: the PCB 352C33 attribution was based on P15 (Hakam 2026), but P15 does not use CWRU data, does not use a PCB 352C33 sensor, and does not use an STM32F4. The CWRU BDC accelerometer model is NOT officially named on the public engineering.case.edu site. The model is now recorded as UNVERIFIED. | Phase 14 audit, 2026-10-11; Phase 15 manifest patch | `[MISATTRIBUTION CORRECTED]` |
| ~~P15 §3 (Hakam 2026) reproduces the canonical CWRU file list~~ — REMOVED: P15 does not use CWRU data and does not reproduce the CWRU file list. | Phase 14 audit, 2026-10-11; Phase 15 manifest patch | `[MISATTRIBUTION CORRECTED]` |
| ~~BA excluded to match Vieira 2026 reporting~~ — REMOVED: "Vieira 2026" is not in `01_literature/paper-matrix.md` and could not be located by Phase 11 audit. The BA exclusion is a project-internal choice. | Phase 11 audit, 2026-10-11 | `[UNVERIFIED CLAIM REMOVED]` |
| ~~CWRU license is CC-BY 4.0~~ — DOWNGRADED: no explicit CC-BY 4.0 statement was found on the CWRU BDC site; no `/license`, `/terms`, `/copyright`, or `/about-this-site` page is accessible. The Zenodo 10987113 (2024-04) third-party mirror carries CC-BY-4.0 but that is a MIRROR statement, not the CWRU BDC's own terms. License is UNVERIFIED. | Phase 11/14/15 audit, 2026-10-11 | `[CONFLICTING EVIDENCE]` |
| ~~2024 CWRU refresh~~ — REMOVED: the 2024 Zenodo 10987113 is a third-party mirror, not an official CWRU release. The 2010 batch remains the most recent public revision on engineering.case.edu. | Phase 11 audit, 2026-10-11 | `[CONFLICT RESOLVED]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Confirm the CWRU release license text
   and the full current file inventory on
   `engineering.case.edu/bearingdatacenter`. The Phase 14 audit did
   not find an explicit CC-BY 4.0 statement on the CWRU BDC page
   (verified 2026-10-11), so the license must be confirmed directly
   before any redistribution.
2. `[TO VERIFY FROM SOURCE DOC]` Confirm exact rpm-per-load mapping for
   the files that will be loaded (needed for the IDFT 1X speed tracker
   ±15 % search window). The 48 kHz DE sub-page (verified 2026-10-11)
   lists 1797/1772/1750/1720 RPM at 0/1/2/3 HP, but the 12 kHz
   Fan-End sub-page has not been re-verified.
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the 3-class collapse
   (Normal / IR / OR) for Experiment 002. The previous "Vieira 2026"
   citation for the Ball-exclusion has been removed (could not be
   located); the Ball exclusion is now a project-internal choice and
   should be re-confirmed against the protocol-freeze decision log
   (2026-10-07) before Experiment 002 consumes CWRU.
4. `[TO VERIFY FROM SOURCE DOC]` Confirm the CWRU BDC accelerometer
   model directly from the readme.pdf bundled with the .mat files
   (or from the CWRU authors by direct contact) before any
   downstream claim about sensor model is made in EXP-002.

## Configuration binding

This dataset feeds Experiment 002 with `source=cwru` and `target=cwru`
(same-domain control) and as the source/target of the 4 cross-machine
pairs in the 6-pair transfer matrix:

| Transfer pair | Source | Target | Use |
|---|---|---|---|
| $T_{11}$ | CWRU | CWRU | Closed-domain control (different physical bearings, same rig) |
| $T_{12}$ | CWRU | PU | Cross-machine, dissimilar rig |
| $T_{13}$ | CWRU | SEU | Cross-machine, different testbed family (gearbox vs. motor) |
| $T_{21}$ | PU | CWRU | Reverse of $T_{12}$ |
| $T_{31}$ | SEU | CWRU | Reverse of $T_{13}$ |

The remaining 2 pairs ($T_{23}$, $T_{32}$) are sourced/destined to PU and
SEU and documented in their respective manifests.
