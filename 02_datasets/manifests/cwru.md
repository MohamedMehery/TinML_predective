# Dataset Manifest — D1: CWRU Bearing Data Center

> **Authoritative source for this manifest**: `01_literature/paper-matrix.md`
> rows P15 (Hakam 2026), P16 (El Boughardini 2026), P20 (Cotrino 2026), and
> P38 (Kolok 2025), plus the ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md` and the
> `00_lab/decision-log.md` Protocol Freeze (2026-10-07). Anything not
> supported by those documents is marked `[TO VERIFY FROM SOURCE DOC]`.

## Card (per repo template)

```
Dataset:               Case Western Reserve University Bearing Data Center
URL:                   https://engineering.case.edu/bearingdatacenter
                       [TO VERIFY FROM SOURCE DOC for the current mirror URL]
Institution / Year:    Case Western Reserve University, original release
                       2003 (most recent public revision 2010; 2024 refresh
                       status [TO VERIFY FROM SOURCE DOC])
Machine:               2-hp (1.49 kW) 4-pole induction motor, relubricated
                       drive-end and fan-end bearings, single-point
                       single-line-to-line fault seeding via EDM
Sensor:                PCB 352C33 accelerometers (Drive End + Fan End);
                       16-channel DAQ; shaft encoder for 1X reference
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
                       (~ 1730 / 1750 / 1772 / 1797 rpm in the public
                       release; exact mapping [TO VERIFY FROM SOURCE DOC])
Fault types:           Inner-race (IR), Outer-race (OR), Ball (BA) fault,
                       each at 0.007", 0.014", 0.021" diameter; normal
                       baseline (no fault); for Experiment 002 we collapse
                       to 3-class (Normal / IR / OR) per the triad
                       protocol (BA excluded to match Vieira 2026
                       reporting [TO VERIFY FROM SOURCE DOC])
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
Publicly available:    Yes
License:               CC-BY 4.0 (per the CWRU public release terms;
                       [TO VERIFY FROM SOURCE DOC] for the 2024 refresh)
Download size:         ~120 MB (P16 reports this; [TO VERIFY FROM SOURCE DOC]
                       for the full current release)
Documentation:         engineering.case.edu/bearingdatacenter (readme.pdf
                       bundled with the .mat files; README in
                       `01_literature/papers/Edge AI-powered vibration
                       monitoring system with IEPE sensors for
                       predictive maintenance in industrial machinery.pdf`
                       §3 reproduces the canonical file list)
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
                       and SEU, not within CWRU itself.
Suitable research questions:
                       EXP-C5C8-002 (Cross-Machine Invariance Transfer)
                       — specifically as the source for the 3 transfer
                       pairs T_CWRU→PU, T_CWRU→SEU, and as the
                       single-machine closed-domain control in T_CWRU→CWRU
                       (same-bearings different-run sanity check).
```

## Evidence grounding (which paper said what)

| Claim | Paper | Tag |
|---|---|---|
| 1D-CNN on CWRU, 12 kHz, 99.1 % accuracy on CWRU/PU/MFPT | P16 | `[DIRECT]` |
| IEPE / accelerometer front-end on CWRU + custom rig | P15 | `[DIRECT]` |
| Quantized 1D-CNN on ESP32, 98.7 % on CWRU | P20 | `[DIRECT]` |
| RMS / kurtosis / crest features on CWRU + custom motor | P38 | `[DIRECT]` |
| CWRU ↔ PU ↔ SEU is the ratified cross-machine triad | Decision log 2026-10-07 | `[DIRECT]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Confirm the 2024 CWRU release license text
   and the 2024 file inventory (`engineering.case.edu/bearingdatacenter`).
2. `[TO VERIFY FROM SOURCE DOC]` Confirm exact rpm-per-load mapping for the
   files that will be loaded (needed for the IDFT 1X speed tracker ±15 %
   search window).
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the fault class mapping for the
   Vieira 2026 3-class collapse (Normal / IR / OR).

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
