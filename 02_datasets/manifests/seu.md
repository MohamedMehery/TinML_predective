# Dataset Manifest — D3: Southeast University (SEU) Gearbox

> **Authoritative source for this manifest**: the canonical SEU DDS
> release at
> `https://mlmechanics.ics.uci.edu` (the URL cited in the original
> release paper abstract: Shao, McAleer, Yan, Baldi, IEEE Trans.
> Industrial Informatics 15(4):2446–2455, 2019,
> DOI 10.1109/TII.2018.2864759; re-verified 2026-10-11 via the
> IEEE Xplore document 8432110 abstract), with the GitHub mirror
> `https://github.com/cathysiyu/Mechanical-datasets` (README
> verified 2026-10-11, last repository commit 2018-01-17) as a
> secondary distribution channel, the ratified C5/C8 charter in
> `03_research_questions/candidate-questions.md`, and the
> `00_lab/decision-log.md` "Dataset Harmonization & Feasibility Audit"
> row (2026-10-07), which ratifies CWRU ↔ PU ↔ SEU as the cross-machine
> triad. SEU is the *gearbox* member of the triad and is the principal
> test of "different testbed family".
>
> **Important correction (2026-10-11, Phase 11 audit)**: The earlier
> draft of this manifest cited `01_literature/paper-matrix.md` row P27
> (Łuczak 2024, MDPI Electronics 13(2):452) as the primary source for
> the SEU DDS. Direct reading of the P27 PDF shows that P27 is a
> 6DOF-IMU fan-blade-demonstrator study, NOT an SEU DDS study, and
> P27 does not have a §Data Availability statement pointing to the
> SEU DDS. P27 is therefore NOT evidence for SEU. Anything not
> directly supported by the SEU DDS release README(s) or
> Shao 2019 IEEE TII paper is marked `[TO VERIFY FROM SOURCE DOC]`.
>
> **Phase 15 corrections (2026-10-11, evidence-based documentation
> patch)**: the Phase 14 audit re-verified the original-paper-cited
> UCI URL `mlmechanics.ics.uci.edu` (Shao 2019 IEEE TII §Abstract
> verbatim: "We created a repository including these datasets
> located at mlmechanics.ics.uci.edu"), clarified that the
> `cathysiyu/Mechanical-datasets` GitHub mirror also bundles a CWRU
> bearing subdataset that is NOT part of the SEU DDS, recorded the
> README-supported condition labels (20-0, 30-2) and the file-name-
> derived class inventory for the SEU bearing and gear subdatasets,
> and labelled the 12 kHz sampling-rate claim as
> `[PARTIALLY VERIFIED]` (the README does not state a sampling rate;
> the 12 kHz figure comes from a secondary source,
> VictorBauler/awesome-bearing-dataset, verified 2026-10-11).
> License, sensor model, and exact physical rpm/load-torque
> values remain UNVERIFIED or `[TO VERIFY FROM SOURCE DOC]`
> because no authoritative source could be located in the Phase 14
> audit.

## Card (per repo template)

```
Dataset:               Southeast University (SEU) Drivetrain Dynamics
                       Simulator (DDS) — "SEU DDS". The SEU DDS
                       consists of TWO sub-datasets (bearing + gear)
                       collected on the DDS rig. The
                       `cathysiyu/Mechanical-datasets` GitHub
                       repository also bundles a separate CWRU
                       bearing subdataset (per the README verbatim
                       "Bearing dataset is from Case Western
                       Reserve University bearing data center");
                       that bundled CWRU subset is NOT part of
                       SEU DDS and is documented under the CWRU
                       manifest.
URL:                   PRIMARY (per Shao 2019 IEEE TII abstract,
                       re-verified 2026-10-11):
                         https://mlmechanics.ics.uci.edu
                       MIRROR (Shao's own GitHub):
                         https://github.com/cathysiyu/Mechanical-datasets
                         (README verified 2026-10-11, last commit
                         2018-01-17)
                       The Shao 2019 paper does NOT cite the
                       GitHub URL as the canonical release; the
                       GitHub repo's "About" section has no
                       description, website, or topics (re-verified
                       2026-10-11), and the repository has no
                       LICENSE file. The UCI URL
                       (mlmechanics.ics.uci.edu) is therefore the
                       original-paper-cited canonical source; the
                       GitHub repo is an authoritative mirror but
                       not the primary.
Institution / Year:    Southeast University, School of Mechanical
                       Engineering (research group of Prof. Ruqiang
                       Yan; "These data are acquired on Drivetrain
                       Dynamics Simulator (DDS) by research group of
                       Pf. Ruqiang Yan, Southeaset University, China"
                       per the `gearbox/dds` file in the
                       cathysiyu/Mechanical-datasets repo, re-verified
                       2026-10-11). The SEU DDS public release
                       accompanies Shao, McAleer, Yan, Baldi, "Highly
                       Accurate Machine Fault Diagnosis Using Deep
                       Transfer Learning", IEEE Trans. Industrial
                       Informatics 15(4):2446-2455, Article 8432110,
                       April 2019, DOI 10.1109/TII.2018.2864759
                       (bibliographic details verified 2026-10-11
                       via IEEE Xplore document 8432110 and the
                       Xi'an Jiaotong University scholar page).
                       The `cathysiyu/Mechanical-datasets` GitHub
                       mirror's last commit is dated 2018-01-17,
                       consistent with the SEU DDS being 2018-era
                       data, NOT 2024-era. The previously-cited
                       2024 attribution to P27 was incorrect (see
                       header note).
Machine:               SEU drivetrain dynamometer (DDS); the rig is a
                       gearbox + bearings combination, which is
                       structurally different from CWRU's induction
                       motor and PU's modular test rig. The gear-mesh
                       excitation is a major additional spectral
                       component that the C5/C8 protocol explicitly
                       addresses via "SEU gear-mesh envelope isolation"
                       in the decision log "Protocol Design Gate Freeze"
                       row.
Sensor:                Vibration accelerometer + motor torque. Per the
                       `cathysiyu/Mechanical-datasets` README each
                       file contains 8 rows of signals: (1) motor
                       vibration, (2-4) planetary gearbox vibration
                       in x/y/z, (5) motor torque, (6-8) parallel
                       gearbox vibration in x/y/z. The README
                       explicitly states "Signals of rows 2,3,4 are
                       all effective" for the bearing sub-dataset.
                       Specific sensor model [TO VERIFY FROM SOURCE
                       DOC] — the README does not name a sensor
                       model.
Sampling Rate:         [PARTIALLY VERIFIED] Secondary source
                       (VictorBauler/awesome-bearing-dataset, GitHub,
                       re-verified 2026-10-11) states: "Data
                       Characteristics: Contains multivariate time-
                       series data with 8 channels, including
                       vibration from multiple components and motor
                       torque. Vibration was sampled at 12 kHz." The
                       `cathysiyu/Mechanical-datasets` README does
                       NOT state a sampling rate; the Shao 2019 IEEE
                       TII paper was not accessed in full text in the
                       Phase 14 audit (only the IEEE Xplore abstract
                       and ResearchGate publication 403928635 were
                       verified). The 12 kHz figure is therefore
                       PARTIALLY VERIFIED — the secondary source
                       supports it, the README does not contradict
                       it, but the authoritative Shao 2019 paper §
                       Experimental Setup has not been independently
                       re-read. For the 12 kHz harmonization in
                       `experiment_config.json`, the secondary-source
                       12 kHz figure is consistent with
                       `experiment_config.json`
                       `sampling_rate_harmonization.target_fs_hz =
                       12000` but is not primary-source verified.
Raw signal:            Yes (raw vibration time series; some downstream
                       papers convert to 2D CWT scalograms, but the
                       raw 1D signal is the input for Experiment 002's
                       R1/R2/R3 pipeline. NOTE: P27 Łuczak 2024 is
                       NOT an SEU paper; P27 converts its own
                       6DOF-IMU fan demonstrator to CWT, not the
                       SEU DDS.)
Number of machines:    1 drivetrain testbed (DDS) per the
                       `cathysiyu/Mechanical-datasets` README.
                       SEU DDS has TWO sub-datasets (bearing + gear)
                       on the same DDS rig.
Operating conditions:  Per the `cathysiyu/Mechanical-datasets` README
                       (re-verified 2026-10-11) the DDS records at
                       TWO rotating-speed / load configurations:
                       "20-0" and "30-2". Per the awesome-bearing-
                       dataset summary, "20-0" = 20 Hz rotating
                       speed, 0 V load, and "30-2" = 30 Hz rotating
                       speed, 2 V load. Exact physical rpm and
                       load-torque values [TO VERIFY FROM SOURCE
                       DOC] from the Shao 2019 paper §Experimental
                       Setup.
Fault types:           The SEU DDS has TWO sub-datasets. Per the
                       `cathysiyu/Mechanical-datasets` GitHub
                       directory listing (re-verified 2026-10-11),
                       the file inventory is:
                         (a) SEU bearing subdataset
                             (`gearbox/bearingset/`):
                             5 classes (health, ball, inner, outer,
                             comb) × 2 conditions (20-0, 30-2) =
                             10 files:
                               health_20_0.csv, health_30_2.csv,
                               ball_20_0.csv,   ball_30_2.csv,
                               inner_20_0.csv,  inner_30_2.csv,
                               outer_20_0.csv,  outer_30_2.csv,
                               comb_20_0.csv,   comb_30_2.csv.
                             "comb" = combined fault (multiple
                             elements); the directory's commit
                             message says "4 kinds of bearing
                             faults" but the file inventory shows
                             4 fault types + 1 healthy = 5
                             classes total.
                         (b) SEU gear subdataset (`gearbox/gearset/`):
                             5 classes (Health, Chipped, Miss,
                             Root, Surface) × 2 conditions
                             (20-0, 30-2) = 10 files:
                               Health_20_0.csv, Health_30_2.csv,
                               Chipped_20_0.csv, Chipped_30_2.csv,
                               Miss_20_0.csv, Miss_30_2.csv,
                               Root_20_0.csv, Root_30_2.csv,
                               Surface_20_0.csv, Surface_30_2.csv.
                       For Experiment 002 the cross-machine collapse
                       is:
                         Healthy / Gear-fault / Bearing-fault
                       (any finer granularity is not present in
                       CWRU / PU at the same level and so cannot be
                       used in the cross-machine comparison).
                       The decision log explicitly notes "SEU
                       gear-mesh envelope isolation" as one of the
                       three technical closures of the C5/C8 protocol
                       freeze. (P27 Łuczak 2024 is NOT the source
                       for SEU fault classes — the canonical source
                       is the `cathysiyu/Mechanical-datasets`
                       directory listing and the Shao 2019 paper.)
Healthy data:          Yes (health/Health files at 20-0 and 30-2)
Fault data:            Yes (ball, inner, outer, comb for bearings;
                       Chipped, Miss, Root, Surface for gears)
Labels:                Per-file metadata; the class is encoded in the
                       filename prefix (health, ball, inner, outer,
                       comb for the bearing subdataset; Health,
                       Chipped, Miss, Root, Surface for the gear
                       subdataset). Per Lessmeier-style combined-
                       damage caveats, the SEU "comb" class may
                       itself be a multiple-damage bearing; this is
                       not stated in the README and would need
                       primary-source verification before any
                       class-collapse approximation is documented.
Standard train/test protocol:
                       Various downstream SEU-DDS studies (NOT P27,
                       which is a 6DOF-IMU study) use random k-fold
                       cross-validation on 2D CWT-scalogram or 1D
                       vibration representations; we reject this in
                       Experiment 002 because it leaks physical
                       bearing / gear identity across folds. Experiment
                       002 uses physical-bearing-UID + run-level
                       separation per charter §2.
Publicly available:    Yes (per the original-paper-cited UCI URL
                       `mlmechanics.ics.uci.edu` and the
                       `cathysiyu/Mechanical-datasets` public GitHub
                       mirror).
License:               UNVERIFIED. The `cathysiyu/Mechanical-datasets`
                       repository has NO LICENSE file at its root
                       (verified 2026-10-11). No explicit CC-BY 4.0
                       statement was located. The MDPI Electronics
                       open-data policy applies to MDPI papers,
                       NOT to the SEU DDS itself, and therefore
                       does NOT establish a CC-BY 4.0 license for
                       the SEU DDS. The UCI
                       `mlmechanics.ics.uci.edu` license was not
                       re-checked in the Phase 14 audit and
                       therefore also remains UNVERIFIED. The SEU
                       DDS license remains UNVERIFIED and must be
                       confirmed before any redistribution.
Download size:         [TO VERIFY FROM SOURCE DOC] (not stated in
                       the `cathysiyu/Mechanical-datasets` README
                       and not located in secondary sources).
Documentation:         PRIMARY:
                         Shao, McAleer, Yan, Baldi, "Highly
                         Accurate Machine Fault Diagnosis Using
                         Deep Transfer Learning", IEEE Trans.
                         Industrial Informatics 15(4):2446-2455,
                         April 2019, DOI
                         10.1109/TII.2018.2864759 (IEEE Xplore
                         document 8432110, re-verified 2026-10-11).
                       MIRROR:
                         `https://github.com/cathysiyu/Mechanical-datasets`
                         README (verified 2026-10-11). The
                         previously cited P27
                         (electronics-13-00452.pdf) is NOT a source
                         for the SEU DDS — P27 is a 6DOF-IMU
                         fan-blade demonstrator study that does not
                         mention the SEU DDS and does not have a
                         §Data Availability URL.
Known limitations:     (a) SEU is a *gearbox* rig, not just a
                       bearing rig. The gear-mesh excitation is at a
                       much higher order than the bearing-fault
                       characteristic orders. The C5/C8 protocol
                       freeze explicitly addresses this via "SEU
                       gear-mesh envelope isolation" (decision log
                       2026-10-07), which must be implemented in
                       `src/preprocess.py` before SEU enters the
                       transfer matrix. (b) Some downstream SEU-DDS
                       studies (NOT P27) use 2D CWT scalograms,
                       which exceed the 8.2 kB RAM budget of the
                       charter, so the 2D path is NOT used in
                       Experiment 002 — the protocol uses R3_Order
                       (1D, MLP) on the SEU channel instead. (c) The
                       casing transfer function of the SEU
                       drivetrain is the third distinct h_casing(t)
                       in the triad; it is the most dissimilar from
                       CWRU's motor casing and therefore the hardest
                       test of the cross-machine hypothesis. (d) The
                       SEU DDS sampling rate is only PARTIALLY
                       VERIFIED (secondary source says 12 kHz,
                       README is silent). (e) The `cathysiyu/
                       Mechanical-datasets` GitHub repository also
                       bundles a CWRU bearing subdataset that is
                       NOT part of SEU DDS; that bundled CWRU
                       subset is out-of-scope for EXP-002 SEU
                       experiments and is documented under the
                       CWRU manifest.
Suitable research questions:
                       EXP-C5C8-002 — as the "different testbed
                       family" target of T_CWRU→SEU and T_PU→SEU and
                       as the source of T_SEU→CWRU, T_SEU→PU,
                       T_SEU→SEU (closed-domain control).
```

## Evidence grounding (which source said what)

| Claim | Source | Tag |
|---|---|---|
| Original-paper-cited canonical SEU DDS URL is `https://mlmechanics.ics.uci.edu` (verbatim from Shao 2019 IEEE TII abstract: "We created a repository including these datasets located at mlmechanics.ics.uci.edu") | IEEE Xplore document 8432110 abstract (re-verified 2026-10-11) | `[DIRECT]` |
| Shao, McAleer, Yan, Baldi (2019) bibliographic details: IEEE Trans. Industrial Informatics 15(4):2446-2455, April 2019, DOI 10.1109/TII.2018.2864759, Article 8432110, Date of Publication 10 August 2018 | IEEE Xplore document 8432110 + Xi'an Jiaotong University scholar page (re-verified 2026-10-11) | `[DIRECT]` |
| `cathysiyu/Mechanical-datasets` GitHub mirror README (re-verified 2026-10-11): "Bearing dataset is from Case Western Reserve University bearing data center" — i.e., the GitHub repo bundles a CWRU bearing subdataset that is NOT part of SEU DDS | `https://github.com/cathysiyu/Mechanical-datasets` README + `https://csegroups.case.edu/bearingdatacenter/...` legacy URL referenced in the README | `[DIRECT]` |
| SEU DDS was acquired on the Drivetrain Dynamics Simulator (DDS) by the research group of Prof. Ruqiang Yan, Southeast University, China (typo "Southeaset" verbatim in the original) | `cathysiyu/Mechanical-datasets/blob/master/gearbox/dds` (re-verified 2026-10-11) | `[DIRECT]` |
| SEU DDS has 2 sub-datasets (bearing + gear) on the DDS rig | `cathysiyu/Mechanical-datasets` README (verified 2026-10-11) | `[DIRECT]` |
| SEU DDS records 8 channels (motor vibration, planetary x/y/z, motor torque, parallel x/y/z) | `cathysiyu/Mechanical-datasets` README (verified 2026-10-11) | `[DIRECT]` |
| SEU DDS records at 2 rotating-speed/load configurations: 20-0 and 30-2 | `cathysiyu/Mechanical-datasets` README (verified 2026-10-11) | `[DIRECT]` |
| Channels 2,3,4 (planetary gearbox x/y/z) are the effective bearing channels | `cathysiyu/Mechanical-datasets` README (verified 2026-10-11) | `[DIRECT]` |
| SEU bearing subdataset file inventory: 5 classes (health, ball, inner, outer, comb) × 2 conditions (20-0, 30-2) = 10 files | `cathysiyu/Mechanical-datasets/tree/master/gearbox/bearingset` directory listing (verified 2026-10-11) | `[DIRECT]` |
| SEU gear subdataset file inventory: 5 classes (Health, Chipped, Miss, Root, Surface) × 2 conditions (20-0, 30-2) = 10 files | `cathysiyu/Mechanical-datasets/tree/master/gearbox/gearset` directory listing (verified 2026-10-11) | `[DIRECT]` |
| SEU DDS sampling rate = 12 kHz (vibration) | VictorBauler/awesome-bearing-dataset GitHub README (re-verified 2026-10-11) | `[PARTIALLY VERIFIED — secondary source; primary README silent]` |
| CWRU ↔ PU ↔ SEU triad ratified as cross-machine benchmark | Decision log 2026-10-07 | `[DIRECT]` |
| "SEU gear-mesh envelope isolation" is one of the three C5/C8 protocol closures | Decision log 2026-10-07, Protocol Freeze row | `[DIRECT]` |
| ~~SEU gearbox used with CWT-Morlet 2D ResNet~~ — REMOVED: P27 is NOT an SEU paper. The actual SEU-DDS release paper is Shao, McAleer, Yan, Baldi (2019), IEEE Trans. Industrial Informatics 15(4):2446-2455, DOI 10.1109/TII.2018.2864759. P27 (Łuczak 2024, MDPI Electronics 13(2):452) is a 6DOF-IMU fan-blade demonstrator study with no SEU content. | Phase 11 audit, 2026-10-11 | `[CONFLICT RESOLVED]` |

## Required pre-experiment checks (before Experiment 002 consumes this dataset)

1. `[TO VERIFY FROM SOURCE DOC]` Pull the canonical license text
   from the SEU DDS release. The
   `cathysiyu/Mechanical-datasets` GitHub mirror (verified
   2026-10-11) has NO LICENSE file at its root, and the UCI
   `mlmechanics.ics.uci.edu` license was not re-checked in the
   Phase 14 audit, so the SEU DDS license remains UNVERIFIED.
2. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU DDS native sampling
   rate from the Shao 2019 IEEE TII paper §Experimental Setup
   directly. The `cathysiyu/Mechanical-datasets` README does not
   state a sampling rate; the 12 kHz figure cited above is from a
   secondary source (awesome-bearing-dataset). Until the Shao 2019
   paper §Experimental Setup is independently re-read, the 12 kHz
   figure is `[PARTIALLY VERIFIED]`.
3. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU DDS fault class
   labels (5 bearing classes health/ball/inner/outer/comb and 5
   gear classes Health/Chipped/Miss/Root/Surface per the file
   inventory) and align with the Healthy / Gear-fault /
   Bearing-fault collapse used in Experiment 002. Note that the
   "comb" bearing class may itself be a multiple-damage bearing
   (analogous to Paderborn KB23/KB24/KB27); the README does not
   state this and the Shao 2019 paper §Experimental Setup should
   be consulted.
4. `[TO VERIFY FROM SOURCE DOC]` Confirm the SEU DDS operating-
   condition enumeration (rotating speed 20 Hz / 30 Hz, load 0 V /
   2 V per the secondary awesome-bearing-dataset summary; exact
   physical rpm and load-torque values needed for the IDFT 1X
   speed tracker ±15 % search window).

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
