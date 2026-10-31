## 1. Summary

- All reported Stage 1 results are conditional on the **gate-passing subset: 599 of 1,020 eligible recordings**; 421 recordings were excluded by the unchanged timestamp-residual or endpoint-rate gates.
- Across five seeds, R3_Order exceeded R1_Raw on the in-condition test view (0.584 ± 0.008 vs 0.311 ± 0.034) and the radial-load-change view (0.550 ± 0.018 vs 0.305 ± 0.017).
- On the primary N09 speed-shift view, R3_Order scored 0.109 ± 0.015 and R1_Raw 0.151 ± 0.033; both were below both reported baselines.
- **The primary speed-shift hypothesis is not supported.**
- Diagnostic B found substantial above-angular-Nyquist envelope power in every condition; it did not rule aliasing in or out as a contributor and did not establish a causal classification effect.
- No failure cause is assigned to N09. No normalization, envelope-filter, or measured-speed change was made after observing the Stage 1 test views; such changes belong to Stage 2.

## 2. Experimental question and design

Stage 1 asked whether the order-envelope representation (R3_Order) improves three-class bearing-condition classification on held-out physical Paderborn bearings relative to the raw time-domain representation (R1_Raw), when trained on N15_M07_F10 and evaluated on (a) the same operating condition, (b) the primary N09_M07_F10 speed-shift condition, and (c) the N15_M07_F04 radial-load-change condition. The allocation is by physical bearing; the three evaluation conditions are views of the same held-out test-bearing allocation, not three independent test sets.

The owner-run comparison used five seeds in this order: 42, 1337, 2026, 7, 99; three fixed labels—Healthy (0), Outer (1), Inner (2); one source-train partition, one validation partition, and three test views. The models were Compact1DCNN for R1_Raw and StandardMLP for R3_Order. Results below use owner-supplied run outputs, not a run performed in this workspace. The Stage 1 split maps train and validation to N15_M07_F10, test-in-condition to N15_M07_F10, test-primary to N09_M07_F10, and test-secondary to N15_M07_F04 (`04_experiments/002_cross_machine_invariance/src/stage1_split.py:35–40`, code commit `980ad2988d4467303e5d1d3cbb04c988b93596ca`).

## 3. Data, eligibility, and gate attrition

The owner reports an eligible inventory of 1,020 recordings across the approved bearing-by-condition pairs. The unchanged quality gates passed 599 and excluded 421: 293 first failed the timestamp-residual gate and 128 first failed the endpoint-rate gate. The cause of a timestamp residual is not identified by that gate; no cause is assigned here. This analysis therefore describes only the complete cases that passed both gates and makes no claim that they represent the excluded recordings or are free of selection bias.

| Audit dimension | Passed / eligible |
|---|---:|
| Train partition | 172 / 260 |
| Validation partition | 50 / 100 |
| Test-in-condition partition | 114 / 220 |
| Test-primary partition | 146 / 220 |
| Test-secondary partition | 117 / 220 |
| Healthy | 114 / 200 |
| Outer | 264 / 440 |
| Inner | 221 / 380 |
| N15_M07_F10 | 336 / 580 |
| N09_M07_F10 | 146 / 220 |
| N15_M07_F04 | 117 / 220 |

The owner reports zero required-pair coverage gaps and a minimum of seven passing recordings for every required pair. The unchanged reader thresholds are an endpoint-rate relative tolerance of 0.001 and a maximum absolute timestamp residual of 1/64,000 s; the checks use the original vibration time axis (`tools/paderborn_mat_reader/recording.py:65–72,153–168`; `tools/paderborn_mat_reader/reader.py:370–445`). Gate exclusions were opt-in in the Stage 1 runner; the runner documents that it excludes only the recognized first-failure cases, not arbitrary read errors (`04_experiments/002_cross_machine_invariance/src/run_stage1.py:198–210`, same code commit).

Gate-passing loaded windows and physical-bearing counts, with class counts in label order [0, 1, 2] = [Healthy, Outer, Inner]:

| Partition / view | Windows | Physical bearings | Class counts [0,1,2] |
|---|---:|---:|---:|
| Train | 7,740 | 13 | [1,935, 2,790, 3,015] |
| Validation | 2,250 | 5 | [405, 1,080, 765] |
| Test-in-condition | 5,133 | 11 | [810, 2,564, 1,759] |
| Test-primary | 6,570 | 11 | [1,125, 3,060, 2,385] |
| Test-secondary | 5,265 | 11 | [855, 2,385, 2,025] |

The same 11 test bearings are used in all three views. Window counts are not counts of independent physical replicates.

## 4. Executed protocol and implementation audit

All repository-code citations in this document refer to code commit `980ad2988d4467303e5d1d3cbb04c988b93596ca`; line numbers are in the repository paths shown.

| Protocol element | Executed behavior and audit reference |
|---|---|
| Input and decimation | The Paderborn reader records the nominal operating condition and 64 kHz nominal sampling rate (`tools/paderborn_mat_reader/recording.py:18–26`). The experiment decimates vibration from 64 kHz to 12 kHz with the configured polyphase Kaiser path (`04_experiments/002_cross_machine_invariance/src/datasets.py:610–624`; `04_experiments/002_cross_machine_invariance/src/preprocess_decim.py:34–58,61–81`). |
| Windowing | Windows are 2,048 samples with 1,024-sample hop and a 128-sample angular-resampling target (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:18–22`; `04_experiments/002_cross_machine_invariance/src/datasets.py:616–624`; `04_experiments/002_cross_machine_invariance/src/preprocess.py:194–214`). |
| Envelope path | The signal is mean-centered; a fourth-order 500–5,000 Hz Butterworth band-pass is applied, then a Hilbert-magnitude envelope is formed and mean-centered (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:23–34`; `04_experiments/002_cross_machine_invariance/src/preprocess.py:47–63,187–192`). No envelope anti-alias filter is present before angular resampling (`04_experiments/002_cross_machine_invariance/src/preprocess.py:187–246`). |
| R3_Order | The envelope window is passed to `estimate_1x_speed`; that returned value is passed to `angular_resample`, followed by a Hann-windowed 128-point RFFT (`04_experiments/002_cross_machine_invariance/src/preprocess.py:224–246`). With 64 samples/revolution and a 128-sample target, the grid spans two revolutions; the RFFT yields 65 natural magnitude bins from order 0 through 32 at 0.5-order spacing. The output vector is 1,024 long, but bins 65–1,023 are zero in this code path; the separate zeroing from bin 512 onward adds no additional padding. The config says `max_orders: 16` and `order_output_bins: 512`, which do not describe the effective 0–32-order, 65-populated-bin output here (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:40–49`). |
| Normalization | Z-score means and standard deviations are fitted on source-train rows only and applied unchanged to validation and each test view (`04_experiments/002_cross_machine_invariance/src/datasets.py:375–392`; `04_experiments/002_cross_machine_invariance/src/run_stage1.py:1056–1079`). |
| Class weights and optimization | Training-only inverse-frequency class weights use N/(3 n_c); the owner reports approximately [1.333, 0.925, 0.856] in [Healthy, Outer, Inner] order. The configured optimizer is AdamW (learning rate 0.001, weight decay 0.0001), batch size 32, at most 60 epochs, patience 12, and weighted cross-entropy (`04_experiments/002_cross_machine_invariance/src/train.py:59–101,245–256,276–304`; `04_experiments/002_cross_machine_invariance/config/experiment_config.json:97–106`). |
| Metrics and uncertainty | Macro-F1 and per-class F1 use the fixed three labels; the runner records per-bearing cluster-bootstrap intervals with 10,000 iterations and a 95% level (`04_experiments/002_cross_machine_invariance/src/evaluate.py:24–47,88–125`; `04_experiments/002_cross_machine_invariance/src/run_stage1.py:2097–2117`; `04_experiments/002_cross_machine_invariance/config/experiment_config.json:108–113`). The config names Hedges' g and a linear mixed-effects model, but those are not reported as implemented outcomes here. |
| Representation/model mapping | R1_Raw (1 × 2,048) uses Compact1DCNN; R3_Order uses StandardMLP on a 1,024-element vector whose actual nonzero order-spectrum values are bins 0–64, with bins 65–1,023 zero in this Paderborn path. The config's `zero_padded_from: 512` does not change that effective vector (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:52–60,87–95`; `04_experiments/002_cross_machine_invariance/src/models.py:27–32,58–78,84–95`; `04_experiments/002_cross_machine_invariance/src/preprocess.py:235–246`). |

**Source discrepancies—configured versus effective preprocessing.** The owner-supplied protocol description calls R1_Raw a “bandpassed time-domain window.” The reference implementation instead assigns R1_Raw from `w_raw`, a window cut from the mean-centered `sig` **before** band-pass filtering; `sig_bp` is used to form the envelope (`04_experiments/002_cross_machine_invariance/src/preprocess.py:187–192,211–222`). Also, the experiment config declares `detrend: linear` (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:23–25`), while this preprocessing path subtracts the signal mean and does not apply linear detrending (`04_experiments/002_cross_machine_invariance/src/preprocess.py:187–192`). The config also declares `max_orders: 16` and `order_output_bins: 512` (`04_experiments/002_cross_machine_invariance/config/experiment_config.json:40–49`), whereas this Paderborn path uses a 128-point angular grid and yields 65 natural RFFT bins through order 32; the remaining vector elements are zero (`04_experiments/002_cross_machine_invariance/src/datasets.py:616–624`; `04_experiments/002_cross_machine_invariance/src/preprocess.py:235–246`). These descriptions/configuration and the effective code behavior conflict. The owner-supplied R1_Raw description is preserved as reported and all three discrepancies are flagged; the code behavior above is not silently rewritten to match them.

## 5. Primary headline results

Owner-reported mean ± sample standard deviation (SD; n = 5 seeds):

| Test view | Operating condition | R1_Raw macro-F1 | R3_Order macro-F1 |
|---|---|---:|---:|
| Test-in-condition | N15_M07_F10 | 0.311 ± 0.034 | 0.584 ± 0.008 |
| Test-primary (speed shift) | N09_M07_F10 | 0.151 ± 0.033 | 0.109 ± 0.015 |
| Test-secondary (radial-load change) | N15_M07_F04 | 0.305 ± 0.017 | 0.550 ± 0.018 |

Owner-reported mean train / validation macro-F1 across seeds: R1_Raw 0.974 / 0.344; R3_Order 0.832 / 0.569. The train and validation values do not change the test-view interpretation. In-condition and secondary-view results favor R3_Order. The primary speed-shift result does not: R3_Order is lower than R1_Raw there, and both are below both baselines in Section 7. **The primary speed-shift hypothesis is not supported.**

## 6. Per-class results and fit diagnostics

Owner-reported mean per-class F1, in [Healthy, Outer, Inner] order:

| Test view | R1_Raw [Healthy, Outer, Inner] | R3_Order [Healthy, Outer, Inner] |
|---|---:|---:|
| Test-in-condition | [0.406, 0.139, 0.387] | [0.529, 0.612, 0.611] |
| Test-primary (N09_M07_F10) | [0.303, 0.151, 0] | [0.012, 0.167, 0.147] |
| Test-secondary (N15_M07_F04) | [0.417, 0, 0.497] | [0.483, 0.553, 0.614] |

These are means of the owner-reported per-seed class metrics; the report does not reconstruct the class metrics from predictions. On the primary view, both representations have low macro-F1 and each misses at least one class severely. No particular mechanical, sensor, preprocessing, or data-quality cause is inferred for the N09 result.

## 7. Baselines and uncertainty

| Test view | Majority-class macro-F1 | Uniform-random Monte Carlo macro-F1 |
|---|---:|---:|
| Test-in-condition | 0.2221 | 0.3172 |
| Test-primary (speed shift) | 0.2118 | 0.3208 |
| Test-secondary (radial-load change) | 0.2078 | 0.3198 |

Both primary-view representation means are below both primary-view baselines. This is part of why the primary speed-shift hypothesis is not supported; the baseline comparison is not a causal explanation of the failure.

The owner reports descriptive cluster-bootstrap intervals as follows. For R3_Order test-in-condition, seed 42 is [0.3231, 0.7501]; across the five seed-specific intervals, lower endpoints range 0.3170–0.3364 and upper endpoints 0.7501–0.7807. For R3_Order test-primary, seed 42 is [0.0468, 0.1475] and all seed-specific intervals lie within [0.0414, 0.1820]. For R1_Raw test-primary, all five lower endpoints are 0 and upper endpoints range 0.2069–0.3639. These intervals are descriptive, not evidence of independent window-level replication. Windows overlap within recordings; windows are not independent physical replicates, and five seeds are not five independent bearings. The same physical test bearings recur across views.

## 8. Diagnostics A and B

**Diagnostic A — synthetic implementation check (not a Paderborn performance test).** Synthetic orders 3 and 5 at 15 and 25 Hz peaked at bins 6 and 10, with zero location error; amplitude differences were −0.000037 and −0.000043. There were 0/128 unsupported samples. In the amplitude-modulated check, 22/22 windows peaked at order 5 at 25 Hz, with amplitude 0.49989–0.49991. Order 45 appeared as order 19 / bin 38 at approximately 97.8% of sampled-reference amplitude (2.153% lower). No envelope anti-alias filter was present. This validates only the tested synthetic signals and does not validate real-data classification or exclude real-data aliasing.

**Diagnostic B — owner-run real-data envelope diagnostic.** The diagnostic results below are owner-supplied; they were not reproduced in this workspace. The following interpretation is retained verbatim:

> “The real-data diagnostic found substantial envelope power above the applicable angular Nyquist frequency in ALL three operating conditions (median fractions 0.42–0.49).”
>
> “The median fraction was HIGHER in N09_M07_F10 (0.4857) than in N15_M07_F10 (0.4193), a difference of +0.0664. The paired bearing-level comparison is consistent: median Δ(N15 − N09) = −0.059 across the 11 shared test bearings.”
>
> “Because above-Nyquist energy is large in every condition, this measurement alone does not explain why performance collapses specifically on N09. It does NOT show that aliasing is absent, harmless, or ruled out as a contributor.”
>
> “This diagnostic did NOT isolate or quantify any causal contribution of aliasing to classification performance. A signal-level counterfactual would be required to investigate that contribution, and even that would not by itself establish a classification effect.”

Additional owner-reported descriptive peaks: median R3 peak order by class was N15_M07_F10—Healthy 5.5, Inner 3.0, Outer 3.0; N09_M07_F10—Healthy 3.5, Inner 7.0, Outer 6.5. Peak magnitudes were N15 in-condition 1.903, N09 primary 1.029, and N15 secondary 1.718. A distribution shift under frozen N15 normalization is a hypothesis only, not an established explanation. Diagnostic B did not rule aliasing in or out as a contributor and did not establish a classification effect.

## 9. Speed-estimation behavior (Section 2.5a reconciliation)

The condition map assigns N09 and N15 nominal speeds of 900 and 1,500 rpm, respectively (15 and 25 Hz after conversion in the dataset loader; `tools/paderborn_mat_reader/recording.py:18–22`; `04_experiments/002_cross_machine_invariance/src/datasets.py:610–612`). At the Stage 1 sampling rate and estimator FFT length, the bin spacing is 12,000 / 2,048 = 5.859375 Hz. The nominal 15 Hz search band (±15%) contains zero positive-frequency bins; the nominal 25 Hz search band contains one (23.4375 Hz). The estimator's fewer-than-three-bins branch returns the supplied nominal speed (`04_experiments/002_cross_machine_invariance/src/preprocess.py:65–90`; it is called on each 2,048-point envelope window at `04_experiments/002_cross_machine_invariance/src/preprocess.py:224–234`).

| Event / counter | Exact definition | Owner/source outcome |
|---|---|---|
| Raw estimator nominal return | `estimate_1x_speed` returns `nominal_hz` when fewer than 3 bins lie in the ±15% search band. | Source-derived: 26,958 of 26,958 windows. This is not a zero-count event. |
| Effective speed passed to `angular_resample` | The raw estimator return `f_speed` is passed directly to the resampler. | Owner-reported and source-consistent: nominal 15 Hz or 25 Hz for all 26,958 windows. |
| Downstream invalid-speed replacement | `angular_resample` replaces speed only if `speed_hz <= 0` or `speed_hz > 200`, setting it to `revs * fs / N` (`04_experiments/002_cross_machine_invariance/src/preprocess.py:104–157`, especially `127–130`). | Source-derived: 0 of 26,958 invocations; nominal 15 Hz and 25 Hz do not trigger this guard. |
| Diagnostic B owner counter | Owner text says “fallbacks occurred in 0 windows,” without specifying which mechanism. | Owner-reported value: 0; counter implementation/definition was not supplied. |

These are distinct events and counts; do not merge the estimator's 26,958 nominal returns with the downstream guard's zero invocations. The Diagnostic B zero is consistent if the owner-side counter counts downstream invalid-speed replacements. If it was intended to count estimator nominal returns, that value is **CONFLICTING / UNRESOLVED** against the source-derived 26,958. The source behavior and effective speed are clear; the counter's intended event remains **UNRESOLVED** from the supplied diagnostic materials. The word “fallbacks” above is retained only as the owner-supplied label and is explicitly disambiguated here.

The measured shaft-speed channel is retained in per-window metadata (`04_experiments/002_cross_machine_invariance/src/datasets.py:633–691`); it is not passed to `estimate_1x_speed` or `angular_resample` in this path. The R3 speed behavior above uses the estimator output, not measured-speed substitution.

## 10. Limitations

1. **Gate-conditioned subset:** results apply to 599/1,020 eligible recordings. The 421 exclusions (293 timestamp-residual, 128 endpoint-rate) may affect representativeness; no no-selection-bias claim is made.
2. **Gate causes:** a first-failure category is not a diagnosis of why a recording failed. In particular, no cause is assigned to timestamp residuals.
3. **Small physical test sample:** each view covers 11 held-out physical bearings; test-view windows do not increase the number of independent physical replicates.
4. **Repeated test bearings:** the same 11 bearings occur in each operating-condition view, so view comparisons are paired/repeated, not independent-sample comparisons.
5. **Overlapping windows:** 2,048-sample windows use 1,024-sample hops; adjacent windows from the same recording are dependent.
6. **Seeds are not bearings:** five optimization seeds measure run-to-run variability, not five new physical specimens.
7. **Single dataset/rig context:** the experiment evaluates this Paderborn dataset and its approved condition/bearing allocation; it does not establish cross-dataset or cross-machine generality.
8. **Condition coverage is finite:** only the source N15_M07_F10, primary N09_M07_F10, and secondary N15_M07_F04 conditions are represented here.
9. **N09 performance cause is unknown:** the measurements do not identify a cause of the primary-view collapse.
10. **Aliasing remains unresolved:** above-Nyquist energy is substantial in every condition. Neither Diagnostic A nor B establishes the causal contribution of aliasing to classification performance.
11. **Confidence intervals are descriptive:** cluster-bootstrap intervals summarize these observed bearing clusters; they do not make windows independent, and the five seed intervals are not independent-bearing replications.
12. **Configured/effective preprocessing discrepancies:** the owner-supplied description of R1_Raw as band-passed conflicts with the reference code, which windows the mean-centered signal before band-pass; the config's `detrend: linear` conflicts with mean-subtraction-only code; and the configured R3 `max_orders: 16` / `order_output_bins: 512` do not match the effective 65 populated bins through order 32. The exact behavior is documented in Section 4; resolve these discrepancies before follow-up protocol claims rely on the configured descriptions.

## 11. Conclusions

For the gate-passing subset, R3_Order performed better than R1_Raw on the in-condition view and the radial-load-change view. The primary N09_M07_F10 speed-shift result was poor for both representations; R3_Order was lower than R1_Raw, and both were below both baselines. **The primary speed-shift hypothesis is not supported.** This is a negative Stage 1 result for that hypothesis, not evidence of a particular failure mechanism. Diagnostic B shows substantial above-Nyquist envelope power but does not rule aliasing in or out as a contributor and does not establish that aliasing caused the N09 result.

## 12. Post-Stage 1 boundary and Stage 2

The Stage 1 test views have already been observed. Do not revise Stage 1 preprocessing, normalization, envelope filtering, or speed input in response and then describe the resulting evaluation as an untouched Stage 1 test. Any future normalization change, envelope anti-alias filter, measured-speed substitution, or other speed-estimation change is a **Stage 2** intervention. Stage 2 must be separately specified and evaluated; any use of these already observed test views must be disclosed as post-Stage-1 reuse rather than presented as a new untouched test. The distribution-shift explanation remains a hypothesis only.

## 13. Appendix A — Per-seed test results (30 rows)

Values are owner-supplied. Seed order is 42, 1337, 2026, 7, 99. Each row lists macro-F1 and [Healthy, Outer, Inner] F1. Window and bearing counts are those reported for each test view.

| Seed | Representation | Test view | Macro-F1 | Healthy F1 | Outer F1 | Inner F1 | Windows | Bearings |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 42 | R1_Raw | Test-in-condition | 0.3412 | 0.4347 | 0.1904 | 0.3984 | 5,133 | 11 |
| 1337 | R1_Raw | Test-in-condition | 0.3004 | 0.4235 | 0.0134 | 0.4641 | 5,133 | 11 |
| 2026 | R1_Raw | Test-in-condition | 0.2712 | 0.4145 | 0.1363 | 0.2628 | 5,133 | 11 |
| 7 | R1_Raw | Test-in-condition | 0.3516 | 0.4103 | 0.1940 | 0.4505 | 5,133 | 11 |
| 99 | R1_Raw | Test-in-condition | 0.2906 | 0.3481 | 0.1629 | 0.3607 | 5,133 | 11 |
| 42 | R1_Raw | Test-primary (N09_M07_F10) | 0.1057 | 0.2938 | 0.0233 | 0 | 6,570 | 11 |
| 1337 | R1_Raw | Test-primary (N09_M07_F10) | 0.1711 | 0.3065 | 0.2069 | 0 | 6,570 | 11 |
| 2026 | R1_Raw | Test-primary (N09_M07_F10) | 0.1471 | 0.3014 | 0.1398 | 0 | 6,570 | 11 |
| 7 | R1_Raw | Test-primary (N09_M07_F10) | 0.1942 | 0.3119 | 0.2706 | 0 | 6,570 | 11 |
| 99 | R1_Raw | Test-primary (N09_M07_F10) | 0.1387 | 0.2998 | 0.1163 | 0 | 6,570 | 11 |
| 42 | R1_Raw | Test-secondary (N15_M07_F04) | 0.3190 | 0.4430 | 0 | 0.5140 | 5,265 | 11 |
| 1337 | R1_Raw | Test-secondary (N15_M07_F04) | 0.3085 | 0.4241 | 0 | 0.5014 | 5,265 | 11 |
| 2026 | R1_Raw | Test-secondary (N15_M07_F04) | 0.3185 | 0.4291 | 0 | 0.5263 | 5,265 | 11 |
| 7 | R1_Raw | Test-secondary (N15_M07_F04) | 0.3019 | 0.4048 | 0 | 0.5009 | 5,265 | 11 |
| 99 | R1_Raw | Test-secondary (N15_M07_F04) | 0.2763 | 0.3851 | 0 | 0.4438 | 5,265 | 11 |
| 42 | R3_Order | Test-in-condition | 0.5782 | 0.5255 | 0.6089 | 0.6003 | 5,133 | 11 |
| 1337 | R3_Order | Test-in-condition | 0.5954 | 0.5530 | 0.6292 | 0.6038 | 5,133 | 11 |
| 2026 | R3_Order | Test-in-condition | 0.5795 | 0.5232 | 0.6052 | 0.6102 | 5,133 | 11 |
| 7 | R3_Order | Test-in-condition | 0.5780 | 0.5239 | 0.5895 | 0.6205 | 5,133 | 11 |
| 99 | R3_Order | Test-in-condition | 0.5905 | 0.5201 | 0.6295 | 0.6219 | 5,133 | 11 |
| 42 | R3_Order | Test-primary (N09_M07_F10) | 0.1014 | 0.0114 | 0.1551 | 0.1376 | 6,570 | 11 |
| 1337 | R3_Order | Test-primary (N09_M07_F10) | 0.1346 | 0.0071 | 0.2217 | 0.1751 | 6,570 | 11 |
| 2026 | R3_Order | Test-primary (N09_M07_F10) | 0.0966 | 0.0107 | 0.1643 | 0.1149 | 6,570 | 11 |
| 7 | R3_Order | Test-primary (N09_M07_F10) | 0.1000 | 0.0141 | 0.1400 | 0.1459 | 6,570 | 11 |
| 99 | R3_Order | Test-primary (N09_M07_F10) | 0.1119 | 0.0178 | 0.1551 | 0.1629 | 6,570 | 11 |
| 42 | R3_Order | Test-secondary (N15_M07_F04) | 0.5292 | 0.4646 | 0.5271 | 0.5961 | 5,265 | 11 |
| 1337 | R3_Order | Test-secondary (N15_M07_F04) | 0.5704 | 0.4983 | 0.5850 | 0.6279 | 5,265 | 11 |
| 2026 | R3_Order | Test-secondary (N15_M07_F04) | 0.5365 | 0.4655 | 0.5396 | 0.6043 | 5,265 | 11 |
| 7 | R3_Order | Test-secondary (N15_M07_F04) | 0.5457 | 0.4764 | 0.5283 | 0.6325 | 5,265 | 11 |
| 99 | R3_Order | Test-secondary (N15_M07_F04) | 0.5675 | 0.5083 | 0.5852 | 0.6091 | 5,265 | 11 |

## 14. Appendix B — Run provenance

| Field | Owner-reported value |
|---|---|
| Run ID | `20261023T061739730879Z_38f09a20` |
| Run date | 2026-10-23 |
| Code commit | `980ad2988d4467303e5d1d3cbb04c988b93596ca` |
| Runner | `04_experiments/002_cross_machine_invariance/src/run_stage1.py` |
| Platform | Windows, local |
| Python / NumPy / SciPy / PyTorch | Python 3.13.1; NumPy 2.4.6; SciPy 1.18.1; PyTorch 2.14.1 CPU |
| Runs | 30 per-seed/view rows; 0 failed or skipped |
| Per-window predictions | 169,680 rows; owner reports file retained locally (file path and file hash not supplied here) |
| Selected-file-list SHA-256 | `42cd53004baa8fb18f39c679ce1d5a2c0393e6859e7baabadb5c28b7e4f8cf93` |
| Run-manifest SHA-256 | `59a4b9811c1216696803899a00b30aca18e01d1d28a084a3e6ba55c6e87a8195` |

The run manifest, selected-file list, predictions, and underlying MAT data were not present in this workspace. Run outputs, counts, hashes, and results above are reproduced from the owner-supplied record; they were not independently rehashed, re-run, recomputed, or rounded here. No Stage 1 training or prediction was performed in this workspace.

## 15. Appendix C — Bearing allocation

| Allocation | Physical bearings |
|---|---|
| Train (13) | `K002 K004 K006`; `KA01 KA05 KA07 KA04 KA15`; `KI01 KI07 KI04 KI14 KI18` |
| Validation (5) | `K005`; `KA09 KA16`; `KI03 KI17` |
| Test (11; shared by all three test views) | `K001 K003`; `KA03 KA06 KA08 KA22 KA30`; `KI05 KI08 KI16 KI21` |
| Excluded Combined bearings | `KB23 KB24 KB27` |

The three test conditions are evaluated on the same 11 held-out test bearings. The included/excluded bearing allocation is the owner-reported approved split; split code defines the condition mapping and validates disjoint allocations (`04_experiments/002_cross_machine_invariance/src/stage1_split.py:34–41,85–146`).
