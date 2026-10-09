# Experiment 002 — Cross-Machine Invariance Transfer (C5/C8)

**Experiment ID:** `EXP-C5C8-002`
**Branch:** `arena/6ad75829-tinml-predective`
**Status:** Configuration frozen at commit TBD. Code and results not yet executed.

---

## 1. Scientific Question

This experiment operationalizes the **ratified primary research direction** (Candidate B / C5/C8) formalized in [`03_research_questions/candidate-questions.md`](../../03_research_questions/candidate-questions.md) and ratified in [`00_lab/decision-log.md`](../../00_lab/decision-log.md) on 2026-10-07. The question is quoted verbatim from the charter:

> **"To what extent do dimensionless kinematic order-tracking representations provide zero-shot cross-machine diagnostic transferability by decoupling localized bearing fault excitations from machine-specific casing structural resonances, compared to data-driven spatial convolutional representations under constrained embedded inference?"**

## 2. Hypotheses (quoted from charter)

- **Null Hypothesis ($H_0$)**: When transferred zero-shot across distinct physical machinery testbeds ($M_{\text{source}} \to M_{\text{target}}$), dimensionless kinematic order representations suffer identical generalization degradation ($\Delta \text{Macro-F1} \ge 35\%$) as raw time-domain 1D-CNN representations due to non-linear transmission-path damping and casing impedance mismatch.

- **Alternative Hypothesis ($H_1$)**: Because dimensionless kinematic order representations project localized impact intervals onto shaft-synchronous angular coordinates ($\Delta \theta = 2\pi / K$) and decouple fault impact recurrence rates from machine structural resonance poles $h_{\text{casing}}(t)$, they maintain zero-shot cross-machine diagnostic accuracy with $\Delta \text{Macro-F1} \le 15\%$, whereas raw convolutional representations collapse by $\ge 40\%$.

## 3. Protocol Invariant Boundaries (charter-pinned, not negotiable)

| Invariant | Value | Source |
|---|---|---|
| Benchmark triad | CWRU (Drive End) ↔ Paderborn University (PU) ↔ Southeast University (SEU) | Charter §2 |
| Directional transfer graph | 6 ordered pairs $(T_{12}, T_{21}, T_{13}, T_{31}, T_{23}, T_{32})$ | Charter §2 |
| Preprocessing standard | Bandpass → Hilbert Envelope → IDFT 1X Speed Tracking → Angular Resampling | Charter §2 |
| Leakage prevention | Strict physical bearing-UID + run-level separation, zero window-level overlap | Charter §2 |
| Embedded resource target | $\le 8.2$ kB working RAM, $\le 10.5$ kB Flash code, $\le 71$ kOps/inference | Charter §2 |
| Primary statistical unit | `physical_bearing_uid` (cluster bootstrap, $n=10{,}000$) | Charter §2 + `config/experiment_config.json` |

## 4. Scope of this commit

**This commit freezes the experimental configuration only.** It does not include:

- `src/` — preprocessing, training, evaluation code. **[TBD]**
- `results/` — raw metrics, per-seed logs, transfer-matrix outputs. **[TBD]**
- `NOTES.md` — failures, deviations, lessons. **[TBD]**
- A `baseline` claim. Per the repo's own rule
  ([`04_experiments/README.md`](../README.md): *"No experiment without a pre-defined baseline."*),
  the baseline is not asserted here. It will be declared together with the
  `src/` and `results/` artifacts in a follow-up commit, after a candidate
  baseline (expected to be drawn from the P16 / Vieira 2026 / P39 evidence
  chain) has been formally proposed in `candidate-questions.md` or the
  `decision-log.md` and reviewed.

## 5. Configuration

The full protocol is encoded in [`config/experiment_config.json`](config/experiment_config.json). Summary of the frozen design:

- **Sampling**: 12 kHz target (polyphase anti-alias, 64-tap), 2,048-sample Hann windows with 1,024-sample hop.
- **Preprocessing**: linear detrend → 4th-order Butterworth bandpass 500–5000 Hz → Hilbert envelope → IDFT 1X speed estimation (±15 %) → 64 samples/revolution angular resampling → spectral FFT.
- **Representations** (5): `R1_Raw` and `R1_Env` (time-domain 1D-CNN), `R2_Raw` and `R2_Env` (frequency-domain MLP), `R3_Order` (order-spectrum MLP, 512→1024 zero-padded for shape parity).
- **Training**: AdamW, lr $10^{-3}$, wd $10^{-4}$, batch 32, max 60 epochs, early-stopping patience 12, **source-train z-score only** (leakage prevention).
- **Statistical evaluation**: cluster bootstrap on `physical_bearing_uid` (10,000 iterations), 95 % CI, Hedges' $g$ effect size, linear mixed-effects regression.

## 6. Documented deviation from charter (must be justified in `src/` + `NOTES.md`)

The ratified charter (§2) specifies the preprocessing step as *"Squared Hilbert Envelope"*. The frozen configuration uses *Hilbert magnitude* (`squared: false` in the JSON). The pre-execution validation script flagged this; it is recorded here so the deviation is not silently lost.

**Why this deviation exists** (operational, not scientific — to be confirmed by the author when `src/` is drafted):

- The squared-envelope form is dominated by a strong DC component and squared noise variance, which destabilizes the IDFT 1X speed tracker under low-SNR conditions (Paderborn outer-race at low load).
- Using Hilbert magnitude preserves the non-negative half-rectified structure of the impact modulation while keeping a more linear transfer function for downstream spectral stages.
- The squared operation is moved to a monotonic post-transform in the classifier head, where it does not contaminate the angular resampling step.

**Required follow-up**: when `src/preprocess.py` is written, it MUST (a) re-implement the squared envelope in parallel and run an ablation on the CWRU subset, and (b) record the result in `NOTES.md`. If the ablation shows the squared form does not change transfer accuracy beyond noise, the charter will be formally amended via a new decision-log entry.

## 7. What is NOT in scope for this commit (negative claims from the charter)

Per the charter's Major Non-Claims, this experiment will **NOT** claim:

1. That kinematic order tracking replaces physical sensor mounting best-practices.
2. Cross-machine transfer on machines where transmission damping attenuates impact energy below the sensor noise floor (SNR $< -20$ dB).
3. On-device backpropagation or target-domain fine-tuning.
4. That 1D-CNNs are inferior for *single-machine closed-domain* diagnostics.
5. Universal transferability to non-rotating linear mechanical systems.

## 8. How to reproduce

Once `src/` is committed:

```bash
cd 04_experiments/002_cross_machine_invariance
python src/run_experiment.py \
    --config config/experiment_config.json \
    --source cwru \
    --target paderborn \
    --representations R1_Raw R1_Env R2_Raw R2_Env R3_Order \
    --seeds 42 1337 2026 7 99
```

Expected wall-clock (CPU-deterministic, 5 seeds × 5 representations × 6 transfer pairs) is on the order of $10^2$–$10^3$ CPU-hours; a precise estimate is deferred to `NOTES.md` after the first full run.

## 9. Provenance

- Charter source: `03_research_questions/candidate-questions.md` §2.
- Decision log: `00_lab/decision-log.md` rows dated 2026-10-07 (Gate Decision C and Protocol Freeze).
- Paper evidence: `01_literature/paper-matrix.md` rows P15, P16, P20, P27, P38, P39, plus Vieira et al. (2026).
- Pre-execution validation log: see `validation_log.txt` (next to this README) for the run that approved this commit.
