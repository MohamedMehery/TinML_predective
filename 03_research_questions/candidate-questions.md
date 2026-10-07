# Candidate Research Questions & Ratified Protocol Charter

---

## 1. Candidate Evaluation & Discrimination Summary

| Candidate ID | Scientific Object | Core Hypothesis | Evidence Grounding | Feasibility & Dataset Triad | Gate Decision |
|---|---|---|---|---|---|
| **C1** | State-Memory Information Scaling | Moment accumulation beats spatial convolutions below critical memory threshold $B^*$. | P16, P39, Vieira et al. | Valid on CWRU / PU | Reformulated into C5/C8 |
| **C2 / C7** | Streaming Baseline Adaptation in $\mathcal{O}(1)$ Memory | Timescale separation allows zero-buffer threshold adaptation without defect masking. | P08 (Open Gap), P39 (Remounting) | High on Paderborn 2024 / Ottawa; Invalid on CWRU concatenation | **Secondary / Deferred** |
| **C3** | Multimodal Low-Rate Sensing | Multimodal low-rate ensemble saves acquisition energy over high-rate vibration. | P05, P28, P29, P39 | Requires specialized multi-sensor testbeds | Weakened (Prior-Art Overlap with P39) |
| **C4** | Quantization Representation Distortion | Uniform INT8 degrades heavy-tailed physical moments more than bounded neural activations. | P16, P17, P31, P39 | Universal across all sets | Weakened (Deep learning theory mature) |
| **C5 / C8** | Kinematic Order Invariance vs. Structural Resonance Overfitting | Dimensionless kinematic order tracking provides zero-shot cross-machine transferability without retraining. | Vieira et al. (2026), P16, P39 | **Complete across CWRU ↔ PU ↔ SEU Triad** | **RATIFIED PRIMARY DIRECTION** |
| **C6** | Compressive / Sub-Nyquist Sensing | Compressed-domain direct inference preserves diagnostic manifolds in vibration. | P17 (Optical only; 0/39 vibration) | Blocked by commercial sensor hardware (no analog CS) | **Killed** (Hardware Infeasible) |

---

## 2. Ratified Primary Research Question Charter (C5/C8)

### Primary Scientific Question
> **“To what extent do dimensionless kinematic order-tracking representations provide zero-shot cross-machine diagnostic transferability by decoupling localized bearing fault excitations from machine-specific casing structural resonances, compared to data-driven spatial convolutional representations under constrained embedded inference?”**

### Central Hypotheses
- **Null Hypothesis ($H_0$)**: When transferred zero-shot across distinct physical machinery testbeds ($M_{\text{source}} \to M_{\text{target}}$), dimensionless kinematic order representations suffer identical generalization degradation ($\Delta \text{Macro-F1} \ge 35\%$) as raw time-domain 1D-CNN representations due to non-linear transmission-path damping and casing impedance mismatch.
- **Alternative Hypothesis ($H_1$)**: Because dimensionless kinematic order representations project localized impact intervals onto shaft-synchronous angular coordinates ($\Delta \theta = 2\pi / K$) and decouple fault impact recurrence rates from machine structural resonance poles $h_{\text{casing}}(t)$, they maintain zero-shot cross-machine diagnostic accuracy with $\Delta \text{Macro-F1} \le 15\%$, whereas raw convolutional representations collapse by $\ge 40\%$.

### Major Non-Claims
1. The study will **NOT** claim that kinematic order tracking replaces physical sensor mounting best-practices.
2. The study will **NOT** claim cross-machine transfer on machines where transmission damping attenuates impact energy completely below the sensor noise floor ($\text{SNR} < -20\text{ dB}$).
3. The study will **NOT** claim on-device backpropagation or target-domain fine-tuning.
4. The study will **NOT** claim that 1D-CNNs are inferior for *single-machine closed-domain* diagnostics.
5. The study will **NOT** claim universal transferability to non-rotating linear mechanical systems.

### Protocol Invariant Boundaries
- **Benchmark Triad**: CWRU (Drive End), Paderborn University (PU), Southeast University (SEU).
- **Directional Transfer Graph**: 6 pairs ($T_{12}, T_{21}, T_{13}, T_{31}, T_{23}, T_{32}$).
- **Preprocessing Standard**: Bandpass filter + Squared Hilbert Envelope + IDFT 1X Speed Tracking + Angular Resampling.
- **Leakage Prevention**: Strictly physical bearing-unit and run-level separation (zero window-level overlap).
- **Embedded Resource Target**: $\le 8.2\text{ kB}$ working RAM, $\le 10.5\text{ kB}$ Flash code, $\le 71\text{ kOps/inference}$.
