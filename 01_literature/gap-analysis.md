# Gap Analysis — Cross-Paper Synthesis & Evidence Matrix

> **Corpus Baseline**: 39 Physical Files / 36 Unique Works at Commit `7f7b110`  
> **Rule**: No gap is identified without at least two explicit, empty cross-paper cells supported by documented evidence.

---

## 1. Cross-Paper Coverage Matrix (Key In-Scope Works)

| Evaluated Dimension | P04 (Katib) | P05 (Bhoi) | P08 (Benmachiche) | P15 (Hakam) | P16 (El Boughardini) | P18 (Gupta) | P20 (Cotrino) | P25 (Fathalla) | P28 (Aung) | P29 (Brito) | P38 (Kolok) | P39 (Garay) | Vieira (2026) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Physical MCU Deployment** | ✓ (ESP32) | ✓ (STM32H7) | ? (Survey) | ✓ (STM32F4) | ✓ (STM32F4) | ✓ (ESP32) | ✓ (ESP32) | ? (Survey) | ✓ (Cortex-M4) | ✓ (ESP32-S3) | ✓ (ESP32) | ✓ (nRF52832) | — |
| **INT8 / Fixed-Point Quantization** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | — |
| **Leakage-Free Cross-Bearing Split** | — | — | — | — | — | — | — | — | — | — | — | — | ✓ |
| **Cross-Machine Transfer Validation** | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Online Streaming Drift Adaptation** | — | — | ? (Gap ID) | — | — | — | — | — | — | — | — | ✓ (Remount) | — |
| **Zero-Shot Kinematic Invariance** | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Stack / Runtime SRAM Measurement** | — | — | — | — | ✓ | — | — | — | ✓ | — | — | ✓ | — |
| **Energy Consumption Measurement** | — | — | — | — | — | — | — | — | — | — | ✓ | — | — |

*Legend: `✓` = Directly demonstrated; `?` = Mentioned or surveyed; `—` = Unaddressed / Empty.*

---

## 2. Identified Primary Scientific Gaps

### Gap G1: Zero-Shot Cross-Machine Structural Transfer Collapse (The C5/C8 Gap)
- **Evidence**: 0 / 39 papers in the corpus evaluate cross-machine transfer between distinct physical machinery testbeds. Vieira et al. (MSSP 2026) proved that deep learning models collapse to $63.17\%$ even across different bearing units on the *same* testbed.
- **Scientific Mechanism**: Purely data-driven 1D-CNNs overfit to the casing structural transfer function $h_{\text{casing}}(t)$ (natural resonance poles) of the training rig.
- **Public Data Verifiability**: Fully verifiable across the public CWRU $\leftrightarrow$ Paderborn $\leftrightarrow$ SEU benchmark triad.
- **Priority**: **PRIMARY RATIFIED GAP (C5/C8)**.

### Gap G2: $\mathcal{O}(1)$ Working-Memory Streaming Baseline Adaptation (The C2/C7 Gap)
- **Evidence**: P08 (Benmachiche et al. 2025) explicitly surveyed embedded anomaly detection and identified *on-device concept drift adaptation and real-time threshold adjustment under strict embedded constraints ($<64\text{ kB}$ RAM) as an unaddressed open gap*. P39 (Garay et al. 2026) proved that sensor remounting creates severe baseline drift.
- **Scientific Mechanism**: Distinguishing operational load step transients from incipient cyclical defect emergence in constant-state memory.
- **Public Data Verifiability**: Conditionally verifiable (requires authentic dynamic time-varying datasets such as Paderborn 2024 Zenodo or Ottawa).
- **Priority**: **SECONDARY / DEFERRED GAP (C2/C7)**.

### Gap G3: Quantization-Induced Geometric Distortion in Heavy-Tailed Moment Manifolds
- **Evidence**: P16, P17, and P39 deployed INT8 models, but 0 / 39 papers analyzed the differential topological distortion between non-linear heavy-tailed physical moments vs. bounded neural activations under extreme quantization ($<8\text{ bits}$).
- **Priority**: Methodological sub-component of G1/G2.
