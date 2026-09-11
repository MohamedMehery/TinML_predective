# Literature Map — Synthesis of Collected Research

> **Purpose**: Map the state of evidence across the 7 collected literature papers.  
> **Rule**: Document recurring patterns, isolated approaches, and contradictions. Do NOT declare final research gaps yet.

---

## 1. Recurring Research Problems
Across the collected literature set, research focus concentrates heavily on:
1. **Bearing & Motor Fault Diagnosis**: 5 out of 7 papers (Kadonzvo, Hakam, El Boughardini, Gupta, Cotrino Herrera) specifically address fault diagnosis in rotating electrical machines or rolling element bearings.
2. **On-Device Edge Inference**: 6 papers focus on transferring inference from centralized/cloud servers to local edge hardware (ESP32, Teensy 4.1, NanoPi Neo, Raspberry Pi 5, optical sensor nodes).
3. **Model Efficiency & Resource Constraints**: All 7 papers address the trade-off between model accuracy, memory footprint (RAM/Flash), latency, and energy consumption.

---

## 2. Datasets Landscape
- **Dominant Public Benchmarks**: Case Western Reserve University (CWRU) is referenced or used in 4 papers (Kadonzvo, El Boughardini, Cotrino Herrera, Alharthi). Other public datasets include MFPT, IMS (NASA), Paderborn, PRONOSTIA, and Huang-Baddour.
- **Custom / Ad-Hoc Laboratory Rigs**: 4 papers (Kadonzvo, Hakam, Gupta, Cotrino Herrera) created custom laboratory test benches to record vibration data under controlled conditions.
- **Dataset Openness**: Only 2 primary experimental papers explicitly link publicly accessible dataset repositories (Hakam via Kaggle, Cotrino Herrera via journal link).

---

## 3. Sensor Modalities & Signal Types
- **Triaxial & Monaxial Accelerometers**: Accelerometer-based vibration signals are the single most dominant sensing modality (used in 6 out of 7 papers). Specific sensors cited include MPU6050 (Kadonzvo, Cotrino Herrera), IEPE AS-062 piezoelectric (Hakam), and ADXL345 (Gupta).
- **Multi-Modal Sensing**: Multi-sensor fusion (combining vibration, thermal imaging via AMG8833, current via SCT-013, and voltage via ZMPT101B) appears in 1 paper (Kadonzvo).
- **Optical Sensors**: Optical transmission spectral sensing appears in 1 paper (Wissing).

---

## 4. Machine Learning Model Families
- **1D Convolutional Neural Networks (1D CNNs)**: Used in 5 papers (Hakam, El Boughardini, Gupta, Cotrino Herrera, Alharthi). 1D CNNs are overwhelmingly preferred for raw time-series vibration classification due to low parameter count (< 45k params) and spatial feature extraction.
- **Classical Machine Learning (SVM, Random Forest, k-NN)**: Evaluated in 4 papers (Kadonzvo, Hakam, El Boughardini, Alharthi). In Hakam et al., Random Forest (99.17% F1, 1.56 ms) and SVM (98.99% F1, 1.33 ms) significantly outperformed deep learning models on a Linux SBC target.
- **Recurrent & Attention Architectures (LSTM, Transformers)**: Evaluated in 3 papers (Gupta, Cotrino Herrera, Alharthi). LSTM models showed slight accuracy gains (e.g., Gupta 93.6% vs 91.4%) but at the cost of double latency (26 ms vs 13 ms) and high memory footprint. Transformers underperformed (< 93.7% accuracy) on small sample regimes.
- **Hierarchical & Early-Exit Networks**: Evaluated in 1 paper (Wissing) for energy optimization via early exits.

---

## 5. Preprocessing & Feature Engineering
- **Dual Processing Paths**:
  - *Path A (DL Inputs)*: Minimal preprocessing (band-pass filtering 500 Hz–5 kHz, order-RPM normalization, sliding window segmentation 64 to 1024 samples).
  - *Path B (Classical ML Inputs)*: Handcrafted feature extraction across time (mean, std, RMS, kurtosis, crest factor), frequency (FFT, spectral centroid, band energy), and time-frequency (STFT, MFCCs) domains (up to 47 features in El Boughardini).

---

## 6. Target Hardware & Embedded Platforms
- **Microcontrollers (MCUs)**:
  - **ESP32** (Xtensa LX6 @ 240 MHz): Used as primary inference target in 2 papers (Gupta, Cotrino Herrera) and as acquisition node in 2 papers (Kadonzvo, Hakam).
  - **Teensy 4.1** (ARM Cortex-M7 @ 600 MHz): Used as high-performance MCU target in 1 paper (El Boughardini).
- **Single-Board Computers (SBCs)**:
  - **Raspberry Pi 5 / NanoPi Neo**: Used in 2 papers (Kadonzvo, Hakam) for edge processing.
- **Deployment Frameworks**: TensorFlow Lite for Microcontrollers (TFLM) is the dominant MCU deployment framework cited across 4 papers (El Boughardini, Gupta, Cotrino Herrera, Alharthi).

---

## 7. Evaluation Metrics & Methodological Patterns
- **Accuracy & F1-Score**: Reported in 100% of experimental papers (typically ranging from 91.4% to 99.95%).
- **Inference Latency**: Reported in 5 papers (Hakam, El Boughardini, Gupta, Cotrino Herrera, Alharthi). Latency ranges from 1 ms to 26 ms on MCU targets.
- **Memory Footprint (RAM / Flash)**: Detailed footprint metrics reported in 3 papers (El Boughardini: 90kB Flash / 42kB RAM; Gupta: 172kB Flash / 52kB RAM; Cotrino Herrera: 398kB firmware image).
- **Energy & Power Measurements**: Severely under-reported. Measured empirically in only 2 papers (Gupta: 93 mW; Cotrino Herrera: 260-400 mW). Alharthi et al. explicitly note that <30% of TinyML IIoT literature measures energy or power draw.

---

## 8. Recurring Limitations Stated by Authors
1. **Data Leakage & Random Window Splitting**: El Boughardini and Alharthi highlight that random sample windowing without grouping by physical bearing ID causes massive artificial inflation of accuracy.
2. **Embedded Performance Degradation**: Cotrino Herrera documented a ~6% drop in accuracy (99.95% offline -> 94.2% on ESP32) due to fixed-point conversion constraints and mechanical thermal expansion of test rig components.
3. **Small Sample Sizes & Lack of Real Defects**: Multiple custom lab setups rely on tiny datasets (e.g., Cotrino Herrera: 150 samples total) or synthetic phase imbalance rather than natural physical bearing wear (Kadonzvo).
4. **Lack of Cross-Machine / Cross-Condition Generalization**: Models trained on single test rigs perform poorly under variable speeds, loads, or distinct physical machines.

---

## 9. Flow of Ideas Across Collected Literature
```
[Public Rigs: CWRU / IMS / Paderborn] ──> Standard Benchmark Reference (El Boughardini, Alharthi)
                                                │
[Custom Lab Testbenches] ───────────────────────┼──> Edge Hardware Deployment (ESP32 / Teensy / NanoPi)
  (Hakam, Gupta, Cotrino Herrera, Kadonzvo)     │     (TFLite Micro, INT8 Quantization)
                                                │
[Systematic Synthesis (Alharthi, 2026)] ───────┘──> Identifies Gaps: Data Leakage, Missing Energy Metrics,
                                                       Lack of Cross-Domain Generalization
```
