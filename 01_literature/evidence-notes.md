# Evidence Notes — Verification & Claim Analysis (7 Papers)

> **Evidence Distinction Scheme**:
> - **[A] Explicitly stated by the paper**
> - **[B] Directly measurable / extractable from paper text/tables**
> - **[C] Our interpretation / synthesis**
> - **[D] Unknown / not demonstrated in paper**

---

## Paper 1: Kadonzvo (2025)
*Anomaly Detection in Rotating Electrical Machinery* (Ashesi University Capstone)

- **Finding 1.1**: The project collected multi-modal data (vibration via MPU6050, thermal via AMG8833, current via SCT-013, voltage via ZMPT101B) from a 3-phase induction motor under artificial phase imbalance.
  - **Type**: [A] Explicitly stated by paper (Chapter 3 & 4).
- **Finding 1.2**: Artificial Neural Network (ANN / MLP) achieved ~94% classification accuracy when deployed on a Raspberry Pi 5 via MQTT stream.
  - **Type**: [A] Explicitly stated by paper (Chapter 5, Section 5.3).
- **Finding 1.3**: Random Forest achieved 99% accuracy but was intentionally discarded by the author due to suspicion of overfitting.
  - **Type**: [A] Explicitly stated by paper (Section 5.3.3).
- **Finding 1.4**: Embedded deployment was executed on a Raspberry Pi 5 SBC, NOT on the ESP32 microcontroller.
  - **Type**: [B] Directly extractable (ESP32 handled sensor data acquisition/transmission only; Pi 5 ran Python models).
- **Finding 1.5**: Physical destructive bearing or gear damage was not induced on the motor.
  - **Type**: [A] Explicitly stated (the teaching motor could not be damaged; only electrical phase imbalance was induced).
- **Finding 1.6**: Sampling rate in Hz, memory consumption in kB, and power draw in mW on ESP32 or RPi 5.
  - **Type**: [D] Unknown / Not demonstrated in paper (`NR`).

---

## Paper 2: Hakam et al. (2025)
*Edge AI-powered vibration monitoring system with IEPE sensors for predictive maintenance in industrial machinery* (Scientific African, Elsevier)

- **Finding 2.1**: Classical ML algorithms (Random Forest & SVM) outperformed Deep Learning models (1D CNN, ANN, RNN) in both classification accuracy and latency on the NanoPi Neo edge platform.
  - **Type**: [A] Explicitly stated (Table 1: RF F1=99.17%, SVM F1=98.99% vs CNN F1=93.16%).
- **Finding 2.2**: Measured end-to-end latency for SVM pipeline is 2.11 ms (0.41 ms high-pass filtering + 0.37 ms FFT feature extraction + 1.33 ms SVM inference).
  - **Type**: [B] Directly extractable from Table 3 timing breakdown.
- **Finding 2.3**: Deep learning models required 12.21 ms to 15.32 ms inference time per frame on NanoPi Neo.
  - **Type**: [B] Directly extractable from Table 1.
- **Finding 2.4**: Dataset of 30,000 recordings (~8.33 hours across 6 classes sampled at 10 kHz) made publicly available on Kaggle ("Vibration Dataset of CCS/IEPE sensor AS-062").
  - **Type**: [A] Explicitly stated (Section 2).
- **Finding 2.5**: The system represents a true microcontroller deployment.
  - **Type**: [C] Our interpretation: FALSE. The model inference was executed on a NanoPi Neo (quad-core ARM Cortex-A7 SBC running Linux), not a resource-constrained microcontroller (MCU).
- **Finding 2.6**: Electrical power consumption in mW or energy in Joules per inference.
  - **Type**: [D] Unknown / Not demonstrated in paper (`NR`).

---

## Paper 3: El Boughardini et al. (2026)
*Effective Classification of Bearing Vibration Signals Using Supervised Machine Learning for Predictive Maintenance: A Lightweight 1D CNN for Embedded Deployment* (JESA, IIETA)

- **Finding 3.1**: Proposed lightweight 1D CNN (<45k params) achieved a macro F1-score of 98.6% ± 0.3% across 7 public bearing datasets harmonized to 3 health states (Healthy, Inner Race, Outer Race).
  - **Type**: [A] Explicitly stated (Abstract & Section 4).
- **Finding 3.2**: When deployed on a Teensy 4.1 microcontroller (ARM Cortex-M7 @ 600 MHz), inference executes in 4.7 ms, requiring 90 kB Flash storage and 42 kB runtime RAM.
  - **Type**: [B] Directly extractable from Abstract & Section 3.8 / Table metrics.
- **Finding 3.3**: Latency was measured empirically using hardware ARM DWT cycle counters over 10,000 iterations with compiler barriers (`__asm volatile("" ::: "memory")`).
  - **Type**: [A] Explicitly stated (Section 3.7 & Algorithm A4).
- **Finding 3.4**: Group-stratified 5-fold cross-validation by physical bearing ID prevented data leakage across train/val/test splits.
  - **Type**: [A] Explicitly stated (Algorithm 1).
- **Finding 3.5**: High accuracy transfers seamlessly to low-end 8-bit or 16-bit microcontrollers (Cortex-M0/M4).
  - **Type**: [D] Unknown / Not demonstrated (tested exclusively on a high-performance 600 MHz Cortex-M7 with 8MB Flash).
- **Finding 3.6**: Electrical power consumption during inference on Teensy 4.1.
  - **Type**: [D] Unknown / Not demonstrated (`NR`).

---

## Paper 4: Wissing (2026)
*Efficient machine learning for sensor systems with an application to computational spectrometers* (PhD Dissertation, Univ. Bamberg)

- **Finding 4.1**: Hierarchical Machine Learning (HiML) with early-exit decision branches reduced energy consumption by up to 47.63% compared to flat classifiers.
  - **Type**: [A] Explicitly stated (Abstract & Chapter 3).
- **Finding 4.2**: Reinforcement learning search accelerated the discovery of optimal classifier hierarchies compared to exhaustive search.
  - **Type**: [A] Explicitly stated (Abstract & Section 3.4).
- **Finding 4.3**: Layer-wise relevance propagation (LRP) achieved a 94% reduction in input dimensions for computational spectrometers while maintaining spectral reconstruction fidelity.
  - **Type**: [A] Explicitly stated (Abstract & Chapter 4).
- **Finding 4.4**: Physics-informed data augmentation expanded optical transmission dataset from 214 physical measurements to >10,000 augmented samples.
  - **Type**: [B] Directly extractable from Abstract & Chapter 4.
- **Finding 4.5**: HiML framework validation on vibration-based bearing predictive maintenance datasets.
  - **Type**: [D] Unknown / Not demonstrated (thesis evaluated optical spectrometry and general benchmark datasets; no bearing vibration testing).

---

## Paper 5: Gupta & Shivhare (2025)
*Embedded TinyML for Predictive Maintenance: Vibration Analysis on ESP32 with Real-Time Fault Detection in Industrial Equipment* (IJCMA)

- **Finding 5.1**: 1D CNN achieved 91.4% accuracy, 13 ms latency, 172 KB Flash, 52 KB RAM, and 93 mW power draw on ESP32.
  - **Type**: [B] Directly extractable from Table 4 & Table 5.
- **Finding 5.2**: Hybrid CNN-LSTM achieved 93.6% accuracy, 26 ms latency, 268 KB Flash, 84 KB RAM, and 115 mW power draw on ESP32.
  - **Type**: [B] Directly extractable from Table 4 & Table 5.
- **Finding 5.3**: 1D CNN consumed ~20% less power than CNN-LSTM (93 mW vs 115 mW) on ESP32.
  - **Type**: [A] Explicitly stated (Section 5.5).
- **Finding 5.4**: The custom dataset size and sample acquisition frequency in Hz.
  - **Type**: [D] Unknown / Not demonstrated in paper text (`NR`).
- **Finding 5.5**: Train/test split prevented data leakage across time windows.
  - **Type**: [D] Unknown / Not demonstrated (split protocol omitted).

---

## Paper 6: Cotrino Herrera et al. (2026)
*Intelligent Bearing Fault Detection: Deep Learning Model Assessment and Embedded System Deployment* (IEEE Latin America Transactions)

- **Finding 6.1**: Offline 1D CNN model achieved 99.95% validation accuracy across 5 bearing health classes.
  - **Type**: [A] Explicitly stated (Abstract & Table V).
- **Finding 6.2**: Deployed 1D CNN model accuracy dropped by nearly 6 percentage points (99.95% -> 94.2%) during real-time hardware validation on ESP32.
  - **Type**: [A] Explicitly stated (Abstract, Section IV, Table VI).
- **Finding 6.3**: Real-time ESP32 implementation metrics: Latency = 1 ms, Compiled firmware size = 398 kB, Estimated power draw = 260–400 mW (80–120 mA @ 3.3V).
  - **Type**: [B] Directly extractable from Table VII.
- **Finding 6.4**: Friction and thermal wear on the 3D-printed PLA bearing support caused physical structural deformation during testing, creating operational signal distortion.
  - **Type**: [A] Explicitly stated (Section IV, Point 2).
- **Finding 6.5**: Total dataset size consists of only 150 samples (30 samples per class; 64x3 window matrix).
  - **Type**: [B] Directly extractable from Section IV (Page 643).
- **Finding 6.6**: High offline accuracy (99.95%) on 150 samples guarantees robustness to industrial noise and domain shift.
  - **Type**: [C] Our interpretation: UNLIKELY. A 150-sample dataset is severely undersized and prone to overfitting.

---

## Paper 7: Alharthi et al. (2026)
*TinyML in Industrial IoT: A Systematic Review of Applications, System Components, and Methodologies* (MDPI Sensors / Systematic Review)

- **Finding 7.1**: Synthesized 35 peer-reviewed studies (2018–2026) on TinyML in IIoT, establishing that Predictive Maintenance (PdM) & Anomaly Detection account for the majority of TinyML IIoT applications.
  - **Type**: [A] Explicitly stated (Abstract & Section 3).
- **Finding 7.2**: Fewer than 30% of published TinyML IIoT studies report empirical energy or power consumption metrics (mW or Joules), relying instead solely on accuracy and latency.
  - **Type**: [A] Explicitly stated (Section 5 & 6).
- **Finding 7.3**: On-device continuous learning or online model updating is present in <5% of surveyed TinyML IIoT literature.
  - **Type**: [B] Directly extractable from Section 4 methodology synthesis.
- **Finding 7.4**: Over-reliance on stationary lab datasets (e.g. CWRU) and random window splitting (causing severe data leakage) represents a widespread methodological weakness across the domain.
  - **Type**: [A] Explicitly stated (Section 6 & 7).
- **Finding 7.5**: Microcontroller hardware is dominated by STM32 (Cortex-M4/M7) and ESP32, with TFLite Micro being the primary deployment framework.
  - **Type**: [B] Directly extractable from Section 4 system component mapping.
