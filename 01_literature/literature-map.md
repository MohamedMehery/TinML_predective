# Literature Map & Thematic Taxonomy

> **Baseline**: 39 Physical Files / 36 Unique Works at Commit `7f7b110` (October 7, 2026)

---

## 1. Domain Topology & Lineage Map

```
+========================================================================================================+
|                                    TOTAL PHYSICAL CORPUS (N = 39)                                      |
|                                     UNIQUE SCIENTIFIC WORKS (N = 36)                                   |
+========================================================================================================+
                                                    │
         ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
         ▼                                                                                     ▼
[I. CORE EMBEDDED PdM & TinyML]                                            [II. METHODOLOGICAL & ADJACENT]
(21 / 39 = 53.8% files | 20 / 36 = 55.6% works)                            (18 / 39 = 46.2% files | 16 / 36 = 44.4% works)
         │                                                                                     │
         ├──> A. Vibration Bearing Fault PdM on MCUs                                           ├──> A. Concept Drift in Data Streams (PC/Server)
         │    - P15 (Hakam 2026): STM32F4/ESP32 1D-CNN                                         │    - P01/P02 (Shang 2024): Novelty-aware NaDD
         │    - P16 (El Boughardini 2026): STM32F446 INT8 1D-CNN                               │    - P11 (He 2025): Android malware drift GNN
         │    - P18 (Gupta 2025): ESP32 vibration FFT MLP                                      │    - P26 (Greco 2025): TKDE RepDrift latent drift
         │    - P20 (Cotrino Herrera 2026): ESP32 spectrogram CNN                              │    - P35 (Tran 2025): AI Review image drift survey
         │    - P36 (Arciniegas 2025): ESP32 motor vibration DNN                               │    [5 / 39 = 12.8% files | 4 / 36 = 11.1% works]
         │    - P38 (Kolok 2025): ESP32 low-cost vibration                                     │
         │    - P39 (Garay 2026): nRF52832 Cortex-M4F Anomaly AE                               ├──> B. Sensor Missing Data Imputation (Server)
         │    [7 / 39 = 17.9% files | 7 / 36 = 19.4% works]                                    │    - P21/P32 (Cao 2025): Wearable IMU GAIN/BiLSTM
         │                                                                                     │    - P03 (Alshuhail 2025): Health IoT sensor fusion
         ├──> B. Industrial Equipment PdM on Edge                                              │    [3 / 39 = 7.7% files | 2 / 36 = 5.6% works]
         │    - P04 (Katib 2025): ESP32 consumer IoT autoencoder                               │
         │    - P05 (Bhoi 2024): STM32H7 power electronics PdM                                 ├──> C. Fault-Tolerant Control & WSN Reliability
         │    - P14 (Kadonzvo 2025): ESP32 motor anomaly detection                             │    - P10 (Munir 2015): ACM TOSN WSN Markov models
         │    - P24 (Ullah 2026): Jetson Nano milling TCN                                      │    - P12 (Wang 2007): Nonlinear active FTC Lyapunov
         │    - P27 (Łuczak 2024): SEU gearbox CWT + 2D-CNN                                    │    - P37 (Chu 2023): BLDC Hall sensor virtual CNN
         │    - P28 (Aung 2024): Cortex-M4 electric fan ML                                     │    [3 / 39 = 7.7% files | 3 / 36 = 8.3% works]
         │    - P29 (Brito 2025): ESP32-S3 industrial pump PdM                                 │
         │    - P34 (Rayhan 2025): STM32 solar inverter PdM                                    └──> D. Out-of-Scope Vision & Biomedical
         │    [8 / 39 = 20.5% files | 8 / 36 = 22.2% works]                                         - P06 (Avuçlu 2025): HOG/LBP/FFT image SVM
         │                                                                                          - P09 (Kheirollahi 2026): EEG seizure review
         └──> C. TinyML System & Hardware Surveys                                                   - P17 (Wissing 2025): Optical spectrometer ML
              - P07 (Ooko 2024): JCTA TinyML in PdM review                                         - P19 (Csurka 2018): CV local features review
              - P08 (Benmachiche 2025): Embedded anomaly survey                                     - P22 (Li 2026): Medical MRI segmentation
              - P13 (Ahmed 2026): nRF52840 TinyML benchmark                                         - P23 (Schönberger 2017): CVPR 2017 SfM features
              - P25 (Fathalla 2026): IIoT TinyML systematic review                                  - P30 (Côté-Allard 2020): sEMG gesture review
              - P31/P33 (Pazmiño Ortiz 2025): System TinyML review                                 [7 / 39 = 17.9% files | 7 / 36 = 19.4% works]
              [6 / 39 = 15.4% files | 5 / 36 = 13.9% works]
```

---

## 2. Cross-Paper Methodological Lineage

1. **The Data Leakage & Generalization Crisis**:
   - **Vieira et al. (MSSP 2026 / arXiv:2509.22267v3)** proved that segment-wise random splitting creates severe data leakage, artificially inflating 1D-CNN (WDCNN) accuracy from $63.17\%$ to $>99\%$.
   - **P39 (Garay et al., Sensors 2026)** confirmed that physical sensor remounting degrades single-machine anomaly detection by $>25\%$, requiring baseline recalibration.
   - **P16 (El Boughardini et al., 2026)** and **P20 (Cotrino Herrera et al., 2026)** demonstrated high closed-domain accuracy on CWRU, but did not evaluate cross-testbed zero-shot generalization.

2. **The Invariant Representation Solution**:
   - The laboratory's ratified scientific direction (Candidate B: C5/C8) builds upon the causal realization that 1D-CNN spatial filters overfit to casing resonance poles ($h_{\text{casing}}(t)$).
   - Projecting vibration energy onto **dimensionless kinematic orders** ($BPFO/f_r, BPFI/f_r$) via envelope demodulation and angular resampling provides the missing physical bridge enabling true zero-shot cross-machine transferability without on-device retraining.

---

## 3. Glossary of Core Scientific Terms

| Term | Formal Scientific Definition | First Corpus Occurrence |
|---|---|---|
| **TinyML** | Machine learning inference and processing executed on resource-constrained embedded systems ($\le 256\text{ kB}$ SRAM, $\le 1\text{ MB}$ Flash, $\le 100\text{ mW}$). | P07, P08, P25, P31 |
| **Order Tracking** | Signal processing technique transforming time-domain signals into the angular domain ($\theta = \int 2\pi f_r dt$), mapping rotational harmonic orders independently of speed. | P17, Ratified Protocol |
| **Casing Transfer Function ($h_{\text{casing}}(t)$)** | The mechanical structural impulse response of the machine housing, characterized by natural resonance poles and modal damping. | Ratified Protocol |
| **Zero-Shot Domain Generalization** | Model training conducted strictly on source machine $M_S$ with evaluation on unseen target machine $M_T$ without target labels, calibration, or retraining. | Ratified Protocol |
| **Ball Pass Frequency Outer Race ($BPFO$)** | Characteristic kinematic impact frequency generated by rolling elements striking an outer race defect: $BPFO = \frac{n}{2} f_r \left(1 - \frac{d_b}{D_p}\cos\theta\right)$. | P16, Ratified Protocol |
| **Ball Pass Frequency Inner Race ($BPFI$)** | Characteristic kinematic impact frequency generated by rolling elements striking an inner race defect: $BPFI = \frac{n}{2} f_r \left(1 + \frac{d_b}{D_p}\cos\theta\right)$. | P16, Ratified Protocol |
