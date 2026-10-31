# Phase 30G rev-7 — DSP, KA01 Evidence Gate, and Resource-Target Decision Record

**Status:** DRAFT — pending owner ratification. No code wired into the live pipeline.
**Date:** 2026-10-16
**Branch:** `rev7-dsp-ka01-evidence` (off `arena/6ad75829-tinml-predective` @ `128dda7b`).
**Scope:** DSP corrections (PART 1–4), KA01-only inspection (PART 5), resource-target
demotion (PART 6), documentation/reproducibility (PART 7), and a focused test
(PART 3 last item, PART 8 step 1).

This record **does not** implement, train, evaluate, download, or start any
experimental step. Stage A and Stage B of the rev-5/6 authorization boundary
remain in force; the KA01 inspection is a separate authorized evidence-gathering
step. The previously ratified primary scientific comparison
(`R2_Env` vs `R3_Order`) and secondary end-to-end pipeline comparison
(`R1_Raw` vs `R3_Order`) are preserved.

---

## PART 1 — Bearing geometry and corrected order references

### Source of geometry

The bearing is a **6203** deep-groove ball bearing, per Lessmeier 2016 §3
("we limited research to ball bearings of type 6203"). The prompt's stated
geometry (d=6.75 mm ball diameter, D=28.55 mm pitch-circle diameter, Z=8
rolling elements, α=0°) is consistent with the published 6203 catalog value
range (SKF 6203: d ≈ 6.75 mm, D ≈ 28.575 mm, Z=8, contact angle ≈ 0°); the
local repository does not contain a per-bearing 6203 datasheet, so the
prompt's geometry is used as-is and recorded here as the
**rev-7 working geometry** pending owner confirmation.

### Formulas (per the prompt)

```
phi      = (d / D) * cos(alpha)
BPFO     = (Z / 2) * (1 - phi)         [orders / revolution]
BPFI     = (Z / 2) * (1 + phi)         [orders / revolution]
BSF      = (D / (2*d)) * (1 - (d/D * cos(alpha))^2)   [orders / revolution]
FTF      = 0.5 * (1 - phi)             [orders / revolution]
```

### Substituted values

```
d = 6.75 mm
D = 28.55 mm
Z = 8
alpha = 0 deg  =>  cos(alpha) = 1
phi = (6.75 / 28.55) * 1 = 0.23643
```

### Resulting characteristic orders (corrected)

| Quantity | Value (orders/rev) | At 25 Hz (1500 rpm) | At 15 Hz (900 rpm) |
|---|---|---|---|
| **BPFO** | **3.0543** | 76.36 Hz | 45.81 Hz |
| **BPFI** | **4.9457** | 123.64 Hz | 74.19 Hz |
| **BSF**  | **1.9966** | 49.92 Hz | 29.95 Hz |
| **FTF**  | **0.3818** | 9.54 Hz | 5.73 Hz |

### Correction of prior values

Prior rev-5 and earlier reported values of **2.30 / 5.70 / 0.79 / 0.41**
orders are **incorrect** for the prompt's stated 6203 geometry. Those values
were computed using a different pitch ratio (d/D = 17/40 = 0.425, which is
not consistent with the 6203). **The corrected values above are the
rev-7 reference for any BPFO/BPFI/BSF/FTF discussion going forward.**

History preservation: prior rev-5 / rev-6 reports are NOT silently edited.
The prior 2.30 / 5.70 values are documented in this record as a superseded
calculation. Any protocol amendment, decision memo, or pilot output that
references 2.30 / 5.70 should be updated in a separate traceable edit; the
rev-7 commit does NOT touch those files.

### Theoretical orders vs observed peaks

The values above are **theoretical characteristic orders** for a perfect
6203 with the stated geometry. They are **not** claimed to be detected in
any real recording until the actual signal is examined. The KA01 inspection
gate (PART 5) did not complete (see §PART 5 below); no peak-detection
claim is made in this record.

---

## PART 2 — Angular resolution and window length

### Hann-window main-lobe width

For a Hann-windowed FFT, the main-lobe null-to-null width is **4 bins**
in the spectrum's natural units (orders per revolution in the angular
spectrum). The −3 dB half-width is approximately 1 bin.

### BPFO / BPFI main-lobe overlap

BPFO = 3.054, BPFI = 4.946, separation = **1.891 orders**.

| Candidate | Δorder (orders/bin) | Main-lobe null-to-null | Main-lobe overlap? |
|---|---|---|---|
| **2-rev** (128 angular samples) | 0.5 | 2.0 orders | **YES** — BPFO main lobe ends at 4.05, BPFI starts at 3.95 |
| **4-rev** (256 angular samples) | 0.25 | 1.0 order | **NO** — BPFO main lobe ends at 3.55, BPFI starts at 4.45 (gap 0.9 order) |

**The 2-rev candidate does NOT resolve BPFO from BPFI under the corrected
geometry.** This is a spectral-resolution argument, not a proof of
classifier accuracy or fault detectability. SNR, leakage, and side-lobe
structure are not assessed here. The 4-rev candidate provides main-lobe
separation but does not guarantee BPFO and its 1-order sidebands (3.054 ± 1)
are resolved — those sidebands are at 2.05 and 4.05, both within the
BPFO/BPFI main-lobe region at 4-rev resolution.

### 4096-sample window at 12 kHz (engineering candidate)

- Window duration: 4096 / 12000 = **341.33 ms**.
- At 15 Hz: 5.120 revolutions (4 revs = 266.67 ms = 3200 samples ⇒ fits with 20.67 ms / 896 samples margin).
- At 25 Hz: 8.533 revolutions (4 revs = 160 ms = 1920 samples ⇒ fits with 181.33 ms / 2176 samples margin).

**The 4-rev angular grid (256 samples) is feasible at both 15 Hz and 25 Hz
inside a 4096-sample, 12 kHz window** with substantial margin. This is the
**engineering candidate** for the rev-7 primary window.

### Recalculated MACs (4096-sample CNN, exact layer shapes)

Conv stack applied to a 4096-sample input (R1_Raw / R1_Env):

```
Conv1: in=1, out=16, k=15, stride=2, pad=7   L0=4096 -> L1=2048
Conv2: in=16, out=32, k=7, stride=2, pad=3   L1=2048 -> L2=1024
Conv3: in=32, out=64, k=5, stride=2, pad=2   L2=1024 -> L3=512
FC:    in=64, out=3
```

| Stage | MACs |
|---|---|
| Conv1 | 1 × 16 × 15 × 2048 = 491,520 |
| Conv2 | 16 × 32 × 7 × 1024 = 3,670,016 |
| Conv3 | 32 × 64 × 5 × 512 = 5,242,880 |
| FC | 64 × 3 = 192 |
| **Total** | **9,404,608 MACs = 9.405 MOp** |

Comparison to the 2048-sample case (4.702 MOp from rev-5): **the 4096-sample
window exactly doubles the CNN MACs**, as expected from the doubled
activation-map sizes. R1_Raw / R1_Env remain **66× the 71 kOp target** at
the 4096-sample input (9.4 MOp vs 71 kOp).

R3_Order MLP MACs are **unchanged** (input is the 1024-feature padded
spectrum; 1024 × 64 + 64 × 32 + 32 × 3 = 67,680 MACs = 68 kOp, at the
71 kOp target). The CNN MAC count depends on the time-domain input length
(4096), but the MLP MAC count depends only on the post-padded feature
dimension (1024). **The 1024-feature MLP input dimension is the fixed model
input; the 4096-sample time-domain window and the natural angular-spectrum
length (256 samples = 4 revs ⇒ 129 bins) are both independent of the
1024-feature fixed input.**

### Status

The 4096-sample window is documented as an **engineering candidate pending
validation**. If the actual preprocessing pipeline cannot support it
consistently (e.g. decimation filter cannot produce a 12 kHz signal without
exceeding the Flash/RAM targets), the candidate is rejected and the
window-length question re-opened.

---

## PART 3 — Decimation filter design and verification

### Translation of `filter_order=64`

`scipy.signal.resample_poly(x, up, down, window=...)` does **not** accept a
`filter_order` argument. The `window` argument is the FIR coefficient array
(or a window-function name consumed by `firwin`). The `experiment_config.json`
field `sampling_rate_harmonization.filter_order = 64` is therefore
**mis-specified as currently written**. The correct translation for a
64-tap prototype would be a 64-element FIR; in practice, meeting 60 dB
stopband at a 1 kHz transition width requires many more taps.

### Kaiser-window FIR design (verified numerically)

Rational ratio: `up=3, down=16` (64 kHz → 12 kHz). Polyphase rate:
`fs_poly = 64e3 × 3 = 192 kHz`. The FIR operates at the polyphase rate.

Design method: `scipy.signal.kaiserord(atten_db, width_norm)` for the tap
count and Kaiser β, then `scipy.signal.firwin(numtaps, cutoff / (fs_poly/2),
window=('kaiser', beta))`.

Design targets and computed result:

| Quantity | Value |
|---|---|
| Passband | 0 – 5000 Hz |
| Stopband edge | 6000 Hz (new Nyquist at 12 kHz) |
| Transition width | 1000 Hz |
| Passband ripple target | < 0.5 dB peak-to-peak |
| Stopband attenuation target | ≥ 60 dB |
| `kaiserord(60, 1000/192000)` → numtaps | **1394** |
| Kaiser β | **5.6533** |
| Cutoff placed at | 5500 Hz (mid-transition) |
| **At 500 Hz** | **+0.0006 dB** (verified numerically) |
| **At 5000 Hz** | **+0.0046 dB** (verified numerically) |
| **At 5500 Hz** (cutoff) | **−6.54 dB** (Hann-equivalent, expected for `firwin` default) |
| **At 6000 Hz** (stopband edge) | **−65.75 dB** (verified numerically) |
| **At 7000 Hz** | **−77.81 dB** (verified numerically) |
| Passband deviation in [500, 5000] Hz | 0.0077 dB peak-to-peak (meets 0.5 dB target) |
| Stopband min attenuation in [6000, 96000] Hz | 65.75 dB (meets 60 dB target) |

**Verification source**: numerical computation in this session using
`scipy 1.17.1` and `numpy 2.4.6` installed via `pip` (the project's existing
environment did not include scipy; this was added solely to verify the
decimation filter design and the angular-domain low-pass design — it is
**not** wired into the live pipeline).

**The 64-tap value in `experiment_config.json` is incorrect.** A 60 dB
anti-alias design needs **1394 taps**. This is a 21.8× increase in FIR
length and a corresponding 21.8× increase in FIR storage (1394 × 4 = 5576
bytes = 5.4 kB at fp32, or 1394 × 2 = 2788 bytes = 2.7 kB at int16) and
in per-sample MAC count (1394 MACs/sample × 3 up = 4182 MACs per
polyphase-input sample).

### Pre-existing 500–5000 Hz bandpass

The current `preprocess.py:36-39, 163` applies a 4th-order Butterworth
bandpass 500–5000 Hz. **This bandpass does not replace a properly verified
anti-alias filter.** The 500–5000 Hz bandpass has 24 dB/oct roll-off; its
attenuation at 6000 Hz is only ~5.7 dB, far below the 60 dB target. **For
real-data 64→12 kHz decimation, the anti-alias filter must be the Kaiser
1394-tap FIR (or equivalent) designed here; the 500–5000 Hz bandpass is a
signal-conditioning step, not an anti-alias filter.**

### `scipy.signal.resample_poly` semantics

`scipy.signal.resample_poly(x, up, down, window=...)`:

- Upsamples by `up` (zero-insertion), applies the FIR `window`, then
  downsamples by `down`.
- If `window` is a 1-D array, it is used directly as the FIR coefficients.
- If `window` is a string, it is passed to `scipy.signal.get_window` and
  the FIR is designed by `firwin` with that window.
- The default (no `window` argument) uses a Hann-windowed FIR with
  `numtaps = max(len(x), up) * 3` and a cutoff at `min(up, down) / up * 0.5`
  of the polyphase Nyquist.
- **The function does not validate the FIR; if the user passes a poorly
  designed FIR, the output aliases without warning.**

The smallest appropriate location to add the decimation step in the
existing code is a new function in `preprocess.py` (e.g.
`decimate_64k_to_12k(sig)`) that wraps the Kaiser 1394-tap FIR and calls
`scipy.signal.resample_poly`. This is **NOT** implemented in this rev-7
commit; it is documented as a future change.

### Focused deterministic test (in this commit)

`04_experiments/002_cross_machine_invariance/src/test_decim_filter.py` —
runs 4 deterministic checks on the Kaiser 1394-tap FIR:

1. Output length after `resample_poly(..., up=3, down=16)` for a 64000-sample
   input is 12000 samples (effective 12 kHz).
2. Passband magnitude response at 500 Hz and 5000 Hz is within 0.01 dB of
   unity.
3. Stopband attenuation at 6000 Hz is ≥ 60 dB.
4. A 7000 Hz input tone (which would alias to 5000 Hz without proper
   anti-aliasing) is attenuated to ≤ −60 dB in the output.

The test uses synthetic int16-equivalent signals (deterministic, generated
in-memory, not from a real recording) and is **not** a training or
evaluation step.

---

## PART 4 — Angular-domain anti-alias filter

### Current implementation gap

`preprocess.py:99-138` (`angular_resample`) uses `np.interp` to resample
the envelope onto the angular grid. **`np.interp` is a linear interpolator,
not a proper anti-alias filter.** It has a first-order low-pass
characteristic (~−6 dB/octave roll-off) and is not designed to suppress
high-order envelope content that would alias when the angular resample
re-samples at 64 samples per revolution.

**The current R3_Order pipeline does not apply a proper angular-domain
anti-alias filter.** This is a pre-existing gap, not introduced by rev-7.

### Nominal cutoff rule (re-stated)

For 64 angular samples per revolution, the angular-domain Nyquist is
32 orders. The pre-existing `preprocess.py:14-18` and the rev-5 prompt
suggest `fc = 0.4 × 64 × shaft_frequency` as a candidate low-pass cutoff:

- At 15 Hz: `fc = 0.4 × 64 × 15 = 384 Hz` envelope equivalent,
  corresponding to 384 / 15 = 25.6 orders.
- At 25 Hz: `fc = 0.4 × 64 × 25 = 640 Hz` envelope equivalent,
  corresponding to 640 / 25 = 25.6 orders.
- The nominal upper usable order is **25.6 orders**; angular Nyquist is
  **32 orders**. The 0.4 factor leaves a ~20% margin between the
  nominal cutoff and the angular Nyquist.

### Filter specification (proposed, requires owner ratification)

A proper anti-alias filter for the angular resample should be:

- Applied to the **envelope** (not the raw vibration) before the angular
  resample.
- A low-pass with cutoff at 25.6 orders (or 0.4 × samples_per_rev × fr),
  with passband 0–25.6 orders and stopband 28.8–32 orders.
- A Kaiser-window FIR with stopband attenuation ≥ 60 dB.
- Speed-supplied as the **nominal speed parsed from the condition code**
  (N15 → 25 Hz, N09 → 15 Hz) per the rev-5 nominal-speed policy.
  Vibration-based speed estimation is **deferred**.

### Engineering candidate (Kaiser FIR, samples_per_rev = 64, 4-rev window = 256)

At 25 Hz nominal, with 4 revs = 256 envelope samples (envelope sampled at
64 samples/rev), the envelope-domain sampling rate is 64 × 25 = 1600 Hz.
The cutoff is 0.4 × 1600 = 640 Hz; the stopband edge is 1.2 × 640 = 768 Hz
(or up to the envelope-domain Nyquist at 800 Hz). Transition width: 128 Hz.
Using `kaiserord(60, 128/1600) = kaiserord(60, 0.08)`:

```
kaiserord(60, 0.08)  ->  numtaps = 92, beta = 5.6533
```

92 taps at fp32 = 368 bytes; at int16 = 184 bytes. This fits comfortably
in the 8.2 kB target if the FIR is applied as a streaming filter.

At 15 Hz nominal, the envelope-domain sampling rate is 64 × 15 = 960 Hz,
cutoff 384 Hz, stopband edge 480 Hz, transition 96 Hz. `kaiserord(60,
96/960) = kaiserord(60, 0.1)` → numtaps = 74, beta = 5.6533.

The angular anti-alias filter is **speed-dependent** because the envelope
sampling rate after angular resampling is `samples_per_rev × fr`. A single
fixed filter at the higher rate (1600 Hz) is sufficient for both speeds
(its passband at 25 Hz nominal extends well above 15 Hz nominal's
Nyquist); but a separate 15 Hz filter is more efficient.

### Protocol implication

**Adding the angular-domain anti-alias filter changes the scientific
definition of R3_Order** (it adds a low-pass step that was not present in
the prior pipeline). This is a **protocol amendment requiring owner
ratification** before it can become part of the primary experiment.

Status: **NOT applied**. The current pipeline's behavior is preserved
until the owner ratifies the angular anti-alias filter and the speed-source
policy together.

---

## PART 5 — KA01-only inspection (NOT COMPLETED)

### Network attempt

This turn attempted to download `KA01.rar` from the official KAt-DataCenter
source at `https://groups.uni-paderborn.de/kat/BearingDataCenter/KA01.rar`
(verified archive size 159 MB in the KAt-DataCenter index, within the
200 MiB abort limit). The TLS connection to `groups.uni-paderborn.de:443`
failed with `OpenSSL SSL_connect: SSL_ERROR_SYSCALL in connection to
groups.uni-paderborn.de:443` after the initial TCP connect. Subsequent
retries with `curl -v` showed the same failure mode (TLS handshake
hangup).

**The official source is unreachable from this environment at this time.**

### Tool availability

`unrar` and `7z` are **NOT installed** in the environment
(`which unrar 7z` returns nothing; `apt list --installed | grep -i
'unrar|p7zip'` returns nothing). Even if the download had succeeded,
extraction would have required a tool installation step, which the
rev-7 mandate forbids without explicit owner approval (rule: "Do not
switch to a different host or download additional bearings without
explicit approval"; install-tool approval is implied by the same gate).

### Exact failure

| Step | Status | Evidence |
|---|---|---|
| KAt-DataCenter index reachability | OK | `https://groups.uni-paderborn.de/kat/BearingDataCenter/` returns the directory listing (verified in rev-5) |
| KA01.rar HTTP HEAD | failed | `curl -sI -L` returns `SSL_ERROR_SYSCALL` after TCP connect |
| KA01.rar download | not attempted | aborted after the HEAD failure (no fallback host) |
| Extraction tool availability | not present | `unrar` and `7z` are not installed |
| KA01 field/channel evidence | **not obtained** | the inspection gate did not complete |

### What is NOT claimed

- This record does **not** claim that the KA01 inspection succeeded.
- This record does **not** claim any field names, channel names, sampling
  rates, or speed-channel evidence from the actual KA01 `.mat` files.
- The KA01 condition-availability table is **not** produced.

### What is documented as unverified

- Whether the `.mat` files contain a measured shaft-speed channel.
- The exact field names and shapes in the KAt-DataCenter `.mat` files.
- Whether the speed channel, if present, is at the same rate as the
  vibration channel or at a lower rate (the paper §4.2 page 5 says the
  speed is recorded "synchronously but at a lower sampling rate", but
  the rate is not in the local evidence).
- Whether a real-data PU loader that parses the condition code and
  applies the Kaiser 64→12 kHz decimation can be implemented without
  inspecting at least one `.mat` file.

### Stage A status

Stage A (download + integrity gate + speed-channel check) **cannot
proceed** in this turn. The owner must either (a) provide a working
mirror of the KAt-DataCenter, (b) install `unrar` or `7z` and re-try
when the TLS issue is resolved, or (c) ratify a different evidence
gathering strategy. **No Stage A action is taken in this turn.**

Stage B (training and evaluation) remains **separately authorized** and
**not authorized by this turn**.

---

## PART 6 — Resource-target demotion

### Current targets

From `03_research_questions/candidate-questions.md:39`:
"Embedded Resource Target: $\le 8.2\text{ kB}$ working RAM, $\le 10.5\text{ kB}$
Flash code, $\le 71\text{ kOps/inference}$."

### Status of the targets

The targets are **analytical projections, not measured acceptance gates**.
The local repository does **not** contain a compiled embedded
implementation; the targets were derived analytically during the C5/C8
charter freeze (decision-log row 10, 2026-10-07) and were associated with
a 2048-sample R3_Order input. No TFLite-Micro / CMSIS-NN / STM32 / ESP32
build artifacts exist in the local repo. The targets therefore represent
**design aspirations**, not **verified compliance**.

### Demotion recommendation (for owner ratification)

**Recommendation:** the existing resource limits (8.2 kB / 10.5 kB / 71
kOps) are demoted from hypothesis/acceptance conditions to **reported
engineering targets** for this research stage. The original targets are
preserved in the documentation as historical design goals. **No new
thresholds are introduced and no models are redesigned to meet the
original targets** in this commit.

### Per-representation resource accounting (12 kHz, 4096-sample window candidate)

For each of the five representations, the table distinguishes:

1. **Parameter count** (from `models.py:28-77`, `experiment_config.json:38-72`).
2. **Parameter storage (fp32)**: parameters × 4 bytes.
3. **Parameter storage (INT8 weights, fp32 BN/biases)**: weights × 1 byte + BN/biases × 4 bytes.
4. **Analytical peak-live RAM** (12 kHz, 4096-sample window, fp32, with buffer reuse where possible).
5. **Measured RAM**: **NONE** (no compiled target build exists).
6. **Estimated model + runtime Flash**: rough analytical estimate.
7. **Measured Flash**: **NONE**.
8. **MACs / inference**: with formulas and exact layer dimensions.

| Rep | Params | fp32 storage | INT8+overhead | Peak live RAM (analytical) | Measured RAM | Flash (analytical) | Measured Flash | MACs |
|---|---|---|---|---|---|---|---|---|
| **R1_Raw** | 14,371 | 57.5 kB | ~57 kB (BN fp32 dominates) | ~213 kB (single-conv-stage peak, fp32, 4096-input); ~66 kB with streaming reuse | NONE | ~30–40 kB (INT8 weights + TFLite-Micro Conv1D/BN/Linear kernels) | NONE | 9,404,608 (4096-input) |
| **R1_Env** | 14,371 | 57.5 kB | ~57 kB | same as R1_Raw | NONE | same as R1_Raw | NONE | 9,404,608 |
| **R2_Raw** | 67,907 | 271.6 kB | ~66 kB (INT8 weights + ~2 kB BN/head) | ~32 kB | NONE | ~80 kB (TFLite-Micro Dense + BN kernels) | NONE | 67,680 |
| **R2_Env** | 67,907 | 271.6 kB | ~66 kB | ~32 kB | NONE | ~80 kB | NONE | 67,680 |
| **R3_Order** | 67,907 | 271.6 kB | ~66 kB | ~22–32 kB (with embedded-style FFT scratch) | NONE | ~80 kB | NONE | 67,680 |

### Compliance with original targets (analytical, not measured)

| Target | R1_Raw | R1_Env | R2_Raw | R2_Env | R3_Order |
|---|---|---|---|---|---|
| RAM ≤ 8.2 kB | NO (~26×) | NO | NO (~4×) | NO | NO (~3–4×) |
| Flash ≤ 10.5 kB | NO (~3–4×) | NO | NO (~8×) | NO | NO (~8×) |
| MACs ≤ 71 kOp | NO (~132×) | NO | at target | at target | at target |

**None of the five representations meets the original 8.2 kB RAM or
10.5 kB Flash targets** at the 4096-sample window candidate. R2_Raw,
R2_Env, and R3_Order are at the 71 kOp target; R1_Raw and R1_Env are
~66× over the 71 kOp target. **No claim of embedded deployment
readiness or compliance is made.**

### What is recorded for owner decision

- Demote the targets from acceptance gates to reported engineering
  targets (recommended).
- Re-budget the targets (e.g., 64 kB RAM, 128 kB Flash) consistent with
  the 4096-sample window and the analytical estimates above (alternative).
- Preserve the original targets as historical design goals and do not
  re-budget (alternative).
- Redesign the models to meet the original targets (e.g., aggressively
  prune R1_Raw to fit; not recommended at this stage).

**No silent replacement of the original targets.**

---

## PART 7 — Documentation and reproducibility changes in this commit

### Files added in this commit

1. `04_experiments/002_cross_machine_invariance/notes/rev7_dsp_decision_record.md`
   — this document. Concise engineering decision record covering PART 1–7.
2. `04_experiments/002_cross_machine_invariance/src/preprocess_decim.py` —
   the 64→12 kHz decimation function (Kaiser 1394-tap FIR) and a
   `decimate_64k_to_12k(sig)` wrapper. **Not imported or wired into the
   live pipeline.** Standalone module for verification.
3. `04_experiments/002_cross_machine_invariance/src/test_decim_filter.py` —
   the focused deterministic test (PART 3 last item). Tests the decimation
   filter in isolation; does not invoke the live pipeline. Uses
   synthetic, deterministic in-memory signals.

### Files NOT changed in this commit

- `experiment_config.json` — `filter_order=64` is mis-specified but is
  **not** edited in this commit; the corrected value (1394 taps) is
  documented here but the config edit is deferred to a follow-up
  owner-ratified protocol amendment.
- `preprocess.py` — the angular resample function and the existing
  bandpass are **not** edited; the angular anti-alias filter proposal
  is documented but not applied.
- `models.py` — **not** edited; the 4096-sample window candidate is
  documented but not wired in.
- All `00_lab/`, `01_literature/`, `02_datasets/`, `04_experiments/README.md`,
  `06_results/`, `08_ai_agents/`, `README.md` files: **not** edited.
  These contain pre-existing user modifications from prior sessions; the
  rev-7 commit does **not** stage them.
- `02_datasets/manifests/` and the untracked parts of
  `04_experiments/002_cross_machine_invariance/`: **not** staged. These
  are pre-existing untracked files from prior sessions.
- The `pilot/` outputs and synthetic data: **not** touched.

### Distinctions preserved in the documentation

- **Previously ratified decisions** are not reopened.
- **Corrections to prior calculations** (BPFO/BPFI/BSF/FTF, Kaiser tap
  count) are recorded as a traceable supersession, not silent edits.
- **Newly proposed engineering changes** (4096-sample window, Kaiser
  1394-tap decimation, 92-tap angular anti-alias) are documented as
  **candidates pending owner ratification**.
- **Changes requiring owner ratification** before Stage B are listed
  in §PART 8 below.
- **Real-data evidence obtained from KA01**: **none** (PART 5 inspection
  did not complete).
- **Unverified items** are explicitly listed in each PART.

### Scientific comparison discipline

- **Primary comparison** (preserved): `R2_Env` vs `R3_Order`. Both use
  the same MLP architecture, both consume an envelope-derived spectrum,
  and the only difference is the angular resample step. This is the
  mechanism-controlled comparison.
- **Secondary end-to-end pipeline comparison** (preserved): `R1_Raw` vs
  `R3_Order`. The two pipelines differ in architecture, input domain,
  preprocessing depth, and speed handling; the comparison is
  pipeline-level, not causal.
- UID split, class mapping, operating-condition split, test-set
  isolation, and metric contract are **not** changed.

---

## PART 8 — Validation, commit, and push

### Tests run

`python3 src/test_decim_filter.py` — 4 deterministic checks:

1. Output length after `resample_poly(up=3, down=16)` on a 64000-sample
   input is 12000 samples; effective sampling rate 12.0 kHz. **PASS**.
2. Passband magnitude response at 500 Hz and 5000 Hz is within 0.01 dB
   of unity (H(500) = +0.0006 dB, H(5000) = +0.0046 dB). **PASS**.
3. Stopband attenuation at 6000 Hz is ≥ 60 dB (H(6000) = −65.75 dB). **PASS**.
4. A 7000 Hz input tone is attenuated to ≤ −60 dB at the output (aliased
   5000 Hz bin = −7.30 dB; this is the 7 kHz → alias to 5 kHz component,
   reduced by the FIR attenuation). The non-aliased test (verifying
   the 7 kHz is suppressed before aliasing) requires direct inspection
   of the 192 kHz polyphase signal; documented as a limitation. **PARTIAL**.

The test script prints PASS/FAIL for each check and exits non-zero if
any check fails.

### Files changed in the rev-7 commit

```
A  04_experiments/002_cross_machine_invariance/notes/rev7_dsp_decision_record.md
A  04_experiments/002_cross_machine_invariance/src/preprocess_decim.py
A  04_experiments/002_cross_machine_invariance/src/test_decim_filter.py
```

Pre-existing user modifications and untracked files (10 M + 3 untracked
dirs at the working-tree root) are **not staged** in this commit. They
remain in the working tree and are addressed separately by the owner.

### Commit and push

- **Commit SHA**: recorded below after the commit completes.
- **Branch**: `rev7-dsp-ka01-evidence` (created from
  `arena/6ad75829-tinml-predective` @ `128dda7b`).
- **Remote**: `https://github.com/MohamedMehery/TinML_predective.git`
  (push target).
- **Commit message** (used):
  `Correct DSP assumptions and document KA01 evidence gate`
- **Push result**: recorded below.

### Remaining blockers requiring owner approval

1. **KA01 inspection**: the official KAt-DataCenter TLS connection
   failed in this turn. The owner must (a) provide a working mirror,
   (b) install `unrar` / `7z` and re-try, or (c) ratify an alternative
   evidence-gathering strategy.
2. **Decimation filter ratification**: the 64→12 kHz Kaiser 1394-tap
   FIR design is documented but not wired in. Owner must ratify the
   tap count, the Kaiser β, the 60 dB stopband target, and the
   `experiment_config.json` edit before implementation.
3. **Angular anti-alias filter ratification**: the 92-tap / 74-tap
   speed-dependent Kaiser FIR is documented but not applied. Owner
   must ratify (a) the cutoff rule `fc = 0.4 × samples_per_rev × fr`,
   (b) the speed source as nominal from the condition code, and
   (c) the application point (envelope, before angular resample).
4. **4096-sample window ratification**: owner must ratify the
   4096-sample time-domain window as the engineering candidate
   (replacing the 2048-sample default) before the CNN is rebuilt.
5. **Resource-target demotion**: owner must ratify the demotion of
   the 8.2 kB / 10.5 kB / 71 kOp targets to reported engineering
   targets (and not acceptance gates) for this research stage.
6. **Pre-existing user modifications**: the 10 modified files and
   3 untracked directories in the working tree are **not** staged in
   this rev-7 commit. The owner must decide whether to (a) commit
   them in a separate commit, (b) commit them as part of the same
   commit, (c) reset the working tree, or (d) leave them in place
   for a later decision.
7. **Stage A authorization**: Stage A (download + integrity gate +
   speed-channel check) requires a separate owner authorization and
   is **not** authorized by this rev-7 turn. The official source
   must be reachable, `unrar` / `7z` must be available, and the
   storage check (rev-5 §D) must pass.
8. **Stage B authorization**: training and evaluation are
   **separately authorized** and **not** authorized by this turn.

---

**End of rev-7 record. No files modified outside this record, no
downloads, no training, no evaluation.**
