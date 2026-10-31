# Decision Memo — Envelope Representation (DRAFT, awaiting research-lead approval)

**Status:** DRAFT — Not yet approved. Recorded for traceability, not as a decision.

**Date drafted:** 2026-10-12
**Phase:** 20 (evidence-driven gap closure, research audit, controlled repair)
**Author:** audit / pending research-lead approval

---

## 1. Question being decided

Should the EXP-002 envelope be:
  (a) Hilbert magnitude (`|H{x}(t)|`), matching the current code and config;
  (b) Squared Hilbert envelope (`|H{x}(t)|²`), matching the ratified C5/C8
      charter text;
  (c) a third representation; or
  (d) remain explicitly OPEN until a low-SNR validation experiment is
      executed?

## 2. Evidence currently in the repository

- `candidate-questions.md` line 37 (charter): "Squared Hilbert Envelope".
- `experiment_config.json` line 23-26: `envelope_extraction.method =
  "hilbert_magnitude"`, `squared = false`.
- `src/preprocess.py:48-57` `hilbert_envelope(x)`: returns
  `np.abs(signal.hilbert(x))` — magnitude, with an explicit NOTE in the
  docstring acknowledging the deviation.
- `src/preprocess.py:170` (in `extract_representations`): subtracts
  `np.mean(env)` from the envelope **before** any subsequent processing
  (R1_Env, R2_Env, R3_Order, and the IDFT 1X tracker input).
- `validation_log.txt` Stage 1b line 34-35: `[WARN] charter specifies
  "Squared Hilbert Envelope" but config uses "Hilbert magnitude"`. Records
  the deviation as an unresolved follow-up.
- `NOTES.md` §3.1: status OPEN, owner tbd; "Plan: when `src/preprocess.py`
  is written, implement the squared envelope in parallel and run a
  CWRU-subset ablation."
- `decision-log.md` 2026-10-11 (Phase 15 row): explicitly OPEN.
- `README.md` §6 (lines 59-67): the operational justification for
  preferring magnitude. Claims: (i) squared form has a strong DC component,
  (ii) squared noise variance destabilizes the IDFT 1X tracker at low SNR,
  (iii) magnitude preserves non-negative half-rectified structure.

## 3. Mathematical analysis (no execution; theoretical)

Let $x(t) = s(t) + n(t)$ where $s$ is the defect-modulated carrier and
$n \sim \mathcal{N}(0, \sigma^2)$ is additive white noise.

- $y_{\text{abs}}(t) = |\mathcal{H}\{x(t)\}|$
- $y_{\text{sq}}(t) = |\mathcal{H}\{x(t)\}|^2$

For a 1X-amplitude-modulated 3.5 kHz carrier with $|s(t)| \sim A$ and
$\sigma^2 = P_n$:

- $E[y_{\text{abs}}] = |s| \cdot \text{erf}(|s|/(\sigma\sqrt{2})) + \sigma\sqrt{2/\pi} e^{-|s|^2/(2\sigma^2)}$
  (at high SNR, $\to |s|$; the DC component is the rectified carrier mean).
- $E[y_{\text{sq}}] = |s|^2 + \sigma^2$ (DC is $|s|^2$ plus noise variance).
- $\text{Var}(y_{\text{sq}}) \approx 2|s|^2 \sigma^2 + 2\sigma^4$.

**DC handling in current code (`preprocess.py:170`):** the mean is
subtracted after envelope extraction. This subtraction applies equally to
both representations. **The "strong DC component" cited in README §6 is
therefore eliminated by the code as written, regardless of which envelope
is chosen.** The DC-difference claim in the README is misleading as a
practical justification for the squared-vs-magnitude choice.

**Spectral SNR difference (the residual effect after mean subtraction):**

For a 1X peak in $|s|$ with amplitude $A$ relative to a noise floor of
variance $P_n$:

- In $y_{\text{abs}}$: $Y_{\text{abs,1X}} \sim A/2$, $Y_{\text{abs,noise}} \sim \sigma\sqrt{2/\pi}$.
  Spectral SNR (1X / noise) $\approx A / (2\sigma\sqrt{2/\pi}) \approx A/(1.6\sigma)$.
- In $y_{\text{sq}} = y_{\text{abs}}^2$: $Y_{\text{sq,1X}} \sim A^2/4$ (DC + 2×1X from squaring),
  $Y_{\text{sq,noise}} \sim 2(A/\pi)\sigma\sqrt{2/\pi} \approx 0.51 A\sigma$.
  Spectral SNR $\approx A^2/4 / (0.51 A\sigma) = A/(2.04\sigma)$.

**Net effect:** the spectral SNR of the 1X component relative to noise is
approximately the **same** for $y_{\text{abs}}$ and $y_{\text{sq}}$
($A/(1.6\sigma)$ vs $A/(2.04\sigma)$ — a ~28% degradation for squared).
The mean subtraction removes the DC bias difference; the residual
spectral SNR difference is modest.

**The "destabilizes the IDFT 1X speed tracker" claim in README §6** is
therefore not strongly supported by the spectral analysis. The tracker
performance depends on the (1X peak height / adjacent-bin noise) ratio,
which is comparable between the two representations after mean
subtraction.

## 4. Empirical evidence in the repository

- `pilot/pre_execution_validation.py` lines 25-32 (TBD): "The real-data
  failure mode (per the decision log) is PU outer-race at low SNR; this
  check does not exercise that case. A low-SNR extension is TBD."
- `pilot/results/.gitkeep` line 17: planned `PILOT_pipeline_sanity.json`
  (tracker convergence, window counts) — **does not exist on disk**.
- No `PILOT_pipeline_sanity.json` exists in `pilot/results/`.

**No low-SNR test result exists in the repository.** The claim that the
squared form "destabilizes the IDFT 1X tracker at low SNR" is **not
backed by any executed experiment in this codebase**.

## 5. Paper-matrix evidence

- P15, P16, P20, P27, P38, P39 (the CWRU/PU/SEU-related papers) do not
  discuss envelope type. None of them compare magnitude vs squared.
- The "Vieira 2026" reference that was once cited as the source of the
  "squared envelope" requirement was removed in Phase 11 after the audit
  could not locate it.

**No paper in the repository's literature matrix supports either
representation choice as superior for the C5/C8 hypothesis.**

## 6. Available alternatives and their consequences

- **Option A: Ratify magnitude (status quo).** Keep
  `experiment_config.json` and `preprocess.py` unchanged; amend the
  charter text in `candidate-questions.md` to match ("Hilbert magnitude"
  instead of "Squared Hilbert Envelope"). Add a low-SNR tracker
  validation test in `pilot/pre_execution_validation.py` to close the
  evidence gap.
- **Option B: Switch to squared.** Change `envelope_extraction.squared`
  to `true` in `experiment_config.json`, change
  `preprocess.py:hilbert_envelope` to return `np.abs(signal.hilbert(x))**2`,
  re-validate (PRE_EXECUTION_VALIDATION.md's low-SNR case is now
  mandatory), and amend the charter text accordingly. If the low-SNR
  validation fails (e.g., the tracker cannot converge at SNR = 3 dB),
  revert to magnitude.
- **Option C: Implement both, default to magnitude, expose a CLI flag for
  the squared form.** Keep the current code as the default; add a
  parallel `hilbert_envelope_squared` function; add a flag in
  `run_pipeline.py` to choose between them. This is the
  "implementation-first" approach that NOTES.md §3.1 proposed.
- **Option D: Remain OPEN.** Do not change the code, config, or charter.
  Add a low-SNR validation test as a separate technical task; defer the
  scientific decision until that test is executed and its result is
  reviewed. This is the most conservative option.

**Recommendation:** **Option A** (ratify magnitude) is the
operationally-cheapest and best-supported by the existing code, but the
README §6 justification should be updated to reflect the actual
mathematics (mean subtraction removes the DC difference; the residual
spectral SNR difference is modest; no paper supports either choice).
**Option D** (remain OPEN + add low-SNR test) is the most
scientifically-defensible but leaves the existing code in a
documented-deviation state. The research lead's call.

## 7. Specific changes proposed (subject to research-lead approval)

For Option A:
1. Amend `candidate-questions.md` line 37 to read "Hilbert magnitude"
   (or "Hilbert envelope, magnitude form"). Document the rationale in
   the charter text.
2. Update `experiment_config.json` `envelope_extraction.method` from
   `"hilbert_magnitude"` to `"hilbert_envelope"` for clarity, OR add an
   explanatory comment in the JSON.
3. Update `README.md` §6 to:
   - Acknowledge that mean subtraction eliminates the DC-difference
     rationale;
   - State the residual spectral SNR difference (~28% degradation for
     squared) is modest;
   - Note that no paper in the literature matrix supports either choice;
   - State that the choice is provisional pending low-SNR validation.
4. Add a `validate_speed_tracking_low_snr` function to
   `pilot/pre_execution_validation.py` that tests the tracker at SNR =
   30, 20, 10, 3, 0, -3 dB on a synthetic 1X-modulated carrier, with the
   pass criterion being `≥ 95% pass rate (2% threshold) at SNR ≥ 10 dB`
   and `≥ 80% pass rate (5% threshold) at SNR = 3 dB`. This is a
   category-2 (unambiguous implementation) addition.

For Option B: change the code, change the config, re-validate. Larger
blast radius; potentially changes downstream results if the pilot is
re-run.

For Option C: implement both, keep magnitude as the default, add the
ablation. The same ablation script in Option A would still be needed
later.

For Option D: do not change any frozen file. Add the low-SNR test as a
new file under `pilot/`. Defer the decision.

## 8. Limitations

- The mathematical analysis in §3 is approximate (assumes Gaussian noise,
  single-tone carrier, no aliasing). The real CWRU/PU/SEU signals are
  more complex.
- The IDFT 1X tracker also depends on the Hann window and parabolic
  interpolation, which interact with the spectral SNR in non-trivial
  ways. A small synthetic validation is the only way to bound the actual
  performance difference.
- The pilot is **synthetic**. The squared-vs-magnitude difference in a
  real noisy environment may differ from the synthetic estimate. The
  proposed low-SNR test, if executed, would be a synthetic-only
  measurement; the real-data equivalent would require real-data
  access, which is currently blocked by the licensing decision.

---

**End of decision memo. Status: DRAFT, awaiting research-lead approval.**
