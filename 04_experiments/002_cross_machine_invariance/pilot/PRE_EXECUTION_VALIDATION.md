# Pre-Execution Validation Script

This document describes `pre_execution_validation.py` (in this directory)
and its three checks. The script is a pre-execution gate; it does not
need to be re-run for every transfer pair.

## What the three checks actually test

| # | Check | What it tests | What it does NOT test |
|---|---|---|---|
| 1 | `validate_filter_stability` | The 4th-order Butterworth bandpass [500, 5000] Hz at fs=12 kHz (the `experiment_config.json` settings) has all poles strictly inside the unit circle. | That the filter is *appropriate* for the experiment (e.g. that 500 Hz is the right low-cut). That's a charter decision, not a math check. |
| 2 | `validate_speed_tracking` | The IDFT 1X tracker recovers a 29.5 Hz (1770 RPM) shaft speed to within 2 % on a clean, noise-free, modulated synthetic signal, using the configured `search_tolerance_percent = 15.0`. | The tracker's behavior under low SNR. The real failure mode is PU outer-race at SNR ≈ 3 dB (per the pilot's `snr_db=3.0`); a low-SNR extension is TBD. |
| 3 | `validate_causal_identity` (corrected) | R2-Env (1024-bin envelope rFFT) and R3-Order (1024-bin zero-padded order rFFT after 64-samples/rev angular resampling) are both derived from the same N=2048 input window; the angular resample does not change the envelope power by more than 10× (i.e. it does not silently consume data from outside the window). | That R2-Env and R3-Order produce the *same* spectrum — they shouldn't; R3-Order is supposed to be more cross-machine stable by design. The check verifies the *shape contract* and *causal containment*, not spectral equality. |

## How to run

```bash
cd 04_experiments/002_cross_machine_invariance/pilot
python pre_execution_validation.py
```

Expected output (last 5 lines):

```
[ok] DSP Bandpass Filter Stable: order=4, [500, 5000] Hz at fs=12000 Hz, max pole magnitude = 0.978... < 1.0
[ok] 1X Speed Tracking Validated: estimated 29.50... Hz (true: 29.500 Hz, error: 0.0...%, tolerance band: [25.08, 33.93] Hz)
[ok] Causal Identity Verified: R2_Env shape (1024,), R3_Order shape (1024,), both derived from same N=2048 window; angular resample output length = ...; power ratio = ...x

ALL PRE-EXECUTION VALIDATION TESTS PASSED.
```

## Why check #3 was rewritten

The original third check (in the script you sent me) was named
`validate_causal_identity` and its docstring said *"Verify that order
tracking and envelope spectra consume strictly causal identical
windows."* But the body was `assert len(env) == N` after a Hilbert
transform on `np.random.randn` — which is a length sanity check, not a
causality check, and it did not construct R2-Env or R3-Order at all.
A check whose name overstates what it tests is exactly the
Confirmation-Bias failure mode the lab's research principles
(`00_lab/research-principles.md` rule 7) warn against. The corrected
check constructs both representations from the same N-sample window
and asserts shape contract + causal containment, which is what the
name promised.

## Where this fits in the experiment

This script is a **pre-execution gate**, not a runtime module. The
runtime pipeline lives in `04_experiments/002_cross_machine_invariance/src/`
(TBD). The smoke-test pilot data lives in this `pilot/` directory
(`synthetic/raw_data/*.npz`, git-ignored) and is consumed by `src/`
when that lands. The pre-execution validation log in
`04_experiments/002_cross_machine_invariance/validation_log.txt` is a
*different* pre-execution check — it validates
`experiment_config.json` against the C5/C8 charter. The two layers
are complementary:

- `validation_log.txt` + `validate_config.py`: does the *config* match
  the *charter*?
- `pre_execution_validation.py`: do the *mathematical preconditions*
  of the configured pipeline hold?
