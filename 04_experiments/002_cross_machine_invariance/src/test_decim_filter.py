"""Focused deterministic test for the rev-7 64->12 kHz decimation filter.

NOT a training, evaluation, or pipeline test. Uses synthetic in-memory
signals only. Does not require a real recording, the KAt-DataCenter
archive, or any external data source.

Run:
    cd /home/user/TinML_predective
    python3 -m 04_experiments.002_cross_machine_invariance.src.test_decim_filter

The script prints PASS/FAIL for each of the 4 checks and exits non-zero
on any failure.
"""

from __future__ import annotations

import sys
import numpy as np
from scipy.signal import freqz

# Allow direct execution from the src/ directory
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess_decim import (
    decimate_64k_to_12k,
    decim_design_summary,
    decim_filter_coefficients,
    _FS_IN, _FS_OUT, _FS_POLY,
)


def _print_check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return ok


def test_output_length_and_fs() -> bool:
    """Check 1: output length matches expected (n * 3 / 16, ceiling) and
    effective sampling rate is 12.0 kHz."""
    print("Check 1: output length and effective sampling rate")
    n_in = 64000
    t = np.arange(n_in) / _FS_IN
    x = np.cos(2 * np.pi * 1000.0 * t).astype(np.float32)
    y = decimate_64k_to_12k(x)
    expected = int(np.ceil(n_in * 3 / 16))
    detail = (
        f"input {n_in} samples -> output {len(y)} samples "
        f"(expected {expected}); effective fs = {len(y) / (n_in / _FS_IN):.4f} kHz "
        f"(expected {_FS_OUT / 1000:.1f} kHz)"
    )
    return _print_check("output_length", len(y) == expected, detail)


def test_passband_response() -> bool:
    """Check 2: passband magnitude response at 500 Hz and 5000 Hz
    is within 0.01 dB of unity (i.e., < 0.5 dB deviation)."""
    print("Check 2: passband magnitude response")
    h = decim_filter_coefficients()
    w, H = freqz(h, worN=4096, fs=_FS_POLY)
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)
    idx_500 = np.argmin(np.abs(w - 500))
    idx_5000 = np.argmin(np.abs(w - 5000))
    pb_500 = mag_db[idx_500]
    pb_5000 = mag_db[idx_5000]
    ok_500 = abs(pb_500) < 0.01
    ok_5000 = abs(pb_5000) < 0.01
    detail = (
        f"|H(500)| = {pb_500:+.4f} dB (target |.| < 0.01 dB); "
        f"|H(5000)| = {pb_5000:+.4f} dB (target |.| < 0.01 dB)"
    )
    return _print_check("passband_response", ok_500 and ok_5000, detail)


def test_stopband_attenuation() -> bool:
    """Check 3: stopband attenuation at 6000 Hz is >= 60 dB."""
    print("Check 3: stopband attenuation at 6000 Hz")
    h = decim_filter_coefficients()
    w, H = freqz(h, worN=8192, fs=_FS_POLY)
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)
    idx_6000 = np.argmin(np.abs(w - 6000))
    sb_6000 = mag_db[idx_6000]
    ok = sb_6000 <= -60.0
    detail = f"|H(6000)| = {sb_6000:+.2f} dB (target <= -60 dB)"
    return _print_check("stopband_attenuation", ok, detail)


def test_alias_suppression() -> bool:
    """Check 4: a 7000 Hz input tone (which would alias to 5000 Hz
    without adequate anti-aliasing) is attenuated by the FIR so the
    output is dominated by the passband content.

    Test strategy: pass a 1000 Hz tone and a 7000 Hz tone, each with
    the same amplitude. The 1000 Hz tone should be preserved; the
    7000 Hz tone should be attenuated by ~60+ dB at the polyphase
    stage before aliasing. At the 12 kHz output, the 7000 Hz tone
    aliases to 5000 Hz; if the anti-alias filter is correct, the
    energy at the 5 kHz bin should be dominated by the residual
    passband ripple, not by the unattenuated aliased 7 kHz.
    """
    print("Check 4: 7000 Hz alias suppression")
    n_in = 64000
    t = np.arange(n_in) / _FS_IN
    # 1000 Hz tone: in passband, should pass.
    # 7000 Hz tone: in stopband, should be attenuated before aliasing.
    x = (np.cos(2 * np.pi * 1000.0 * t) + np.cos(2 * np.pi * 7000.0 * t)).astype(np.float32)
    y = decimate_64k_to_12k(x)
    Y = np.abs(np.fft.rfft(y))
    fy = np.fft.rfftfreq(len(y), d=1.0 / _FS_OUT)
    idx_1k = np.argmin(np.abs(fy - 1000))
    idx_5k = np.argmin(np.abs(fy - 5000))  # where the 7 kHz aliases to
    # Ratio of energy at the 5 kHz bin to the 1 kHz bin
    ratio_db = 20 * np.log10(Y[idx_5k] / (Y[idx_1k] + 1e-20))
    # With a perfect anti-alias filter, the 7 kHz tone is attenuated by
    # ~60 dB before aliasing, so the aliased 5 kHz component is ~60 dB
    # below the 1 kHz component. With no filter, the 7 kHz aliases to
    # 5 kHz with full amplitude (ratio ~ 0 dB).
    ok = ratio_db <= -50.0  # allow some numerical slack
    detail = (
        f"output spectrum: 1 kHz bin = {Y[idx_1k]:.4f}, 5 kHz bin (aliased 7 kHz) = "
        f"{Y[idx_5k]:.4f}; ratio = {ratio_db:+.2f} dB (target <= -50 dB)"
    )
    return _print_check("alias_suppression", ok, detail)


def main() -> int:
    print(f"=== rev-7 decimation filter test ===")
    print(f"Design summary: {decim_design_summary()}")
    print()
    results = [
        test_output_length_and_fs(),
        test_passband_response(),
        test_stopband_attenuation(),
        test_alias_suppression(),
    ]
    print()
    if all(results):
        print("All checks PASS.")
        return 0
    failed = sum(1 for r in results if not r)
    print(f"{failed} of {len(results)} checks FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
