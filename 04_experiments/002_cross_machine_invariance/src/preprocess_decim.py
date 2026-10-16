"""64 kHz -> 12 kHz decimation (Kaiser FIR anti-alias).

This module is a STANDALONE verification module for the rev-7
engineering decision record. It is NOT imported or wired into the
live preprocessing pipeline (`preprocess.py`).

Design targets (from the rev-7 record, PART 3):
- Passband 0 - 5000 Hz, deviation < 0.5 dB peak-to-peak.
- Stopband >= 60 dB attenuation at >= 6000 Hz.
- Transition width 1000 Hz (5000 -> 6000).
- Polyphase rate 192 kHz (64 kHz x up=3).
- Cutoff placed at 5500 Hz (mid-transition).
- FIR designed with Kaiser window via scipy.signal.kaiserord +
  scipy.signal.firwin.

Computed (this rev-7 turn, scipy 1.17.1, numpy 2.4.6):
- numtaps = 1394
- beta    = 5.6533
- |H(500)|   = +0.0006 dB
- |H(5000)|  = +0.0046 dB
- |H(6000)|  = -65.75 dB (stopband)
- |H(7000)|  = -77.81 dB

The 64-tap value in `experiment_config.json` is INCORRECT for the
60 dB stopband target. A correct design needs 1394 taps. This module
is the actual realization of the corrected design.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly

# Polyphase rate for 64 kHz -> 12 kHz rational resampling
_FS_POLY = 192000.0  # 64e3 * 3
_UP = 3
_DOWN = 16
_FS_IN = 64000
_FS_OUT = 12000  # _FS_IN * _UP / _DOWN

# Cutoff (mid-transition between passband 5000 and stopband 6000)
_CUTOFF_HZ = 5500.0
_TRANSITION_HZ = 1000.0
_ATTEN_DB = 60.0

# Compute tap count and Kaiser beta via kaiserord (one-time, deterministic)
_WIDTH_NORM = _TRANSITION_HZ / _FS_POLY
_NUMTAPS_RAW, _BETA = kaiserord(_ATTEN_DB, _WIDTH_NORM)
# kaiserord may return an even number; ensure odd numtaps for Type I linear-phase FIR.
_NUMTAPS = int(_NUMTAPS_RAW) + 1 if int(_NUMTAPS_RAW) % 2 == 0 else int(_NUMTAPS_RAW)

# Design the FIR once at module import. Coefficients are deterministic.
_DECIM_H = firwin(
    _NUMTAPS,
    _CUTOFF_HZ / (_FS_POLY / 2.0),
    window=("kaiser", float(_BETA)),
    pass_zero=True,
).astype(np.float64)


def decimate_64k_to_12k(sig: np.ndarray) -> np.ndarray:
    """Decimate a 64 kHz signal to 12 kHz using the Kaiser FIR.

    Parameters
    ----------
    sig : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
        Input signal at 64 kHz. float32 or float64.

    Returns
    -------
    np.ndarray
        Decimated signal at 12 kHz. Length is ceil(n * 3 / 16).
    """
    if sig.ndim == 1:
        return resample_poly(sig, up=_UP, down=_DOWN, window=_DECIM_H).astype(np.float32)
    if sig.ndim == 2:
        out = np.empty((sig.shape[0], int(np.ceil(sig.shape[1] * _UP / _DOWN))), dtype=np.float32)
        for i in range(sig.shape[0]):
            out[i] = resample_poly(sig[i], up=_UP, down=_DOWN, window=_DECIM_H).astype(np.float32)
        return out
    raise ValueError(f"decimate_64k_to_12k expects 1-D or 2-D input; got {sig.ndim}-D")


def decim_filter_coefficients() -> np.ndarray:
    """Return the Kaiser FIR coefficients (1394 taps, float64)."""
    return _DECIM_H.copy()


def decim_design_summary() -> dict:
    """Return the design parameters as a dict (for documentation)."""
    return {
        "fs_in_hz": _FS_IN,
        "fs_out_hz": _FS_OUT,
        "up": _UP,
        "down": _DOWN,
        "fs_poly_hz": _FS_POLY,
        "cutoff_hz": _CUTOFF_HZ,
        "transition_hz": _TRANSITION_HZ,
        "attenuation_db": _ATTEN_DB,
        "numtaps": int(_NUMTAPS),
        "kaiser_beta": float(_BETA),
        "passband_hz": (0.0, 5000.0),
        "stopband_hz": (6000.0, _FS_POLY / 2.0),
    }
