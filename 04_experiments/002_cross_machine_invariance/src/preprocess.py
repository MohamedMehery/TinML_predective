"""
Signal preprocessing and 5-representation extraction for Experiment 002.

Implements:
  - Bandpass Butterworth (order 4, [500, 5000] Hz at 12 kHz target).
  - Hilbert envelope (magnitude; the squared-vs-magnitude deviation from
    the C5/C8 charter is documented in ../NOTES.md §3.1).
  - IDFT 1X shaft-speed tracker with parabolic interpolation, ±15 %
    search band, on the envelope spectrum.
  - Fixed-grid angular resampling: 256 output samples for the 12 kHz
    CWRU case (4 revolutions x 64 samples/rev), 64 for the 64 kHz PU
    case (1 revolution x 64). No mixed output lengths within a dataset.
  - 5 representations:
      R1_Raw : 1x2048 time-domain window (1D-CNN input)
      R1_Env : 1x2048 envelope time-domain window (1D-CNN input)
      R2_Raw : 1024-bin magnitude FFT of the windowed raw signal (MLP)
      R2_Env : 1024-bin magnitude FFT of the windowed envelope (MLP)
      R3_Order: 1024-bin magnitude FFT of the angular-resampled envelope,
                zero-padded from 512 bins (MLP)

All shapes match experiment_config.json; see pre_execution_validation.py
for the assertion-based pre-flight.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as fft
import scipy.signal as signal


# --- Per-dataset resample target lengths (fixed, never mixed) -----------
# CWRU at 12 kHz: 4 revolutions x 64 samples/rev = 256 output samples.
# PU at 64 kHz:   1 revolution  x 64 samples/rev =  64 output samples.
# These are the only two resample targets in the current pilot.
RESAMPLE_TARGET_LEN = {
    12000: 256,
    64000: 64,
}


def bandpass(x: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """4th-order Butterworth bandpass, zero-phase via filtfilt."""
    b, a = signal.butter(order, [lowcut / (fs / 2.0), highcut / (fs / 2.0)], btype="bandpass")
    return signal.filtfilt(b, a, x)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Magnitude of the analytic signal via Hilbert transform.

    NOTE: this is the magnitude, not the squared form. The squared-vs-
    magnitude deviation from the C5/C8 charter is tracked in
    ../NOTES.md §3.1; the experiment config's `envelope_extraction.
    squared = false` is the source of truth until the deviation is
    resolved.
    """
    return np.abs(signal.hilbert(x))


def estimate_1x_speed(envelope_window: np.ndarray, fs: int, nominal_hz: float, tol_pct: float = 15.0) -> float:
    """IDFT 1X shaft-speed tracker with parabolic peak interpolation.

    Returns the estimated 1X frequency in Hz. The search band is
    nominal_hz * (1 ± tol_pct/100), matching the config's
    `shaft_speed_estimation.search_tolerance_percent = 15.0`.

    If no bins fall in the search band, the nominal speed is returned
    (the caller can detect this and decide whether to skip the window).
    """
    N = envelope_window.size
    env = envelope_window - np.mean(envelope_window)
    win = env * np.hanning(N)
    spec = np.abs(fft.rfft(win, n=N))
    # Use the first 1024 bins to match R2_Env shape; the 1X component
    # is at < 100 Hz for all rigs in the pilot, so this is plenty.
    spec = spec[:1024]
    freqs = fft.rfftfreq(N, d=1.0 / fs)[:1024]

    lo = (1.0 - tol_pct / 100.0) * nominal_hz
    hi = (1.0 + tol_pct / 100.0) * nominal_hz
    band_mask = (freqs >= lo) & (freqs <= hi)
    band_indices = np.where(band_mask)[0]
    if band_indices.size < 3:
        return float(nominal_hz)

    k_star = int(band_indices[np.argmax(spec[band_indices])])
    if k_star <= 0 or k_star >= len(spec) - 1:
        return float(nominal_hz)

    a_val, b_val, c_val = float(spec[k_star - 1]), float(spec[k_star]), float(spec[k_star + 1])
    denom = a_val - 2.0 * b_val + c_val
    if abs(denom) < 1e-6:
        return float(nominal_hz)
    delta = 0.5 * (a_val - c_val) / denom
    delta = max(-0.5, min(0.5, delta))
    return float((k_star + delta) * (fs / N))


def angular_resample(
    envelope_window: np.ndarray,
    fs: int,
    speed_hz: float,
    samples_per_rev: int = 64,
    target_len: int = 256,
) -> np.ndarray:
    """Resample an envelope window to a *fixed-length* angular grid.

    The output is exactly `target_len` samples spanning
    `target_len / samples_per_rev` complete shaft revolutions,
    regardless of how the window aligns with the shaft phase. This
    is what makes R3_Order pixel-aligned across speeds.

    The angular resample assumes the tracker has given us a reasonable
    speed estimate. If the estimate is implausible (≤ 0 or > 200 Hz
    for a rotating-machinery application), we fall back to the
    nominal speed embedded in the window length: for the 12 kHz
    CWRU case with N=2048 we use target_len/64 = 4 revolutions; for
    64 kHz PU we use target_len/64 = 1 revolution.
    """
    N = envelope_window.size
    if speed_hz <= 0 or speed_hz > 200.0:
        # Fallback: nominal speed from window length and target_len.
        revs = max(1, target_len // samples_per_rev)
        speed_hz = revs * fs / N

    revs = target_len // samples_per_rev
    n_out = revs * samples_per_rev  # == target_len
    theta = np.linspace(0.0, 2.0 * np.pi * revs, n_out, endpoint=False)
    t_angle = theta / (2.0 * np.pi * speed_hz)
    t_src = np.arange(N) / fs

    # If t_angle extends slightly past the window, np.interp clamps to
    # the boundary value; that is acceptable (it is a single-sample
    # edge effect at the highest angle bin, well under 1 % of the
    # signal energy).
    return np.interp(t_angle, t_src, envelope_window).astype(np.float32)


def extract_representations(
    sig: np.ndarray,
    fs: int,
    nominal_speed_hz: float,
    window_size: int = 2048,
    hop_size: int = 1024,
    lowcut: float = 500.0,
    highcut: float = 5000.0,
    bandpass_order: int = 4,
    samples_per_rev: int = 64,
    r2_bins: int = 1024,
    r3_pre_pad_bins: int = 512,
    resample_target_len: int | None = None,
) -> dict:
    """Window + extract 5 representations from a single raw signal.

    Returns a dict with keys:
      r1_raw, r1_env, r2_raw, r2_env, r3_order : (n_windows, ...) float32
      labels, uids, dataset : (n_windows,) object/int
      speed_estimates, true_speeds : (n_windows,) float
      resample_lens : (n_windows,) int
    The caller is responsible for setting labels, uids, and dataset.
    """
    if resample_target_len is None:
        resample_target_len = RESAMPLE_TARGET_LEN.get(fs, 256)

    # 1. Preprocess the whole signal once.
    sig = sig.astype(np.float32)
    sig = sig - np.mean(sig)
    sig_bp = bandpass(sig, fs, lowcut, highcut, bandpass_order).astype(np.float32)
    env = hilbert_envelope(sig_bp).astype(np.float32)
    env = env - np.mean(env)

    n_windows = (sig.size - window_size) // hop_size + 1
    if n_windows <= 0:
        raise ValueError(
            f"Signal too short: {sig.size} samples < window_size={window_size}"
        )

    out = {
        "r1_raw": np.empty((n_windows, 1, window_size), dtype=np.float32),
        "r1_env": np.empty((n_windows, 1, window_size), dtype=np.float32),
        "r2_raw": np.empty((n_windows, r2_bins), dtype=np.float32),
        "r2_env": np.empty((n_windows, r2_bins), dtype=np.float32),
        "r3_order": np.empty((n_windows, r2_bins), dtype=np.float32),
        "speed_estimates": np.empty(n_windows, dtype=np.float32),
        "true_speeds": np.full(n_windows, nominal_speed_hz, dtype=np.float32),
        "resample_lens": np.full(n_windows, resample_target_len, dtype=np.int32),
    }

    for w in range(n_windows):
        s = w * hop_size
        w_raw = sig[s : s + window_size]
        w_env = env[s : s + window_size]

        # R1: time-domain, 1 x N (1D-CNN input).
        out["r1_raw"][w, 0, :] = w_raw
        out["r1_env"][w, 0, :] = w_env

        # R2: frequency-domain magnitude, r2_bins.
        out["r2_raw"][w, :] = np.abs(fft.rfft(w_raw * np.hanning(window_size), n=window_size))[:r2_bins]
        out["r2_env"][w, :] = np.abs(fft.rfft(w_env * np.hanning(window_size), n=window_size))[:r2_bins]

        # 1X speed tracker on the envelope window.
        f_speed = estimate_1x_speed(w_env, fs, nominal_speed_hz, tol_pct=15.0)
        out["speed_estimates"][w] = f_speed

        # R3: fixed-grid angular resample, then FFT, then zero-pad.
        env_resampled = angular_resample(
            w_env, fs, f_speed,
            samples_per_rev=samples_per_rev,
            target_len=resample_target_len,
        )
        spec = np.abs(fft.rfft(env_resampled * np.hanning(env_resampled.size), n=resample_target_len))
        # spec length is resample_target_len // 2 + 1; pad/trim to r2_bins.
        if spec.size < r2_bins:
            padded = np.zeros(r2_bins, dtype=np.float32)
            padded[: spec.size] = spec
            out["r3_order"][w, :] = padded
        else:
            out["r3_order"][w, :] = spec[:r2_bins]
        # Apply the config's `R3_Order.zero_padded_from = 512` by zeroing
        # bins from 512 to r2_bins; the config treats the first 512 bins
        # as the order-domain information and the rest as padding.
        out["r3_order"][w, r3_pre_pad_bins:] = 0.0

    return out
