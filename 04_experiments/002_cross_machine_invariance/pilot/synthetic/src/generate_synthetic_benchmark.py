"""
Deterministic Physical-Dynamics Generator — PILOT SMOKE TEST ONLY.

This script generates **synthetic** vibration waveforms for a
pipeline-shape smoke test of Experiment 002. It is NOT a substitute for
the real CWRU / Paderborn / SEU public benchmarks. The synthetic signals
are produced by a deterministic kinematic + structural-resonance model
parameterized with the *bearing geometry constants* of the real rigs
(SKF 6205-2RS for CWRU, 6203 for Paderborn 2016), but the signals
themselves are constructed from `np.sin` + `np.convolve` + Gaussian
noise, not from measurements.

DO NOT cite the output of this script as a result of Experiment 002.
The real experiment result comes from the public benchmarks documented
in `02_datasets/manifests/cwru.md`, `paderborn.md`, and `seu.md`.

Output paths default to inside this experiment's `pilot/` directory,
NOT to `02_datasets/`, so the synthetic data cannot be confused with
the real dataset cards.

The .npz files in `raw_data/` are git-ignored (see repo-root
`.gitignore`). Only this script and the `pilot_manifest.json` index are
tracked.
"""

import json
import os

import numpy as np

# --- Output paths: locked to the pilot folder, not 02_datasets/ -----
# File layout (relative to this script):
#   .../pilot/synthetic/src/generate_synthetic_benchmark.py   <-- __file__
#   .../pilot/synthetic/src/                                 <-- dirname 1
#   .../pilot/synthetic/                                     <-- dirname 2
#   .../pilot/                                               <-- dirname 3  (= PILOT_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PILOT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
OUTPUT_DIR = os.path.join(PILOT_ROOT, "synthetic", "raw_data")
MANIFEST_DIR = os.path.join(PILOT_ROOT, "synthetic", "manifests")

# --- Bearing geometry (SKF 6205-2RS for CWRU drive end, 6203 for PU 2016) -
CWRU_BEARINGS = {
    "6205_2RS_JEM_SKF": {
        "n_balls": 9,
        "ball_dia_mm": 7.94,
        "pitch_dia_mm": 39.04,
        "contact_angle_deg": 0.0,
        "bpfo_order": 3.5848,
        "bpfi_order": 5.4152,
        "bsf_order": 2.3220,
        "ftf_order": 0.3983,
    }
}

PU_BEARINGS = {
    "6203": {
        "n_balls": 8,
        "ball_dia_mm": 6.75,
        "pitch_dia_mm": 28.5,
        "contact_angle_deg": 0.0,
        "bpfo_order": 3.0526,
        "bpfi_order": 4.9474,
        "bsf_order": 1.9904,
        "ftf_order": 0.3816,
    }
}


def generate_bearing_signal(
    fs,
    duration_sec,
    speed_rpm,
    fault_type,
    bearing_geom,
    resonance_freq=3500.0,
    resonance_damping=0.08,
    snr_db=5.0,
    seed=42,
):
    """Generate a SYNTHETIC bearing vibration signal.

    This is a deterministic physical-dynamics model, NOT a measurement.
    See module docstring.
    """
    np.random.seed(seed)
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    f_shaft = speed_rpm / 60.0

    # 1. Shaft rotation fundamental and harmonics
    x_shaft = 0.4 * np.sin(2.0 * np.pi * f_shaft * t) + 0.15 * np.sin(4.0 * np.pi * f_shaft * t)

    # 2. Fault impulse train
    if fault_type == "Healthy":
        fault_impulses = np.zeros(n_samples)
    else:
        if fault_type == "Outer_Race":
            f_impact = f_shaft * bearing_geom["bpfo_order"]
        elif fault_type == "Inner_Race":
            f_impact = f_shaft * bearing_geom["bpfi_order"]
        elif fault_type == "Ball":
            f_impact = f_shaft * (2.0 * bearing_geom["bsf_order"])
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        period_samples = fs / f_impact
        impulse_indices = np.arange(0, n_samples, period_samples).astype(int)
        impulse_indices = impulse_indices[impulse_indices < n_samples]

        fault_impulses = np.zeros(n_samples)
        # Add jitter
        for idx in impulse_indices:
            jitter = int(np.random.normal(0, period_samples * 0.02))
            actual_idx = max(0, min(n_samples - 1, idx + jitter))
            fault_impulses[actual_idx] = 1.0

        # Inner race amplitude modulation by 1X shaft rotation
        if fault_type == "Inner_Race":
            mod_envelope = 1.0 + 0.8 * np.cos(2.0 * np.pi * f_shaft * t)
            fault_impulses = fault_impulses * mod_envelope

    # 3. Structural resonance response (decaying sinusoid convolution)
    impulse_response_len = int(fs * 0.02)
    t_ir = np.arange(impulse_response_len) / fs
    h = (
        np.exp(-resonance_damping * 2.0 * np.pi * resonance_freq * t_ir)
        * np.sin(2.0 * np.pi * resonance_freq * t_ir)
    )

    structural_vibration = np.convolve(fault_impulses, h, mode="same")

    # 4. Total signal + background noise
    raw_signal = x_shaft + 2.5 * structural_vibration
    sig_power = np.mean(raw_signal ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(max(1e-6, noise_power)), n_samples)

    return (raw_signal + noise).astype(np.float32)


def materialize_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    manifest_records = []
    seed_idx = 100

    # --- CWRU (Source) ---------------------------------------------
    cwru_speeds = [1797, 1772, 1750, 1730]
    cwru_faults = [
        # (fault_type, uid_base, fault_size_mils)
        ("Healthy", "CWRU_N", 0),
        ("Outer_Race", "CWRU_OR_007", 7),
        ("Outer_Race", "CWRU_OR_014", 14),
        ("Outer_Race", "CWRU_OR_021", 21),
        ("Inner_Race", "CWRU_IR_007", 7),
        ("Inner_Race", "CWRU_IR_014", 14),
        ("Inner_Race", "CWRU_IR_021", 21),
        ("Ball", "CWRU_B_007", 7),
        ("Ball", "CWRU_B_014", 14),
        ("Ball", "CWRU_B_021", 21),
    ]

    # CWRU Healthy naming: use one record per speed; the file already
    # contains the speed in its name, so the UID is just the class label.
    for f_type, uid_base, f_size in cwru_faults:
        for speed_rpm in cwru_speeds:
            if f_type == "Healthy":
                uid = "CWRU_N"
            else:
                uid = uid_base

            fname = f"{uid}_speed{speed_rpm}.npz"
            fpath = os.path.join(OUTPUT_DIR, fname)

            sig = generate_bearing_signal(
                fs=12000,
                duration_sec=10.0,
                speed_rpm=speed_rpm,
                fault_type=f_type,
                bearing_geom=CWRU_BEARINGS["6205_2RS_JEM_SKF"],
                resonance_freq=3800.0,
                resonance_damping=0.06,
                snr_db=6.0,
                seed=seed_idx,
            )
            seed_idx += 1
            np.savez_compressed(fpath, DE_time=sig)

            manifest_records.append(
                {
                    "dataset": "CWRU",
                    "uid": uid,
                    "filepath": os.path.relpath(fpath, PILOT_ROOT),
                    "fault_class": f_type,
                    "fault_size_mils": f_size,
                    "fs_hz": 12000,
                    "speed_rpm": speed_rpm,
                    "nominal_speed_hz": speed_rpm / 60.0,
                    "num_samples": len(sig),
                    "synthetic": True,   # explicit tag, see pilot/README.md
                }
            )

    # --- Paderborn (Target) -----------------------------------------
    pu_speeds = [1500, 900]
    pu_faults = [
        ("Healthy", ["PU_K001", "PU_K002", "PU_K003"]),
        ("Outer_Race", ["PU_KA01", "PU_KA03", "PU_KA05", "PU_KA07", "PU_KA09"]),
        ("Inner_Race", ["PU_KI01", "PU_KI03", "PU_KI05", "PU_KI07", "PU_KI08"]),
        ("Ball", ["PU_KB23", "PU_KB24", "PU_KB27"]),
    ]

    for f_type, uids in pu_faults:
        for uid in uids:
            for speed_rpm in pu_speeds:
                fname = f"{uid}_speed{speed_rpm}.npz"
                fpath = os.path.join(OUTPUT_DIR, fname)

                sig = generate_bearing_signal(
                    fs=64000,
                    duration_sec=5.0,
                    speed_rpm=speed_rpm,
                    fault_type=f_type,
                    bearing_geom=PU_BEARINGS["6203"],
                    resonance_freq=2400.0,
                    resonance_damping=0.10,
                    snr_db=3.0,
                    seed=seed_idx,
                )
                seed_idx += 1
                np.savez_compressed(fpath, vibration_radial=sig)

                manifest_records.append(
                    {
                        "dataset": "PU",
                        "uid": uid,
                        "filepath": os.path.relpath(fpath, PILOT_ROOT),
                        "fault_class": f_type,
                        "fault_size_mils": 0,
                        "fs_hz": 64000,
                        "speed_rpm": speed_rpm,
                        "nominal_speed_hz": speed_rpm / 60.0,
                        "num_samples": len(sig),
                        "synthetic": True,   # explicit tag
                    }
                )

    manifest_path = os.path.join(MANIFEST_DIR, "pilot_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2)

    print(
        f"[pilot] Materialized {len(manifest_records)} SYNTHETIC records "
        f"to {manifest_path}."
    )
    print(
        "[pilot] NOTE: these signals are generated by a deterministic "
        "kinematic + structural-resonance model. They are NOT the real "
        "CWRU / Paderborn public benchmarks. See pilot/README.md."
    )


if __name__ == "__main__":
    materialize_datasets()
