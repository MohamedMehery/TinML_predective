# Synthetic Manifests — PILOT ONLY

The files in this directory describe **synthetic vibration data** generated
by `../src/generate_synthetic_benchmark.py`. They are **not** part of the
C5/C8 evidence base. The real CWRU / Paderborn / SEU manifests live in
`02_datasets/manifests/` and are the authoritative source for
Experiment 002.

| File | Purpose |
|------|---------|
| `pilot_manifest.json` | Per-record index of the synthetic `.npz` files (uid, fault_class, fs_hz, speed_rpm, num_samples, file path). Regenerated every run. |
| `README.md` | This file. |
