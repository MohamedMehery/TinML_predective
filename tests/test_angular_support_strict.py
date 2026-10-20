import os, sys, math
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_ROOT = os.path.join(REPO_ROOT, "04_experiments", "002_cross_machine_invariance")
SRC = os.path.join(EXP_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from src.preprocess import angular_resample, extract_representations
except Exception:
    from preprocess import angular_resample, extract_representations


def test_supported_request_passes():
    env = np.ones(2048, dtype=np.float32)
    out = angular_resample(env, fs=12000, speed_hz=15.0, samples_per_rev=64,
                          target_len=128, strict_support=True)
    assert out.shape == (128,)


def test_unsupported_request_raises():
    env = np.ones(2048, dtype=np.float32)
    # 256 samples at 15 Hz need ~0.266 s, but the window only supports 0.1706 s.
    try:
        angular_resample(env, fs=12000, speed_hz=15.0, samples_per_rev=64,
                         target_len=256, strict_support=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_strict_error_message_is_diagnostic():
    env = np.ones(2048, dtype=np.float32)
    try:
        angular_resample(env, fs=12000, speed_hz=10.0, samples_per_rev=64,
                         target_len=128, strict_support=True)
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert "outside observed window support" in msg
    assert "requested times" in msg


def test_synthetic_path_retains_clamping_behavior():
    sig = np.ones(4096, dtype=np.float32)
    # Default (non-strict) path must not raise even when the requested span
    # exceeds the observed window support.
    reps = extract_representations(sig, fs=12000, nominal_speed_hz=10.0,
                                 resample_target_len=256)
    assert reps["r3_order"].shape == (3, 1024)
    # Strict path must reject the same request.
    try:
        extract_representations(sig, fs=12000, nominal_speed_hz=10.0,
                               resample_target_len=256,
                               strict_angular_support=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_real_files_all_windows_zero_out_of_support():
    n09 = "/tmp/audit_samples/N09_M07_F10_KA01_1.mat"
    n15 = "/tmp/audit_samples/N15_M07_F10_KA01_1.mat"
    if not (os.path.exists(n09) and os.path.exists(n15)):
        import pytest
        pytest.skip("real KA01 samples not available")
    try:
        from src.datasets import load_real_paderborn_dataset
    except Exception:
        from datasets import load_real_paderborn_dataset
    for path in (n09, n15):
        ds = load_real_paderborn_dataset(path)
        assert ds.r3_order.shape[1] == 1024
        assert ds.r3_order.shape[0] > 0
