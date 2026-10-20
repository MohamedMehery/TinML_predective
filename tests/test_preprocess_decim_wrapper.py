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
    from src.preprocess import decimate_64k_to_12k as wrapper
except Exception:
    from preprocess import decimate_64k_to_12k as wrapper
try:
    from src.preprocess_decim import decimate_64k_to_12k as original
except Exception:
    from preprocess_decim import decimate_64k_to_12k as original


def test_wrapper_delegates_to_verified_routine():
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    a = wrapper(x)
    b = original(x)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_wrapper_output_length_convention():
    for n in (16, 100, 256, 1000):
        x = np.zeros(n, dtype=np.float32)
        out = wrapper(x)
        expected = math.ceil(n * 3 / 16)
        assert out.shape == (expected,)
