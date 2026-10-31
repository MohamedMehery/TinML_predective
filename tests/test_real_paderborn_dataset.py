import os, sys, tempfile, math
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

from tests.test_paderborn_mat_reader import make_mat
try:
    from src.datasets import load_real_paderborn_dataset
except Exception:
    from datasets import load_real_paderborn_dataset

try:
    import scipy
    SCIPY = True
except Exception:
    SCIPY = False


def _make_mat(td, name, n=64000):
    t = [i/64000.0 for i in range(n)]
    x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
    y = [{"Name": "vibration_1", "Data": [1.0]*n, "XIndex": 1, "Unit": "m/s2"},
         {"Name": "speed", "Data": [1500.0]*n, "XIndex": 2, "Unit": "rpm"}]
    p = os.path.join(td, name)
    make_mat(p, x_entries=x, y_entries=y)
    return p


def test_window_count_and_tail():
    td = tempfile.mkdtemp()
    p = _make_mat(td, "N15_M07_F10_KA01_1.mat", n=64000)
    ds = load_real_paderborn_dataset(p, window_size=2048, hop_size=1024)
    vib_12k_len = len(ds.r1_raw)*0  # placeholder
    # 64000 samples -> 12000 samples at 12k; windows = (12000-2048)//1024 + 1
    expected = (12000 - 2048)//1024 + 1
    assert ds.r1_raw.shape[0] == expected
    assert ds.window_indices.shape[0] == expected
    # unused tail
    used = (expected - 1)*1024 + 2048
    assert 12000 - used >= 0


def test_required_metadata_fields():
    td = tempfile.mkdtemp()
    p = _make_mat(td, "N15_M07_F10_KA01_1.mat", n=64000)
    ds = load_real_paderborn_dataset(p, window_size=2048, hop_size=1024)
    m = ds.window_metadata[0]
    for key in ["uid", "bearing_id", "physical_bearing_uid", "recording_id",
                "condition_code", "source_file", "window_index", "class_name",
                "label", "nominal_speed_hz", "measured_speed_hz_mean",
                "measured_speed_hz_min", "measured_speed_hz_max"]:
        assert key in m
    n = ds.r1_raw.shape[0]
    assert ds.measured_speed_hz_mean.shape[0] == n
    assert ds.physical_bearing_uids.shape == (n,)
    assert np.all(ds.physical_bearing_uids == "KA01")
    assert np.all(ds.labels == 1)
    assert ds.condition_codes[0] == "N15_M07_F10"
    assert m["bearing_id"] == m["physical_bearing_uid"] == "KA01"
    assert m["recording_id"] == "1"
    assert m["class_name"] == "Outer_Race"
    assert m["label"] == 1


def test_condition_code_rejection():
    td = tempfile.mkdtemp()
    t = [i/64000.0 for i in range(64000)]
    x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
    y = [{"Name": "vibration_1", "Data": [1.0]*64000, "XIndex": 1, "Unit": "m/s2"},
         {"Name": "speed", "Data": [1500.0]*64000, "XIndex": 2, "Unit": "rpm"}]
    p = os.path.join(td, "N99_M07_F10_KA01_1.mat")
    make_mat(p, x_entries=x, y_entries=y)
    try:
        load_real_paderborn_dataset(p)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_speed_selection_by_time_interval():
    td = tempfile.mkdtemp()
    # speed ramps over time so window stats differ; ensure not using 12k indices
    n = 64000
    t = [i/64000.0 for i in range(n)]
    speed = [1500.0 + (i/64000.0)*100.0 for i in range(n)]
    x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
    y = [{"Name": "vibration_1", "Data": [1.0]*n, "XIndex": 1, "Unit": "m/s2"},
         {"Name": "speed", "Data": speed, "XIndex": 2, "Unit": "rpm"}]
    p = os.path.join(td, "N15_M07_F10_KA01_1.mat")
    make_mat(p, x_entries=x, y_entries=y)
    ds = load_real_paderborn_dataset(p, window_size=2048, hop_size=1024)
    # measured mean should vary across windows (not constant)
    means = ds.measured_speed_hz_mean
    assert np.max(means) - np.min(means) > 1e-3


def test_nominal_speed_passed_to_extract(monkeypatch):
    try:
        import src.datasets as D
    except Exception:
        import datasets as D
    captured = {}
    orig = D.extract_representations
    def fake(sig, fs, nominal_speed_hz, **kw):
        captured["nominal"] = nominal_speed_hz
        return orig(sig, fs, nominal_speed_hz, **kw)
    monkeypatch.setattr(D, "extract_representations", fake)
    td = tempfile.mkdtemp()
    p = _make_mat(td, "N15_M07_F10_KA01_1.mat", n=64000)
    load_real_paderborn_dataset(p, window_size=2048, hop_size=1024)
    assert captured["nominal"] == 25.0
