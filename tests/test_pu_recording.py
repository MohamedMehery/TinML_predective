import os, sys, tempfile, unittest
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
try:
    import numpy as np
    from scipy.io import savemat
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
from tests.test_paderborn_mat_reader import make_mat
from tools.paderborn_mat_reader.recording import load_pu_recording, PuRecordingError
from tools.paderborn_mat_reader.reader import MatReaderError

@unittest.skipUnless(SCIPY_AVAILABLE, "scipy/numpy not available")
class TestPuRecording(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
    def _make(self, name, **kw):
        p = os.path.join(self.td, name)
        make_mat(p, **kw)
        return p
    def test_speed_unit_required(self):
        p = self._make("N15_M07_F10_KA01_1.mat")
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm")
        with self.assertRaises(TypeError):
            load_pu_recording(p)
    def test_unsupported_condition_code(self):
        p = self._make("N99_M07_F10_KA01_1.mat")
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_filename_parsing(self):
        t = [i/64000.0 for i in range(5)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*5, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [900.0]*5, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N09_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertEqual(rec.metadata["bearing_id"], "KA01")
        self.assertEqual(rec.metadata["condition_code"], "N09_M07_F10")
        self.assertEqual(rec.metadata["nominal_rpm"], 900)
    def test_rpm_to_hz_conversion(self):
        t = [i/64000.0 for i in range(10)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*10, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*10, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        import numpy as np
        self.assertTrue(np.allclose(rec.speed_hz, 25.0))
    def test_interpolation_uses_speed_axis(self):
        tv = [0.0, 1.0/64000.0, 2.0/64000.0]; ts = [0.0, 2.0/64000.0]
        x = [{"Name": "TimeVib", "Data": tv, "Unit": "s"}, {"Name": "TimeSpeed", "Data": ts, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1490.0, 1510.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertAlmostEqual(rec.speed_hz[1], 25.0, places=6)
    def test_common_support_truncation_and_drops(self):
        tv = [i/64000.0 for i in range(6)]; ts = [1.5/64000.0, 2.5/64000.0, 3.5/64000.0]
        x = [{"Name": "TimeVib", "Data": tv, "Unit": "s"}, {"Name": "TimeSpeed", "Data": ts, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0, 1500.0, 1500.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertEqual(rec.vibration_time.tolist(), [2/64000.0, 3/64000.0])
        self.assertEqual(rec.metadata["dropped_begin"], 2)
        self.assertEqual(rec.metadata["dropped_end"], 2)
        self.assertEqual(rec.metadata["dropped_total"], 4)
        self.assertEqual(rec.metadata["retained_count"], 2)
    def test_preservation_of_values_and_timestamps(self):
        t = [i/64000.0 for i in range(4)]; vals = [1.5, 2.5, 3.5, 4.5]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": vals, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*4, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertEqual(rec.vibration.tolist(), vals)
        self.assertEqual(rec.vibration_time.tolist(), t)
    def test_nonfinite_timestamps_rejected(self):
        t = [0.0, float("nan"), 0.2]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": [0.0, 0.1, 0.2], "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises((PuRecordingError, MatReaderError)):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_nonincreasing_timestamps_rejected(self):
        t = [0.0, 0.2, 0.1]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": [0.0, 0.1, 0.2], "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises((PuRecordingError, MatReaderError)):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_nonfinite_values_rejected(self):
        t = [0.0, 0.1, 0.2]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, float("nan"), 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_no_common_support_rejected(self):
        tv = [0.0, 0.1, 0.2]; ts = [10.0, 10.1, 10.2]
        x = [{"Name": "TimeVib", "Data": tv, "Unit": "s"}, {"Name": "TimeSpeed", "Data": ts, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_speed_gate_fails(self):
        t = [i/64000.0 for i in range(10)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*10, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1000.0]*10, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_output_arrays(self):
        t = [i/64000.0 for i in range(10)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*10, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*10, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        import numpy as np
        self.assertEqual(rec.vibration.dtype, np.float64)
        self.assertEqual(rec.vibration_time.dtype, np.float64)
        self.assertEqual(rec.speed_hz.dtype, np.float64)
        self.assertEqual(rec.fs_nominal_hz, 64000)
        self.assertEqual(len(rec.vibration), len(rec.vibration_time))
        self.assertEqual(len(rec.vibration), len(rec.speed_hz))
        self.assertTrue(np.all(rec.vibration_time >= rec.metadata["t_start"]))
        self.assertTrue(np.all(rec.vibration_time <= rec.metadata["t_end"]))
    def test_missing_speed_unit_keyword(self):
        p = self._make("N15_M07_F10_KA01_1.mat")
        with self.assertRaises(TypeError):
            load_pu_recording(p)
    def test_endpoint_rate_gate_fails(self):
        t = [i*0.001 for i in range(100)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*100, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*100, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
    def test_residual_gate_fails(self):
        import math
        t = [i/64000.0 + 5e-5*math.sin(math.pi*i/199) for i in range(200)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*200, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*200, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")

    def test_custom_nominal_sampling_rate(self):
        t = [i/32000.0 for i in range(10)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*10, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*10, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed", fs_nominal_hz=32000)
        self.assertEqual(rec.fs_nominal_hz, 32000)
        self.assertAlmostEqual(rec.fs_endpoint_hz, 32000.0, places=3)

    def test_custom_endpoint_rate_tolerance(self):
        t = [i/63000.0 for i in range(100)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*100, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*100, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed", fs_nominal_hz=64000, endpoint_rate_rel_tol=0.001)
        rec = load_pu_recording(p, speed_unit="rpm_assumed", fs_nominal_hz=64000, endpoint_rate_rel_tol=0.02)
        self.assertAlmostEqual(rec.fs_endpoint_hz, 63000.0, places=3)

    def test_custom_residual_limit(self):
        import math
        t = [i/64000.0 + 5e-5*math.sin(math.pi*i/199) for i in range(200)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*200, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*200, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
        rec = load_pu_recording(p, speed_unit="rpm_assumed", max_abs_residual_s=1e-4)
        self.assertEqual(rec.metadata["retained_count"], 200)

    def test_custom_speed_tolerance(self):
        t = [i/64000.0 for i in range(10)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*10, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1400.0]*10, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        with self.assertRaises(PuRecordingError):
            load_pu_recording(p, speed_unit="rpm_assumed")
        rec = load_pu_recording(p, speed_unit="rpm_assumed", speed_rel_tol=0.1)
        self.assertEqual(rec.metadata["retained_count"], 10)

    def test_invalid_parameter_values(self):
        p = self._make("N15_M07_F10_KA01_1.mat")
        for kw in [{"fs_nominal_hz": 0}, {"fs_nominal_hz": -1}, {"fs_nominal_hz": 1.5},
                   {"endpoint_rate_rel_tol": 0}, {"endpoint_rate_rel_tol": 1}, {"endpoint_rate_rel_tol": 1.5},
                   {"max_abs_residual_s": 0}, {"max_abs_residual_s": -1},
                   {"speed_rel_tol": 0}, {"speed_rel_tol": 1}, {"speed_rel_tol": 1.5}]:
            with self.assertRaises(PuRecordingError):
                load_pu_recording(p, speed_unit="rpm_assumed", **kw)

    def test_all_supported_condition_codes(self):
        cases = [("N15_M07_F10", 1500, 0.7, 1000), ("N09_M07_F10", 900, 0.7, 1000),
                 ("N15_M01_F10", 1500, 0.1, 1000), ("N15_M07_F04", 1500, 0.7, 400)]
        for code, rpm, torque, force in cases:
            t = [i/64000.0 for i in range(5)]
            x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
            y = [{"Name": "vibration_1", "Data": [1.0]*5, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [float(rpm)]*5, "XIndex": 2, "Unit": "rpm"}]
            p = self._make(f"{code}_KA01_1.mat", x_entries=x, y_entries=y)
            rec = load_pu_recording(p, speed_unit="rpm_assumed")
            self.assertEqual(rec.metadata["condition_code"], code)
            self.assertEqual(rec.metadata["nominal_rpm"], rpm)
            self.assertEqual(rec.metadata["torque_nm"], torque)
            self.assertEqual(rec.metadata["radial_force_n"], force)

    def test_reject_unknown_incomplete_condition_codes(self):
        t = [i/64000.0 for i in range(5)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*5, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*5, "XIndex": 2, "Unit": "rpm"}]
        for name in ["N99_M07_F10_KA01_1.mat", "N15_M07_KA01_1.mat", "N15_M07_F99_KA01_1.mat"]:
            p = self._make(name, x_entries=x, y_entries=y)
            with self.assertRaises(PuRecordingError):
                load_pu_recording(p, speed_unit="rpm_assumed")

    def test_raw_speed_statistics_from_original_values(self):
        t = [i/64000.0 for i in range(3)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1490.0, 1500.0, 1510.0], "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertEqual(rec.metadata["raw_speed_min"], 1490.0)
        self.assertEqual(rec.metadata["raw_speed_mean"], 1500.0)
        self.assertEqual(rec.metadata["raw_speed_max"], 1510.0)
        self.assertEqual(rec.metadata["speed_raw_min"], 1490.0)
        self.assertEqual(rec.metadata["speed_raw_mean"], 1500.0)
        self.assertEqual(rec.metadata["speed_raw_max"], 1510.0)
        self.assertEqual(rec.metadata["raw_speed_statistics"]["min"], 1490.0)

    def test_default_behavior_preserved(self):
        t = [i/64000.0 for i in range(5)]
        x = [{"Name": "TimeVib", "Data": t, "Unit": "s"}, {"Name": "TimeSpeed", "Data": t, "Unit": "s"}]
        y = [{"Name": "vibration_1", "Data": [1.0]*5, "XIndex": 1, "Unit": "m/s2"}, {"Name": "speed", "Data": [1500.0]*5, "XIndex": 2, "Unit": "rpm"}]
        p = self._make("N15_M07_F10_KA01_1.mat", x_entries=x, y_entries=y)
        rec = load_pu_recording(p, speed_unit="rpm_assumed")
        self.assertEqual(rec.fs_nominal_hz, 64000)
        self.assertEqual(rec.metadata["fs_nominal_hz"], 64000)
        self.assertEqual(rec.metadata["max_abs_residual_limit_s"], 1.0/64000.0)
