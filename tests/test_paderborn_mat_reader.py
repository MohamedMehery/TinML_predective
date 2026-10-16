"""Synthetic-fixture tests for the isolated Paderborn MAT reader.

These tests validate parser behavior against a synthetic fixture schema.
They do NOT prove compatibility with actual KA01 files.
"""

import math
import os
import sys
import tempfile
import unittest

# Make the package importable when running from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import numpy as np
    from scipy.io import savemat, loadmat
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False

from tools.paderborn_mat_reader.reader import (
    MatReaderError,
    read_mat,
    inspect_mat,
)


def _skip_if_no_scipy(test):
    if not SCIPY_AVAILABLE:
        return unittest.skip("scipy/numpy not available")(test)
    return test


def make_mat(path, root_name="N15_M07_F10_KA01_1",
             x_entries=None, y_entries=None,
             root_fields=("Info", "X", "Y", "Description")):
    """Build a synthetic MATLAB 5.0 file. x_entries/y_entries are lists of dicts."""
    if x_entries is None:
        x_entries = [
            {"Name": "TimeSpeed", "Data": [0.0, 0.25, 0.5], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
    if y_entries is None:
        y_entries = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0],
             "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0],
             "XIndex": 1, "Unit": "rpm"},
        ]

    # Build X cell array of structs
    X = np.empty((1, len(x_entries)), dtype=object)
    for i, xe in enumerate(x_entries):
        names = list(xe.keys())
        # ensure field order
        dt = [(k, object) for k in ("Name", "Data", "Unit") if k in xe or True]
        # build with all standard fields
        st = np.zeros(1, dtype=[("Name", object), ("Data", object), ("Unit", object)])
        st[0]["Name"] = np.array([xe.get("Name", "")])
        st[0]["Data"] = np.asarray(xe.get("Data", []), dtype=float).ravel()
        st[0]["Unit"] = np.array([xe.get("Unit", "")]) if "Unit" in xe else np.array([""])
        # handle missing unit by empty string -> UNKNOWN
        if "Unit" not in xe:
            st[0]["Unit"] = np.array([""])
        X[0, i] = st

    Y = np.empty((1, len(y_entries)), dtype=object)
    for i, ye in enumerate(y_entries):
        st = np.zeros(1, dtype=[("Name", object), ("Data", object),
                                ("XIndex", object), ("Unit", object)])
        st[0]["Name"] = np.array([ye.get("Name", "")])
        st[0]["Data"] = np.asarray(ye.get("Data", []), dtype=float).ravel()
        st[0]["XIndex"] = np.array([[ye.get("XIndex", 0)]])
        if "Unit" in ye:
            st[0]["Unit"] = np.array([ye["Unit"]])
        else:
            st[0]["Unit"] = np.array([""])
        Y[0, i] = st

    fields = []
    if "Info" in root_fields:
        fields.append(("Info", object))
    if "X" in root_fields:
        fields.append(("X", object))
    if "Y" in root_fields:
        fields.append(("Y", object))
    if "Description" in root_fields:
        fields.append(("Description", object))
    root = np.zeros(1, dtype=fields)
    if "Info" in root_fields:
        root[0]["Info"] = np.array([["info"]], dtype=object)
    if "X" in root_fields:
        root[0]["X"] = X
    if "Y" in root_fields:
        root[0]["Y"] = Y
    if "Description" in root_fields:
        root[0]["Description"] = np.array(["desc"])

    savemat(path, {root_name: root}, do_compression=True)
    return path


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy/numpy not available")
class TestReaderBasic(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def _path(self, name="t.mat"):
        return os.path.join(self.td, name)

    def test_reordered_channels(self):
        # speed first, vibration second
        y = [
            {"Name": "speed", "Data": [1.0, 2.0, 3.0], "XIndex": 1, "Unit": "rpm"},
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0],
             "XIndex": 2, "Unit": "m/s2"},
        ]
        p = make_mat(self._path(), y_entries=y)
        res = read_mat(p)
        self.assertIn("vibration_1", res["channels"])
        self.assertIn("speed", res["channels"])
        self.assertEqual(res["channels"]["vibration_1"].values.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(res["channels"]["speed"].values.tolist(), [1.0, 2.0, 3.0])

    def test_xindex_one_based(self):
        # XIndex=2 -> X[1]
        p = make_mat(self._path())
        res = read_mat(p)
        ch = res["channels"]["vibration_1"]
        self.assertEqual(ch.x_index, 2)
        # time should come from second X entry
        self.assertEqual(ch.timestamps.tolist(), [0.0, 0.1, 0.2, 0.3])

    def test_different_counts_and_ranges(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, 0.5, 1.0], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.25, 0.5, 0.75, 1.0], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0]*5, "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [1.0]*3, "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        res = read_mat(p)
        self.assertEqual(len(res["channels"]["vibration_1"].values), 5)
        self.assertEqual(len(res["channels"]["speed"].values), 3)
        rep = inspect_mat(p)
        sup = rep["common_time_support"]
        self.assertTrue(sup["valid"])
        self.assertAlmostEqual(sup["support_start"], 0.0)
        self.assertAlmostEqual(sup["support_end"], 1.0)

    def test_missing_unit_unknown(self):
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2},  # no Unit
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), y_entries=y)
        res = read_mat(p)
        self.assertEqual(res["channels"]["vibration_1"].unit, "UNKNOWN")

    def test_missing_channel_raises(self):
        y = [
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_nan_timestamps_raises(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, float("nan"), 1.0], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_inf_timestamps_raises(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, float("inf"), 1.0], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_nonincreasing_timestamps_raises(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, 0.5, 0.5], "Unit": "s"},  # duplicate
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_decreasing_timestamps_raises(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, 0.5, 0.3], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_mismatched_lengths_raises(self):
        x = [
            {"Name": "TimeSpeed", "Data": [0.0, 0.5, 1.0], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0], "XIndex": 1, "Unit": "rpm"},  # mismatch
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_invalid_xindex_raises(self):
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 5, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_xindex_zero_raises(self):
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 0, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_missing_root_x_raises(self):
        p = make_mat(self._path(), root_fields=("Info", "Y", "Description"))
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_missing_root_y_raises(self):
        p = make_mat(self._path(), root_fields=("Info", "X", "Description"))
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_malformed_x_entry_raises(self):
        # X entry without Data
        x = [
            {"Name": "TimeSpeed"},  # no Data
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        with self.assertRaises(MatReaderError):
            read_mat(p)

    def test_inspect_report_fields(self):
        p = make_mat(self._path())
        rep = inspect_mat(p)
        self.assertIn("vibration_1", rep["channels"])
        ch = rep["channels"]["vibration_1"]
        self.assertEqual(ch["unit"], "m/s2")
        self.assertEqual(ch["n_samples"], 4)
        self.assertEqual(ch["nonpositive_diff_count"], 0)
        self.assertEqual(ch["validation_status"], "OK")
        # conditional rate present
        self.assertTrue(isinstance(ch["conditional_sample_rate_hz"], float))

    def test_inspect_missing_unit(self):
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), y_entries=y)
        rep = inspect_mat(p)
        self.assertEqual(rep["channels"]["vibration_1"]["unit"], "UNKNOWN")

    def test_no_overlap_support_invalid(self):
        x = [
            {"Name": "TimeSpeed", "Data": [10.0, 10.5, 11.0], "Unit": "s"},
            {"Name": "TimeVib", "Data": [0.0, 0.1, 0.2, 0.3], "Unit": "s"},
        ]
        y = [
            {"Name": "vibration_1", "Data": [1.0, 2.0, 3.0, 4.0], "XIndex": 2, "Unit": "m/s2"},
            {"Name": "speed", "Data": [10.0, 20.0, 30.0], "XIndex": 1, "Unit": "rpm"},
        ]
        p = make_mat(self._path(), x_entries=x, y_entries=y)
        rep = inspect_mat(p)
        self.assertFalse(rep["common_time_support"]["valid"])

    def test_values_preserved(self):
        p = make_mat(self._path())
        res = read_mat(p)
        self.assertEqual(res["channels"]["speed"].values.tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(res["channels"]["speed"].timestamps.tolist(), [0.0, 0.25, 0.5])

    def test_multiple_top_level_vars_raises(self):
        import numpy as np
        from scipy.io import savemat
        x = np.empty((1, 1), dtype=object)
        xs = np.zeros(1, dtype=[("Name", object), ("Data", object), ("Unit", object)])
        xs[0]["Name"] = np.array(["T"])
        xs[0]["Data"] = np.array([0.0, 1.0])
        xs[0]["Unit"] = np.array(["s"])
        x[0, 0] = xs
        y = np.empty((1, 1), dtype=object)
        ys = np.zeros(1, dtype=[("Name", object), ("Data", object),
                                ("XIndex", object), ("Unit", object)])
        ys[0]["Name"] = np.array(["vibration_1"])
        ys[0]["Data"] = np.array([1.0, 2.0])
        ys[0]["XIndex"] = np.array([[1]])
        ys[0]["Unit"] = np.array(["m/s2"])
        y[0, 0] = ys
        root = np.zeros(1, dtype=[("X", object), ("Y", object)])
        root[0]["X"] = x
        root[0]["Y"] = y
        savemat(self._path(), {"a": root, "b": root})
        with self.assertRaises(MatReaderError):
            read_mat(self._path())


if __name__ == "__main__":
    unittest.main()
