"""Tests for the owner-approved, source-traceable PU bearing class map."""

import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "02_datasets/manifests/paderborn_bearing_class_map.csv"
DOC_PATH = ROOT / "02_datasets/manifests/paderborn_bearing_class_map.md"
FIELDNAMES = [
    "bearing_id",
    "class_name",
    "label",
    "damage_origin",
    "source_table",
    "source_component",
    "source_combination",
    "extent_level",
    "stage1_status",
]

# Explicit owner-provided mapping, checked against Lessmeier et al. (2016),
# Tables 4, 5, and 7. Labels are not derived from bearing-ID prefixes.
EXPECTED_ROWS = [
    ("K001", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("K002", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("K003", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("K004", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("K005", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("K006", "Healthy", "0", "healthy", "Table 7", "n/a", "n/a", "n/a", "include"),
    ("KA01", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L1", "include"),
    ("KA03", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L2", "include"),
    ("KA05", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L1", "include"),
    ("KA06", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L2", "include"),
    ("KA07", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L1", "include"),
    ("KA08", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L2", "include"),
    ("KA09", "Outer_Race", "1", "artificial", "Table 4", "OR", "single point", "L2", "include"),
    ("KI01", "Inner_Race", "2", "artificial", "Table 4", "IR", "single point", "L1", "include"),
    ("KI03", "Inner_Race", "2", "artificial", "Table 4", "IR", "single point", "L1", "include"),
    ("KI05", "Inner_Race", "2", "artificial", "Table 4", "IR", "single point", "L1", "include"),
    ("KI07", "Inner_Race", "2", "artificial", "Table 4", "IR", "single point", "L2", "include"),
    ("KI08", "Inner_Race", "2", "artificial", "Table 4", "IR", "single point", "L2", "include"),
    ("KA04", "Outer_Race", "1", "real", "Table 5", "OR", "S", "L1", "include"),
    ("KA15", "Outer_Race", "1", "real", "Table 5", "OR", "S", "L1", "include"),
    ("KA16", "Outer_Race", "1", "real", "Table 5", "OR", "R", "L2", "include"),
    ("KA22", "Outer_Race", "1", "real", "Table 5", "OR", "S", "L1", "include"),
    ("KA30", "Outer_Race", "1", "real", "Table 5", "OR", "R", "L1", "include"),
    ("KI04", "Inner_Race", "2", "real", "Table 5", "IR", "M", "L1", "include"),
    ("KI14", "Inner_Race", "2", "real", "Table 5", "IR", "M", "L1", "include"),
    ("KI16", "Inner_Race", "2", "real", "Table 5", "IR", "S", "L3", "include"),
    ("KI17", "Inner_Race", "2", "real", "Table 5", "IR", "R", "L1", "include"),
    ("KI18", "Inner_Race", "2", "real", "Table 5", "IR", "S", "L2", "include"),
    ("KI21", "Inner_Race", "2", "real", "Table 5", "IR", "S", "L1", "include"),
    ("KB23", "Combined", "-1", "real", "Table 5", "IR (+OR)", "M", "L2", "exclude"),
    ("KB24", "Combined", "-1", "real", "Table 5", "IR (+OR)", "M", "L3", "exclude"),
    ("KB27", "Combined", "-1", "real", "Table 5", "OR + IR", "M", "L1", "exclude"),
]


class PaderbornBearingClassMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            cls.fieldnames = reader.fieldnames
            cls.rows = list(reader)
        cls.rows_by_id = {row["bearing_id"]: row for row in cls.rows}

    def test_exact_schema_and_complete_owner_mapping(self):
        self.assertEqual(self.fieldnames, FIELDNAMES)
        self.assertEqual(len(self.rows), 32)
        self.assertEqual(len(self.rows_by_id), 32)
        expected = [dict(zip(FIELDNAMES, values)) for values in EXPECTED_ROWS]
        self.assertEqual(self.rows, expected)

    def test_label_contract_and_explicit_stage1_exclusions(self):
        self.assertEqual(
            Counter(row["label"] for row in self.rows),
            Counter({"0": 6, "1": 12, "2": 11, "-1": 3}),
        )
        labels_by_class = {
            "Healthy": "0",
            "Outer_Race": "1",
            "Inner_Race": "2",
            "Combined": "-1",
        }
        for row in self.rows:
            self.assertEqual(row["label"], labels_by_class[row["class_name"]])
            expected_status = "exclude" if row["label"] == "-1" else "include"
            self.assertEqual(row["stage1_status"], expected_status)

        excluded_ids = {
            row["bearing_id"] for row in self.rows if row["stage1_status"] == "exclude"
        }
        self.assertEqual(excluded_ids, {"KB23", "KB24", "KB27"})
        self.assertFalse(
            any(
                row["class_name"] == "Combined" and row["stage1_status"] == "include"
                for row in self.rows
            )
        )

    def test_source_combination_and_extent_conventions_are_preserved(self):
        table4 = [row for row in self.rows if row["source_table"] == "Table 4"]
        self.assertEqual(len(table4), 12)
        self.assertEqual({row["source_combination"] for row in table4}, {"single point"})

        table5 = [row for row in self.rows if row["source_table"] == "Table 5"]
        self.assertEqual(len(table5), 14)
        self.assertTrue(
            all(row["source_combination"] in {"S", "R", "M"} for row in table5)
        )
        self.assertEqual(
            {
                bearing_id: (
                    self.rows_by_id[bearing_id]["source_component"],
                    self.rows_by_id[bearing_id]["source_combination"],
                    self.rows_by_id[bearing_id]["extent_level"],
                )
                for bearing_id in ("KB23", "KB24", "KB27")
            },
            {
                "KB23": ("IR (+OR)", "M", "L2"),
                "KB24": ("IR (+OR)", "M", "L3"),
                "KB27": ("OR + IR", "M", "L1"),
            },
        )
        self.assertEqual(
            {row["extent_level"] for row in self.rows if row["source_table"] in {"Table 4", "Table 5"}},
            {"L1", "L2", "L3"},
        )

        healthy = [row for row in self.rows if row["source_table"] == "Table 7"]
        self.assertEqual(len(healthy), 6)
        self.assertTrue(
            all(
                row[field] == "n/a"
                for row in healthy
                for field in ("source_component", "source_combination", "extent_level")
            )
        )

    def test_documentation_records_source_provenance_and_license(self):
        documentation = DOC_PATH.read_text(encoding="utf-8")
        required_text = (
            "Lessmeier 2016.pdf",
            "359ac7f068c65b87a396f86da22abe50a82140ad",
            "Table 4",
            "Table 5",
            "Table 7",
            "CC BY-NC 4.0",
            "10.36001/phme.2016.v3i1.1577",
            "Combined",
            "single point",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, documentation)


if __name__ == "__main__":
    unittest.main()
