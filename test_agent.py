"""
tests/test_agent.py
--------------------
Unit tests for the parts of the project that don't need an API key or
network access: JSON extraction from model output, and dataset loading.

Run with:  pytest
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_tools
from utils import extract_answer_value


class TestAgentUtils(unittest.TestCase):
    def test_extract_answer_value_plain_json(self):
        self.assertEqual(extract_answer_value('{"state": "Assam"}'), {"state": "Assam"})

    def test_extract_answer_value_code_fenced(self):
        raw = '```json\n{"count": 42}\n```'
        self.assertEqual(extract_answer_value(raw), {"count": 42})

    def test_extract_answer_value_surrounded_by_text(self):
        raw = 'Here you go: {"total": 7} thanks'
        self.assertEqual(extract_answer_value(raw), {"total": 7})

    def test_extract_answer_value_bare_string_fallback(self):
        self.assertEqual(extract_answer_value("Assam"), "Assam")

    def test_extract_answer_value_none_input(self):
        self.assertIsNone(extract_answer_value(None))

    def test_load_tabular_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            csv_path.write_text("state,rate\nAssam,215\nKerala,30\n")
            df = data_tools.load_tabular(csv_path)
            self.assertEqual(list(df.columns), ["state", "rate"])
            self.assertEqual(len(df), 2)

    def test_preview_shape_and_head(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            csv_path.write_text("a,b\n1,2\n3,4\n")
            df = data_tools.load_tabular(csv_path)
            prev = data_tools.preview(df, n=1)
            self.assertEqual(prev["shape"], [2, 2])
            self.assertEqual(len(prev["head"]), 1)
            self.assertEqual(prev["columns"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

