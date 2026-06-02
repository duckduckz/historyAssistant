from __future__ import annotations

import pathlib
import sys
import unittest
from importlib import import_module

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


extract_dates = import_module("historyAssistant.planning.extraction.temporal_parser").extract_dates


class TemporalParserMonthTypoTests(unittest.TestCase):
    def test_fabruary_year_resolves_to_february_range(self) -> None:
        date_rule, date_range, date_params = extract_dates("show me movement in fabruary 2024", {})

        self.assertEqual(date_rule, "custom_range")
        self.assertEqual(date_range, {"start_date": "2024-02-01", "end_date": "2024-02-29"})
        self.assertIsNone(date_params)


if __name__ == "__main__":
    unittest.main()
