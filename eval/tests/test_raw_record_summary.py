from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.app.chatbot_service import raw_record_summary_answer
from historyAssistant.app.narration import deterministic_narration


class RawRecordSummaryTests(unittest.TestCase):
    def test_location_area_question_summarizes_distinct_areas(self) -> None:
        execution_result = {
            "result_type": "raw_records",
            "total_count": 3,
            "records": [
                {"area": "0A_K 010", "timestamp": "2024-01-31 18:48:32"},
                {"area": "0A_GANG 002", "timestamp": "2024-01-31 18:47:37"},
                {"area": "0A_K 010", "timestamp": "2024-01-31 18:42:24"},
            ],
        }

        answer = raw_record_summary_answer(
            "Which areas did patient 72 pass through on 2024-01-31?",
            execution_result,
        )

        self.assertIsNotNone(answer)
        self.assertIn("2 distinct areas", answer or "")
        self.assertIn("0A_K 010", answer or "")
        self.assertIn("0A_GANG 002", answer or "")

    def test_non_summary_lookup_returns_none(self) -> None:
        execution_result = {
            "result_type": "raw_records",
            "total_count": 1,
            "records": [{"call_id": "abc", "call_type": "Panic"}],
        }

        self.assertIsNone(raw_record_summary_answer("Show me the Panic call.", execution_result))

    def test_location_who_question_can_use_person_id(self) -> None:
        execution_result = {
            "result_type": "raw_records",
            "total_count": 2,
            "records": [
                {"personId": 266, "area": "0A_K 010"},
                {"personId": 267, "area": "0A_K 010"},
            ],
        }

        answer = raw_record_summary_answer(
            "Which patients were in area 0A_K 010 at 4:02 on 2024-01-31?",
            execution_result,
        )

        self.assertIsNotNone(answer)
        self.assertIn("2 distinct persons", answer or "")
        self.assertIn("266", answer or "")
        self.assertIn("267", answer or "")

    def test_grouped_lookup_narration_includes_counts(self) -> None:
        execution_result = {
            "result_type": "aggregated",
            "records": [
                {"personFirstName": "patient-a", "metric_value": 3},
                {"personFirstName": "patient-b", "metric_value": 1},
            ],
            "total_count": 2,
        }

        answer = deterministic_narration(
            "Show all patients between 15:00 to 16:00 in 0A_LEEFRUIMTE 002 at 2024-02-03",
            execution_result,
        )

        self.assertIn("matching patient value", answer)
        self.assertIn("patient-a (3 record(s))", answer)
        self.assertIn("patient-b (1 record(s))", answer)


if __name__ == "__main__":
    unittest.main()
