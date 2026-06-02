from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.config.loader import load_all_configs
from historyAssistant.planning import normalize_question
from historyAssistant.planning.intent_planner.slot_planner import try_build_slot_plan


def build_plan(question: str, bundle, detected_entities=None):
    normalized = normalize_question(question, bundle.vocabulary)
    result = try_build_slot_plan(
        question=question,
        normalized_question=normalized.normalized_question,
        source_id="calls",
        target_table="historyCalls",
        detected_entities=detected_entities or normalized.detected_entities,
        request_type="qa",
        chart_suppressed=False,
        preferred_chart_type=None,
        bundle=bundle,
        client=None,
        use_llm_slots=False,
    )
    assert result.used, result.to_debug_dict()
    return result.canonical_plan


class CallSimpleLookupRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_all_configs()

    def test_cancelled_calls_with_response_time_stays_raw_call_records(self) -> None:
        plan = build_plan("show me cancelled calls with response time last week", self.bundle)

        self.assertEqual(plan["target_tables"], ["historyCalls"])
        self.assertEqual(plan["metric"], "call_details")
        self.assertEqual(plan["aggregation"], "raw_records")
        self.assertEqual(plan["group_by"], [])
        self.assertIn("response_time_sec", plan["source_fields"])
        self.assertIn("is_cancelled", plan["source_fields"])
        self.assertIn("cancel_location", plan["source_fields"])

    def test_this_month_lookup_gets_date_rule(self) -> None:
        plan = build_plan(
            "I'd like to see the calls from this month.",
            self.bundle,
            detected_entities={"date_rules": ["this_month"], "tables": ["historyCalls"]},
        )

        self.assertEqual(plan["target_tables"], ["historyCalls"])
        self.assertEqual(plan["aggregation"], "raw_records")
        self.assertEqual(plan["date_rule"], "this_month")
        self.assertEqual(plan["filters"], [])

    def test_ward_zone_combined_value_does_not_add_absorbed_zone_filter(self) -> None:
        plan = build_plan(
            "Can you show me which calls received in ward 2V ZONE 2A on 2024-01-31?",
            self.bundle,
            detected_entities={
                "tables": ["historyCalls"],
                "dimensions": ["ward_name", "zone_name"],
            },
        )

        self.assertEqual(plan["target_tables"], ["historyCalls"])
        self.assertEqual(plan["aggregation"], "raw_records")
        self.assertEqual(
            plan["filters"],
            [{"field": "ward_name", "op": "=", "value": "2V ZONE 2A"}],
        )


if __name__ == "__main__":
    unittest.main()
