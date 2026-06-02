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


def build_plan(question: str, bundle):
    normalized = normalize_question(question, bundle.vocabulary)
    result = try_build_slot_plan(
        question=question,
        normalized_question=normalized.normalized_question,
        source_id="calls",
        target_table="historyCalls",
        detected_entities=normalized.detected_entities,
        request_type="qa",
        chart_suppressed=False,
        preferred_chart_type=None,
        bundle=bundle,
        client=None,
        use_llm_slots=False,
    )
    assert result.used, result.to_debug_dict()
    return result.canonical_plan


class CallUrgentFilterRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_all_configs()

    def test_urgent_scalar_uses_priority_filter(self) -> None:
        plan = build_plan("how many urgent calls in February 2024", self.bundle)

        self.assertEqual(plan["metric"], "total_calls")
        self.assertEqual(plan["group_by"], [])
        self.assertIn({"field": "priority", "op": "in", "value": [4, 5]}, plan["filters"])

    def test_urgent_by_ward_keeps_group_by_and_filter(self) -> None:
        plan = build_plan("urgent calls by ward in February 2024", self.bundle)

        self.assertEqual(plan["metric"], "total_calls")
        self.assertEqual(plan["group_by"], ["ward_name"])
        self.assertIn({"field": "priority", "op": "in", "value": [4, 5]}, plan["filters"])
        self.assertNotIn({"field": "ward_name", "op": "=", "value": "in February 2024"}, plan["filters"])


if __name__ == "__main__":
    unittest.main()
