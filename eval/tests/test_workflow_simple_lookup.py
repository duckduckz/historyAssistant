from __future__ import annotations

import pathlib
import sys
import unittest
from importlib import import_module

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.app.source_plan_repairs import repair_workflow_presence_lookup
from historyAssistant.config import load_source_configs
from historyAssistant.planning import normalize_question
from historyAssistant.planning.intent_planner.slot_planner import try_build_slot_plan


validator_mod = import_module("historyAssistant.planning.core.validator")


def build_plan(question: str, bundle, detected_entities=None):
    normalized = normalize_question(question, bundle.vocabulary)
    result = try_build_slot_plan(
        question=question,
        normalized_question=normalized.normalized_question,
        source_id="workflows",
        target_table="historyWorkflows",
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


class WorkflowSimpleLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_source_configs("workflows")

    def test_presence_records_for_ward_keeps_ward_filter_and_raw_records(self) -> None:
        question = "Show me presence records for ward 2A on 2014-08-15"

        normalized = normalize_question(question, self.bundle.vocabulary)
        plan = build_plan(question, self.bundle, detected_entities=normalized.detected_entities)
        plan = repair_workflow_presence_lookup(plan, question=question, bundle=self.bundle)
        validation = validator_mod.validate_and_repair_plan(
            plan,
            policy=validator_mod.build_validation_policy(self.bundle),
        )
        canonical = validation["canonical_plan"]

        self.assertEqual(canonical["target_tables"], ["historyWorkflows"])
        self.assertEqual(canonical["metric"], "presence_details")
        self.assertEqual(canonical["aggregation"], "raw_records")
        self.assertEqual(canonical["group_by"], [])
        self.assertCountEqual(
            canonical["filters"],
            [
                {"field": "ward_name", "op": "=", "value": "2V ZONE 2A"},
                {"field": "workflow_name", "op": "=", "value": "Presence"},
            ],
        )
        self.assertIn("room_name", canonical["source_fields"])
        self.assertIn("ward_name", canonical["source_fields"])

    def test_who_presence_normal_runtime_stays_raw_presence_details(self) -> None:
        question = "can you show me name who presence in 2024-02-01"
        plan = build_plan(question, self.bundle)

        self.assertEqual(plan["target_tables"], ["historyWorkflows"])
        self.assertEqual(plan["metric"], "presence_details")
        self.assertEqual(plan["aggregation"], "raw_records")
        self.assertEqual(plan["group_by"], [])
        self.assertEqual(plan["date_range"], {"start_date": "2024-02-01", "end_date": "2024-02-01"})

    def test_who_was_present_repair_preserves_distinct_person_grouping(self) -> None:
        question = "who was present on 2024-02-01"
        plan = build_plan(question, self.bundle)
        plan = repair_workflow_presence_lookup(plan, question=question, bundle=self.bundle)

        self.assertEqual(plan["target_tables"], ["historyWorkflows"])
        self.assertEqual(plan["metric"], "presence_event_count")
        self.assertEqual(plan["aggregation"], "count_threshold")
        self.assertEqual(plan["group_by"], ["initiator_person"])
        self.assertIn({"field": "workflow_name", "op": "=", "value": "Presence"}, plan["filters"])

if __name__ == "__main__":
    unittest.main()
