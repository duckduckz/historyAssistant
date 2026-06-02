from __future__ import annotations

import pathlib
import sys
import unittest
from importlib import import_module

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.app.source_plan_repairs import repair_location_person_lookup
from historyAssistant.config import load_source_configs
from historyAssistant.execution import canonical_plan_to_query_plan, compile_es_query
from historyAssistant.planning import normalize_question
from historyAssistant.planning.intent_planner.raw_detail_transform import build_raw_detail_plan
from historyAssistant.planning.intent_planner.slot_planner import try_build_slot_plan


validator_mod = import_module("historyAssistant.planning.core.validator")


def build_plan(question: str, bundle, detected_entities=None):
    normalized = normalize_question(question, bundle.vocabulary)
    result = try_build_slot_plan(
        question=question,
        normalized_question=normalized.normalized_question,
        source_id="locations",
        target_table="historyLocations",
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


class LocationSimpleLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_source_configs("locations")

    def test_movement_records_stays_raw_records_with_location_filters(self) -> None:
        question = (
            "Show me the movement records for person_name "
            "72a9fdd6c18461162a37f0c8f30aa4fd1afdb9860439adb304181ab79c4e16e2 "
            "on 2024-01-31 around zone_name ZONEKSWA"
        )

        normalized = normalize_question(question, self.bundle.vocabulary)
        plan = build_plan(question, self.bundle, detected_entities=normalized.detected_entities)
        plan = repair_location_person_lookup(
            plan,
            question=normalized.normalized_question,
            bundle=self.bundle,
        )
        validation = validator_mod.validate_and_repair_plan(
            plan,
            policy=validator_mod.build_validation_policy(self.bundle),
        )
        canonical = validation["canonical_plan"]

        self.assertEqual(canonical["target_tables"], ["historyLocations"])
        self.assertEqual(canonical["metric"], "movement_details")
        self.assertEqual(canonical["aggregation"], "raw_records")
        self.assertEqual(canonical["group_by"], [])
        self.assertCountEqual(
            canonical["filters"],
            [
                {
                    "field": "personFirstName",
                    "op": "=",
                    "value": "72a9fdd6c18461162a37f0c8f30aa4fd1afdb9860439adb304181ab79c4e16e2",
                },
                {"field": "zoneName", "op": "=", "value": "ZONEKSWA"},
            ],
        )

    def test_area_patient_lookup_becomes_distinct_patient_values(self) -> None:
        question = "Which patients were in area 0A_K 010 at 4:02 on 2024-01-31?"

        normalized = normalize_question(question, self.bundle.vocabulary)
        plan = build_plan(question, self.bundle, detected_entities=normalized.detected_entities)

        self.assertEqual(plan["aggregation"], "cardinality")
        self.assertEqual(plan["group_by"], ["personFirstName"])
        self.assertIn({"field": "area", "op": "=", "value": "0A_K 010"}, plan["filters"])

    def test_location_person_filter_uses_keyword_field_directly(self) -> None:
        plan = {
            "request_type": "qa",
            "target_tables": ["historyLocations"],
            "metric": "movement_details",
            "aggregation": "raw_records",
            "filters": [
                {
                    "field": "personFirstName",
                    "op": "=",
                    "value": "72a9fdd6c18461162a37f0c8f30aa4fd1afdb9860439adb304181ab79c4e16e2",
                }
            ],
            "group_by": [],
            "order_by": None,
            "limit": None,
            "scope_policy": "raw",
            "outlier_policy": "none",
            "source_fields": ["personFirstName", "timestamp"],
            "date_rule": "custom_range",
            "date_range": {"start_date": "2024-01-31", "end_date": "2024-01-31"},
            "time_range": {"start_time": "08:00", "end_time": "12:00"},
            "normalized_sub_request": "test",
            "sub_request_id": 1,
            "classification_confidence": 1.0,
        }

        query_plan = canonical_plan_to_query_plan(plan, config_bundle=self.bundle)
        compiled = compile_es_query(query_plan, config_bundle=self.bundle)
        filters = compiled.query_body["query"]["bool"]["filter"]

        self.assertIn(
            {
                "term": {
                    "personFirstName": "72a9fdd6c18461162a37f0c8f30aa4fd1afdb9860439adb304181ab79c4e16e2"
                }
            },
            filters,
        )
        self.assertNotIn("personFirstName.keyword", str(compiled.query_body))

    def test_raw_detail_transform_preserves_location_filters_and_date(self) -> None:
        prior = {
            "target_tables": ["historyLocations"],
            "metric": "movement_event_count",
            "aggregation": "cardinality",
            "filters": [
                {
                    "field": "personFirstName",
                    "op": "=",
                    "value": "59104929c3db63ef49fc86f881fa9ce4623e769ec2ee5118ed87f0866cdbe5af",
                }
            ],
            "date_rule": "custom_range",
            "date_range": {"start_date": "2024-02-01", "end_date": "2024-02-01"},
            "group_by": ["movement_detail"],
        }

        detail = build_raw_detail_plan(prior, bundle=self.bundle)

        self.assertIsNotNone(detail)
        self.assertEqual(detail["metric"], "movement_details")
        self.assertEqual(detail["aggregation"], "raw_records")
        self.assertEqual(detail["filters"], prior["filters"])
        self.assertEqual(detail["date_range"], prior["date_range"])
        self.assertEqual(detail["group_by"], [])
        self.assertIn("timestamp", detail["source_fields"])
        self.assertIn("personFirstName", detail["source_fields"])


if __name__ == "__main__":
    unittest.main()
