from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historyAssistant.config import load_source_configs
from historyAssistant.execution import canonical_plan_to_query_plan, compile_es_query
from historyAssistant.planning.core.validator import build_validation_policy, validate_and_repair_plan


class ValidatorDateFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_source_configs("calls")

    def test_invalid_empty_date_filter_is_dropped_when_date_range_is_valid(self) -> None:
        plan = {
            "request_type": "chart",
            "target_tables": ["historyCalls"],
            "metric": "total_calls",
            "aggregation": "count",
            "date_rule": "custom_range",
            "date_range": {"start_date": "2024-02-01", "end_date": "2024-02-29"},
            "filters": [
                {"field": "shift", "op": "=", "value": "night"},
                {"field": "call_date", "op": "between", "value": ["", "]]"]},
            ],
            "group_by": ["person_name"],
        }

        validation = validate_and_repair_plan(
            plan,
            policy=build_validation_policy(self.bundle),
        )
        canonical = validation["canonical_plan"]

        self.assertTrue(validation["valid"], validation)
        self.assertEqual(canonical["filters"], [{"field": "shift", "op": "=", "value": "night"}])
        self.assertIn("Dropped invalid date filter for call_date", " ".join(validation["repairs"]))

        query_plan = canonical_plan_to_query_plan(canonical, config_bundle=self.bundle)
        compiled = compile_es_query(query_plan, config_bundle=self.bundle)

        self.assertIn({"range": {"call_date": {"gte": "2024-02-01", "lte": "2024-02-29"}}}, compiled.query_body["query"]["bool"]["filter"])


if __name__ == "__main__":
    unittest.main()
