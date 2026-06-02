from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from historyAssistant.execution.aggregation_strategies import (
    AvgCountPerGroupStrategy,
    PercentileStrategy,
    RateThresholdStrategy,
)
from historyAssistant.execution.es_builder import compile_es_query
from historyAssistant.execution.query_models import QueryPlan


class RankingOverfetchTests(unittest.TestCase):
    def test_percentile_strategy_overfetches_target_rank_group(self) -> None:
        strategy = PercentileStrategy(default_percent=50.0)
        plan = QueryPlan(
            metric="median_response_time",
            aggregation="median",
            group_by=["ward_name"],
            ranking_order={"field": "metric_value", "direction": "desc", "target_group": "ward_name"},
            limit=10,
        )
        body = strategy.build_query_body(
            query_filter={"match_all": {}},
            metric_cfg={"field": "response_time_sec", "percentile": 0.5},
            group_cols=["ward_name"],
            max_result_rows=50,
            ranking_order=plan.ranking_order or {},
            plan=plan,
            pre_sort_fetch_size=500,
        )
        self.assertEqual(body["aggs"]["group_ward_name"]["terms"]["size"], 500)

    def test_avg_count_per_group_overfetches_target_rank_group(self) -> None:
        strategy = AvgCountPerGroupStrategy()
        plan = QueryPlan(
            metric="avg_calls_per_shift",
            aggregation="avg_count_per_group",
            group_by=["ward_name"],
            ranking_order={"field": "metric_value", "direction": "desc", "target_group": "ward_name"},
            limit=10,
        )
        body = strategy.build_query_body(
            query_filter={"match_all": {}},
            metric_cfg={"count_field": "call_id", "base_group": ["shift"]},
            group_cols=["ward_name"],
            max_result_rows=50,
            ranking_order=plan.ranking_order or {},
            plan=plan,
            pre_sort_fetch_size=500,
        )
        self.assertEqual(body["aggs"]["group_ward_name"]["terms"]["size"], 500)

    def test_rate_threshold_grouped_body_overfetches_when_post_sorting(self) -> None:
        strategy = RateThresholdStrategy()
        plan = QueryPlan(
            metric="response_within_2_min_rate",
            aggregation="rate_threshold",
            group_by=["ward_name"],
            ranking_order={"field": "metric_value", "direction": "desc", "target_group": "ward_name"},
            limit=10,
        )
        body = strategy.build_query_body(
            query_filter={"match_all": {}},
            metric_cfg={"field": "response_time_sec", "threshold": 120, "comparator": "<="},
            group_cols=["ward_name"],
            max_result_rows=50,
            ranking_order=plan.ranking_order or {},
            plan=plan,
            pre_sort_fetch_size=500,
        )
        self.assertEqual(body["aggs"]["group_ward_name"]["terms"]["size"], 500)

    def test_compile_es_query_exposes_overfetch_debug_for_ranked_percentile(self) -> None:
        plan = QueryPlan(
            metric="median_response_time",
            aggregation="median",
            group_by=["ward_name"],
            ranking_order={"field": "metric_value", "direction": "desc", "target_group": "ward_name"},
            limit=10,
        )
        compiled = compile_es_query(plan)
        self.assertIsNotNone(compiled.overfetch_debug)
        self.assertTrue(compiled.overfetch_debug["activated"])
        self.assertEqual(compiled.overfetch_debug["pre_sort_fetch_multiplier"], 10)
        self.assertEqual(compiled.overfetch_debug["pre_sort_fetch_size"], 500)


if __name__ == "__main__":
    unittest.main()
