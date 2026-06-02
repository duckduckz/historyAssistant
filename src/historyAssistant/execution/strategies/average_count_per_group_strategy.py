from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import (
    AggregationStrategy,
    es_field_name,
    resolve_pre_sort_fetch_size,
    sort_and_limit_grouped_df,
)


class AvgCountPerGroupStrategy(AggregationStrategy):
    needs_raw_values = True

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {}, True

    def _base_group_col(self, metric_cfg: dict) -> str:
        base_group = metric_cfg.get("base_group", [])
        return base_group[0] if base_group else "shift"

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        count_field = metric_cfg.get("count_field", "call_id")
        base_col = self._base_group_col(metric_cfg)
        base_terms = {
            "terms": {"field": es_field_name(base_col, plan), "size": max_result_rows},
            "aggs": {"group_count": {"value_count": {"field": count_field}}},
        }

        if group_cols:
            body = {"size": 0, "query": query_filter, "aggs": {}}
            current_aggs = body["aggs"]
            has_metric_ranking = (
                isinstance(ranking_order, dict)
                and ranking_order.get("field") == "metric_value"
                and ranking_order.get("target_group") in group_cols
            )
            fetch_size = resolve_pre_sort_fetch_size(max_result_rows, pre_sort_fetch_size) if has_metric_ranking else max_result_rows
            for col in group_cols:
                agg_name = f"group_{col}"
                current_aggs[agg_name] = {
                    "terms": {"field": es_field_name(col, plan), "size": fetch_size if col == ranking_order.get("target_group") else max_result_rows},
                    "aggs": {},
                }
                current_aggs = current_aggs[agg_name]["aggs"]
            current_aggs["base_groups"] = base_terms
            current_aggs["metric_value"] = {"avg_bucket": {"buckets_path": "base_groups>group_count"}}
            return body

        return {
            "size": 0,
            "query": query_filter,
            "aggs": {
                "base_groups": base_terms,
                "metric_value": {"avg_bucket": {"buckets_path": "base_groups>group_count"}},
            },
        }

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        def extractor(agg_data: Dict[str, Any]) -> Any:
            return agg_data.get("metric_value", {}).get("value")

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


__all__ = ["AvgCountPerGroupStrategy"]
