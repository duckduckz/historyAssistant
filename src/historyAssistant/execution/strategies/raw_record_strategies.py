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


class RawRecordsStrategy(AggregationStrategy):
    needs_raw_values = True

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {}, True

    def resolve_limit(self, plan: QueryPlan, metric_cfg: dict, default_limit: int, max_result_rows: int) -> Optional[int]:
        metric_default_limit = metric_cfg.get("default_limit")
        effective_default = int(metric_default_limit) if metric_default_limit else default_limit
        limit_val = plan.limit if plan.limit else effective_default
        if plan.export_all and plan.limit is not None:
            return max(int(limit_val), 1)
        return min(int(limit_val), max_result_rows)

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        limit_val = self.resolve_limit(plan, metric_cfg, default_limit=max_result_rows, max_result_rows=max_result_rows)
        order_by = plan.effective_order_by() or metric_cfg.get("default_order_by") or {}
        sort_field = order_by.get("field", metric_cfg.get("_timestamp_field", "timestamp_utc"))
        if sort_field == "metric_value":
            sort_field = metric_cfg.get("_timestamp_field", "timestamp_utc")
        sort_direction = order_by.get("direction", "desc")
        body: Dict[str, Any] = {
            "size": limit_val,
            "query": query_filter,
            "sort": [{sort_field: {"order": sort_direction}}],
            "track_total_hits": True,
        }
        display_columns = plan.source_fields or metric_cfg.get("display_columns")
        if isinstance(display_columns, list) and display_columns:
            body["_source"] = display_columns
        return body

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        hits = es_response.get("hits", {}).get("hits", [])
        records = [hit.get("_source", {}) for hit in hits]
        return pd.DataFrame(records) if records else pd.DataFrame()


class FirstNumericStrategy(AggregationStrategy):
    needs_raw_values = True

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {}, True

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        field = metric_cfg.get("field", "priority")
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
            ts_field = metric_cfg.get("_timestamp_field", "timestamp_utc")
            current_aggs["metric_value"] = {
                "top_hits": {
                    "size": 1,
                    "sort": [{ts_field: {"order": "asc"}}],
                    "_source": [field],
                }
            }
            return body

        ts_field = metric_cfg.get("_timestamp_field", "timestamp_utc")
        return {
            "size": 1,
            "query": query_filter,
            "sort": [{ts_field: {"order": "asc"}}],
            "_source": [field],
        }

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        field = metric_cfg.get("field", "priority")
        if group_cols:
            def extractor(agg_data: Dict[str, Any]) -> Any:
                hits = agg_data.get("metric_value", {}).get("hits", {}).get("hits", [])
                if hits:
                    return hits[0].get("_source", {}).get(field)
                return None

            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)

        hits = es_response.get("hits", {}).get("hits", [])
        value = hits[0].get("_source", {}).get(field) if hits else None
        return pd.DataFrame([{"metric_value": value}])


__all__ = ["FirstNumericStrategy", "RawRecordsStrategy"]
