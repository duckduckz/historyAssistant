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


class PercentileStrategy(AggregationStrategy):
    needs_raw_values = True

    def __init__(self, default_percent: Optional[float] = None) -> None:
        self.default_percent = default_percent

    def _percent(self, metric_cfg: dict) -> float:
        if self.default_percent is not None:
            return float(self.default_percent)
        return float(metric_cfg.get("percentile", 0.5)) * 100.0

    def _percent_key(self, metric_cfg: dict) -> str:
        return str(self._percent(metric_cfg))

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {"percentiles": {"field": field, "percents": [self._percent(metric_cfg)]}}, True

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        metric_agg, _ = self.build_metric_agg(metric_cfg, metric_cfg.get("field", "call_id"))
        if group_cols:
            has_metric_ranking = (
                isinstance(ranking_order, dict)
                and ranking_order.get("field") == "metric_value"
                and ranking_order.get("target_group") in group_cols
            )
            fetch_size = resolve_pre_sort_fetch_size(max_result_rows, pre_sort_fetch_size) if has_metric_ranking else max_result_rows
            body = {"size": 0, "query": query_filter, "aggs": {}}
            current_aggs = body["aggs"]
            for col in group_cols:
                agg_name = f"group_{col}"
                col_size = fetch_size if col == ranking_order.get("target_group") else max_result_rows
                current_aggs[agg_name] = {
                    "terms": {"field": es_field_name(col, plan), "size": col_size},
                    "aggs": {},
                }
                current_aggs = current_aggs[agg_name]["aggs"]
            current_aggs["percentile_values"] = metric_agg
            return body
        return {"size": 0, "query": query_filter, "aggs": {"percentile_values": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        percent_key = self._percent_key(metric_cfg)

        def extractor(agg_data: Dict[str, Any]) -> Any:
            return agg_data.get("percentile_values", {}).get("values", {}).get(percent_key)

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return pd.DataFrame([{"metric_value": extractor(es_response.get("aggregations", {}))}])


__all__ = ["PercentileStrategy"]
