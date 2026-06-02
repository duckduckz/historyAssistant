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


class NumericSummaryStrategy(AggregationStrategy):
    needs_raw_values = True

    _ES_AGG_MAP = {
        "avg": "avg",
        "min": "min",
        "max": "max",
        "sum": "sum",
        "median": "percentiles",
        "percentile": "percentiles",
    }

    def _numeric_aggregation(self, metric_cfg: dict) -> str:
        return str(metric_cfg.get("numeric_aggregation") or "avg").strip().lower()

    def _percent(self, metric_cfg: dict) -> float:
        numeric_aggregation = self._numeric_aggregation(metric_cfg)
        if numeric_aggregation == "median":
            return 50.0
        percentile = metric_cfg.get("percentile", 0.9)
        return float(percentile) * 100.0 if float(percentile) <= 1 else float(percentile)

    def _percent_key(self, metric_cfg: dict) -> str:
        return str(self._percent(metric_cfg))

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        numeric_aggregation = self._numeric_aggregation(metric_cfg)
        es_aggregation = self._ES_AGG_MAP.get(numeric_aggregation)
        if not es_aggregation:
            raise ValueError(f"Unsupported numeric summary aggregation: {numeric_aggregation}")
        if es_aggregation == "percentiles":
            return {"percentiles": {"field": field, "percents": [self._percent(metric_cfg)]}}, True
        return {es_aggregation: {"field": field}}, False

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        field = metric_cfg.get("field")
        if not field:
            raise ValueError("numeric_summary requires metric_cfg.field")
        metric_agg, _ = self.build_metric_agg(metric_cfg, field)
        if group_cols:
            has_metric_ranking = (
                isinstance(ranking_order, dict)
                and ranking_order.get("field") == "metric_value"
                and ranking_order.get("target_group") in group_cols
                and self._numeric_aggregation(metric_cfg) in {"median", "percentile"}
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
            current_aggs["metric_value"] = metric_agg
            return body
        return {"size": 0, "query": query_filter, "aggs": {"metric_value": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        numeric_aggregation = self._numeric_aggregation(metric_cfg)

        def extractor(agg_data: Dict[str, Any]) -> Any:
            metric_val = agg_data.get("metric_value", {})
            if numeric_aggregation in {"median", "percentile"}:
                return metric_val.get("values", {}).get(self._percent_key(metric_cfg))
            return metric_val.get("value")

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


__all__ = ["NumericSummaryStrategy"]
