from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import AggregationStrategy, sort_and_limit_grouped_df


class _SimpleMetricStrategy(AggregationStrategy):
    """ for count/avg/cardinality"""

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        metric_agg, _ = self.build_metric_agg(metric_cfg, metric_cfg.get("field", "call_id"))
        if group_cols:
            return self._build_grouped_body(query_filter, metric_agg, group_cols, max_result_rows, ranking_order, plan, pre_sort_fetch_size)
        return {"size": 0, "query": query_filter, "aggs": {"metric_value": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        def extractor(agg_data: Dict[str, Any]) -> Any:
            metric_val = agg_data.get("metric_value", {})
            if "value" in metric_val:
                return metric_val["value"]
            if "doc_count" in agg_data and "metric_value" not in agg_data:
                return agg_data.get("doc_count")
            return metric_val.get("value")

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


class CountStrategy(_SimpleMetricStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {"value_count": {"field": field}}, False


class AvgStrategy(_SimpleMetricStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {"avg": {"field": field}}, False


class CardinalityStrategy(_SimpleMetricStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {"cardinality": {"field": field}}, False


__all__ = ["AvgStrategy", "CardinalityStrategy", "CountStrategy"]
