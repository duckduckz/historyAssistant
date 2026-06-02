from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import AggregationStrategy, sort_and_limit_grouped_df


class RateThresholdStrategy(AggregationStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        comp = metric_cfg.get("comparator", "<=")
        threshold = metric_cfg.get("threshold", 0)
        comp_op = {"<=": "<=", ">=": ">=", "<": "<", ">": ">", "=": "=="}.get(comp, "<=")
        return {
            "scripted_metric": {
                "init_script": "state.total = 0; state.matched = 0",
                "map_script": (
                    f"if (doc['{field}'].size() > 0 && doc['{field}'].value != null) {{"
                    f"  state.total++;"
                    f"  if (doc['{field}'].value {comp_op} {threshold}) {{ state.matched++; }}"
                    f"}}"
                ),
                "combine_script": "return ['total': state.total, 'matched': state.matched]",
                "reduce_script": (
                    "double total = 0; double matched = 0; "
                    "for (s in states) { total += s.total; matched += s.matched; } "
                    "return total > 0 ? (matched / total) * 100.0 : 0"
                ),
            }
        }, False

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        metric_agg, _ = self.build_metric_agg(metric_cfg, metric_cfg.get("field", "call_id"))
        if group_cols:
            return self._build_grouped_body(query_filter, metric_agg, group_cols, max_result_rows, ranking_order, plan, pre_sort_fetch_size)
        return {"size": 0, "query": query_filter, "aggs": {"metric_value": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        def extractor(agg_data: Dict[str, Any]) -> Any:
            metric_val = agg_data.get("metric_value")
            if isinstance(metric_val, dict) and "value" in metric_val:
                return metric_val["value"]
            return metric_val

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


class CountThresholdStrategy(AggregationStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        comp = metric_cfg.get("comparator", "<=")
        threshold = metric_cfg.get("threshold", 0)
        count_field = metric_cfg.get("count_field", "call_id")
        if comp == "=":
            threshold_filter = {"term": {field: threshold}}
        elif comp == "!=":
            threshold_filter = {"bool": {"must_not": [{"term": {field: threshold}}]}}
        else:
            op_map = {"<=": "lte", ">=": "gte", "<": "lt", ">": "gt"}
            range_op = op_map.get(comp)
            if not range_op:
                raise ValueError(f"Unsupported comparator for count_threshold: {comp}")
            threshold_filter = {"range": {field: {range_op: threshold}}}
        return {
            "filter": threshold_filter,
            "aggs": {"count": {"value_count": {"field": count_field}}},
        }, False

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        metric_agg, _ = self.build_metric_agg(metric_cfg, metric_cfg.get("field", "call_id"))
        if group_cols:
            return self._build_grouped_body(query_filter, metric_agg, group_cols, max_result_rows, ranking_order, plan, pre_sort_fetch_size)
        return {"size": 0, "query": query_filter, "aggs": {"metric_value": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        def extractor(agg_data: Dict[str, Any]) -> Any:
            metric_val = agg_data.get("metric_value", {})
            return metric_val.get("count", {}).get("value", 0)

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


__all__ = ["CountThresholdStrategy", "RateThresholdStrategy"]
