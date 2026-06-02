from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import AggregationStrategy, sort_and_limit_grouped_df
from historyAssistant.execution.strategies.threshold_metric_strategies import RateThresholdStrategy


class DqRateStrategy(RateThresholdStrategy):
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        dq_flag = metric_cfg.get("dq_flag")
        return {
            "scripted_metric": {
                "init_script": "state.total = 0; state.flagged = 0",
                "map_script": (
                    "state.total++; "
                    f"if (doc['{dq_flag}'].size() > 0 && doc['{dq_flag}'].value == 1) {{ state.flagged++; }}"
                ),
                "combine_script": "return ['total': state.total, 'flagged': state.flagged]",
                "reduce_script": (
                    "double total = 0; double flagged = 0; "
                    "for (s in states) { total += s.total; flagged += s.flagged; } "
                    "return total > 0 ? (flagged / total) * 100.0 : 0"
                ),
            }
        }, False


class DqCountStrategy(AggregationStrategy):
    _CLAUSE_BUILDERS = {
        "single": lambda flags, flag: {"term": {flag: 1}},
        "any": lambda flags, flag: {"bool": {"should": [{"term": {f: 1}} for f in flags], "minimum_should_match": 1}},
        "all": lambda flags, flag: {"bool": {"filter": [{"term": {f: 1}} for f in flags]}},
    }

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        dq_flags = metric_cfg.get("dq_flags", [])
        dq_flag = metric_cfg.get("dq_flag")
        builder = self._CLAUSE_BUILDERS[self.mode]
        bool_clause: Dict[str, Any] = builder(dq_flags, dq_flag)

        return {
            "filter": bool_clause,
            "aggs": {"count": {"value_count": {"field": "call_id"}}},
        }, False

    def build_query_body(self, query_filter: dict, metric_cfg: dict, group_cols: list, max_result_rows: int, ranking_order: dict, plan: QueryPlan, pre_sort_fetch_size: Optional[int] = None) -> dict:
        metric_agg, _ = self.build_metric_agg(metric_cfg, metric_cfg.get("field", "call_id"))
        if group_cols:
            return self._build_grouped_body(query_filter, metric_agg, group_cols, max_result_rows, ranking_order, plan, pre_sort_fetch_size)
        return {"size": 0, "query": query_filter, "aggs": {"metric_value": metric_agg}}

    def extract_result(self, es_response: dict, metric_cfg: dict, group_cols: list, plan: QueryPlan) -> pd.DataFrame:
        def extractor(agg_data: Dict[str, Any]) -> Any:
            metric_val = agg_data.get("metric_value", {})
            return metric_val.get("count", {}).get("value", metric_val.get("doc_count", 0))

        if group_cols:
            df = pd.DataFrame(self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor))
            return sort_and_limit_grouped_df(df, group_cols, plan)
        return self._extract_scalar_value(es_response, extractor)


__all__ = ["DqCountStrategy", "DqRateStrategy"]
