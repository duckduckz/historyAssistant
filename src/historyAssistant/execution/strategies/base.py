from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan

YamlDict = Dict[str, Any]

_OVERFETCH_CAP = 1000


def es_field_name(field: str, plan: Optional[QueryPlan] = None) -> str:
    keyword_fields = set(getattr(plan, "keyword_fields", []) or []) if plan is not None else set()
    if field in keyword_fields:
        return f"{field}.keyword"
    return field


def supports_metric_bucket_order(metric_agg: Dict[str, Any]) -> bool:
    supported_metric_aggs = {"avg", "sum", "min", "max", "value_count", "cardinality"}
    return any(name in metric_agg for name in supported_metric_aggs)


def sort_and_limit_grouped_df(df: pd.DataFrame, group_cols: List[str], plan: QueryPlan) -> pd.DataFrame:
    if df.empty or "metric_value" not in df.columns:
        return df

    ranking_order = plan.effective_ranking_order() or {}
    if not isinstance(ranking_order, dict) or not ranking_order.get("field"):
        return df

    direction = ranking_order.get("direction", "desc")
    ascending = direction == "asc"
    target_group = ranking_order.get("target_group")
    outer_groups = [g for g in group_cols if g != target_group] if target_group and len(group_cols) > 1 else []
    if outer_groups:
        tie_break_cols = [target_group] if target_group and target_group in df.columns else []
        sort_cols = outer_groups + ["metric_value"] + tie_break_cols
        sort_asc = [True] * len(outer_groups) + [ascending] + [True] * len(tie_break_cols)
        df = df.sort_values(sort_cols, ascending=sort_asc)
        chart_has_series = (
            plan.request_type == "chart"
            and isinstance(getattr(plan, "chart_spec", None), dict)
            and plan.chart_spec.get("series")
        )
        if plan.limit and not chart_has_series:
            df = df.groupby(outer_groups, sort=False).head(plan.limit).reset_index(drop=True)
        return df

    tie_break_cols = [col for col in group_cols if col in df.columns]
    sort_cols = ["metric_value"] + tie_break_cols
    sort_asc = [ascending] + [True] * len(tie_break_cols)
    df = df.sort_values(sort_cols, ascending=sort_asc)
    if plan.limit:
        df = df.head(plan.limit)
    return df


def resolve_pre_sort_fetch_size(max_result_rows: int, pre_sort_fetch_size: Optional[int]) -> int:
    if pre_sort_fetch_size is None:
        return max_result_rows
    return max(max_result_rows, min(int(pre_sort_fetch_size), _OVERFETCH_CAP))


class AggregationStrategy(ABC):
    needs_raw_values = False

    @abstractmethod
    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        pass

    @abstractmethod
    def build_query_body(
        self,
        query_filter: dict,
        metric_cfg: dict,
        group_cols: list,
        max_result_rows: int,
        ranking_order: dict,
        plan: QueryPlan,
        pre_sort_fetch_size: Optional[int] = None,
    ) -> dict:
        pass

    @abstractmethod
    def extract_result(
        self,
        es_response: dict,
        metric_cfg: dict,
        group_cols: list,
        plan: QueryPlan,
    ) -> pd.DataFrame:
        pass

    def resolve_limit(
        self,
        plan: QueryPlan,
        metric_cfg: dict,
        default_limit: int,
        max_result_rows: int,
    ) -> Optional[int]:
        if group_cols := (plan.group_by or []):
            limit_val = plan.limit if plan.limit is not None else default_limit
            return min(int(limit_val), max_result_rows)
        return None

    def _build_grouped_body(
        self,
        query_filter: dict,
        metric_agg: dict,
        group_cols: list,
        max_result_rows: int,
        ranking_order: dict,
        plan: QueryPlan,
        pre_sort_fetch_size: Optional[int] = None,
    ) -> dict:
        body = {"size": 0, "query": query_filter, "aggs": {}}
        current_aggs = body["aggs"]
        order_field = ranking_order.get("field")
        order_direction = ranking_order.get("direction", "desc")
        order_target_group = ranking_order.get("target_group")

        for col in group_cols:
            agg_name = f"group_{col}"
            order_clause = {}
            child_aggs: YamlDict = {}

            if order_field == col and order_target_group == col:
                order_clause = {"_key": order_direction}
            elif order_field == "metric_value" and order_target_group == col and supports_metric_bucket_order(metric_agg):
                sort_metric_name = f"sort_metric_{col}"
                child_aggs[sort_metric_name] = metric_agg
                order_clause = {sort_metric_name: order_direction}

            needs_overfetch = (
                not order_clause
                and order_field == "metric_value"
                and order_target_group == col
            )
            fetch_size = resolve_pre_sort_fetch_size(max_result_rows, pre_sort_fetch_size) if needs_overfetch else max_result_rows
            terms_config = {"field": es_field_name(col, plan), "size": fetch_size}
            if order_clause:
                terms_config["order"] = order_clause

            current_aggs[agg_name] = {"terms": terms_config, "aggs": child_aggs}
            current_aggs = current_aggs[agg_name]["aggs"]

        current_aggs["metric_value"] = metric_agg
        return body

    def _extract_grouped_records(
        self,
        agg_data: Dict[str, Any],
        group_cols: List[str],
        extractor: Callable[[Dict[str, Any]], Any],
        prefix: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        prefix = prefix or {}
        if not group_cols:
            row = dict(prefix)
            row["metric_value"] = extractor(agg_data)
            return [row]

        col = group_cols[0]
        agg_name = f"group_{col}"
        remaining = group_cols[1:]
        records: List[Dict[str, Any]] = []
        for bucket in agg_data.get(agg_name, {}).get("buckets", []):
            row_prefix = dict(prefix)
            row_prefix[col] = bucket.get("key_as_string", bucket.get("key"))
            records.extend(self._extract_grouped_records(bucket, remaining, extractor, row_prefix))
        return records

    def _extract_scalar_value(self, es_response: Dict[str, Any], extractor: Callable[[Dict[str, Any]], Any]) -> pd.DataFrame:
        aggs = es_response.get("aggregations", {})
        return pd.DataFrame([{"metric_value": extractor(aggs)}])


__all__ = [
    "AggregationStrategy",
    "es_field_name",
    "resolve_pre_sort_fetch_size",
    "sort_and_limit_grouped_df",
]
