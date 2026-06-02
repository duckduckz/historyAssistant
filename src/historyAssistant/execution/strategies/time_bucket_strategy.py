from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.execution.strategies.base import AggregationStrategy, es_field_name, sort_and_limit_grouped_df


class TimeBucketStrategy(AggregationStrategy):
    needs_raw_values = False

    _TIME_SERIES_RESULT_RE = re.compile(
        r"(?i)\b(?:trend|over\s+time|changed?|change\s+over\s+time|"
        r"day\s+by\s+day|week\s+by\s+week|month\s+by\s+month|year\s+by\s+year|"
        r"daily|weekly|monthly|hourly|"
        r"per\s+(?:day|week|month|hour|minute)s?|"
        r"by\s+(?:day|week|month|hour))\b"
    )

    _INNER_AGG_MAP = {
        "count": "value_count",
        "avg": "avg",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "cardinality": "cardinality",
    }

    def _es_interval_params(self, interval: str) -> tuple[str, str]:
        if interval.endswith("w") or interval.endswith("M"):
            return "calendar_interval", interval
        return "fixed_interval", interval

    def _build_inner_metric_agg(self, metric_cfg: dict) -> dict:
        base_agg = metric_cfg.get("aggregation", "count")
        field = metric_cfg.get("field", "call_id")
        es_agg_key = self._INNER_AGG_MAP.get(base_agg, "value_count")
        count_field = metric_cfg.get("count_field", field) if es_agg_key == "value_count" else field
        return {es_agg_key: {"field": count_field}}

    def _apply_summary(self, values: list, summary: str) -> Any:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        if summary == "avg":
            return sum(clean) / len(clean)
        if summary == "min":
            return min(clean)
        if summary == "max":
            return max(clean)
        if summary in ("sum", "count"):
            return sum(clean)
        if summary == "median":
            sorted_values = sorted(clean)
            mid = len(sorted_values) // 2
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2 if len(sorted_values) % 2 == 0 else sorted_values[mid]
        return sum(clean) / len(clean)

    def _wants_series_output(self, plan: QueryPlan) -> bool:
        if getattr(plan, "request_type", None) == "chart":
            return True
        question = str(getattr(plan, "normalized_sub_request", "") or "")
        return bool(self._TIME_SERIES_RESULT_RE.search(question))

    def _extract_bucket_rows(
        self,
        agg_data: Dict[str, Any],
        group_cols: List[str],
        prefix: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        prefix = prefix or {}
        if not group_cols:
            rows: List[Dict[str, Any]] = []
            for bucket in agg_data.get("time_buckets", {}).get("buckets", []):
                row = dict(prefix)
                date_col = self._date_field
                row[date_col] = bucket.get("key_as_string", bucket.get("key"))
                row["metric_value"] = bucket.get("bucket_metric", {}).get("value")
                rows.append(row)
            return rows

        col = group_cols[0]
        agg_name = f"group_{col}"
        remaining = group_cols[1:]
        records: List[Dict[str, Any]] = []
        for bucket in agg_data.get(agg_name, {}).get("buckets", []):
            row_prefix = dict(prefix)
            row_prefix[col] = bucket.get("key_as_string", bucket.get("key"))
            records.extend(self._extract_bucket_rows(bucket, remaining, row_prefix))
        return records

    def build_metric_agg(self, metric_cfg: dict, field: str) -> tuple[dict, bool]:
        return {}, False

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
        interval = getattr(plan, "bucket_interval", None) or "1d"
        interval_key, interval_val = self._es_interval_params(interval)
        inner_metric_agg = self._build_inner_metric_agg(metric_cfg)
        self._date_field = metric_cfg.get("_date_field", "call_date")

        histogram_agg: Dict[str, Any] = {
            "date_histogram": {
                interval_key: interval_val,
                "field": self._date_field,
                "min_doc_count": 1,
            },
            "aggs": {"bucket_metric": inner_metric_agg},
        }

        if not group_cols:
            return {"size": 0, "query": query_filter, "aggs": {"time_buckets": histogram_agg}}

        body: Dict[str, Any] = {"size": 0, "query": query_filter, "aggs": {}}
        current_aggs = body["aggs"]
        for col in group_cols:
            agg_name = f"group_{col}"
            current_aggs[agg_name] = {
                "terms": {"field": es_field_name(col, plan), "size": max_result_rows},
                "aggs": {},
            }
            current_aggs = current_aggs[agg_name]["aggs"]
        current_aggs["time_buckets"] = histogram_agg
        return body

    def extract_result(
        self,
        es_response: dict,
        metric_cfg: dict,
        group_cols: list,
        plan: QueryPlan,
    ) -> pd.DataFrame:
        if self._wants_series_output(plan):
            rows = self._extract_bucket_rows(es_response.get("aggregations", {}), group_cols)
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            date_col = getattr(self, "_date_field", "call_date")
            sort_cols = [col for col in [*group_cols, date_col] if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
            return df

        summary = getattr(plan, "bucket_summary", None) or "avg"

        def _bucket_values(agg_data: Dict[str, Any]) -> list:
            buckets = agg_data.get("time_buckets", {}).get("buckets", [])
            return [b.get("bucket_metric", {}).get("value") for b in buckets]

        def extractor(agg_data: Dict[str, Any]) -> Any:
            return self._apply_summary(_bucket_values(agg_data), summary)

        if not group_cols:
            aggs = es_response.get("aggregations", {})
            return pd.DataFrame([{"metric_value": extractor(aggs)}])

        df = pd.DataFrame(
            self._extract_grouped_records(es_response.get("aggregations", {}), group_cols, extractor)
        )
        return sort_and_limit_grouped_df(df, group_cols, plan)


__all__ = ["TimeBucketStrategy"]
