from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from historyAssistant.config import ConfigBundle, load_all_configs
from historyAssistant.execution.query_models import QueryPlan
from historyAssistant.planning import REQUEST_TYPE_CHART

YamlDict = Dict[str, Any]

def build_chart_payload(plan: QueryPlan, records: list[YamlDict]) -> Optional[YamlDict]:
    if plan.request_type != REQUEST_TYPE_CHART:
        return None
    if plan.aggregation =="raw_records":
        return None
    return {
        "chart_spec": plan.chart_spec,
        "metric": plan.metric,
        "group_by": plan.group_by,
        "rows": records,
        "bucket_interval": getattr(plan, "bucket_interval", None),
        "bucket_summary": getattr(plan, "bucket_summary", None),
    }

def normalize_scalar_answer(value: Any, metric: str, *, is_time_bucket: bool = False, metric_cfg: Optional[YamlDict] = None) -> Any:
    if value is None:
        return None
    is_count = (metric_cfg or {}).get("output_type") == "count"
    if isinstance(value, (int, float)) and (is_time_bucket or not is_count):
        return round(float(value), 2)
    return value


def get_display_value_maps(group_by: List[str], config_bundle: Optional[ConfigBundle]) -> Dict[str, Dict]:
    if not group_by or config_bundle is None:
        return {}
    tables = config_bundle.table_registry.get("tables", {})
    active_table = next(iter(tables.keys()), "historyCalls")
    table_def = tables.get(active_table, {})

    dvm_all = table_def.get("dimension_value_maps", {})
    maps: Dict[str, Dict] = {}
    for col in group_by:
        dvm = dvm_all.get(col, {})
        dm = dvm.get("display_value_map")
        if dm:
            maps[col] = {int(k): v for k, v in dm.items()}
    return maps


def get_configured_display_orders(group_by: List[str], config_bundle: Optional[ConfigBundle]) -> Dict[str, List[Any]]:
    if not group_by or config_bundle is None:
        return {}
    tables = config_bundle.table_registry.get("tables", {})
    active_table = next(iter(tables.keys()), "historyCalls")
    table_def = tables.get(active_table, {})

    dvm_all = table_def.get("dimension_value_maps", {})
    orders: Dict[str, List[Any]] = {}
    for col in group_by:
        dvm = dvm_all.get(col, {})
        configured = dvm.get("display_order") or dvm.get("allowed_values")
        if configured:
            orders[col] = list(configured)
    return orders


def apply_display_maps(df: pd.DataFrame, records: list, display_maps: Dict[str, Dict]) -> tuple:
    if not display_maps:
        return df, records
    for col, mapping in display_maps.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v, m=mapping: m.get(int(v), v) if pd.notna(v) else v)
    records = df.to_dict(orient="records") if not df.empty else records
    return df, records


def apply_display_order(df: pd.DataFrame, plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> pd.DataFrame:
    if df.empty or not plan.display_order:
        return df
    if isinstance(plan.ranking_order, dict) and plan.ranking_order.get("field"):
        return df

    configured_orders = get_configured_display_orders(plan.group_by, config_bundle)
    sort_cols: List[str] = []
    ascending: List[bool] = []
    temp_cols: List[str] = []

    for idx, spec in enumerate(plan.display_order):
        field_name = spec.get("field")
        if field_name not in df.columns:
            continue
        direction = spec.get("direction", "asc")
        mode = spec.get("mode", "configured")
        if mode == "configured" and field_name in configured_orders:
            value_order = configured_orders[field_name]
            rank_map = {value: pos for pos, value in enumerate(value_order)}
            temp_col = f"__display_order_{idx}"
            df[temp_col] = df[field_name].map(lambda value, m=rank_map: m.get(value, len(m)))
            sort_cols.append(temp_col)
            ascending.append(direction != "desc")
            temp_cols.append(temp_col)
        else:
            sort_cols.append(field_name)
            ascending.append(direction != "desc")

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    if temp_cols:
        df = df.drop(columns=temp_cols)
    return df


def has_explicit_rank_limit(question: str) -> bool:
    text = str(question or "")
    return bool(
        re.search(
            r"(?i)\b(?:top|bottom)\s*[-]?\s*\d+\b|\b(?:which|what(?:\s+are)?|show(?:\s+me)?(?:\s+the)?)\s+\d+\b",
            text,
        )
    )


def infer_rank_winner_scalar(df: pd.DataFrame, plan: QueryPlan) -> Any:
    if df.empty or len(plan.group_by) != 1:
        return None
    if not isinstance(plan.ranking_order, dict) or not plan.ranking_order.get("field"):
        return None
    if has_explicit_rank_limit(plan.normalized_sub_request or ""):
        return None

    question = str(plan.normalized_sub_request or "").strip().lower()
    if not question.startswith(("which ", "what ")):
        return None

    target_group = plan.ranking_order.get("target_group") or plan.group_by[0]
    if target_group not in df.columns:
        target_group = plan.group_by[0]
    return df.iloc[0].get(target_group)


def to_raw_result(execution_result: YamlDict, plan: QueryPlan, *, config_bundle: Optional[ConfigBundle] = None) -> YamlDict:
    # start from the df created by execution.then derive the other result views that the app needs later.
    df = execution_result.get("dataframe", pd.DataFrame())
    records = df.to_dict(orient="records") if not df.empty else []

    scalar_answer = None
    if not df.empty and "metric_value" in df.columns and len(df.columns) == 1:
        # if the df just one metric_value column, will take the first value as the scalar answer.
        scalar_answer = df.iloc[0]["metric_value"]
    elif not df.empty and "metric_value" in df.columns and len(df) == 1 and len(plan.group_by) == 0:
        scalar_answer = df.iloc[0]["metric_value"] # here use iloc[0] is to handle when the df has only one row but the index is not starting from 0

    # clean up scalar formatting before returning the result.
    is_time_bucket = bool(getattr(plan, "bucket_interval", None))
    _metric_cfg: YamlDict = (getattr(config_bundle, "metrics", None) or {}).get(plan.metric, {})
    scalar_answer = normalize_scalar_answer(scalar_answer, plan.metric, is_time_bucket=is_time_bucket, metric_cfg=_metric_cfg)

    if not df.empty:
        df = apply_display_order(df, plan, config_bundle)
        records = df.to_dict(orient="records")

    display_maps = get_display_value_maps(plan.group_by, config_bundle)
    if display_maps and not df.empty:
        df, records = apply_display_maps(df, records, display_maps)

    if scalar_answer is None and not df.empty:
        inferred_scalar = infer_rank_winner_scalar(df, plan)
        if inferred_scalar is not None:
            scalar_answer = inferred_scalar

    if plan.aggregation == "room_switch_within_minutes" and execution_result.get("ok", False):
        window = None
        if config_bundle is not None:
            metric_cfg = (config_bundle.metric_catalog.get("metrics", {}) or {}).get(plan.metric, {}) or {}
            window = metric_cfg.get("window_minutes")
        window_text = f" within {window} minutes" if window else ""
        if records:
            scalar_answer = f"Yes, found at least {len(records)} same-person different-room Presence event pair(s){window_text}."
        else:
            scalar_answer = f"No same-person different-room Presence event pairs were found{window_text}."

    result_type = "raw_records" if plan.aggregation == "raw_records" else "aggregated"

    # return the final raw result structure used across the app.
    return {
        "ok": execution_result.get("ok", False),
        "request_type": plan.request_type,
        "result_type": result_type,
        "query": execution_result.get("query"),
        "data_source": execution_result.get("data_source"),
        "where_debug": execution_result.get("where_debug", []),
        "overfetch_debug": execution_result.get("overfetch_debug"),
        "dataframe": df,
        "records": records,
        "total_count": execution_result.get("total_hits", len(records)),
        "scalar_answer": scalar_answer,
        "chart_payload": build_chart_payload(plan, records),
        "empty": df.empty,
        "plan": asdict(plan),
        "error": execution_result.get("error"),
        "warnings": list(execution_result.get("warnings", []) or []),
    }


__all__ = [
    "build_chart_payload",
    "normalize_scalar_answer",
    "to_raw_result",
]
