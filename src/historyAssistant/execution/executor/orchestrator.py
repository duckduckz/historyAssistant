from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from historyAssistant.config import (
    ConfigBundle,
    load_all_configs,
    resolve_elasticsearch_config,
    resolve_elasticsearch_config_for_table,
)
from historyAssistant.execution.strategies import STRATEGY_REGISTRY
from historyAssistant.execution.query_builder import compile_es_query, get_date_bounds, get_max_date
from historyAssistant.execution.query_models import QueryPlan, canonical_plan_to_query_plan
from historyAssistant.execution.result import to_raw_result
from historyAssistant.planning import REQUEST_TYPE_CHART, REQUEST_TYPE_QA

from .comparison_executor import execute_comparison_query
from .entity_intersection import apply_entity_intersection

YamlDict = Dict[str, Any]


# es requires interpolation method to be specified when asking for percentiles.
def get_quantile_interpolation(app_config: YamlDict) -> str:
    runtime_cfg = app_config.get("runtime", {})
    quantile_interpolation = str(runtime_cfg.get("quantile_interpolation", "linear")).strip().lower()
    if quantile_interpolation not in {"linear", "nearest", "lower", "higher", "midpoint"}:
        return "linear"
    return quantile_interpolation


def open_es_connection(*, config_bundle: Optional[ConfigBundle] = None) -> Elasticsearch:
    bundle = config_bundle or load_all_configs()
    es_cfg = resolve_elasticsearch_config(bundle.app_config)
    return Elasticsearch([f"http://{es_cfg['host']}:{es_cfg['port']}"])


def get_es_index(*, config_bundle: Optional[ConfigBundle] = None, canonical_table: Optional[str] = None) -> str:
    bundle = config_bundle or load_all_configs()
    if canonical_table:
        es_cfg = resolve_elasticsearch_config_for_table(bundle.app_config, canonical_table)
        return es_cfg["index"]
    es_cfg = resolve_elasticsearch_config(bundle.app_config)
    return es_cfg["index"]


def _parse_iso_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    text = value.strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _date_filter_range(plan: QueryPlan, date_field: str) -> tuple[Optional[date], Optional[date]]:
    starts: List[date] = []
    ends: List[date] = []
    for filt in plan.filters:
        if filt.field != date_field:
            continue
        if filt.op == "between" and isinstance(filt.value, (list, tuple)) and len(filt.value) == 2:
            start = _parse_iso_date(filt.value[0])
            end = _parse_iso_date(filt.value[1])
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        elif filt.op == "=":
            parsed = _parse_iso_date(filt.value)
            if parsed:
                starts.append(parsed)
                ends.append(parsed)
        elif filt.op in {">", ">="}:
            parsed = _parse_iso_date(filt.value)
            if parsed:
                starts.append(parsed)
        elif filt.op in {"<", "<="}:
            parsed = _parse_iso_date(filt.value)
            if parsed:
                ends.append(parsed)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _dataset_range_warning(
    plan: QueryPlan, min_date: Optional[date], max_date: Optional[date], date_field: str
) -> Optional[str]:
    if min_date is None or max_date is None:
        return None
    query_start, query_end = _date_filter_range(plan, date_field)
    if query_start is None and query_end is None:
        return None
    if query_end and query_end < min_date:
        return f"The requested date range is before the available {plan.canonical_table} data ({min_date} to {max_date})."
    if query_start and query_start > max_date:
        return f"The requested date range is after the available {plan.canonical_table} data ({min_date} to {max_date})."
    return None


def execute_plan(plan: QueryPlan, es: Elasticsearch, *, config_bundle: Optional[ConfigBundle] = None) -> YamlDict:
    bundle = config_bundle or load_all_configs()
    es_index = get_es_index(config_bundle=bundle, canonical_table=getattr(plan, "canonical_table", None))

    has_relative = any(f.op == "relative" for f in plan.filters)
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(plan.canonical_table, {}) or {}
    date_field = table_def.get("date_field", "call_date")
    reference_date = get_max_date(es, es_index, date_field) if has_relative else None

    dataset_min_date: Optional[date] = None
    dataset_max_date: Optional[date] = None
    if any(f.field == date_field for f in plan.filters):
        dataset_min_date, dataset_max_date = get_date_bounds(es, es_index, date_field)

    compiled = compile_es_query(plan, config_bundle=bundle, reference_date=reference_date)
    query_body = compiled.query_body
    metric_cfg = compiled.metric_cfg

    if getattr(plan, "bucket_interval", None):
        aggregation = "time_bucket_summary"
    else:
        aggregation = metric_cfg.get("aggregation", "count")

    strategy = STRATEGY_REGISTRY.get(aggregation)
    if not strategy:
        raise ValueError(f"No strategy for aggregation: {aggregation}")

    query_debug = str(query_body)
    try:
        result = es.search(index=es_index, body=query_body)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "query": query_debug,
            "data_source": {
                "canonical_table": plan.canonical_table,
                "physical_table": plan.target_table,
                "es_index": es_index,
                "date_field": date_field,
            },
            "dataframe": pd.DataFrame(),
            "where_debug": compiled.filter_debug,
            "overfetch_debug": compiled.overfetch_debug,
        }

    group_cols = plan.group_by or []
    df = strategy.extract_result(result, metric_cfg, group_cols, plan)
    total_hits = result.get("hits", {}).get("total", {}).get("value")

    warnings: List[str] = []
    if (total_hits == 0 or df.empty) and dataset_min_date and dataset_max_date:
        warning = _dataset_range_warning(plan, dataset_min_date, dataset_max_date, date_field)
        if warning:
            warnings.append(warning)

    return {
        "ok": True,
        "query": query_debug,
        "data_source": {
            "canonical_table": plan.canonical_table,
            "physical_table": plan.target_table,
            "es_index": es_index,
            "date_field": date_field,
        },
        "where_debug": compiled.filter_debug,
        "overfetch_debug": compiled.overfetch_debug,
        "error": None,
        "dataframe": df,
        "total_hits": total_hits,
        "warnings": warnings,
    }


def execute_canonical_plan(canonical_plan: YamlDict,es: Elasticsearch,*,config_bundle: Optional[ConfigBundle] = None) -> YamlDict:
    bundle = config_bundle or load_all_configs()
    request_type = canonical_plan.get("request_type")

    if request_type not in {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}:
        return {
            "ok": False,
            "request_type": request_type,
            "error": f"Execution path not implemented for request_type: {request_type}",
            "canonical_plan": canonical_plan,
        }

    try:
        query_plan = canonical_plan_to_query_plan(canonical_plan, config_bundle=bundle)
    except Exception as exc:
        return {
            "ok": False,
            "request_type": request_type,
            "error": f"Invalid canonical plan for execution: {exc}",
            "canonical_plan": canonical_plan,
        }

    execution_result = execute_plan(query_plan, es, config_bundle=bundle)
    raw_result = to_raw_result(execution_result, query_plan, config_bundle=bundle)
    raw_result["canonical_plan"] = canonical_plan

    if canonical_plan.get("query_shape") == "entity_intersection" and raw_result.get("ok", False):
        raw_result = apply_entity_intersection(canonical_plan, raw_result)

    compare_to = canonical_plan.get("compare_to")
    if compare_to and raw_result.get("ok", False):
        from historyAssistant.app.comparison_presenter import (
            build_comparison_summary,
            label_from_comparison_side,
        )

        comparison_result = execute_comparison_query(canonical_plan, compare_to, es, bundle, raw_result)
        if comparison_result:
            raw_result["comparison"] = comparison_result
            date_rules_config = getattr(bundle, "date_rules", None)
            display_value_maps = (bundle.table_registry.get("tables", {}) or {}).get(
                query_plan.canonical_table, {}
            ).get("display_value_maps")
            comparison_summary = build_comparison_summary(
                canonical_plan, compare_to, comparison_result,
                date_rules_config=date_rules_config,
                display_value_maps=display_value_maps,
            )
            if comparison_summary:
                raw_result["comparison_summary"] = comparison_summary
                if comparison_summary.get("winner_label") is not None:
                    raw_result["scalar_answer"] = comparison_summary["winner_label"]

            chart_payload = raw_result.get("chart_payload")
            if chart_payload and comparison_result.get("ok"):
                chart_payload["comparison_rows"] = comparison_result.get("records") or []
                chart_payload["comparison_is_date_swap"] = bool(
                    compare_to.get("date_rule") or compare_to.get("date_range")
                )
                chart_payload["main_period_label"] = (
                    label_from_comparison_side(
                        canonical_plan, compare_to,
                        date_rules_config=date_rules_config,
                        display_value_maps=display_value_maps,
                    ) or "Main"
                )
                chart_payload["comparison_period_label"] = (
                    label_from_comparison_side(
                        compare_to, canonical_plan,
                        date_rules_config=date_rules_config,
                        display_value_maps=display_value_maps,
                    ) or "Comparison"
                )

    return raw_result


def execute_full_raw_records_export(canonical_plan: YamlDict, *, es: Any, bundle: Any) -> YamlDict:
    from historyAssistant.execution.query_models import canonical_plan_to_query_plan
    from historyAssistant.execution.result import to_raw_result

    query_plan = canonical_plan_to_query_plan(canonical_plan, config_bundle=bundle)
    es_index = get_es_index(config_bundle=bundle, canonical_table=query_plan.canonical_table)
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(query_plan.canonical_table, {}) or {}
    date_field = table_def.get("date_field", "call_date")
    reference_date = get_max_date(es, es_index, date_field) if any(f.op == "relative" for f in query_plan.filters) else None
    compiled = compile_es_query(query_plan, config_bundle=bundle, reference_date=reference_date)

    query_body = deepcopy(compiled.query_body)
    query_body.pop("sort", None)
    query_body["size"] = 1000

    rows = [hit.get("_source", {}) for hit in scan(es, index=es_index, query=query_body, size=1000, scroll="2m")]
    df = pd.DataFrame(rows)
    execution_result = {
        "ok": True,
        "query": str(query_body),
        "data_source": {
            "canonical_table": query_plan.canonical_table,
            "physical_table": query_plan.target_table,
            "es_index": es_index,
            "date_field": date_field,
        },
        "where_debug": compiled.filter_debug,
        "overfetch_debug": compiled.overfetch_debug,
        "dataframe": df,
        "total_hits": len(rows),
    }
    return to_raw_result(execution_result, query_plan, config_bundle=bundle)


__all__ = [
    "execute_canonical_plan",
    "execute_full_raw_records_export",
    "execute_plan",
    "get_es_index",
    "get_quantile_interpolation",
    "open_es_connection",
]
