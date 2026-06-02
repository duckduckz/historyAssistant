from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from historyAssistant.config import ConfigBundle
from historyAssistant.execution.query_builder.date_filters import relative_date_filter
from historyAssistant.execution.query_builder.field_resolver import (
    boolean_fields_for_plan,
    coerce_boolean,
    coerce_filter_value,
    es_field_name,
    expand_date_only_range,
    expand_relative_date_clause_for_datetime_field,
    field_uses_datetime_date_range,
    is_date_only,
    keyword_fields_for_plan,
    numeric_outlier_fields_for_plan,
    table_definition_for_plan,
    value_map_for_field,
)
from historyAssistant.execution.query_models import QueryFilter, QueryPlan

YamlDict = Dict[str, Any]
_RANGE_OPS = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
_COMPARISON_OPS = {"=", "!=", ">", ">=", "<", "<="}
_DQ_AGGREGATIONS_WITHOUT_REQUIRED_NULL_FILTERS = {
    "dq_rate",
    "dq_count",
    "dq_count_any_true",
    "dq_count_all_true",
}


def build_filters(
    plan: QueryPlan,
    metric_cfg: YamlDict,
    reference_date: Optional[date] = None,
    config_bundle: Optional[ConfigBundle] = None,
) -> Tuple[List[Dict], List[str]]:
    context = _FilterBuildContext(plan=plan, config_bundle=config_bundle)
    filters, debug = _policy_filters(plan, metric_cfg, context)

    for query_filter in plan.filters:
        clause, desc = _filter_clause(query_filter, plan, context, reference_date)
        if clause:
            filters.append(clause)
            debug.append(desc)

    required_filters, required_debug = _required_column_filters(plan, metric_cfg, context)
    filters.extend(required_filters)
    debug.extend(required_debug)
    return filters, debug


class _FilterBuildContext:
    def __init__(self, *, plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> None:
        self.config_bundle = config_bundle
        self.table_def = table_definition_for_plan(plan, config_bundle)
        self.keyword_fields = keyword_fields_for_plan(plan, config_bundle)
        self.boolean_fields = boolean_fields_for_plan(plan, config_bundle)
        self.numeric_outlier_fields = numeric_outlier_fields_for_plan(plan, config_bundle)
        self.time_fields = set(self.table_def.get("time_field", []) or [])
        self.timestamp_fields = set(self.table_def.get("timestamp_fields", []) or [])

    def es_field(self, field: str) -> str:
        return es_field_name(field, self.keyword_fields)

    def is_time_field(self, field: str) -> bool:
        return field in self.time_fields or field in self.timestamp_fields


def _policy_filters(plan: QueryPlan, metric_cfg: YamlDict, context: _FilterBuildContext) -> Tuple[List[Dict], List[str]]:
    filters: List[Dict] = []
    debug: List[str] = []

    if plan.scope_policy == "qa_ready":
        for flag in context.table_def.get("scope_flags", []) or []:
            filters.append({"term": {flag: 1}})
            debug.append(f"{flag} = 1")

    metric_field = metric_cfg.get("field")
    if plan.outlier_policy == "exclude_negative" and metric_field in context.numeric_outlier_fields:
        filters.append({"range": {metric_field: {"gte": 0}}})
        debug.append(f"{metric_field} >= 0")
    return filters, debug


def _filter_clause(
    filt: QueryFilter,
    plan: QueryPlan,
    context: _FilterBuildContext,
    reference_date: Optional[date],
) -> Tuple[Optional[Dict], str]:
    if filt.op == "relative":
        clause, desc = relative_date_filter(
            filt.value, filt.field, reference_date,
            date_rules_config=getattr(context.config_bundle, "date_rules", None),
        )
        return (
            expand_relative_date_clause_for_datetime_field(clause, plan, context.config_bundle, filt.field),
            desc,
        )
    if filt.op == "time_between":
        return _time_between_clause(filt), f"time({filt.field}) between {filt.value[0]} and {filt.value[1]}"
    if filt.op == "wildcard":
        return {"wildcard": {context.es_field(filt.field): {"value": str(filt.value).lower()}}}, f"{filt.field} wildcard {filt.value}"
    if filt.op in _COMPARISON_OPS:
        return _comparison_clause(filt, plan, context)
    if filt.op == "in":
        return _in_clause(filt, plan, context)
    if filt.op == "between":
        return _between_clause(filt, plan, context)
    return None, ""


def _comparison_clause(filt: QueryFilter, plan: QueryPlan, context: _FilterBuildContext) -> Tuple[Optional[Dict], str]:
    if filt.field in context.boolean_fields:
        value = coerce_boolean(filt.value)
        return _term_or_not_clause(filt.field, filt.op, value), f"{filt.field} = {value}"

    if context.is_time_field(filt.field):
        return _time_field_comparison_clause(filt, plan, context), f"{filt.field} {filt.op} {filt.value}"

    es_field = context.es_field(filt.field)
    value = coerce_filter_value(filt.value, value_map_for_field(filt.field, plan, context.config_bundle))
    if filt.op in {"=", "!="}:
        return _term_or_not_clause(es_field, filt.op, value), f"{filt.field} {filt.op} {value}"
    return {"range": {es_field: {_RANGE_OPS[filt.op]: value}}}, f"{filt.field} {filt.op} {value}"


def _time_field_comparison_clause(filt: QueryFilter, plan: QueryPlan, context: _FilterBuildContext) -> Dict:
    if filt.op == "=":
        start_val, end_val = expand_date_only_range(plan, context.config_bundle, filt.field, filt.value, filt.value)
        return {"range": {filt.field: {"gte": start_val, "lte": end_val}}}
    if filt.op == "!=":
        start_val, end_val = expand_date_only_range(plan, context.config_bundle, filt.field, filt.value, filt.value)
        return {"bool": {"must_not": [{"range": {filt.field: {"gte": start_val, "lte": end_val}}}]}}
    value = filt.value
    if field_uses_datetime_date_range(plan, context.config_bundle, filt.field) and is_date_only(value):
        value = f"{value.strip()} 00:00:00" if filt.op in {">", ">="} else f"{value.strip()} 23:59:59"
    return {"range": {filt.field: {_RANGE_OPS[filt.op]: value}}}


def _term_or_not_clause(field: str, op: str, value: Any) -> Dict:
    if op == "!=":
        return {"bool": {"must_not": [{"term": {field: value}}]}}
    return {"term": {field: value}}


def _in_clause(filt: QueryFilter, plan: QueryPlan, context: _FilterBuildContext) -> Tuple[Optional[Dict], str]:
    if not isinstance(filt.value, (list, tuple)):
        return None, ""
    value_map = value_map_for_field(filt.field, plan, context.config_bundle)
    values = [coerce_filter_value(value, value_map) for value in filt.value]
    return {"terms": {context.es_field(filt.field): values}}, f"{filt.field} IN {values}"


def _between_clause(filt: QueryFilter, plan: QueryPlan, context: _FilterBuildContext) -> Tuple[Optional[Dict], str]:
    if not isinstance(filt.value, (list, tuple)) or len(filt.value) != 2:
        return None, ""
    start_val, end_val = expand_date_only_range(plan, context.config_bundle, filt.field, filt.value[0], filt.value[1])
    return {"range": {filt.field: {"gte": start_val, "lte": end_val}}}, f"{filt.field} between {start_val} and {end_val}"


def _time_between_clause(filt: QueryFilter) -> Dict:
    start_t, end_t = filt.value
    return {
        "script": {
            "script": {
                "source": (
                    "if (doc[params.field].size() == 0) return false; "
                    "def ts = doc[params.field].value.toString(); "
                    "def sep = ts.indexOf('T'); "
                    "if (sep < 0) sep = ts.indexOf(' '); "
                    "if (sep < 0 || ts.length() < sep + 6) return false; "
                    "def h = Integer.parseInt(ts.substring(sep + 1, sep + 3)); "
                    "def m = Integer.parseInt(ts.substring(sep + 4, sep + 6)); "
                    "def t = h * 60 + m; "
                    "def s = params.sh * 60 + params.sm; "
                    "def e = params.eh * 60 + params.em; "
                    "return t >= s && t <= e"
                ),
                "params": {
                    "field": filt.field,
                    "sh": int(start_t.split(":")[0]),
                    "sm": int(start_t.split(":")[1]) if ":" in start_t else 0,
                    "eh": int(end_t.split(":")[0]),
                    "em": int(end_t.split(":")[1]) if ":" in end_t else 0,
                },
            }
        }
    }


def _required_column_filters(plan: QueryPlan, metric_cfg: YamlDict, context: _FilterBuildContext) -> Tuple[List[Dict], List[str]]:
    if metric_cfg.get("aggregation", "count") in _DQ_AGGREGATIONS_WITHOUT_REQUIRED_NULL_FILTERS:
        return [], []
    primary_key = context.table_def.get("primary_key", "call_id")
    filters = [{"exists": {"field": col}} for col in metric_cfg.get("required_columns", []) if col != primary_key]
    debug = [f"{col} IS NOT NULL" for col in metric_cfg.get("required_columns", []) if col != primary_key]
    return filters, debug
