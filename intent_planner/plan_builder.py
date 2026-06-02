from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

from historyAssistant.config import ConfigBundle

from .answer_shape import DISTINCT_VALUES, RAW_RECORDS
from .intent_slots import IntentSlots

YamlDict = Dict[str, Any]

_chart_rules = import_module("..chart.chart_rules", __package__)
_build_chart_rule_set = _chart_rules.build_chart_rule_set
_build_default_chart_spec = _chart_rules.build_default_chart_spec
_should_plan_auto_chart = _chart_rules.should_plan_auto_chart


def build_plan_from_slots(
    slots: IntentSlots,
    bundle: ConfigBundle,
    *,
    chart_suppressed: bool = False,
    preferred_chart_type: Optional[str] = None,
) -> YamlDict:
    metric_params = dict(slots.metric_params or {})
    entity_intersection = metric_params.pop("_entity_intersection", None)

    if slots.answer_shape == RAW_RECORDS:
        return _build_raw_records_plan(slots)

    # for distinct_values the projection_field must also be the group_by dimension. the LLM slot extractor puts it in projection_field 
    resolved_group_by = list(slots.group_by or [])
    if slots.answer_shape == DISTINCT_VALUES and slots.projection_field:
        if slots.projection_field not in resolved_group_by:
            resolved_group_by = [slots.projection_field] + resolved_group_by[:1]

    plan: YamlDict = {
        "request_type": slots.request_type if slots.request_type in {"qa", "chart"} else "qa",
        "target_tables": [slots.target_table],
        "metric": slots.metric,
        "aggregation": "time_bucket_summary" if slots.bucket_interval else slots.aggregation,
        "metric_params": metric_params,
        "filters": list(slots.filters or []),
        "group_by": resolved_group_by,
        "order_by": slots.order_by,
        "ranking_order": None,
        "display_order": [],
        "limit": slots.limit,
        "date_rule": slots.date_rule or "none",
        "date_range": slots.date_range,
        "date_params": slots.date_params,
        "time_range": slots.time_range,
        "compare_to": slots.comparison.to_compare_to()
        if slots.comparison and slots.comparison.comparison_type == "period"
        else None,
        "comparison_preferred_direction": slots.comparison.preferred_direction
        if slots.comparison
        else None,
        "bucket_interval": slots.bucket_interval,
        "bucket_summary": slots.bucket_summary,
        "normalized_sub_request": None,
        "sub_request_id": 1,
        "classification_confidence": slots.confidence,
    }
    _attach_intent_packet(plan, slots)
    if slots.answer_shape == DISTINCT_VALUES and slots.projection_field:
        plan["scope_policy"] = "raw"
        plan["outlier_policy"] = "none"

    if entity_intersection:
        plan["request_type"] = "qa"
        plan["query_shape"] = "entity_intersection"
        plan["entity_intersection"] = dict(entity_intersection)

    metric_cfg = (bundle.metric_catalog.get("metrics", {}) or {}).get(slots.metric or "", {}) or {}
    if metric_cfg.get("default_scope"):
        plan["scope_policy"] = metric_cfg.get("default_scope")
    if metric_cfg.get("default_outlier_policy"):
        plan["outlier_policy"] = metric_cfg.get("default_outlier_policy")

    if slots.ranking.detected and resolved_group_by:
        ranking_field = slots.ranking.field or "metric_value"
        if ranking_field == "percent_change":
            # percent_change is produced after compare_to execution, so ES cannot sort by it directly. Do not create a ranking_order that validator and es cannot execute.
            plan["_post_compare_ranking"] = {
                "field": "percent_change",
                "direction": slots.ranking.direction or "desc",
                "limit": slots.ranking.limit or 10,
            }
        else:
            plan["ranking_order"] = {
                "field": ranking_field,
                "direction": slots.ranking.direction or "desc",
                "target_group": resolved_group_by[-1],
            }
            plan["order_by"] = dict(plan["ranking_order"])
            plan["limit"] = slots.ranking.limit or 10

    auto_chart, _ = _should_plan_auto_chart(plan, bundle, chart_suppressed=chart_suppressed)
    if entity_intersection:
        pass
    elif slots.bucket_interval and not chart_suppressed:
        date_field = _date_field_for_table(bundle, slots.target_table)
        chart_group_by = [date_field, *resolved_group_by[:1]] if date_field else resolved_group_by[:1]
        plan["request_type"] = "chart"
        plan["chart_spec"] = _build_default_chart_spec(
            metric_name=slots.metric,
            group_by=chart_group_by,
            chart_type=preferred_chart_type or "line",
            rule_set=_build_chart_rule_set(bundle.chart_catalog),
        )
        plan["_chart_intent_source"] = "explicit" if preferred_chart_type else "auto_time_bucket"
    elif preferred_chart_type or (auto_chart and len(resolved_group_by) <= 1):
        plan["request_type"] = "chart"
        plan["chart_spec"] = _build_default_chart_spec(
            metric_name=slots.metric,
            group_by=resolved_group_by,
            chart_type=preferred_chart_type or "bar",
            rule_set=_build_chart_rule_set(bundle.chart_catalog),
        )
        plan["_chart_intent_source"] = "explicit" if preferred_chart_type else "auto"

    return plan


def _build_raw_records_plan(slots: IntentSlots) -> YamlDict:
    plan: YamlDict = {
        "request_type": "qa",
        "target_tables": [slots.target_table],
        "metric": slots.metric,
        "aggregation": "raw_records",
        "metric_params": dict(slots.metric_params or {}),
        "filters": list(slots.filters or []),
        "group_by": [],
        "order_by": slots.order_by,
        "ranking_order": None,
        "display_order": [],
        "limit": slots.limit,
        "scope_policy": "raw",
        "outlier_policy": "none",
        "source_fields": list(slots.source_fields or []),
        "date_rule": slots.date_rule or "none",
        "date_range": slots.date_range,
        "date_params": slots.date_params,
        "time_range": slots.time_range,
        "compare_to": None,
        "bucket_interval": None,
        "bucket_summary": None,
        "normalized_sub_request": None,
        "sub_request_id": 1,
        "classification_confidence": slots.confidence,
    }
    _attach_intent_packet(plan, slots)
    return plan


def _attach_intent_packet(plan: YamlDict, slots: IntentSlots) -> None:
    field_selection = dict(slots.narration_fields or {})
    plan["_intent_packet"] = {
        "source_id": slots.source_id,
        "target_table": slots.target_table,
        "answer_shape": slots.answer_shape,
        "metric": slots.metric,
        "filters": list(slots.filters or []),
        "source_fields": list(slots.source_fields or []),
        "projection_field": slots.projection_field,
        "field_selection": field_selection,
        "extraction_path": slots.extraction_path,
        "reason": slots.reason,
    }
    if field_selection:
        plan["_narration_fields"] = field_selection


def _date_field_for_table(bundle: ConfigBundle, target_table: str) -> Optional[str]:
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(target_table, {}) or {}
    time_field = table_def.get("time_field") or []
    timestamp_fields = table_def.get("timestamp_fields") or []
    return (
        table_def.get("date_field")
        or (time_field[0] if isinstance(time_field, list) and time_field else None)
        or (timestamp_fields[0] if isinstance(timestamp_fields, list) and timestamp_fields else None)
    )
