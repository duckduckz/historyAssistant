from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from historyAssistant.config import ConfigBundle

YamlDict = Dict[str, Any]


def build_raw_detail_plan(plan: YamlDict, *, bundle: ConfigBundle) -> Optional[YamlDict]:
    if not isinstance(plan, dict) or not plan.get("target_tables"):
        return None

    raw_metric = _raw_record_metric(bundle)
    if not raw_metric:
        return None

    table_name = str((plan.get("target_tables") or [""])[0])
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(table_name, {}) or {}
    timestamp_field = _timestamp_field(table_def)

    updated = deepcopy(plan)
    updated["request_type"] = "qa"
    updated["metric"] = raw_metric
    updated["aggregation"] = "raw_records"
    updated["group_by"] = []
    updated["ranking_order"] = None
    updated["display_order"] = []
    updated["chart_spec"] = None
    updated["_chart_intent_source"] = None
    updated["scope_policy"] = "raw"
    updated["outlier_policy"] = "none"
    updated["source_fields"] = _source_fields(table_def, updated)
    if timestamp_field:
        updated["order_by"] = {"field": timestamp_field, "direction": "desc"}
    else:
        updated["order_by"] = None
    updated["_detail_transform"] = {
        "from_metric": plan.get("metric"),
        "from_aggregation": plan.get("aggregation"),
        "reason": "raw_detail_followup",
    }
    return updated


def _raw_record_metric(bundle: ConfigBundle) -> Optional[str]:
    metrics = bundle.metric_catalog.get("metrics", {}) or {}
    for metric_name, metric_cfg in metrics.items():
        if isinstance(metric_cfg, dict) and metric_cfg.get("aggregation") == "raw_records":
            return str(metric_name)
    return None


def _timestamp_field(table_def: YamlDict) -> Optional[str]:
    for field in [
        table_def.get("timestamp_field"),
        *_as_list(table_def.get("timestamp_fields")),
        *_as_list(table_def.get("time_field")),
        table_def.get("date_field"),
    ]:
        field_name = str(field or "").strip()
        if field_name:
            return field_name
    return None


def _source_fields(table_def: YamlDict, plan: YamlDict) -> List[str]:
    fields: List[str] = []
    fields.extend(str(item) for item in (table_def.get("default_source_fields") or []) if str(item).strip())
    for field in [
        table_def.get("date_field"),
        table_def.get("timestamp_field"),
        *_as_list(table_def.get("timestamp_fields")),
        *_as_list(table_def.get("time_field")),
    ]:
        if field:
            fields.append(str(field))
    for filt in plan.get("filters") or []:
        field = str((filt or {}).get("field") or "").strip()
        if field:
            fields.append(field)
    return _dedupe(fields)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _dedupe(values: List[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


__all__ = ["build_raw_detail_plan"]
