from __future__ import annotations

import re
from typing import Any, Dict, Optional

from historyAssistant.config import ConfigBundle
from historyAssistant.execution.query_models import QueryPlan

YamlDict = Dict[str, Any]

_TRUTHY = {"true", "1", "yes"}
_FALSY = {"false", "0", "no"}


def table_definition_for_plan(plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> YamlDict:
    if config_bundle is None:
        return {}
    tables = config_bundle.table_registry.get("tables", {}) or {}
    return tables.get(plan.canonical_table, {}) or {}


def keyword_fields_for_plan(plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> set[str]:
    return set(table_definition_for_plan(plan, config_bundle).get("keyword_fields", []) or [])


def boolean_fields_for_plan(plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> set[str]:
    return set(table_definition_for_plan(plan, config_bundle).get("boolean_fields", []) or [])


def numeric_outlier_fields_for_plan(plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> set[str]:
    return set(table_definition_for_plan(plan, config_bundle).get("numeric_outlier_fields", []) or [])


def value_map_for_field(field: str, plan: QueryPlan, config_bundle: Optional[ConfigBundle]) -> dict:
    table_def = table_definition_for_plan(plan, config_bundle)
    field_info = (table_def.get("dimension_value_maps") or {}).get(field, {}) or {}
    return field_info.get("value_map") or {}


def coerce_filter_value(value: Any, value_map: dict) -> Any:
    if not value_map or not isinstance(value, str):
        return value
    key = value.strip().lower()
    return value_map[key] if key in value_map else value


def coerce_boolean(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUTHY:
            return 1
        if low in _FALSY:
            return 0
    raise ValueError(f"cannot coerce {value!r} to boolean")


def es_field_name(field: str, keyword_fields: Optional[set[str]] = None) -> str:
    return f"{field}.keyword" if field in (keyword_fields or set()) else field


def field_uses_datetime_date_range(plan: QueryPlan, config_bundle: Optional[ConfigBundle], field: str) -> bool:
    table_def = table_definition_for_plan(plan, config_bundle)
    return field == table_def.get("date_field") and field == table_def.get("timestamp_field")


def is_date_only(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def expand_date_only_range(
    plan: QueryPlan,
    config_bundle: Optional[ConfigBundle],
    field: str,
    start_val: Any,
    end_val: Any,
) -> tuple[Any, Any]:
    if not field_uses_datetime_date_range(plan, config_bundle, field):
        return start_val, end_val
    if is_date_only(start_val):
        start_val = f"{str(start_val).strip()} 00:00:00"
    if is_date_only(end_val):
        end_val = f"{str(end_val).strip()} 23:59:59"
    return start_val, end_val


def expand_relative_date_clause_for_datetime_field(
    clause: Optional[Dict],
    plan: QueryPlan,
    config_bundle: Optional[ConfigBundle],
    field: str,
) -> Optional[Dict]:
    if not clause or not field_uses_datetime_date_range(plan, config_bundle, field):
        return clause
    range_body = clause.get("range") if isinstance(clause, dict) else None
    if not isinstance(range_body, dict) or field not in range_body:
        return clause
    bounds = range_body.get(field)
    if not isinstance(bounds, dict):
        return clause

    updated = dict(bounds)
    if is_date_only(updated.get("gte")):
        updated["gte"] = f"{updated['gte'].strip()} 00:00:00"
    if is_date_only(updated.get("gt")):
        updated["gt"] = f"{updated['gt'].strip()} 23:59:59"
    if is_date_only(updated.get("lte")):
        updated["lte"] = f"{updated['lte'].strip()} 23:59:59"
    if is_date_only(updated.get("lt")):
        updated["lt"] = f"{updated['lt'].strip()} 00:00:00"
    return {"range": {field: updated}}
