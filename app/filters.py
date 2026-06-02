from __future__ import annotations
from typing import List

from historyAssistant.config import YamlDict


def has_date_filter(filters: List[YamlDict], *, date_field: str = "call_date") -> bool:
    return any(
        isinstance(f, dict) and f.get("field") == date_field
        and f.get("op") in {"relative", "between", "=", "!=", ">", ">=", "<", "<="}
        for f in filters
    )


def apply_date_fallback(canonical_plan: YamlDict, *, picker_start: str, picker_end: str, date_field: str = "call_date") -> YamlDict:
    filters = canonical_plan.get("filters", [])
    if has_date_filter(filters, date_field=date_field):
        return canonical_plan

    picker_start = (picker_start or "").strip()
    picker_end = (picker_end or "").strip()
    if not picker_start or not picker_end:
        return canonical_plan

    updated = dict(canonical_plan)
    updated_filters = list(filters)
    updated_filters.insert(0, {
        "field": date_field,
        "op": "between",
        "value": [picker_start, picker_end],
    })
    updated["filters"] = updated_filters
    return updated


def has_time_filter(filters: List[YamlDict]) -> bool:
    return any(
        isinstance(f, dict) and f.get("op") == "time_between"
        for f in filters
    )


def apply_time_fallback(canonical_plan: YamlDict, *, picker_time_start: str, picker_time_end: str, timestamp_field: str = "timestamp_utc") -> YamlDict:
    filters = canonical_plan.get("filters", [])
    if has_time_filter(filters):
        return canonical_plan

    picker_time_start = (picker_time_start or "").strip()
    picker_time_end = (picker_time_end or "").strip()
    if not picker_time_start or not picker_time_end:
        return canonical_plan

    updated = dict(canonical_plan)
    updated_filters = list(filters)
    updated_filters.append({
        "field": timestamp_field,
        "op": "time_between",
        "value": [picker_time_start, picker_time_end],
    })
    updated["filters"] = updated_filters
    return updated
