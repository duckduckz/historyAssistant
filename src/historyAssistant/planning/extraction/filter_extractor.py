from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .deterministic_filter_fallback import collect_fallback_filter_candidates
from .filter_value_resolver import dedupe_filters, normalize_room_code, resolve_filter_candidates
from .measure_threshold_extractor import extract_measure_threshold_filters

YamlDict = Dict[str, Any]

_AREA_OR_RE = re.compile(
    r"(?i)\b([0-9A-Z][A-Z0-9_]*(?:\s+[A-Z0-9_]+){0,2})\s+or\s+([0-9A-Z][A-Z0-9_]*(?:\s+[A-Z0-9_]+){0,2})\b"
)


def extract_dimension_filters(
    question: str,
    table_definition: YamlDict,
    *,
    active_table: str = "",
    detected_entities: Optional[YamlDict] = None,
    filter_candidates: Optional[List[YamlDict]] = None,
    normalized_question: str = "",
    client: Any = None,
) -> List[YamlDict]:
    """Resolve filters through provided candidates first, then deterministic fallback."""

    del active_table, normalized_question, client
    question = str(question or "")
    provided_filters = resolve_filter_candidates(filter_candidates or [], table_definition)
    if provided_filters:
        base = _with_area_or_fallback(question, table_definition, provided_filters)
    else:
        fallback_candidates = collect_fallback_filter_candidates(
            question,
            table_definition,
            detected_entities=detected_entities,
        )
        fallback_filters = resolve_filter_candidates(fallback_candidates, table_definition)
        base = _with_area_or_fallback(question, table_definition, fallback_filters)
    return _with_measure_thresholds(question, table_definition, base)


def _with_measure_thresholds(
    question: str,
    table_definition: YamlDict,
    filters: List[YamlDict],
) -> List[YamlDict]:
    """Append numeric threshold filters on measure fields (e.g. time_gap_minutes > 30).

    These coexist with dimension filters; a threshold for a field already filtered
    is skipped so an explicit dimension filter wins.
    """
    thresholds = extract_measure_threshold_filters(question, table_definition)
    if not thresholds:
        return filters
    existing_fields = {str(item.get("field") or "") for item in filters}
    additions = [item for item in thresholds if str(item.get("field") or "") not in existing_fields]
    return [*filters, *additions] if additions else filters


def _with_area_or_fallback(question: str, table_definition: YamlDict, filters: List[YamlDict]) -> List[YamlDict]:
    semantic_fields = table_definition.get("semantic_fields", {}) or {}
    area_field = semantic_fields.get("area")
    if area_field and not any(item.get("field") == area_field for item in filters):
        values = _extract_area_or_values(question)
        if values:
            filters = [*filters, {"field": area_field, "op": "in", "value": values}]
    return dedupe_filters(filters)


def _extract_area_or_values(question: str) -> List[str]:
    for match in _AREA_OR_RE.finditer(question or ""):
        left = match.group(1).strip()
        right = match.group(2).strip()
        if "_" in left or "_" in right:
            return [left, right]
    return []


__all__ = [
    "extract_dimension_filters",
    "normalize_room_code",
    "resolve_filter_candidates",
]
