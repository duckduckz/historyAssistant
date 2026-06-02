from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from historyAssistant.config import ConfigBundle
from historyAssistant.planning._text_utils import dedupe
from historyAssistant.planning.extraction.filter_extractor import extract_dimension_filters

from .answer_shape import DISTINCT_VALUES, RAW_RECORDS
from .intent_slots import IntentSlots, RankingSlots, SlotExtractionResult

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class ProjectionSelection:
    used: bool
    answer_shape: Optional[str] = None
    projection_field: Optional[str] = None
    projection_mode: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    debug: YamlDict = field(default_factory=dict)

    def to_dict(self) -> YamlDict:
        return {
            "used": self.used,
            "answer_shape": self.answer_shape,
            "projection_field": self.projection_field,
            "projection_mode": self.projection_mode,
            "confidence": self.confidence,
            "reason": self.reason,
            **dict(self.debug),
        }

_LIMIT_PATTERNS = [
    re.compile(r"(?i)\b(?:first|top)\s+(\d+)\b"),
    re.compile(r"(?i)\b(?:latest|most\s+recent|last|recent)\s+(\d+)\s+(?:calls?|records?|entries?|results?)\b"),
    re.compile(r"(?i)\b(\d+)\s+(?:most\s+recent|latest|newest)\s+(?:calls?|records?|entries?|results?)?\b"),
    re.compile(r"(?i)\b(?:show|give|list|display|get)\s+(?:me\s+)?(\d+)\s+(?:calls?|records?|entries?|results?)\b"),
    re.compile(r"(?i)\blimit\s+(?:to\s+)?(\d+)\b"),
]
_MAX_LIMIT = 500
_FIRST_RECORD_RE = re.compile(r"(?i)\b(?:first|earliest|oldest)\s+(?:matching\s+)?(?:call|record|entry|event)\b")
_LATEST_RECORD_RE = re.compile(r"(?i)\b(?:latest|most recent|last|newest)\s+(?:matching\s+)?(?:call|record|entry|event)\b")
_LIST_REQUEST_RE = re.compile(r"(?i)\b(?:show|list|give|display|all)\b")


def extract_record_intent_slots(
    *,
    question: str,
    normalized_question: str,
    source_id: str,
    target_table: str,
    detected_entities: Optional[YamlDict],
    request_type: str,
    bundle: ConfigBundle,
    client: Any = None,
    requested_answer_shape: Optional[str] = None,
) -> SlotExtractionResult:
    from importlib import import_module

    temporal_parser = import_module("..extraction.temporal_parser", __package__)
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(target_table, {}) or {}
    if not table_def:
        return SlotExtractionResult(False, reason=f"unknown_target_table:{target_table}")

    filters = extract_dimension_filters(
        question,
        table_def,
        active_table=target_table,
        detected_entities=detected_entities,
        normalized_question=normalized_question,
        client=client,
    )
    date_rule, date_range, date_params = temporal_parser.extract_dates(question, getattr(bundle, "date_rules", {}))
    if date_rule is None and detected_entities:
        rules = detected_entities.get("date_rules", []) or []
        if rules:
            date_rule = rules[0]
    has_shift_filter = any(f.get("field") == "shift" for f in filters)
    time_range = temporal_parser.extract_time_range(
        question,
        getattr(bundle, "date_rules", {}),
        skip_day_part=has_shift_filter,
    )
    limit = extract_limit(question)
    order_by = _extract_single_record_order(question, table_def)
    if order_by and limit is None:
        limit = 1

    filter_fields = {f.get("field") for f in filters if f.get("field")}
    projection_selection = _select_projection(
        question=question,
        normalized_question=normalized_question,
        target_table=target_table,
        bundle=bundle,
        client=client,
        filter_fields=filter_fields,
        detected_entities=detected_entities,
        requested_answer_shape=requested_answer_shape,
    )
    projection_field, projection_mode = (None, None)
    if projection_selection.answer_shape == DISTINCT_VALUES:
        projection_field = projection_selection.projection_field
        projection_mode = projection_selection.projection_mode or "list"
    if (
        projection_selection.answer_shape != RAW_RECORDS
        and not projection_field
        and _LIST_REQUEST_RE.search(question or "")
    ):
        projection_field, projection_mode = _projection_from_detected_dimensions(
            detected_entities=detected_entities,
            table_def=table_def,
            filter_fields=filter_fields,
        )

    if projection_field and projection_mode in {"list", "single"}:
        metric = _default_count_metric_for_lookup(bundle)
        cfg = (bundle.metric_catalog.get("metrics", {}) or {}).get(metric, {}) or {}
        return SlotExtractionResult(
            True,
            slots=IntentSlots(
                source_id=source_id,
                target_table=target_table,
                answer_shape=DISTINCT_VALUES,
                request_type=request_type if request_type in {"qa", "chart"} else "qa",
                metric=metric,
                aggregation=cfg.get("aggregation") or "count",
                filters=filters,
                group_by=[projection_field],
                limit=limit or 50,
                projection_field=projection_field,
                projection_mode=projection_mode,
                date_rule=date_rule or "none",
                date_range=date_range,
                date_params=date_params,
                time_range=time_range,
                confidence=1.0,
                extraction_path="deterministic_record_slots",
                reason=projection_selection.reason or "deterministic_distinct_values",
            ),
            reason=projection_selection.reason or "deterministic_distinct_values",
            debug={"projection_selection": projection_selection.to_dict()},
        )

    metric = _detail_metric(bundle)
    metric_cfg = (bundle.metric_catalog.get("metrics", {}) or {}).get(metric, {}) or {}
    default_order = metric_cfg.get("default_order_by") if isinstance(metric_cfg.get("default_order_by"), dict) else None
    default_limit = metric_cfg.get("default_limit")
    if order_by is None and default_order:
        order_by = dict(default_order)
    if limit is None and isinstance(default_limit, int):
        limit = default_limit

    return SlotExtractionResult(
        True,
        slots=IntentSlots(
            source_id=source_id,
            target_table=target_table,
            answer_shape=RAW_RECORDS,
            request_type="qa",
            metric=metric,
            aggregation="raw_records",
            filters=filters,
            group_by=[],
            source_fields=_default_source_fields(table_def, question, filters),
            order_by=order_by,
            limit=limit,
            ranking=RankingSlots(detected=False),
            date_rule=date_rule or "none",
            date_range=date_range,
            date_params=date_params,
            time_range=time_range,
            confidence=1.0,
            extraction_path="deterministic_record_slots",
            reason="deterministic_raw_records",
        ),
        reason="deterministic_raw_records",
        debug={"projection_selection": projection_selection.to_dict()},
    )


def extract_limit(question: str) -> Optional[int]:
    for pattern in _LIMIT_PATTERNS:
        match = pattern.search(question or "")
        if match:
            limit = int(match.group(1))
            if 1 <= limit <= _MAX_LIMIT:
                return limit
            if limit > _MAX_LIMIT:
                return _MAX_LIMIT
    return None


def _extract_single_record_order(question: str, table_def: YamlDict) -> Optional[YamlDict]:
    timestamp = str(table_def.get("timestamp_field") or (table_def.get("timestamp_fields") or ["timestamp_utc"])[0])
    if _FIRST_RECORD_RE.search(question or ""):
        return {"field": timestamp, "direction": "asc"}
    if _LATEST_RECORD_RE.search(question or ""):
        return {"field": timestamp, "direction": "desc"}
    return None


def _select_projection(
    *,
    question: str,
    normalized_question: str,
    target_table: str,
    bundle: ConfigBundle,
    client: Any,
    filter_fields: set,
    detected_entities: Optional[YamlDict],
    requested_answer_shape: Optional[str],
) -> ProjectionSelection:
    _ = (normalized_question, client)

    if requested_answer_shape == RAW_RECORDS:
        return ProjectionSelection(
            True,
            answer_shape=RAW_RECORDS,
            confidence=0.95,
            reason="answer_shape_raw_records",
            debug={},
        )

    _, table_def = _active_table(bundle)
    projection_field, projection_mode = _projection_from_detected_dimensions(
        detected_entities=detected_entities,
        table_def=table_def,
        filter_fields=filter_fields,
    )
    if projection_field:
        return ProjectionSelection(
            True,
            answer_shape=DISTINCT_VALUES,
            projection_field=projection_field,
            projection_mode=projection_mode or "list",
            confidence=0.75,
            reason="detected_dimension_projection",
            debug={},
        )

    return ProjectionSelection(False, reason="projection_not_resolved", debug={})


def _projection_from_detected_dimensions(
    *,
    detected_entities: Optional[YamlDict],
    table_def: YamlDict,
    filter_fields: set,
) -> Tuple[Optional[str], Optional[str]]:
    allowed = set(table_def.get("dimensions", []) or [])
    for raw in (detected_entities or {}).get("dimensions", []) or []:
        field = str(raw or "").strip()
        if field and field in allowed and field not in filter_fields:
            return field, "list"
    return None, None


def _detail_metric(bundle: ConfigBundle) -> str:
    metrics = bundle.metric_catalog.get("metrics", {}) or {}
    for name, cfg in metrics.items():
        if cfg.get("aggregation") == "raw_records":
            return str(name)
    return "call_details"


def _default_count_metric_for_lookup(bundle: ConfigBundle) -> str:
    metrics = bundle.metric_catalog.get("metrics", {}) or {}
    supported_count_aggs = {"count", "cardinality", "count_threshold"}
    for preferred in ("movement_event_count", "location_update_count", "presence_event_count", "total_calls"):
        cfg = metrics.get(preferred)
        if cfg and cfg.get("aggregation") in supported_count_aggs:
            return preferred
    for name, cfg in metrics.items():
        if cfg.get("aggregation") in supported_count_aggs:
            return str(name)
    return "total_calls"


def _default_source_fields(table_def: YamlDict, question: str, filters: List[YamlDict]) -> List[str]:
    configured = list(table_def.get("default_source_fields", []) or [])
    if configured:
        fields = configured
    else:
        fields = list(table_def.get("identifiers", []) or [])[:2]
        fields += list(table_def.get("time_field", []) or [])[:2]
        fields += list(table_def.get("dimensions", []) or [])[:4]

    for filt in filters:
        field = str(filt.get("field") or "").strip()
        if field:
            fields.append(field)
    return _dedupe(fields)


def _active_table(bundle: ConfigBundle) -> Tuple[str, YamlDict]:
    tables = bundle.table_registry.get("tables", {}) or {}
    active_table = next(iter(tables.keys()), "historyCalls")
    return active_table, tables.get(active_table, {}) or {}


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    return [
        str(item).strip()
        for item in raw_values
        if str(item).strip()
    ]


def _dedupe(values: List[str]) -> List[str]:
    return dedupe(values)


__all__ = [
    "DISTINCT_VALUES",
    "extract_limit",
    "extract_record_intent_slots",
]
