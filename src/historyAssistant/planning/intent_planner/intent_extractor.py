from __future__ import annotations

from importlib import import_module
import re
from typing import Any, Dict, List, Optional

from historyAssistant.config import ConfigBundle
from historyAssistant.planning._text_utils import compact_text, dedupe_by_norm, norm_text
from historyAssistant.planning.extraction.filter_extractor import extract_dimension_filters

from .comparison_intent import (
    extract_comparison_slots,
    extract_main_period_from_front_loaded_compare,
)
from .intent_slots import IntentSlots, RankingSlots, SlotExtractionResult
from .value_filter_inference import infer_value_filters, remove_filtered_group_by

YamlDict = Dict[str, Any]

_temporal_parser = import_module("..extraction.temporal_parser", __package__)
_extract_dates = _temporal_parser.extract_dates
_extract_time_range = _temporal_parser.extract_time_range
_match_configured_date_alias = _temporal_parser.match_configured_date_alias

_ranking_signals = import_module("..extraction.ranking_signals", __package__)
_detect_ranking_signals = _ranking_signals.detect_ranking_signals

_chart_rules = import_module("..chart.chart_rules", __package__)
_should_plan_auto_chart = _chart_rules.should_plan_auto_chart


def extract_intent_slots(
    *,
    question: str,
    normalized_question: str,
    source_id: str,
    target_table: str,
    detected_entities: Optional[YamlDict],
    request_type: str,
    chart_suppressed: bool,
    bundle: ConfigBundle,
    client: Any = None,
) -> SlotExtractionResult:
    table_def = _table_definition(bundle, target_table)
    if not table_def:
        return SlotExtractionResult(False, reason=f"unknown_target_table:{target_table}")

    entities = {
        key: list(value) if isinstance(value, list) else value
        for key, value in (detected_entities or {}).items()
    }
    comparison = extract_comparison_slots(
        question,
        normalized_question,
        source_id=source_id,
        target_table=target_table,
        table_definition=table_def,
        date_rules_config=getattr(bundle, "date_rules", {}),
    )

    metric = _first_allowed_metric(entities.get("metrics"), bundle)
    if not metric and _should_use_source_default_count(question, normalized_question, comparison):
        metric = _default_count_metric_for_source(source_id, target_table, bundle)
    if not metric:
        return SlotExtractionResult(
            False,
            reason="metric_not_resolved",
            debug={"detected_entities": detected_entities or {}},
        )

    metric_cfg = (bundle.metric_catalog.get("metrics", {}) or {}).get(metric, {}) or {}
    aggregation = str(metric_cfg.get("aggregation") or "").strip() or None
    if aggregation == "raw_records":
        return SlotExtractionResult(False, reason=f"raw_metric_not_slot_planned:{metric}")

    date_rule, date_range, date_params = _resolve_main_date(
        question,
        normalized_question,
        comparison,
        getattr(bundle, "date_rules", {}),
    )
    if not date_rule and not date_range:
        date_rule = _first_detected_date_rule(entities, getattr(bundle, "date_rules", {}))
    if not date_rule and not date_range:
        date_rule = _date_rule_from_config_alias(
            question,
            normalized_question,
            getattr(bundle, "date_rules", {}),
        )
    bucket_interval, bucket_summary = _resolve_bucket_interval(entities, bundle)
    filters = _extract_filters(
        question,
        normalized_question,
        entities,
        bundle,
        target_table,
        client=client,
    )
    filters = _merge_filters(
        filters,
        infer_value_filters(
            question,
            bundle=bundle,
            target_table=target_table,
            existing_filters=filters,
        ),
    )
    group_by = _resolve_group_by(entities, comparison, table_def)
    metric_params: YamlDict = {}
    intersection = _detect_entity_intersection(
        question=question,
        bundle=bundle,
        target_table=target_table,
        table_def=table_def,
        group_by=group_by,
    )
    if intersection:
        condition_field = str(intersection["condition_field"])
        required_values = list(intersection["required_values"])
        filters = [
            f for f in filters
            if str(f.get("field") or "") != condition_field
            and not _filter_conflicts_with_intersection(f, required_values)
        ]
        filters.append({"field": condition_field, "op": "in", "value": required_values})
        group_by = [
            str(intersection["entity_field"]),
            condition_field,
        ]
        metric_params["_entity_intersection"] = intersection
    if comparison and comparison.comparison_type == "entity":
        filters = _merge_filters(filters, comparison.filters)
    if not intersection:
        group_by = remove_filtered_group_by(group_by, filters)

    ranking_signal = _detect_ranking_signals(question)
    ranking = RankingSlots(
        detected=ranking_signal.detected,
        direction=ranking_signal.direction,
        limit=ranking_signal.limit,
    )

    slots = IntentSlots(
        source_id=source_id,
        target_table=target_table,
        answer_shape="analytic_summary",
        request_type=request_type if request_type in {"qa", "chart"} else "qa",
        metric=metric,
        aggregation=aggregation,
        metric_params=metric_params,
        group_by=group_by,
        filters=filters,
        date_rule=date_rule or "none",
        date_range=date_range,
        date_params=date_params,
        time_range=(
            _extract_time_range(question, getattr(bundle, "date_rules", {}))
            or _extract_time_range(normalized_question, getattr(bundle, "date_rules", {}))
        ),
        comparison=comparison,
        ranking=ranking,
        bucket_interval=bucket_interval,
        bucket_summary=bucket_summary,
        confidence=0.9 if comparison or entities.get("metrics") else 0.78,
        extraction_path="deterministic_slots",
        reason=_slot_reason(metric, comparison),
    )
    return SlotExtractionResult(True, slots=slots, reason=slots.reason)


def _table_definition(bundle: ConfigBundle, target_table: str) -> YamlDict:
    return (bundle.table_registry.get("tables", {}) or {}).get(target_table, {}) or {}


def _first_allowed_metric(values: Any, bundle: ConfigBundle) -> Optional[str]:
    allowed = set((bundle.metric_catalog.get("metrics", {}) or {}).keys())
    for raw in values or []:
        value = str(raw)
        if value in allowed:
            return value
    return None


def _default_count_metric_for_source(source_id: str, target_table: str, bundle: ConfigBundle) -> Optional[str]:
    preferred = {
        "calls": "total_calls",
        "locations": "movement_event_count",
        "workflows": "presence_event_count",
    }.get(source_id)
    metrics = bundle.metric_catalog.get("metrics", {}) or {}
    if preferred in metrics:
        return preferred
    for metric_name, cfg in metrics.items():
        if cfg.get("table") == target_table and cfg.get("aggregation") in {"count", "cardinality"}:
            return str(metric_name)
    return None


def _should_use_source_default_count(question: str, normalized_question: str, comparison: Any) -> bool:
    text = f"{question or ''} {normalized_question or ''}".lower()
    count_words = ("call", "calls", "activity", "movement", "presence", "event", "events", "volume")
    return bool(comparison) or any(word in text for word in count_words)


def _resolve_main_date(
    question: str,
    normalized_question: str,
    comparison: Any,
    date_rules_config: Optional[YamlDict],
) -> tuple[Optional[str], Optional[YamlDict], Optional[YamlDict]]:
    if comparison and comparison.reason == "front_loaded_compare_and":
        front = extract_main_period_from_front_loaded_compare(
            question,
            normalized_question,
            date_rules_config,
        )
        if front[0]:
            return front
    return _extract_dates(
        " ".join(part for part in [question or "", normalized_question or ""] if part),
        date_rules_config,
    )


def _first_detected_date_rule(entities: YamlDict, date_rules_config: Optional[YamlDict]) -> Optional[str]:
    rules = _known_date_rule_names(date_rules_config)
    for raw in entities.get("date_rules", []) or []:
        value = str(raw).strip()
        if value and (not rules or value in rules):
            return value
    return None


def _date_rule_from_config_alias(
    question: str,
    normalized_question: str,
    date_rules_config: Optional[YamlDict],
) -> Optional[str]:
    text = f"{question or ''} {normalized_question or ''}"
    return _match_configured_date_alias(text, date_rules_config)


def _known_date_rule_names(date_rules_config: Optional[YamlDict]) -> set[str]:
    return {
        str(item.get("name") or "").strip()
        for item in (date_rules_config or {}).get("canonical_date_rules", []) or []
        if item.get("name")
    }


def _resolve_bucket_interval(entities: YamlDict, bundle: ConfigBundle) -> tuple[Optional[str], Optional[str]]:
    intervals = list((entities or {}).get("bucket_intervals", []) or [])
    if not intervals:
        return None, None

    interval_id = str(intervals[0]).strip()
    if not interval_id:
        return None, None

    bucket_cfg = (
        (bundle.vocabulary or {})
        .get("bucket_intervals", {})
        .get(interval_id, {})
    ) or {}
    default_summary = str(bucket_cfg.get("default_summary") or "avg").strip() or "avg"
    return interval_id, default_summary


def _extract_filters(
    question: str,
    normalized_question: str,
    entities: YamlDict,
    bundle: ConfigBundle,
    target_table: str,
    *,
    client: Any = None,
) -> List[YamlDict]:
    _ = client
    table_def = _table_definition(bundle, target_table)
    return extract_dimension_filters(
        question,
        table_def,
        active_table=target_table,
        detected_entities=entities,
        normalized_question=normalized_question,
    )


def _resolve_group_by(entities: YamlDict, comparison: Any, table_def: YamlDict) -> List[str]:
    allowed = set(table_def.get("dimensions", []) or [])
    group_by: List[str] = []
    if comparison and comparison.group_by:
        group_by.extend(comparison.group_by)
    for raw in entities.get("dimensions", []) or []:
        dim = str(raw)
        if dim in allowed and dim not in group_by:
            group_by.append(dim)
    return group_by[:2]


def _merge_filters(base: List[YamlDict], extra: List[YamlDict]) -> List[YamlDict]:
    merged = [
        filt for filt in base
        if str(filt.get("field") or "") not in {str(item.get("field") or "") for item in extra}
    ]
    merged.extend(extra)
    return merged


def _detect_entity_intersection(
    *,
    question: str,
    bundle: ConfigBundle,
    target_table: str,
    table_def: YamlDict,
    group_by: List[str],
) -> Optional[YamlDict]:
    text = str(question or "")
    if not re.search(r"(?i)\b(?:both|all|each of|every)\b", text):
        return None

    dim_maps = table_def.get("dimension_value_maps", {}) or {}
    best_field: Optional[str] = None
    best_values: List[str] = []
    for field in table_def.get("dimensions", []) or []:
        field = str(field)
        values = _known_dimension_values(dim_maps.get(field, {}) or {})
        if not values:
            continue
        matches = _mentioned_values(text, values)
        if len(matches) >= 2 and len(matches) > len(best_values):
            best_field = field
            best_values = matches

    if not best_field or len(best_values) < 2:
        return None

    entity_field = _entity_dimension_for_intersection(
        question=text,
        table_def=table_def,
        group_by=group_by,
        condition_field=best_field,
    )
    if not entity_field:
        return None

    return {
        "entity_field": entity_field,
        "condition_field": best_field,
        "required_values": best_values,
        "match": "all",
    }


def _entity_dimension_for_intersection(
    *,
    question: str,
    table_def: YamlDict,
    group_by: List[str],
    condition_field: str,
) -> Optional[str]:
    for field in group_by:
        if field != condition_field:
            return field

    dimensions = set(str(item) for item in table_def.get("dimensions", []) or [])
    text = question.lower()
    candidates = [
        ("personFirstName", ("patient", "patients", "person", "people", "who")),
        ("person_name", ("patient", "patients", "person", "people", "who")),
        ("initiator_person", ("nurse", "nurses", "staff", "caregiver", "who")),
        ("room_name", ("room", "rooms")),
        ("ward_name", ("ward", "wards")),
        ("ward", ("ward", "wards")),
    ]
    for field, words in candidates:
        if field in dimensions and any(word in text for word in words):
            return field
    return None


def _known_dimension_values(dim_info: YamlDict) -> List[str]:
    values: List[str] = []
    for value in dim_info.get("allowed_values", []) or []:
        text = str(value or "").strip()
        if text and text.lower() != "unknown":
            values.append(text)
    return sorted(set(values), key=len, reverse=True)


def _mentioned_values(question: str, values: List[str]) -> List[str]:
    text_norm = _norm_for_value(question)
    text_compact = _compact_for_value(question)
    matches: List[str] = []
    for value in values:
        value_norm = _norm_for_value(value)
        value_compact = _compact_for_value(value)
        if not value_norm:
            continue
        if (
            re.search(rf"(?<![a-z0-9]){re.escape(value_norm)}(?![a-z0-9])", text_norm)
            or (value_compact and value_compact in text_compact)
        ):
            matches.append(value)
    return _dedupe_preserve_order(matches)


def _filter_conflicts_with_intersection(filt: YamlDict, required_values: List[str]) -> bool:
    value = filt.get("value")
    values = value if isinstance(value, list) else [value]
    required_compact = [_compact_for_value(item) for item in required_values]
    for raw in values:
        value_compact = _compact_for_value(raw)
        if not value_compact:
            continue
        if any(value_compact in required or required in value_compact for required in required_compact):
            return True
    return False


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    return dedupe_by_norm(values)


def _norm_for_value(value: Any) -> str:
    return norm_text(value)


def _compact_for_value(value: Any) -> str:
    return compact_text(value)


def _slot_reason(metric: str, comparison: Any) -> str:
    if comparison:
        return f"deterministic_{comparison.comparison_type}_comparison:{metric}"
    return f"deterministic_metric:{metric}"
