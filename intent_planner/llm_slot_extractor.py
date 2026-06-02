from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from historyAssistant.config import ConfigBundle
from historyAssistant.planning._text_utils import parse_json_text as _parse_json_text_shared
from historyAssistant.llm.llm_client import call_json_prompt, get_default_model_name
from historyAssistant.planning.extraction.filter_extractor import resolve_filter_candidates
from historyAssistant.planning.extraction.temporal_parser import resolve_temporal_slots

from .intent_slots import ComparisonSlots, IntentSlots, RankingSlots, SlotExtractionResult

YamlDict = Dict[str, Any]

LLM_SLOT_SYSTEM_TEXT = """
You extract structured analytics intent slots for a hospital history assistant.
Return strict JSON only. Do not write Elasticsearch queries.

Use only the allowed metrics and dimensions provided by the user payload.
Use allowed_source_fields when the user asks to see exact raw fields, for example timestamps or identifiers.
Choose answer_shape:
- raw_records when the user wants exact source rows/events.
- distinct_values when the user asks which/who/what values and the answer should be a list grouped by one dimension.
- analytic_summary when the user wants a computed number, grouped summary, ranking, comparison, trend, or chartable aggregate.
Return one primary metric only. If the user asks for multiple metrics that cannot be represented as one plan, set usable=false and explain briefly.
If the question compares periods, fill comparison.type = "period" and put the second period in comparison.compare_to.
For "from A to B", "increase from A to B", or "decrease from A to B", use B as the main date_range/date_rule and A as comparison.compare_to.
For percentage increase/decrease questions, set metric_params.calculation = "percentage_change" and rank by field "percent_change" when the user asks for biggest/highest change.
If the question compares two entities or filter values, use one grouped filter with op "in" instead of compare_to.
For generic numeric questions, use metric "numeric_summary" when available and choose metric_params.field from allowed_measure_fields.
For numeric_summary, put the requested operation in metric_params.numeric_aggregation using one of avg, min, max, sum, median, or percentile.
If the user asks for p75/p90/p95, set metric_params.numeric_aggregation="percentile" and metric_params.percentile to 0.75, 0.90, or 0.95.
If a field or metric is unclear, return {"usable": false, "reason": "..."}.
""".strip()


def extract_llm_intent_slots(
    *,
    question: str,
    normalized_question: str,
    source_id: str,
    target_table: str,
    request_type: str,
    bundle: ConfigBundle,
    client: Any,
    model: Optional[str] = None,
) -> SlotExtractionResult:
    if client is None:
        return SlotExtractionResult(False, reason="llm_slot_client_missing")

    table_def = (bundle.table_registry.get("tables", {}) or {}).get(target_table, {}) or {}
    metrics = bundle.metric_catalog.get("metrics", {}) or {}
    dim_maps = table_def.get("dimension_value_maps", {}) or {}
    measure_maps = table_def.get("measure_field_maps", {}) or {}
    payload = {
        "question": question,
        "normalized_question": normalized_question,
        "source_id": source_id,
        "target_table": target_table,
        "allowed_metrics": sorted(metrics.keys()),
        "metric_descriptions": {
            name: {
                "description": cfg.get("description"),
                "aggregation": cfg.get("aggregation"),
                "field": cfg.get("field"),
            }
            for name, cfg in metrics.items()
            if isinstance(cfg, dict)
        },
        "allowed_dimensions": list(table_def.get("dimensions", []) or []),
        "allowed_source_fields": sorted(_allowed_source_fields(table_def)),
        "dimension_descriptions": {
            name: (dim_maps.get(name, {}) or {}).get("description")
            for name in table_def.get("dimensions", []) or []
        },
        "allowed_measure_fields": list(table_def.get("measure_fields", []) or []),
        "measure_field_descriptions": {
            name: {
                "description": (measure_maps.get(name, {}) or {}).get("description"),
                "unit": (measure_maps.get(name, {}) or {}).get("unit"),
                "allowed_aggregations": (measure_maps.get(name, {}) or {}).get("allowed_aggregations", []),
            }
            for name in table_def.get("measure_fields", []) or []
        },
        "allowed_filter_ops": ["=", "!=", ">", ">=", "<", "<=", "in", "between", "relative", "time_between"],
        "response_schema": {
            "usable": True,
            "answer_shape": "raw_records|distinct_values|analytic_summary",
            "metric": "one allowed metric",
            "aggregation": "for numeric_summary only: avg|min|max|sum|median|percentile",
            "metric_params": {
                "field": "required only for numeric_summary; one allowed_measure_field",
                "numeric_aggregation": "avg|min|max|sum|median|percentile",
                "percentile": 0.9,
            },
            "group_by": ["zero, one, or two allowed dimensions"],
            "filters": [{"field": "allowed dimension", "op": "=", "value": "value"}],
            "filters_detailed": [
                {
                    "field": "allowed dimension",
                    "op": "=|!=|>|>=|<|<=|in|between|relative|time_between",
                    "value": "resolved value or list",
                    "value_text": "the exact text span or user wording if useful",
                    "confidence": 0.0,
                }
            ],
            "source_fields": ["only for raw_records; field names from allowed_source_fields"],
            "narration_fields": {
                "requested_field": "one allowed_source_field/result field or null",
                "summary_field": "one allowed_source_field/result field or null",
                "display_fields": ["field names in useful display order"],
            },
            "order_by": {"field": "allowed field", "direction": "asc|desc"},
            "limit": 10,
            "projection_field": "only for distinct_values; one allowed dimension",
            "projection_mode": "single|list|null",
            "date_rule": "none|custom_range|rolling_days|this_month|last_month|calendar_week|last_calendar_week|last_7_days|yesterday",
            "date_range": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
            "date_params": {"days": 7},
            "time_range": {"start_time": "HH:MM", "end_time": "HH:MM"},
            "comparison": {
                "type": "period|entity|filter_value|null",
                "preferred_direction": "asc|desc — which direction is better for the metric in this question (e.g. lower response_time = asc, more calls = desc)",
                "compare_to": {
                    "date_rule": "custom_range",
                    "date_range": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
                    "filters": [],
                },
            },
            "ranking": {"detected": False, "direction": "desc", "limit": 10, "field": "metric_value|percent_change"},
            "bucket_interval": None,
            "bucket_summary": None,
            "confidence": 0.0,
            "reason": "short reason",
        },
    }

    try:
        raw = call_json_prompt(
            client,
            model=model or get_default_model_name(),
            system_text=LLM_SLOT_SYSTEM_TEXT,
            user_payload=payload,
            max_tokens=600,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, str):
            parsed = _parse_json_text(parsed)
    except Exception as exc:
        return SlotExtractionResult(False, reason=f"llm_slot_failed:{exc}")

    if not isinstance(parsed, dict) or not parsed.get("usable", False):
        return SlotExtractionResult(
            False,
            reason=str((parsed or {}).get("reason") or "llm_slot_not_usable"),
            debug={"raw_response": parsed},
        )

    metric = str(parsed.get("metric") or "")
    if metric not in metrics:
        return SlotExtractionResult(False, reason=f"llm_slot_metric_not_allowed:{metric}", debug={"raw_response": parsed})

    allowed_dimensions = set(table_def.get("dimensions", []) or [])
    group_by = [str(item) for item in parsed.get("group_by", []) or [] if str(item) in allowed_dimensions][:2]
    raw_filters = parsed.get("filters_detailed") if isinstance(parsed.get("filters_detailed"), list) else parsed.get("filters", [])
    filters = _valid_filters(raw_filters, allowed_dimensions, table_def)
    comparison = _comparison_from_llm(
        parsed.get("comparison"),
        allowed_dimensions,
        table_def,
        getattr(bundle, "date_rules", {}),
    )
    ranking = parsed.get("ranking") if isinstance(parsed.get("ranking"), dict) else {}
    metric_cfg = metrics.get(metric, {}) or {}
    metric_params = _valid_metric_params(parsed.get("metric_params"), metric, table_def, parsed.get("aggregation"))
    if metric == "numeric_summary" and not metric_params:
        return SlotExtractionResult(False, reason="llm_slot_numeric_summary_invalid_params", debug={"raw_response": parsed})
    answer_shape = str(parsed.get("answer_shape") or "analytic_summary").strip().lower()
    if answer_shape not in {"raw_records", "distinct_values", "analytic_summary"}:
        answer_shape = "analytic_summary"
    source_fields = _valid_source_fields(parsed.get("source_fields"), table_def)
    narration_fields = _valid_narration_fields(
        parsed.get("narration_fields"),
        table_def,
        group_by=group_by,
        source_fields=source_fields,
        answer_shape=answer_shape,
    )
    source_fields = _merge_source_fields_from_narration(
        source_fields,
        narration_fields,
        table_def,
        answer_shape=answer_shape,
    )
    order_by = _valid_order_by(parsed.get("order_by"), table_def)
    limit = parsed.get("limit") if isinstance(parsed.get("limit"), int) else None
    projection_field = parsed.get("projection_field")
    if projection_field not in allowed_dimensions:
        projection_field = None
    date_rule, date_range, date_params = resolve_temporal_slots(
        date_rule=parsed.get("date_rule") or "none",
        date_range=parsed.get("date_range") if isinstance(parsed.get("date_range"), dict) else None,
        date_params=parsed.get("date_params") if isinstance(parsed.get("date_params"), dict) else None,
        date_rules_config=getattr(bundle, "date_rules", {}),
    )

    slots = IntentSlots(
        source_id=source_id,
        target_table=target_table,
        answer_shape=answer_shape,
        request_type=request_type if request_type in {"qa", "chart"} else "qa",
        metric=metric,
        aggregation=str(metric_cfg.get("aggregation") or "") or None,
        metric_params=metric_params,
        group_by=group_by,
        filters=filters,
        source_fields=source_fields,
        narration_fields=narration_fields,
        order_by=order_by,
        limit=limit,
        projection_field=projection_field,
        projection_mode=parsed.get("projection_mode") if isinstance(parsed.get("projection_mode"), str) else None,
        date_rule=date_rule,
        date_range=date_range,
        date_params=date_params,
        time_range=parsed.get("time_range") if isinstance(parsed.get("time_range"), dict) else None,
        comparison=comparison,
        ranking=RankingSlots(
            detected=bool(ranking.get("detected")),
            direction=ranking.get("direction"),
            limit=ranking.get("limit") if isinstance(ranking.get("limit"), int) else None,
            field=ranking.get("field") if isinstance(ranking.get("field"), str) else None,
        ),
        bucket_interval=parsed.get("bucket_interval"),
        bucket_summary=parsed.get("bucket_summary"),
        confidence=float(parsed.get("confidence") or 0.65),
        extraction_path="llm_slots",
        reason=str(parsed.get("reason") or "llm_slots"),
    )
    return SlotExtractionResult(True, slots=slots, reason=slots.reason, debug={"raw_response": parsed})


def _parse_json_text(raw: str) -> YamlDict:
    return _parse_json_text_shared(raw)


def _valid_filters(raw_filters: Any, allowed_dimensions: set[str], table_def: YamlDict) -> List[YamlDict]:
    candidates: List[YamlDict] = []
    if not isinstance(raw_filters, list):
        return []
    for item in raw_filters:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "")
        op = str(item.get("op") or "")
        if field in allowed_dimensions and op:
            candidate: YamlDict = {
                "field": field,
                "op": op,
                "value": item.get("value"),
                "source": "llm_slots",
            }
            if item.get("value_text") not in (None, ""):
                candidate["value_text"] = item.get("value_text")
            if isinstance(item.get("confidence"), (int, float)) and not isinstance(item.get("confidence"), bool):
                candidate["confidence"] = float(item.get("confidence"))
            candidates.append(candidate)
    return resolve_filter_candidates(candidates, table_def)


def _valid_source_fields(raw_fields: Any, table_def: YamlDict) -> List[str]:
    if not isinstance(raw_fields, list):
        return []
    allowed = _allowed_source_fields(table_def)
    fields: List[str] = []
    for raw in raw_fields:
        field = str(raw)
        if field in allowed and field not in fields:
            fields.append(field)
    return fields


def _allowed_source_fields(table_def: YamlDict) -> set[str]:
    allowed = set(str(item) for item in (table_def.get("dimensions", []) or []))
    allowed.update(str(item) for item in (table_def.get("identifiers", []) or []))
    allowed.update(str(item) for item in _as_list(table_def.get("time_field")))
    allowed.update(str(item) for item in _as_list(table_def.get("timestamp_fields")))
    allowed.update(str(item) for item in _as_list(table_def.get("date_field")))
    allowed.update(str(item) for item in _as_list(table_def.get("timestamp_field")))
    allowed.update(str(item) for item in _as_list(table_def.get("end_timestamp_field")))
    allowed.update(str(item) for item in (table_def.get("measure_fields", []) or []))
    allowed.update(str(item) for item in (table_def.get("event_fields", []) or []))
    return {field for field in allowed if field}


def _as_list(value: Any) -> List[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    return [value]


def _valid_narration_fields(
    raw_fields: Any,
    table_def: YamlDict,
    *,
    group_by: List[str],
    source_fields: List[str],
    answer_shape: str,
) -> YamlDict:
    if not isinstance(raw_fields, dict):
        return {}
    allowed = _allowed_source_fields(table_def)
    allowed.update(str(item) for item in group_by or [])
    allowed.update(str(item) for item in source_fields or [])
    allowed.add("metric_value")

    requested_field = _valid_optional_field(raw_fields.get("requested_field"), allowed)
    summary_field = _valid_optional_field(raw_fields.get("summary_field"), allowed)
    display_fields: List[str] = []
    for raw in raw_fields.get("display_fields", []) or []:
        field = str(raw)
        if field in allowed and field not in display_fields:
            display_fields.append(field)

    if answer_shape == "raw_records" and requested_field and requested_field not in display_fields:
        display_fields.insert(0, requested_field)

    result: YamlDict = {}
    if requested_field:
        result["requested_field"] = requested_field
    if summary_field:
        result["summary_field"] = summary_field
    if display_fields:
        result["display_fields"] = display_fields
    return result


def _valid_optional_field(value: Any, allowed: set[str]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    field = str(value)
    return field if field in allowed else None


def _merge_source_fields_from_narration(
    source_fields: List[str],
    narration_fields: YamlDict,
    table_def: YamlDict,
    *,
    answer_shape: str,
) -> List[str]:
    if answer_shape != "raw_records":
        return source_fields
    allowed = _allowed_source_fields(table_def)
    merged = list(source_fields or [])
    for field in [
        narration_fields.get("requested_field"),
        narration_fields.get("summary_field"),
        *(narration_fields.get("display_fields") or []),
    ]:
        field_name = str(field or "")
        if field_name in allowed and field_name not in merged:
            merged.append(field_name)
    return merged


def _valid_order_by(raw_order: Any, table_def: YamlDict) -> Optional[YamlDict]:
    if not isinstance(raw_order, dict):
        return None
    field = str(raw_order.get("field") or "")
    direction = str(raw_order.get("direction") or "").lower()
    allowed = set(table_def.get("dimensions", []) or [])
    allowed.update(table_def.get("identifiers", []) or [])
    allowed.update(table_def.get("time_field", []) or [])
    allowed.update(table_def.get("timestamp_fields", []) or [])
    allowed.update(table_def.get("measure_fields", []) or [])
    if field not in allowed or direction not in {"asc", "desc"}:
        return None
    return {"field": field, "direction": direction}


def _valid_metric_params(raw_params: Any, metric: str, table_def: YamlDict, raw_aggregation: Any = None) -> YamlDict:
    if metric != "numeric_summary":
        return dict(raw_params or {}) if isinstance(raw_params, dict) else {}
    if not isinstance(raw_params, dict):
        raw_params = {}

    measure_maps = table_def.get("measure_field_maps", {}) or {}
    measure_fields = set(table_def.get("measure_fields", []) or [])
    field = str(raw_params.get("field") or raw_params.get("measure_field") or "").strip()
    if field not in measure_fields:
        return {}

    allowed_aggs = set((measure_maps.get(field, {}) or {}).get("allowed_aggregations", []) or [])
    numeric_aggregation = str(
        raw_params.get("numeric_aggregation")
        or raw_params.get("aggregation")
        or raw_aggregation
        or ""
    ).strip().lower()
    if numeric_aggregation not in allowed_aggs:
        return {}

    params: YamlDict = {
        "field": field,
        "numeric_aggregation": numeric_aggregation,
    }
    if numeric_aggregation == "percentile":
        percentile = raw_params.get("percentile")
        if isinstance(percentile, (int, float)) and not isinstance(percentile, bool):
            if percentile > 1:
                percentile = percentile / 100.0
            if 0 < percentile <= 1:
                params["percentile"] = float(percentile)
    return params


def _comparison_from_llm(
    raw: Any,
    allowed_dimensions: set[str],
    table_def: YamlDict,
    date_rules_config: Optional[YamlDict] = None,
) -> Optional[ComparisonSlots]:
    if not isinstance(raw, dict):
        return None
    comparison_type = str(raw.get("type") or raw.get("comparison_type") or "").strip()
    if comparison_type in {"", "null", "none"}:
        return None
    compare_to = raw.get("compare_to") if isinstance(raw.get("compare_to"), dict) else {}
    filters = _valid_filters(compare_to.get("filters", []), allowed_dimensions, table_def)
    date_rule, date_range, date_params = resolve_temporal_slots(
        date_rule=compare_to.get("date_rule"),
        date_range=compare_to.get("date_range") if isinstance(compare_to.get("date_range"), dict) else None,
        date_params=compare_to.get("date_params") if isinstance(compare_to.get("date_params"), dict) else None,
        date_rules_config=date_rules_config,
    )
    raw_direction = str(raw.get("preferred_direction") or "").strip().lower()
    preferred_direction = raw_direction if raw_direction in {"asc", "desc"} else None
    return ComparisonSlots(
        comparison_type=comparison_type,
        date_rule=date_rule,
        date_range=date_range,
        date_params=date_params,
        filters=filters,
        preferred_direction=preferred_direction,
        reason="llm_slots",
    )
