from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from historyAssistant.config import YamlDict
from historyAssistant.execution.result.display_labels import FIELD_LABELS
from historyAssistant.execution.result.narration_config import get_narration_table_config, get_narration_table_definition
from historyAssistant.llm.llm_client import call_json_prompt, get_default_model_name
from historyAssistant.planning._text_utils import parse_json_text


SYSTEM_TEXT = """
You select fields for narrating database results in a hospital-history analytics assistant.
Return strict JSON only.

Use only fields listed in available_fields. Never invent field names.

Choose:
- requested_field when the user asks for one specific attribute of a matching record, otherwise null.
- summary_field when the user asks "who", "which", "where", "when", or similar and a distinct-value summary would answer better than listing rows, otherwise null.
- display_fields as the most useful fields to show first for the user's question and result records.

Do not write prose. Do not choose filters, metrics, dates, or Elasticsearch.
If the question is best answered by the normal table/list, set requested_field and summary_field to null.
""".strip()


@dataclass(frozen=True)
class NarrationFieldSelection:
    used: bool
    requested_field: Optional[str] = None
    requested_label: Optional[str] = None
    summary_field: Optional[str] = None
    summary_label: Optional[str] = None
    display_fields: tuple[str, ...] = ()
    confidence: float = 0.0
    reason: str = ""
    debug: Optional[YamlDict] = None

    def to_dict(self) -> YamlDict:
        return {
            "used": self.used,
            "requested_field": self.requested_field,
            "requested_label": self.requested_label,
            "summary_field": self.summary_field,
            "summary_label": self.summary_label,
            "display_fields": list(self.display_fields),
            "confidence": self.confidence,
            "reason": self.reason,
            "debug": dict(self.debug or {}),
        }


def select_narration_fields(
    *,
    question: str,
    execution_result: YamlDict,
    client: Any,
    model: Optional[str] = None,
    min_confidence: float = 0.55,
) -> NarrationFieldSelection:
    if client is None:
        return NarrationFieldSelection(False, reason="client_missing")

    records = execution_result.get("records") or []
    if not records:
        return NarrationFieldSelection(False, reason="no_records")

    available_fields = _available_fields(execution_result, records)
    if not available_fields:
        return NarrationFieldSelection(False, reason="no_available_fields")

    payload = {
        "question": question,
        "result_type": execution_result.get("result_type"),
        "metric": (execution_result.get("plan") or {}).get("metric"),
        "available_fields": available_fields,
        "records_sample": records[:3],
        "record_count": execution_result.get("total_count", len(records)),
        "response_schema": {
            "requested_field": "field name or null",
            "summary_field": "field name or null",
            "display_fields": ["field names in desired order"],
            "confidence": 0.0,
            "reason": "short reason",
        },
    }

    try:
        raw = call_json_prompt(
            client,
            model=model or get_default_model_name(),
            system_text=SYSTEM_TEXT,
            user_payload=payload,
            max_tokens=350,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, str):
            parsed = parse_json_text(parsed)
    except Exception as exc:
        return NarrationFieldSelection(False, reason=f"selector_failed:{exc}", debug={})

    if not isinstance(parsed, dict):
        return NarrationFieldSelection(False, reason="selector_bad_response", debug={"raw_response": parsed})

    allowed = set(available_fields.keys())
    requested_field = _valid_optional_field(parsed.get("requested_field"), allowed)
    summary_field = _valid_optional_field(parsed.get("summary_field"), allowed)
    display_fields = tuple(
        field
        for field in parsed.get("display_fields", [])
        if isinstance(field, str) and field in allowed
    )

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < min_confidence:
        return NarrationFieldSelection(
            False,
            confidence=confidence,
            reason="selector_low_confidence",
            debug={"raw_response": parsed},
        )

    if not requested_field and not summary_field and not display_fields:
        return NarrationFieldSelection(
            False,
            confidence=confidence,
            reason="selector_empty_choice",
            debug={"raw_response": parsed},
        )

    return NarrationFieldSelection(
        True,
        requested_field=requested_field,
        requested_label=_label_for_field(requested_field, available_fields) if requested_field else None,
        summary_field=summary_field,
        summary_label=_label_for_field(summary_field, available_fields) if summary_field else None,
        display_fields=display_fields,
        confidence=confidence,
        reason=str(parsed.get("reason") or "llm_field_selection"),
        debug={"raw_response": parsed},
    )


def select_narration_fields_from_intent_packet(
    *,
    execution_result: YamlDict,
) -> NarrationFieldSelection:
    records = execution_result.get("records") or []
    if not records:
        return NarrationFieldSelection(False, reason="no_records")

    available_fields = _available_fields(execution_result, records)
    if not available_fields:
        return NarrationFieldSelection(False, reason="no_available_fields")

    plan = execution_result.get("canonical_plan") or execution_result.get("plan") or {}
    intent_packet = plan.get("_intent_packet") if isinstance(plan.get("_intent_packet"), dict) else {}
    field_selection = plan.get("_narration_fields") if isinstance(plan.get("_narration_fields"), dict) else {}
    if not field_selection:
        field_selection = intent_packet.get("field_selection") if isinstance(intent_packet.get("field_selection"), dict) else {}
    if not field_selection:
        return NarrationFieldSelection(False, reason="intent_packet_missing_fields")

    allowed = set(available_fields.keys())
    requested_field = _valid_optional_field(field_selection.get("requested_field"), allowed)
    summary_field = _valid_optional_field(field_selection.get("summary_field"), allowed)
    display_fields = tuple(
        field
        for field in field_selection.get("display_fields", []) or []
        if isinstance(field, str) and field in allowed
    )
    if not requested_field and not summary_field and not display_fields:
        return NarrationFieldSelection(False, reason="intent_packet_fields_not_in_result")

    return NarrationFieldSelection(
        True,
        requested_field=requested_field,
        requested_label=_label_for_field(requested_field, available_fields) if requested_field else None,
        summary_field=summary_field,
        summary_label=_label_for_field(summary_field, available_fields) if summary_field else None,
        display_fields=display_fields,
        confidence=1.0,
        reason="intent_packet_field_selection",
        debug={"source": "canonical_plan._intent_packet"},
    )


def _available_fields(execution_result: YamlDict, records: List[YamlDict]) -> YamlDict:
    table_def = get_narration_table_definition(execution_result)
    dim_maps = table_def.get("dimension_value_maps", {}) or {}
    measure_maps = table_def.get("measure_field_maps", {}) or {}
    record_fields = set().union(*(record.keys() for record in records[:5]))
    fields: YamlDict = {}
    for field in sorted(record_fields):
        info = dim_maps.get(field) or measure_maps.get(field) or {}
        fields[str(field)] = {
            "label": _display_label(field),
            "description": info.get("description") or _display_label(field),
            "field_type": info.get("field_type"),
        }
    return fields


def _valid_optional_field(value: Any, allowed: set[str]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    field = str(value)
    return field if field in allowed else None


def _label_for_field(field: str, available_fields: YamlDict) -> str:
    return str((available_fields.get(field) or {}).get("label") or _display_label(field))


def _display_label(field: Any) -> str:
    field_name = str(field)
    return FIELD_LABELS.get(field_name, field_name.replace("_", " ").strip().title())


def display_fields_from_selection(selection: Optional[NarrationFieldSelection]) -> List[str]:
    if not selection or not selection.used:
        return []
    return list(selection.display_fields)


__all__ = [
    "NarrationFieldSelection",
    "display_fields_from_selection",
    "select_narration_fields",
    "select_narration_fields_from_intent_packet",
]
