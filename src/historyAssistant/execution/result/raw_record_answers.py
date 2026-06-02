from __future__ import annotations

from typing import Any, Dict, List, Optional

from historyAssistant.execution.result.narration_config import get_narration_table_definition
from historyAssistant.execution.result.narration_field_selector import NarrationFieldSelection

YamlDict = Dict[str, Any]


_STRUCTURAL_METRIC_COLUMNS = {"metric_value"}


def detect_lookup_projection(question: str, records: list[YamlDict]) -> tuple[str | None, str]:
    if not records:
        return None, ""

    candidate_fields = [
        field_name
        for field_name in records[0].keys()
        if field_name not in _STRUCTURAL_METRIC_COLUMNS
    ]
    if len(candidate_fields) != 1:
        return None, ""

    field_name = candidate_fields[0]
    label = field_name.replace("_", " ").strip().title()
    return field_name, label


def format_raw_record_list(
    records: list[YamlDict],
    total_count: int,
    preview_count: int,
    *,
    display_fields: list[str] | None = None,
) -> str:
    from historyAssistant.execution.result.deterministic_narration import format_value

    field_order = (
        [(key, key.replace("_", " ").strip().title()) for key in display_fields if records and key in records[0]]
        if display_fields
        else [(key, key.replace("_", " ").strip().title()) for key in records[0].keys()] if records else []
    )

    lines = [f"Found {total_count} matching record(s).", "", f"Showing the first {preview_count}:"]
    for idx, record in enumerate(records[:preview_count], start=1):
        title = record.get("call_id")
        title_text = f"Record {idx}" if title in (None, "") else f"Record {idx}: {title}"
        lines.append(f"{idx}. {title_text}")
        for key, label in field_order:
            value = record.get(key)
            if key == "call_id" or value in (None, "", "unknown"):
                continue
            lines.append(f"   {label}: {format_value(value)}")
        lines.append("")

    if total_count > preview_count:
        lines.append(
            f"There are {total_count} matching records in total. If you want the full result set, use the download/export option."
        )
    return "\n".join(lines).rstrip()


def detect_requested_record_field(question: str, record: YamlDict) -> tuple[str | None, str]:
    return None, ""


def needs_full_result_download(execution_result: YamlDict) -> bool:
    if execution_result.get("result_type") != "raw_records":
        return False
    if (execution_result.get("plan") or {}).get("metric") == "last_known_location":
        return False
    total_count = int(execution_result.get("total_count", 0) or 0)
    shown_count = len(execution_result.get("records", []) or [])
    return total_count > shown_count > 0


def _clean_display_value(value: Any) -> Optional[str]:
    if value in (None, "", "unknown"):
        return None
    return str(value)


def _unique_values(records: List[YamlDict], field: str) -> List[str]:
    seen: set[str] = set()
    values: List[str] = []
    for record in records:
        value = _clean_display_value(record.get(field))
        if value is None or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _format_value_list(values: List[str], *, max_items: int = 20) -> str:
    shown = values[:max_items]
    text = ", ".join(shown)
    if len(values) > max_items:
        text += f", and {len(values) - max_items} more"
    return text


def _requested_summary_field(question: str, records: List[YamlDict]) -> tuple[Optional[str], str]:
    if not records:
        return None, ""
    return _schema_summary_field(question, records)


def _schema_summary_field(question: str, records: List[YamlDict]) -> tuple[Optional[str], str]:
    available = set().union(*(record.keys() for record in records[:10]))
    table_def = get_narration_table_definition({"records": records})
    semantic_fields = table_def.get("semantic_fields") or {}
    text = str(question or "").lower()

    if not _has_summary_intent(text):
        return None, ""

    if _asks_for_entity(text):
        for semantic_name, field_name in semantic_fields.items():
            field = str(field_name)
            if field in available and semantic_name in {"person", "person_id"}:
                return field, "person"

    for semantic_name, field_name in semantic_fields.items():
        field = str(field_name)
        if field in available and _mentions(text, str(semantic_name)):
            return field, _human_label(semantic_name)

    field_scores: list[tuple[int, str]] = []
    table_fields = _field_metadata(table_def)
    for field in available:
        haystack = " ".join([
            field,
            _human_label(field),
            str((table_fields.get(field) or {}).get("description") or ""),
        ]).lower()
        score = sum(1 for token in _tokens(text) if len(token) > 2 and token in haystack)
        if score:
            field_scores.append((score, field))
    if field_scores:
        field_scores.sort(key=lambda item: (-item[0], item[1]))
        field = field_scores[0][1]
        return field, _human_label(field)
    return None, ""


def _field_metadata(table_def: YamlDict) -> YamlDict:
    metadata: YamlDict = {}
    metadata.update(table_def.get("dimension_value_maps") or {})
    metadata.update(table_def.get("measure_field_maps") or {})
    return metadata


def _mentions(text: str, value: str) -> bool:
    words = {_singular(token) for token in _tokens(value)}
    if not words:
        return False
    text_words = {_singular(token) for token in _tokens(text)}
    return bool(words & text_words)


def _asks_for_entity(text: str) -> bool:
    words = {_singular(token) for token in _tokens(text)}
    return bool(words & {"who", "which", "patient", "person", "people", "nurse", "staff", "user"})


def _has_summary_intent(text: str) -> bool:
    words = {_singular(token) for token in _tokens(text)}
    return bool(words & {"who", "which", "where", "when", "what"})


def _tokens(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
    return [token for token in normalized.split() if token]


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _human_label(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text or "value"


def raw_record_summary_answer(
    question: str,
    execution_result: YamlDict,
    *,
    field_selection: NarrationFieldSelection | None = None,
) -> Optional[str]:
    if execution_result.get("error"):
        return None
    if execution_result.get("result_type") != "raw_records":
        return None
    if execution_result.get("empty"):
        return None

    records = execution_result.get("records", []) or []
    if field_selection and field_selection.used and field_selection.summary_field:
        field = field_selection.summary_field
        label = field_selection.summary_label or field.replace("_", " ").strip().title()
    else:
        field, label = _requested_summary_field(question, records)
    if not field:
        return None

    values = _unique_values(records, field)
    if not values:
        return None

    total_count = int(execution_result.get("total_count", len(records)) or 0)
    shown_count = len(records)
    plural = label if len(values) == 1 else f"{label}s"
    lines = [
        f"Found {len(values)} distinct {plural} across {total_count} matching record(s): {_format_value_list(values)}."
    ]
    if total_count > shown_count > 0:
        lines.append(
            f"The table below shows the first {shown_count} matching records; use the download option for all records."
        )
    elif total_count > 0:
        lines.append("The table below contains the matching records.")
    return "\n\n".join(lines)


def table_first_answer(
    execution_result: YamlDict,
    question: str = "",
    *,
    field_selection: NarrationFieldSelection | None = None,
) -> Optional[str]:
    if execution_result.get("error"):
        return None
    if execution_result.get("result_type") != "raw_records":
        return None
    if execution_result.get("empty"):
        return "No matching records found."

    summary = raw_record_summary_answer(question, execution_result, field_selection=field_selection)
    if summary:
        return summary

    plan = execution_result.get("plan") or {}
    metric = plan.get("metric")
    records = execution_result.get("records", []) or []
    if metric == "last_known_location" and records:
        record = records[0]
        area = record.get("area")
        ward = record.get("ward")
        zone = record.get("zoneName")
        timestamp = record.get("timestamp")
        movement = record.get("movement_detail") or record.get("movement_status")
        location_parts = []
        if area not in (None, "", "unknown"):
            location_parts.append(f"area {area}")
        if ward not in (None, "", "unknown"):
            location_parts.append(f"ward {ward}")
        if zone not in (None, "", "unknown"):
            location_parts.append(f"zone {zone}")
        location_text = ", ".join(location_parts) or "the latest matching location record"
        answer = f"The person was last seen in {location_text}"
        if timestamp not in (None, "", "unknown"):
            answer += f" at {timestamp}"
        if movement not in (None, "", "unknown"):
            answer += f" ({movement})"
        return answer + "."

    total_count = int(execution_result.get("total_count", len(execution_result.get("records", []) or [])))
    if total_count <= 0:
        return "No matching records found."
    shown_count = len(records)
    if total_count == 1:
        return "Showing 1 matching record in the table."
    if shown_count > 0 and total_count > shown_count:
        return (
            f"Showing the first {shown_count} of {total_count} matching records in the table. "
            "Use the download option to get all results."
        )
    return f"Showing {total_count} matching records in the table."
