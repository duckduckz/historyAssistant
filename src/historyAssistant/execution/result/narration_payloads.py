from __future__ import annotations

import json as _json
from typing import Any, Dict

from historyAssistant.config import YamlDict
from historyAssistant.execution.result.narration_field_selector import NarrationFieldSelection, display_fields_from_selection
from historyAssistant.llm.llm_client import _json_default


def build_comparison_payload(comparison: YamlDict) -> YamlDict:
    payload: YamlDict = {
        "comparison_records": comparison.get("records", []),
        "comparison_scalar_answer": comparison.get("scalar_answer"),
    }
    change = comparison.get("change", {})
    if change.get("has_data"):
        payload["change"] = change
    if change.get("suppressed"):
        payload["comparison_suppressed"] = True
        payload["suppression_reason"] = change.get("reason")
    if change.get("partial_overlap"):
        payload["partial_overlap"] = True
        payload["groups_only_in_main"] = change.get("groups_only_in_main", [])
        payload["groups_only_in_comparison"] = change.get("groups_only_in_comparison", [])
    return payload


def compact_record(
    record: YamlDict,
    result_type: str,
    *,
    display_fields: list[str] | None = None,
) -> YamlDict:
    if result_type == "raw_records":
        keep = display_fields or list(record.keys())[:8]
    else:
        keep = list(record.keys())[:6]
    compact = {key: record.get(key) for key in keep if key in record and record.get(key) not in (None, "")}
    if compact:
        return compact
    return {
        key: value
        for key, value in list(record.items())[:8]
        if value not in (None, "")
    }


def payload_debug(payload: YamlDict, *, mode: str, payload_kind: str) -> YamlDict:
    payload_text = _json.dumps(payload, ensure_ascii=False, default=_json_default)
    records = payload.get("records") or []
    return {
        "mode": mode,
        "payload_kind": payload_kind,
        "payload_chars": len(payload_text),
        "payload_record_count": len(records),
        "payload_sample_records": records[:2],
    }


def build_fast_narration_payload(
    question: str,
    execution_result: YamlDict,
    *,
    field_selection: NarrationFieldSelection | None = None,
) -> YamlDict:
    records = execution_result.get("records", []) or []
    result_type = execution_result.get("result_type", "aggregated")
    payload: YamlDict = {
        "question": question,
        "request_type": execution_result.get("request_type"),
        "result_type": result_type,
        "scalar_answer": execution_result.get("scalar_answer"),
        "record_count": execution_result.get("total_count", len(records)),
        "records": [
            compact_record(
                row,
                result_type,
                display_fields=display_fields_from_selection(field_selection),
            )
            for row in records[:10]
        ],
        "empty": execution_result.get("empty"),
        "error": execution_result.get("error"),
        "warnings": list(execution_result.get("warnings", []) or []),
    }
    comparison = execution_result.get("comparison")
    if comparison and comparison.get("ok"):
        compact_comparison: Dict[str, Any] = {
            "scalar_answer": comparison.get("scalar_answer"),
            "records": [compact_record(row, "aggregated") for row in (comparison.get("records") or [])[:5]],
        }
        change = comparison.get("change", {})
        if change:
            compact_comparison["change"] = change
        payload["comparison"] = compact_comparison
    return payload


def build_llm_narration_payload(
    question: str,
    execution_result: YamlDict,
    *,
    field_selection: NarrationFieldSelection | None = None,
) -> YamlDict:
    records = execution_result.get("records", [])
    payload = {
        "question": question,
        "request_type": execution_result.get("request_type"),
        "result_type": execution_result.get("result_type", "aggregated"),
        "scalar_answer": execution_result.get("scalar_answer"),
        "record_count": len(records),
        "records": records,
        "empty": execution_result.get("empty"),
        "error": execution_result.get("error"),
        "warnings": list(execution_result.get("warnings", []) or []),
        "chart_payload": execution_result.get("chart_payload"),
    }
    if field_selection and field_selection.used:
        payload["field_selection"] = field_selection.to_dict()
    comparison = execution_result.get("comparison")
    if comparison and comparison.get("ok"):
        payload["comparison"] = build_comparison_payload(comparison)
    return payload


def fallback_narration_after_llm_error(execution_result: YamlDict) -> str:
    scalar_answer = execution_result.get("scalar_answer")
    records = execution_result.get("records", [])
    result_type = execution_result.get("result_type", "aggregated")

    if scalar_answer is not None:
        return f"Result: {scalar_answer}"
    if records:
        if result_type == "raw_records":
            cols = ", ".join(list(records[0].keys())[:8]) if records else ""
            return f"Found {len(records)} record(s). Fields: {cols}."
        return f"Returned {len(records)} rows."
    if execution_result.get("error"):
        return f"Execution failed: {execution_result['error']}"
    return "No grounded answer was available."
