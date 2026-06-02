from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

YamlDict = Dict[str, Any]


def empty_conversation_state() -> YamlDict:
    return {
        "last_user_question": "",
        "last_effective_question": "",
        "last_valid_plan": None,
        "last_execution_summary": None,
        "last_source_id": None,
        "last_source_confidence": 0.0,
    }


def normalize_conversation_state(state: Optional[YamlDict]) -> YamlDict:
    normalized = empty_conversation_state()
    if isinstance(state, dict):
        normalized.update({k: deepcopy(v) for k, v in state.items() if k in normalized})
    return normalized


def execution_succeeded(execution_result: YamlDict) -> bool:
    if not isinstance(execution_result, dict) or execution_result.get("error"):
        return False
    if execution_result.get("ok") is True:
        return True
    return str(execution_result.get("status") or "").strip().lower() == "ok"


def summarize_execution_result(execution_result: YamlDict) -> YamlDict:
    records = execution_result.get("records") or []
    return {
        "ok": execution_succeeded(execution_result),
        "result_type": execution_result.get("result_type"),
        "record_count": len(records),
        "total_count": execution_result.get("total_count"),
        "empty": execution_result.get("empty"),
        "has_chart": bool(execution_result.get("chart_payload")),
        "chart_type": (
            (execution_result.get("chart_payload") or {})
            .get("chart_spec", {})
            .get("chart_type")
        ),
        "last_execution_result": deepcopy(execution_result),
    }


def _update_from_fast_path(
    updated: YamlDict,
    *,
    user_question: str,
    debug_payload: YamlDict,
    key: str,
) -> Optional[YamlDict]:
    fast_path = debug_payload.get(key)
    if not fast_path:
        return None
    plan = fast_path.get("plan")
    execution_result = fast_path.get("execution_result") or {}
    if plan and execution_succeeded(execution_result):
        updated["last_user_question"] = user_question
        updated["last_effective_question"] = debug_payload.get("question") or user_question
        updated["last_valid_plan"] = deepcopy(plan)
        updated["last_execution_summary"] = summarize_execution_result(execution_result)
        return updated
    return None


def update_conversation_state_from_debug(
    state: Optional[YamlDict],
    *,
    user_question: str,
    debug_payload: Optional[YamlDict],
) -> YamlDict:
    updated = normalize_conversation_state(state)
    if not debug_payload:
        return updated

    for fast_path_key in ("chart_restyle_result", "same_task_refinement_result"):
        result = _update_from_fast_path(updated, user_question=user_question, debug_payload=debug_payload, key=fast_path_key)
        if result is not None:
            return result

    selected_source = debug_payload.get("selected_source") or {}
    if selected_source.get("source_id"):
        updated["last_source_id"] = selected_source["source_id"]
        updated["last_source_confidence"] = float(selected_source.get("source_confidence") or 0.0)

    planning_items = list(debug_payload.get("planning") or [])
    execution_items = list(debug_payload.get("execution") or [])
    if not planning_items or not execution_items:
        return updated

    for planning_item, execution_item in zip(planning_items, execution_items):
        validation = planning_item.get("validation") or {}
        plan = validation.get("canonical_plan") or planning_item.get("canonical_plan")
        execution_result = execution_item.get("execution_result") or {}
        if validation.get("valid") and plan and execution_succeeded(execution_result):
            updated["last_user_question"] = user_question
            updated["last_effective_question"] = debug_payload.get("question") or user_question
            updated["last_valid_plan"] = deepcopy(plan)
            updated["last_execution_summary"] = summarize_execution_result(execution_result)
            return updated

    return updated
