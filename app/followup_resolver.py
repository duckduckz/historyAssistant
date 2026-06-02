from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from historyAssistant.app.conversation_state import normalize_conversation_state
from historyAssistant.llm import FOLLOWUP_RESOLUTION_SYSTEM_TEXT
from historyAssistant.llm.llm_client import call_json_prompt, get_default_model_name

YamlDict = Dict[str, Any]
ChatHistory = List[Dict[str, str]]

FOLLOWUP_CONFIDENCE_THRESHOLD = 0.75
FOLLOWUP_CONFIDENCE_THRESHOLD_RAW_RECORDS = 0.55

_ANALYTICS_INTENT_RE = re.compile(
    r"\b(?:how\s+many|how\s+long|how\s+often"
    r"|average|avg|mean|median|percentile"
    r"|total|sum|count|percentage|percent|rate"
    r"|compare|trend|breakdown|ranking"
    r"|top\s+\d+|bottom\s+\d+"
    r"|per\s+\w+|by\s+\w+|for\s+each|grouped?\s+by)\b",
    re.IGNORECASE,
)


_FOLLOWUP_EXAMPLES = [
    {
        "prior_question": "Show me the calls on 2024-05-17.",
        "prior_aggregation": "raw_records",
        "current_message": "and only night shift?",
        "mode": "followup", "operation": "filter_change", "confidence": 0.95,
        "rewrite": "Show me the night shift calls on 2024-05-17.",
        "reason": "adding shift filter to prior record lookup",
    },
    {
        "prior_question": "Show me the calls from last week.",
        "prior_aggregation": "raw_records",
        "current_message": "Now only the night shift.",
        "mode": "followup", "operation": "filter_change", "confidence": 0.95,
        "rewrite": "Show me the night shift calls from last week.",
        "reason": "narrowing prior lookup to night shift",
    },
    {
        "prior_question": "Show me the calls for ward 2V ZONE 2B last week.",
        "prior_aggregation": "raw_records",
        "current_message": "Now only the high priority ones.",
        "mode": "followup", "operation": "filter_change", "confidence": 0.95,
        "rewrite": "Show me the high priority calls for ward 2V ZONE 2B last week.",
        "reason": "adding priority filter to prior ward lookup",
    },
    {
        "prior_question": "Show me the calls from last week.",
        "prior_aggregation": "raw_records",
        "current_message": "and ward 2V ZONE 2A?",
        "mode": "followup", "operation": "filter_change", "confidence": 0.9,
        "rewrite": "Show me the calls for ward 2V ZONE 2A from last week.",
        "reason": "adding ward filter to prior date lookup",
    },
    {
        "prior_question": "Show me the Panic calls from this month.",
        "prior_aggregation": "raw_records",
        "current_message": "Now only for the night shift.",
        "mode": "followup", "operation": "filter_change", "confidence": 0.95,
        "rewrite": "Show me the night shift Panic calls from this month.",
        "reason": "adding shift filter to prior call-type lookup",
    },
    # raw_records follow-up — new analytics question (should NOT be treated as followup)
    {
        "prior_question": "Show me the calls on 2024-05-17.",
        "prior_aggregation": "raw_records",
        "current_message": "how many were there?",
        "mode": "new_question", "operation": "new_task", "confidence": 0.9,
        "rewrite": "how many were there?",
        "reason": "asking for a count — different intent from a record lookup",
    },
    # aggregation follow-ups
    {
        "prior_question": "What is the average response time per ward?",
        "prior_aggregation": "avg",
        "current_message": "Can you show that as a bar chart?",
        "mode": "followup", "operation": "chart_restyle", "confidence": 0.95,
        "rewrite": "Show the average response time per ward as a bar chart.",
        "reason": "chart restyle of prior aggregation result",
    },
    {
        "prior_question": "How many calls did each ward receive last month?",
        "prior_aggregation": "count",
        "current_message": "What about this month instead?",
        "mode": "followup", "operation": "date_change", "confidence": 0.9,
        "rewrite": "How many calls did each ward receive this month?",
        "reason": "changing the date window of prior aggregation",
    },
]


def default_resolution(message: str, *, reason: str = "no usable previous state") -> YamlDict:
    return {
        "mode": "new_question",
        "operation": "new_task",
        "confidence": 1.0,
        "rewrite": message,
        "plan_patch": {},
        "reason": reason,
    }


def _recent_messages(history: Optional[ChatHistory], limit: int = 6) -> list[YamlDict]:
    items = list(history or [])[-limit:]
    return [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in items
        if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]


def _coerce_resolution(raw_response: str, message: str, confidence_threshold: float = FOLLOWUP_CONFIDENCE_THRESHOLD) -> YamlDict:
    try:
        text = (raw_response or "{}").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in resolver response")
        parsed = json.loads(match.group(0))
    except Exception as exc:
        result = default_resolution(message, reason=f"resolver parse failed: {exc}")
        result["raw_response"] = raw_response
        return result

    mode = str(parsed.get("mode") or "unknown").strip().lower()
    if mode not in {"followup", "new_question", "topic_switch", "clarification", "unknown"}:
        mode = "unknown"
    operation = str(parsed.get("operation") or "unknown").strip().lower()
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    rewrite = str(parsed.get("rewrite") or message).strip() or message
    plan_patch = parsed.get("plan_patch") if isinstance(parsed.get("plan_patch"), dict) else {}

    result = {
        "mode": mode,
        "operation": operation,
        "confidence": confidence,
        "rewrite": rewrite,
        "plan_patch": deepcopy(plan_patch),
        "reason": str(parsed.get("reason") or ""),
        "raw_response": raw_response,
    }
    if mode != "followup" or confidence < confidence_threshold:
        result["rewrite"] = message
        result["plan_patch"] = {}
    return result


def resolve_followup(
    *,
    current_message: str,
    history: Optional[ChatHistory],
    conversation_state: Optional[YamlDict],
    client: Any,
) -> YamlDict:
    message = (current_message or "").strip()
    state = normalize_conversation_state(conversation_state)
    if not message:
        return default_resolution(message, reason="empty message")
    if not state.get("last_valid_plan"):
        return default_resolution(message, reason="no last valid analytics plan")
    # no prior turns means no possible follow-up — skip the LLM call
    if not history or len([m for m in history if m.get("role") == "user"]) == 0:
        return default_resolution(message, reason="no prior conversation turns")


    last_plan = state.get("last_valid_plan") or {}
    if str(last_plan.get("aggregation") or "").strip().lower() == "raw_records":
        original = str(state.get("last_effective_question") or state.get("last_user_question") or "").strip()
        if original and not _ANALYTICS_INTENT_RE.search(message):
            return {
                "mode": "followup",
                "operation": "filter_change",
                "confidence": 0.9,
                "rewrite": f"{original} {message}",
                "plan_patch": {},
                "reason": "deterministic raw_records filter refinement",
            }

    payload = {
        "examples": _FOLLOWUP_EXAMPLES,
        "current_message": message,
        "recent_messages": _recent_messages(history),
        "conversation_state": state,
        "allowed_modes": ["followup", "new_question", "topic_switch", "clarification", "unknown"],
        "allowed_operations": [
            "chart_restyle",
            "filter_change",
            "date_change",
            "grouping_change",
            "comparison",
            "same_task_refinement",
            "raw_detail",
            "new_task",
            "clarification_reply",
            "unknown",
        ],
    }
    try:
        raw_response = call_json_prompt(
            client,
            model=get_default_model_name(),
            system_text=FOLLOWUP_RESOLUTION_SYSTEM_TEXT,
            user_payload=payload,
            max_tokens=200,
        )
    except Exception as exc:
        result = default_resolution(message, reason=f"resolver call failed: {exc}")
        result["payload"] = payload
        return result

    result = _coerce_resolution(raw_response, message)
    result["payload"] = payload
    return result


def effective_message_from_resolution(message: str, resolution: Optional[YamlDict]) -> str:
    if not resolution:
        return message

    if resolution.get("mode") == "followup":
        rewrite = str(resolution.get("rewrite") or "").strip()
        if rewrite and rewrite != message:
            return rewrite
    return message

def is_confident_followup_resolution(resolution: Optional[YamlDict]) -> bool:
    if not resolution or resolution.get("mode") != "followup":
        return False

    try:
        confidence = float(resolution.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return confidence >= FOLLOWUP_CONFIDENCE_THRESHOLD


def apply_same_task_refinement(cached_result: YamlDict, plan_patch: YamlDict) -> Optional[YamlDict]:
    safe_keys = {"limit", "ranking_order", "display_order"}
    patch_keys = set(plan_patch.keys())
    if not patch_keys or not patch_keys.issubset(safe_keys):
        return None

    df = cached_result.get("dataframe")
    if df is None or not hasattr(df, "empty") or df.empty:
        return None

    patched = deepcopy(cached_result)
    df = patched["dataframe"].copy()

    ranking_order = plan_patch.get("ranking_order")
    if ranking_order and isinstance(ranking_order, dict):
        sort_field = ranking_order.get("field")
        direction = str(ranking_order.get("direction", "desc")).lower()
        if not sort_field or sort_field not in df.columns:
            return None
        df = df.sort_values(sort_field, ascending=(direction == "asc"), kind="mergesort")

    new_limit = plan_patch.get("limit")
    if new_limit is not None:
        try:
            new_limit = int(new_limit)
        except (TypeError, ValueError):
            return None
        if new_limit > len(df):
            return None
        df = df.head(new_limit)

    records = df.to_dict(orient="records")
    patched["dataframe"] = df
    patched["records"] = records
    patched["empty"] = df.empty

    chart_payload = patched.get("chart_payload")
    if isinstance(chart_payload, dict):
        chart_payload["rows"] = records

    if len(df) == 1 and "metric_value" in df.columns:
        patched["scalar_answer"] = df.iloc[0]["metric_value"]

    return patched
