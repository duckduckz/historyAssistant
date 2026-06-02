from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from historyAssistant.config import load_source_configs, source_id_for_table
from historyAssistant.execution import execute_canonical_plan, open_es_connection
from historyAssistant.app.export import build_export_file, preview_dataframe
from historyAssistant.app.narration import narrate
from historyAssistant.planning.core.validator import build_validation_policy, validate_and_repair_plan

YamlDict = Dict[str, Any]

_NEW_QUESTION_RE = re.compile(
    r"(?i)^\s*(?:how|what|which|who|where|when|show|list|display|get|compare|"
    r"give|find|count|tell|does|did|is|are|was|were)\b"
)

_OPTION_A_RE = re.compile(
    r"(?i)^\s*(?:option\s*a|a\.?|ranked?|ranking|yes,?\s*rank)"
    r"(?:\s*[.!?]?\s*$|\b)",
)
_OPTION_B_RE = re.compile(
    r"(?i)^\s*(?:option\s*b|b\.?|flat|no\s*rank)"
    r"(?:\s*[.!?]?\s*$|\b)",
)


@dataclass
class PendingClarification:
    clarification_type: str
    original_question: str
    normalized_question: str
    missing_slots: List[str] = field(default_factory=list)
    candidate_interpretations: List[str] = field(default_factory=list)
    canonical_plan: YamlDict = field(default_factory=dict)
    candidate_plans: List[YamlDict] = field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 2

    def to_dict(self) -> YamlDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[YamlDict]) -> Optional["PendingClarification"]:
        if not data:
            return None
        return cls(
            clarification_type=str(data.get("clarification_type", "")),
            original_question=str(data.get("original_question", "")),
            normalized_question=str(data.get("normalized_question", "")),
            missing_slots=list(data.get("missing_slots", []) or []),
            candidate_interpretations=list(data.get("candidate_interpretations", []) or []),
            canonical_plan=dict(data.get("canonical_plan", {}) or {}),
            candidate_plans=list(data.get("candidate_plans", []) or []),
            attempt_count=int(data.get("attempt_count", 0) or 0),
            max_attempts=int(data.get("max_attempts", 2) or 2),
        )


def default_candidate_interpretations(clarification_type: str) -> List[str]:
    if clarification_type == "ranking_ambiguity":
        return ["Option A: ranked view", "Option B: flat breakdown"]
    return []


def build_clarification_question(pending: PendingClarification) -> str:
    question_builders: Dict[str, Callable[[PendingClarification], str]] = {
        "ranking_ambiguity": question_for_ranking_ambiguity,
        "low_confidence": question_for_low_confidence,
        "source_ambiguity": question_for_source_ambiguity,
    }
    builder = question_builders.get(pending.clarification_type, question_default)
    return builder(pending)


def question_for_ranking_ambiguity(pending: PendingClarification) -> str:
    if pending.clarification_type == "ranking_ambiguity":
        clarification = (pending.canonical_plan or {}).get("_clarification", {}) or {}
        if clarification.get("message"):
            return str(clarification["message"])
    if pending.candidate_interpretations:
        lines = [
            "I found more than one possible interpretation.",
            "Please choose one of these options:",
        ]
        for idx, option in enumerate(pending.candidate_interpretations, start=1):
            lines.append(f"{idx}. {option}")
        return "\n".join(lines)

    return "Could you choose whether you want a ranked view or a flat breakdown?"


def _primary_target_table(pending: PendingClarification) -> str:
    plan = pending.canonical_plan or {}
    target_tables = plan.get("target_tables") or []
    if isinstance(target_tables, list) and target_tables:
        return str(target_tables[0])
    return ""


def question_for_low_confidence(pending: PendingClarification) -> str:
    target_table = _primary_target_table(pending)
    if target_table == "historyLocations":
        return (
            "I'm not sure which location result you want. Could you clarify whether you're asking about "
            "where the person was, whether they moved/exercised, movement records, or last known location? "
            "If possible, please also mention the day or time."
        )
    if target_table == "historyWorkflows":
        return (
            "I'm not sure which workflow result you want. Could you clarify whether you're asking about "
            "who was present, presence event counts, room presence, or ward presence? "
            "If possible, please also mention the day, shift, ward, or room."
        )
    return (
        "I'm not sure what you're looking for. Could you clarify whether you're asking about "
        "response times, call volumes, staying times, or something else? "
        "If possible, please also mention the time period."
    )


def question_for_source_ambiguity(pending: PendingClarification) -> str:
    lines = [
        "I found more than one plausible way to answer this.",
        "Please choose one of these options:",
    ]
    for idx, option in enumerate(pending.candidate_interpretations or [], start=1):
        lines.append(f"{idx}. {option}")
    return "\n".join(lines) if pending.candidate_interpretations else question_default(pending)



def question_default(_pending: PendingClarification) -> str:
    return "Could you clarify your request?"


def is_clarification_reply(message: str, pending: PendingClarification) -> bool:
    if pending is None:
        return False
    text = (message or "").strip()
    if not text:
        return False
    if pending.clarification_type in {"ranking_ambiguity", "source_ambiguity"}:
        return extract_choice(text, pending) is not None
    if text.endswith("?") or _NEW_QUESTION_RE.match(text):
        return False
    return len(text.split()) <= 20


def merge_question(pending: PendingClarification, followup: str) -> str:

    original = (pending.original_question or "").rstrip(".?! ")
    follow = (followup or "").strip()
    if not original:
        return follow
    if not follow:
        return original
    return f"{original} {follow}"


def extract_choice(message: str, pending: PendingClarification) -> Optional[str]:
    if pending.clarification_type not in {"ranking_ambiguity", "source_ambiguity"}:
        return None
    if not message:
        return None
    if pending.clarification_type == "source_ambiguity":
        text = str(message or "").strip().lower().strip(". )")
        if text in {"1", "option 1", "first", "a", "option a"}:
            return "1"
        if text in {"2", "option 2", "second", "b", "option b"}:
            return "2"
        if text in {"3", "option 3", "third", "c", "option c"}:
            return "3"
        for idx, option in enumerate(pending.candidate_interpretations or [], start=1):
            option_text = str(option or "").strip().lower().strip(". )")
            first_word = option_text.split(":", 1)[0].strip()
            if first_word and (text == first_word or text.startswith(f"{first_word}:")):
                return str(idx)
            if option_text and (text == option_text or option_text.startswith(text)):
                return str(idx)
        return None
    if _OPTION_A_RE.search(message):
        return "A"
    if _OPTION_B_RE.search(message):
        return "B"
    return None


def advance_pending_attempt(pending: PendingClarification) -> PendingClarification:
    return replace(pending, attempt_count=pending.attempt_count + 1)


def carry_followup_attempt(updated_pending: Optional[YamlDict], prior_pending: PendingClarification) -> Optional[YamlDict]:
    pending_obj = PendingClarification.from_dict(updated_pending)
    if not pending_obj:
        return updated_pending
    next_attempt = prior_pending.attempt_count + 1
    if pending_obj.attempt_count < next_attempt:
        pending_obj = replace(pending_obj, attempt_count=next_attempt)
    return pending_obj.to_dict()


def clarification_debug_block(
    *,
    pending: Optional[PendingClarification] = None,
    followup_answer: Optional[str] = None,
    merged_question: Optional[str] = None,
    reason: Optional[str] = None,
) -> YamlDict:
    return {
        "triggered": pending is not None,
        "type": (pending.clarification_type if pending else None),
        "reason": reason or (pending.clarification_type if pending else None),
        "missing_slots": list(pending.missing_slots) if pending else [],
        "original_question": (pending.original_question if pending else None),
        "followup_answer": followup_answer,
        "merged_question": merged_question,
        "attempt_count": (pending.attempt_count if pending else 0),
    }


def set_clarification_debug(
    debug_payload: YamlDict,
    *,
    pending: Optional[PendingClarification] = None,
    followup_answer: Optional[str] = None,
    merged_question: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    debug_payload["clarification"] = clarification_debug_block(
        pending=pending,
        followup_answer=followup_answer,
        merged_question=merged_question,
        reason=reason,
    )


def record_deferred_ranking_clarification(debug_payload: YamlDict, plan: YamlDict) -> None:
    clarification = plan.get("_clarification")
    if not clarification:
        return
    debug_payload["deferred_clarification"] = {
        "type": "ranking_ambiguity",
        "surface_to_user": False,
        "reason": "valid flat breakdown already returned",
        "clarification": clarification,
    }


def graceful_fallback_response() -> str:
    return (
        "I wasn't able to understand your question after a couple of tries. "
        "Could you try rephrasing it from scratch? For example:\n"
        "- What is the average response time per ward last month?\n"
        "- Show me the calls for ward 2V on 24 June 2024\n"
        "- Which 3 wards had the most calls last week?"
    )


def execute_clarification_choice(choice: str,pending: PendingClarification,*,bundle: Any,client: Any,export_mode: str) -> Tuple[str, pd.DataFrame, Optional[str], YamlDict]:
    if pending.clarification_type == "source_ambiguity":
        return _execute_source_ambiguity_choice(choice, pending, bundle=bundle, client=client, export_mode=export_mode)

    plan = deepcopy(pending.canonical_plan or {})
    clarification = plan.get("_clarification", {}) or {}
    question = pending.original_question

    debug_payload: YamlDict = {
        "clarification_choice": choice,
        "original_question": question,
        "plan_used": plan,
    }

    if choice == "A":
        original_ranking = clarification.get("original_ranking")
        if original_ranking:
            plan["ranking_order"] = deepcopy(original_ranking)
            plan["order_by"] = deepcopy(original_ranking)
        plan.pop("_clarification", None)

        es = open_es_connection(config_bundle=bundle)
        execution_result = execute_canonical_plan(plan, es, config_bundle=bundle)
        narration_text = narrate(question, execution_result, client)
        execution_item: YamlDict = {
            "sub_request_id": 1,
            "sub_question": question,
            "request_type": "qa",
            "execution_result": execution_result,
            "narration": narration_text,
        }
        debug_payload["execution"] = [execution_item]
        preview_df = preview_dataframe([execution_item])
        export_file = build_export_file([execution_item], export_mode)
        return narration_text or "No results found.", preview_df, export_file, debug_payload

    answer = "Got it - keeping the flat breakdown as shown above."
    return answer, pd.DataFrame(), None, debug_payload


def _execute_source_ambiguity_choice(
    choice: str,
    pending: PendingClarification,
    *,
    bundle: Any,
    client: Any,
    export_mode: str,
) -> Tuple[str, pd.DataFrame, Optional[str], YamlDict]:
    try:
        idx = int(choice) - 1
    except Exception:
        idx = -1
    plans = list(pending.candidate_plans or [])
    if idx < 0 or idx >= len(plans):
        return "I could not match that option. Please choose one of the listed options.", pd.DataFrame(), None, {
            "clarification_choice": choice,
            "original_question": pending.original_question,
            "error": "source_ambiguity_choice_out_of_range",
        }

    plan = deepcopy(plans[idx])
    runtime_source_id = source_id_for_table((plan.get("target_tables") or [None])[0])
    runtime_bundle = load_source_configs(runtime_source_id) if runtime_source_id else bundle
    validation = validate_and_repair_plan(plan, policy=build_validation_policy(runtime_bundle))
    if not validation.get("valid"):
        debug_payload: YamlDict = {
            "clarification_choice": choice,
            "original_question": pending.original_question,
            "plan_used": plan,
            "validation": validation,
            "error": "source_ambiguity_choice_plan_invalid",
        }
        blocking = list(validation.get("blocking_errors") or [])
        reason = "; ".join(blocking) if blocking else "The selected plan did not pass validation."
        return f"The selected option could not be executed safely: {reason}", pd.DataFrame(), None, debug_payload

    plan = validation.get("canonical_plan") or plan
    es = open_es_connection(config_bundle=runtime_bundle)
    execution_result = execute_canonical_plan(plan, es, config_bundle=runtime_bundle)
    narration_text = narrate(pending.original_question, execution_result, client)
    execution_item: YamlDict = {
        "sub_request_id": 1,
        "sub_question": pending.original_question,
        "request_type": plan.get("request_type", "qa"),
        "execution_result": execution_result,
        "narration": narration_text,
    }
    debug_payload: YamlDict = {
        "clarification_choice": choice,
        "original_question": pending.original_question,
        "plan_used": plan,
        "validation": validation,
        "execution": [execution_item],
    }
    preview_df = preview_dataframe([execution_item])
    export_file = build_export_file([execution_item], export_mode)
    return narration_text or "No results found.", preview_df, export_file, debug_payload
