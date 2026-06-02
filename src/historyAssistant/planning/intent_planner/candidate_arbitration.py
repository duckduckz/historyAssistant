from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from historyAssistant.config import load_source_configs
from historyAssistant.planning import normalize_question

from .slot_planner import SlotPlannerResult, try_build_slot_plan

YamlDict = Dict[str, Any]

_SOURCE_TABLES = {
    "calls": "historyCalls",
    "locations": "historyLocations",
    "workflows": "historyWorkflows",
}


@dataclass(frozen=True)
class CandidatePlan:
    source_id: str
    target_table: str
    result: SlotPlannerResult
    score: float
    interpretation: str

    @property
    def canonical_plan(self) -> Optional[YamlDict]:
        return self.result.canonical_plan

    def to_debug_dict(self) -> YamlDict:
        plan = self.canonical_plan or {}
        result_debug = self.result.to_debug_dict()
        return {
            "source_id": self.source_id,
            "target_table": self.target_table,
            "used": self.result.used,
            "score": round(self.score, 3),
            "reason": self.result.reason,
            "interpretation": self.interpretation,
            "llm_slot_fired": "llm_slot_extraction" in result_debug,
            "plan_preview": {
                "metric": plan.get("metric"),
                "aggregation": plan.get("aggregation"),
                "filters": plan.get("filters"),
                "group_by": plan.get("group_by"),
                "ranking_order": plan.get("ranking_order"),
                "date_rule": plan.get("date_rule"),
                "date_range": plan.get("date_range"),
            },
        }


@dataclass(frozen=True)
class ArbitrationResult:
    should_clarify: bool
    candidates: List[CandidatePlan]
    reason: str
    replacement: Optional[CandidatePlan] = None

    def to_debug_dict(self) -> YamlDict:
        return {
            "should_clarify": self.should_clarify,
            "reason": self.reason,
            "replacement": self.replacement.to_debug_dict() if self.replacement else None,
            "candidates": [candidate.to_debug_dict() for candidate in self.candidates],
        }


def arbitrate_low_confidence_source(
    *,
    question: str,
    selected_source_id: str,
    selected_result: SlotPlannerResult,
    source_confidence: float,
    request_type: str,
    chart_suppressed: bool,
    preferred_chart_type: Optional[str],
    client: Any,
    threshold: float = 0.75,
) -> ArbitrationResult:
    if client is None:
        return ArbitrationResult(False, [], "client_missing")
    if source_confidence >= threshold:
        return ArbitrationResult(False, [], "source_confident")

    candidates: List[CandidatePlan] = []
    if selected_result.used and selected_result.canonical_plan:
        selected_table = _target_table(selected_result.canonical_plan)
        candidates.append(
            CandidatePlan(
                source_id=selected_source_id,
                target_table=selected_table,
                result=selected_result,
                score=_score_plan(selected_result.canonical_plan),
                interpretation=_describe_plan(selected_source_id, selected_result.canonical_plan),
            )
        )

    for source_id in _candidate_sources(selected_source_id):
        target_table = _SOURCE_TABLES[source_id]
        try:
            bundle = load_source_configs(source_id)
            normalized = normalize_question(question, bundle.vocabulary)
            result = try_build_slot_plan(
                question=question,
                normalized_question=normalized.normalized_question,
                source_id=source_id,
                target_table=target_table,
                detected_entities=normalized.detected_entities,
                request_type=request_type,
                chart_suppressed=chart_suppressed,
                preferred_chart_type=preferred_chart_type,
                bundle=bundle,
                client=client,
                use_llm_slots=True,
            )
        except Exception:
            continue
        if not result.used or not result.canonical_plan:
            continue
        candidates.append(
            CandidatePlan(
                source_id=source_id,
                target_table=_target_table(result.canonical_plan),
                result=result,
                score=_score_plan(result.canonical_plan),
                interpretation=_describe_plan(source_id, result.canonical_plan),
            )
        )

    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        return ArbitrationResult(False, [], "no_valid_candidate_plans")
    if len(candidates) == 1 and (not selected_result.used or not selected_result.canonical_plan):
        return ArbitrationResult(False, candidates, "single_replacement_candidate", replacement=candidates[0])
    if len({_target_table(candidate.canonical_plan or {}) for candidate in candidates}) < 2:
        return ArbitrationResult(False, candidates, "no_cross_source_alternative")
    return ArbitrationResult(True, candidates[:3], "low_confidence_multiple_valid_source_plans")


def _candidate_sources(selected_source_id: str) -> Iterable[str]:
    for source_id in _SOURCE_TABLES:
        if source_id != selected_source_id:
            yield source_id


def _target_table(plan: YamlDict) -> str:
    target_tables = plan.get("target_tables") or []
    return str(target_tables[0]) if target_tables else ""


def _score_plan(plan: YamlDict) -> float:
    score = 0.0
    if plan.get("metric"):
        score += 1.0
    if plan.get("date_rule") and plan.get("date_rule") != "none":
        score += 0.5
    if plan.get("filters"):
        score += 0.5
    if plan.get("group_by"):
        score += 1.0
    if plan.get("ranking_order"):
        score += 1.0
    if plan.get("aggregation") == "raw_records":
        score += 0.25
    if (plan.get("_intent_packet") or {}).get("extraction_path") == "llm_slots":
        score += 0.25
    return score


def _describe_plan(source_id: str, plan: YamlDict) -> str:
    metric = str(plan.get("metric") or "the selected metric").replace("_", " ")
    group_by = [str(item).replace("_", " ") for item in (plan.get("group_by") or [])]
    source_label = source_id.replace("_", " ").title()
    if group_by:
        return f"{source_label}: use {metric}, grouped by {', '.join(group_by)}"
    return f"{source_label}: use {metric}"


def _dedupe_candidates(candidates: List[CandidatePlan]) -> List[CandidatePlan]:
    result: List[CandidatePlan] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        plan = candidate.canonical_plan or {}
        key = (
            candidate.source_id,
            _target_table(plan),
            str(plan.get("metric") or ""),
            repr(plan.get("group_by") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


__all__ = [
    "ArbitrationResult",
    "CandidatePlan",
    "arbitrate_low_confidence_source",
]
