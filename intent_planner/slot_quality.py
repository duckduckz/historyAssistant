from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .intent_slots import SlotExtractionResult
from .neutral_intent import SourceIntentResolution

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class SlotQualityResult:
    trust_deterministic: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = "use_deterministic"

    def to_dict(self) -> YamlDict:
        return asdict(self)


_COMPARISON_RE = re.compile(
    r"\b(?:compare|compared|versus|vs\.?|against|from\s+\w+\s+\d{4}\s+to\s+\w+\s+\d{4})\b",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"\b(?:percentage|percent|increase|decrease|difference|change|growth|drop)\b",
    re.IGNORECASE,
)
_RANKING_RE = re.compile(
    r"\b(?:top|bottom|highest|lowest|biggest|smallest|most|least|busiest|slowest|fastest)\b",
    re.IGNORECASE,
)
_NUMERIC_SUMMARY_RE = re.compile(
    r"\b(?:average|avg|mean|minimum|min|maximum|max|total|sum|median|percentile|p\d{2}|shortest|longest)\b"
    r".*\b(?:time|duration|response|staying|activation|gap)\b"
    r"|\b(?:time|duration|response|staying|activation|gap)\b"
    r".*\b(?:average|avg|mean|minimum|min|maximum|max|total|sum|median|percentile|p\d{2}|shortest|longest)\b",
    re.IGNORECASE,
)
_BUCKET_RE = re.compile(
    r"\b(?:hourly|daily|weekly|monthly|per\s+(?:hour|day|week|month)|each\s+(?:hour|day|week|month)|every\s+\d+\s*(?:m|min|minute|h|hour|d|day))\b",
    re.IGNORECASE,
)
_DISTINCT_PROJECTION_RE = re.compile(
    r"\b(?:distinct|unique|list|show|give|display)\b.*\b(?:names?|patients?|nurses?|staff|wards?|rooms?|areas?|zones?)\b",
    re.IGNORECASE,
)
_ENTITY_ANSWER_RE = re.compile(
    r"\b(?:who|which|what)\b",
    re.IGNORECASE,
)


def assess_slot_quality(
    *,
    question: str,
    normalized_question: str,
    detected_entities: Optional[YamlDict],
    source_resolution: SourceIntentResolution,
    extraction: SlotExtractionResult,
) -> SlotQualityResult:

    if not extraction.used or not extraction.slots:
        return SlotQualityResult(
            trust_deterministic=False,
            score=0.0,
            reasons=["deterministic_slots_missing"],
            recommended_action="use_llm_slots",
        )

    slots = extraction.slots
    entities = detected_entities or {}
    text = f"{question or ''} {normalized_question or ''}"
    score = 1.0
    reasons: List[str] = []

    metrics = [str(item) for item in entities.get("metrics", []) or [] if str(item).strip()]
    if len(set(metrics)) > 1:
        score -= 0.35
        reasons.append("multiple_metric_entities")

    if not metrics:
        score -= 0.18
        reasons.append("metric_defaulted_or_not_explicit")

    if source_resolution.changed_source:
        score -= 0.08
        reasons.append("source_changed_by_neutral_intent")

    if source_resolution.confidence < 0.8:
        score -= 0.18
        reasons.append("source_confidence_below_0_8")

    if _COMPARISON_RE.search(text) and not slots.comparison:
        score -= 0.3
        reasons.append("comparison_signal_missing_compare_to")

    metric_params = slots.metric_params or {}
    if _CHANGE_RE.search(text) and metric_params.get("calculation") not in {"percentage_change", "difference", "rate"}:
        score -= 0.28
        reasons.append("change_or_rate_signal_not_represented")

    if _RANKING_RE.search(text) and not slots.ranking.detected:
        score -= 0.18
        reasons.append("ranking_signal_missing")

    if _expects_entity_answer(text) and _is_scalar_analytic_plan(slots):
        score -= 0.35
        reasons.append("entity_answer_target_missing_group_by")

    metric_name = str(slots.metric or "")
    count_like_metric = metric_name.endswith("_count") or metric_name in {
        "total_calls",
        "presence_event_count",
        "location_update_count",
        "movement_event_count",
    }
    if _NUMERIC_SUMMARY_RE.search(text) and count_like_metric and not metric_params.get("field"):
        score -= 0.35
        reasons.append("numeric_measure_signal_not_represented")

    if _BUCKET_RE.search(text) and not slots.bucket_interval:
        score -= 0.22
        reasons.append("bucket_signal_missing")

    if _DISTINCT_PROJECTION_RE.search(text) and not slots.group_by:
        score -= 0.22
        reasons.append("projection_or_distinct_signal_missing_group_by")

    score = max(0.0, min(1.0, round(score, 3)))
    trust = score >= 0.72
    return SlotQualityResult(
        trust_deterministic=trust,
        score=score,
        reasons=reasons,
        recommended_action="use_deterministic" if trust else "use_llm_slots",
    )


def _expects_entity_answer(text: str) -> bool:
    return bool(_ENTITY_ANSWER_RE.search(text))


def _is_scalar_analytic_plan(slots: Any) -> bool:
    if slots.answer_shape != "analytic_summary":
        return False
    if slots.group_by or slots.projection_field or slots.comparison:
        return False
    if slots.aggregation == "raw_records":
        return False
    return bool(slots.metric)


__all__ = ["SlotQualityResult", "assess_slot_quality"]
