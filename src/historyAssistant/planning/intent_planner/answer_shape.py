from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from historyAssistant.config import ConfigBundle

logger = logging.getLogger(__name__)

YamlDict = Dict[str, Any]

RAW_RECORDS = "raw_records"
ANALYTIC_SUMMARY = "analytic_summary"
DISTINCT_VALUES = "distinct_values"
AMBIGUOUS = "ambiguous"

_ANALYTIC_RE = re.compile(
    r"\b(?:top|bottom)\s+\d+\b"
    r"|\b(?:highest|lowest|most|fewest|busiest|fastest|slowest|longest|shortest|best|worst)\b"
    r"|\b(?:how many|how much|how often|count|number of|total|sum|average|avg|mean|median|rate|percentage|percent)\b"
    r"|\b(?:per|by each|grouped by|breakdown|distribution|trend|changed|change over|over time|week by week|month by month|day by day)\b"
    r"|\b(?:daily|hourly|weekly|monthly|yearly)\b"
    r"|\b(?:more|less|fewer|longer|shorter|faster|slower|busier)\b.+\b(?:than|compared with|compared to|versus|vs)\b"
    r"|\b(?:than|compared with|compared to|versus|vs)\b.+\b(?:more|less|fewer|longer|shorter|faster|slower|busier)\b"
    r"|\bcompare(?:s|d|ing)?\b(?!.*\b(?:by|across|breakdown|grouped)\b).+?\b(?:and|versus|vs\.?|to|with)\b.+",
    re.IGNORECASE,
)

_RECORD_VERB_RE = re.compile(r"\b(?:show|list|display|fetch|get|give me|see|check|pull up)\b", re.IGNORECASE)
_RECORD_NOUN_RE = re.compile(
    r"\b(?:historycalls|history calls|calls?|call records?|records?|rows?|entries|events?)\b"
    r"|\b(?:historylocations|history locations|locations?|movement records?|movements?)\b"
    r"|\b(?:historyworkflows|history workflows|workflows?|presence records?|presence|activations?)\b",
    re.IGNORECASE,
)
_CHART_KEYWORD_RE = re.compile(r"\b(?:chart|graph|plot|pie|bar|line|visuali[sz]e?)\b", re.IGNORECASE)
_DISTINCT_VALUE_RE = re.compile(
    r"\b(?:which|what|who)\s+(?:patients?|people|persons?|users?|nurses?|staff|wards?|rooms?|zones?|areas?|locations?)\b"
    r"|\bwho\s+was\s+present\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerShapeHint:
    shape: str
    confidence: float
    reason: str
    debug: YamlDict = field(default_factory=dict)

    def to_dict(self) -> YamlDict:
        return {
            "shape": self.shape,
            "confidence": self.confidence,
            "reason": self.reason,
            **dict(self.debug),
        }


def resolve_answer_shape(
    *,
    question: str,
    normalized_question: str,
    detected_entities: Optional[YamlDict],
    bundle: ConfigBundle,
    client: Any = None,
    model: Optional[str] = None,
) -> AnswerShapeHint:
    combined = " ".join(part for part in [question or "", normalized_question or ""] if part)

    # 1. direct answer-shape signals. Analytic calculation words win over
    # record-browsing openers; record browsing wins over incidental metric
    # mentions like "with response time".
    if _ANALYTIC_RE.search(combined):
        return AnswerShapeHint(ANALYTIC_SUMMARY, 0.86, "analytic_signal")
    if _has_group_by_signal(combined, detected_entities):
        return AnswerShapeHint(ANALYTIC_SUMMARY, 0.84, "group_by_signal")
    if _DISTINCT_VALUE_RE.search(combined):
        return AnswerShapeHint(DISTINCT_VALUES, 0.82, "distinct_value_signal")
    if (
        _RECORD_VERB_RE.search(combined)
        and _RECORD_NOUN_RE.search(combined)
        and not _CHART_KEYWORD_RE.search(combined)
    ):
        return AnswerShapeHint(RAW_RECORDS, 0.9, "explicit_record_request")
    # A record/list verb plus a numeric threshold on a measure field (e.g.
    # "show gaps larger than 30 minutes") is a row-filtering request even when
    # the noun isn't a record keyword. General across sources/measures.
    if (
        _RECORD_VERB_RE.search(combined)
        and not _CHART_KEYWORD_RE.search(combined)
        and _has_measure_threshold(combined, bundle)
    ):
        return AnswerShapeHint(RAW_RECORDS, 0.85, "measure_threshold_record_request")

    # 2. resolved metrics aggregation is the shape.
    metric = _first_detected_metric(detected_entities, bundle)
    if metric:
        aggregation = _metric_aggregation(metric, bundle)
        if aggregation == "raw_records":
            return AnswerShapeHint(RAW_RECORDS, 0.95, "metric_aggregation:raw_records",
                                   {"metric": metric})
        if aggregation:
            return AnswerShapeHint(ANALYTIC_SUMMARY, 0.95, "metric_aggregation:computed",
                                   {"metric": metric, "aggregation": aggregation})

    # 3. ambiguous openers to ask LLM.
    if client is not None:
        llm_shape = _llm_answer_shape(combined, client=client, model=model,
                                      detected_entities=detected_entities)
        if llm_shape is not None:
            return llm_shape


    return AnswerShapeHint(ANALYTIC_SUMMARY, 0.3, "default_analytic")


def _has_measure_threshold(question: str, bundle: ConfigBundle) -> bool:
    from historyAssistant.planning.extraction.measure_threshold_extractor import (
        question_has_measure_threshold,
    )

    try:
        tables = (bundle.table_registry.get("tables", {}) or {}).values()
    except Exception:
        return False
    return any(question_has_measure_threshold(question, table_def) for table_def in tables)


def _first_detected_metric(detected_entities: Optional[YamlDict], bundle: ConfigBundle) -> Optional[str]:
    allowed = set((bundle.metric_catalog.get("metrics", {}) or {}).keys())
    for raw in (detected_entities or {}).get("metrics", []) or []:
        value = str(raw)
        if value in allowed:
            return value
    return None


def _metric_aggregation(metric: str, bundle: ConfigBundle) -> Optional[str]:
    cfg = (bundle.metric_catalog.get("metrics", {}) or {}).get(metric, {}) or {}
    return str(cfg.get("aggregation") or "").strip() or None


def _has_group_by_signal(question: str, detected_entities: Optional[YamlDict]) -> bool:
    if not (detected_entities or {}).get("dimensions"):
        return False
    return bool(re.search(r"(?i)\b(?:by|per|across|grouped by|breakdown by)\b", question or ""))


_LLM_SYSTEM_TEXT = (
    "You decide the ANSWER SHAPE for a hospital history analytics question.\n"
    "Ignore the opener ('can I see', 'show me', 'pull up', etc.).\n"
    "Ask: what would the answer look like?\n"
    "  raw_records      -> a table of rows fetched as-is, no number computed.\n"
    "  distinct_values  -> a list of entities/dimension values, usually answering which/who/what.\n"
    "  analytic_summary -> a computed number, summary table, ranking, or comparison.\n"
    'Return strict JSON only: {"shape": "raw_records|distinct_values|analytic_summary", "confidence": <0.0-1.0>}'
)


def _llm_answer_shape(
    question: str,
    *,
    client: Any,
    model: Optional[str],
    detected_entities: Optional[YamlDict],
) -> Optional[AnswerShapeHint]:
    from historyAssistant.llm.llm_client import call_json_prompt, get_default_model_name

    try:
        raw = call_json_prompt(
            client,
            model=model or get_default_model_name(),
            system_text=_LLM_SYSTEM_TEXT,
            user_payload={"question": question, "detected_entities": detected_entities or {}},
            max_tokens=60,
        )
        parsed = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer-shape LLM call failed: %s - defaulting to analytic_summary", exc)
        return None

    shape = str(parsed.get("shape", "")).strip().lower()
    if shape not in {RAW_RECORDS, DISTINCT_VALUES, ANALYTIC_SUMMARY}:
        return None
    raw_conf = parsed.get("confidence", 0.5)
    confidence = max(0.0, min(1.0, float(raw_conf))) if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) else 0.5
    return AnswerShapeHint(shape, confidence, "llm_classified")


def _parse_json(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in answer-shape response: {text[:200]}")
    return json.loads(match.group(0))


__all__ = [
    "RAW_RECORDS",
    "ANALYTIC_SUMMARY",
    "DISTINCT_VALUES",
    "AMBIGUOUS",
    "AnswerShapeHint",
    "resolve_answer_shape",
]
