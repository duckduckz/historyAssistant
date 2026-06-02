from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from historyAssistant.planning import normalize_question, screen_question_for_guardrails, route_source, SourceRoutingResult
from historyAssistant.app.chart_detection import detect_chart_intent, extract_preferred_chart_type, suppress_chart_intent

logger = logging.getLogger(__name__)

YamlDict = Dict[str, Any]

_SOURCE_TABLES = {
    "calls": "historyCalls",
    "locations": "historyLocations",
    "workflows": "historyWorkflows",
}


@dataclass
class ClassificationResult:
    blocked: bool = False
    blocked_response: str = ""
    # populated only when not blocked
    normalized_message: str = ""
    detected_entities: YamlDict = field(default_factory=dict)
    request_type: str = "qa"
    preferred_chart_type: Optional[str] = None
    chart_suppressed: bool = False
    chart_intent_source: Optional[str] = None
    debug_updates: YamlDict = field(default_factory=dict)
    
    source_ids: list[str] = field(default_factory=lambda: ["calls"])
    canonical_tables: list[str] = field(default_factory=lambda: ["historyCalls"])
    source_confidence: float = 0.0



class ClassificationAgent:
    def __init__(self, *, clean_message: str, bundle: Any, client: Any, cached_source_id: Optional[str] = None) -> None:
        self.clean_message = clean_message
        self.bundle = bundle
        self.client = client
        self.cached_source_id = cached_source_id

    def run(self) -> ClassificationResult:
        debug_updates: YamlDict = {}
        timing_rows: list[YamlDict] = []
        classification_start = time.perf_counter()
        step_start = classification_start

        def _record_timing(step: str, **extra: Any) -> None:
            nonlocal step_start
            now = time.perf_counter()
            row: YamlDict = {
                "step": step,
                "duration_ms": round((now - step_start) * 1000, 1),
            }
            row.update(extra)
            timing_rows.append(row)
            step_start = now

        # guardrails, no LLM call
        guardrail_result = screen_question_for_guardrails(self.clean_message, config_bundle=self.bundle)
        _record_timing("guardrails", llm=False, blocked=guardrail_result.blocked)
        debug_updates["guardrails"] = {
            "blocked": guardrail_result.blocked,
            "response_text": guardrail_result.response_text,
            "hits": [
                {
                    "rule_id": hit.rule_id,
                    "category": hit.category,
                    "reason": hit.reason,
                    "matched_keywords": hit.matched_keywords,
                }
                for hit in guardrail_result.hits
            ],
        }
        if guardrail_result.blocked:
            debug_updates["classification_timing"] = {
                "steps": timing_rows,
                "total_ms": round((time.perf_counter() - classification_start) * 1000, 1),
            }
            return ClassificationResult(
                blocked=True,
                blocked_response=guardrail_result.response_text or "This request is out of scope for the assistant.",
                debug_updates=debug_updates,
            )

        # normalize vocabulary
        normalization_result = normalize_question(self.clean_message, self.bundle.vocabulary)
        normalized_message = normalization_result.normalized_question
        _record_timing(
            "vocabulary_normalization",
            llm=False,
            replacements=len(normalization_result.replacements),
            detected_entity_categories=list(normalization_result.detected_entities.keys()),
        )
        debug_updates["normalization"] = {
            "original": normalization_result.original_question,
            "normalized": normalized_message,
            "replacements": [
                {"matched": r[0], "canonical": r[1], "category": r[2]}
                for r in normalization_result.replacements
            ],
            "detected_entities": normalization_result.detected_entities,
        }
        
        # source routing — skip LLM call if caller supplied a cached source_id
        if self.cached_source_id and self.cached_source_id in _SOURCE_TABLES:
            canonical_table = _SOURCE_TABLES[self.cached_source_id]
            source_routing = SourceRoutingResult(
                source_ids=[self.cached_source_id],
                canonical_tables=[canonical_table],
                confidence=1.0,
                reason="session_cache",
                debug={"router": "cached", "cached_source_id": self.cached_source_id},
            )
            _record_timing("source_routing", llm=False, router="cached", source_ids=source_routing.source_ids, confidence=1.0)
        else:
            source_routing = route_source(
                normalized_message,
                detected_entities=normalization_result.detected_entities,
                client=self.client,
                original_question=self.clean_message,
            )
            source_router_kind = (
                (source_routing.debug or {}).get("router")
                or (source_routing.debug or {}).get("rule_fallback", {}).get("router")
            )
            _record_timing(
                "source_routing",
                llm=source_router_kind == "llm",
                router=source_router_kind,
                source_ids=source_routing.source_ids,
                confidence=source_routing.confidence,
            )
        debug_updates["source_routing"] = {
            "source_ids": source_routing.source_ids,
            "canonical_tables": source_routing.canonical_tables,
            "confidence": source_routing.confidence,
            "reason": source_routing.reason,
            "debug": source_routing.debug,
        }


        # chart intent detection
        chart_suppressed = suppress_chart_intent(self.clean_message)
        is_chart = detect_chart_intent(self.clean_message)
        preferred_chart_type = extract_preferred_chart_type(self.clean_message) if is_chart else None
        request_type = "chart" if is_chart else "qa"
        chart_intent_source = "explicit" if is_chart else ("suppressed" if chart_suppressed else None)
        _record_timing(
            "chart_intent_detection",
            llm=False,
            request_type=request_type,
            chart_suppressed=chart_suppressed,
            preferred_chart_type=preferred_chart_type,
        )
        debug_updates["classification_timing"] = {
            "steps": timing_rows,
            "total_ms": round((time.perf_counter() - classification_start) * 1000, 1),
        }

        return ClassificationResult(
            blocked=False,
            normalized_message=normalized_message,
            detected_entities=normalization_result.detected_entities,
            request_type=request_type,
            preferred_chart_type=preferred_chart_type,
            chart_suppressed=chart_suppressed,
            chart_intent_source=chart_intent_source,
            debug_updates=debug_updates,
            
            source_ids=source_routing.source_ids,
            canonical_tables=source_routing.canonical_tables,
            source_confidence=source_routing.confidence,

        )
