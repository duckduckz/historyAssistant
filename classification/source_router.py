from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from historyAssistant.planning._text_utils import as_list, norm_text


YamlDict = Dict[str, Any]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRoutingResult:
    source_ids: List[str]
    canonical_tables: List[str]
    confidence: float
    reason: str
    debug: YamlDict = field(default_factory=dict)


_SOURCE_TABLES = {
    "calls": "historyCalls",
    "locations": "historyLocations",
    "workflows": "historyWorkflows",
}


_ENTITY_SOURCE_WEIGHTS = {
    "tables": 5,
    "metrics": 5,
    "dimensions": 2,
    "time": 2,
    "filters": 1,
}


SOURCE_ROUTER_SYSTEM_TEXT = (
    "You are a data-source router for HistoryAssistant, a hospital analytics chatbot.\n"
    "Choose which Elasticsearch-backed fact table should answer the user's question.\n\n"
    "Available sources:\n"
    "1. calls -> historyCalls\n"
    "   Use for nurse call events, call volume, call type, priority, response time,\n"
    "   answer time, cancellation, call duration, staying time, and call trends.\n\n"
    "2. locations -> historyLocations\n"
    "   Use for tracked person/patient/staff location, where someone was, last seen,\n"
    "   movement, walking, exercise, mobility, activity, moved from/to, zone, area,\n"
    "   ward, previous location, and location at a time.\n\n"
    "3. workflows -> historyWorkflows\n"
    "   Use for workflow/presence records, who was present, nurse/staff presence in\n"
    "   a ward or room, initiator person, presence events, and workflow names.\n\n"
    "Rules:\n"
    "- Return strict JSON only.\n"
    "- Select exactly one primary source unless the user explicitly asks to combine data.\n"
    "- For patient walking/exercise/movement/activity questions, choose locations.\n"
    "- For 'who was present' or nurse/staff present in a room, choose workflows.\n"
    "- For call count, response time, call priority, call type, or cancelled calls, choose calls.\n"
    "- If the question mentions both calls and presence/location, include both source_ids with\n"
    "  the primary source first, but still use only allowed source names.\n\n"
    "Return JSON shape:\n"
    "{\n"
    '  "source_ids": ["calls" | "locations" | "workflows"],\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "reason": "<short reason>"\n'
    "}"
)

#TODO be aware of extra examples should add in to improve the accuracy
_SOURCE_ROUTER_EXAMPLES = [
    {
        "question": "How many calls happened during the night shift?",
        "source_ids": ["calls"],
        "reason": "Call volume question.",
    },
    {
        "question": "How long did patient abc walk on 2024-02-28?",
        "source_ids": ["locations"],
        "reason": "Walking/exercise is movement/location history.",
    },
    {
        "question": "Did patient abc exercise today?",
        "source_ids": ["locations"],
        "reason": "Exercise is inferred from patient movement/location updates.",
    },
    {
        "question": "Where was patient abc last seen?",
        "source_ids": ["locations"],
        "reason": "Last seen asks for tracked location.",
    },
    {
        "question": "Who was present in room K208 during the night shift?",
        "source_ids": ["workflows"],
        "reason": "Presence in a room is workflow presence data.",
    },
    {
        "question": "When a call happened, was a nurse already present?",
        "source_ids": ["calls", "workflows"],
        "reason": "This combines call events with workflow presence records.",
    },
]


def _embedding_route_source(question: str, q_embedding: Any = None) -> SourceRoutingResult:
    """Score each source by semantic similarity between the question and source vocabulary profiles."""
    try:
        from historyAssistant.config import load_source_configs
        from historyAssistant.planning import _entity_embedder as _emb_mod
        score_source = _emb_mod.score_source
        embed_question = _emb_mod.embed_question
    except Exception as exc:
        logger.warning("Embedding source router unavailable: %s", exc)
        return SourceRoutingResult(
            source_ids=["calls"],
            canonical_tables=["historyCalls"],
            confidence=0.35,
            reason="Embedding router unavailable; defaulted to calls.",
            debug={"router": "embedding", "error": str(exc)},
        )

    if q_embedding is None:
        q_embedding = embed_question(question or "")

    scores: Dict[str, float] = {}
    for source_id in _SOURCE_TABLES:
        try:
            bundle = load_source_configs(source_id)
            scores[source_id] = score_source(question or "", source_id, bundle.vocabulary or {}, q_embedding=q_embedding)
        except Exception as exc:
            logger.warning("Could not score source %s: %s", source_id, exc)
            scores[source_id] = 0.0

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_source, top_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = top_score - second_score

    if top_score < 0.30:
        confidence = 0.35
    elif margin >= 0.10:
        confidence = 0.9
    else:
        confidence = 0.6

    return SourceRoutingResult(
        source_ids=[top_source],
        canonical_tables=[_SOURCE_TABLES[top_source]],
        confidence=confidence,
        reason=f"Embedding matched {top_source} (score={top_score:.3f}, margin={margin:.3f}).",
        debug={"router": "embedding", "scores": {k: round(v, 3) for k, v in scores.items()}},
    )


def _as_list(value: Any) -> list[Any]:
    return as_list(value)


def _normalise_key(value: Any) -> str:
    return norm_text(value)


def _source_id_for_table(canonical_table: str) -> Optional[str]:
    table = str(canonical_table or "").strip()
    for source_id, source_table in _SOURCE_TABLES.items():
        if source_table.lower() == table.lower():
            return source_id
    return None


def _add_entity_mapping(index: dict[str, dict[str, str]], category: str, key: Any, source_id: str) -> None:
    normalised = _normalise_key(key)
    if normalised:
        index.setdefault(category, {})[normalised] = source_id


@lru_cache(maxsize=1)
def _load_source_entity_index() -> dict[str, dict[str, str]]:

    index: dict[str, dict[str, str]] = {}
    try:
        from historyAssistant.config import load_source_configs
    except Exception as exc:  
        logger.warning("Could not import source config loader for routing index: %s", exc)
        return index

    for source_id, canonical_table in _SOURCE_TABLES.items():
        _add_entity_mapping(index, "tables", canonical_table, source_id)
        _add_entity_mapping(index, "tables", source_id, source_id)
        try:
            bundle = load_source_configs(source_id)
        except Exception as exc:
            logger.warning("Could not load %s source config for routing index: %s", source_id, exc)
            continue

        for metric_name, metric_cfg in (bundle.metric_catalog.get("metrics") or {}).items():
            _add_entity_mapping(index, "metrics", metric_name, source_id)
            if isinstance(metric_cfg, dict):
                table_source = _source_id_for_table(str(metric_cfg.get("table") or ""))
                if table_source:
                    _add_entity_mapping(index, "metrics", metric_name, table_source)

        vocab = bundle.vocabulary or {}
        for table_name in (vocab.get("tables") or {}).keys():
            _add_entity_mapping(index, "tables", table_name, source_id)
        for metric_name in (vocab.get("metrics") or {}).keys():
            _add_entity_mapping(index, "metrics", metric_name, source_id)
        for dimension_name in (vocab.get("dimensions") or {}).keys():
            _add_entity_mapping(index, "dimensions", dimension_name, source_id)
            _add_entity_mapping(index, "filters", dimension_name, source_id)
        for time_name in (vocab.get("time") or {}).keys():
            _add_entity_mapping(index, "time", time_name, source_id)

        for table_cfg in (bundle.table_registry.get("tables") or {}).values():
            if not isinstance(table_cfg, dict):
                continue
            for metric_name in _as_list(table_cfg.get("metrics")):
                _add_entity_mapping(index, "metrics", metric_name, source_id)
            for dimension_name in _as_list(table_cfg.get("dimensions")):
                _add_entity_mapping(index, "dimensions", dimension_name, source_id)
                _add_entity_mapping(index, "filters", dimension_name, source_id)
            for time_name in _as_list(table_cfg.get("time_field")) + _as_list(table_cfg.get("timestamp_fields")):
                _add_entity_mapping(index, "time", time_name, source_id)

    return index


def _entity_route_source(detected_entities: YamlDict | None) -> SourceRoutingResult | None:
    if not detected_entities:
        return None

    entity_index = _load_source_entity_index()
    hits: dict[str, int] = {}
    matched: list[YamlDict] = []

    for category, values in detected_entities.items():
        category_index = entity_index.get(str(category), {})
        weight = _ENTITY_SOURCE_WEIGHTS.get(str(category), 1)
        for value in _as_list(values):
            source_id = category_index.get(_normalise_key(value))
            if not source_id:
                continue
            hits[source_id] = hits.get(source_id, 0) + weight
            matched.append({"category": category, "value": value, "source_id": source_id, "weight": weight})

    if not hits:
        return None

    sorted_hits = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    top_source, top_score = sorted_hits[0]
    second_score = sorted_hits[1][1] if len(sorted_hits) > 1 else 0
    confidence = 0.95 if top_score >= second_score + 3 else 0.65

    return SourceRoutingResult(
        source_ids=[top_source],
        canonical_tables=[_SOURCE_TABLES[top_source]],
        confidence=confidence,
        reason=f"Matched canonical entities to {top_source} source.",
        debug={"router": "entity_rules", "hits": hits, "matched_entities": matched},
    )




def _parse_source_router_response(text: str) -> YamlDict:
    stripped = (text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in source router response: {text[:200]}")
    return json.loads(match.group(0))


def _normalise_source_ids(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = raw_value
    else:
        values = []

    normalised: list[str] = []
    for value in values:
        source_id = str(value or "").strip().lower()
        if source_id in _SOURCE_TABLES and source_id not in normalised:
            normalised.append(source_id)
    return normalised


def route_source(
    question: str,
    detected_entities: YamlDict | None = None,
    *,
    client: Any | None = None,
    model: Optional[str] = None,
    original_question: Optional[str] = None,
) -> SourceRoutingResult:
    # Encode question once and share across both routers.
    q_text = original_question or question or ""
    q_embedding = None
    try:
        from historyAssistant.planning import _entity_embedder as _emb_mod
        q_embedding = _emb_mod.embed_question(q_text)
    except Exception:
        pass

    entity_result = _entity_route_source(detected_entities)

    embedding_result = _embedding_route_source(q_text, q_embedding=q_embedding)

    if entity_result is not None and entity_result.confidence >= 0.9:
        entity_source = entity_result.source_ids[0] if entity_result.source_ids else None
        emb_source = embedding_result.source_ids[0] if embedding_result.source_ids else None
        if embedding_result.confidence >= 0.85 and emb_source != entity_source:
          
            return embedding_result
        return entity_result

    if embedding_result.confidence >= 0.85:
        return embedding_result

    if entity_result is not None and entity_result.confidence >= 0.6:
        return entity_result
    if embedding_result.confidence >= 0.6:
        return embedding_result

    if client is None:
        return embedding_result

    from historyAssistant.llm.llm_client import call_json_prompt, get_default_model_name

    user_payload: YamlDict = {
        "question": question,
        "original_question": original_question,
        "detected_entities": detected_entities or {},
        "allowed_sources": [
            {"source_id": source_id, "canonical_table": table}
            for source_id, table in _SOURCE_TABLES.items()
        ],
        "examples": _SOURCE_ROUTER_EXAMPLES,
        "entity_router_result": (
            {
                "source_ids": entity_result.source_ids,
                "confidence": entity_result.confidence,
                "reason": entity_result.reason,
            }
            if entity_result is not None
            else None
        ),
        "embedding_router_result": {
            "source_ids": embedding_result.source_ids,
            "confidence": embedding_result.confidence,
            "reason": embedding_result.reason,
            "debug": embedding_result.debug,
        },
    }

    try:
        raw = call_json_prompt(
            client,
            model=model or get_default_model_name(),
            system_text=SOURCE_ROUTER_SYSTEM_TEXT,
            user_payload=user_payload,
            max_tokens=160,
        )
        parsed = _parse_source_router_response(raw)
        source_ids = _normalise_source_ids(parsed.get("source_ids") or parsed.get("source_id"))
        if not source_ids:
            raise ValueError(f"No valid source_ids in source router response: {raw[:200]}")

        raw_confidence = parsed.get("confidence", 0.5)
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        else:
            confidence = 0.5

        return SourceRoutingResult(
            source_ids=source_ids,
            canonical_tables=[_SOURCE_TABLES[source_id] for source_id in source_ids],
            confidence=confidence,
            reason=str(parsed.get("reason") or "LLM selected data source."),
            debug={
                "router": "llm",
                "raw_response": raw,
                "parsed": parsed,
                "entity_fallback": entity_result.debug if entity_result is not None else None,
                "embedding_fallback": embedding_result.debug,
            },
        )
    except Exception as exc:
        logger.warning("LLM source router failed: %s; falling back to embedding result", exc)
        return SourceRoutingResult(
            source_ids=embedding_result.source_ids,
            canonical_tables=embedding_result.canonical_tables,
            confidence=embedding_result.confidence,
            reason=f"{embedding_result.reason} (LLM failed; used embedding fallback)",
            debug={
                "router": "local_fallback",
                "llm_error": str(exc),
                "entity_fallback": entity_result.debug if entity_result is not None else None,
                "embedding_fallback": embedding_result.debug,
            },
        )
