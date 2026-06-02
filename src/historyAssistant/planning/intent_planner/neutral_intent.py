from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from historyAssistant.config import ConfigBundle, load_source_configs
from historyAssistant.planning import normalize_question
from historyAssistant.planning._text_utils import as_list, norm_text

YamlDict = Dict[str, Any]

_SOURCE_TABLES = {
    "calls": "historyCalls",
    "locations": "historyLocations",
    "workflows": "historyWorkflows",
}


@dataclass(frozen=True)
class SourceIntentResolution:
    source_id: str
    target_table: str
    bundle: ConfigBundle
    normalized_question: str
    detected_entities: YamlDict
    confidence: float
    changed_source: bool
    reason: str
    debug: YamlDict = field(default_factory=dict)

    def to_dict(self) -> YamlDict:
        data = asdict(self)
        data.pop("bundle", None)
        return data


def resolve_source_intent(
    *,
    question: str,
    initial_source_id: str,
    initial_target_table: str,
    initial_bundle: ConfigBundle,
    initial_normalized_question: str,
    initial_detected_entities: Optional[YamlDict],
) -> SourceIntentResolution:
    
    scored = [_score_source(question, source_id) for source_id in _SOURCE_TABLES]
    ranked = sorted(scored, key=lambda row: row["score"], reverse=True)
    top = ranked[0] if ranked else {"source_id": initial_source_id, "score": 0.0}
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0

    initial_score = next(
        (row["score"] for row in scored if row["source_id"] == initial_source_id),
        0.0,
    )
    margin = float(top["score"]) - max(float(second_score), 0.0)
    changed = (
        top["source_id"] != initial_source_id
        and float(top["score"]) >= 5.0
        and (
            margin >= 2.0
            or float(top["score"]) >= initial_score + 3.0
            or bool(top.get("explicit_metric_hits"))
        )
    )

    selected_source = str(top["source_id"]) if changed else initial_source_id
    if selected_source not in _SOURCE_TABLES:
        selected_source = initial_source_id
        changed = False

    if changed:
        selected_bundle = load_source_configs(selected_source)
        selected_table = _SOURCE_TABLES[selected_source]
        source_normalization = normalize_question(question, selected_bundle.vocabulary)
        normalized_question = source_normalization.normalized_question
        detected_entities = source_normalization.detected_entities
        reason = f"neutral_source_intent:{initial_source_id}->{selected_source}"
    else:
        selected_bundle = initial_bundle
        selected_table = initial_target_table
        normalized_question = initial_normalized_question
        detected_entities = initial_detected_entities or {}
        reason = f"neutral_source_intent:kept_{selected_source}"

    confidence = _confidence(float(top["score"]), margin, changed)
    return SourceIntentResolution(
        source_id=selected_source,
        target_table=selected_table,
        bundle=selected_bundle,
        normalized_question=normalized_question,
        detected_entities=detected_entities,
        confidence=confidence,
        changed_source=changed,
        reason=reason,
        debug={
            "ranked_sources": ranked,
            "initial_source_id": initial_source_id,
            "initial_score": round(initial_score, 3),
            "top_score": round(float(top["score"]), 3),
            "margin": round(margin, 3),
        },
    )


def _confidence(score: float, margin: float, changed: bool) -> float:
    if score >= 10 and margin >= 4:
        return 0.95
    if changed:
        return 0.88
    if score >= 5:
        return 0.78
    return 0.6


def _score_source(question: str, source_id: str) -> YamlDict:
    bundle = load_source_configs(source_id)
    text = _normalise_text(question)
    tokens = set(_tokens(text))
    intent_cfg = _source_intent_config(bundle)
    source_anchor_hits = _matching_terms(text, intent_cfg.get("anchors", []))
    generic_terms = set(_normalise_text(term) for term in intent_cfg.get("generic_terms", []) if _normalise_text(term))
    owned_phrase_hits = _owned_phrase_hits(text, intent_cfg.get("intent_phrases", {}))
    phrases = _source_phrases(bundle)
    score = 0.0
    hits: List[YamlDict] = []
    explicit_metric_hits: List[str] = []

    for anchor in source_anchor_hits:
        contribution = 5.0
        score += contribution
        hits.append(
            {
                "category": "source_anchor",
                "owner": source_id,
                "text": anchor,
                "contribution": round(contribution, 3),
                "exact": True,
            }
        )

    for owner, phrase_text in owned_phrase_hits:
        contribution = 14.0
        score += contribution
        explicit_metric_hits.append(str(owner))
        hits.append(
            {
                "category": "intent_phrase",
                "owner": owner,
                "text": phrase_text,
                "contribution": round(contribution, 3),
                "exact": True,
            }
        )

    for phrase in phrases:
        phrase_text = _normalise_text(phrase["text"])
        if not phrase_text:
            continue
        phrase_tokens = set(_tokens(phrase_text))
        if not phrase_tokens:
            continue

        exact = _contains_phrase(text, phrase_text)
        overlap = len(tokens & phrase_tokens) / max(len(phrase_tokens), 1)
        contribution = 0.0
        if exact:
            contribution += float(phrase["weight"])
            generic_unanchored = (
                phrase_text in generic_terms
                and not source_anchor_hits
                and not owned_phrase_hits
            )
            if generic_unanchored:
                contribution *= 0.2
            if phrase["category"] == "metrics" and not generic_unanchored:
                explicit_metric_hits.append(str(phrase["owner"]))
        elif overlap >= 0.75 and len(phrase_tokens) >= 2:
            contribution += float(phrase["weight"]) * overlap * 0.45
        elif overlap >= 0.5 and len(phrase_tokens) >= 3:
            contribution += float(phrase["weight"]) * overlap * 0.25

        if contribution:
            score += contribution
            hits.append(
                {
                    "category": phrase["category"],
                    "owner": phrase["owner"],
                    "text": phrase["text"],
                    "contribution": round(contribution, 3),
                    "exact": exact,
                    "generic_downweighted": bool(
                        exact
                        and phrase_text in generic_terms
                        and not source_anchor_hits
                        and not owned_phrase_hits
                    ),
                }
            )

    hits = sorted(hits, key=lambda row: row["contribution"], reverse=True)[:8]
    return {
        "source_id": source_id,
        "target_table": _SOURCE_TABLES[source_id],
        "score": round(score, 3),
        "hits": hits,
        "explicit_metric_hits": sorted(set(explicit_metric_hits)),
        "source_anchor_hits": source_anchor_hits,
    }


def _source_phrases(bundle: ConfigBundle) -> List[YamlDict]:
    phrases: List[YamlDict] = []
    vocab = bundle.vocabulary or {}

    for category, weight in (("tables", 8.0), ("metrics", 9.0), ("dimensions", 2.0)):
        for owner, cfg in (vocab.get(category) or {}).items():
            phrases.append({"category": category, "owner": owner, "text": owner, "weight": weight})
            if isinstance(cfg, dict):
                for text in _as_list(cfg.get("synonyms")):
                    phrases.append({"category": category, "owner": owner, "text": text, "weight": weight})
                for text in _as_list(cfg.get("aliases_for_embedding")):
                    phrases.append({"category": category, "owner": owner, "text": text, "weight": weight * 0.8})
                if category in {"tables", "metrics"} and cfg.get("description"):
                    phrases.append(
                        {
                            "category": category,
                            "owner": owner,
                            "text": cfg.get("description"),
                            "weight": weight * 0.45,
                        }
                    )

    for metric_name, cfg in (bundle.metric_catalog.get("metrics") or {}).items():
        phrases.append({"category": "metrics", "owner": metric_name, "text": metric_name, "weight": 7.0})
        if isinstance(cfg, dict) and cfg.get("description"):
            phrases.append(
                {
                    "category": "metrics",
                    "owner": metric_name,
                    "text": cfg.get("description"),
                    "weight": 4.0,
                }
            )

    for table_name, cfg in (bundle.table_registry.get("tables") or {}).items():
        phrases.append({"category": "tables", "owner": table_name, "text": table_name, "weight": 7.0})
        if isinstance(cfg, dict) and cfg.get("description"):
            phrases.append(
                {
                    "category": "tables",
                    "owner": table_name,
                    "text": cfg.get("description"),
                    "weight": 4.0,
                }
            )

    return phrases


def _normalise_text(value: Any) -> str:
    return norm_text(value)


def _tokens(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _as_list(value: Any) -> Iterable[Any]:
    return as_list(value)


def _source_intent_config(bundle: ConfigBundle) -> YamlDict:
    cfg = (bundle.vocabulary or {}).get("source_intent", {}) or {}
    return cfg if isinstance(cfg, dict) else {}


def _matching_terms(text: str, terms: Iterable[Any]) -> List[str]:
    hits: List[str] = []
    for raw in terms or []:
        term = _normalise_text(raw)
        if term and _contains_phrase(text, term):
            hits.append(term)
    return sorted(set(hits))


def _owned_phrase_hits(text: str, intent_phrases: Any) -> List[tuple[str, str]]:
    hits: List[tuple[str, str]] = []
    if not isinstance(intent_phrases, dict):
        return hits
    for owner, phrases in intent_phrases.items():
        for raw in _as_list(phrases):
            phrase = _normalise_text(raw)
            if phrase and _contains_phrase(text, phrase):
                hits.append((str(owner), phrase))
    return sorted(set(hits))
