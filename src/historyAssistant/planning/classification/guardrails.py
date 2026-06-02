from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from historyAssistant.config import ConfigBundle, load_all_configs

YamlDict = Dict[str, Any]

logger = logging.getLogger(__name__)


DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.6
DEFAULT_IN_SCOPE_MARGIN = 0.1
DEFAULT_MIN_IN_SCOPE_SCORE = 0.0  
LOCAL_EMBEDDING_MODEL = r"C:\models\all-MiniLM-L6-v2"

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def normalize_text(text: str) -> str:
    clean = text.lower().strip()
    clean = re.sub(r"[\t\n\r]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"[^a-z0-9\s\-_/,:;?]+", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


@dataclass(frozen=True)
class GuardrailHit:
    rule_id: str
    category: str
    reason: str
    response_key: str
    matched_keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    response_text: Optional[str] = None
    hits: List[GuardrailHit] = field(default_factory=list)


_st_model: Optional[SentenceTransformer] = None


def get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, local_files_only=True)
        logger.info("Loaded local embedding model: %s", LOCAL_EMBEDDING_MODEL)
    return _st_model


def encode_texts(texts: List[str]) -> Any:
    model = get_st_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def build_seed_texts(rule: YamlDict) -> List[str]:
    seeds: List[str] = []
    reason_text = str(rule.get("reason", "")).strip()
    if reason_text:
        seeds.append(reason_text)
    for example in rule.get("examples", []):
        text = str(example).strip()
        if text:
            seeds.append(text)
    for keyword in rule.get("keywords", []):
        text = str(keyword).strip()
        if text:
            seeds.append(text)
    return seeds


@dataclass
class _RuleEmbedding:
    rule: YamlDict
    embeddings: Any


_embedding_cache: Optional[List[_RuleEmbedding]] = None
_embedding_cache_signature: Optional[Tuple[Tuple[str, ...], ...]] = None


def _rules_signature(rules: List[YamlDict]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(build_seed_texts(rule)) for rule in rules)


def _build_rule_embedding_cache(
    rules: List[YamlDict],
) -> Tuple[List[_RuleEmbedding], Tuple[Tuple[str, ...], ...], int]:
    signature = _rules_signature(rules)
    all_texts: List[str] = []
    rule_text_ranges: List[Tuple[int, int]] = []

    for rule in rules:
        seeds = build_seed_texts(rule)
        start = len(all_texts)
        all_texts.extend(seeds)
        rule_text_ranges.append((start, len(all_texts)))

    if not all_texts:
        return [], signature, 0

    all_embeddings = encode_texts(all_texts)
    cache = [
        _RuleEmbedding(rule=rule, embeddings=all_embeddings[start:end])
        for rule, (start, end) in zip(rules, rule_text_ranges)
    ]
    return cache, signature, len(all_texts)


def build_embedding_cache(rules: List[YamlDict]) -> List[_RuleEmbedding]:
    global _embedding_cache, _embedding_cache_signature

    current_signature = _rules_signature(rules)
    if _embedding_cache is not None and _embedding_cache_signature == current_signature:
        return _embedding_cache

    _embedding_cache, _embedding_cache_signature, text_count = _build_rule_embedding_cache(rules)
    logger.info(
        "built guardrail embedding cache: %d rules, %d seed texts",
        len(_embedding_cache),
        text_count,
    )
    return _embedding_cache


_in_scope_embeddings: Optional[Any] = None
_in_scope_texts_signature: Optional[Tuple[str, ...]] = None


def build_in_scope_texts(cfg: YamlDict) -> List[str]:
    texts: List[str] = []

    allowlist = cfg.get("allowlist_hints", {})
    for focus in allowlist.get("supported_focus", []):
        text = str(focus).strip()
        if text:
            texts.append(text)

    for example in allowlist.get("in_scope_examples", []):
        text = str(example).strip()
        if text:
            texts.append(text)

    return texts


def build_in_scope_cache(cfg: YamlDict) -> Optional[Any]:
    global _in_scope_embeddings, _in_scope_texts_signature

    texts = build_in_scope_texts(cfg)
    if not texts:
        return None

    current_sig = tuple(texts)
    if _in_scope_embeddings is not None and _in_scope_texts_signature == current_sig:
        return _in_scope_embeddings

    _in_scope_embeddings = encode_texts(texts)
    _in_scope_texts_signature = current_sig
    logger.info("Built in-scope embedding cache: %d texts", len(texts))
    return _in_scope_embeddings


def compute_in_scope_score(q_embedding: Any, cfg: YamlDict) -> float:
    in_scope_embs = build_in_scope_cache(cfg)
    if in_scope_embs is None or len(in_scope_embs) == 0:
        return 0.0

    scores = in_scope_embs @ q_embedding
    return float(np.max(scores))


def semantic_check(question: str, rules: List[YamlDict], threshold: float, in_scope_margin: float = DEFAULT_IN_SCOPE_MARGIN, cfg: Optional[YamlDict] = None) -> Optional[Tuple[YamlDict, float]]:
    cache = build_embedding_cache(rules)
    if not cache:
        return None

    q_embedding = encode_texts([normalize_text(question)])[0]

    best_score = 0.0
    best_rule: Optional[YamlDict] = None

    for entry in cache:
        # for the entry's embeddings shape here is (N, dim)
        # i use cosine similarity since embeddings are normalized here
        scores = entry.embeddings @ q_embedding
        max_score = float(np.max(scores))
        if max_score > best_score:
            best_score = max_score
            best_rule = entry.rule

    if best_score < threshold or best_rule is None:
        return None

    if cfg is not None:
        in_scope_score = compute_in_scope_score(q_embedding, cfg)
        if in_scope_score > 0 and best_score < (in_scope_score + in_scope_margin):
            logger.debug(
                "Question closer to in-scope (%.2f) than out-of-scope (%.2f), not blocking",
                in_scope_score, best_score
            )
            return None

    return best_rule, best_score


def _make_hit(rule: YamlDict, *, reason: Optional[str] = None, matched_keywords: Optional[List[str]] = None) -> GuardrailHit:
    return GuardrailHit(
        rule_id=rule.get("id", "unknown_rule"),
        category=rule.get("category", "out_of_scope"),
        reason=reason if reason is not None else rule.get("reason", ""),
        response_key=rule.get("response_key", "out_of_scope"),
        matched_keywords=matched_keywords or [],
    )


def _keyword_hits(clean_question: str, rules: List[YamlDict]) -> List[GuardrailHit]:
    hits: List[GuardrailHit] = []
    for rule in rules:
        matched_keywords = [
            keyword
            for keyword in rule.get("keywords", [])
            if (normalized_keyword := normalize_text(keyword)) and normalized_keyword in clean_question
        ]
        if matched_keywords:
            hits.append(_make_hit(rule, matched_keywords=matched_keywords))
    return hits


def screen_question_for_guardrails(question: str, *, config_bundle: Optional[ConfigBundle] = None) -> GuardrailResult:
    bundle = config_bundle or load_all_configs()
    cfg = bundle.unsupported_topic
    responses = cfg.get("responses", {})
    rules = cfg.get("unsupported_topics", [])

    # read configurable thresholds from policy section
    policy = cfg.get("policy", {})
    semantic_threshold = policy.get("semantic_threshold", DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD)
    in_scope_margin = policy.get("in_scope_margin", DEFAULT_IN_SCOPE_MARGIN)
    min_in_scope_score = policy.get("min_in_scope_score", DEFAULT_MIN_IN_SCOPE_SCORE)

    clean_question = normalize_text(question)
    hits = _keyword_hits(clean_question, rules)

    if hits:
        first_hit = hits[0]
        response_text = responses.get(first_hit.response_key)
        return GuardrailResult(blocked=True, response_text=response_text, hits=hits)

    try:
        result = semantic_check(
            question,
            rules,
            threshold=semantic_threshold,
            in_scope_margin=in_scope_margin,
            cfg=cfg,
        )
        if result is not None:
            matched_rule, score = result
            hit = _make_hit(
                matched_rule,
                reason=f"{matched_rule.get('reason', '')} (semantic match, score={score:.2f})",
                matched_keywords=[f"[semantic:{score:.2f}]"],
            )
            response_text = responses.get(hit.response_key)
            return GuardrailResult(blocked=True, response_text=response_text, hits=[hit])
    except Exception as exc:
        logger.warning("Semantic guardrail check failed, continuing with keyword-only: %s", exc)

    # A scenaio: block if question is too far from all in-scope examples
    if min_in_scope_score > 0.0:
        try:
            q_embedding = encode_texts([normalize_text(question)])[0]
            in_scope_score = compute_in_scope_score(q_embedding, cfg)
            if in_scope_score < min_in_scope_score:
                logger.debug(
                    "Question failed minimum in-scope threshold (%.2f < %.2f), blocking as out-of-scope",
                    in_scope_score, min_in_scope_score,
                )
                response_text = responses.get("out_of_scope")
                hit = GuardrailHit(
                    rule_id="out_of_scope_low_in_scope_score",
                    category="out_of_scope",
                    reason=f"Question does not appear to be about nurse call data (in-scope score={in_scope_score:.2f} < min={min_in_scope_score:.2f})",
                    response_key="out_of_scope",
                    matched_keywords=[f"[in_scope_score:{in_scope_score:.2f}]"],
                )
                return GuardrailResult(blocked=True, response_text=response_text, hits=[hit])
        except Exception as exc:
            logger.warning("Min in-scope score check failed, continuing: %s", exc)

    return GuardrailResult(blocked=False, response_text=None, hits=[])


__all__ = [
    "GuardrailHit",
    "GuardrailResult",
    "encode_texts",
    "screen_question_for_guardrails",
]
