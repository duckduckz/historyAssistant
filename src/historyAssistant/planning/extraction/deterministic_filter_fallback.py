from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from historyAssistant.planning._text_utils import norm_text
from .filter_value_resolver import (
    FilterCandidate,
    compact_value,
    contains_phrase,
    ignored_allowed_value,
)

YamlDict = Dict[str, Any]

_HASH_RE = re.compile(r"(?i)\b([0-9a-f]{64})\b")
_TRUNCATED_HASH_RE = re.compile(r"(?i)\b([0-9a-f]{8})\.\.\.([0-9a-f]{4})\b")
_ROOM_CODE_RE = re.compile(r"(?i)\b((?:K|SANITAIR)\s*\d{1,3}|DVC_[A-Z]+\s*\d{1,3})\b")
_SHIFT_CONTEXT_RE = re.compile(
    r"(?i)(?:(day|evening|night|morning|afternoon)\s+shift"
    r"|shift\s*[:=]?\s*(day|evening|night|morning|afternoon)"
    r"|during\s+(?:the\s+)?(day|evening|night|morning|afternoon))"
)
_BOOLEAN_TRUE_RE = re.compile(
    r"(?i)\b(?:cancelled|canceled|cleared|closed|answered)\s+calls?\b"
    r"|\bcalls?\s+(?:that\s+were\s+)?(?:cancelled|canceled|cleared|closed|answered)\b"
)
_PRIORITY_NUMBER_RE = re.compile(r"(?i)\bpriority\s*(?:=|is)?\s*(\d+)\b|\b(\d+)\s*priority\s+calls?\b")
_PRIORITY_WORD_RE = re.compile(
    r"(?i)\bpriority\s*(?:=|is)?\s*(zero|one|two|three|four|five)\b"
    r"|\b(zero|one|two|three|four|five)\s*priority\s+calls?\b"
)
_NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_GROUP_BY_PREFIX_RE = re.compile(r"(?i)(?:\bby|\bper|\bacross|\bgrouped\s+by|\bbreakdown\s+by)\s*$")
_EXPLICIT_STOP_RE = re.compile(
    r"(?i)\s+(?:at|on|around|in|during|from|to|between|where|which|who|when|"
    r"last|this|yesterday|today|tomorrow|with|for|by|per|across|grouped\s+by|"
    r"had|has|have|was|were|is|are)\b|[?.!,]"
)


def collect_fallback_filter_candidates(
    question: str,
    table_definition: YamlDict,
    *,
    detected_entities: YamlDict | None = None,
) -> List[FilterCandidate]:
    question = str(question or "")
    dim_maps = table_definition.get("dimension_value_maps", {}) or {}
    candidates: List[FilterCandidate] = []

    candidates.extend(_value_map_candidates(question, dim_maps))
    candidates.extend(_allowed_value_candidates(question, dim_maps))
    candidates.extend(_explicit_field_candidates(question, table_definition))
    candidates.extend(_typed_candidates(question, table_definition, detected_entities))
    return candidates


def _value_map_candidates(question: str, dim_maps: YamlDict) -> List[FilterCandidate]:
    candidates: List[FilterCandidate] = []
    for field, info in dim_maps.items():
        for label, mapped in sorted(((info or {}).get("value_map", {}) or {}).items(), key=lambda item: len(str(item[0])), reverse=True):
            if contains_phrase(question, label):
                candidates.append(FilterCandidate(field=field, op="in" if isinstance(mapped, list) else "=", raw_value=mapped, source="value_map"))
                break

    priority_field = next((field for field in dim_maps if str(field).lower() == "priority"), None)
    if priority_field:
        match = _PRIORITY_NUMBER_RE.search(question)
        raw = next((group for group in (match.groups() if match else []) if group), None)
        if raw is not None:
            candidates.append(FilterCandidate(field=priority_field, raw_value=int(raw), source="typed_priority"))
        else:
            match = _PRIORITY_WORD_RE.search(question)
            raw_word = next((group for group in (match.groups() if match else []) if group), None)
            if raw_word:
                candidates.append(FilterCandidate(field=priority_field, raw_value=_NUMBER_WORDS[raw_word.lower()], source="typed_priority"))
    return candidates


def _allowed_value_candidates(question: str, dim_maps: YamlDict) -> List[FilterCandidate]:
    candidates: List[FilterCandidate] = []
    used_spans: List[Tuple[int, int]] = []
    for field, info in dim_maps.items():
        values = sorted((info or {}).get("allowed_values", []) or [], key=lambda value: len(str(value)), reverse=True)
        for value in values:
            if ignored_allowed_value(value):
                continue
            match = _find_allowed_value(question, value)
            if match and not _overlaps(match.start(), match.end(), used_spans):
                candidates.append(FilterCandidate(field=field, raw_value=value, source="allowed_value", span=(match.start(), match.end())))
                used_spans.append((match.start(), match.end()))
                break
    return candidates


def _explicit_field_candidates(question: str, table_definition: YamlDict) -> List[FilterCandidate]:
    candidates: List[FilterCandidate] = []
    dimensions = set(table_definition.get("dimensions", []) or [])
    for concept, field in (table_definition.get("semantic_fields", {}) or {}).items():
        field = str(field)
        if field not in dimensions:
            continue
        for label in _field_labels(concept, field):
            raw_value = _value_after_label(question, label)
            if raw_value:
                candidates.append(FilterCandidate(field=field, raw_value=raw_value, source=f"explicit:{label}"))
                break
    return candidates


def _typed_candidates(question: str, table_definition: YamlDict, detected_entities: YamlDict | None) -> List[FilterCandidate]:
    semantic_fields = table_definition.get("semantic_fields", {}) or {}
    dim_maps = table_definition.get("dimension_value_maps", {}) or {}
    candidates: List[FilterCandidate] = []

    room_field = semantic_fields.get("room")
    room_match = _ROOM_CODE_RE.search(question)
    if room_field and room_match:
        candidates.append(FilterCandidate(field=room_field, raw_value=room_match.group(1), source="room_code"))

    person_field = semantic_fields.get("person")
    if person_field:
        full_hash = _HASH_RE.search(question)
        if full_hash:
            candidates.append(FilterCandidate(field=person_field, raw_value=full_hash.group(1).lower(), source="person_hash"))
        else:
            truncated = _TRUNCATED_HASH_RE.search(question)
            if truncated:
                prefix, suffix = truncated.groups()
                candidates.append(FilterCandidate(field=person_field, op="wildcard", raw_value=f"{prefix.lower()}*{suffix.lower()}", source="person_hash"))

    shift_field = next((field for field in dim_maps if str(field).lower() == "shift"), None)
    shift_match = _SHIFT_CONTEXT_RE.search(question)
    shift_value = next((group for group in (shift_match.groups() if shift_match else []) if group), None)
    if not shift_value and "shift" in ((detected_entities or {}).get("dimensions", []) or []):
        shift_value = next((value for value in ["morning", "afternoon", "evening", "night", "day"] if contains_phrase(question, value)), None)
    if shift_field and shift_value:
        candidates.append(FilterCandidate(field=shift_field, raw_value=shift_value.lower(), source="shift_context"))

    cancelled_field = next((field for field in dim_maps if str(field).lower() == "is_cancelled"), None)
    if cancelled_field and _BOOLEAN_TRUE_RE.search(question):
        candidates.append(FilterCandidate(field=cancelled_field, raw_value=True, source="boolean_phrase"))

    person_type_field = semantic_fields.get("person_type")
    if person_type_field and not _has_person_reference(question):
        for value in (dim_maps.get(person_type_field, {}) or {}).get("allowed_values", []) or []:
            if not ignored_allowed_value(value) and contains_phrase(question, value):
                candidates.append(FilterCandidate(field=person_type_field, raw_value=value, source="person_type"))
                break
    return candidates


def _field_labels(concept: str, field: str) -> List[str]:
    labels = {_humanize(concept), _humanize(field), str(concept), str(field)}
    if field.endswith("_name"):
        labels.add(_humanize(field[:-5]))
    return sorted((label for label in labels if label), key=len, reverse=True)


def _value_after_label(question: str, label: str) -> str:
    match = re.search(rf"(?i)\b{re.escape(label)}\b\s*(?:=|is|named|called)?\s+", question)
    if not match or _GROUP_BY_PREFIX_RE.search(question[: match.start()]):
        return ""
    tail = question[match.end() :]
    stop = _EXPLICIT_STOP_RE.search(tail)
    value = tail[: stop.start() if stop else len(tail)].strip(" :='\"")
    return "" if _looks_like_non_value(value) else value


def _find_allowed_value(question: str, value: Any) -> re.Match[str] | None:
    text = str(value or "").strip()
    if not text or len(compact_value(text)) < 4:
        return None
    return re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])", question)


def _looks_like_non_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if re.search(r"(?i)^(?:custom_range|date_range|last_|this_|today|yesterday|tomorrow|in\s+|during\s+|from\s+|to\s+|between\s+|on\s+|at\s+)", text):
        return True
    return bool(re.fullmatch(r"(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?", text))


def _has_person_reference(question: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:from|by|for|of)\s+(?:user|patient|person|caller|staff)\b"
            r"|\b(?:user|patient|person|caller|staff)\s+[0-9a-f]{8}",
            question,
        )
    )


def _overlaps(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    return any(existing_start < end and start < existing_end for existing_start, existing_end in spans)


def _humanize(value: Any) -> str:
    return norm_text(value)
