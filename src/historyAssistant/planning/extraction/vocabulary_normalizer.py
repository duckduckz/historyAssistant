from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

YamlDict = Dict[str, Any]


@dataclass
class NormalizationResult:
    original_question: str
    normalized_question: str
    replacements: List[Tuple[str, str, str]]
    detected_entities: Dict[str, List[str]]  


_ROOM_CODE_PATTERNS = [
    re.compile(r"(?i)\b(K|SANITAIR)\s*(\d{1,3})\b"),
    re.compile(r"(?i)\b(DVC_[A-Z]+)\s*(\d{1,3})\b"),
]


def _normalize_room_codes(question: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    # rewrite room codes so all downstream stages see the canonical "K 226" form.
    replacements: List[Tuple[str, str, str]] = []
    rewritten = question
    for pattern in _ROOM_CODE_PATTERNS:
        def _sub(match: re.Match) -> str:
            prefix = match.group(1).upper()
            digits = match.group(2).zfill(3)
            canonical = f"{prefix} {digits}"
            if match.group(0) != canonical:
                replacements.append((match.group(0), canonical, "room_code"))
            return canonical
        rewritten = pattern.sub(_sub, rewritten)
    return rewritten, replacements
    
def _synonym_variants(synonym: str) -> List[str]:
    text = " ".join(str(synonym or "").strip().lower().split())
    if not text:
        return []

    variants = [text]
    words = text.split()
    if not words:
        return variants

    last = words[-1]
    stems: List[str] = []
    if last.endswith("s") and len(last) > 2:
        stems = []
    elif last.endswith("y") and len(last) > 2:
        stems.append(last[:-1] + "ies")
    elif last.endswith(("ch", "sh", "x", "z")):
        stems.append(last + "es")
    else:
        stems.append(last + "s")

    for replacement in stems:
        if replacement and replacement != last:
            candidate = " ".join(words[:-1] + [replacement])
            if candidate not in variants:
                variants.append(candidate)
    return variants


_CATEGORY_PRIORITY = ["metrics", "dimensions", "time", "bucket_intervals", "tables"]


def _build_synonym_lookup(vocabulary: YamlDict) -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}

    ordered = sorted(
        vocabulary.keys(),
        key=lambda c: _CATEGORY_PRIORITY.index(c) if c in _CATEGORY_PRIORITY else len(_CATEGORY_PRIORITY),
    )

    for category in ordered:
        category_data = vocabulary.get(category, {})
        if not isinstance(category_data, dict):
            continue

        for canonical_name, entry in category_data.items():
            if not isinstance(entry, dict):
                continue
            synonyms = entry.get("synonyms", [])
            if not isinstance(synonyms, list):
                continue

            for synonym in synonyms:
                if isinstance(synonym, str) and synonym.strip():
                    for key in _synonym_variants(synonym):
                        if key not in lookup:
                            lookup[key] = (canonical_name, category)

    return lookup


def _sort_synonyms_by_length(lookup: Dict[str, Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    # check longer phrases first 
    items = [(syn, canonical, cat) for syn, (canonical, cat) in lookup.items()]
    return sorted(items, key=lambda x: len(x[0]), reverse=True)


def normalize_question(question: str,vocabulary: YamlDict, *,replace_in_text: bool = True) -> NormalizationResult:
    pre_question, room_replacements = _normalize_room_codes(question) if replace_in_text else (question, [])

    if not vocabulary:
        return NormalizationResult(
            original_question=question,
            normalized_question=pre_question,
            replacements=list(room_replacements),
            detected_entities={},
        )

    lookup = _build_synonym_lookup(vocabulary)
    sorted_synonyms = _sort_synonyms_by_length(lookup)

    normalized = pre_question
    question_lower = pre_question.lower()
    synonym_replacements: List[Tuple[str, str, str]] = []
    detected_entities: Dict[str, List[str]] = {}

    matched_positions: List[Tuple[int, int]] = []

    for synonym, canonical_name, category in sorted_synonyms:
      
        pattern = r'\b' + re.escape(synonym) + r'\b'

        for match in re.finditer(pattern, question_lower):
            start, end = match.start(), match.end()

            overlaps = any(
                not (end <= existing_start or start >= existing_end)
                for existing_start, existing_end in matched_positions
            )

            if not overlaps:
                matched_positions.append((start, end))
                matched_text = pre_question[start:end]
                synonym_replacements.append((matched_text, canonical_name, category))

                if category not in detected_entities:
                    detected_entities[category] = []
                if canonical_name not in detected_entities[category]:
                    detected_entities[category].append(canonical_name)

    if replace_in_text and synonym_replacements:
        sorted_matches = sorted(
            zip(matched_positions, synonym_replacements),
            key=lambda x: x[0][0],
            reverse=True
        )

        for (start, end), (matched_text, canonical_name, category) in sorted_matches:
     
            if matched_text.lower() != canonical_name.lower():
                normalized = normalized[:start] + canonical_name + normalized[end:]

    replacements: List[Tuple[str, str, str]] = list(room_replacements) + synonym_replacements

    return NormalizationResult(
        original_question=question,
        normalized_question=normalized,
        replacements=replacements,
        detected_entities=detected_entities,
    )


__all__ = [
    "NormalizationResult",
    "normalize_question",
   
]
