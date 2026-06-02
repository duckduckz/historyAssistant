"""Extract numeric threshold filters on measure fields from free text.

General, source-agnostic: turns phrases like "gaps larger than 30 minutes",
"response time over 5 minutes", or "staying time under 2 hours" into a canonical
filter ``{"field": <measure_field>, "op": ">|>=|<|<=", "value": <number>}``.

The measure field is identified from the table registry's ``measure_fields`` and
``measure_field_maps`` — by the field's own name tokens (e.g. ``time_gap_minutes``
yields the term "gap"), by an optional ``synonyms`` list, and the threshold value
is converted into the field's native unit (so "5 minutes" on a seconds field
becomes 300). No per-question keywords or regexes are hardcoded.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

YamlDict = Dict[str, object]

# Comparator phrases/symbols mapped to canonical ops. Compound comparators
# (>=, <=) are listed first so they win over the bare > / < that they contain
# (e.g. "no more than" must not be read as "more than").
_COMPARATORS: List[Tuple[str, str]] = [
    (">=", r"at least|no less than|no fewer than|minimum(?:\s+of)?|>="),
    ("<=", r"at most|no more than|no greater than|up to|maximum(?:\s+of)?|<="),
    (">", r"greater than|larger than|longer than|more than|over|above|exceeding|beyond|>"),
    ("<", r"less than|fewer than|smaller than|shorter than|under|below|<"),
]

_NUMBER = r"(\d+(?:\.\d+)?)"

# Phrase unit -> seconds, used to convert a spoken unit into the field's unit.
_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}

# Field-map unit names -> seconds, used as the conversion target.
_FIELD_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1,
    "minute": 60, "minutes": 60, "min": 60,
    "hour": 3600, "hours": 3600,
}

# Calendar units signal a date phrase ("over the last 30 days"), never a measure
# threshold — reject them so temporal language doesn't become a numeric filter.
_DATE_UNITS = {
    "day", "days", "week", "weeks", "month", "months",
    "year", "years", "quarter", "quarters",
}

# Tokens in a measure field name that don't help identify it in language.
_GENERIC_TOKENS = {
    "time", "sec", "secs", "second", "seconds", "min", "mins", "minute",
    "minutes", "hour", "hours", "count", "value", "values", "id", "num",
    "number", "total", "avg", "average", "per",
}


def extract_measure_threshold_filters(
    question: str,
    table_definition: YamlDict,
) -> List[YamlDict]:
    """Return measure-field threshold filters found in ``question``.

    Empty list when no measure field is configured, no comparator+number phrase
    is present, or no measure field can be associated with a threshold.
    """
    text = str(question or "")
    measure_fields = list(table_definition.get("measure_fields", []) or [])
    if not text.strip() or not measure_fields:
        return []

    measure_maps = table_definition.get("measure_field_maps", {}) or {}
    field_terms = {
        field: _terms_for_field(field, measure_maps.get(field, {}) or {})
        for field in measure_fields
    }
    low = text.lower()
    mentioned = _mentioned_fields(low, field_terms)

    thresholds = _find_thresholds(low)
    if not thresholds:
        return []

    filters: List[YamlDict] = []
    seen: set[Tuple[str, str, object]] = set()
    for op, number, unit, position in thresholds:
        field = _pick_field(mentioned, measure_fields, position)
        if not field:
            continue
        value = _convert_value(number, unit, measure_maps.get(field, {}) or {})
        key = (field, op, value)
        if key in seen:
            continue
        seen.add(key)
        filters.append({"field": field, "op": op, "value": value, "source": "measure_threshold"})
    return filters


def question_has_measure_threshold(question: str, table_definition: YamlDict) -> bool:
    """Lightweight check used by answer-shape routing."""
    return bool(extract_measure_threshold_filters(question, table_definition))


def _terms_for_field(field: str, field_map: YamlDict) -> List[str]:
    terms: set[str] = set()
    for synonym in field_map.get("synonyms", []) or []:
        text = str(synonym).strip().lower()
        if text:
            terms.add(text)

    tokens = [token for token in re.split(r"[_\s]+", str(field).lower()) if token]
    for token in tokens:
        if token not in _GENERIC_TOKENS:
            terms.add(token)

    # Multi-word phrase from the name minus pure unit/id suffixes
    # (e.g. response_time_sec -> "response time").
    phrase_tokens = [token for token in tokens if token not in {"sec", "secs", "min", "mins", "id"}]
    if len(phrase_tokens) >= 2:
        terms.add(" ".join(phrase_tokens))
    return sorted(terms, key=len, reverse=True)


def _mentioned_fields(low_text: str, field_terms: Dict[str, List[str]]) -> Dict[str, int]:
    """Map each mentioned measure field to the earliest character index it appears at."""
    mentioned: Dict[str, int] = {}
    for field, terms in field_terms.items():
        for term in terms:
            idx = _term_index(low_text, term)
            if idx >= 0:
                mentioned[field] = idx if field not in mentioned else min(mentioned[field], idx)
    return mentioned


def _term_index(low_text: str, term: str) -> int:
    # Single words match plural/suffixed forms ("gap" -> "gaps"); phrases match as-is.
    pattern = r"\b" + re.escape(term) + (r"\w*" if " " not in term else "")
    match = re.search(pattern, low_text)
    return match.start() if match else -1


def _find_thresholds(low_text: str) -> List[Tuple[str, float, Optional[str], int]]:
    results: List[Tuple[str, float, Optional[str], int]] = []
    consumed: List[Tuple[int, int]] = []
    for op, comparator in _COMPARATORS:
        pattern = re.compile(
            r"(?<![a-z])(?:" + comparator + r")\s*" + _NUMBER + r"\s*([a-z]+)?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(low_text):
            start, end = match.start(), match.end()
            if any(start < c_end and c_start < end for c_start, c_end in consumed):
                continue
            consumed.append((start, end))
            raw_unit = (match.group(2) or "").strip().lower() or None
            if raw_unit in _DATE_UNITS:
                # Temporal phrase ("over the last 30 days"), not a measure threshold.
                continue
            number = float(match.group(1))
            unit = raw_unit if raw_unit in _UNIT_SECONDS else None
            results.append((op, number, unit, start))
    return results


def _pick_field(
    mentioned: Dict[str, int],
    measure_fields: List[str],
    threshold_position: int,
) -> Optional[str]:
    if not mentioned:
        # Require an explicit measure reference so plain numbers in temporal or
        # count phrases ("over the last 30 days", "over 30 records") never become
        # a measure threshold.
        return None
    if len(mentioned) == 1:
        return next(iter(mentioned))
    # Multiple measures named: associate with the closest one to the threshold.
    return min(mentioned.items(), key=lambda item: abs(item[1] - threshold_position))[0]


def _convert_value(number: float, unit: Optional[str], field_map: YamlDict) -> object:
    field_unit = str(field_map.get("unit") or "").strip().lower()
    spoken_seconds = _UNIT_SECONDS.get(unit) if unit else None
    field_seconds = _FIELD_UNIT_SECONDS.get(field_unit)
    if spoken_seconds and field_seconds and spoken_seconds != field_seconds:
        number = number * spoken_seconds / field_seconds
    if float(number).is_integer():
        return int(number)
    return round(float(number), 4)


__all__ = ["extract_measure_threshold_filters", "question_has_measure_threshold"]
