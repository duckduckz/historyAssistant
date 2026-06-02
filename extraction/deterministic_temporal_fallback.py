from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Dict, Optional

from .temporal_constants import MONTH_MAP, MONTH_PATTERN
from .temporal_value_resolver import (
    DateResult,
    custom_range,
    eu_to_iso,
    month_range,
    parse_iso,
    range_from_iso_values,
    week_range_in_month,
    week_range_in_year,
)

logger = logging.getLogger(__name__)
YamlDict = Dict[str, Any]

try:
    from dateparser import parse as _dateparser_parse
except Exception:
    _dateparser_parse = None

_MONTH_TYPO_RE = re.compile(r"\b([A-Za-z]{4,})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b")
_EU_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_RANGE_RE = re.compile(r"(?i)\b(?:from|between)\s+(\d{4}-\d{1,2}-\d{1,2})\s+(?:and|to)\s+(\d{4}-\d{1,2}-\d{1,2})\b")
_MONTH_DAY_YEAR_RE = re.compile(rf"(?i)\b({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
_DAY_MONTH_YEAR_RE = re.compile(rf"(?i)\b(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})\b")
_MONTH_YEAR_RE = re.compile(rf"(?i)\b({MONTH_PATTERN})\s+(\d{{4}})\b")
_YEAR_RE = re.compile(r"(?i)\b(?:in|of|for|during)?\s*((?:19|20)\d{2})\b")
_WEEK_OF_YEAR_RE = re.compile(r"(?i)\b(first|second|third|fourth|fifth|last|\d{1,2})(?:st|nd|rd|th)?\s+week\s+(?:of|in)\s+(\d{4})\b")
_WEEK_OF_MONTH_RE = re.compile(rf"(?i)\b(first|second|third|fourth|fifth|last|\d{{1,2}})(?:st|nd|rd|th)?\s+week(?:\s+of|\s+in)?\s+({MONTH_PATTERN})(?:\s+in)?\s+(\d{{4}})\b")
_ROLLING_DAYS_RE = re.compile(r"(?i)\b(?:last|past|most recent|recent|previous)\s+(\d+)\s+days?\b")


def extract_dates_deterministic(question: str, date_rules_config: Optional[YamlDict] = None) -> DateResult:
    text = normalise_month_typos(question)
    for handler in _DATE_HANDLERS:
        result = handler(text)
        if result is not None:
            return result

    configured_rule = match_configured_date_alias(text, date_rules_config)
    if configured_rule:
        return configured_rule, None, None

    parsed = _resolve_dateparser_result(text)
    if parsed is not None:
        logger.debug("dateparser resolved temporal phrase '%s' -> %s", question, parsed)
        return parsed
    return None, None, None


def match_configured_date_alias(question: str, date_rules_config: Optional[YamlDict]) -> Optional[str]:
    candidates = []
    for order, item in enumerate((date_rules_config or {}).get("canonical_date_rules", []) or []):
        name = str(item.get("name") or "").strip()
        if not name or name in {"none", "custom_range", "rolling_days"}:
            continue
        for alias in item.get("aliases", []) or []:
            alias_text = str(alias or "").strip().lower()
            if alias_text and not re.search(r"\bn\b", alias_text):
                candidates.append((order, alias_text, name))

    text = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    for _, alias, name in sorted(candidates, key=lambda item: (-len(item[1]), item[0])):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
            return name
    return None


def normalise_month_typos(question: str) -> str:
    month_names = tuple(MONTH_MAP)

    def replacement(match: re.Match[str]) -> str:
        token = match.group(1)
        lower = token.lower()
        if lower in MONTH_MAP:
            return token
        candidates = [name for name in month_names if name[0] == lower[0]]
        best = difflib.get_close_matches(lower, candidates, n=1, cutoff=0.78) if candidates else []
        return best[0] if best else token

    return _MONTH_TYPO_RE.sub(replacement, question or "")


def _handle_explicit_range(text: str) -> Optional[DateResult]:
    match = _RANGE_RE.search(text)
    if match and parse_iso(match.group(1)) and parse_iso(match.group(2)):
        return custom_range(min(match.group(1), match.group(2)), max(match.group(1), match.group(2)))
    return None


def _handle_iso_dates(text: str) -> Optional[DateResult]:
    return range_from_iso_values([parse_iso(item) for item in _ISO_DATE_RE.findall(text)])


def _handle_eu_dates(text: str) -> Optional[DateResult]:
    return range_from_iso_values([eu_to_iso(dd, mm, yyyy) for dd, mm, yyyy in _EU_DATE_RE.findall(text)])


def _handle_named_month_date(text: str) -> Optional[DateResult]:
    match = _MONTH_DAY_YEAR_RE.search(text)
    if match:
        month, day, year = match.group(1), match.group(2), match.group(3)
    else:
        match = _DAY_MONTH_YEAR_RE.search(text)
        if not match:
            return None
        day, month, year = match.group(1), match.group(2), match.group(3)

    month_num = MONTH_MAP.get(month.lower())
    iso = f"{year}-{month_num:02d}-{int(day):02d}" if month_num else ""
    return custom_range(iso, iso) if parse_iso(iso) else None


def _handle_week_of_month(text: str) -> Optional[DateResult]:
    match = _WEEK_OF_MONTH_RE.search(text)
    rng = week_range_in_month(match.group(1), match.group(2), int(match.group(3))) if match else None
    return custom_range(rng[0], rng[1]) if rng else None


def _handle_month_year(text: str) -> Optional[DateResult]:
    match = _MONTH_YEAR_RE.search(text)
    rng = month_range(match.group(1), int(match.group(2))) if match else None
    return custom_range(rng[0], rng[1]) if rng else None


def _handle_week_of_year(text: str) -> Optional[DateResult]:
    match = _WEEK_OF_YEAR_RE.search(text)
    rng = week_range_in_year(match.group(1), int(match.group(2))) if match else None
    return custom_range(rng[0], rng[1]) if rng else None


def _handle_year(text: str) -> Optional[DateResult]:
    match = _YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        return custom_range(f"{year}-01-01", f"{year}-12-31")
    return None


def _handle_rolling_days(text: str) -> Optional[DateResult]:
    match = _ROLLING_DAYS_RE.search(text)
    return ("rolling_days", None, {"days": int(match.group(1))}) if match else None


def _resolve_dateparser_result(question: str) -> Optional[DateResult]:
    if _dateparser_parse is None:
        return None
    parsed = _dateparser_parse(question, settings={"STRICT_PARSING": True, "REQUIRE_PARTS": ["day", "month", "year"]})
    if parsed is None:
        return None
    value = parsed.strftime("%Y-%m-%d")
    return custom_range(value, value)


_DATE_HANDLERS = (
    _handle_explicit_range,
    _handle_iso_dates,
    _handle_eu_dates,
    _handle_named_month_date,
    _handle_week_of_month,
    _handle_month_year,
    _handle_week_of_year,
    _handle_year,
    _handle_rolling_days,
)
