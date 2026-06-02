from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

from .intent_slots import ComparisonSlots

YamlDict = Dict[str, Any]

_temporal_parser = import_module("..extraction.temporal_parser", __package__)
_extract_dates = _temporal_parser.extract_dates
_month_range = _temporal_parser._month_range
_MONTH_MAP = _temporal_parser._MONTH_MAP

_HASH_RE = re.compile(r"\b([0-9a-fA-F]{32,})\b")
_COMPARE_AND_RE = re.compile(
    r"(?is)\bcompare(?:s|d|ing)?\s+(.+?)\s+\band\b\s+(.+?)(?:[:?.]|$)"
)
_MONTH_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(month) for month in _MONTH_MAP) + r")\b"
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_CONNECTOR_RE = re.compile(r"(?i)\b(?:vs\.?|versus|compare(?:d)?\s+(?:to|with))\b")


def extract_comparison_slots(
    question: str,
    normalized_question: str,
    *,
    source_id: str,
    target_table: str,
    table_definition: YamlDict,
    date_rules_config: Optional[YamlDict] = None,
) -> Optional[ComparisonSlots]:
    text = " ".join(part for part in [question or "", normalized_question or ""] if part)

    entity = _extract_entity_comparison(text, table_definition)
    if entity is not None:
        return entity

    period = _extract_period_comparison(text, question, normalized_question, date_rules_config)
    if period is not None:
        return period

    return None


def _extract_entity_comparison(text: str, table_definition: YamlDict) -> Optional[ComparisonSlots]:
    hashes: List[str] = []
    for raw in _HASH_RE.findall(text or ""):
        value = raw.lower()
        if value not in hashes:
            hashes.append(value)
    if len(hashes) < 2:
        return None

    dimensions = set(table_definition.get("dimensions", []) or [])
    group_field = None
    for candidate in ("personFirstName", "person_name", "initiator_person", "personId"):
        if candidate in dimensions:
            group_field = candidate
            break
    if not group_field:
        return None

    return ComparisonSlots(
        comparison_type="entity",
        filters=[{"field": group_field, "op": "in", "value": hashes[:2]}],
        group_by=[group_field],
        values=hashes[:2],
        reason="two_hashed_entities",
    )


def _extract_period_comparison(
    text: str,
    question: str,
    normalized_question: str,
    date_rules_config: Optional[YamlDict],
) -> Optional[ComparisonSlots]:
    match = detect_period_comparison(
        question,
        normalized_question=normalized_question,
        date_rules_config=date_rules_config,
    )
    if match is not None:
        return ComparisonSlots(
            comparison_type="period",
            date_rule=match.compare_date_rule,
            date_range=match.compare_date_range,
            reason=match.reason,
        )

    front_loaded = _extract_front_loaded_period_comparison(text, date_rules_config)
    if front_loaded is not None:
        return front_loaded[1]
    return None


def extract_main_period_from_front_loaded_compare(
    question: str,
    normalized_question: str,
    date_rules_config: Optional[YamlDict],
) -> Tuple[Optional[str], Optional[YamlDict], Optional[YamlDict]]:
    text = " ".join(part for part in [question or "", normalized_question or ""] if part)
    front_loaded = _extract_front_loaded_period_comparison(text, date_rules_config)
    if front_loaded is None:
        return None, None, None
    return front_loaded[0]


def _extract_front_loaded_period_comparison(
    text: str,
    date_rules_config: Optional[YamlDict],
) -> Optional[Tuple[Tuple[Optional[str], Optional[YamlDict], Optional[YamlDict]], ComparisonSlots]]:
    match = _COMPARE_AND_RE.search(text or "")
    if not match:
        return None

    left_text = match.group(1).strip(" ,;:-")
    right_text = match.group(2).strip(" ,;:-")
    shared_year = _shared_year(f"{left_text} {right_text}")

    left_rule, left_range, left_params = _extract_side_period(left_text, date_rules_config, shared_year)
    right_rule, right_range, right_params = _extract_side_period(right_text, date_rules_config, shared_year)
    if not left_rule or not right_rule:
        return None
    if left_rule == right_rule and left_range == right_range and left_params == right_params:
        return None

    comparison = ComparisonSlots(
        comparison_type="period",
        date_rule=right_rule,
        date_range=right_range,
        date_params=right_params,
        reason="front_loaded_compare_and",
    )
    return (left_rule, left_range, left_params), comparison


def _extract_side_period(
    text: str,
    date_rules_config: Optional[YamlDict],
    shared_year: Optional[int],
) -> Tuple[Optional[str], Optional[YamlDict], Optional[YamlDict]]:
    month = _MONTH_RE.search(text or "")
    year = _first_year(text) or shared_year
    if month and year:
        rng = _month_range(month.group(1), year)
        if rng:
            return "custom_range", {"start_date": rng[0], "end_date": rng[1]}, None
    return _extract_dates(text, date_rules_config)


def _first_year(text: str) -> Optional[int]:
    match = _YEAR_RE.search(text or "")
    return int(match.group(1)) if match else None


def _shared_year(text: str) -> Optional[int]:
    years = [int(item) for item in _YEAR_RE.findall(text or "")]
    unique = set(years)
    return years[0] if len(unique) == 1 else None


# Connector-style period comparison ("X vs Y", "compare X to Y"), merged from the
# former period_comparison_detector module.

@dataclass(frozen=True)
class PeriodComparisonMatch:
    main_date_rule: Optional[str]
    main_date_range: Optional[Dict[str, str]]
    compare_date_rule: Optional[str]
    compare_date_range: Optional[Dict[str, str]]
    reason: str


def detect_period_comparison(
    question: str,
    normalized_question: Optional[str] = None,
    date_rules_config: Optional[YamlDict] = None,
) -> Optional[PeriodComparisonMatch]:
    for source_text in (question, normalized_question):
        if not source_text:
            continue
        match = _try_split(source_text, date_rules_config)
        if match is not None:
            return match
    return None


def _try_split(
    text: str,
    date_rules_config: Optional[YamlDict],
) -> Optional[PeriodComparisonMatch]:
    connector = _CONNECTOR_RE.search(text)
    if not connector:
        return None

    left = text[: connector.start()].strip(" ,;.-")
    right = text[connector.end() :].strip(" ,;.-")
    if not left or not right:
        return None

    shared_year = _shared_year(text)
    main_rule, main_range = _extract_connector_side(left, date_rules_config, shared_year)
    comp_rule, comp_range = _extract_connector_side(right, date_rules_config, shared_year)
    if not main_rule or not comp_rule:
        return None
    if main_rule == comp_rule and main_range == comp_range:
        # Same period on both sides — not actually a comparison.
        return None

    return PeriodComparisonMatch(
        main_date_rule=main_rule,
        main_date_range=main_range,
        compare_date_rule=comp_rule,
        compare_date_range=comp_range,
        reason=f"explicit_connector:{connector.group(0).lower()}",
    )


def _extract_connector_side(
    text: str,
    date_rules_config: Optional[YamlDict],
    shared_year: Optional[int],
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    if shared_year is not None:
        month = _MONTH_RE.search(text or "")
        if month:
            rng = _month_range(month.group(1), shared_year)
            if rng:
                return "custom_range", {"start_date": rng[0], "end_date": rng[1]}

    rule, date_range, _ = _extract_dates(text, date_rules_config)
    if rule:
        return rule, date_range
    return None, None
