from __future__ import annotations

from typing import Any, Dict, Optional

from .deterministic_temporal_fallback import (
    extract_dates_deterministic,
    match_configured_date_alias,
)
from .temporal_constants import MONTH_MAP as _MONTH_MAP
from .temporal_value_resolver import DateResult, month_range as _month_range, validate_temporal_slots
from .time_range_parser import detect_day_part, extract_time_range

YamlDict = Dict[str, Any]


def extract_dates(
    question: str,
    date_rules_config: Optional[YamlDict] = None,
) -> DateResult:
    return extract_dates_deterministic(question, date_rules_config)


def resolve_temporal_slots(
    *,
    date_rule: Any,
    date_range: Any,
    date_params: Any,
    date_rules_config: Optional[YamlDict] = None,
) -> DateResult:
    return validate_temporal_slots(
        date_rule=date_rule,
        date_range=date_range,
        date_params=date_params,
        date_rules_config=date_rules_config,
    )


__all__ = [
    "detect_day_part",
    "extract_dates",
    "extract_time_range",
    "match_configured_date_alias",
    "resolve_temporal_slots",
]
