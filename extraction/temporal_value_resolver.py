from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from .temporal_constants import MONTH_MAP, ORDINAL_TO_WEEK

YamlDict = Dict[str, Any]
DateResult = Tuple[Optional[str], Optional[Dict[str, str]], Optional[Dict[str, int]]]


def parse_iso(text: str) -> Optional[str]:
    try:
        year_s, month_s, day_s = str(text or "").split("-")
        return datetime(int(year_s), int(month_s), int(day_s)).strftime("%Y-%m-%d")
    except Exception:
        return None


def eu_to_iso(dd: str, mm: str, yyyy: str) -> Optional[str]:
    try:
        return datetime(int(yyyy), int(mm), int(dd)).strftime("%Y-%m-%d")
    except Exception:
        return None


def month_range(month_name: str, year: int) -> Optional[Tuple[str, str]]:
    month_num = MONTH_MAP.get(str(month_name or "").lower())
    if not month_num:
        return None
    first = datetime(year, month_num, 1)
    last = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def week_range_in_year(week_value: str, year: int) -> Optional[Tuple[str, str]]:
    ordinal = ORDINAL_TO_WEEK.get(str(week_value or "").lower())
    if ordinal == -1:
        week_num = datetime(year, 12, 28).isocalendar().week
    elif ordinal is not None:
        week_num = ordinal
    else:
        try:
            week_num = int(week_value)
        except Exception:
            return None

    if week_num <= 0 or week_num > 53:
        return None
    try:
        start = datetime.fromisocalendar(year, week_num, 1)
        end = datetime.fromisocalendar(year, week_num, 7)
    except ValueError:
        return None
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def week_range_in_month(week_value: str, month_name: str, year: int) -> Optional[Tuple[str, str]]:
    month_num = MONTH_MAP.get(str(month_name or "").lower())
    if not month_num:
        return None

    ordinal = ORDINAL_TO_WEEK.get(str(week_value or "").lower())
    if ordinal is None:
        try:
            ordinal = int(week_value)
        except Exception:
            return None

    first_day = datetime(year, month_num, 1)
    last_day = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
    first_full_week_start = first_day + timedelta(days=(7 - first_day.weekday()) % 7)

    if ordinal == -1:
        last_full_week_end = last_day - timedelta(days=(last_day.weekday() - 6) % 7)
        start = last_full_week_end - timedelta(days=6)
        if start.month != month_num or last_full_week_end.month != month_num:
            return None
        return start.strftime("%Y-%m-%d"), last_full_week_end.strftime("%Y-%m-%d")

    if ordinal <= 0:
        return None
    start = first_full_week_start + timedelta(days=(ordinal - 1) * 7)
    end = start + timedelta(days=6)
    if start.month != month_num or end.month != month_num:
        return None
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def custom_range(start: str, end: str) -> DateResult:
    return "custom_range", {"start_date": start, "end_date": end}, None


def range_from_iso_values(iso_values: list) -> Optional[DateResult]:
    valid = [item for item in iso_values if item]
    if len(valid) >= 2:
        return custom_range(min(valid), max(valid))
    if len(valid) == 1:
        return custom_range(valid[0], valid[0])
    return None


def validate_temporal_slots(
    *,
    date_rule: Any,
    date_range: Any,
    date_params: Any,
    date_rules_config: Optional[YamlDict] = None,
) -> DateResult:
    rule = str(date_rule or "none").strip() or "none"
    known_rules = _known_date_rules(date_rules_config)
    if rule not in known_rules and rule not in {"none", "custom_range", "rolling_days"}:
        rule = "none"

    clean_range = _clean_date_range(date_range)
    if clean_range:
        return "custom_range", clean_range, None

    clean_params = _clean_date_params(date_params)
    if rule == "rolling_days" and clean_params:
        return rule, None, clean_params
    if rule in {"custom_range", "rolling_days"}:
        return "none", None, None
    return rule, None, None


def _clean_date_range(raw: Any) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        return None
    start = parse_iso(str(raw.get("start_date") or ""))
    end = parse_iso(str(raw.get("end_date") or ""))
    if not start or not end:
        return None
    return {"start_date": min(start, end), "end_date": max(start, end)}


def _clean_date_params(raw: Any) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None
    try:
        days = int(raw.get("days"))
    except Exception:
        return None
    if days <= 0 or days > 3660:
        return None
    return {"days": days}


def _known_date_rules(date_rules_config: Optional[YamlDict]) -> set[str]:
    return {
        str(item.get("name") or "").strip()
        for item in (date_rules_config or {}).get("canonical_date_rules", []) or []
        if item.get("name")
    }
