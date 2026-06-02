from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from elasticsearch import Elasticsearch

YamlDict = Dict[str, Any]


def get_max_date(es: Elasticsearch, index: str, field: str = "call_date") -> Optional[date]:
    try:
        result = es.search(
            index=index,
            body={"size": 0, "aggs": {"max_date": {"max": {"field": field}}}},
        )
        max_epoch = result.get("aggregations", {}).get("max_date", {}).get("value")
        if max_epoch is not None:
            return datetime.fromtimestamp(max_epoch / 1000, tz=timezone.utc).date()
    except Exception:
        pass
    return None


def get_date_bounds(es: Elasticsearch, index: str, field: str = "call_date") -> tuple[Optional[date], Optional[date]]:
    try:
        result = es.search(
            index=index,
            body={
                "size": 0,
                "aggs": {
                    "min_date": {"min": {"field": field}},
                    "max_date": {"max": {"field": field}},
                },
            },
        )
        aggs = result.get("aggregations", {})
        min_epoch = aggs.get("min_date", {}).get("value")
        max_epoch = aggs.get("max_date", {}).get("value")
        min_date = datetime.fromtimestamp(min_epoch / 1000, tz=timezone.utc).date() if min_epoch is not None else None
        max_date = datetime.fromtimestamp(max_epoch / 1000, tz=timezone.utc).date() if max_epoch is not None else None
        return min_date, max_date
    except Exception:
        return None, None


def get_max_call_date(es: Elasticsearch, index: str) -> Optional[date]:
    return get_max_date(es, index, "call_date")


def relative_date_filter(
    rule: Any,
    column: str = "call_date",
    reference_date: Optional[date] = None,
    date_rules_config: Optional[YamlDict] = None,
) -> Tuple[Optional[Dict], str]:
    rule_name, rolling_days = _normalize_rule_input(rule)
    if rule_name is None:
        return None, ""
    rule_cfg = _find_rule_config(rule_name, date_rules_config)
    if rule_cfg is None:
        return None, ""
    if reference_date is not None:
        return _build_filter_from_reference(rule_cfg, rolling_days, column, reference_date)
    return _build_filter_using_es_math(rule_cfg, rolling_days, column)


def _find_rule_config(rule_name: str, date_rules_config: Optional[YamlDict]) -> Optional[YamlDict]:
    for rule in (date_rules_config or {}).get("canonical_date_rules") or []:
        if rule.get("name") == rule_name:
            return rule
    return None


def _normalize_rule_input(rule: Any) -> Tuple[Optional[str], Optional[int]]:
    if isinstance(rule, dict):
        if rule.get("mode") == "rolling_days":
            days = rule.get("days")
            if isinstance(days, int) and not isinstance(days, bool) and days > 0:
                return "rolling_days", days
        return None, None
    if not isinstance(rule, str):
        return None, None
    return rule, None


def _build_filter_from_reference(
    rule_cfg: YamlDict,
    rolling_days: Optional[int],
    column: str,
    ref: date,
) -> Tuple[Optional[Dict], str]:
    fmt = "%Y-%m-%d"
    mode = rule_cfg.get("mode")
    name = rule_cfg.get("name", "")

    def _range(gte: date, lte: date) -> Dict:
        return {"range": {column: {"gte": gte.strftime(fmt), "lte": lte.strftime(fmt)}}}

    if mode == "rolling_days":
        start = ref + timedelta(days=rule_cfg.get("start_day_offset", 0))
        end = ref + timedelta(days=rule_cfg.get("end_day_offset", 0))
        return _range(start, end), f"{column} {name} ({start} to {end})"

    if mode == "rolling_days_dynamic" and rolling_days is not None:
        start = ref - timedelta(days=rolling_days - 1)
        return _range(start, ref), f"{column} rolling_days ({start} to {ref}, days={rolling_days})"

    if mode == "calendar_week_current":
        start = ref - timedelta(days=ref.weekday())
        end = start + timedelta(days=6)
        return _range(start, end), f"{column} {name} ({start} to {end})"

    if mode == "calendar_week_previous":
        current_week_start = ref - timedelta(days=ref.weekday())
        end = current_week_start - timedelta(days=1)
        start = current_week_start - timedelta(days=7)
        return _range(start, end), f"{column} {name} ({start} to {end})"

    if mode == "calendar_month_current":
        start = ref.replace(day=1)
        return _range(start, ref), f"{column} {name} ({start} to {ref})"

    if mode == "calendar_month_previous":
        first_this = ref.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return _range(first_prev, last_prev), f"{column} {name} ({first_prev} to {last_prev})"

    return None, ""


def _build_filter_using_es_math(
    rule_cfg: YamlDict,
    rolling_days: Optional[int],
    column: str,
) -> Tuple[Optional[Dict], str]:
    mode = rule_cfg.get("mode")
    name = rule_cfg.get("name", "")

    if mode == "rolling_days_dynamic" and rolling_days is not None:
        return (
            {"range": {column: {"gte": f"now-{rolling_days - 1}d/d", "lte": "now/d"}}},
            f"{column} rolling_days ({rolling_days})",
        )

    es_gte = rule_cfg.get("es_gte")
    es_lte = rule_cfg.get("es_lte")
    es_lt = rule_cfg.get("es_lt")

    if es_gte:
        clause: Dict[str, Any] = {"gte": es_gte}
        if es_lte:
            clause["lte"] = es_lte
        elif es_lt:
            clause["lt"] = es_lt
        return {"range": {column: clause}}, f"{column} {name}"

    return None, ""
