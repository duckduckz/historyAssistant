from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

YamlDict = Dict[str, Any]


def _date_rule_label(date_rule: Optional[str], date_rules_config: Optional[YamlDict]) -> Optional[str]:
    if not date_rule or date_rule in {"none", "custom_range"}:
        return None
    rules = (date_rules_config or {}).get("canonical_date_rules") or []
    for rule in rules:
        if rule.get("name") == date_rule:
            label = rule.get("display_label")
            return str(label) if label else None
    return None


def datelabel_from_comparison_side(
    side: YamlDict,
    date_rules_config: Optional[YamlDict] = None,
) -> Optional[str]:
    date_rule = side.get("date_rule")
    date_range = side.get("date_range") or {}

    label = _date_rule_label(date_rule, date_rules_config)
    if label:
        return label

    if date_rule == "custom_range":
        start = date_range.get("start_date")
        end = date_range.get("end_date")
        if start and end:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d")
                if (end_dt - start_dt).days == 6:
                    return f"week of {start_dt.strftime('%d/%m/%Y')}"
            except ValueError:
                pass
            if start == end:
                return start
            return f"{start} to {end}"
    return None


def format_filter_value_for_label(
    field_name: str,
    value: Any,
    display_value_maps: Optional[YamlDict] = None,
) -> Optional[str]:
    if value in {None, ""}:
        return None
    field_map = (display_value_maps or {}).get(field_name) or {}
    if field_map:
        key = str(value).strip().lower()
        mapped = field_map.get(key) or field_map.get(str(value))
        if mapped:
            return str(mapped)
    return str(value)


def equality_filter_map(side: YamlDict) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for filt in side.get("filters") or []:
        if filt.get("op") == "=" and filt.get("value") not in {None, ""}:
            mapping[str(filt.get("field"))] = filt.get("value")
    return mapping


def label_from_comparison_side(
    side: YamlDict,
    other_side: Optional[YamlDict] = None,
    *,
    date_rules_config: Optional[YamlDict] = None,
    display_value_maps: Optional[YamlDict] = None,
) -> Optional[str]:
    own_filters = equality_filter_map(side)
    other_filters = equality_filter_map(other_side or {})
    differing_fields = [
        field_name
        for field_name, value in own_filters.items()
        if other_filters.get(field_name) != value
    ]

    filter_labels: List[str] = []
    for field_name in differing_fields:
        label = format_filter_value_for_label(
            field_name, own_filters.get(field_name), display_value_maps
        )
        if label:
            filter_labels.append(label)

    own_date = datelabel_from_comparison_side(side, date_rules_config)
    other_date = datelabel_from_comparison_side(other_side or {}, date_rules_config)
    date_differs = own_date and own_date != other_date

    parts: List[str] = []
    parts.extend(filter_labels)
    if date_differs:
        parts.append(own_date)
    if parts:
        return " ".join(parts) if len(parts) == 2 and date_differs else " / ".join(parts)
    if own_date:
        return own_date
    if filter_labels:
        return " / ".join(filter_labels)
    return None


def comparison_intent_direction(
    question: str,
    metric_name: Optional[str],
    canonical_plan: Optional[YamlDict] = None,
) -> str:
    # LLM already determined direction during slot extraction — use it if present
    if canonical_plan and canonical_plan.get("comparison_preferred_direction") in {"asc", "desc"}:
        return canonical_plan["comparison_preferred_direction"]

    # signal words as fast fallback when LLM path was not used
    text = (question or "").lower()
    desc_signals = {"more", "higher", "longer", "slowest", "slower", "highest", "most", "worse", "worst"}
    asc_signals = {"less", "lower", "fewer", "shorter", "faster", "lowest", "least"}
    if any(word in text for word in desc_signals):
        return "desc"
    if any(word in text for word in asc_signals):
        return "asc"
    return "desc"


def build_comparison_summary(
    canonical_plan: YamlDict,
    compare_to: YamlDict,
    comparison_result: YamlDict,
    *,
    date_rules_config: Optional[YamlDict] = None,
    display_value_maps: Optional[YamlDict] = None,
) -> Optional[YamlDict]:
    change = comparison_result.get("change") or {}
    if not change.get("has_data"):
        return None

    question = canonical_plan.get("normalized_sub_request") or ""
    metric_name = canonical_plan.get("metric")
    direction = comparison_intent_direction(question, metric_name, canonical_plan)
    main_label = label_from_comparison_side(
        canonical_plan, compare_to,
        date_rules_config=date_rules_config,
        display_value_maps=display_value_maps,
    ) or "main"
    comp_label = label_from_comparison_side(
        compare_to, canonical_plan,
        date_rules_config=date_rules_config,
        display_value_maps=display_value_maps,
    ) or "comparison"
    main_value = change.get("main_value")
    comp_value = change.get("comparison_value")

    if main_value is None or comp_value is None:
        grouped_changes = change.get("grouped_changes") or []
        if grouped_changes:
            if direction == "asc":
                main_wins = all(float(row.get("absolute_change", 0)) <= 0 for row in grouped_changes)
                comp_wins = all(float(row.get("absolute_change", 0)) >= 0 for row in grouped_changes)
            else:
                main_wins = all(float(row.get("absolute_change", 0)) >= 0 for row in grouped_changes)
                comp_wins = all(float(row.get("absolute_change", 0)) <= 0 for row in grouped_changes)

            if main_wins and not comp_wins:
                winner_label = main_label
            elif comp_wins and not main_wins:
                winner_label = comp_label
            else:
                return None

            return {
                "main_label": main_label,
                "comparison_label": comp_label,
                "winner_label": winner_label,
                "preferred_direction": direction,
                "grouped_changes": grouped_changes,
            }
        return None

    if direction == "asc":
        winner_label = main_label if float(main_value) <= float(comp_value) else comp_label
    else:
        winner_label = main_label if float(main_value) >= float(comp_value) else comp_label

    return {
        "main_label": main_label,
        "comparison_label": comp_label,
        "winner_label": winner_label,
        "preferred_direction": direction,
        "main_value": main_value,
        "comparison_value": comp_value,
    }


__all__ = [
    "datelabel_from_comparison_side",
    "format_filter_value_for_label",
    "equality_filter_map",
    "label_from_comparison_side",
    "comparison_intent_direction",
    "build_comparison_summary",
]
