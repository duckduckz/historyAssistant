from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, List, Optional, Set

from historyAssistant.config import load_config, resolve_config_path

REQUEST_TYPE_CHART = import_module("..core.canonical_schema", __package__).REQUEST_TYPE_CHART


YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class ChartRuleSet:
    config_path: Optional[str]
    raw_config: YamlDict
    default_chart_type: str
    allowed_chart_types: Set[str]
    chart_types: Dict[str, YamlDict]
    dimensions: Dict[str, YamlDict]
    numeric_fields: Dict[str, YamlDict]
    metrics: Dict[str, YamlDict]
    disallowed_dimensions: Set[str]
    selection_rules: YamlDict = field(default_factory=dict)
    validation_hints: YamlDict = field(default_factory=dict)
    titles: YamlDict = field(default_factory=dict)
    formatting: YamlDict = field(default_factory=dict)


@dataclass(frozen=True)
class ChartValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repaired_chart_spec: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> YamlDict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "repaired_chart_spec": dict(self.repaired_chart_spec),
        }


__all__ = [
    "ChartRuleSet",
    "ChartValidationResult",
    "build_chart_rule_set",
    "is_allowed_chart_type",
    "is_chart_dimension",
    "is_chart_numeric_field",
    "metric_allowed_chart_types",
    "metric_allowed_series",
    "metric_recommended_dimensions",
    "default_chart_type_for_metric",
    "build_default_chart_spec",
    "validate_chart_spec",
    "should_plan_auto_chart",
]


def should_plan_auto_chart(
    plan: YamlDict,
    bundle: Any,
    *,
    chart_suppressed: bool = False,
) -> tuple[bool, str]:

    if chart_suppressed:
        return False, "user_suppressed"

    aggregation = str(plan.get("aggregation") or "").strip().lower()
    if aggregation == "raw_records":
        return False, "raw_records"

    metric = str(plan.get("metric") or "")
    metric_cfg: YamlDict = getattr(bundle, "metrics", {}).get(metric, {})
    if not metric_cfg.get("chartable", True):
        return False, "detail_lookup_metric"

    rule_set = build_chart_rule_set(getattr(bundle, "chart_catalog", None))
    if metric:
        allowed = metric_allowed_chart_types(metric, rule_set)

        if allowed is not None and len(allowed) == 0 and metric in rule_set.metrics:
            return False, "metric_not_chartable"

    if plan.get("compare_to"):
        return True, "comparison_overlay"

    if plan.get("bucket_interval"):
        return True, "time_bucket_trend"

    group_by = [g for g in (plan.get("group_by") or []) if isinstance(g, str)]
    if group_by:
        return True, "grouped_result"

    return False, "scalar_result"


def build_chart_rule_set(chart_catalog: Optional[YamlDict] = None) -> ChartRuleSet:
    raw_config = chart_catalog or load_config("chart_catalog")
    policy = raw_config.get("policy", {})

    config_path = None
    if chart_catalog is None:
        try:
            config_path = str(resolve_config_path("chart_catalog"))
        except FileNotFoundError:
            config_path = None

    allowed_chart_types = set(policy.get("allowed_chart_types", []))
    default_chart_type = str(policy.get("default_chart_type", "line")).strip().lower()
    if default_chart_type not in allowed_chart_types and allowed_chart_types:
        default_chart_type = sorted(allowed_chart_types)[0]

    return ChartRuleSet(
        config_path=config_path,
        raw_config=raw_config,
        default_chart_type=default_chart_type or "line",
        allowed_chart_types=allowed_chart_types,
        chart_types=dict(raw_config.get("chart_types", {})),
        dimensions=dict(raw_config.get("dimensions", {})),
        numeric_fields=dict(raw_config.get("numeric_fields", {})),
        metrics=dict(raw_config.get("metrics", {})),
        disallowed_dimensions=set(raw_config.get("disallowed_dimensions", [])),
        selection_rules=dict(raw_config.get("selection_rules", {})),
        validation_hints=dict(raw_config.get("validation_hints", {})),
        titles=dict(raw_config.get("titles", {})),
        formatting=dict(raw_config.get("formatting", {})),
    )


def is_allowed_chart_type(chart_type: Optional[str], rule_set: ChartRuleSet) -> bool:
    return isinstance(chart_type, str) and chart_type in rule_set.allowed_chart_types


def is_chart_dimension(field_name: Optional[str], rule_set: ChartRuleSet) -> bool:
    return isinstance(field_name, str) and field_name in rule_set.dimensions


def is_chart_numeric_field(field_name: Optional[str], rule_set: ChartRuleSet) -> bool:
    return isinstance(field_name, str) and field_name in rule_set.numeric_fields


def metric_allowed_chart_types(metric_name: Optional[str], rule_set: ChartRuleSet) -> Set[str]:
    if not isinstance(metric_name, str):
        return set()
    metric_rule = rule_set.metrics.get(metric_name, {})
    return set(metric_rule.get("allowed_chart_types", []))


def metric_allowed_series(metric_name: Optional[str], rule_set: ChartRuleSet) -> Set[str]:
    if not isinstance(metric_name, str):
        return set()
    metric_rule = rule_set.metrics.get(metric_name, {})
    return set(metric_rule.get("allowed_series", []))


def metric_recommended_dimensions(metric_name: Optional[str], rule_set: ChartRuleSet) -> List[str]:
    if not isinstance(metric_name, str):
        return []
    metric_rule = rule_set.metrics.get(metric_name, {})
    return list(metric_rule.get("recommended_dimensions", []))


def default_chart_type_for_metric(metric_name: Optional[str], rule_set: ChartRuleSet) -> str:
    allowed_types = metric_allowed_chart_types(metric_name, rule_set)
    if allowed_types:
        recommended_dimensions = metric_recommended_dimensions(metric_name, rule_set)
        dimension_roles = [
            str(rule_set.dimensions.get(field_name, {}).get("role", "")).strip().lower()
            for field_name in recommended_dimensions
        ]

        selection_preferences = (
            ("prefer_line_when_x_role", "line"),
            ("prefer_bar_when_x_role", "bar"),
        )
        for rule_name, chart_type in selection_preferences:
            if chart_type not in allowed_types:
                continue
            preferred_roles = {
                str(role).strip().lower()
                for role in rule_set.selection_rules.get(rule_name, [])
                if isinstance(role, str)
            }
            if preferred_roles and any(role in preferred_roles for role in dimension_roles):
                return chart_type

        if rule_set.selection_rules.get("prefer_horizontal_bar_for_ranking") and "horizontal_bar" in allowed_types:
            if any(role in {"categorical", "ordered_category"} for role in dimension_roles):
                return "horizontal_bar"

    if rule_set.default_chart_type in allowed_types:
        return rule_set.default_chart_type
    if allowed_types:
        return sorted(allowed_types)[0]
    return rule_set.default_chart_type


def dimension_allows_chart_type(field_name: str, chart_type: str, rule_set: ChartRuleSet) -> bool:
    dim_rule = rule_set.dimensions.get(field_name, {})
    return chart_type in set(dim_rule.get("allowed_chart_types", []))


def dimension_allows_axis_slot(field_name: str, axis_slot: str, rule_set: ChartRuleSet) -> bool:
    dim_rule = rule_set.dimensions.get(field_name, {})
    return axis_slot in set(dim_rule.get("allowed_axis_slots", []))


def auto_title(chart_type: str, metric_name: Optional[str], x_field: Optional[str], series_field: Optional[str], rule_set: ChartRuleSet) -> Optional[str]:
    templates = rule_set.titles.get("auto_title_templates", {})
    template = templates.get(chart_type)
    if not template:
        return None

    metric_label = rule_set.metrics.get(metric_name or "", {}).get("label", metric_name or "Metric")
    x_label = rule_set.dimensions.get(x_field or "", {}).get("label", x_field or "dimension")
    series_label = rule_set.dimensions.get(series_field or "", {}).get("label", series_field or "series")
    return (
        template.replace("{metric_label}", str(metric_label))
        .replace("{x_label}", str(x_label))
        .replace("{series_label}", str(series_label))
    )


def chart_type_for_dimension(x_field: Optional[str], metric_name: Optional[str], rule_set: ChartRuleSet) -> str:

    if x_field and x_field in rule_set.dimensions:
        role = str(rule_set.dimensions[x_field].get("role", "")).strip().lower()
        allowed_metric = metric_allowed_chart_types(metric_name, rule_set) or rule_set.allowed_chart_types
        if role == "time" and "line" in allowed_metric:
            return "line"
        if role in {"categorical", "ordered_category"} and "bar" in allowed_metric:
            return "bar"
    return default_chart_type_for_metric(metric_name, rule_set)


def build_default_chart_spec(*, metric_name: Optional[str], group_by: Optional[List[str]], chart_type: Optional[str], rule_set: Optional[ChartRuleSet] = None) -> Dict[str, Any]:
    rules = rule_set or build_chart_rule_set()
    resolved_group_by = list(group_by or [])
    x_field_candidate = resolved_group_by[0] if resolved_group_by else None
    resolved_chart_type = chart_type or chart_type_for_dimension(x_field_candidate, metric_name, rules)
    x_field = x_field_candidate
    series_field = resolved_group_by[1] if len(resolved_group_by) > 1 else None

    spec = {
        "chart_type": resolved_chart_type,
        "x": x_field,
        "y": metric_name,
        "series": series_field,
        "title": auto_title(resolved_chart_type, metric_name, x_field, series_field, rules),
    }

    if resolved_chart_type == "horizontal_bar":
        spec["x"] = metric_name
        spec["y"] = x_field
    elif resolved_chart_type == "scatter":
        spec["x"] = x_field
        spec["y"] = metric_name
    elif resolved_chart_type == "pie":
        spec["series"] = None

    return spec


def validate_chart_spec(*, metric_name: Optional[str], group_by: Optional[List[str]], chart_spec: Optional[Dict[str, Any]], rule_set: Optional[ChartRuleSet] = None) -> ChartValidationResult:
    rules = rule_set or build_chart_rule_set()
    resolved_group_by = list(group_by or [])
    repaired = dict(chart_spec or {})
    errors: List[str] = []
    warnings: List[str] = []

    def add_error(code: str, message: str) -> None:
        errors.append(f"{code}: {message}")

    def add_warning(code: str, message: str) -> None:
        warnings.append(f"{code}: {message}")

    chart_type = repaired.get("chart_type")
    if not is_allowed_chart_type(chart_type, rules):
        add_error("E_CHART_TYPE_DISALLOWED", f"Disallowed chart type: {chart_type}")
        chart_type = default_chart_type_for_metric(metric_name, rules)
        repaired["chart_type"] = chart_type

    allowed_metric_types = metric_allowed_chart_types(metric_name, rules)
    if metric_name and allowed_metric_types and chart_type not in allowed_metric_types:
        add_error("E_METRIC_CHART_TYPE_UNSUPPORTED", f"Metric '{metric_name}' does not support chart type '{chart_type}'")
        fallback_type = default_chart_type_for_metric(metric_name, rules)
        repaired["chart_type"] = fallback_type
        chart_type = fallback_type

    if "x" not in repaired:
        if chart_type == "horizontal_bar":
            repaired["x"] = metric_name
        else:
            repaired["x"] = resolved_group_by[0] if resolved_group_by else None
    if "y" not in repaired:
        if chart_type == "horizontal_bar":
            repaired["y"] = resolved_group_by[0] if resolved_group_by else None
        else:
            repaired["y"] = metric_name
    if "series" not in repaired:
        repaired["series"] = resolved_group_by[1] if len(resolved_group_by) > 1 else None

    x_field = repaired.get("x")
    y_field = repaired.get("y")
    series_field = repaired.get("series")

    if x_field is not None and not isinstance(x_field, str):
        add_error("E_X_INVALID_TYPE", "chart_spec.x must be a string or null")
        repaired["x"] = None
        x_field = None
    if y_field is not None and not isinstance(y_field, str):
        add_error("E_Y_INVALID_TYPE", "chart_spec.y must be a string or null")
        repaired["y"] = None
        y_field = None
    if series_field is not None and not isinstance(series_field, str):
        add_error("E_SERIES_INVALID_TYPE", "chart_spec.series must be a string or null")
        repaired["series"] = None
        series_field = None

    if isinstance(x_field, str) and x_field in rules.disallowed_dimensions:
        add_error("E_X_FIELD_DISALLOWED", f"Disallowed chart field in x: {x_field}")
        repaired["x"] = None
        x_field = None
    if isinstance(series_field, str) and series_field in rules.disallowed_dimensions:
        add_error("E_SERIES_FIELD_DISALLOWED", f"Disallowed chart field in series: {series_field}")
        repaired["series"] = None
        series_field = None

    x_is_metric_axis = chart_type == "horizontal_bar" and x_field == metric_name
    if isinstance(x_field, str) and not x_is_metric_axis and not is_chart_dimension(x_field, rules) and not is_chart_numeric_field(x_field, rules):
        add_error("E_X_FIELD_UNKNOWN", f"Unknown chart field in x: {x_field}")
        repaired["x"] = None
        x_field = None
    if isinstance(series_field, str) and not is_chart_dimension(series_field, rules):
        add_error("E_SERIES_FIELD_UNKNOWN", f"Unknown chart field in series: {series_field}")
        repaired["series"] = None
        series_field = None

    if isinstance(x_field, str) and is_chart_dimension(x_field, rules):
        if not dimension_allows_chart_type(x_field, chart_type, rules):
            add_error("E_X_DIMENSION_CHART_TYPE_INVALID", f"Dimension '{x_field}' is not allowed for chart type '{chart_type}'")
            repaired["x"] = None
            x_field = None
        elif chart_type != "horizontal_bar" and not dimension_allows_axis_slot(x_field, "x", rules):
            add_error("E_X_DIMENSION_AXIS_INVALID", f"Dimension '{x_field}' is not allowed in chart x")
            repaired["x"] = None
            x_field = None

    if isinstance(series_field, str) and is_chart_dimension(series_field, rules):
        if not dimension_allows_chart_type(series_field, chart_type, rules):
            add_error("E_SERIES_DIMENSION_CHART_TYPE_INVALID", f"Dimension '{series_field}' is not allowed for chart type '{chart_type}'")
            repaired["series"] = None
            series_field = None
        elif not dimension_allows_axis_slot(series_field, "series", rules):
            add_error("E_SERIES_DIMENSION_AXIS_INVALID", f"Dimension '{series_field}' is not allowed in chart series")
            repaired["series"] = None
            series_field = None

    allowed_series = metric_allowed_series(metric_name, rules)
    if isinstance(series_field, str) and allowed_series and series_field not in allowed_series:
        add_warning("W_SERIES_NOT_RECOMMENDED", f"Series '{series_field}' is not recommended for metric '{metric_name}'")

    if chart_type == "scatter":
        if x_field is not None and not is_chart_dimension(x_field, rules) and not is_chart_numeric_field(x_field, rules):
            add_error("E_SCATTER_X_UNKNOWN", "Scatter chart x must be a known dimension or numeric field")
            repaired["x"] = None
        if y_field not in {metric_name, None} and not is_chart_numeric_field(y_field, rules):
            add_error("E_SCATTER_Y_NOT_NUMERIC", "Scatter chart y must be a numeric field or the selected metric")
            repaired["y"] = metric_name

    if chart_type == "pie":
        if len(resolved_group_by) > 1:
            add_error("E_PIE_TOO_MANY_GROUPS", "Pie charts support only one grouping dimension")
        if series_field:
            repaired["series"] = None
            add_warning("W_PIE_SERIES_REMOVED", "Removed pie chart series")

    if isinstance(x_field, str) and metric_name:
        recommended_dimensions = set(metric_recommended_dimensions(metric_name, rules))
        if recommended_dimensions and x_field not in recommended_dimensions and not is_chart_numeric_field(x_field, rules):
            add_warning("W_X_DIMENSION_NOT_RECOMMENDED", f"Dimension '{x_field}' is not in recommended_dimensions for metric '{metric_name}'")

    if not repaired.get("title"):
        repaired["title"] = auto_title(chart_type, metric_name, repaired.get("x"), repaired.get("series"), rules)

    return ChartValidationResult(valid=not errors, errors=errors, warnings=warnings, repaired_chart_spec=repaired)
