from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional, Set

from historyAssistant.config import (
    ConfigBundle,
    get_runtime_defaults,
    load_all_configs,
)

_canonical_schema = import_module(".canonical_schema", __package__)
REQUEST_TYPE_CHART = _canonical_schema.REQUEST_TYPE_CHART
REQUEST_TYPE_QA = _canonical_schema.REQUEST_TYPE_QA

_chart_rules = import_module("..chart.chart_rules", __package__)
ChartRuleSet = _chart_rules.ChartRuleSet
build_chart_rule_set = _chart_rules.build_chart_rule_set


YamlDict = Dict[str, Any]

ALLOWED_FILTER_OPS = {"=", "!=", ">", ">=", "<", "<=", "in", "between", "relative", "time_between", "wildcard"}
ALLOWED_REQUEST_TYPES = {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}


@dataclass(frozen=True)
class ValidationPolicy:
    config_bundle: ConfigBundle
    chart_rules: ChartRuleSet
    active_table: str
    allowed_tables: Set[str]
    allowed_metrics: Set[str]
    allowed_columns: Set[str]
    groupable_dimensions: Set[str]
    allowed_filter_ops: Set[str]
    allowed_request_types: Set[str]
    runtime_defaults: YamlDict
    table_definition: YamlDict


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    blocking_errors: List[str] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    ambiguity_reason: str = ""
    canonical_plan: Dict[str, Any] = field(default_factory=dict)
    chart_validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> YamlDict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "repairs": list(self.repairs),
            "blocking_errors": list(self.blocking_errors),
            "missing_slots": list(self.missing_slots),
            "ambiguity_reason": self.ambiguity_reason,
            "canonical_plan": deepcopy(self.canonical_plan),
            "chart_validation": deepcopy(self.chart_validation),
        }


__all__ = [
    "ALLOWED_FILTER_OPS",
    "ALLOWED_REQUEST_TYPES",
    "ValidationPolicy",
    "ValidationResult",
    "build_validation_policy",
    "infer_plan_policies",
    "validate_and_repair_plan",
]


def build_validation_policy(config_bundle: Optional[ConfigBundle] = None) -> ValidationPolicy:
    bundle = config_bundle or load_all_configs()
    app_config = bundle.app_config
    table_registry = bundle.table_registry
    metric_catalog = bundle.metric_catalog

    source_tables = table_registry.get("tables", {})
    active_table = next(iter(source_tables.keys()), None)

    if not active_table:
        active_table = (
            app_config.get("tables", {}).get("active_table")
            or bundle.chart_catalog.get("policy", {}).get("active_table")
            or "historyCalls"
        )

    table_definition = table_registry.get("tables", {}).get(active_table, {})

    allowed_columns = (
        set(table_definition.get("dimensions", []))
        | set(table_definition.get("measure_fields", []))
        | set(table_definition.get("time_field", []))
        | set(table_definition.get("timestamp_fields", []))
        | set(table_definition.get("identifiers", []))
        | set(table_definition.get("event_fields", []))
        | set(table_definition.get("dq_flags", []))
        | set(table_definition.get("scope_flags", []))
    )

    return ValidationPolicy(
        config_bundle=bundle,
        chart_rules=build_chart_rule_set(bundle.chart_catalog),
        active_table=active_table,
        allowed_tables={active_table},
        allowed_metrics=set(metric_catalog.get("metrics", {}).keys()),
        allowed_columns=allowed_columns,
        groupable_dimensions=set(table_definition.get("dimensions", [])),
        allowed_filter_ops=set(ALLOWED_FILTER_OPS),
        allowed_request_types=set(ALLOWED_REQUEST_TYPES),
        runtime_defaults=get_runtime_defaults(app_config),
        table_definition=table_definition,
    )


def infer_plan_policies(metric_name: Optional[str], policy: ValidationPolicy) -> tuple[Optional[str], Optional[str]]:
    metric_cfg = policy.config_bundle.metric_catalog.get("metrics", {}).get(metric_name or "", {})
    default_scope = metric_cfg.get("default_scope", policy.runtime_defaults.get("scope_policy"))
    default_outlier = metric_cfg.get("default_outlier_policy", policy.runtime_defaults.get("outlier_policy"))

    if metric_name in {"total_calls", "priority_value"}:
        default_scope = "raw"
        default_outlier = "none"

    if default_scope == "qa_ready" and not policy.table_definition.get("scope_flags"):
        default_scope = "raw"

    return default_scope, default_outlier


def validate_and_repair_plan(
    plan: Dict[str, Any],
    *,
    policy: Optional[ValidationPolicy] = None,
    return_dataclass: bool = False,
    config_bundle: Optional[ConfigBundle] = None,
    scenario: Optional[str] = None,
) -> ValidationResult | Dict[str, Any]:

    _ = scenario 
    runtime_policy = policy or build_validation_policy(config_bundle or load_all_configs())
    repaired = deepcopy(plan)
    repairs: List[str] = []

    metric_name = repaired.get("metric")
    errors: List[str] = []
    blocking_errors: List[str] = []
    missing_slots: List[str] = []
    if metric_name:
        metric_info = runtime_policy.config_bundle.metric_catalog.get("metrics", {}).get(metric_name, {})
        if not metric_info:
            errors.append(f"unknown_metric:{metric_name}")
            blocking_errors.append(f"unknown_metric:{metric_name}")
        expected_aggregation = metric_info.get("aggregation")
        if expected_aggregation and repaired.get("aggregation") is None:
            repaired["aggregation"] = expected_aggregation
            repairs.append("Filled missing aggregation from metric catalog")

        expected_scope, expected_outlier = infer_plan_policies(metric_name, runtime_policy)
        if repaired.get("scope_policy") is None and expected_scope is not None:
            repaired["scope_policy"] = expected_scope
            repairs.append("Filled missing scope_policy from metric catalog")
        if repaired.get("outlier_policy") is None and expected_outlier is not None:
            repaired["outlier_policy"] = expected_outlier
            repairs.append("Filled missing outlier_policy from metric catalog")
        if metric_name == "numeric_summary":
            numeric_errors, numeric_repairs = _validate_numeric_summary(repaired, runtime_policy)
            errors.extend(numeric_errors)
            blocking_errors.extend(numeric_errors)
            repairs.extend(numeric_repairs)
    else:
        missing_slots.append("metric")
        errors.append("missing_metric")
        blocking_errors.append("missing_metric")

    request_type = repaired.get("request_type")
    valid = request_type in runtime_policy.allowed_request_types and not blocking_errors
    if request_type not in runtime_policy.allowed_request_types:
        errors.append(f"invalid_request_type:{request_type}")
        blocking_errors.append(f"invalid_request_type:{request_type}")

    if repaired.get("aggregation") == "raw_records":
        temporal_repairs = _ensure_raw_record_temporal_source_fields(repaired, runtime_policy)
        repairs.extend(temporal_repairs)

    filter_errors, filter_repairs = _validate_and_repair_filters(repaired, runtime_policy)
    errors.extend(filter_errors)
    blocking_errors.extend(filter_errors)
    repairs.extend(filter_repairs)

    date_errors, date_repairs = _validate_date_slots(repaired)
    errors.extend(date_errors)
    blocking_errors.extend(date_errors)
    repairs.extend(date_repairs)

    valid = request_type in runtime_policy.allowed_request_types and not blocking_errors

    result = ValidationResult(
        valid=valid,
        errors=errors,
        repairs=repairs,
        blocking_errors=blocking_errors,
        missing_slots=missing_slots,
        canonical_plan=repaired,
    )
    return result if return_dataclass else result.to_dict()


def _ensure_raw_record_temporal_source_fields(plan: YamlDict, policy: ValidationPolicy) -> List[str]:
    source_fields = plan.get("source_fields")
    if not isinstance(source_fields, list) or not source_fields:
        return []

    temporal_fields: List[str] = []
    table_def = policy.table_definition
    for field in [
        table_def.get("date_field"),
        table_def.get("timestamp_field"),
        table_def.get("end_timestamp_field"),
        *(table_def.get("timestamp_fields") or []),
        *(table_def.get("time_field") or []),
    ]:
        field_name = str(field or "").strip()
        if field_name and field_name in policy.allowed_columns and field_name not in temporal_fields:
            temporal_fields.append(field_name)

    added = [field for field in temporal_fields if field not in source_fields]
    if not added:
        return []

    plan["source_fields"] = [*added, *source_fields]
    return [f"Added raw-record temporal source field(s): {', '.join(added)}"]


def _validate_and_repair_filters(plan: YamlDict, policy: ValidationPolicy) -> tuple[List[str], List[str]]:
    filters = plan.get("filters")
    if filters is None:
        return [], []
    if not isinstance(filters, list):
        plan["filters"] = []
        return ["invalid_filters:not_list"], ["Reset invalid filters to an empty list"]

    errors: List[str] = []
    repairs: List[str] = []
    cleaned: List[YamlDict] = []
    has_valid_date_slot = _has_valid_custom_date_range(plan)
    date_fields = _date_like_fields(policy)

    for item in filters:
        if not isinstance(item, dict):
            repairs.append("Dropped invalid filter that was not an object")
            continue
        field = str(item.get("field") or "").strip()
        op = str(item.get("op") or "").strip()
        value = item.get("value")

        if not field or field not in policy.allowed_columns:
            errors.append(f"filter_field_not_allowed:{field or '<empty>'}")
            continue
        if op not in policy.allowed_filter_ops:
            errors.append(f"filter_op_not_allowed:{op or '<empty>'}:{field}")
            continue
        if not _filter_value_is_valid(op, value, date_like=field in date_fields):
            if field in date_fields and has_valid_date_slot:
                repairs.append(f"Dropped invalid date filter for {field}; canonical date_range is already valid")
                continue
            errors.append(f"filter_value_invalid:{field}:{op}")
            continue

        cleaned.append(deepcopy(item))

    if cleaned != filters:
        plan["filters"] = cleaned
    return errors, repairs


def _validate_date_slots(plan: YamlDict) -> tuple[List[str], List[str]]:
    date_rule = str(plan.get("date_rule") or "none").strip() or "none"
    date_range = plan.get("date_range")
    if date_rule != "custom_range":
        return [], []
    if _has_valid_custom_date_range(plan):
        return [], []
    plan["date_rule"] = "none"
    plan["date_range"] = None
    return ["invalid_custom_date_range"], ["Removed invalid custom date range"]


def _date_like_fields(policy: ValidationPolicy) -> Set[str]:
    table_def = policy.table_definition
    return {
        str(field).strip()
        for field in [
            table_def.get("date_field"),
            table_def.get("timestamp_field"),
            table_def.get("end_timestamp_field"),
            *(table_def.get("timestamp_fields") or []),
            *(table_def.get("time_field") or []),
        ]
        if str(field or "").strip()
    }


def _filter_value_is_valid(op: str, value: Any, *, date_like: bool) -> bool:
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        left = str(value[0] or "").strip()
        right = str(value[1] or "").strip()
        if not left or not right:
            return False
        return (not date_like) or (_is_valid_iso_date_prefix(left) and _is_valid_iso_date_prefix(right))
    if op == "in":
        return isinstance(value, (list, tuple)) and any(str(item or "").strip() for item in value)
    if op == "time_between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        return all(str(item or "").strip() for item in value)
    if op == "relative":
        return bool(str(value or "").strip())
    return value is not None and str(value).strip() != ""


def _has_valid_custom_date_range(plan: YamlDict) -> bool:
    date_range = plan.get("date_range")
    if not isinstance(date_range, dict):
        return False
    start = str(date_range.get("start_date") or "").strip()
    end = str(date_range.get("end_date") or "").strip()
    return _is_valid_iso_date_prefix(start) and _is_valid_iso_date_prefix(end)


def _is_valid_iso_date_prefix(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 10:
        return False
    try:
        datetime.strptime(text[:10], "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _validate_numeric_summary(plan: YamlDict, policy: ValidationPolicy) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    repairs: List[str] = []
    params = plan.get("metric_params")
    if not isinstance(params, dict):
        params = {}
        plan["metric_params"] = params
        repairs.append("Initialized missing metric_params for numeric_summary")

    field = str(params.get("field") or "").strip()
    if not field:
        errors.append("numeric_summary_missing_field")
        return errors, repairs

    measure_fields = set(policy.table_definition.get("measure_fields", []) or [])
    if field not in measure_fields:
        errors.append(f"numeric_summary_field_not_allowed:{field}")
        return errors, repairs

    measure_maps = policy.table_definition.get("measure_field_maps", {}) or {}
    measure_info = measure_maps.get(field, {}) or {}
    allowed_aggs = set(measure_info.get("allowed_aggregations", []) or [])
    numeric_aggregation = str(
        params.get("numeric_aggregation")
        or params.get("aggregation")
        or ""
    ).strip().lower()
    if not numeric_aggregation:
        errors.append("numeric_summary_missing_aggregation")
        return errors, repairs
    if allowed_aggs and numeric_aggregation not in allowed_aggs:
        errors.append(f"numeric_summary_aggregation_not_allowed:{numeric_aggregation}:{field}")
        return errors, repairs

    params["numeric_aggregation"] = numeric_aggregation
    plan["aggregation"] = "numeric_summary"
    params.setdefault("unit", measure_info.get("unit"))
    params.setdefault("field_label", measure_info.get("label") or field)
    for group_field in plan.get("group_by", []) or []:
        if group_field not in policy.groupable_dimensions:
            errors.append(f"group_by_not_allowed:{group_field}")
    return errors, repairs
