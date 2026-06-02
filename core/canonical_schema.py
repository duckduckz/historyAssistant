from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)

REQUEST_TYPE_QA = "qa"
REQUEST_TYPE_CHART = "chart"

EXECUTABLE_REQUEST_TYPES = {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}
ALL_REQUEST_TYPES = EXECUTABLE_REQUEST_TYPES

CANONICAL_TABLE_HISTORY_CALLS = "historyCalls"

ALLOWED_FILTER_OPS = {"=", "!=", ">", ">=", "<", "<=", "in", "between", "relative", "time_between", "wildcard"}

ALLOWED_AGGREGATIONS = {
    "count", "sum", "avg", "min", "max",
    "median", "percentile",
    "cardinality",
    "rate_threshold", "count_threshold", "avg_count_per_group",
    "dq_rate", "dq_count", "dq_count_any_true", "dq_count_all_true",
    "raw_records",  
    "room_switch_within_minutes",
    "time_bucket_summary",  
}

ALLOWED_BUCKET_SUMMARIES = {"avg", "min", "max", "sum", "median", "count"}

ALLOWED_DATE_RULES = {
    "none", "yesterday", "last_7_days", "last_week", "calendar_week",
    "last_calendar_week", "this_month", "last_month", "custom_range", "rolling_days",
}

# these are the allowed sort directions.
ALLOWED_ORDER_DIRECTIONS = {"asc", "desc"}


@dataclass(frozen=True)
class CanonicalFilter:
    # this stores one canonical filter item.
    field: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.op not in ALLOWED_FILTER_OPS:
            logger.warning("Filter op '%s' not in allowed ops: %s", self.op, ALLOWED_FILTER_OPS)


@dataclass(frozen=True)
class CanonicalTimeRange:
    # timeofday range like 03:40 to 03:50.
    start_time: Optional[str] = None  # HH:MM or HH:MM:SS format
    end_time: Optional[str] = None    # HH:MM or HH:MM:SS format


@dataclass(frozen=True)
class CanonicalDateRange:
    # date range for custom_range.
    start_date: Optional[str] = None  # YYYY-MM-DD or DD/MM/YYYY
    end_date: Optional[str] = None    # YYYY-MM-DD or DD/MM/YYYY


@dataclass(frozen=True)
class CanonicalDateParams:
    # extra parameters for date rules such as rolling_days.
    days: Optional[int] = None


@dataclass(frozen=True)
class CanonicalOrderBy:
    # field and direction for sorting.
    field: str
    direction: str
    target_group: Optional[str] = None

    def __post_init__(self) -> None:
        if self.direction not in ALLOWED_ORDER_DIRECTIONS:
            logger.warning("Order direction '%s' not in allowed: %s", self.direction, ALLOWED_ORDER_DIRECTIONS)


@dataclass(frozen=True)
class CanonicalDisplayOrder:
    # presentation ordering for a grouped field independent from ranking.
    field: str
    direction: str
    mode: str = "configured"

    def __post_init__(self) -> None:
        if self.direction not in ALLOWED_ORDER_DIRECTIONS:
            logger.warning("Display order direction '%s' not in allowed: %s", self.direction, ALLOWED_ORDER_DIRECTIONS)


def default_target_tables() -> List[str]:
    return [CANONICAL_TABLE_HISTORY_CALLS]


@dataclass(frozen=True)
class CanonicalPlanBase:
    request_type: str
    target_tables: List[str] = field(default_factory=default_target_tables)
    normalized_sub_request: Optional[str] = None
    sub_request_id: Optional[int] = None
    classification_confidence: Optional[float] = None


@dataclass(frozen=True)
class CanonicalCompareTo:
    date_rule: Optional[str] = None  # last_week, last_month, custom_range
    date_range: Optional[CanonicalDateRange] = None  # custom_range: explicit start/end dates
    filters: List[CanonicalFilter] = field(default_factory=list)  # dimension filters for the comparison side

@dataclass(frozen=True)
class QaPlan(CanonicalPlanBase):
    # canonical plan type for normal qa requests that later become es queries.
    request_type: str = REQUEST_TYPE_QA
    metric: str = "total_calls"
    aggregation: Optional[str] = None
    metric_params: Dict[str, Any] = field(default_factory=dict)
    filters: List[CanonicalFilter] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    ranking_order: Optional[CanonicalOrderBy] = None
    display_order: List[CanonicalDisplayOrder] = field(default_factory=list)
    order_by: Optional[CanonicalOrderBy] = None
    limit: Optional[int] = None
    scope_policy: Optional[str] = None
    outlier_policy: Optional[str] = None
    
    # keep date and time filtering structured before es builder converts them.
    date_rule: Optional[str] = None  # eg: yesterday, last_7_days, last_week, custom_range
    date_params: Optional[CanonicalDateParams] = None
    date_range: Optional[CanonicalDateRange] = None  #custom_range: explicit start/end dates
    time_range: Optional[CanonicalTimeRange] = None  # timeofday filtering
    
    # comparison period for queries like compare last week to the week before
    compare_to: Optional[CanonicalCompareTo] = None

    bucket_interval: Optional[str] = None
    bucket_summary: Optional[str] = None

    def __post_init__(self) -> None:
        if self.aggregation and self.aggregation not in ALLOWED_AGGREGATIONS:
            logger.warning("Aggregation '%s' not in allowed: %s", self.aggregation, ALLOWED_AGGREGATIONS)
        if self.date_rule and self.date_rule not in ALLOWED_DATE_RULES:
            logger.warning("Date rule '%s' not in allowed: %s", self.date_rule, ALLOWED_DATE_RULES)


CanonicalPlan = QaPlan

__all__ = [
    # constants
    "REQUEST_TYPE_QA",
    "REQUEST_TYPE_CHART",
    "EXECUTABLE_REQUEST_TYPES",
    "ALL_REQUEST_TYPES",
    "CANONICAL_TABLE_HISTORY_CALLS",
    "ALLOWED_FILTER_OPS",
    "ALLOWED_AGGREGATIONS",
    "ALLOWED_BUCKET_SUMMARIES",
    "ALLOWED_DATE_RULES",
    "ALLOWED_ORDER_DIRECTIONS",
    # data classes
    "CanonicalFilter",
    "CanonicalOrderBy",
    "CanonicalDisplayOrder",
    "CanonicalTimeRange",
    "CanonicalDateRange",
    "CanonicalDateParams",
    "CanonicalCompareTo",
    "CanonicalPlanBase",
    "QaPlan",
    "CanonicalPlan",
    # helper functions
    "default_target_tables",
    # from dict parsers
    "filter_from_dict",
    "order_by_from_dict",
    "display_order_from_list",
    "time_range_from_dict",
    "date_range_from_dict",
    "date_params_from_dict",
    "qa_plan_from_dict",
    "canonical_plan_from_dict",
    "compare_to_from_dict",
]


def validate_request_type(request_type: str) -> str:
    # request type is one of the known canonical request types.
    if request_type not in ALL_REQUEST_TYPES:
        raise ValueError(f"Unsupported canonical request_type: {request_type}")
    return request_type


def require_mapping(data: Any, field_name: str) -> Mapping[str, Any]:
    # value is a mapping before reading fields from it.
    if not isinstance(data, Mapping):
        raise TypeError(f"Field '{field_name}' must be a mapping")
    return data


def ensure_list(value: Any, field_name: str) -> List[Any]:
    # make sure a field is a list and normalise null into an empty list.
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Field '{field_name}' must be a list")
    return list(value)


def string_list(value: Any, field_name: str) -> List[str]:
    # make sure a list field contains only strings.
    items = ensure_list(value, field_name)
    normalized: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"Field '{field_name}' must contain only strings")
        normalized.append(item)
    return normalized


def target_tables_from_dict(data: Mapping[str, Any]) -> List[str]:
    # read and validate the target_tables field before execution maps it 
    target_tables = data.get("target_tables", default_target_tables())
    if not isinstance(target_tables, list):
        raise TypeError("Field 'target_tables' must be a list")
    normalized: List[str] = []
    for table_name in target_tables:
        if not isinstance(table_name, str):
            raise TypeError("Field 'target_tables' must contain only strings")
        normalized.append(table_name)
    return normalized


def optional_int(value: Any, field_name: str) -> Optional[int]:
    # validate an optional integer field.
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Field '{field_name}' must be an integer or null")
    return value


def optional_float(value: Any, field_name: str) -> Optional[float]:
    # validate an optional float field.
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Field '{field_name}' must be a float or null")
    return float(value)


def optional_string(value: Any, field_name: str) -> Optional[str]:
    # validate an optional string field.
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Field '{field_name}' must be a string or null")
    return value


def base_kwargs_from_dict(data: Mapping[str, Any], request_type: str) -> Dict[str, Any]:
    # collect the shared base fields used by canonical plan objects.
    return {
        "request_type": request_type,
        "target_tables": target_tables_from_dict(data),
        "normalized_sub_request": optional_string(data.get("normalized_sub_request"), "normalized_sub_request"),
        "sub_request_id": optional_int(data.get("sub_request_id"), "sub_request_id"),
        "classification_confidence": optional_float(
            data.get("classification_confidence"), "classification_confidence"
        ),
    }


def filter_from_dict(data: Mapping[str, Any]) -> CanonicalFilter:
    # convert one raw filter dictionary into a CanonicalFilter object.
    raw = require_mapping(data, "filter")
    field_name = raw.get("field")
    op = raw.get("op")
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("Filter field 'field' must be a non-empty string")
    if not isinstance(op, str) or not op:
        raise TypeError("Filter field 'op' must be a non-empty string")
    return CanonicalFilter(field=field_name, op=op, value=raw.get("value"))


def order_by_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[CanonicalOrderBy]:
    # convert a raw order_by dictionary into a CanonicalOrderBy object.
    if data is None:
        return None
    raw = require_mapping(data, "order_by")
    field_name = raw.get("field")
    direction = raw.get("direction")
    target_group = raw.get("target_group")
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("Field 'order_by.field' must be a non-empty string")
    if not isinstance(direction, str) or not direction:
        raise TypeError("Field 'order_by.direction' must be a non-empty string")
    if target_group is not None and (not isinstance(target_group, str) or not target_group):
        raise TypeError("Field 'order_by.target_group' must be a non-empty string or null")
    return CanonicalOrderBy(field=field_name, direction=direction, target_group=target_group)


def display_order_from_list(data: Optional[List[Mapping[str, Any]]]) -> List[CanonicalDisplayOrder]:
    # convert a raw display_order list into CanonicalDisplayOrder objects.
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError("Field 'display_order' must be a list or null")
    items: List[CanonicalDisplayOrder] = []
    for idx, raw_item in enumerate(data):
        raw = require_mapping(raw_item, f"display_order[{idx}]")
        field_name = raw.get("field")
        direction = raw.get("direction", "asc")
        mode = raw.get("mode", "configured")
        if not isinstance(field_name, str) or not field_name:
            raise TypeError(f"Field 'display_order[{idx}].field' must be a non-empty string")
        if not isinstance(direction, str) or not direction:
            raise TypeError(f"Field 'display_order[{idx}].direction' must be a non-empty string")
        if not isinstance(mode, str) or not mode:
            raise TypeError(f"Field 'display_order[{idx}].mode' must be a non-empty string")
        items.append(CanonicalDisplayOrder(field=field_name, direction=direction, mode=mode))
    return items


def time_range_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[CanonicalTimeRange]:
    # convert a raw time_range dictionary into a CanonicalTimeRange object.
    if data is None:
        return None
    raw = require_mapping(data, "time_range")
    start_time = optional_string(raw.get("start_time"), "time_range.start_time")
    end_time = optional_string(raw.get("end_time"), "time_range.end_time")
    if start_time is None and end_time is None:
        return None
    return CanonicalTimeRange(start_time=start_time, end_time=end_time)


def date_range_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[CanonicalDateRange]:
    # convert a raw date_range dictionary into a CanonicalDateRange object.
    if data is None:
        return None
    raw = require_mapping(data, "date_range")
    start_date = optional_string(raw.get("start_date"), "date_range.start_date")
    end_date = optional_string(raw.get("end_date"), "date_range.end_date")
    if start_date is None and end_date is None:
        return None
    return CanonicalDateRange(start_date=start_date, end_date=end_date)


def date_params_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[CanonicalDateParams]:
    # convert a raw date_params dictionary into a CanonicalDateParams object.
    if data is None:
        return None
    raw = require_mapping(data, "date_params")
    days = optional_int(raw.get("days"), "date_params.days")
    if days is None:
        return None
    return CanonicalDateParams(days=days)


def compare_to_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[CanonicalCompareTo]:
    # convert a raw compare_to dictionary into a CanonicalCompareTo object.
    if data is None:
        return None
    raw = require_mapping(data, "compare_to")
    date_rule = optional_string(raw.get("date_rule"), "compare_to.date_rule")
    date_range = date_range_from_dict(raw.get("date_range"))
    filters = [filter_from_dict(item) for item in ensure_list(raw.get("filters"), "compare_to.filters")]
    if date_rule is None and date_range is None and not filters:
        return None
    return CanonicalCompareTo(date_rule=date_rule, date_range=date_range, filters=filters)


def qa_plan_from_dict(data: Mapping[str, Any]) -> QaPlan:
    
    raw = require_mapping(data, "plan")
    request_type = raw.get("request_type", REQUEST_TYPE_QA)
    validate_request_type(request_type)
    if request_type not in {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}:
        raise ValueError(f"qa_plan_from_dict expected request_type '{REQUEST_TYPE_QA}' or '{REQUEST_TYPE_CHART}', got '{request_type}'")
    return QaPlan(
        **base_kwargs_from_dict(raw, request_type),
        metric=optional_string(raw.get("metric"), "metric") or "total_calls",
        aggregation=optional_string(raw.get("aggregation"), "aggregation"),
        metric_params=dict(raw.get("metric_params") or {}) if isinstance(raw.get("metric_params"), Mapping) else {},
        filters=[filter_from_dict(item) for item in ensure_list(raw.get("filters"), "filters")],
        group_by=string_list(raw.get("group_by"), "group_by"),
        ranking_order=order_by_from_dict(raw.get("ranking_order") or raw.get("order_by")),
        display_order=display_order_from_list(raw.get("display_order")),
        order_by=order_by_from_dict(raw.get("order_by")),
        limit=optional_int(raw.get("limit"), "limit"),
        scope_policy=optional_string(raw.get("scope_policy"), "scope_policy"),
        outlier_policy=optional_string(raw.get("outlier_policy"), "outlier_policy"),
        date_rule=optional_string(raw.get("date_rule"), "date_rule"),
        date_params=date_params_from_dict(raw.get("date_params")),
        date_range=date_range_from_dict(raw.get("date_range")),
        time_range=time_range_from_dict(raw.get("time_range")),
        compare_to=compare_to_from_dict(raw.get("compare_to")),
        bucket_interval=optional_string(raw.get("bucket_interval"), "bucket_interval"),
        bucket_summary=optional_string(raw.get("bucket_summary"), "bucket_summary"),
    )


def canonical_plan_from_dict(data: Mapping[str, Any]) -> CanonicalPlan:
    # choose right canonical plan object before later steps run validation and execution.
    raw = require_mapping(data, "plan")
    request_type = raw.get("request_type")
    if not isinstance(request_type, str):
        raise TypeError("Field 'request_type' must be a string")
    request_type = validate_request_type(request_type)
    if request_type in {REQUEST_TYPE_QA, REQUEST_TYPE_CHART}:
        return qa_plan_from_dict(raw)
    raise ValueError(f"request_type '{request_type}' is not a canonical plan type")
