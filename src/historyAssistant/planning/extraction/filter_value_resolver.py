from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from historyAssistant.planning._text_utils import compact_text

YamlDict = Dict[str, Any]


@dataclass(frozen=True)
class FilterCandidate:
    field: str
    raw_value: Any
    op: str = "="
    source: str = ""
    span: Optional[Tuple[int, int]] = None


def resolve_filter_candidates(raw_candidates: Iterable[Any], table_definition: YamlDict) -> List[YamlDict]:
    filters: List[YamlDict] = []
    allowed_fields = set(table_definition.get("dimensions", []) or [])
    dim_maps = table_definition.get("dimension_value_maps", {}) or {}

    for raw in raw_candidates or []:
        candidate = coerce_filter_candidate(raw)
        if not candidate or candidate.field not in allowed_fields:
            continue
        if any(item.get("field") == candidate.field for item in filters):
            continue
        resolved = _resolve_candidate(candidate, dim_maps.get(candidate.field, {}) or {}, table_definition)
        if resolved is not None:
            filters.append(resolved)

    return dedupe_filters(_drop_absorbed_filters(filters, dim_maps))


def coerce_filter_candidate(raw: Any) -> Optional[FilterCandidate]:
    if isinstance(raw, FilterCandidate):
        return raw
    if not isinstance(raw, dict):
        return None
    field = str(raw.get("field") or "")
    if not field:
        return None
    return FilterCandidate(
        field=field,
        op=str(raw.get("op") or "="),
        raw_value=raw.get("value"),
        source=str(raw.get("source") or "raw"),
    )


def normalize_room_code(raw_value: str) -> Optional[str]:
    text = str(raw_value or "").strip().upper()
    simple_match = re.fullmatch(r"(K|SANITAIR)\s*(\d{1,3})", text)
    if simple_match:
        prefix, digits = simple_match.groups()
        return f"{prefix} {digits.zfill(3)}"
    dvc_match = re.fullmatch(r"(DVC_[A-Z]+)\s*(\d{1,3})", text)
    if dvc_match:
        prefix, digits = dvc_match.groups()
        return f"{prefix} {digits.zfill(3)}"
    return None


def dedupe_filters(filters: List[YamlDict]) -> List[YamlDict]:
    result: List[YamlDict] = []
    seen = set()
    for filt in filters:
        key = (filt.get("field"), filt.get("op"), repr(filt.get("value")))
        if key in seen:
            continue
        seen.add(key)
        result.append(filt)
    return result


def _drop_absorbed_filters(filters: List[YamlDict], dim_maps: YamlDict) -> List[YamlDict]:
    if not filters:
        return filters

    absorbed_fields: set[str] = set()
    for filt in filters:
        field = str(filt.get("field") or "")
        field_info = dim_maps.get(field, {}) or {}
        absorbed_field = str(field_info.get("absorbs_field") or "")
        if not absorbed_field:
            continue
        absorbed_filter = next(
            (item for item in filters if str(item.get("field") or "") == absorbed_field),
            None,
        )
        if absorbed_filter and _filter_value_absorbs(field_info, filt.get("value"), absorbed_filter.get("value")):
            absorbed_fields.add(absorbed_field)

    if not absorbed_fields:
        return filters
    return [filt for filt in filters if str(filt.get("field") or "") not in absorbed_fields]


def _filter_value_absorbs(field_info: YamlDict, value: Any, absorbed_value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    absorbed_values = absorbed_value if isinstance(absorbed_value, list) else [absorbed_value]
    marker = str(field_info.get("already_merged_marker") or "").strip()
    separator = str(field_info.get("absorb_separator") or "").strip()

    for raw_value in values:
        value_text = str(raw_value or "")
        value_compact = compact_value(value_text)
        if not value_compact:
            continue
        has_marker = bool(marker and marker.lower() in value_text.lower())
        has_separator = bool(separator and separator.lower() in value_text.lower())
        for raw_absorbed in absorbed_values:
            absorbed_compact = compact_value(raw_absorbed)
            if absorbed_compact and absorbed_compact in value_compact:
                return True
        if has_marker or has_separator:
            return True
    return False


def ignored_allowed_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "unknown", "none", "null"}


def contains_phrase(question: str, phrase: Any) -> bool:
    text = str(phrase or "").strip()
    if not text:
        return False
    return bool(re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])", question or ""))


def compact_value(value: Any) -> str:
    return compact_text(value)


def norm_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").lower()).strip()


def _resolve_candidate(candidate: FilterCandidate, field_info: YamlDict, table_definition: YamlDict) -> Optional[YamlDict]:
    value = candidate.raw_value
    op = candidate.op if candidate.op in {"=", "!=", "in", "wildcard"} else "="

    if op == "wildcard":
        return {"field": candidate.field, "op": "wildcard", "value": str(value)}
    if op == "in" and isinstance(value, list):
        values = [
            resolved for item in value
            for resolved in [_resolve_single_value(candidate.field, item, field_info, table_definition)]
            if resolved is not None
        ]
        return {"field": candidate.field, "op": "in", "value": _unique_values(values)} if values else None

    resolved = _resolve_single_value(candidate.field, value, field_info, table_definition)
    if resolved is None:
        return None
    return {"field": candidate.field, "op": op, "value": resolved}


def _resolve_single_value(field: str, raw_value: Any, field_info: YamlDict, table_definition: YamlDict) -> Optional[Any]:
    if raw_value is None:
        return None

    for label, mapped in (field_info.get("value_map", {}) or {}).items():
        if norm_value(label) == norm_value(raw_value):
            return mapped

    allowed_values = [
        value for value in field_info.get("allowed_values", []) or []
        if not ignored_allowed_value(value)
    ]
    if allowed_values:
        resolved = _resolve_known_value(raw_value, allowed_values)
        if resolved is not None:
            return resolved
        normalized = _normalize_typed_value(field, raw_value, table_definition)
        return _resolve_known_value(normalized, allowed_values) if normalized != raw_value else None

    normalized = _normalize_typed_value(field, raw_value, table_definition)
    if normalized is None or _looks_like_non_dimension_value(normalized):
        return None
    if _allows_free_text_filter(field_info) or field_info.get("field_type") in {"integer", "float", "boolean"}:
        return normalized
    return None


def _resolve_known_value(raw_value: Any, allowed_values: List[Any]) -> Optional[Any]:
    raw_norm = norm_value(raw_value)
    raw_compact = compact_value(raw_value)
    raw_tokens = set(_value_tokens(raw_value))
    if not raw_norm:
        return None

    exact = [value for value in allowed_values if norm_value(value) == raw_norm]
    if len(exact) == 1:
        return exact[0]

    compact_exact = [value for value in allowed_values if compact_value(value) == raw_compact]
    if len(compact_exact) == 1:
        return compact_exact[0]

    token_matches = []
    for value in allowed_values:
        candidate_tokens = set(_value_tokens(value))
        candidate_compact = compact_value(value)
        if raw_tokens and raw_tokens.issubset(candidate_tokens):
            token_matches.append(value)
        elif raw_compact and candidate_compact.endswith(raw_compact):
            token_matches.append(value)

    token_matches = _unique_values(token_matches)
    return token_matches[0] if len(token_matches) == 1 else None


def _normalize_typed_value(field: str, raw_value: Any, table_definition: YamlDict) -> Any:
    semantic_fields = table_definition.get("semantic_fields", {}) or {}
    concept = next((name for name, candidate in semantic_fields.items() if candidate == field), "")

    if concept == "room":
        return normalize_room_code(str(raw_value)) or str(raw_value).strip()
    if concept == "zone" and str(field) == "zoneName":
        compact = re.sub(r"\s+", "", str(raw_value or "").strip()).upper()
        return compact if compact.startswith("ZONE") else f"ZONE{compact}"

    text = str(raw_value).strip() if isinstance(raw_value, str) else raw_value
    if isinstance(text, str) and field.lower().endswith("hour") and text.isdigit():
        return int(text)
    if isinstance(text, str) and text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def _looks_like_non_dimension_value(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if re.search(r"(?i)^(?:custom_range|date_range|last_|this_|today|yesterday|tomorrow|in\s+|during\s+|from\s+|to\s+|between\s+|on\s+|at\s+)", text):
        return True
    return bool(re.fullmatch(r"(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?", text))


def _allows_free_text_filter(field_info: YamlDict) -> bool:
    if "allow_free_text_filter" in field_info:
        return bool(field_info.get("allow_free_text_filter"))
    if field_info.get("load_values_from_es") or field_info.get("allowed_values"):
        return False
    return True


def _value_tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", norm_value(value))


def _unique_values(values: List[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
