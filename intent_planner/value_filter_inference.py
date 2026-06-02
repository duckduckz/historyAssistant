from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from historyAssistant.config import ConfigBundle
from historyAssistant.planning._text_utils import as_list, compact_text, norm_text

YamlDict = Dict[str, Any]


def infer_value_filters(
    question: str,
    *,
    bundle: ConfigBundle,
    target_table: str,
    existing_filters: Optional[List[YamlDict]] = None,
) -> List[YamlDict]:
    
    table_def = (bundle.table_registry.get("tables", {}) or {}).get(target_table, {}) or {}
    dim_maps = table_def.get("dimension_value_maps", {}) or {}
    if not dim_maps:
        return []

    existing_fields = {str(f.get("field") or "") for f in existing_filters or []}
    filters: List[YamlDict] = []
    for field in table_def.get("dimensions", []) or []:
        field = str(field)
        if field in existing_fields:
            continue
        allowed_values = _known_values(dim_maps.get(field, {}) or {})
        if not allowed_values:
            continue

        aliases = _dimension_aliases(bundle, field)
        if not aliases:
            aliases = [field]
        value_text = _extract_value_after_alias(question, aliases)
        if not value_text:
            continue

        matches = _match_known_values(value_text, allowed_values)
        if len(matches) == 1:
            filters.append({"field": field, "op": "=", "value": matches[0]})
        elif len(matches) > 1:
            filters.append({"field": field, "op": "in", "value": matches})

    return filters


def remove_filtered_group_by(group_by: List[str], filters: List[YamlDict]) -> List[str]:
    filtered_fields = {str(f.get("field") or "") for f in filters or []}
    return [field for field in group_by if field not in filtered_fields]


def _dimension_aliases(bundle: ConfigBundle, field: str) -> List[str]:
    aliases = [field]
    cfg = ((bundle.vocabulary or {}).get("dimensions") or {}).get(field, {}) or {}
    if isinstance(cfg, dict):
        aliases.extend(str(v) for v in _as_list(cfg.get("synonyms")) if str(v).strip())
    # Longest aliases first so "location ward" wins before "ward".
    seen = set()
    result = []
    for alias in sorted(aliases, key=lambda v: len(str(v)), reverse=True):
        key = _norm(alias)
        if key and key not in seen:
            seen.add(key)
            result.append(str(alias))
    return result


def _extract_value_after_alias(question: str, aliases: List[str]) -> Optional[str]:
    text = str(question or "")
    for alias in aliases:
        alias_pat = re.escape(str(alias).replace("_", " "))
        match = re.search(
            rf"(?i)(?:\bfor\b|\bin\b|\bat\b|\bfrom\b|\bonly\b|\bwhere\b)?\s*\b{alias_pat}\b\s+(?:is\s+|=\\s*)?([A-Za-z0-9_ -]{{1,40}})",
            text.replace("_", " "),
        )
        if not match:
            continue
        raw = match.group(1)
        value = re.split(
            r"(?i)\b(?:last|this|next|today|yesterday|tomorrow|by|per|each|compare|vs|versus|and|with|group|count|movement|calls?|presence)\b",
            raw,
            maxsplit=1,
        )[0].strip(" ,.;:")
        if value:
            return value
    return None


def _match_known_values(raw_value: str, allowed_values: List[str]) -> List[str]:
    raw_norm = _norm(raw_value)
    if not raw_norm:
        return []

    exact = [value for value in allowed_values if _norm(value) == raw_norm]
    if exact:
        return exact

    prefix = [
        value
        for value in allowed_values
        if _norm(value).startswith(raw_norm)
        or _compact_norm(value).startswith(_compact_norm(raw_value))
    ]
    if prefix:
        return prefix

    contained = [
        value
        for value in allowed_values
        if raw_norm in _norm(value)
        or _compact_norm(raw_value) in _compact_norm(value)
    ]
    return contained


def _known_values(dim_info: YamlDict) -> List[str]:
    values = []
    for value in dim_info.get("allowed_values", []) or []:
        text = str(value or "").strip()
        if text and text.lower() != "unknown":
            values.append(text)
    seen = set()
    result = []
    for value in sorted(values, key=len):
        key = _norm(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _norm(value: Any) -> str:
    return norm_text(value)


def _compact_norm(value: Any) -> str:
    return compact_text(value)


def _as_list(value: Any) -> Iterable[Any]:
    return as_list(value)
