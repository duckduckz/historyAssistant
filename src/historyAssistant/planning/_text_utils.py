from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

YamlDict = Dict[str, Any]


def norm_text(value: Any) -> str:
    """lowercase, collapse whitespace, replace underscores with spaces."""
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").lower()).strip()


def compact_text(value: Any) -> str:
    """strip everything except lowercase letters and digits."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def as_list(value: Any) -> List[Any]:
    """wrap a scalar in a list; return lists as-is; return [] for None."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedupe(values: List[str]) -> List[str]:
    """remove duplicates while preserving insertion order (exact string match)."""
    seen: set = set()
    result: List[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def dedupe_by_norm(values: List[str]) -> List[str]:
    """remove duplicates using norm_text as the key (used for dimension values)."""
    seen: set = set()
    result: List[str] = []
    for v in values:
        key = norm_text(v)
        if key and key not in seen:
            seen.add(key)
            result.append(v)
    return result


def parse_json_text(raw: str) -> YamlDict:
    """parse a JSON string that may be wrapped in markdown code fences."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start: end + 1])
        raise


__all__ = [
    "norm_text",
    "compact_text",
    "as_list",
    "dedupe",
    "dedupe_by_norm",
    "parse_json_text",
]
