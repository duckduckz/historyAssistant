from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import Any, List, Optional

from .loader import ConfigBundle, YamlDict, resolve_elasticsearch_config

logger = logging.getLogger(__name__)

DEFAULT_MAX_VALUES = 500


def _build_es_client(app_config: YamlDict) -> Optional[Any]:
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        logger.warning("elasticsearch package not available; skipping dynamic allowed_values load")
        return None
    try:
        es_cfg = resolve_elasticsearch_config(app_config)
        return Elasticsearch([f"http://{es_cfg['host']}:{es_cfg['port']}"])
    except Exception as exc:
        logger.warning("Could not build ES client for dynamic value loading: %s", exc)
        return None


def _terms_agg(es: Any, index: str, agg_field: str, size: int) -> List[str]:
    result = es.search(
        index=index,
        size=0,
        body={
            "aggs": {
                "distinct_values": {
                    "terms": {"field": agg_field, "size": size}
                }
            }
        },
    )
    buckets = (
        result.get("aggregations", {})
        .get("distinct_values", {})
        .get("buckets", [])
    )
    return [str(bucket["key"]) for bucket in buckets if "key" in bucket]


def _fetch_distinct_values(
    es: Any, index: str, field: str, size: int, *, prefer_keyword: bool
) -> List[str]:
    candidates: List[str] = (
        [f"{field}.keyword", field] if prefer_keyword else [field, f"{field}.keyword"]
    )
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            values = _terms_agg(es, index, candidate, size)
            if values:
                return values
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        logger.warning(
            "ES distinct values fetch failed for index=%s field=%s: %s",
            index, field, last_error,
        )
    return []


def _any_dimension_flagged(tables: YamlDict) -> bool:
    for table_def in (tables or {}).values():
        if not isinstance(table_def, dict):
            continue
        for field_cfg in (table_def.get("dimension_value_maps") or {}).values():
            if isinstance(field_cfg, dict) and field_cfg.get("load_values_from_es"):
                return True
    return False


def populate_dynamic_allowed_values(bundle: ConfigBundle) -> ConfigBundle:
    
    table_registry = bundle.table_registry or {}
    tables = table_registry.get("tables") or {}
    if not tables or not _any_dimension_flagged(tables):
        return bundle

    es = _build_es_client(bundle.app_config)
    if es is None:
        return bundle

    new_tables = copy.deepcopy(tables)
    populated = False
    for table_name, table_def in new_tables.items():
        if not isinstance(table_def, dict):
            continue
        index = table_def.get("es_index")
        if not index:
            continue
        keyword_fields = set(table_def.get("keyword_fields") or [])
        dvms = table_def.get("dimension_value_maps") or {}
        for field_name, field_cfg in dvms.items():
            if not isinstance(field_cfg, dict):
                continue
            if not field_cfg.get("load_values_from_es"):
                continue
            max_size = int(field_cfg.get("load_values_max") or DEFAULT_MAX_VALUES)
            prefer_keyword = field_name in keyword_fields or str(field_cfg.get("field_type") or "").lower() == "keyword"
            values = _fetch_distinct_values(es, index, field_name, max_size, prefer_keyword=prefer_keyword)
            if values:
                field_cfg["allowed_values"] = values
                populated = True
                logger.info(
                    "Loaded %d allowed_values for %s.%s from ES",
                    len(values), table_name, field_name,
                )

    if not populated:
        return bundle

    new_registry = dict(table_registry)
    new_registry["tables"] = new_tables
    return replace(bundle, table_registry=new_registry)


__all__ = ["populate_dynamic_allowed_values"]
