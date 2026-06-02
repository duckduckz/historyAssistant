from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


YamlDict = Dict[str, Any]


CONFIG_NAME_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "app_config": ("app_config.yaml",),
    "metric_catalog": ("metric_catalog.yaml",),
    "date_rules": ("date_rules.yaml",),
    "table_registry": ("table_registry.yaml",),
    "vocabulary": ("vocabulary.yaml",),
    "chart_catalog": ("chart_catalog.yaml",),
    "unsupported_topic": ("unsupported_topic.yaml",),
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_dir: Path
    data_dir: Path
    notebooks_dir: Path
    src_dir: Path


@dataclass(frozen=True)
class ConfigBundle:
    paths: ProjectPaths
    app_config: YamlDict
    metric_catalog: YamlDict
    date_rules: YamlDict
    table_registry: YamlDict
    vocabulary: YamlDict
    chart_catalog: YamlDict
    unsupported_topic: YamlDict

    @property
    def defaults(self) -> YamlDict:
        return self.app_config.get("defaults", {})

    @property
    def runtime(self) -> YamlDict:
        return self.app_config.get("runtime", {})

    @property
    def metrics(self) -> YamlDict:
        return self.metric_catalog.get("metrics", {})


def find_project_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").exists() and (
            (candidate / "data").exists()
        ):
            return candidate
    raise FileNotFoundError("Could not find project root containing configs and data folder")


def build_project_paths(root: Optional[Path] = None) -> ProjectPaths:
    resolved_root = find_project_root(root)
    return ProjectPaths(
        root=resolved_root,
        config_dir=resolved_root / "configs",
        data_dir=resolved_root / "data",
        notebooks_dir=resolved_root / "notebooks",
        src_dir=resolved_root / "src",
    )


def load_yaml(path: Path) -> YamlDict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_config_path(config_key: str, *, config_dir: Optional[Path] = None, root: Optional[Path] = None) -> Path:
    paths = build_project_paths(root)
    cfg_dir = config_dir or paths.config_dir
    names = CONFIG_NAME_CANDIDATES.get(config_key)
    if not names:
        raise KeyError(f"Unknown config key: {config_key}")

    search_dirs = [cfg_dir]

    if cfg_dir == paths.config_dir:  # if using default config_dir
        search_dirs.extend([
            cfg_dir / "shared",
            cfg_dir / "calls",
            cfg_dir / "locations",
            cfg_dir / "workflows",
        ])
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for name in names:
            candidate = search_dir / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not locate config for '{config_key}' in {cfg_dir} or subdirectories")


def load_config(config_key: str, *, config_dir: Optional[Path] = None, root: Optional[Path] = None) -> YamlDict:
    return load_yaml(resolve_config_path(config_key, config_dir=config_dir, root=root))


def load_optional_config(config_key: str, *, config_dir: Optional[Path] = None, root: Optional[Path] = None) -> YamlDict:
    try:
        return load_config(config_key, config_dir=config_dir, root=root)
    except FileNotFoundError:
        return {}


def _apply_dynamic_allowed_values(bundle: ConfigBundle) -> ConfigBundle:
    """Late-bind ES-driven allowed_values without creating a circular import."""
    from .dynamic_dimension_loader import populate_dynamic_allowed_values
    return populate_dynamic_allowed_values(bundle)


@lru_cache(maxsize=4)
def load_all_configs(root: Optional[Path] = None) -> ConfigBundle:
    paths = build_project_paths(root)

    bundle = ConfigBundle(
        paths=paths,
        app_config=load_optional_config("app_config", config_dir=paths.config_dir, root=paths.root),
        metric_catalog=load_optional_config("metric_catalog", config_dir=paths.config_dir, root=paths.root),
        date_rules=load_optional_config("date_rules", config_dir=paths.config_dir, root=paths.root),
        table_registry=load_optional_config("table_registry", config_dir=paths.config_dir, root=paths.root),
        vocabulary=load_optional_config("vocabulary", config_dir=paths.config_dir, root=paths.root),
        chart_catalog=load_optional_config("chart_catalog", config_dir=paths.config_dir, root=paths.root),
        unsupported_topic=load_optional_config("unsupported_topic", config_dir=paths.config_dir, root=paths.root),
    )
    return _apply_dynamic_allowed_values(bundle)


@lru_cache(maxsize=16)
def load_source_configs(source_id: str, root: Optional[Path] = None) -> ConfigBundle:
    paths = build_project_paths(root)
    source_dir = paths.config_dir / source_id

    if not source_dir.exists():
        raise FileNotFoundError(f"Unknown source config folder: {source_dir}")

    bundle = ConfigBundle(
        paths=paths,
        app_config=load_optional_config("app_config", config_dir=paths.config_dir, root=paths.root),
        metric_catalog=load_optional_config("metric_catalog", config_dir=source_dir, root=paths.root),
        date_rules=load_optional_config("date_rules", config_dir=paths.config_dir, root=paths.root),
        table_registry=load_optional_config("table_registry", config_dir=source_dir, root=paths.root),
        vocabulary=load_optional_config("vocabulary", config_dir=source_dir, root=paths.root),
        chart_catalog=load_optional_config("chart_catalog", config_dir=source_dir, root=paths.root),
        unsupported_topic=load_optional_config("unsupported_topic", config_dir=paths.config_dir, root=paths.root),
    )
    return _apply_dynamic_allowed_values(bundle)


def resolve_elasticsearch_config(app_config: YamlDict) -> Dict[str, Any]:
    db_cfg = app_config.get("database", {})
    return {
        "host": db_cfg.get("host", "localhost"),
        "port": int(db_cfg.get("port", 9200)),
        "index": db_cfg.get("index", "televic.history.call"),
    }


def get_data_sources(app_config: YamlDict) -> YamlDict:
    sources = app_config.get("data_sources", {})
    return sources if isinstance(sources, dict) else {}


def get_data_source_by_id(app_config: YamlDict, source_id: str) -> YamlDict:
    source = get_data_sources(app_config).get(source_id, {})
    return source if isinstance(source, dict) else {}


def get_data_source_by_table(app_config: YamlDict, canonical_table: str) -> YamlDict:
    for source in get_data_sources(app_config).values():
        if isinstance(source, dict) and source.get("canonical_table") == canonical_table:
            return source
    return {}


def get_allowed_tables(app_config: YamlDict) -> list[str]:
    tables_cfg = app_config.get("tables", {})
    configured = tables_cfg.get("allowed_tables") if isinstance(tables_cfg, dict) else None
    if isinstance(configured, list):
        return [str(value) for value in configured if str(value).strip()]
    sources = get_data_sources(app_config)
    return [
        str(source.get("canonical_table"))
        for source in sources.values()
        if isinstance(source, dict) and source.get("canonical_table")
    ]


def resolve_data_source_config(app_config: YamlDict, *, canonical_table: Optional[str] = None, source_id: Optional[str] = None) -> YamlDict:
    source = get_data_source_by_id(app_config, source_id) if source_id else {}
    if not source and canonical_table:
        source = get_data_source_by_table(app_config, canonical_table)
    if source:
        return dict(source)

    es_cfg = resolve_elasticsearch_config(app_config)
    active_table = app_config.get("tables", {}).get("active_table", "historyCalls")
    return {
        "source_id": "calls",
        "canonical_table": active_table,
        "index": es_cfg["index"],
        "label": "call records",
        "date_field": "call_date",
        "timestamp_field": "timestamp_utc",
    }


def resolve_elasticsearch_config_for_table(app_config: YamlDict, canonical_table: str) -> Dict[str, Any]:
    es_cfg = resolve_elasticsearch_config(app_config)
    source_cfg = resolve_data_source_config(app_config, canonical_table=canonical_table)
    es_cfg["index"] = source_cfg.get("index", es_cfg["index"])
    es_cfg["canonical_table"] = source_cfg.get("canonical_table", canonical_table)
    es_cfg["source_id"] = source_cfg.get("source_id")
    return es_cfg


def get_runtime_defaults(app_config: YamlDict) -> YamlDict:
    defaults = app_config.get("defaults", {})
    runtime = app_config.get("runtime", {})

    quantile_interpolation = str(runtime.get("quantile_interpolation", "linear")).strip().lower()
    if quantile_interpolation not in {"linear", "nearest", "lower", "higher", "midpoint"}:
        quantile_interpolation = "linear"

    return {
        "scope_policy": defaults.get("scope_policy", "qa_ready"),
        "outlier_policy": defaults.get("outlier_policy", "exclude_negative"),
        "default_limit": int(runtime.get("default_limit", 10)),
        "max_result_rows": int(runtime.get("max_result_rows", 50)),
        "quantile_interpolation": quantile_interpolation,
    }


_TABLE_SOURCE_MAP: Dict[str, str] = {
    "historyCalls": "calls",
    "historyLocations": "locations",
    "historyWorkflows": "workflows",
}


def source_id_for_table(canonical_table: Any) -> Optional[str]:
    return _TABLE_SOURCE_MAP.get(str(canonical_table or ""))


def existing_config_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    paths = build_project_paths(root)
    found: Dict[str, Path] = {}
    for key in CONFIG_NAME_CANDIDATES:
        try:
            found[key] = resolve_config_path(key, config_dir=paths.config_dir, root=paths.root)
        except FileNotFoundError:
            continue
    return found


def iter_config_keys() -> Iterable[str]:
    return CONFIG_NAME_CANDIDATES.keys()
