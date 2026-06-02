from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
YamlDict = Dict[str, Any]

from historyAssistant.execution.result.display_labels import FIELD_LABELS as _FIELD_LABELS

_DATE_COLUMN_CANDIDATES = ["call_date", "workflow_date", "timestamp", "date"]
_ENCODED_PERSON_RE = re.compile(r"^[a-fA-F0-9]{32,}$")
_PERSON_LABEL_FIELDS = {
    "personFirstName",
    "person_name",
    "initiator_person",
}

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in _DATE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None

_BUCKET_X_LABELS: Dict[str, str] = {
    "1d": "Day",
    "7d": "Week",
    "1w": "Week",
    "14d": "2-week period",
    "1M": "Month",
    "1h": "Hour",
    "30m": "30 min",
    "15m": "15 min",
}

_BUCKET_TITLE_WORDS: Dict[str, str] = {
    "1d": "daily",
    "7d": "weekly",
    "1w": "weekly",
    "14d": "bi-weekly",
    "1M": "monthly",
    "1h": "hourly",
    "30m": "every 30 min",
    "15m": "every 15 min",
}


def _field_label(field: Optional[str]) -> str:
    if not field:
        return ""
    return _FIELD_LABELS.get(field, field.replace("_", " "))


def _build_title(
    chart_type: str,
    metric: Optional[str],
    x_col: Optional[str],
    series_col: Optional[str],
    bucket_interval: Optional[str],
    bucket_summary: Optional[str],
) -> str:
    metric_label = _field_label(metric)
    x_label = _field_label(x_col)

    # timebucket charts
    if bucket_interval:
        bucket_word = _BUCKET_TITLE_WORDS.get(bucket_interval, bucket_interval)
        summary_word = (bucket_summary or "avg").lower()
        if summary_word == "avg":
            summary_word = "average"
        title = f"{metric_label} — {summary_word} per {bucket_word.replace('every ', '')}"
        if series_col:
            title += f" by {_field_label(series_col)}"
        return title

    # stacked / series
    if series_col and chart_type in {"stacked_bar", "bar", "line", "area"}:
        return f"{metric_label} by {x_label} and {_field_label(series_col)}"

    # standard grouped
    if x_label and chart_type in {"line", "area"}:
        return f"{metric_label} over {x_label}"

    if x_label:
        return f"{metric_label} by {x_label}"

    return metric_label or "Chart"


_PERIOD_COLUMN = "__period__"
_DAY_OFFSET_COLUMN = "__day_offset__"
_COMPARISON_META_COLUMNS = {"metric_value", _PERIOD_COLUMN, _DAY_OFFSET_COLUMN}


def _detect_grouped_comparison_x(
    chart_payload: YamlDict,
    main_side: pd.DataFrame,
    comp_side: pd.DataFrame,
) -> Optional[str]:
    chart_spec = chart_payload.get("chart_spec") or {}
    configured_x = chart_spec.get("x")
    if isinstance(configured_x, str) and configured_x in main_side.columns and configured_x in comp_side.columns:
        return configured_x

    common_cols = [col for col in main_side.columns if col in comp_side.columns]
    for col in common_cols:
        if col not in _COMPARISON_META_COLUMNS and not _is_date_like_column(col):
            return col
    return None


def _is_date_like_column(column: str) -> bool:
    return column in _DATE_COLUMN_CANDIDATES or column.endswith("_date") or column.endswith("_timestamp")


def _prepare_comparison_overlay(
    chart_payload: YamlDict,
    main_df: pd.DataFrame,
    bucket_interval: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
   
    comp_rows = chart_payload.get("comparison_rows")
    if not comp_rows:
        return main_df, None, None, None, None

    main_label = str(chart_payload.get("main_period_label") or "Main")
    comp_label = str(chart_payload.get("comparison_period_label") or "Comparison")
    is_date_swap = bool(chart_payload.get("comparison_is_date_swap"))

    main_side = main_df.copy() if main_df is not None else pd.DataFrame()
    comp_side = pd.DataFrame(comp_rows)
    if main_side.empty and comp_side.empty:
        return main_df, None, None, None, None

    main_side[_PERIOD_COLUMN] = main_label
    comp_side[_PERIOD_COLUMN] = comp_label
    combined = pd.concat([main_side, comp_side], ignore_index=True, sort=False)

    date_col = _detect_date_column(combined)
    metric_name = chart_payload.get("metric")
    metric_label = _field_label(metric_name)

    grouped_x = _detect_grouped_comparison_x(chart_payload, main_side, comp_side)
    if grouped_x:
        x_label = _field_label(grouped_x)
        title = f"{metric_label} by {x_label}: {main_label} vs {comp_label}".strip(" :,")
        return combined, grouped_x, _PERIOD_COLUMN, "bar", title

    if bucket_interval and date_col and is_date_swap:
        # Align by ordinal position within each period so periods overlay.
        combined[_DAY_OFFSET_COLUMN] = (
            combined.groupby(_PERIOD_COLUMN)[date_col]
            .rank(method="dense")
            .astype(int)
        )
        bucket_word = _BUCKET_TITLE_WORDS.get(bucket_interval, bucket_interval or "")
        title = f"{metric_label} — {bucket_word}, {main_label} vs {comp_label}".strip(" —,")
        return combined, _DAY_OFFSET_COLUMN, _PERIOD_COLUMN, "line", title

    if bucket_interval and date_col:
        # Entity swap: same dates on both sides — use the real date column.
        bucket_word = _BUCKET_TITLE_WORDS.get(bucket_interval, bucket_interval or "")
        title = f"{metric_label} — {bucket_word}, {main_label} vs {comp_label}".strip(" —,")
        return combined, date_col, _PERIOD_COLUMN, "line", title

    # Scalar comparison — two grouped bars.
    title = f"{metric_label}: {main_label} vs {comp_label}".strip(" :,")
    return combined, _PERIOD_COLUMN, _PERIOD_COLUMN, "bar", title


def render_chart(chart_payload: Optional[YamlDict]) -> Optional[Any]:
    if not chart_payload:
        return None
    try:
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not installed — cannot render chart")
        return None

    chart_spec = chart_payload.get("chart_spec") or {}
    rows = chart_payload.get("rows") or []
    if not rows and not chart_payload.get("comparison_rows"):
        return None
    if not chart_spec:
        return None

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty and not chart_payload.get("comparison_rows"):
        return None

    chart_type = chart_spec.get("chart_type", "bar")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    series_col = chart_spec.get("series")

    metric_name = chart_payload.get("metric")
    bucket_interval: Optional[str] = chart_payload.get("bucket_interval")
    bucket_summary: Optional[str] = chart_payload.get("bucket_summary")

    # If a comparison side is present, merge both sides and override x/series/type.
    df, comp_x, comp_series, comp_chart_type, comp_title = _prepare_comparison_overlay(
        chart_payload, df, bucket_interval,
    )
    if comp_x is not None:
        x_col = comp_x
        series_col = comp_series
        chart_type = comp_chart_type or "bar"
    elif chart_type == "bar" and series_col:
        # auto-upgrade bar → stacked_bar when a series dimension is present
        chart_type = "stacked_bar"

    # y column in execution records is use metric_value
    if "metric_value" in df.columns:
        metric_aliases = {name for name in (metric_name, x_col, y_col) if isinstance(name, str)}
        for alias in metric_aliases:
            if alias not in df.columns:
                df[alias] = df["metric_value"]

    x_col = _resolve_chart_x_column(df, x_col, bucket_interval, series_col)
    y_col = _resolve_chart_y_column(df, y_col, metric_name)

    if x_col and x_col not in df.columns:
        detected = _detect_date_column(df)
        if detected:
            logger.debug("chart x '%s' not in df; falling back to '%s'", x_col, detected)
            x_col = detected

    for axis_name, column in (("x", x_col), ("y", y_col), ("series", series_col)):
        if column and column not in df.columns:
            logger.warning("%s column '%s' not in dataframe columns %s", axis_name, column, list(df.columns))
            return None

    df = sort_for_chart(df, chart_type, x_col, series_col)
    title_x_col = x_col
    title_series_col = series_col
    df, x_col, series_col, labels = _apply_chart_display_labels(
        df,
        x_col=x_col,
        series_col=series_col,
        labels={},
    )

    #  x column is call_date bucketed
    x_label_override: Optional[str] = None
    if bucket_interval:
        x_label_override = _BUCKET_X_LABELS.get(bucket_interval, bucket_interval)

    title = (
        comp_title
        if comp_title is not None
        else _build_title(chart_type, metric_name, title_x_col, title_series_col, bucket_interval, bucket_summary)
    )

    base_labels = _build_label_map(x_col, y_col, series_col, metric_name, x_label_override)
    base_labels.update(labels)
    labels = base_labels
    # Don't show the synthetic "__period__" column name on the axis.
    if _PERIOD_COLUMN in labels and labels[_PERIOD_COLUMN] in {_PERIOD_COLUMN, "  period  "}:
        labels[_PERIOD_COLUMN] = ""
    if x_col == _PERIOD_COLUMN:
        labels[_PERIOD_COLUMN] = ""

    try:
        fig = build_figure(df, chart_type, x_col, y_col, series_col, title, labels, px)
        if fig is None:
            return None
        apply_layout(fig, title, has_series=bool(series_col))
        return fig
    except Exception as exc:
        logger.warning("Chart rendering failed: %s", exc)
        return None


def _resolve_chart_x_column(
    df: pd.DataFrame,
    x_col: Optional[str],
    bucket_interval: Optional[str],
    series_col: Optional[str],
) -> Optional[str]:
    if x_col and x_col in df.columns:
        return x_col

    if bucket_interval:
        detected = _detect_date_column(df)
        if detected:
            logger.debug("chart x missing/invalid; using time bucket column '%s'", detected)
            return detected

    preferred_time_cols = [
        *_DATE_COLUMN_CANDIDATES,
        "call_date_day",
        "call_date_week",
        "call_date_month",
        "call_date_year",
        "hour_bucket",
    ]
    for candidate in preferred_time_cols:
        if candidate in df.columns:
            logger.debug("chart x missing/invalid; using temporal column '%s'", candidate)
            return candidate

    non_metric_cols = [
        col for col in df.columns
        if col not in {"metric_value", series_col}
    ]
    if len(non_metric_cols) == 1:
        logger.debug("chart x missing/invalid; using sole non-metric column '%s'", non_metric_cols[0])
        return non_metric_cols[0]
    if non_metric_cols:
        logger.debug("chart x missing/invalid; using first non-metric column '%s'", non_metric_cols[0])
        return non_metric_cols[0]
    return x_col


def _resolve_chart_y_column(
    df: pd.DataFrame,
    y_col: Optional[str],
    metric_name: Optional[str],
) -> Optional[str]:
    if y_col and y_col in df.columns:
        return y_col
    if metric_name and metric_name in df.columns:
        return metric_name
    if "metric_value" in df.columns:
        return "metric_value"
    return y_col


def _build_label_map(
    x_col: Optional[str],
    y_col: Optional[str],
    series_col: Optional[str],
    metric_name: Optional[str],
    x_label_override: Optional[str],
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if x_col:
        labels[x_col] = x_label_override if x_label_override else _field_label(x_col)
    if y_col:
        labels[y_col] = _field_label(y_col)
    if series_col:
        labels[series_col] = _field_label(series_col)
    if metric_name and metric_name not in labels:
        labels[metric_name] = _field_label(metric_name)
    if "metric_value" not in labels:
        labels["metric_value"] = _field_label(metric_name) if metric_name else "Value"
    return labels


def _shorten_encoded_person_label(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if _ENCODED_PERSON_RE.fullmatch(text):
        return text[:5]
    return value


def _apply_chart_display_labels(
    df: pd.DataFrame,
    *,
    x_col: Optional[str],
    series_col: Optional[str],
    labels: Dict[str, str],
) -> tuple[pd.DataFrame, Optional[str], Optional[str], Dict[str, str]]:
    
    display_df = df
    updated_labels = dict(labels)

    def apply_column(column: Optional[str]) -> Optional[str]:
        nonlocal display_df
        if not column or column not in display_df.columns:
            return column
        if column not in _PERSON_LABEL_FIELDS:
            return column
        display_col = f"{column}__chart_label"
        display_df = display_df.copy()
        display_df[display_col] = display_df[column].map(_shorten_encoded_person_label)
        updated_labels[display_col] = _field_label(column)
        return display_col

    updated_x_col = apply_column(x_col)
    updated_series_col = apply_column(series_col)
    return display_df, updated_x_col, updated_series_col, updated_labels


def sort_for_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str],
    series_col: Optional[str],
) -> pd.DataFrame:
    if df.empty or not x_col or x_col not in df.columns:
        return df
    if chart_type not in {"line", "area", "scatter", "dot_plot", "bar", "stacked_bar", "histogram"}:
        return df

    sorted_df = df.copy()
    sort_col = "__chart_x_sort"
    if x_col in _PERSON_LABEL_FIELDS:
        sorted_df[sort_col] = sorted_df[x_col].astype(str)
    else:
        parsed_x = pd.to_datetime(sorted_df[x_col], errors="coerce")
        if parsed_x.notna().any():
            sorted_df[sort_col] = parsed_x
        else:
            numeric_x = pd.to_numeric(sorted_df[x_col], errors="coerce")
            if numeric_x.notna().any():
                sorted_df[sort_col] = numeric_x
            else:
                sorted_df[sort_col] = sorted_df[x_col].astype(str)

    sort_columns = [sort_col]
    if series_col and series_col in sorted_df.columns:
        sort_columns.insert(0, series_col)
    sorted_df = sorted_df.sort_values(sort_columns, kind="mergesort").drop(columns=[sort_col])
    return sorted_df


def build_figure(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str],
    y: Optional[str],
    series: Optional[str],
    title: str,
    labels: Dict[str, str],
    px: Any,
) -> Optional[Any]:
    kw: Dict[str, Any] = {"title": title, "labels": labels}

    if chart_type == "bar":
        return px.bar(df, x=x, y=y, color=series, barmode="group", **kw)

    if chart_type == "horizontal_bar":
        return px.bar(df, x=x, y=y, color=series, orientation="h", **kw)

    if chart_type == "stacked_bar":
        return px.bar(df, x=x, y=y, color=series, barmode="stack", **kw)

    if chart_type == "line":
        return px.line(df, x=x, y=y, color=series, **kw)

    if chart_type == "area":
        return px.area(df, x=x, y=y, color=series, **kw)

    if chart_type == "pie":
        return px.pie(df, names=x, values=y, **kw)

    if chart_type == "scatter":
        return px.scatter(df, x=x, y=y, color=series, **kw)

    if chart_type == "histogram":
        return px.bar(df, x=x, y=y, **kw)

    if chart_type == "dot_plot":
        return px.scatter(df, x=x, y=y, color=series, **kw)

    logger.warning("Unknown chart_type '%s', falling back to bar", chart_type)
    return px.bar(df, x=x, y=y, color=series, barmode="group", **kw)


def apply_layout(fig: Any, title: str, *, has_series: bool = False) -> None:
    # Right-side vertical legend keeps the title row clean and never overlaps it,
    # even on narrow viewports — Plotly auto-wraps long legends down the side.
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        title=dict(x=0.5, xanchor="center", font=dict(size=14)),
        showlegend=has_series,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ) if has_series else {},
    )
