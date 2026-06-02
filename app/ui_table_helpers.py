from __future__ import annotations

from html import escape as _escape
from pathlib import Path
from urllib.parse import quote as _url_quote

from historyAssistant.execution.result.display_labels import FIELD_LABELS as _DISPLAY_COLUMN_NAMES


_HIDDEN_TABLE_COLUMNS: frozenset[str] = frozenset({
    "qa_ready_flag",
    "timestamp_utc",
    "cancel_timestamp_utc",
    "cancellation_trigger",
    "triggered_by",
    "call_name",
    "cancel_person",
    "person_location",
    "cancel_location",
})

_TIME_SORT_COLUMNS: tuple[str, ...] = ("call_date", "call_dayofweek", "call_hour")

_METRIC_SORT_COLUMNS: frozenset[str] = frozenset({
    "metric_value", "total_calls", "avg_response_time", "median_response_time",
    "avg_staying_time", "median_staying_time", "p75_response_time", "p90_response_time",
    "p90_staying_time", "p95_staying_time", "response_within_2_min_rate",
    "response_within_3_min_rate", "response_over_5_min_rate", "avg_calls_per_shift",
})


def prepare_display_df(df):
    import pandas as pd

    df = df.copy()
    drop_cols = [c for c in df.columns if c in _HIDDEN_TABLE_COLUMNS]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    sort_col = next((c for c in _TIME_SORT_COLUMNS if c in df.columns), None)
    if sort_col is None:
        sort_col = next((c for c in _METRIC_SORT_COLUMNS if c in df.columns), None)
    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=True, kind="mergesort").reset_index(drop=True)

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(2)

    return df.rename(columns=lambda c: _DISPLAY_COLUMN_NAMES.get(c, c.replace("_", " ").capitalize()))


def build_inline_chat_table_html(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<div class='chat-inline-table-empty'>No rows to preview.</div>"

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    header_html = "".join(f"<th>{_escape(str(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cell_html = "".join(
            f"<td>{_escape('' if row.get(col) is None else str(row.get(col)))}</td>"
            for col in columns
        )
        body_rows.append(f"<tr>{cell_html}</tr>")

    table_html = "".join(body_rows)
    return (
        "<div class='chat-inline-table-wrap'>"
        "<table class='chat-inline-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{table_html}</tbody>"
        "</table>"
        "</div>"
    )


def build_inline_chat_table_from_dataframe(table) -> str:
    if table is None or getattr(table, "empty", True):
        return ""
    display = prepare_display_df(table).fillna("")
    rows = display.to_dict(orient="records")
    return build_inline_chat_table_html(rows)


def debug_has_raw_records(debug_payload: dict | None) -> bool:
    if not debug_payload:
        return False
    for item in debug_payload.get("execution", []) or []:
        result = item.get("execution_result", {}) if isinstance(item, dict) else {}
        if result.get("result_type") == "raw_records":
            return True
    return False


def should_show_inline_chat_table(table, debug_payload: dict | None = None) -> bool:
    if table is None or getattr(table, "empty", True):
        return False
    if not debug_has_raw_records(debug_payload):
        return False
    columns_attr = getattr(table, "columns", [])
    columns = list(columns_attr)
    if len(table) == 1 and set(columns).issubset({"metric_value", "scalar_answer"}):
        return False
    return True


def build_chat_download_link(file_path: str | None) -> str:
    if not file_path:
        return ""
    resolved_path = str(Path(file_path).resolve()).replace("\\", "/")
    filename = Path(file_path).name
    href = f"/gradio_api/file={_url_quote(resolved_path, safe='/:')}"
    return (
        "<div class='chat-download-wrap'>"
        f"<a class='chat-download-link' href='{_escape(href, quote=True)}' "
        f"download='{_escape(filename, quote=True)}'>"
        "Download all matching rows"
        "</a>"
        "</div>"
    )


def table_has_rows(table) -> bool:
    if table is None:
        return False

    empty = getattr(table, "empty", None)
    if isinstance(empty, bool):
        return not empty

    try:
        return len(table) > 0
    except Exception:
        return False


def make_result_table_preview(table, max_rows: int = 50):
    if not table_has_rows(table):
        return None

    head = getattr(table, "head", None)
    preview = table.head(max_rows) if callable(head) else table
    return prepare_display_df(preview)


def should_show_result_panel_table(table, debug_payload: dict | None = None) -> bool:
    if not table_has_rows(table):
        return False

    if debug_has_raw_records(debug_payload):
        return False

    if debug_payload:
        for item in debug_payload.get("execution", []) or []:
            result = item.get("execution_result", {}) if isinstance(item, dict) else {}
            if result.get("chart_payload"):
                return False

    columns_attr = getattr(table, "columns", [])
    columns = list(columns_attr)

    try:
        row_count = len(table)
    except Exception:
        row_count = 0

    if row_count == 1 and set(columns).issubset({"metric_value", "scalar_answer"}):
        return False

    return True
