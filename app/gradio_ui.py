from __future__ import annotations

import ast
from pathlib import Path

import gradio as gr

from historyAssistant.app.chatbot_service import handle_chat_turn_stream, handle_show_source_records
from historyAssistant.app.conversation_state import update_conversation_state_from_debug
from historyAssistant.app.ui_table_helpers import (
    build_chat_download_link,
    build_inline_chat_table_from_dataframe,
    make_result_table_preview,
    should_show_inline_chat_table,
    should_show_result_panel_table,
    table_has_rows,
)
from historyAssistant.execution.chart import render_chart


def load_app_css() -> str:
    CSS_PATH = Path(__file__).resolve().parent / "assets" / "gradio_ui.css"
    return CSS_PATH.read_text(encoding="utf-8")


def drop_empty(value):
    if isinstance(value, dict):
        cleaned = {k: drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if not is_empty_value(v)}
    if isinstance(value, list):
        cleaned = [drop_empty(v) for v in value]
        return [v for v in cleaned if not is_empty_value(v)]
    return value


def is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size == 0
    return False


def compact_planning_entry(entry: dict) -> dict:
    validation = entry.get("validation", {}) or {}
    plan = validation.get("canonical_plan") or entry.get("canonical_plan") or {}
    compact = {
        "sub_request_id": entry.get("sub_request_id"),
        "scenario": entry.get("scenario"),
        "attempt": entry.get("attempt"),
        "deterministic": entry.get("deterministic"),
        "valid": validation.get("valid"),
        "errors": validation.get("errors"),
        "warnings": validation.get("warnings"),
        "blocking_errors": validation.get("blocking_errors"),
        "repairs": validation.get("repairs"),
        "chart_auto": entry.get("chart_auto"),
        "chart_selection": entry.get("chart_selection"),
        "chart_validation": validation.get("chart_validation"),
        "plan": {
            "request_type": plan.get("request_type"),
            "target_tables": plan.get("target_tables"),
            "chart_intent_source": plan.get("_chart_intent_source"),
            "metric": plan.get("metric"),
            "aggregation": plan.get("aggregation"),
            "filters": plan.get("filters"),
            "group_by": plan.get("group_by"),
            "ranking_order": plan.get("ranking_order"),
            "limit": plan.get("limit"),
            "date_rule": plan.get("date_rule"),
            "date_range": plan.get("date_range"),
            "time_range": plan.get("time_range"),
            "compare_to": plan.get("compare_to"),
            "chart_spec": plan.get("chart_spec"),
        },
    }
    return drop_empty(compact)


def _compact_question_type_label(*, result_type: str | None, bucket_interval: str | None, compare_to, group_by: list | None) -> str:
    if result_type == "raw_records":
        return "simple lookup"
    if compare_to:
        return "comparison"
    if bucket_interval:
        return "time trend"
    if group_by:
        return "aggregation / ranking"
    return "single metric"


def _compact_plan_fields(plan: dict | None) -> dict | None:
    plan = plan or {}
    compact = {
        "request_type": plan.get("request_type"),
        "target_tables": plan.get("target_tables"),
        "metric": plan.get("metric"),
        "aggregation": plan.get("aggregation"),
        "group_by": plan.get("group_by"),
        "filters": plan.get("filters"),
        "ranking_order": plan.get("ranking_order"),
        "limit": plan.get("limit"),
        "date_rule": plan.get("date_rule"),
        "date_range": plan.get("date_range"),
        "time_range": plan.get("time_range"),
        "compare_to": plan.get("compare_to"),
        "bucket_interval": plan.get("bucket_interval"),
        "bucket_summary": plan.get("bucket_summary"),
        "chart_spec": plan.get("chart_spec"),
    }
    return drop_empty(compact)


def _sample_records(records: list | None, max_rows: int = 4) -> list:
    rows = records or []
    if not isinstance(rows, list):
        return []
    return rows[:max_rows]


def _parse_query_debug(query_debug):
    if isinstance(query_debug, dict):
        return query_debug
    if not isinstance(query_debug, str) or not query_debug.strip():
        return None
    try:
        return ast.literal_eval(query_debug)
    except (SyntaxError, ValueError):
        return query_debug


def _compact_parsed_plan(parsed_plan: dict | None) -> dict | None:
    if not isinstance(parsed_plan, dict):
        return parsed_plan
    return drop_empty({
        "request_type": parsed_plan.get("request_type"),
        "metric": parsed_plan.get("metric"),
        "filters": parsed_plan.get("filters"),
        "group_by": parsed_plan.get("group_by"),
        "ranking_order": parsed_plan.get("ranking_order"),
        "limit": parsed_plan.get("limit"),
        "date_rule": parsed_plan.get("date_rule"),
        "date_range": parsed_plan.get("date_range"),
        "compare_to": parsed_plan.get("compare_to"),
        "bucket_interval": parsed_plan.get("bucket_interval"),
        "bucket_summary": parsed_plan.get("bucket_summary"),
    })


def _compact_followup_resolution(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    return drop_empty({
        "resolved": value.get("resolved"),
        "strategy": value.get("strategy"),
        "plan_patch": _compact_parsed_plan(value.get("plan_patch")),
        "needs_clarification": value.get("needs_clarification"),
    })


def _compact_detected_entities(entities: dict | None) -> dict | None:
    if not isinstance(entities, dict):
        return None
    compact: dict = {}
    for key, value in entities.items():
        if isinstance(value, list):
            compact[key] = value[:3]
        else:
            compact[key] = value
    return drop_empty(compact)


def _scenario_relevant_plan_view(plan: dict | None, *, question_type: str) -> dict | None:
    plan = plan or {}
    base = {
        "request_type": plan.get("request_type"),
        "target_tables": plan.get("target_tables"),
        "metric": plan.get("metric"),
        "date_rule": plan.get("date_rule"),
        "date_range": plan.get("date_range"),
        "time_range": plan.get("time_range"),
        "filters": plan.get("filters"),
    }

    if question_type == "simple lookup":
        base["source_fields"] = plan.get("source_fields")
    else:
        base["aggregation"] = plan.get("aggregation")
        base["group_by"] = plan.get("group_by")

    if question_type in {"aggregation / ranking", "comparison"}:
        base["ranking_order"] = plan.get("ranking_order")
        base["limit"] = plan.get("limit")

    if question_type == "comparison":
        base["compare_to"] = plan.get("compare_to")

    if question_type == "time trend":
        base["bucket_interval"] = plan.get("bucket_interval")
        base["bucket_summary"] = plan.get("bucket_summary")

    if plan.get("request_type") == "chart" or plan.get("chart_spec"):
        base["chart_spec"] = plan.get("chart_spec")

    return drop_empty(base)


def _same_as_previous_note(previous: dict | None, current: dict | None) -> str | None:
    if not previous or not current:
        return None
    if drop_empty(previous) == drop_empty(current):
        return "Same as previous step."
    return None


def compact_chart_payload(chart_payload: dict | None) -> dict | None:
    if not chart_payload:
        return None

    rows = chart_payload.get("rows") or []
    compact = {
        "metric": chart_payload.get("metric"),
        "group_by": chart_payload.get("group_by"),
        "row_count": len(rows),
        "chart_spec": chart_payload.get("chart_spec"),
        "sample_rows": rows[:3],
    }
    return drop_empty(compact)


def compact_execution_item(item: dict) -> dict:
    result = item.get("execution_result", {}) or {}
    records = result.get("records") or []
    compact = {
        "sub_request_id": item.get("sub_request_id"),
        "ok": result.get("ok"),
        "empty": result.get("empty"),
        "data_source": result.get("data_source"),
        "result_type": result.get("result_type"),
        "record_count": len(records),
        "scalar_answer": result.get("scalar_answer"),
        "has_chart": bool(result.get("chart_payload")),
        "where_debug": result.get("where_debug"),
        "overfetch_debug": result.get("overfetch_debug"),
        "chart_result": compact_chart_payload(result.get("chart_payload")),
        "error": result.get("error"),
        "narration": item.get("narration"),
    }
    return drop_empty(compact)


def format_debug_payload_compact(debug_payload: dict | None) -> dict | None:
    if not debug_payload:
        return debug_payload

    controls = debug_payload.get("controls", {}) or {}
    normalization = debug_payload.get("normalization", {}) or {}
    selected_source = debug_payload.get("selected_source", {}) or {}
    routing = debug_payload.get("scenario_routing", {}) or {}
    hints = debug_payload.get("planning_hints", {}) or {}
    guardrails = debug_payload.get("guardrails", {}) or {}

    compact = {
        "question": debug_payload.get("question"),
        "controls": {k: v for k, v in controls.items() if v not in ("", "No export", None)},
        "classification_timing": debug_payload.get("classification_timing"),
        "guardrails": {
            "blocked": guardrails.get("blocked"),
            "hits": guardrails.get("hits"),
        },
        "normalization": {
            "normalized": normalization.get("normalized"),
            "detected_entities": normalization.get("detected_entities"),
        },
        "scenario": {
            "name": routing.get("scenario"),
            "confidence": routing.get("confidence"),
        },
        "selected_source": {
            "source_id": selected_source.get("source_id"),
            "source_ids": selected_source.get("source_ids"),
            "canonical_table": selected_source.get("canonical_table"),
            "canonical_tables": selected_source.get("canonical_tables"),
            "es_index": selected_source.get("es_index"),
            "source_confidence": selected_source.get("source_confidence"),
        },
        "planning_hints": {
            "metric_signals": hints.get("metric_signals"),
            "group_by_signals": hints.get("group_by_signals"),
            "ranking": hints.get("ranking"),
            "detected_dates": hints.get("detected_dates"),
            "detected_filters": hints.get("detected_filters"),
            "detected_time_range": hints.get("detected_time_range"),
            "detected_limit": hints.get("detected_limit"),
        },
        "followup_resolution": debug_payload.get("followup_resolution"),
        "clarification": debug_payload.get("clarification"),
        "deferred_clarification": debug_payload.get("deferred_clarification"),
        "planning": [compact_planning_entry(entry) for entry in debug_payload.get("planning", [])],
        "execution": [compact_execution_item(item) for item in debug_payload.get("execution", [])],
    }
    return drop_empty(compact)


def _first_item(items: list | None) -> dict:
    if not items:
        return {}
    first = items[0]
    return first if isinstance(first, dict) else {}


def split_debug_sections(debug_payload: dict | None) -> dict:
    
    if not debug_payload:
        return {"source_routing": None, "planner": None, "execute": None, "elasticsearch_query": None}

    _timing = debug_payload.get("_timing") or {}
    classification_timing = debug_payload.get("classification_timing") or {}
    normalization = debug_payload.get("source_normalization", {}) or debug_payload.get("normalization", {}) or {}
    selected_source = debug_payload.get("selected_source", {}) or {}
    guardrails = debug_payload.get("guardrails", {}) or {}
    planning_items = debug_payload.get("planning", []) or []
    execution_items = debug_payload.get("execution", []) or []
    first_planning = _first_item(planning_items)
    first_execution = _first_item(execution_items)
    execution_result = first_execution.get("execution_result", {}) or {}
    validation = first_planning.get("validation", {}) or {}
    canonical_plan = validation.get("canonical_plan") or first_planning.get("canonical_plan") or {}
    chart_payload = execution_result.get("chart_payload")
    result_type = execution_result.get("result_type")
    group_by = canonical_plan.get("group_by") or []
    bucket_interval = canonical_plan.get("bucket_interval")
    compare_to = canonical_plan.get("compare_to")
    question_type = _compact_question_type_label(
        result_type=result_type,
        bucket_interval=bucket_interval,
        compare_to=compare_to,
        group_by=group_by,
    )

    # SOURCE ROUTIN
    cls_steps = {
        s["step"]: s
        for s in (classification_timing.get("steps") or [])
    }
    scenario_step = cls_steps.get("scenario_routing", {})
    source_step = cls_steps.get("source_routing", {})
    vocab_step = cls_steps.get("vocabulary_normalization", {})

    source_routing = drop_empty({
        "_duration_ms": _timing.get("classification_ms"),
        "source_id": selected_source.get("source_id") or selected_source.get("router_table") or source_step.get("source_ids", [None])[0],
        "source_confidence": selected_source.get("source_confidence") or source_step.get("confidence"),
        "source_router": source_step.get("router"),
        "scenario": scenario_step.get("scenario") or debug_payload.get("scenario"),
        "scenario_confidence": scenario_step.get("confidence"),
        "scenario_router": scenario_step.get("router"),
        "normalized_question": normalization.get("normalized") or normalization.get("normalized_question"),
        "detected_entities": _compact_detected_entities(
            normalization.get("detected_entities") or vocab_step.get("detected_entities")
        ),
        "in_scope": not bool(guardrails.get("blocked")),
        "followup": _compact_followup_resolution(debug_payload.get("followup_resolution")),
    })

    # ── PLANNER ────────────────────────────────────────────────────────────────
    planning_path = (debug_payload.get("planning_path") or
                     debug_payload.get("scenario_subtype", {}).get("planning_path") or
                     "unknown")
    slot_debug = debug_payload.get("slot_planner") or {}

    # figure out which planner was actually used
    if slot_debug.get("used"):
        planner_path_label = "intent_planner"
        extraction_path = (slot_debug.get("slot_extraction") or {}).get("extraction_path", "deterministic_slots")
        slot_reason = slot_debug.get("reason", "")
        llm_slots_used = bool((slot_debug.get("llm_slot_extraction") or {}).get("used"))
        intent_extraction = drop_empty({
            "path": ("llm_slots" if llm_slots_used else "deterministic"),
            "reason": slot_reason,
            "extraction_path": extraction_path,
        })
    else:
        planner_path_label = planning_path
        intent_extraction = drop_empty({
            "path": "not_planned",
            "reason": first_planning.get("fast_path_reason") or (slot_debug.get("reason") if slot_debug else None),
        })

    compact_plan = _scenario_relevant_plan_view(canonical_plan, question_type=question_type)
    validation_summary = drop_empty({
        "valid": validation.get("valid"),
        "repairs": validation.get("repairs"),
        "errors": validation.get("errors"),
        "blocking_errors": validation.get("blocking_errors"),
    })

    planner = drop_empty({
        "_duration_ms": _timing.get("planning_ms"),
        "planning_path": planner_path_label,
        "question_type": question_type,
        "intent_extraction": intent_extraction,
        "canonical_plan": compact_plan,
        "validation": validation_summary,
    })
    execution_records = execution_result.get("records") or []
    chart_spec = (chart_payload or {}).get("chart_spec") or (canonical_plan.get("chart_spec") if isinstance(canonical_plan, dict) else None)
    execute = drop_empty({
        "_duration_ms": _timing.get("execution_ms"),
        "status": "ok" if execution_result.get("ok") else ("error" if execution_result.get("error") else None),
        "error": execution_result.get("error"),
        "data_source": execution_result.get("data_source"),
        "result_type": result_type,
        "record_count": len(execution_records),
        "scalar_answer": execution_result.get("scalar_answer"),
        "sample_records": _sample_records(execution_records, max_rows=4),
        "chart": drop_empty({
            "chart_spec": chart_spec,
            "row_count": len((chart_payload or {}).get("rows") or []),
        }) if chart_payload or chart_spec else None,
        "narration_mode": (first_execution.get("narration_debug") or {}).get("mode"),
        "narration_ms": _timing.get("narration_ms"),
    })
    elasticsearch_query = {
        "index": (execution_result.get("data_source") or {}).get("es_index"),
        "body": _parse_query_debug(execution_result.get("query")),
        "where_debug": execution_result.get("where_debug"),
        "overfetch_debug": execution_result.get("overfetch_debug"),
    }
    elasticsearch_query = {
        key: value
        for key, value in elasticsearch_query.items()
        if not is_empty_value(value)
    }

    return {
        "source_routing": source_routing or None,
        "planner": planner or None,
        "execute": execute or None,
        "elasticsearch_query": elasticsearch_query or None,
    }

SHOW_RESULTS_PANEL_JS = """
(details) => {
    return details;
}
"""

HIDE_RESULTS_PANEL_JS = """
() => {
}
"""

HIDE_ADVANCED_KEEP_RESULTS_JS = """
(details) => {
    return details;
}
"""

HIDE_RESULTS_ON_SUBMIT_JS = """
(message) => {
    return message;
}
"""

PIPELINE_DIAGRAM_HTML = """
<div style="font-family:inherit;padding:12px 8px 8px">
  <div style="display:flex;align-items:stretch;gap:0">

    <!-- Source Routing -->
    <div style="flex:1;border:2px solid #94a3b8;border-radius:10px;padding:10px 8px;background:#f8fafc">
      <div style="font-size:0.72rem;font-weight:700;color:#64748b;text-align:center;margin-bottom:8px;letter-spacing:.04em">SOURCE ROUTING</div>
      <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
        <div style="background:#dbeafe;border:1.5px solid #93c5fd;border-radius:999px;padding:4px 14px;font-size:0.75rem;font-weight:600;color:#1e40af">Question</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #cbd5e1;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Guardrails</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #cbd5e1;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Vocab Normalize</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #cbd5e1;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Source Route</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #cbd5e1;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Scenario Detect</div>
      </div>
    </div>

    <div style="display:flex;align-items:center;padding:0 6px;color:#94a3b8;font-size:1rem">→</div>

    <!-- Planner -->
    <div style="flex:1.4;border:2px solid #6366f1;border-radius:10px;padding:10px 8px;background:#f5f3ff">
      <div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-align:center;margin-bottom:8px;letter-spacing:.04em">PLANNER</div>
      <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
        <div style="background:#fff;border:1.5px solid #a5b4fc;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Intent Extraction</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="display:flex;gap:4px;width:100%">
          <div style="flex:1;background:#fff;border:1.5px solid #a5b4fc;border-radius:7px;padding:5px 4px;font-size:0.7rem;font-weight:600;color:#334155;text-align:center">Deterministic<br><span style="font-weight:400;color:#64748b">rules + config</span></div>
          <div style="flex:1;background:#fff;border:1.5px solid #a5b4fc;border-radius:7px;padding:5px 4px;font-size:0.7rem;font-weight:600;color:#334155;text-align:center">LLM Slots<br><span style="font-weight:400;color:#64748b">no query</span></div>
        </div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #a5b4fc;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Validate &amp; Repair</div>
      </div>
    </div>

    <div style="display:flex;align-items:center;padding:0 6px;color:#94a3b8;font-size:1rem">→</div>

    <!-- Execute -->
    <div style="flex:1;border:2px solid #22c55e;border-radius:10px;padding:10px 8px;background:#f0fdf4">
      <div style="font-size:0.72rem;font-weight:700;color:#16a34a;text-align:center;margin-bottom:8px;letter-spacing:.04em">EXECUTE</div>
      <div style="display:flex;flex-direction:column;align-items:center;gap:2px">
        <div style="background:#fff;border:1.5px solid #86efac;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">ES Query</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #86efac;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Aggregate / Fetch</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#fff;border:1.5px solid #86efac;border-radius:7px;padding:5px 10px;font-size:0.75rem;font-weight:600;color:#334155;text-align:center">Narrate &amp; Chart</div>
        <div style="color:#94a3b8;font-size:0.85rem">↓</div>
        <div style="background:#dcfce7;border:1.5px solid #86efac;border-radius:999px;padding:4px 14px;font-size:0.75rem;font-weight:600;color:#166534">Answer</div>
      </div>
    </div>

  </div>
</div>
"""

_STEP_KEYS = [
    "source_routing",
    "planner",
    "execute",
    "elasticsearch_query",
]
_STEP_LABELS = [
    "Source Routing",
    "Planner",
    "Execute Summary",
    "Elasticsearch Query",
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="HistoryAssistant Demo", css=load_app_css(), theme=gr.themes.Soft()) as demo:
        history_state = gr.State([])
        saved_msg = gr.State("")
        pending_state = gr.State(None)
        conversation_state = gr.State({})
        date_start = gr.State("")
        date_end = gr.State("")
        time_start = gr.State("")
        time_end = gr.State("")
        export_mode = gr.State("No export")
        show_debug = gr.State(True)
        latest_details_state = gr.State({})

        gr.Markdown("# HistoryAssistant Chat Demo", elem_classes=["app-shell"])

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=5, elem_classes=["chat-panel"]):
                gr.Markdown("### Chat", elem_classes=["panel-title"])
                chatbot = gr.Chatbot(
                    value=[],
                    height=420,
                    show_label=False,
                    placeholder="Ask about call counts, shifts, wards, priorities, or trends.",
                )
                with gr.Row(elem_classes=["detail-button-row"]):
                    result_details_button = gr.Button(
                        "Result details",
                        variant="secondary",
                        visible=False,
                    )
                    source_records_button = gr.Button(
                        "Show source records",
                        variant="secondary",
                        visible=False,
                    )
                with gr.Row(elem_classes=["input-action-row"]):
                    with gr.Column(scale=6):
                        message_box = gr.Textbox(
                            label="Question",
                            placeholder="Example: Which ward had the highest number of calls last week?",
                            lines=3,
                            elem_classes=["question-box"],
                        )
                    with gr.Column(scale=1, min_width=140, elem_classes=["action-col"]):
                        send_button = gr.Button("Send", variant="primary")
                        clear_button = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=3, elem_id="results-column", elem_classes=["control-panel"]):
                with gr.Group(elem_id="results-panel"):
                    gr.HTML("<div class='result-detail-heading'>Result details</div>")

                    # Chart and table shown immediately when available
                    chart_plot = gr.Plot(label="Chart", visible=False)
                    chart_empty_html = gr.HTML(visible=False)
                    result_table = gr.Dataframe(
                        label="Table", visible=False, interactive=False, wrap=True,
                    )

                    # Advanced details: accordions matching pipeline groups
                    _STEP_COLORS = ["#64748b", "#6366f1", "#16a34a", "#d97706"]
                    with gr.Accordion("Advanced details", open=False, visible=False) as advanced_details_box:
                        step_boxes = []
                        step_jsons = []
                        for label, color in zip(_STEP_LABELS, _STEP_COLORS):
                            gr.HTML(
                                f"<div style='font-size:0.7rem;font-weight:700;color:{color};"
                                f"letter-spacing:.06em;padding:6px 2px 2px;text-transform:uppercase'>"
                                f"{label}</div>"
                            )
                            with gr.Accordion(label, open=False) as box:
                                j = gr.JSON(show_label=False)
                            step_boxes.append(box)
                            step_jsons.append(j)

                    with gr.Row(elem_classes=["hide-detail-row"]):
                        with gr.Column(elem_classes=["hide-detail-button"]):
                            hide_results_button = gr.Button(
                                "Hide result details", size="sm", variant="secondary", visible=False,
                            )

        has_results = gr.State(False)

        def _build_details_state(*, debug_sections, plotly_fig, result_table_preview, show_debug_value):
            return {
                "plotly_fig": plotly_fig,
                "result_table": result_table_preview,
                "debug_sections": debug_sections or {},
                "show_debug": bool(show_debug_value),
            }

        def _detail_panel_updates(details_state, *, show_button: bool, show_advanced: bool):
            details_state = details_state or {}
            plotly_fig = details_state.get("plotly_fig")
            result_table_preview = details_state.get("result_table")
            debug_sections = details_state.get("debug_sections", {}) or {}
            show_debug_value = bool(details_state.get("show_debug", True))

            has_advanced = show_debug_value and any(debug_sections.get(k) for k in _STEP_KEYS)
            step_updates = [gr.update(value=debug_sections.get(k)) for k in _STEP_KEYS]

            return (
                gr.update(visible=show_button),
                gr.update(value=plotly_fig, visible=plotly_fig is not None),
                gr.update(value=result_table_preview, visible=table_has_rows(result_table_preview)),
                gr.update(visible=show_advanced and has_advanced, open=False),
                gr.update(visible=show_advanced and has_advanced),
                *step_updates,
            )

        def _render_outputs(chat, debug, table, dl_file, sd, current_conv_state):
            has_inline_table = should_show_inline_chat_table(table, debug)
            rendered_chat = list(chat or [])
            if has_inline_table and rendered_chat:
                inline_table_html = build_inline_chat_table_from_dataframe(table)
                download_link_html = build_chat_download_link(dl_file)
                if inline_table_html or download_link_html:
                    last_message = dict(rendered_chat[-1])
                    content = str(last_message.get("content", "") or "")
                    if "chat-inline-table-wrap" not in content:
                        content = f"{content}<br><br>{inline_table_html}"
                    if download_link_html and "chat-download-link" not in content:
                        content = f"{content}{download_link_html}"
                    last_message["content"] = content
                    rendered_chat[-1] = last_message

            chart_payload = None
            if debug and "execution" in debug:
                for exec_item in debug.get("execution", []):
                    exec_result = exec_item.get("execution_result", {})
                    candidate = exec_result.get("chart_payload")
                    if candidate:
                        chart_payload = candidate
                        break

            plotly_fig = render_chart(chart_payload) if chart_payload else None
            show_table_in_panel = should_show_result_panel_table(table, debug)
            result_table_preview = make_result_table_preview(table) if show_table_in_panel else None

            debug_sections = split_debug_sections(debug)
            has_advanced = bool(sd and any(debug_sections.get(k) for k in _STEP_KEYS))

            chart_was_requested = False
            if debug:
                chart_section = (split_debug_sections(debug) or {}).get("chart") or {}
                chart_was_requested = bool(chart_section.get("chart_requested"))
            show_empty_chart = chart_was_requested and plotly_fig is None
            empty_chart_html = (
                "<div style='padding:24px 16px;text-align:center;color:#64748b;font-size:0.9rem;'>"
                "No chart available for this result.</div>"
                if show_empty_chart else ""
            )

            details_state = _build_details_state(
                debug_sections=debug_sections,
                plotly_fig=plotly_fig,
                result_table_preview=result_table_preview,
                show_debug_value=sd,
            )
            step_updates = [gr.update(value=debug_sections.get(k)) for k in _STEP_KEYS]
            source_records_visible = bool((current_conv_state or {}).get("last_valid_plan"))
            return (
                rendered_chat,
                details_state,
                plotly_fig,
                empty_chart_html,
                result_table_preview,
                has_advanced,
                step_updates,
                source_records_visible,
            )

        def _wrap_stream_with_visibility(message, history, ds, de, ts, te, em, sd, pending, conv_state):
            current_conv_state = conv_state or {}
            for chat, hist, debug, table, dl_file, updated_pending in handle_chat_turn_stream(
                message, history, ds, de, ts, te, em, sd, pending, conversation_state=current_conv_state
            ):
                current_conv_state = update_conversation_state_from_debug(
                    current_conv_state, user_question=message, debug_payload=debug,
                )
                (
                    rendered_chat,
                    details_state,
                    plotly_fig,
                    empty_chart_html,
                    result_table_preview,
                    has_advanced,
                    step_updates,
                    source_records_visible,
                ) = _render_outputs(chat, debug, table, dl_file, sd, current_conv_state)

                yield (
                    rendered_chat,
                    rendered_chat,
                    gr.update(visible=False),
                    gr.update(visible=source_records_visible),
                    details_state,
                    gr.update(value=plotly_fig, visible=plotly_fig is not None),
                    gr.update(value=empty_chart_html, visible=bool(empty_chart_html)),
                    gr.update(value=result_table_preview, visible=table_has_rows(result_table_preview)),
                    gr.update(visible=has_advanced, open=False),
                    gr.update(visible=False),
                    *step_updates,
                    bool(rendered_chat),
                    updated_pending,
                    current_conv_state,
                )

        def _wrap_source_records_click(history, em, sd, conv_state):
            current_conv_state = conv_state or {}
            chat, hist, debug, table, dl_file, updated_pending = handle_show_source_records(
                history=history,
                conversation_state=current_conv_state,
                export_mode=em,
                show_debug=sd,
            )
            current_conv_state = update_conversation_state_from_debug(
                current_conv_state,
                user_question="Show source records",
                debug_payload=debug,
            )
            (
                rendered_chat,
                details_state,
                plotly_fig,
                empty_chart_html,
                result_table_preview,
                has_advanced,
                step_updates,
                source_records_visible,
            ) = _render_outputs(chat, debug, table, dl_file, sd, current_conv_state)
            return (
                rendered_chat,
                rendered_chat,
                gr.update(visible=False),
                gr.update(visible=source_records_visible),
                details_state,
                gr.update(value=plotly_fig, visible=plotly_fig is not None),
                gr.update(value=empty_chart_html, visible=bool(empty_chart_html)),
                gr.update(value=result_table_preview, visible=table_has_rows(result_table_preview)),
                gr.update(visible=has_advanced, open=False),
                gr.update(visible=False),
                *step_updates,
                bool(rendered_chat),
                updated_pending,
                current_conv_state,
            )

        _stream_outputs = [
            chatbot, history_state, result_details_button, source_records_button, latest_details_state,
            chart_plot, chart_empty_html, result_table,
            advanced_details_box, hide_results_button, *step_jsons,
            has_results, pending_state, conversation_state,
        ]

        send_button.click(
            fn=lambda msg: (msg, "", gr.update(visible=False), gr.update(visible=False), {}),
            inputs=[message_box],
            outputs=[saved_msg, message_box, result_details_button, source_records_button, latest_details_state],
            js=HIDE_RESULTS_ON_SUBMIT_JS,
        ).then(
            fn=_wrap_stream_with_visibility,
            inputs=[saved_msg, history_state, date_start, date_end, time_start, time_end, export_mode, show_debug, pending_state, conversation_state],
            outputs=_stream_outputs,
        )

        message_box.submit(
            fn=lambda msg: (msg, "", gr.update(visible=False), gr.update(visible=False), {}),
            inputs=[message_box],
            outputs=[saved_msg, message_box, result_details_button, source_records_button, latest_details_state],
            js=HIDE_RESULTS_ON_SUBMIT_JS,
        ).then(
            fn=_wrap_stream_with_visibility,
            inputs=[saved_msg, history_state, date_start, date_end, time_start, time_end, export_mode, show_debug, pending_state, conversation_state],
            outputs=_stream_outputs,
        )

        _detail_outputs = [
            result_details_button, chart_plot, result_table,
            advanced_details_box, hide_results_button, *step_jsons,
        ]

        result_details_button.click(
            fn=lambda details: _detail_panel_updates(details, show_button=False, show_advanced=True),
            inputs=[latest_details_state],
            outputs=_detail_outputs,
            js=SHOW_RESULTS_PANEL_JS,
        )

        source_records_button.click(
            fn=_wrap_source_records_click,
            inputs=[history_state, export_mode, show_debug, conversation_state],
            outputs=_stream_outputs,
        )

        hide_results_button.click(
            fn=lambda details: _detail_panel_updates(details, show_button=True, show_advanced=False),
            inputs=[latest_details_state],
            outputs=_detail_outputs,
            js=HIDE_ADVANCED_KEEP_RESULTS_JS,
        )

        clear_button.click(
            fn=lambda: (
                [], [],
                gr.update(visible=False),
                gr.update(visible=False),
                {},
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                *[gr.update(value=None) for _ in _STEP_KEYS],
                "", False, None, {},
            ),
            outputs=[
                chatbot, history_state, result_details_button, source_records_button, latest_details_state,
                chart_plot, chart_empty_html, result_table,
                advanced_details_box, hide_results_button, *step_jsons,
                message_box, has_results, pending_state, conversation_state,
            ],
            js=HIDE_RESULTS_PANEL_JS,
        )

    return demo
