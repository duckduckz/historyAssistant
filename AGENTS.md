# AGENTS.md

This file is the working guide for coding agents in this repository. Use it as the default operating manual before making changes.

The goal is simple: an agent should be able to add a feature, fix a bug, or investigate a wrong answer without re-learning the whole codebase from scratch.

## Project Summary

HistoryAssistant is a natural-language analytics assistant for Televic hospital history data.

Users ask questions in plain language. The system:

1. classifies the request for safety, vocabulary, source, and chart intent
2. resolves the source and answer shape inside the intent planner
3. builds or repairs a canonical plan
4. executes the plan against Elasticsearch
5. returns narration, tables, charts, and exports

The architecture is config-driven. Prefer changing config or the correct pipeline layer over adding one-off logic in the UI.

## Presentation Brief For Slide Generation

Use this section when asked to create a 15-20 minute research-project presentation about HistoryAssistant.

Recommended story:

1. **Context And Motivation**
   Explain the hospital-history analytics problem: operational history data exists across nurse-call events, location updates, and workflow/presence records, but non-technical users need to ask questions in plain language rather than manually query Elasticsearch or CSV files.
2. **Research Goal**
   Present the goal as building a safe, explainable, multi-source natural-language analytics assistant that converts a user question into a validated canonical plan, executes against Elasticsearch, and returns narration, tables, charts, and exports.
3. **Data Sources**
   Show the three active sources:
   - calls: nurse-call event history
   - locations: patient/staff location update history
   - workflows: workflow and presence event history
   Mention that each source has its own canonical table, Elasticsearch index, date/timestamp fields, metrics, dimensions, and vocabulary config.
4. **Architecture Overview**
   Use one pipeline diagram:
   `User question -> classification/source routing -> intent planner -> canonical plan -> validator -> query builder -> execution strategies -> Elasticsearch -> result formatter -> narration/chart/table/export`.
5. **Planning Approach**
   Explain that old scenario routing was removed. The current planner uses answer shape (`raw_records`, `distinct_values`, `analytic_summary`), deterministic extraction when reliable, and one unified LLM slot call when deterministic slots are weak. The LLM fills structured slots only; it does not write Elasticsearch.
6. **Configuration-Driven Design**
   Emphasize that metrics, dimensions, date rules, table metadata, chart rules, source metadata, vocabulary, allowed fields, and numeric measure fields live in YAML config. Code validates and executes these structures instead of hardcoding every question pattern.
7. **Execution Approach**
   Explain the canonical plan to `QueryPlan` adapter, the Elasticsearch query builder, strategy registry, generic numeric summaries, raw-record strategies, time buckets, comparisons, entity intersections, and result formatting.
8. **User Experience**
   Show the Gradio chat UI with streaming stages, debug panel, chart/table output, export support, clarification flow, follow-up handling, deterministic raw-record narration, and LLM-assisted narration for richer non-raw answers.
9. **Representative Demo Questions**
   Pick 4-6 examples that cover the system:
   - `how many urgent calls in February 2024`
   - `urgent calls by ward in February 2024`
   - `who was present on 2024-02-01`
   - `last known location of staff <hash>`
   - `who moved most last week`
   - `maximum activation time by ward in February 2024`
10. **Evaluation And Debugging**
    Show how the debug payload exposes selected source, normalized question, answer shape, slot extraction path, canonical plan, validation, query execution, timings, and narration mode. Mention focused unit tests and targeted question debugging.
11. **Results And Lessons Learned**
    Highlight the main research result: the project moved from brittle scenario-specific routing toward a modular, inspectable, config-driven assistant that supports three sources and multiple answer shapes. Mention remaining tradeoffs: source ambiguity, domain vocabulary, latency from LLM calls, and transitional source-specific repairs.
12. **Limitations**
    Be honest: not yet a full semantic parser, some fallback rules remain, source-specific repairs still exist, Elasticsearch must be available for full runtime, and medical/operational interpretation still requires human review.
13. **Future Work**
    Suggest: reduce transitional repairs, improve evaluation set, add more schema-aware retrieval for values, support careful cross-source questions, improve follow-up memory, add stronger privacy controls, make comparison execution more generic, and investigate Elasticsearch MCP as a simpler future architecture.
14. **Conclusion**
    Close with the contribution: a working prototype that turns plain-language hospital-history questions into validated, executable analytics with transparent debug traces.

Future-work note for Elasticsearch MCP:

- The README describes a possible future migration to the Elasticsearch MCP server.
- Present MCP as future work, not as the current implementation.
- Current implementation uses a controlled pipeline: classification, intent planning, canonical validation, query building, execution strategies, result formatting, and narration.
- MCP could simplify the architecture by letting the model write and run Elasticsearch queries through a tool call.
- The likely migration path is hybrid:
  1. keep guardrails, source routing, UI, exports, follow-up handling, and config loading
  2. replace the slot planner, query builder, execution strategies, and execution agent with a model + Elasticsearch MCP tool call
  3. keep YAML configs as schema/context for the model
- Be honest about the tradeoff: MCP could remove a large amount of code and make new question types easier, but it may reduce deterministic control, validator guarantees, reproducibility, and predictable latency.

Suggested 15-slide structure:

1. Title: HistoryAssistant
2. Problem Context
3. Research Goal
4. Data Sources
5. System Overview Diagram
6. Classification And Source Routing
7. Intent Planner And Canonical Plan
8. Validation And Guardrails
9. Elasticsearch Execution
10. UI And Debug Experience
11. Demo Flow
12. Results / What Works
13. Limitations
14. Future Work / Elasticsearch MCP
15. Conclusion / Q&A

Slide design guidance:

- Use one clean architecture diagram rather than many dense code screenshots.
- Use short bullets; put implementation details in speaker notes.
- Show at least one real debug payload or canonical plan excerpt, but crop it to the fields that matter.
- Use screenshots from the Gradio UI for demo/result slides.
- Keep the tone research-oriented: problem, method, implementation, evidence, limitations, future work.
- Avoid claiming clinical decision support. Describe the system as an analytics and exploration assistant over historical operational data.

## Core Rules

- Keep the pipeline modular: classify -> plan -> validate -> execute -> narrate.
- Prefer config changes over code changes when the issue is semantic.
- Do not hardcode behavior in `gradio_ui.py` or other presentation files to compensate for planning or execution bugs.
- Do not bypass validator safeguards just to make one prompt work.
- For single-source questions, select one target table first. Only add cross-source orchestration intentionally.
- Preserve dataset-relative time behavior. Relative dates should be anchored to dataset/reference-date logic, not blindly to machine time.
- Respect scope and outlier policies:
  - `scope_policy`: default is `qa_ready`
  - `outlier_policy`: default is `exclude_negative`
- Runtime config is cached with `lru_cache`. After changing YAML config, assume the app must be restarted.
- Old scenario routing has been removed from the active runtime. Do not reintroduce a separate `simple_lookup | aggregation | filtering_ranking` scenario router unless explicitly requested.

## Current Runtime Shape

Main entry point:

- `python src/run_gradio.py`
- The app currently launches on `127.0.0.1:7861`.

Primary runtime path:

1. `src/run_gradio.py`
2. `src/historyAssistant/app/gradio_ui.py`
3. `src/historyAssistant/app/chatbot_service.py`
4. `src/historyAssistant/agents/classification_agent.py`
5. `src/historyAssistant/planning/intent_planner/slot_planner.py`
6. `src/historyAssistant/agents/execution_agent.py`
7. `src/historyAssistant/app/narration.py`

High-level pipeline:

```text
User question
  -> ClassificationAgent
       guardrails
       vocabulary normalization
       source routing
       chart intent detection
  -> Intent Planner
       neutral source intent resolution
       answer-shape resolution: raw_records or analytic_summary
       deterministic slots or LLM-assisted slots
       canonical plan building
       validation
  -> Source-specific post-plan repairs
  -> Date/time picker fallbacks
  -> Validation again
  -> ExecutionAgent
  -> Result formatting
       raw-record table-first answer, chart payload, scalar/record payloads
  -> Narration / Chart / Table / Export
```

Important current transition state:

- `planning/08_scenario_router.py` has been deleted from the active code path.
- `agents/planning_agent.py`, `app/planning.py`, `llm/03_planning_prompt.py`, and `llm/04_canonical_bridge.py` have been removed from the active intent-planner runtime.
- `planning/simple_lookup.py` has been removed. Raw-record and distinct-value lookup now go through `intent_planner/record_intent.py`, shared filtering is in `planning/extraction/filter_extractor.py`, and canonical plan construction is centralized in `intent_planner/plan_builder.py`.
- `app/gradio_ui.py` may still contain stale labels such as "Scenario Detect" in debug UI text. Treat those as presentation debt, not evidence that scenario routing is still active.
- `planning/core/validator.py` still accepts a `scenario` argument for compatibility, but currently ignores it.

## Folder Structure

### Root folders

- `configs/`
  Source of truth for metrics, vocabularies, chart rules, date rules, table metadata, source metadata, and runtime settings.
- `src/`
  Application and pipeline code.
- `data/`
  Processed fact tables and local data assets.
- `eval/`
  Evaluation scripts, notebooks, benchmark files, and tests.
- `docs/`
  Human-facing documentation.
- `notebooks/`
  Exploration, validation, and dataset-specific analysis.
- `outputs/`, `exports/`
  Generated outputs.

### Important code folders

- `src/historyAssistant/agents/`
  Active runtime agents:
  - `classification_agent.py`
  - `execution_agent.py`
  The previous `planning_agent.py` is no longer part of the active runtime.
- `src/historyAssistant/planning/`
  Planning package root. Keep broad exports here; most implementation lives in `classification/`, `chart/`, `core/`, `extraction/`, and `intent_planner/`.
- `src/historyAssistant/planning/classification/`
  Classification helpers:
  - `guardrails.py`
  - `source_router.py`
- `src/historyAssistant/planning/chart/`
  Chart policy and chart rule helpers:
  - `chart_rules.py`
- `src/historyAssistant/planning/core/`
  Canonical plan schema and validation:
  - `canonical_schema.py`
  - `validator.py`
- `src/historyAssistant/planning/extraction/`
  Reusable extraction helpers shared by the planner:
  - `filter_extractor.py`
  - `filter_value_resolver.py`
  - `deterministic_filter_fallback.py`
  - `ranking_signals.py`
  - `temporal_parser.py`
  - `deterministic_temporal_fallback.py`
  - `temporal_value_resolver.py`
  - `time_range_parser.py`
  - `temporal_constants.py`
  - `vocabulary_normalizer.py`
- `src/historyAssistant/planning/intent_planner/`
  Current planning center:
  - `slot_planner.py` - top-level intent-planner orchestration
  - `answer_shape.py` - replaces the old scenario router with `raw_records` vs `analytic_summary`
  - `neutral_intent.py` - source re-resolution after the first source route
  - `intent_extractor.py` - deterministic metric, filter, group, date, bucket, ranking, comparison, and intersection slots
  - `llm_slot_extractor.py` - LLM-assisted slot extraction; it returns slots, not Elasticsearch
  - `plan_builder.py` - turns slots into canonical plans
  - `slot_quality.py` - decides when deterministic slots are trustworthy
  - `comparison_intent.py` and `value_filter_inference.py` - focused intent helpers
- `src/historyAssistant/llm/`
  LLM client setup, example loading, response parsing compatibility, and system prompt helpers. The old full-plan prompt and canonical bridge files are removed from the active path.
- `src/historyAssistant/execution/`
  Query models, Elasticsearch query compilation, execution orchestration, aggregation strategies, chart rendering, result formatting, and display labels.
  Important execution modules:
  - `query_models.py` - canonical plan to execution `QueryPlan`
  - `query_builder/` - compiles `QueryPlan` into Elasticsearch request bodies:
    `compiler.py`, `filter_builder.py`, `field_resolver.py`, and `date_filters.py`
  - `executor/` - runs canonical plans and query plans:
    `orchestrator.py`, `comparison_executor.py`, and `entity_intersection.py`
  - `result/` - converts execution dataframes into raw results, labels, chart payloads, scalar answers, raw-record text, narration payloads, and validated narration field selections
    - `formatter.py` - dataframe/result to raw result payload
    - `display_labels.py` - shared display labels
    - `raw_record_answers.py` - deterministic raw-record/table-first answer helpers
    - `deterministic_narration.py` - deterministic fallback narration for errors, warnings, scalar and simple record outputs
    - `narration_payloads.py` - compact/full payloads for LLM narration
    - `narration_field_selector.py` - LLM chooses requested/summary/display fields from actual result columns only
    - `narration_config.py` - resolves table/schema context for result narration
  - `strategies/` - aggregation-specific query bodies and response extraction split by execution family:
    `base.py`, `basic_metric_strategies.py`, `percentile_metric_strategy.py`,
    `threshold_metric_strategies.py`, `data_quality_metric_strategies.py`,
    `average_count_per_group_strategy.py`, `raw_record_strategies.py`,
    `time_bucket_strategy.py`, `workflow_room_switch_strategy.py`,
    `numeric_summary_strategy.py`, and `registry.py`
  Generic numeric summaries use `metric: numeric_summary` with `metric_params.field` and
  `metric_params.numeric_aggregation`. The LLM may select these from table-registry
  `measure_fields` / `measure_field_maps`; the validator must reject fields or
  aggregations not allowed by that table metadata.
- `src/historyAssistant/app/`
  Chat orchestration, clarification, follow-up handling, narration orchestration, runtime loading, picker filters, Gradio UI, export flow, source-specific plan repairs, and UI table helpers. `app/narration.py` should stay a thin facade around narration strategy selection and LLM calls; result-specific formatting belongs in `execution/result/`.
- `src/historyAssistant/config/`
  Config loading, source-specific config bundles, dynamic allowed-values loading, and data/index preparation helpers.

## Data Sources And Tables

Configured sources in `configs/shared/app_config.yaml`:

- `calls`
  - canonical table: `historyCalls`
  - index: `televic.history.call`
  - date field: `call_date`
  - timestamp field: `timestamp_utc`
  - data path: `data/historyCalls/03_processed/fact_history_calls_v3.csv`
- `locations`
  - canonical table: `historyLocations`
  - index: `televic.history.location`
  - date field: `timestamp`
  - timestamp field: `timestamp`
  - data path: `data/historyLocations/processed/fact_history_locations_v1.csv`
- `workflows`
  - canonical table: `historyWorkflows`
  - index: `televic.history.workflow`
  - date field: `workflow_date`
  - timestamp field: `timestamp_utc`
  - end timestamp field: `cancel_timestamp_utc`
  - data path: `data/historyWorkflows/processed/fact_history_workflows_v1.csv`

When changing routing, planning, or execution, always think in terms of source id + canonical table + table config, not just a hardcoded Elasticsearch index.

## Current Planning Model

### ClassificationAgent

`src/historyAssistant/agents/classification_agent.py` currently does:

1. guardrails via `planning/classification/guardrails.py`
2. vocabulary normalization via `planning/extraction/vocabulary_normalizer.py`
3. source routing via `planning/classification/source_router.py`
4. chart intent detection via `app/chart_detection.py`

It does not call a scenario router. It returns source ids, canonical tables, detected entities, request type, and chart preferences.

### Source Routing

`planning/classification/source_router.py` combines:

- entity-rule routing from normalized detected entities
- embedding-based source scoring
- LLM fallback when available and the deterministic/embedding signals are weak

If entity routing and embedding routing strongly disagree, embedding may override a high-confidence entity route. Check `debug["source_routing"]` and `debug["selected_source"]`.

### New Structure Slot Planner

`planning/intent_planner/slot_planner.py` is the active planner.

The planner first calls `resolve_source_intent()` from `neutral_intent.py`, which can switch source after classification using source-specific vocabularies and intent phrases.

Then it calls `resolve_answer_shape()` from `answer_shape.py`:

- `raw_records`: fetch rows without computing a summary
- `analytic_summary`: compute a count, average, trend, ranking, comparison, grouped table, or chart-ready result

For `raw_records` and `distinct_values`, the planner calls `intent_planner/record_intent.py` and then uses `plan_builder.py`.

For `analytic_summary`, it calls deterministic slot extraction first. If deterministic slots are missing, low quality, or fail validation, it calls `llm_slot_extractor.py` as the single unified LLM planning fallback. The LLM fills structured slots only; it must not write Elasticsearch queries.

Current LLM-call shape:

- Source routing uses local entity/embedding routing first and only calls LLM when those signals are weak.
- Answer shape is resolved deterministically in `slot_planner.py`; do not add a separate answer-shape LLM call.
- Separate filter and projection LLM calls have been removed from the active planner. Use the unified LLM slot extractor when deterministic planning is not trustworthy.
- The unified `llm_slot_extractor.py` call covers answer shape, metric, group_by, filters, dates, projection, comparison, and ranking when deterministic planning is not trustworthy.
- Narration may still call LLM depending on `runtime.narration_mode`; deterministic narration is used for errors, warnings, raw-record table-first answers, and when configured or forced.
- Raw-record narration should not add an extra LLM call. If a raw-record answer is slow, inspect source routing and slot planning first.
- For non-raw results, `execution/result/narration_field_selector.py` may use one structured LLM call to choose requested, summary, or display fields. The selector must only choose from actual result columns and table schema; it must not invent fields or write prose.
- For a normal first-turn question, keep the target around 2-3 LLM calls: optional source-router LLM, optional unified slot LLM, and optional narration LLM/field-selection call. Raw-record paths should usually avoid narration LLM calls after execution.

### Raw Record And Distinct-Value Lookup

`planning/intent_planner/record_intent.py` owns raw-record and distinct-value slot extraction.

It currently owns:

- raw-record slot extraction
- distinct-value projection slot extraction
- lookup limit extraction
- first/latest record ordering
- default source field selection
- raw lookup projection detection

For raw-record canonical plans, the validator ensures configured table date/timestamp fields are present in `source_fields` when source fields are explicitly limited. Do not hardcode timestamp field names in narration or UI code; fix table metadata or validator behavior if time columns are missing.

`planning/extraction/filter_extractor.py` coordinates cross-source dimension filter extraction used by both raw lookup and analytic slot extraction. The active planner currently uses deterministic fallback plus config/ES-backed resolution, not a separate filter LLM call.

`planning/extraction/temporal_parser.py` is a small facade. It does not make a separate LLM call. Existing LLM slot extraction may propose `date_rule`, `date_range`, and `date_params`, then `temporal_value_resolver.py` validates them. Deterministic date parsing lives in `deterministic_temporal_fallback.py`, calendar math lives in `temporal_value_resolver.py`, and time/day-part parsing lives in `time_range_parser.py`.

Do not reintroduce a separate simple lookup planner. Raw lookup should stay in the intent-planner slot -> plan-builder path.

### Source-Specific Repairs

`app/source_plan_repairs.py` still runs after intent planning in `chatbot_service.py`.

It handles source-specific domain patches, including:

- location person hash handling and location-area normalization
- last-known-location plan repair
- workflow presence filters
- workflow room normalization
- raw-record sort repair
- date picker fallback guard for last-known-location

This is a transitional layer. Prefer moving durable, source-aware planning semantics into config or `planning/intent_planner/` when touching these rules, but do not bypass it without understanding the behavior it protects.

## Where Changes Should Go

Use this routing table before editing code.

### If the problem is vocabulary, synonyms, or naming

Edit:

- `configs/*/vocabulary.yaml`

Maybe also inspect:

- `src/historyAssistant/planning/extraction/vocabulary_normalizer.py`
- `src/historyAssistant/planning/intent_planner/neutral_intent.py`
- `src/historyAssistant/planning/intent_planner/answer_shape.py`

### If the problem is source selection

Inspect/edit:

- `src/historyAssistant/planning/classification/source_router.py`
- `src/historyAssistant/planning/intent_planner/neutral_intent.py`
- `configs/*/vocabulary.yaml` under source intent, metric, table, and dimension phrases

Always inspect both initial `source_routing` and `slot_planner.source_intent_resolution` in the debug payload.

### If the problem is answer shape

Inspect/edit:

- `src/historyAssistant/planning/intent_planner/answer_shape.py`
- `configs/*/metric_catalog.yaml` for metric aggregation type
- `configs/*/vocabulary.yaml` for phrases that normalize to raw-record or analytic metrics
- `src/historyAssistant/planning/intent_planner/record_intent.py` if the issue is raw-record or distinct-value slot extraction after shape selection

Answer shape replaces old scenario routing. Do not add a new scenario router for wording variants.

### If the problem is supported metric semantics

Edit:

- `configs/*/metric_catalog.yaml`
- `configs/*/vocabulary.yaml`
- `configs/*/chart_catalog.yaml` if charts are affected

Only edit code if the metric needs a new aggregation family, slot extraction behavior, or validator/executor support.

### If the problem is allowed filters, dimensions, value mapping, or field metadata

Edit:

- `configs/*/table_registry.yaml`

Maybe also inspect:

- `src/historyAssistant/planning/extraction/filter_extractor.py` for dimension filter extraction
- `src/historyAssistant/planning/intent_planner/value_filter_inference.py`
- `src/historyAssistant/app/source_plan_repairs.py` for source-specific normalization

### If the problem is date parsing or supported date windows

Edit:

- `configs/shared/date_rules.yaml`
- `src/historyAssistant/planning/extraction/temporal_parser.py`
- `src/historyAssistant/planning/core/validator.py`

### If the intent planner engaged incorrectly or failed

Inspect/edit:

- `src/historyAssistant/planning/intent_planner/slot_planner.py`
- `src/historyAssistant/planning/intent_planner/answer_shape.py`
- `src/historyAssistant/planning/intent_planner/neutral_intent.py`
- `src/historyAssistant/planning/intent_planner/intent_extractor.py`
- `src/historyAssistant/planning/intent_planner/llm_slot_extractor.py`
- `src/historyAssistant/planning/intent_planner/plan_builder.py`
- `src/historyAssistant/planning/intent_planner/slot_quality.py`

The debug payload's `debug["slot_planner"]` field shows source resolution, answer shape, slot extraction, LLM fallback, validation, and plan preview.

### If the problem is raw-record lookup

Inspect/edit:

- `src/historyAssistant/planning/intent_planner/answer_shape.py` if the wrong shape was selected
- `src/historyAssistant/planning/extraction/filter_extractor.py` if raw-record filters are wrong
- `src/historyAssistant/planning/intent_planner/record_intent.py` if raw-record limits, ordering, projection, or source fields are wrong
- `src/historyAssistant/app/source_plan_repairs.py` if the issue is source-specific location/workflow behavior after raw plan construction

### If the problem is invalid or weak analytical planning

Inspect/edit:

- `src/historyAssistant/planning/intent_planner/intent_extractor.py`
- `src/historyAssistant/planning/intent_planner/llm_slot_extractor.py`
- `src/historyAssistant/planning/intent_planner/plan_builder.py`
- `src/historyAssistant/planning/intent_planner/comparison_intent.py`
- `src/historyAssistant/planning/intent_planner/value_filter_inference.py`
- `src/historyAssistant/planning/extraction/ranking_signals.py`
- `src/historyAssistant/planning/core/validator.py`

### If the problem is Elasticsearch query behavior or result extraction

Inspect/edit:

- `src/historyAssistant/execution/query_models.py`
- `src/historyAssistant/execution/query_builder/`
- `src/historyAssistant/execution/strategies/`
- `src/historyAssistant/execution/executor/`
- `src/historyAssistant/execution/result/`

### If the problem is answer wording, formatting, chart output, or export behavior

Inspect/edit:

- `src/historyAssistant/app/narration.py`
- `src/historyAssistant/execution/result/formatter.py`
- `src/historyAssistant/execution/result/raw_record_answers.py`
- `src/historyAssistant/execution/result/deterministic_narration.py`
- `src/historyAssistant/execution/result/narration_payloads.py`
- `src/historyAssistant/execution/result/narration_field_selector.py`
- `src/historyAssistant/execution/chart/renderer.py`
- `src/historyAssistant/app/export.py`
- `src/historyAssistant/app/ui_table_helpers.py`

Keep `app/narration.py` as orchestration only. Do not add source-specific field aliases, display order lists, or regex-heavy wording rules there. For narration field choices, prefer schema descriptions plus validated LLM selection from actual result columns. For raw records, prefer deterministic table-first output.

### If the problem is UI-only

Inspect/edit:

- `src/historyAssistant/app/gradio_ui.py`
- `src/historyAssistant/app/assets/gradio_ui.css`

Do not fix semantic bugs in the UI layer.

## Agent Loop

Every agent working in this repo should follow this loop:

1. Read this file first.
2. Identify the affected layer:
   - config
   - normalization
   - source routing
   - answer shape
   - slot extraction
   - raw lookup helper
   - validation
   - execution
   - narration
   - UI
3. Reproduce the issue with the smallest useful path.
4. Inspect the generated plan and query before changing prompts.
5. Make the smallest change in the correct layer.
6. Validate with the narrowest test or script that proves the fix.
7. If config changed, restart-dependent paths before concluding the change works.
8. Summarize the fix in pipeline terms, not just file terms.

## Debugging Order

When an answer is wrong, check in this order:

1. Did guardrails block or distort the request?
2. Did vocabulary normalization map the request correctly?
3. Did initial source routing choose the right source/table? Check `debug["source_routing"]` and `debug["selected_source"]`.
4. Did intent-planner source intent change or keep the selected source correctly? Check `debug["slot_planner"]["source_intent_resolution"]`.
5. Did answer shape choose `raw_records` or `analytic_summary` correctly? Check `debug["slot_planner"]["answer_shape"]`.
6. For raw records, did `record_intent` and `record_plan` produce the intended filters, source fields, limit, order, and metric?
7. For analytics, did deterministic or LLM slot extraction produce the intended metric, filters, group_by, comparison, ranking, and date slots?
8. Did source-specific repairs in `source_plan_repairs.py` alter the plan in an unintended way?
9. Did validator repair the plan into something unintended?
10. For raw records, did validator add the table's configured date/timestamp fields to `source_fields`?
11. Did execution target the correct index/date/timestamp fields?
12. Did result formatting or narration misdescribe a correct result?

Before touching prompts, inspect the debug payload and generated Elasticsearch query.

## Subagents

Use subagents only for bounded, parallelizable work. Do not split the main reasoning path if the next step depends on it immediately.

Recommended subagent roles:

- `Config Investigator`
  Reads `configs/` and identifies whether a bug is semantic, schema-related, vocabulary-related, or chart-related.
- `Planner Investigator`
  Checks source intent, answer shape, deterministic slots, LLM slots, plan builder, and validator behavior.
- `Raw Lookup Investigator`
  Checks `filter_extractor.py`, `record_intent.py`, source fields, ordering, limits, projections, and source repairs.
- `Execution Investigator`
  Checks query compilation, aggregation strategy selection, index selection, and result extraction.
- `UI Verifier`
  Confirms whether a problem is presentation-only or truly semantic.
- `Eval Runner`
  Runs targeted evaluation scripts or tests and reports regressions.

Subagent ownership guidance:

- Split by layer, not by random file batches.
- Avoid overlapping writes.
- Never have multiple subagents editing the same config file or the same pipeline module at once.

## Skills

If the environment supports project skills, use these in this order.

### `historyassistant-repo`

Use first for almost any task in this repository. It matches the architecture, config-driven workflow, and validation expectations of this project.

### `historyassistant-debugger`

Use when the goal is to localize a failure before fixing it. This skill is for tracing bugs to the correct layer: config semantics, normalization, source routing, answer shape, slot planning, validator repair, execution, or presentation.

## Change Playbooks

### Add or change a metric

1. Update `configs/*/metric_catalog.yaml`.
2. Update `configs/*/vocabulary.yaml` with synonyms and source-intent phrases.
3. Update `configs/*/chart_catalog.yaml` if charts are affected.
4. Check `intent_planner/intent_extractor.py` and `intent_planner/llm_slot_extractor.py` if the metric needs new slot behavior.
5. Edit `planning/core/validator.py` only if structural repair is needed.
6. Edit aggregation strategy code only if this is a new aggregation family.
7. Edit or migrate `app/source_plan_repairs.py` only if source-specific domain patching is needed.

### Add or change a dimension/filter

1. Update `configs/*/table_registry.yaml` with allowed dimensions, display maps, and `dimension_value_maps`.
2. Update `configs/*/vocabulary.yaml`.
3. Check `planning/extraction/filter_extractor.py` for filter extraction behavior.
4. Check `intent_planner/value_filter_inference.py` if the field needs inferred filter values.
5. Check `intent_planner/record_intent.py` if raw lookup source fields or projection behavior are affected.
6. Check `app/source_plan_repairs.py` if the field needs source-specific normalization.

If the field is a time/date column used in raw-record output, make sure the table's `date_field`, `timestamp_field`, `timestamp_fields`, and/or `time_field` metadata are correct. Raw-record temporal source-field repair depends on those schema entries.

### Tune answer shape

1. Check whether the metric aggregation already implies the correct shape.
2. Adjust `intent_planner/answer_shape.py` only for high-confidence, source-independent shape signals.
3. Prefer vocabulary/metric config when the issue is domain language.
4. Avoid building a large hardcoded keyword router.
5. Keep ambiguous wording flowing to LLM-assisted slot extraction when possible.

### Tune the new structure planner

1. Add or adjust source phrases in `configs/*/vocabulary.yaml`.
2. Check `intent_planner/neutral_intent.py` for source scoring behavior.
3. Check `intent_planner/intent_extractor.py` for deterministic slot extraction.
4. Check `intent_planner/slot_quality.py` for deterministic-vs-LLM fallback decisions.
5. If deterministic slots are not enough, extend `intent_planner/llm_slot_extractor.py` so the LLM fills slots, not raw Elasticsearch queries.

### Improve language understanding

1. Start with vocabulary and metric descriptions.
2. Then inspect answer shape and source intent.
3. Then inspect deterministic slots and LLM slot schema.
4. Then inspect validator or execution if the plan is structurally valid but behavior is wrong.

Do not add narration-only aliases or large regex lists for field requests. Non-raw narration field selection should use `narration_field_selector.py`, table schema metadata, and actual result columns. Raw-record result text should remain deterministic unless there is a deliberate latency tradeoff.

### Add a new source-aware feature

1. Confirm the feature is table-aware, not hardcoded to calls.
2. Route by source id and canonical table first.
3. Resolve index and date/timestamp fields from config.
4. Validate against the selected table schema.
5. Execute through the normal execution path.
6. Add source-specific repairs only if config and slot extraction cannot express the rule cleanly.

### Fix a wrong chart

1. Check chart intent detection and whether the plan requested chart mode intentionally.
2. Check chart rules and metric compatibility.
3. Check plan `chart_spec`.
4. Check result formatting payload.
5. Check `execution/chart/renderer.py`.
6. Only touch the UI if rendering is correct but display wiring is broken.

## Validation Rules

After changes, validate as narrowly as possible but enough to prove behavior.

Useful commands:

```bash
python src/run_gradio.py
python eval/debug_single_question.py "how many calls last week"
python -m unittest eval.tests.test_calls_simple_lookup_regression
python -m unittest eval.tests.test_location_simple_lookup
python -m unittest eval.tests.test_workflow_simple_lookup
python -m unittest eval.tests.test_ranking_overfetch
python -m unittest eval.tests.test_raw_record_summary
python -m unittest eval.tests.test_temporal_parser_month_typos
python eval/eval_gold_baseline.py
```

Notes:

- Some tests load dynamic allowed values from Elasticsearch. If Elasticsearch is not running on `localhost:9200`, those lookups may warn, slow down, or fail depending on the path.
- The old `eval/eval_simple_lookup.py`, `eval/eval_aggregation.py`, `eval/eval_filtering_ranking.py`, and `eval/eval_nurse_benchmark.py` scripts are not present in the current tree.
- Some notebooks and older eval code still refer to scenarios. Treat those as legacy evaluation labels unless the active runtime code also references them.

Validation expectations by change type:

- Config-only change: verify the affected loader/validator path and one real question flow.
- Source-routing change: verify selected source id, canonical table, and ES index.
- Answer-shape change: verify `slot_planner.answer_shape` and whether the raw or analytic path was used.
- Planning change: inspect canonical plan before and after validator repair.
- Raw lookup change: inspect filters, source fields, limit, order, date/time, and raw-record metric.
- Execution change: inspect generated Elasticsearch query and result dataframe/records shape.
- Chart change: verify chart payload and rendered figure.
- Multi-source change: verify selected table, index, date field, timestamp field, and output labels.

## Coding Style For This Repo

- Prefer small, explicit functions.
- Preserve existing naming and module boundaries.
- Keep semantic rules in config when possible.
- Keep canonical-plan behavior centralized.
- Add concise comments only when the code would otherwise be hard to follow.
- Avoid introducing side paths that bypass the normal pipeline.
- Avoid reviving removed old-planner files unless the user explicitly asks for a compatibility restoration.

## Definition Of Done

Do not consider a task done until all of these are true:

- The change is in the correct layer.
- The behavior is verified with a focused validation path.
- No new hardcoded semantic shortcut was introduced where config should own the rule.
- If YAML changed, restart-sensitive behavior was accounted for.
- The final explanation can clearly say:
  - what layer changed
  - why that layer was the correct fix
  - how it was validated

## Quick Start Checklist For Any New Agent

1. Read `AGENTS.md`.
2. Read `CLAUDE.md`, but note that it may lag behind this file and still mention removed old-planner pieces.
3. Inspect `configs/shared/app_config.yaml`.
4. Identify the target source/table and affected layer.
5. Reproduce with `eval/debug_single_question.py` or the smallest equivalent path.
6. Inspect `debug["selected_source"]`, `debug["slot_planner"]`, the canonical plan, and the generated ES query.
7. Fix the issue in the right module or config.
8. Validate narrowly.
9. Summarize in pipeline terms.
