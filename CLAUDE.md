# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

HistoryAssistant is a natural-language analytics assistant for hospital nurse-call history data. Users ask questions in plain language; the system routes them through a pipeline (classify → slot-plan → execute) and returns a text answer, inline table, Plotly chart, or downloadable export, all backed by Elasticsearch.

Three data sources are supported with parallel config directories:
- **historyCalls** — nurse-call records (response time, staying time, call counts, etc.)
- **historyLocations** — staff location tracking (movement, last-known location)
- **historyWorkflows** — staff workflow presence events (presence counts, room-switch patterns)

## How to run

```bash
# Start the Gradio UI (main entry point)
python src/run_gradio.py
# Opens at http://127.0.0.1:7860

# Debug a single question end-to-end
python eval/debug_single_question.py "how many calls last week"

# Run evaluation suites
python eval/eval_simple_lookup.py
python eval/eval_aggregation.py
python eval/eval_filtering_ranking.py
python eval/eval_nurse_benchmark.py
```

Required `.env` variables at the repo root:
```
OPENAI_API_KEY=
OPENAI_DEPLOYMENT=
OPEN_BASE_URL=          # Azure OpenAI base URL
OPENAI_API_VERSION=     # e.g. 2025-04-01-preview
HF_TOKEN=               # for sentence-transformers embeddings
```

Elasticsearch must be running and indexed. Use `src/historyAssistant/config/dump_csv_to_elasticsearch_*.py` to load data for each source.

## Full pipeline architecture

All orchestration lives in `app/chatbot_service.py`.

```
User question
  ↓
ClassificationAgent        (guardrails → normalization → source routing → chart detection)
  ↓  source_id, detected_entities, request_type
  ↓
Slot Planner               (intent extraction → slot filling → quality assessment → plan building)
  │  SlotPlannerResult(used=True)  → CanonicalPlan → ExecutionAgent
  │  SlotPlannerResult(used=False) → fall through to LLM planner
  ↓
LLM Planner                (few-shot retrieval → LLM JSON plan → source-specific repairs → validate/retry)
  ↓  CanonicalPlan
  ↓
ExecutionAgent             (CanonicalPlan → QueryPlan → ES query → strategy.extract_results())
  ↓
Narration / Inline Table / Chart / Export
```

---

### ClassificationAgent (`agents/classification_agent.py`)

Steps run in order:

1. **Guardrails** (`planning/guardrails.py`) — sentence-transformer embeddings compare the question against blocked topics in `configs/calls/unsupported_topic.yaml`. Returns structured `GuardrailResult` (with `GuardrailHit` entries). Blocked questions return an error message without planning.
2. **Vocabulary normalization** (`planning/extraction/vocabulary_normalizer.py`) — replaces synonyms with canonical names using the active source's `vocabulary.yaml`. Returns `NormalizationResult` (normalized_question, replacements, detected_entities).
3. **Source routing** (`planning/source_router.py`) — determines `source_id` (`calls | locations | workflows`) via rule-based, entity-mapping, and embedding fallback. Returns `SourceRoutingResult` with confidence score and debug info.
4. **Chart detection** (`app/chart_detection.py`) — regex detects chart requests and preferred chart type; sets `request_type = REQUEST_TYPE_CHART`.

Note: Explicit scenario routing (`simple_lookup | aggregation | filtering_ranking`) was removed. Answer shape is now resolved inside the slot planner via `planning/new_structure/answer_shape.py`.

---

### Slot Planner (`planning/new_structure/slot_planner.py`)

**Purpose**: Unified intent-to-plan translation using slot extraction and quality assessment. Replaces the old `fast_planner.py` and the deterministic fast path.

**Entry point**: `try_build_slot_plan(question, source_id, config_bundle, detected_entities)` — called in `chatbot_service.py` immediately after classification.

**Returns**: `SlotPlannerResult` dataclass:
```python
@dataclass
class SlotPlannerResult:
    used: bool
    canonical_plan: Optional[dict]   # None if used=False
    reason: str
    confidence: float
    subtype: str
    debug: dict
```

**Pipeline inside the slot planner**:
1. `intent_extractor.py` — deterministic signal detection (metric, group_by, ranking, bucket).
2. `llm_slot_extractor.py` — LLM-based slot extraction for ambiguous cases.
3. `slot_quality.py` — `assess_slot_quality()` returns `SlotQualityResult`; low confidence falls through to LLM planner.
4. `answer_shape.py` — determines query intent: `raw_records | aggregation | distinct_values | …`
5. Intent-specific builders:
   - `record_intent.py` — raw record / lookup queries
   - `neutral_intent.py` — source-specific intent normalization
   - `comparison_intent.py` — comparison period queries
6. `plan_builder.py` — converts filled `IntentSlots` → `CanonicalPlan`.

**Key slot types** (`planning/new_structure/intent_slots.py`):
- `IntentSlots` — base (metric, aggregation, filters, group_by, order_by)
- `RankingSlots` — extends with ranking signal info
- `ComparisonSlots` — extends with compare_to fields
- `SlotExtractionResult` — wraps slots + confidence + debug

---

### Extraction helpers (`planning/extraction/`)

Shared utilities used by both the slot planner and the LLM planner path:

| Module | Purpose |
|---|---|
| `vocabulary_normalizer.py` | Synonym replacement; returns `NormalizationResult` |
| `filter_extractor.py` | Deterministic dimension filter extraction from question text |
| `filter_value_resolver.py` | Maps filter text to canonical dimension values via `dimensions_value_maps` |
| `ranking_signals.py` | Regex-based detection of sort/ranking language (`RankingSignal` dataclass) |
| `temporal_parser.py` | Date/time parsing |
| `temporal_value_resolver.py` | Resolves temporal entities to canonical date rules |
| `temporal_constants.py` | Time entity constants (days, months, durations) |
| `time_range_parser.py` | Parses time-of-day ranges |
| `deterministic_filter_fallback.py` | Fallback filter extraction when primary extraction fails |
| `deterministic_temporal_fallback.py` | Fallback temporal extraction |
| `llm_filter_candidate_extractor.py` | LLM-based filter candidate extraction |

**Filter extraction design rules** (implemented in `filter_extractor.py`):
- **`allowed_values` guard**: for ward, zone, and area fields — exact substring matching against config values; regex fallbacks only run when no `allowed_values` are configured.
- **Cross-source field names**: `_FIELD_NAME_MAP` maps generic concepts (`ward`, `zone`, `area`, `person`, `person_type`, `room`) to actual ES field names per table.
- **Area→ward redirect**: when `area` allowed_values don't match, value is alpha↔digit-boundary-normalised and checked against ward `allowed_values`.
- **`personType` filter**: only fires on exact word-boundary match; plural forms ("patients") do NOT match "Patient" — intentional so ranking questions keep `personFirstName` in group_by.

---

### LLM Planner (`agents/planning_agent.py`, `app/planning.py`)

Used when the slot planner returns `used=False`.

1. Retrieves similar few-shot examples (`llm/00_example_retriever.py`).
2. LLM generates JSON plan.
3. **Source-specific repairs** applied (`app/source_plan_repairs.py`).
4. Validated and repaired by `planning/core/validator.py`. One retry if blocking errors remain.
5. Date/time from UI pickers injected as fallback after planning (`app/filters.py`).
6. Clarification questions generated before and after planning (`app/clarification.py`).

---

### Source-specific plan repairs (`app/source_plan_repairs.py`)

**Purpose**: Post-LLM semantic patching for domain patterns that the LLM cannot reliably handle.

#### historyLocations repairs (`repair_location_person_lookup()`):
- Normalizes room/area values to canonical format: `"1A Room 5"` → `"1A_ROOM 005"`
- Extracts SHA-256 hashed person IDs from questions (`_HASHED_PERSON_RE`)
- Detects "last seen / last known location" patterns → sets metric to `last_known_location`, aggregation to `raw_records`, limit 1
- Detects movement/mobility patterns → switches to `movement_event_count`
- Detects comparison patterns with 2+ people → uses `compare_to` field

#### historyWorkflows repairs (`repair_workflow_presence_lookup()`):
- Normalizes room values (K/SANITAIR/DVC prefixes with zero-padded numbers)
- Enforces required workflow filters
- Detects "who was present" → `raw_records` aggregation
- Handles person comparison queries via `initiator_person` filters

#### Workflow special patterns:
- `build_workflow_room_switch_plan()` — creates deterministic plan for "present in 2 different rooms within X minutes"
- `build_workflow_presence_lookup_plan()` — wraps simple_lookup with workflow repairs
- `should_apply_picker_date_fallback()` — guard: suppresses UI date-picker fallback for `last_known_location` metric

---

### ExecutionAgent (`agents/execution_agent.py`)

- Converts `CanonicalPlan` → `QueryPlan` (`execution/query_models.py`).
- `execution/es_builder.py` compiles the full Elasticsearch query body (filters, aggregation strategy, scope/outlier policies, date windows).
- `execution/aggregation_strategies.py` — `STRATEGY_REGISTRY` maps aggregation type → strategy class. Each implements `build_query_body()` and `extract_results()`. Current strategies: `Count`, `Sum`, `Avg`, `Percentile`, `Median`, `RawRecords`, `RateThreshold`, `ThresholdCount`, `Comparison`, `TimeBucket`, …
- Reference date for relative date rules is the **maximum call_date in the index**, not the machine clock.

---

### Canonical plan schema (`planning/core/canonical_schema.py`)

Central data contract between planner and executor. Key types:
- `QaPlan` — top-level plan (metric, aggregation, group_by, filters, date_rule, date_range, time_range, order_by, compare_to, bucket_interval, bucket_summary, scope_policy, outlier_policy, request_type, target_tables, display_order)
- `CanonicalFilter`, `CanonicalDateRange`, `CanonicalDateParams`, `CanonicalTimeRange`, `CanonicalOrderBy`, `CanonicalCompareTo`
- `CanonicalDisplayOrder` — presentation-only ordering independent from ranking
- `REQUEST_TYPE_QA` and `REQUEST_TYPE_CHART` control rendering path
- New fields on `CanonicalPlanBase`: `normalized_sub_request`, `sub_request_id`, `classification_confidence`

---

### Validator (`planning/core/validator.py`)

- Validates a `CanonicalPlan` dict; returns `ValidationResult` (with `missing_slots`, `ambiguity_reason` for sub-request handling).
- `infer_plan_policies()` applies scope/outlier policy defaults.
- Simplified: complex repair logic moved upstream into the slot planner.

---

### UI table helpers (`app/ui_table_helpers.py`)

Pandas DataFrame prep and HTML rendering for the Gradio chat interface.

Key functions:
- `prepare_display_df(df)` — drops hidden columns (`qa_ready_flag`, `timestamp_utc`, etc.), sorts by time/metric columns, rounds floats, renames columns to friendly display names
- `build_inline_chat_table_html(rows)` — converts list-of-dicts to HTML `<table>` for inline chat preview
- `should_show_inline_chat_table()` — only shows inline table for `raw_records` results
- `should_show_result_panel_table()` — hides result panel table if raw_records or chart is present
- `build_chat_download_link()` — creates `/gradio_api` file download link

---

## Planning module structure

```
planning/
├── guardrails.py              # Guardrail screening (GuardrailResult, GuardrailHit)
├── chart_rules.py             # Chart rule validation (ChartRuleSet, ChartValidationResult)
├── source_router.py           # Source routing with embedding fallback (SourceRoutingResult)
├── core/
│   ├── canonical_schema.py    # Canonical plan data types (QaPlan, CanonicalFilter, …)
│   └── validator.py           # Plan validation (ValidationResult, infer_plan_policies)
├── extraction/
│   ├── vocabulary_normalizer.py
│   ├── filter_extractor.py
│   ├── filter_value_resolver.py
│   ├── ranking_signals.py
│   ├── temporal_parser.py
│   ├── temporal_value_resolver.py
│   ├── temporal_constants.py
│   ├── time_range_parser.py
│   ├── deterministic_filter_fallback.py
│   ├── deterministic_temporal_fallback.py
│   └── llm_filter_candidate_extractor.py
└── new_structure/
    ├── slot_planner.py          # Entry point: try_build_slot_plan()
    ├── intent_extractor.py      # Deterministic intent signal detection
    ├── intent_slots.py          # IntentSlots, RankingSlots, ComparisonSlots, SlotExtractionResult
    ├── llm_slot_extractor.py    # LLM-based slot extraction
    ├── slot_quality.py          # SlotQualityResult, assess_slot_quality()
    ├── plan_builder.py          # Slots → CanonicalPlan
    ├── answer_shape.py          # Query intent shape resolution
    ├── record_intent.py         # Raw record / lookup handling
    ├── record_projection_selector.py
    ├── comparison_intent.py     # Comparison period handling
    ├── neutral_intent.py        # Source-specific intent normalization
    └── value_filter_inference.py
```

---

## Config files (prefer config over code for semantic changes)

All in `configs/`. Three source namespaces: `calls/`, `locations/`, `workflows/`. Each has its own `vocabulary.yaml`, `metric_catalog.yaml`, `table_registry.yaml`, etc.

| File | Controls |
|---|---|
| `shared/app_config.yaml` | ES connection, active table, row limits, defaults (scope/outlier policy), narration mode |
| `shared/date_rules.yaml` | Supported date windows and regex patterns |
| `<source>/metric_catalog.yaml` | Metric definitions, aggregation types, fields, units, thresholds |
| `<source>/table_registry.yaml` | Dimensions, measure fields, DQ flags, display value maps, ordering |
| `<source>/vocabulary.yaml` | Synonym → canonical name mappings; also contains `fast_path:` inference rules used by slot planner |
| `calls/chart_catalog.yaml` | Allowed chart types per metric, dimension roles |
| `calls/unsupported_topic.yaml` | Guardrail keywords and blocked-response text |

### Slot planner config (in `vocabulary.yaml`)

```yaml
fast_path:
  metric_inference:
    - all: ["last seen"]           # all keywords must match
      metric: "last_known_location"
      mode: "replace"              # or "add_if_missing"
  group_by_inference:
    - any: ["by ward"]             # any keyword must match
      dimension: "ward_name"
  group_by_from_time_entities:    # promote these time entity types to group_by
    - "call_dayofweek"
    - "call_hour"
  date_rule_overrides:
    - from: "last_7_days"
      to: "last_week"
      metric: "total_calls"
      any: ["trend"]
```

**Practical rule**: if a bug can be fixed by config, prefer config. Config changes do not require validator or prompt changes unless the new semantic is structurally new.

---

## Time-bucket analysis

Questions like "average calls per day", "median response time every 30 minutes", or "total per 15 days" use a two-level aggregation. Handled via two optional plan slots:

- `bucket_interval` — ES interval string: `"1d"`, `"15d"`, `"30m"`, `"1h"`, `"1w"`, `"1M"`. Slot planner or LLM fills this; validator normalises free-text (e.g., "30 minutes" → `"30m"`).
- `bucket_summary` — cross-bucket summary: `avg | min | max | sum | median | count`.

When `bucket_interval` is set, `compile_es_query` in `es_builder.py` routes to `TimeBucketStrategy`, which builds a `date_histogram` agg with an inner per-bucket metric, then computes the summary stat in Python. Groups from `group_by` wrap as outer `terms` aggs. No hardcoded interval values — any numeric + unit combination works.

Key files: `planning/core/canonical_schema.py` (schema), `planning/core/validator.py` (`normalize_bucket_interval`, `repair_bucket_fields`), `execution/aggregation_strategies.py` (`TimeBucketStrategy`).

---

## Change playbooks

### Add/change a metric
1. `configs/<source>/metric_catalog.yaml` (aggregation type, field, scope, outlier policy)
2. `configs/<source>/vocabulary.yaml` (synonyms; add `fast_path.metric_inference` rule if deterministic)
3. `configs/<source>/chart_catalog.yaml` (if charts apply)
4. `planning/core/validator.py` only if new repair logic needed
5. `execution/aggregation_strategies.py` only if aggregation family is new
6. `app/source_plan_repairs.py` only if source-specific domain patterns apply

### Add/change a dimension or filter field
1. `configs/<source>/table_registry.yaml` (allowed dimensions, display maps, `dimensions_value_maps`)
2. `configs/<source>/vocabulary.yaml` (synonyms)
3. Check `planning/core/validator.py` filter repair logic if needed
4. Check `app/source_plan_repairs.py` if field needs normalization (e.g., room naming)

### Tune the slot planner
1. Add/adjust `fast_path.metric_inference` or `fast_path.group_by_inference` rules in `vocabulary.yaml`
2. Add `date_rule_overrides` entries for specific metric + pattern combos
3. To block an aggregation type from the slot planner, update `planning/new_structure/slot_planner.py`

### Add a new data source
1. Create `configs/<source>/` with: `metric_catalog.yaml`, `table_registry.yaml`, `vocabulary.yaml`, `chart_catalog.yaml`
2. Add source loader to `config/dump_csv_to_elasticsearch_<source>.py`
3. Add source routing rules in `planning/source_router.py`
4. Add source-specific plan repairs in `app/source_plan_repairs.py` if needed
5. Update `shared/app_config.yaml` to register the new table name

### Improve language understanding
1. `configs/<source>/vocabulary.yaml` (synonyms + fast_path rules)
2. Few-shot examples in `data/<source>/04_ground_truth/`
3. Prompt builders in `src/historyAssistant/llm/`
4. Check whether validator is aggressively repairing the intended output

### Debug a wrong answer

Work through this chain before touching prompts:
1. Did guardrails block or distort the question?
2. Did normalization map language to the right metric/dimension?
3. Did source routing pick the right source (calls/locations/workflows)?
4. Did the **slot planner** engage when it shouldn't (or fail when it should)? Check `debug["slot_planner"]` in the debug payload.
5. Did `answer_shape.py` resolve the correct intent shape (aggregation vs raw_records vs distinct_values)?
6. Did the LLM plan produce the intended fields (if slot planner returned `used=False`)?
7. Did **source-specific repairs** alter the plan in an unintended way? Check `app/source_plan_repairs.py`.
8. Did validator repair into something unintended?
9. Does the compiled ES query match the intended semantics?
10. Is narration misdescribing a correct result?

---

## Key conventions

- **Scope policy**: `qa_ready` (cleaned data, default) vs `raw` (DQ investigations). Never mix them silently.
- **Outlier policy**: `exclude_negative` (default) removes records with negative durations. Use `none` intentionally.
- **Runtime caching**: `app/runtime.py` LRU-caches configs, LLM client, and few-shot examples. Changes to YAML files require a process restart.
- **Streaming**: `chatbot_service.handle_chat_turn_stream()` is the live UI path; `handle_chat_turn()` is the synchronous path used by eval scripts.
- **Chart rendering**: Plotly figures are produced by `execution/chart_renderer.py` from a chart payload built in `execution/result_formatter.py`. Chart hacks belong there, not in `gradio_ui.py`.
- **Table rendering**: DataFrame prep and HTML generation live in `app/ui_table_helpers.py`. Inline chat table only shows for `raw_records` results; result panel table shows for aggregation results without charts.
- **Presentation-only changes**: keep in `app/narration.py` or `execution/result_formatter.py`, not in the planner or validator.
- **Source-specific domain logic**: belongs in `app/source_plan_repairs.py`, not in the LLM prompt or validator.
- **Confidence scores**: slot planner emits `confidence=0.96` (or 0.85 for embedding-matched); LLM path does not set confidence. Use this to distinguish paths in debug payloads.
- **New extraction logic**: belongs in `planning/extraction/` (shared utilities) or `planning/new_structure/` (intent-specific builders), not in `app/` or `llm/`.
