# Plan: Introduce Standalone Stage 1 (Preparation) Pipeline

**Created:** 2026-04-24
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-24 17:07
**Branch:** `feature/align-clustering-pipeline-with-guidelines` (extended)

---

## Overview

**What:** Create a standalone Stage 1 pipeline (`preparation_pipeline.py`) that loads
raw session CSVs, applies block-ID synthesis and NaN-gap interpolation, and saves a
cleaned `<session>_prepared.csv` artifact. Wire it into the Prefect workflow and DAG
config as a `touch_preparation` task that gates both Stage 2a and Stage 2b.

**Why:** The guidelines at `docs/design/timeseries_clustering_pipeline_guidelines.md`
require Stage 1 to produce a distinct cleaned artifact. Currently Stage 1 operations
are embedded in `series_pipeline.py` (Stage 2a), and `extraction_pipeline.py` (Stage
2b) can silently receive un-interpolated raw CSVs when Stage 2a has not run.

**How:** Introduce `preparation_pipeline.py` as a thin Stage 1 driver that reuses
existing `preparation/` modules. Update Stage 2a and Stage 2b to consume the prepared
CSV. Move the misclassified `direction.py` (semantic inference, not data cleaning) into
`representation/series_level/`.

---

## Problem Statement

### D1 — Stage 1 has no standalone pipeline or artifact

`series_pipeline.py` (labeled "Stage 2a") runs Stage 1 prep as a preamble:

```python
df = load_session_csv(input_file)     # Stage 1: load
df = ensure_block_id_column(df)       # Stage 1: block metadata
df = interpolate_touch_columns(df)    # Stage 1: NaN imputation
groups = group_touches(df)            # Stage 1: grouping
# --- Stage 2a transforms below ---
vel, accel = get_kinematics(group, fps)
```

The cleaned intermediate is never saved to disk. Stage 1 is not independently runnable.
No `touch_preparation` task exists in the DAG or GUI.

### D2 — Stage 2b bypasses Stage 1 when Stage 2a is disabled

`extraction_pipeline.py` loads the Stage 2a augmented CSV when `series_dir` is
provided, otherwise falls back to the **raw session CSV** — NaN gaps intact in
`contact_depth`, `contact_area`, sticker position columns. `interpolate_touch_columns`
is never called on this path. Feature values derived from those columns are silently
corrupted.

### D3 — `preparation/direction.py` is misclassified as Stage 1 code

`infer_direction(group)` derives a categorical stroke label (proximal/distal/static)
from the movement trajectory. The guidelines state Stage 1 must NOT contain "anything
that changes the semantic content of a series." Direction inference is a Stage 2
operation, not data cleaning.

### D4 — Interpolation method is hardcoded, not configurable

`interpolate_touch_columns(df, method='cubic')` has a `method` param, but
`series_pipeline.py` always uses the default. No YAML key exposes it.

---

## Goals

### In Scope

1. Create `preparation_pipeline.py` with `run_preparation()` as the Stage 1 driver.
2. Add a `touch_preparation` Prefect flow and DAG task, gating both Stage 2a and 2b.
3. Update `series_pipeline.py` (Stage 2a) to read the prepared CSV when
   `preparation_dir` is provided; raise if the file is missing (fail-fast).
4. Update `extraction_pipeline.py` (Stage 2b) to read the prepared CSV when
   `preparation_dir` is provided; fall back to inline interpolation on raw CSV only as
   a last resort.
5. Expose `interpolation.method` as a YAML config key under `touch_preparation`.
6. Move `preparation/direction.py` to `representation/series_level/direction.py` and
   update its import in `extraction_pipeline.py`.

### Out of Scope

- Adding new Stage 1 operations (outlier handling, denoising/smoothing) — out of scope
  for this research dataset.
- GUI widgets for `touch_preparation` configuration — the task renders with the
  existing scalar-section widgets automatically; no custom dialog needed.
- Changes to Stage 3 (reduction), Stage 4 (clustering), or Stage 5 (evaluation).
- Wiring `preparation/direction.py` into any Stage 2a transform — only the import
  path changes.

---

## Success Criteria

- [ ] Running `touch_preparation` on a session produces
  `4_analysed/preparation/<session_id>_prepared.csv` with no NaN gaps in
  `contact_depth`, `sticker_*_position_*`, and `contact_area` columns within each
  touch group.
- [ ] Running `touch_series_transforms` after `touch_preparation` reads the prepared
  CSV and produces `_series_augmented.csv` with `velocity_magnitude` and
  `acceleration_magnitude` columns.
- [ ] Running `touch_feature_extraction` with `touch_series_transforms` disabled reads
  the prepared CSV (not the raw CSV) and computes features on interpolated data.
- [ ] Running `touch_feature_extraction` with both stages above disabled falls back to
  raw CSV + inline interpolation; features match the prepared-CSV path numerically.
- [ ] Setting `preparation.interpolation.method: linear` in the YAML changes the
  interpolation method applied by `touch_preparation`.
- [ ] `infer_direction` is importable from `representation.series_level.direction`;
  old `preparation.direction` path no longer exists.
- [ ] No regression in `touch_clustering`, `touch_comparing`, or
  `map_receptive_fields_clustered` outputs.

---

## Technical Design

### Approach

`preparation_pipeline.py` is a thin Stage 1 driver that reuses the three existing
`preparation/` modules (`loader`, `block_id`, `interpolation`). The output is a CSV
with the same columns as the raw session CSV — no new columns, no semantic transforms.

Both Stage 2a and Stage 2b accept an optional `preparation_dir` parameter. When
provided and the prepared CSV exists, they skip re-running Stage 1. When not provided,
Stage 2a falls back to its current inline preamble (backward compat); Stage 2b falls
back to loading raw CSV + inline interpolation.

**Fail-fast rule for Stage 2a:** if `preparation_dir` is given but the prepared CSV is
not found, raise `FileNotFoundError` immediately. Stage 1 must have run successfully
before Stage 2a can proceed — no silent fallback to raw CSV.

**DAG dependency chain after fix:**
```
touch_preparation (Stage 1)       → 4_analysed/preparation/<session>_prepared.csv
  └── touch_series_transforms (Stage 2a) → 4_analysed/series_transforms/<session>_series_augmented.csv
        └── touch_feature_extraction (Stage 2b) → 4_analysed/extraction/<feature>/<session>_touch_summary.csv
              └── touch_clustering → ...
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Full separation — standalone Stage 1 artifact** (chosen) | Clean stage boundary; Stage 2b always receives interpolated data; Stage 1 independently runnable and inspectable; matches guidelines | Extra CSV per session (~same size as raw session CSV) | **Chosen** |
| Minimal fix — embed Stage 1 in Stage 2a, fix Stage 2b bypass inline | No new task; no new artifact | Stage 1 still invisible as a distinct stage; discrepancy vs guidelines persists | Rejected |
| Rename `touch_series_transforms` to `touch_preparation_and_series_transforms` | Near-zero code change | Stage 1 still not independently runnable; D2 (Stage 2b bypass) unresolved | Rejected |

### Architecture Constraints (from knowledge base)

- **CuPy import order** (`note-cupy-import-order.md`): N/A — `preparation_pipeline.py`
  imports only from `preparation/` and `utils/`, no CuPy dependency.
- **Qt `itemChanged` signal recursion** (`note-qt-itemchanged-signal-recursion.md`):
  N/A — no GUI widgets added for Stage 1.
- No other knowledge-base notes apply.

### Architecture Changes

```
code/src/analysis/touch_analytics/
├── preparation_pipeline.py                          # NEW — Stage 1 driver
├── series_pipeline.py                               # MODIFIED — add preparation_dir param
├── extraction_pipeline.py                           # MODIFIED — add preparation_dir, fix fallback
├── __init__.py                                      # MODIFIED — export run_preparation
├── preparation/
│   ├── loader.py                                    # unchanged
│   ├── block_id.py                                  # unchanged
│   ├── grouping.py                                  # unchanged
│   ├── interpolation.py                             # unchanged
│   └── direction.py                                 # MOVED (see below)
└── representation/
    └── series_level/
        └── direction.py                             # MOVED from preparation/direction.py

code/scripts/
└── analysis_workflow.py                             # MODIFIED — add flow + dispatcher branch

configs/
└── analyse_workflow_dag.yaml                        # MODIFIED — add task, update depends_on
```

---

## Implementation Plan

### Phase 1 — Create `preparation_pipeline.py`

**Goal:** Standalone Stage 1 driver that loads, block-IDs, interpolates, and saves
the cleaned CSV artifact.

- [x] Task 1.1 — Create `preparation_pipeline.py` with
  `run_preparation(input_items, preparation_cfg, output_dir, force) -> List[Path]`.
  Per session: idempotency via `should_process_task`; load via
  `preparation/loader.load_session_csv`; block-id via
  `preparation/block_id.ensure_block_id_column`; interpolate via
  `preparation/interpolation.interpolate_touch_columns(df, method=...)` where
  `method` comes from `preparation_cfg.get('interpolation', {}).get('method', 'cubic')`;
  save to `output_dir/<session_id>_prepared.csv`. Raise on load or save failure.
- [x] Task 1.2 — Export `run_preparation` from `touch_analytics/__init__.py`.

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — NEW
- `code/src/analysis/touch_analytics/__init__.py` — add `run_preparation` export

**Dependencies:** None

---

### Phase 2 — Update Stage 2a (`series_pipeline.py`) to read prepared CSV

**Goal:** Stage 2a accepts an optional `preparation_dir`; when provided it reads the
prepared CSV and skips inline Stage 1 preamble.

- [x] Task 2.1 — Add `preparation_dir: Path = None` to `run_series_transforms` and
  `_transform_session` signatures.
- [x] Task 2.2 — In `_transform_session`: if `preparation_dir` is provided, look for
  `preparation_dir/<session_id>_prepared.csv`. If found, load it (already
  interpolated — skip `interpolate_touch_columns`). If not found, raise
  `FileNotFoundError` with a clear message.
- [x] Task 2.3 — If `preparation_dir` is None, keep the current inline preamble
  (load raw CSV, block-id, interpolate) for backward compatibility when Stage 2a
  is run standalone.

**Files Modified:**
- `code/src/analysis/touch_analytics/series_pipeline.py` — add param, branch load logic

**Dependencies:** Phase 1

---

### Phase 3 — Update Stage 2b (`extraction_pipeline.py`) to read prepared CSV

**Goal:** Stage 2b always receives interpolated data: augmented CSV → prepared CSV →
raw CSV + inline interpolation (last resort only).

- [x] Task 3.1 — Add `preparation_dir: Path = None` parameter to
  `run_feature_extraction` and `_extract_session`. Also added `series_dir: Path = None`
  (required by the fallback chain in Task 3.2).
- [x] Task 3.2 — In `_extract_session`, update the fallback chain:
  1. If `series_dir` provided and `<session>_series_augmented.csv` exists → load it
     (Stage 2a output; already prepared + transformed).
  2. Else if `preparation_dir` provided and `<session>_prepared.csv` exists → load it
     (Stage 1 output; already interpolated).
  3. Else → load raw CSV and call `interpolate_touch_columns(df)` inline (last resort;
     warn via `logging.warning`).
- [x] Task 3.3 — Pass `preparation_dir` through from `run_feature_extraction` to
  `_extract_session`.

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — add param, updated
  fallback chain

**Dependencies:** Phase 1

---

### Phase 4 — Wire into `analysis_workflow.py` and DAG YAML

**Goal:** `touch_preparation` appears as a distinct Prefect flow and DAG task,
gating Stage 2a and 2b.

- [x] Task 4.1 — Add `touch_preparation_flow` to `analysis_workflow.py`.
- [x] Task 4.2 — Add `touch_series_transforms_flow` (was missing) and wire `preparation_dir`
  forwarding to `run_series_transforms`.
- [x] Task 4.3 — Update `touch_feature_extraction_flow`: accept `series_dir` and
  `preparation_dir`, forward to `run_feature_extraction`.
- [x] Task 4.4 — In `run_batch_analysis`: add `"touch_preparation"` and
  `"touch_series_transforms"` to `available_tasks`; compute `preparation_dir` and
  `series_dir` after items are collected; forward to downstream flows.
- [x] Task 4.5 — Add `run_preparation` and `run_series_transforms` imports to
  `analysis_workflow.py`.
- [x] Task 4.6 — Add `touch_preparation` task to `analyse_workflow_dag.yaml`.
- [x] Task 4.7 — Update `touch_series_transforms.depends_on: [touch_preparation]`.

**Files Modified:**
- `code/scripts/analysis_workflow.py` — new flow, updated flows, dispatcher
- `configs/analyse_workflow_dag.yaml` — new task, updated depends_on

**Dependencies:** Phases 2 and 3

---

### Phase 5 — Move `direction.py` out of `preparation/`

**Goal:** Fix the semantic misclassification of stroke-direction inference (D3).

- [x] Task 5.1 — Move
  `preparation/direction.py` → `representation/series_level/direction.py`.
  Content is unchanged; only the file location changes.
- [x] Task 5.2 — Update the import in `extraction_pipeline.py` and also in
  `representation/feature_characterization/touch_category.py` (additional consumer
  found by grep).
- [x] Task 5.3 — Delete `preparation/direction.py`.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/direction.py` — NEW
  (content copied from preparation/direction.py)
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — update import
- `code/src/analysis/touch_analytics/preparation/direction.py` — DELETE

**Dependencies:** None (independent of Phases 1–4)

---

## Testing Plan

### Unit Tests

- [ ] `test_run_preparation_creates_prepared_csv` — call `run_preparation` on a
  synthetic session CSV with NaN gaps in `contact_depth`. Assert the prepared CSV
  exists and `contact_depth` has no NaN within touch groups.
- [ ] `test_run_preparation_idempotent` — run twice without force. Assert the second
  run skips processing (same output mtime).
- [ ] `test_run_preparation_interpolation_method_forwarded` — call with
  `preparation={'interpolation': {'method': 'linear'}}`. Assert the interpolated values
  differ from cubic (they will for a curved signal).
- [ ] `test_series_pipeline_reads_prepared_csv` — create a fake prepared CSV (already
  interpolated); call `run_series_transforms` with `preparation_dir` pointing to it.
  Assert `_transform_session` does not call `interpolate_touch_columns`.
- [ ] `test_series_pipeline_raises_if_prepared_missing` — call `run_series_transforms`
  with `preparation_dir` set but no prepared CSV present. Assert `FileNotFoundError`.
- [ ] `test_extraction_fallback_to_prepared_csv` — call `run_feature_extraction` with
  no `series_dir` but `preparation_dir` pointing to a prepared CSV. Assert it loads the
  prepared CSV (not raw).
- [ ] `test_extraction_inline_fallback_interpolates` — call `run_feature_extraction`
  with neither `series_dir` nor `preparation_dir`. Assert `interpolate_touch_columns`
  is called on the raw CSV.
- [ ] `test_infer_direction_importable_from_new_path` — `from
  touch_analytics.representation.series_level.direction import infer_direction` does not
  raise.

### Integration Tests

- [ ] `test_stage1_then_stage2a_then_stage2b` — run all three in sequence on a test
  session. Assert per-feature CSV values match a reference run with the old inline
  preamble.
- [ ] `test_stage2b_standalone_matches_full_pipeline` — run Stage 2b alone (no Stage
  1/2a). Assert per-feature CSVs are numerically equal to the full-pipeline run.

### Manual Verification

- [ ] Run `touch_preparation` from the GUI on a single session. Inspect
  `_prepared.csv` — confirm no NaN gaps in `contact_depth` within any touch group.
- [ ] Run `touch_series_transforms` after. Confirm `_series_augmented.csv` has
  `velocity_magnitude` and `acceleration_magnitude`.
- [ ] Disable `touch_preparation` and `touch_series_transforms`; run
  `touch_feature_extraction` alone. Confirm it completes and features are not
  obviously wrong (same range as full-pipeline run).
- [ ] Change `preparation.interpolation.method: linear` in the YAML; run
  `touch_preparation`. Inspect `_prepared.csv` — confirm interpolation changed
  (check a short touch group with few sticker-position values).
- [ ] Run full pipeline through `touch_clustering`. Confirm cluster output unchanged.

### Edge Cases

- [ ] Session CSV where all sticker columns are NaN for a touch group — Stage 1 leaves
  those NaN (0 non-NaN branch in `_interpolate_group`); Stage 2a logs a warning.
- [ ] `force_processing: false` and prepared CSV already exists but raw CSV is newer
  — idempotency check triggers reprocessing.
- [ ] `touch_preparation` disabled in DAG YAML but `touch_series_transforms` enabled
  — Stage 2a runs without `preparation_dir`; falls back to inline Stage 1 preamble.

---

## Documentation Plan

- [ ] Docstring on `run_preparation` explaining Stage 1 responsibility and output path.
- [ ] No changes to `CLAUDE.md` or `README.md` — internal pipeline change, no
  architectural constraint added.

---

## Rollback Plan

1. Revert commits from this plan. All new params default to `None`, so callers are
   unaffected.
2. Delete `4_analysed/preparation/` directories if any prepared CSVs were written.
3. No data migration — Stage 2b output format is unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Prepared CSV doubles per-session disk usage | High | Low | Same trade-off accepted for `_series_augmented.csv`; disk is cheap; Stage 1 artifact is the same width as raw CSV plus no new columns |
| Stage 2a fails fast if `preparation_dir` set but prepared CSV missing — breaks incremental runs where only Stage 2a re-runs | Medium | Medium | Re-enable `touch_preparation` when re-running Stage 2a, or pass `preparation_dir=None` to fall back to inline preamble |
| `infer_direction` consumers other than `extraction_pipeline.py` break on import path change | Low | Medium | Grep for `from .preparation.direction` and `from preparation.direction` before Phase 5 |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Stage 1 driver | 0.5 day | None |
| Phase 2 — Stage 2a reads prepared CSV | 0.25 day | Phase 1 |
| Phase 3 — Stage 2b reads prepared CSV | 0.25 day | Phase 1 |
| Phase 4 — Workflow + DAG integration | 0.5 day | Phases 2, 3 |
| Phase 5 — Move direction.py | 0.1 day | None |

---

## References

- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md` — Stage 1 definition
- **Parent alignment plan:**
  `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md`
- **Stage 2a/2b split plan:**
  `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md`
- **Config forwarding fix (companion):**
  `docs/development/plans/pending/fix-analysis-workflow-stage-config-forwarding.md`
- **Stage 1 modules (ready to reuse):**
  - `code/src/analysis/touch_analytics/preparation/loader.py` — `load_session_csv`
  - `code/src/analysis/touch_analytics/preparation/block_id.py` — `ensure_block_id_column`
  - `code/src/analysis/touch_analytics/preparation/interpolation.py` — `interpolate_touch_columns`
- **Stage 2a driver (pattern to follow):**
  `code/src/analysis/touch_analytics/series_pipeline.py`
- **Idempotency utility:** `code/src/utils/should_process_task.py` — `should_process_task`
