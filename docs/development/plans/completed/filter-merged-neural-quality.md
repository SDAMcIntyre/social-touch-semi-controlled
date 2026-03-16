# Plan: Filter Merged Data by Neural Quality

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/filter-merged-neural-quality`

---

## Overview

Add a filtering step to the merging pipeline that removes trials marked "Not2Use" in an external xlsx metadata file. Produces filtered per-block CSVs in a `sessions_filtered/` subfolder and a filtered aggregated session CSV. This keeps original merged data untouched while providing clean datasets for downstream analysis.

## Problem Statement

The merging pipeline includes all trials regardless of neural recording quality. An experimenter-maintained xlsx file annotates which trials are unusable ("Not2Use"), but there is no automated way to exclude them. This pollutes downstream analysis and requires manual post-hoc filtering every time the data is re-processed.

## Goals

### In Scope

1. Parse xlsx metadata to identify Not2Use trial IDs per unit/block
2. Filter per-block merged CSVs, writing results to `sessions_filtered/`
3. Re-aggregate filtered blocks into `{session_id}_semicontrolled_aggregated_session_filtered.csv`
4. Integrate as a DAG task with `depends_on: [unify_dataset]`

### Out of Scope

- Modifying the original merged CSVs (read-only)
- GUI for editing quality annotations
- Filtering at the preprocessing stage (this operates on merged data only)

## Success Criteria

- [ ] Per-block filtered CSVs in `sessions_filtered/` exclude all rows belonging to Not2Use trials
- [ ] Filtered aggregated CSV produced with `_filtered` suffix
- [ ] `filter_by_neural_quality` task in DAG YAML with `depends_on: [unify_dataset]`
- [ ] Blocks with no Not2Use trials are copied as-is to `sessions_filtered/`
- [ ] NaN `trial_id` rows (nerve-rate interpolation) correctly removed when belonging to a Not2Use trial

---

## Technical Design

### Approach

1. **Parse xlsx** into lookup: `{(unit, block_order): set_of_not2use_trial_ids}`
2. **Per block:** extract unit + block_order from filename, look up Not2Use trials, forward-fill `trial_id` to assign NaN rows, filter, write to `sessions_filtered/`
3. **Aggregate:** reuse existing `aggregate_session_blocks()` on `sessions_filtered/` folder

**Handling NaN `trial_id`:** The merged CSV has `trial_id` only on Kinect-rate rows (30 Hz). Nerve-rate rows (1000 Hz) have NaN. Forward-fill assigns each NaN row to its preceding trial. `trial_id == 0` (inter-trial gaps) are always kept. NaN rows before the first Kinect frame get `fillna(0)` so they are also kept.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Filter per-block then re-aggregate | Clean separation, reuses aggregation, per-block filtered files for debugging | Two-step process | **Chosen** |
| Filter only at aggregated level | Single step | Loses per-block filtered files; harder to debug | Rejected |
| Add filter column instead of removing rows | Non-destructive | Downstream tools would need to check filter column everywhere | Rejected |

### Architecture Changes

No new classes or modules beyond a single new file. Changes are confined to the merging pipeline:

```
code/scripts/_4_merging/
  filter_merged_by_neural_quality.py   # NEW - core filtering logic (~80 lines)
  __init__.py                          # Add export
  aggregate_blocks_session.py          # Add optional params (backward-compatible)

code/scripts/
  merging_pipeline_neuron_to_kinect_auto.py  # Add filter flow + wire in

configs/
  merging_pipeline_neuron_to_kinect_auto_dag.yaml  # Add param + task
```

**Knowledge base check:** No applicable notes. The feature does not involve Open3D widgets, CuPy import order, ICP registration, or somatosensory metrics.

---

## Implementation Plan

### Phase 1: Core Filtering Module

**Goal:** Pure data-processing functions, no pipeline coupling.

- [ ] Task 1.1 - Create `filter_merged_by_neural_quality.py` with three functions:
  - `parse_neural_quality_xlsx(xlsx_path) -> Dict[Tuple[str, int], Set[int]]`
  - `filter_block_by_neural_quality(input_csv, output_csv, not2use_trials, *, force_processing) -> Path`
  - `extract_unit_and_block_order(filename) -> Tuple[str, int]`
- [ ] Task 1.2 - Export from `_4_merging/__init__.py`

**Files Modified:**

- `code/scripts/_4_merging/filter_merged_by_neural_quality.py` - NEW (~80 lines)
- `code/scripts/_4_merging/__init__.py` - Add import

**Key logic:**

- `parse_neural_quality_xlsx`: Read xlsx with `pd.read_excel()`. Iterate rows, check columns "1"-"12" for "Not2Use" substring. Build `{(unit, block_order): {trial_ids}}` dict.
- `filter_block_by_neural_quality`: Use `should_process_task()` for idempotency. Read CSV, forward-fill trial_id (`df['trial_id'].ffill().fillna(0)`), drop rows where filled trial_id is in not2use set, write output.
- `extract_unit_and_block_order`: Regex `r'(ST\d+-\d+)'` for unit, `r'block-order-(\d+)'` for block order (convert to int, strips leading zero).

**Dependencies:** None

### Phase 2: Pipeline Integration

**Goal:** Wire into DAG and pipeline script.

- [ ] Task 2.1 - Add `neural_quality_xlsx` parameter to DAG YAML
- [ ] Task 2.2 - Add `filter_by_neural_quality` task definition (`depends_on: [unify_dataset]`)
- [ ] Task 2.3 - Add `@flow` function `filter_by_neural_quality_flow` in pipeline script
- [ ] Task 2.4 - Wire into `run_single_session_pipeline()` after `unify_dataset` block
- [ ] Task 2.5 - Add filtered aggregation in `run_batch_processing()` - call `aggregate_blocks()` with `input_subfolder="sessions_filtered"` and `output_suffix="_filtered"`
- [ ] Task 2.6 - Add `input_subfolder` and `output_suffix` optional params to `aggregate_blocks()` flow (backward-compatible defaults)

**Files Modified:**

- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` - Add parameter + task
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` - Add flow, wire into pipeline, update `aggregate_blocks` call

**DAG YAML additions:**

```yaml
parameters:
  neural_quality_xlsx: "path/to/neural_quality.xlsx"

tasks:
  filter_by_neural_quality:
    enabled: true
    options:
      force_processing: false
    depends_on: [unify_dataset]
```

**Pipeline flow (in `run_single_session_pipeline`):**

- Read xlsx path from `dag_handler.get_parameter('neural_quality_xlsx')`
- Resolve relative to project data root
- Parse xlsx once, extract unit + block_order from config
- Look up Not2Use trials for this block
- If Not2Use trials exist: filter and write to `sessions_filtered/`
- If no Not2Use trials: copy merged CSV as-is to `sessions_filtered/`
- Mark task completed

**Filtered aggregation (in `run_batch_processing`):**

- After existing aggregation loop, add second loop for filtered aggregation
- Read from `sessions_filtered/`, write `_aggregated_session_filtered.csv`

**Dependencies:** Phase 1

### Phase 3: Edge Cases

**Goal:** Harden against missing/malformed inputs.

- [ ] Task 3.1 - No xlsx path configured or file missing: log warning, skip task
- [ ] Task 3.2 - Block has 0 Not2Use trials: copy as-is to `sessions_filtered/`
- [ ] Task 3.3 - ALL trials Not2Use: write header-only CSV
- [ ] Task 3.4 - Empty/NaN xlsx cells: treat as valid (not Not2Use)
- [ ] Task 3.5 - `trial_id` column missing from merged CSV: log warning, copy as-is
- [ ] Task 3.6 - Not2Use trial_id not found in CSV data: log warning, no crash

**Files Modified:**

- `code/scripts/_4_merging/filter_merged_by_neural_quality.py` - Add guards and logging

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Run pipeline on a known block with Not2Use trials; compare row counts between `sessions/` and `sessions_filtered/`
- [ ] Verify no rows with a Not2Use `trial_id` remain in filtered output
- [ ] Verify filtered aggregated CSV exists and has `_filtered` suffix
- [ ] Verify blocks with no Not2Use trials are copied identically

### Edge Cases

- [ ] Block with all 12 trials marked Not2Use produces header-only CSV
- [ ] xlsx with variable trial counts (3-12) parsed correctly
- [ ] "Not2Use" detected as substring (e.g., "Not2Use - noisy")

---

## Documentation Plan

- [ ] Add inline docstrings to the three public functions in `filter_merged_by_neural_quality.py`
- [ ] Document `neural_quality_xlsx` parameter in DAG YAML with a comment

---

## Rollback Plan

All outputs go to new paths (`sessions_filtered/`, `*_filtered.csv`). No existing data is modified.

1. Remove `filter_by_neural_quality` task + `neural_quality_xlsx` param from DAG YAML
2. Revert pipeline script changes
3. Delete `sessions_filtered/` directories if desired

No data migrations, no schema changes. `git revert` of the merge commit is sufficient.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| xlsx column names differ from expected | Medium | High | Validate columns on parse, clear error message listing expected vs found |
| Forward-fill misassigns rows at trial boundaries | Low | Medium | `trial_id=0` rows (inter-trial) never filtered; only in-trial NaN rows get ffilled |
| `aggregate_blocks` signature change breaks callers | Low | Medium | New params have backward-compatible defaults (`input_subfolder=None`, `output_suffix=""`) |

---

## References

- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` - pipeline script
- `code/scripts/_4_merging/aggregate_blocks_session.py` - aggregation logic
- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` - DAG config
- `code/src/utils/pipeline/pipeline_config_manager.py` - `DagConfigHandler`
