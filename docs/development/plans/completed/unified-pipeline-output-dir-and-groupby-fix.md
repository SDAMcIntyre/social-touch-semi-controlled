# Plan: Unified Pipeline — Output Dir, Groupby, and Cleanup Fixes

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-18 11:29
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

**What:** Fix three issues in the unified touch analysis pipeline: the output directory should be defined in the Prefect flow and passed down, the groupby must include `block_order_id` to uniquely identify touches, and the backward-compatibility dual-write should be removed.

**Why:** The output directory is currently hardcoded deep inside `unified_pipeline.py`, making it invisible to the flow that orchestrates it. The groupby on `['trial_id', 'single_touch_id']` alone is incorrect because input files are aggregated blocks of one session — `trial_id` and `single_touch_id` can repeat across blocks. The backward-compat dual-write to the root `unified_touches/` is no longer needed.

**How:** Add an `output_dir` parameter to `run_unified_touch_analysis` and its internal helpers, compute it in the flow, fix the groupby keys, reorder columns to put `block_order_id` first, and delete the dual-write block.

## Problem Statement

1. **Output dir not controlled by flow:** `_extract_session` and `_cluster_profile` each construct `database_path / '4_analysed' / 'unified_touches'` internally. The flow has no visibility or control over where output goes.
2. **Incorrect groupby:** `_extract_all_touches` groups by `['trial_id', 'single_touch_id']` (line 230), but since the aggregated session CSV contains multiple blocks, these two columns are not unique. `block_order_id` must be part of the groupby.
3. **Column ordering:** `block_order_id` should appear before `trial_id` and `single_touch_id` in output CSVs to reflect the logical hierarchy (block > trial > touch).
4. **Stale backward-compat:** The dual-write of the `max` profile to root `unified_touches/` (lines 211-218) is no longer needed and creates maintenance overhead.

## Goals

### In Scope

1. Define `output_dir` in `unified_touch_analysis_flow` and pass it through to all pipeline functions
2. Fix groupby to `['block_order_id', 'trial_id', 'single_touch_id']`
3. Reorder `_SHARED_COLUMNS` and `shared` dict so `block_order_id` comes first
4. Remove backward-compat dual-write block for the `max` profile

### Out of Scope

- Updating downstream tasks (`analyse_ap_efficacy`, `map_receptive_fields`) to use profile subdirectories — separate concern
- Changes to extraction or clustering logic
- Changes to DAG YAML config

## Success Criteria

- [ ] `unified_touch_analysis_flow` constructs `output_dir` and passes it to `run_unified_touch_analysis`
- [ ] `unified_pipeline.py` has no hardcoded `'4_analysed' / 'unified_touches'` path construction
- [ ] `_extract_all_touches` groups by `['block_order_id', 'trial_id', 'single_touch_id']`
- [ ] Output CSV columns are ordered: `block_order_id, trial_id, single_touch_id, type_metadata, direction, ...`
- [ ] No dual-write to root `unified_touches/` for the `max` profile
- [ ] Pipeline runs end-to-end without errors

---

## Technical Design

### Approach

Straightforward parameter threading: the flow computes a single `output_dir` from the anchor `database_path` and passes it down. Internal functions stop constructing paths from `database_path` and use the provided `output_dir` directly. The groupby fix adds `block_order_id` to the existing groupby tuple and reads it from the group key instead of `group['block_order_id'].iloc[0]`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pass `output_dir` from flow | Flow controls output location, single source of truth | Requires signature changes to 3 functions | **Chosen** |
| Keep constructing internally, add override param | Backward compatible | Still hides default behavior, more complex | Rejected |

### Architecture Changes

No new modules. Signature changes to existing functions only.

---

## Implementation Plan

### Phase 1: All fixes
**Goal:** Apply all four changes in one pass since they are tightly scoped and interdependent.

- [ ] Task 1.1 — `analysis_workflow.py`: In `unified_touch_analysis_flow`, compute `output_dir = input_items[0][1] / '4_analysed' / 'unified_touches'` and pass to `run_unified_touch_analysis`
- [ ] Task 1.2 — `unified_pipeline.py`: Add `output_dir: Path` parameter to `run_unified_touch_analysis`, pass it to `_extract_session` and `_cluster_profile`
- [ ] Task 1.3 — `unified_pipeline.py`: In `_extract_session`, replace `database_path: Path` with `output_dir: Path`, remove `unified_root = database_path / '4_analysed' / 'unified_touches'`, use `output_dir` directly
- [ ] Task 1.4 — `unified_pipeline.py`: In `_cluster_profile`, replace `database_path: Path` with `output_dir: Path`, use `output_dir / profile_name / 'clustering' / clusterer_name` directly
- [ ] Task 1.5 — `unified_pipeline.py`: In `_extract_all_touches`, change groupby to `['block_order_id', 'trial_id', 'single_touch_id']`, unpack group key tuple accordingly
- [ ] Task 1.6 — `unified_pipeline.py`: Reorder `_SHARED_COLUMNS` to `['block_order_id', 'trial_id', 'single_touch_id', ...]` and reorder `shared` dict in `_extract_all_touches` to match
- [ ] Task 1.7 — `unified_pipeline.py`: Delete backward-compat dual-write block (lines 211-218 in `_extract_session`)

**Files Modified:**
- `code/scripts/analysis_workflow.py` — `unified_touch_analysis_flow` computes and passes `output_dir`
- `code/src/analysis/touch_analytics/unified_pipeline.py` — Signature changes, groupby fix, column reorder, dual-write removal

**Dependencies:** None

---

## Testing Plan

### Manual Verification

- [ ] Run `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml` end-to-end
- [ ] Verify output CSVs exist under `unified_touches/<profile>/` (not at root)
- [ ] Open a per-session CSV and confirm columns are ordered `block_order_id, trial_id, single_touch_id, ...`
- [ ] Confirm no CSV files are written to root `unified_touches/` directory
- [ ] Grep `unified_pipeline.py` for `'4_analysed'` — should find zero matches

### Edge Cases

- [ ] Session with multiple blocks — verify touches from different blocks with same `trial_id`+`single_touch_id` are kept separate (not collapsed)

---

## Documentation Plan

- [ ] No external documentation changes needed (internal refactoring)

---

## Rollback Plan

1. Revert the commit — all changes are in 2 files
2. No data migration needed — output directory structure is unchanged (only the code path to get there changed)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream tasks expect root-level CSVs after dual-write removal | Medium | Medium | Those tasks are currently disabled; update them when re-enabled |
| `block_order_id` is `None` for some sessions (no `source_block_file` column) | Low | Low | groupby with `None` still works — groups by the null value |

---

## References

- Unified pipeline: `code/src/analysis/touch_analytics/unified_pipeline.py`
- Analysis workflow: `code/scripts/analysis_workflow.py`
- Parent plan: `docs/development/plans/pending/unified-touch-analysis-pipeline.md`
