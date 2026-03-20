# Plan: Stale Output Cleanup

**Date:** 2026-03-19
**Author:** Basil Duvernoy
**Status:** Pending — under reconsideration
**Branch:** _(none — previous branch rolled back 2026-03-20)_

---

## Overview

When a pipeline task fails after upstream inputs have changed, its old outputs from a previous successful run remain on disk. Downstream tasks silently consume these stale files, producing incorrect results. This plan introduces a pre-run cleanup mechanism that deletes pure output files before task execution, so that failures leave no stale artifacts for downstream consumption.

## Problem Statement

The pipeline processes data in sequential stages (A -> B -> C). When Task A's outputs change and Task B is re-run but fails on some sessions, Task B's old outputs from a previous successful run remain on disk. Task C — or other downstream pipelines that discover inputs via directory globbing — silently picks up these stale outputs as if they were current.

This is particularly dangerous because:
- The analysis pipeline discovers inputs via `glob("*_touch_summary.csv")`, finding ALL files including stale ones from sessions where upstream failed
- The pipeline runs per-block/per-session in batch — some blocks may succeed while others fail, leaving a mix of current and stale outputs
- There is no warning or error — the pipeline appears to succeed

**Constraint — living metadata files:** Some tasks use metadata files that serve as both input AND output (e.g., `metadata_path` in `track_objects_in_video`, `roi_manual_annotation.json`). These files accumulate state across runs and cannot be blindly deleted before a task runs.

## Goals

### In Scope
1. Prevent downstream tasks from consuming stale outputs when an upstream task fails
2. Protect living metadata files from deletion (ROI annotations, curation metadata)
3. Apply to all four pipeline types: preprocessing, postprocessing, merging, and analysis

### Out of Scope
- Manifest/registry system for tracking pipeline run provenance
- Sentinel/lock file mechanism for cross-pipeline failure signaling
- Changes to `should_process_task.py` or `TaskExecutor`
- Changes to individual task function signatures

## Success Criteria

- [ ] When a stage-loop task fails, its pure output files are absent (not stale)
- [ ] When an analysis sub-pipeline task fails for a session, its output CSV is absent
- [ ] Living metadata files (e.g., `roi_manual_annotation.json`) are never deleted
- [ ] Happy-path execution produces identical results to current behavior
- [ ] `should_process_task()` correctly triggers reprocessing when outputs are missing after cleanup

---

## Technical Design

### Approach

**Pre-run cleanup with explicit pure output declaration.** Each pipeline stage declares which output files are "pure outputs" (written fresh each run, safe to delete). Before the task function executes, these files are deleted. If the task succeeds, it recreates them. If it fails, no stale files remain.

This works because:
- `should_process_task()` already handles "output missing" → returns `True` (processing required)
- Downstream tasks that glob directories will simply not find the file
- Living metadata files are never declared as pure outputs, so they are preserved

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pre-run cleanup of pure outputs | Minimal changes, no new abstractions, works with existing `should_process_task` | Requires enumerating outputs per stage | **Chosen** |
| Failure sentinel files | No file deletion needed | Requires downstream awareness of sentinels, couples tasks | Rejected |
| Output quarantine (move to `.quarantine/`) | Preserves files for recovery | Complex state machine, quarantine dir management | Rejected |
| Manifest system | Full provenance tracking | Over-engineered for this codebase | Rejected |

### Architecture Changes

No new modules or classes. Changes are localized to:

1. **Stage-loop pipelines** (preprocessing, postprocessing): A 5-line cleanup block is added to the shared stage-loop pattern, and each stage definition gains an optional `"pure_outputs"` lambda.

2. **Analysis sub-pipelines** (extraction, clustering, comparing): Each sub-pipeline deletes its target output file before attempting to write it.

3. **Merging pipeline**: Pre-write deletion at the output site (direct task calls, no stage-loop).

### Distinguishing Pure Outputs from Living Metadata

| Category | Definition | Examples | Action |
|----------|-----------|----------|--------|
| Pure output | Written fresh from scratch each run | CSVs, PLY files, XYZ metadata JSON | Listed in `pure_outputs`, deleted before run |
| Living metadata | Read at task start, updated incrementally, written back | `roi_manual_annotation.json`, `curation_metadata.json` | **Not listed**, never deleted |

During implementation, each output file will be verified by checking whether the task reads it before writing.

---

## Implementation Plan

### Phase 1: Stage-Loop Cleanup Infrastructure
**Goal:** Add the generic pre-run cleanup mechanism to the two stage-loop pipelines.

**Tasks:**
- [ ] Task 1.1 — Add `pure_outputs` cleanup block to the stage loop in `preprocess_workflow_kinect_auto.py` (after `executor.can_run` check, before `stage["func"](**params)`)
- [ ] Task 1.2 — Add `pure_outputs` cleanup block to the stage loop in `postprocess_workflow_kinect_auto.py` (same pattern)
- [ ] Task 1.3 — Add `"pure_outputs"` lambdas to all preprocessing stage definitions (excluding living metadata files)
- [ ] Task 1.4 — Add `"pure_outputs"` lambdas to all postprocessing stage definitions

**Files Modified:**
- `code/scripts/preprocess_workflow_kinect_auto.py` — cleanup block in stage loop + `pure_outputs` on all stages
- `code/scripts/postprocess_workflow_kinect_auto.py` — cleanup block in stage loop + `pure_outputs` on all stages

**Dependencies:** None

### Phase 2: Analysis Pipeline Cleanup
**Goal:** Add pre-write output deletion to the analysis sub-pipelines that use glob-based discovery.

**Tasks:**
- [ ] Task 2.1 — In `extraction_pipeline.py::_extract_session`, delete `output_path` before the extraction attempt (after idempotency check decides to proceed)
- [ ] Task 2.2 — In `clustering_pipeline.py`, delete `pooled_touch_summary_clustered.csv` and `cluster_metadata.json` before writing
- [ ] Task 2.3 — In `comparing_pipeline.py`, delete comparison result files before writing

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — pre-write deletion in `_extract_session`
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — pre-write deletion before output
- `code/src/analysis/touch_analytics/comparing_pipeline.py` — pre-write deletion before output

**Dependencies:** None (independent of Phase 1)

### Phase 3: Merging Pipeline Cleanup
**Goal:** Add pre-write output deletion to the merging pipeline's direct task calls.

**Tasks:**
- [ ] Task 3.1 — In `merging_pipeline_neuron_to_kinect_auto.py::run_single_session_pipeline`, delete `output_file_path` before `unify_dataset()` call
- [ ] Task 3.2 — Delete `filtered_output` before `filter_by_neural_quality_flow()` call

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — pre-write deletion before each task

**Dependencies:** None (independent of Phase 1 and 2)

---

## Testing Plan

### Manual Verification
- [ ] **Failure cleanup:** Run preprocessing on a session, force a middle task to fail (e.g., via bad input) → confirm its pure output files are absent after the run
- [ ] **Analysis cleanup:** Run extraction on a session that will fail → confirm `_touch_summary.csv` for that session is removed → confirm clustering skips it
- [ ] **Living metadata preservation:** Run `track_stickers_raw` with existing `roi_manual_annotation.json` → confirm the annotation file is intact after the run
- [ ] **Happy path:** Run full pipeline on a session that succeeds → confirm all outputs are produced correctly (no regression)
- [ ] **Idempotency:** Run the same successful pipeline twice → confirm `should_process_task` correctly skips up-to-date tasks (outputs exist, mtimes are valid)

### Edge Cases
- [ ] Task that produces no outputs (e.g., `validate_forearm_extraction` with empty `outputs: []`) — cleanup block should be a no-op
- [ ] Stage with `pure_outputs` lambda that returns an empty list — no error, no action
- [ ] `force_processing: true` combined with cleanup — outputs are deleted then recreated regardless of staleness

---

## Documentation Plan

- [ ] Add knowledge base note explaining the pattern, when to use `pure_outputs`, and the living-metadata distinction

---

## Rollback Plan

All changes are additive (new `pure_outputs` keys on stage dicts, new `unlink()` calls before writes). To rollback:

1. Remove the cleanup block from the two stage loops
2. Remove `"pure_outputs"` keys from stage definitions
3. Revert the pre-write `unlink()` calls in analysis and merging pipelines

No data migrations or breaking changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A living metadata file is accidentally listed in `pure_outputs` | Low | High — loses accumulated tracking state | Verify each file's role during implementation by checking if the task reads it before writing |
| `pure_outputs` lambda raises an exception (e.g., context key not yet set) | Low | Med — task crashes before running | Use defensive `lambda` with fallback to empty list; test with context state at each stage |
| Output file is in use by another process during deletion | Low | Low — `unlink()` fails with PermissionError | Wrap in try/except, log warning, let task proceed (stale output is better than crash) |
