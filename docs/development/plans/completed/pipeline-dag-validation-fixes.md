# Plan: Pipeline DAG Validation Fixes

**Date:** 2026-02-23
**Author:** Basil Duvernoy
**Status:** Completed
**Completion Date:** 2026-03-06
**Branch:** `feature/pipeline-dag-validation-fixes`

---

## Overview

The two preprocessing workflow scripts (`preprocess_workflow_kinect_auto.py` and
`preprocess_workflow_kinect_manual.py`) contain several validation and dependency
flaws that cause wasted computation, incorrect failure cascading, and fragile
manual-to-auto handoff checks. This plan addresses all seven identified defects
without changing the scientific logic of any processing step.

## Problem Statement

The auto and manual preprocessing pipelines form a human-in-the-loop pair: manual
steps produce annotations/metadata that auto steps consume via validation gates.
An audit revealed these flaws:

1. **One failure kills independent branches** — The auto pipeline marks ALL
   subsequent tasks as SKIPPED on any failure, even tasks on unrelated DAG branches.
2. **`validate_hand_extraction` runs after the computation it should gate** — The
   expensive 3D hand motion generation runs before hand model validity is checked.
3. **`find_single_touches` has a wrong DAG dependency** — Declares dependency on
   `generate_3d_hand_in_motion` but actually consumes `generate_xyz_stickers` output.
4. **Manual pipeline execution order contradicts DAG for colorspace steps** —
   `review_handstickers_color_threshold` is hardcoded before
   `prepare_stickers_colorspace`, but auto needs colorspace first.
5. **Manual DAG missing LED dependency for `define_trial_chunks`** — Uses LED data
   but does not depend on `prepare_led_tracking`.
6. **Validation gates are file-existence checks only** — Stale or corrupted files
   pass all gates.
7. **Manual pipeline lacks `TaskExecutor`** — Inconsistent error handling and no
   monitor status updates for skipped tasks.

## Goals

### In Scope

1. Fix failure-cascading logic so independent DAG branches continue when one fails
2. Reorder `validate_hand_extraction` to run before `generate_3d_hand_in_motion`
3. Correct `find_single_touches` DAG dependency from `generate_3d_hand_in_motion`
   to `generate_xyz_stickers`
4. Fix manual DAG ordering for colorspace/threshold steps
5. Add missing `prepare_led_tracking` dependency to `define_trial_chunks`
6. Add content-level validation to critical gates (not just file existence)
7. Adopt `TaskExecutor` in the manual pipeline for consistent error handling

### Out of Scope

- Parallelising the auto pipeline execution (true concurrent DAG scheduling)
- Changing any scientific processing logic (sticker tracking, hand models, etc.)
- Adding new pipeline stages or removing existing ones
- GPU acceleration or performance optimisation of individual tasks
- Restructuring the DAG config YAML schema

## Success Criteria

- [x] A failure in `track_led_blinking` no longer prevents `track_stickers_raw`
      (and other independent branches) from running
- [x] `validate_hand_extraction` executes before `generate_3d_hand_in_motion` in
      both code and DAG config
- [x] `find_single_touches.depends_on` lists `generate_xyz_stickers` instead of
      `generate_3d_hand_in_motion`
- [x] Manual DAG enforces `prepare_stickers_colorspace` before
      `review_handstickers_color_threshold`
- [x] `define_trial_chunks` in manual DAG depends on `prepare_led_tracking`
- [x] At least the top 3 validation gates check file size + basic schema (not just
      existence)
- [x] Manual pipeline uses `TaskExecutor` for all task dispatch
- [x] All existing tests still pass
- [x] A new knowledge-base note documents the DAG validation ordering pattern

---

## Technical Design

### Approach

Fix each flaw at its source — mostly YAML dependency declarations and the
error-cascading loop — rather than introducing a new orchestration framework.
The pipeline's flat-list execution model is kept, but the failure handler is
made DAG-aware so it only skips true dependents of the failed task.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Patch existing loop + YAML (chosen)** | Minimal churn, preserves Prefect compatibility, reviewable diffs | Still sequential, no true parallelism | Chosen |
| **Replace with Prefect native DAG** | True parallel execution, built-in retry/skip | Major rewrite, couples tightly to Prefect internals, harder to debug | Rejected — too large for this scope |
| **Topological sort at runtime** | Auto-derives order from YAML | Adds runtime complexity, harder to reason about stage order | Rejected — overkill given current sequential model |

### Architecture Changes

No new modules or classes. Changes are confined to:

- `code/scripts/preprocess_workflow_kinect_auto.py` — error cascade logic, stage
  ordering, context-passing
- `code/scripts/preprocess_workflow_kinect_manual.py` — adopt `TaskExecutor`,
  reorder colorspace steps
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — dependency corrections
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — dependency corrections
- `code/src/utils/pipeline/pipeline_config_manager.py` — add
  `get_dependents(task_name)` helper to `DagConfigHandler`
- Validation functions in `_3_preprocessing/` — strengthen checks (size + schema)

---

## Implementation Plan

### Phase 1: DAG Dependency Corrections (YAML + code ordering)

**Goal:** Make declared dependencies match actual data flow.

- [x] Task 1.1 — In `preprocess_workflow_kinect_auto_dag.yaml`: change
      `find_single_touches.depends_on` from `generate_3d_hand_in_motion` to
      `generate_xyz_stickers` (keep `define_trial_id` and
      `generate_stimuli_metadata`)
- [x] Task 1.2 — In `preprocess_workflow_kinect_auto_dag.yaml`: move
      `validate_hand_extraction` to depend on `track_hands_model` (not
      `generate_3d_hand_in_motion`) and add it as a dependency OF
      `generate_3d_hand_in_motion`
- [x] Task 1.3 — In `preprocess_workflow_kinect_auto.py`: reorder
      `pipeline_stages` so `validate_hand_extraction` appears before
      `generate_3d_hand_in_motion`
- [x] Task 1.4 — In `preprocess_workflow_kinect_manual_dag.yaml`: add
      `prepare_led_tracking` to `define_trial_chunks.depends_on`
- [x] Task 1.5 — In `preprocess_workflow_kinect_manual_dag.yaml`: make
      `review_handstickers_color_threshold` depend on
      `prepare_stickers_colorspace` (not just `review_2d_stickers`)
- [x] Task 1.6 — In `preprocess_workflow_kinect_manual.py`: swap the code
      ordering of steps 4 and 5 so `prepare_stickers_colorspace` runs before
      `review_handstickers_color_threshold`

**Files Modified:**

- `configs/preprocess_workflow_kinect_auto_dag.yaml` — dependency list edits
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — dependency list edits
- `code/scripts/preprocess_workflow_kinect_auto.py` — stage list reorder
- `code/scripts/preprocess_workflow_kinect_manual.py` — step reorder

**Dependencies:** None

### Phase 2: DAG-Aware Failure Cascading

**Goal:** When a task fails, only skip its transitive dependents — not the
entire remaining pipeline.

- [x] Task 2.1 — Add `get_dependents(task_name) -> set[str]` method to
      `DagConfigHandler` that walks the DAG and returns all tasks transitively
      depending on `task_name`
- [x] Task 2.2 — In the auto pipeline's main loop, replace the "skip everything
      after index" logic with: compute the transitive dependents of the failed
      task, mark only those as SKIPPED, and `continue` the loop for other tasks
- [x] Task 2.3 — Track failed tasks in a set so `can_run()` can implicitly
      block dependents (a task whose dependency is failed or skipped cannot run)

**Files Modified:**

- `code/src/utils/pipeline/pipeline_config_manager.py` — new method
- `code/scripts/preprocess_workflow_kinect_auto.py` — replace error cascade block

**Dependencies:** Phase 1 (correct dependencies must be in place first)

### Phase 3: Manual Pipeline — Adopt `TaskExecutor`

**Goal:** Consistent error handling and monitor integration across both pipelines.

- [x] Task 3.1 — Import `TaskExecutor` and optionally `PipelineMonitor` in the
      manual script
- [x] Task 3.2 — Replace each `if dag_handler.can_run(...): ... mark_completed()`
      block with a `TaskExecutor` context manager, mirroring the auto pattern
- [x] Task 3.3 — Verify that exception suppression in `TaskExecutor.__exit__`
      is acceptable for manual (interactive GUI) tasks — if not, add an option
      to re-raise

**Files Modified:**

- `code/scripts/preprocess_workflow_kinect_manual.py` — task dispatch refactor

**Dependencies:** Phase 1

### Phase 4: Strengthen Validation Gates

**Goal:** Critical handoff checks verify file content, not just existence.

- [x] Task 4.1 — `is_2d_stickers_tracking_valid()`: verify the JSON contains the
      expected `"validated"` key **and** that the referenced ROI CSV has >0 rows
- [x] Task 4.2 — `is_correlation_videos_threshold_defined()`: verify the
      threshold value is a finite number, not just that the key exists
- [x] Task 4.3 — `.SUCCESS` sentinel for curated hands: verify the curated `.pkl`
      file size is >0 and the sentinel was written after the pkl
- [x] Task 4.4 — `_single-touches-corrected.csv` check: verify the file has a
      header row and >0 data rows
- [x] Task 4.5 — Add a shared utility `validate_csv_not_empty(path, min_rows=1)`
      to reduce duplication across gates

**Files Modified:**

- `code/src/_3_preprocessing/_1_sticker_tracking/` — validation functions
- `code/src/_3_preprocessing/_2_hand_tracking/` — validation functions
- `code/scripts/preprocess_workflow_kinect_auto.py` — single-touches check
- `code/src/utils/pipeline/` — optional shared validator utility

**Dependencies:** Phase 1 (so the correct gates are being strengthened)

### Phase 5: Documentation

**Goal:** Capture the DAG validation pattern for future reference.

- [x] Task 5.1 — Create
      `docs/development/knowledge-base/note-dag-validation-ordering.md`
      documenting: the flaw classes found, root causes, fix patterns, and a
      checklist for future DAG additions
- [x] Task 5.2 — Add inline comments at the top of both workflow scripts
      describing the manual-auto dependency contract

**Files Modified:**

- `docs/development/knowledge-base/note-dag-validation-ordering.md` — new file
- `docs/development/knowledge-base/README.md` — add index entry
- `code/scripts/preprocess_workflow_kinect_auto.py` — header comment
- `code/scripts/preprocess_workflow_kinect_manual.py` — header comment

**Dependencies:** Phases 1–4

---

## Testing Plan

### Unit Tests

- [x] `DagConfigHandler.get_dependents()` returns correct transitive closure for
      a known test DAG
- [x] `DagConfigHandler.get_dependents()` returns empty set for a leaf task
- [x] `validate_csv_not_empty()` rejects zero-byte, header-only, and missing files
- [x] `validate_csv_not_empty()` accepts a file with header + 1 data row

### Integration Tests

- [x] Simulate a `track_led_blinking` failure and verify `track_stickers_raw`
      still executes (mocked pipeline)
- [x] Simulate a `validate_hand_extraction` failure and verify
      `generate_3d_hand_in_motion` is blocked (since validation now precedes it)

### Manual Verification

- [x] Run the manual pipeline on one session config and confirm all `TaskExecutor`
      context managers log correctly
- [x] Run the auto pipeline with one task forced to fail and confirm only its
      dependents are marked SKIPPED in the Excel report

### Edge Cases

- [x] A task with no dependents fails — no other tasks should be skipped
- [x] All tasks disabled except one leaf — pipeline runs just that task
- [x] Stale `.SUCCESS` sentinel from a previous incompatible run — validation
      rejects it based on timestamp comparison

---

## Documentation Plan

- [x] Create knowledge-base note: `docs/development/knowledge-base/note-dag-validation-ordering.md`
- [x] Update `docs/development/knowledge-base/README.md` index
- [x] Add header docstrings to both workflow scripts describing the manual-auto contract

---

## Rollback Plan

All changes are to Python scripts, YAML configs, and docs. No data migrations,
no database changes, no external service calls.

1. **Before deployment:** All changes on a feature branch; `main` is untouched.
2. **Rollback procedure:** `git revert` the merge commit. The only runtime
   artefacts are the Excel status reports, which are timestamped and
   non-destructive.
3. **Data considerations:** None — pipeline outputs are idempotent
   (`force_processing` flag re-derives from source).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Reordering `validate_hand_extraction` breaks sessions where metadata is generated inside `generate_3d_hand_in_motion` | Low | High | Read validation function source to confirm it only reads files produced by manual pipeline, not by auto |
| `get_dependents()` has a bug and skips too many / too few tasks | Medium | Medium | Unit tests with a known graph; run full pipeline on one real session before merge |
| `TaskExecutor` exception suppression hides GUI errors in manual pipeline | Medium | Low | Task 3.3 adds optional re-raise; manual pipeline can opt in |
| Strengthened validation gates reject previously-accepted files | Low | Medium | Run gates on existing processed data before merge to check false-rejection rate |

---

## References

- Analysis conversation: Claude Code session 2026-02-23 (auto/manual pipeline audit)
- `code/scripts/preprocess_workflow_kinect_auto.py` — auto pipeline
- `code/scripts/preprocess_workflow_kinect_manual.py` — manual pipeline
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — auto DAG config
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — manual DAG config
- `code/src/utils/pipeline/pipeline_config_manager.py` — `DagConfigHandler`
- `code/src/utils/pipeline/task_executor.py` — `TaskExecutor`
