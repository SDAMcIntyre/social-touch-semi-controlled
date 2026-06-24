# Plan: Keep Stale Outputs Option

**Date:** 2026-03-21
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-23 11:02
**Branch:** `feature/keep-stale-outputs`

---

## Overview

Add a `keep_stale` option to `should_process_task()` that refreshes the modification
timestamp of stale output files instead of triggering reprocessing. This is particularly
valuable for expensive manual/interactive tasks like `review_single_touches`, where
re-doing the step costs significant human time. `force_processing` still overrides
`keep_stale` and forces actual reprocessing.

## Problem Statement

When an upstream input is modified (e.g. a dependency is regenerated), `should_process_task()`
marks all downstream outputs as stale and requires reprocessing. For automated tasks this is
correct. But for interactive review tasks — where a human has already validated the output —
re-running the task wastes time with no benefit if the upstream change is cosmetic or irrelevant
to the reviewed output.

Currently the only way to avoid re-review is to manually touch the output file outside the
pipeline, which is error-prone and undiscoverable.

## Goals

### In Scope
1. Add `keep_stale` parameter to `should_process_task()` that touches stale outputs instead of returning `True`
2. Wire the option through to `review_single_touches` via the manual workflow
3. Expose the option in the DAG YAML config so it appears in the GUI detail panel

### Out of Scope
- Adding `keep_stale` to other tasks (can be done later by adding the option to their YAML config)
- Any changes to `clean_task_outputs()` — it is only called after `should_process_task()` returns `True`
- Changing the GUI layout — `keep_stale` is a regular boolean option, so it will render automatically as a checkbox in the existing `TaskDetailPanel`

## Success Criteria

- [ ] `should_process_task(..., keep_stale=True)` touches stale outputs and returns `False`
- [ ] `should_process_task(..., keep_stale=True, force=True)` still returns `True` (force wins)
- [ ] Missing outputs still trigger processing regardless of `keep_stale`
- [ ] `review_single_touches` accepts and passes through `keep_stale`
- [ ] DAG config has `keep_stale: false` option for `review_single_touches`
- [ ] Option appears as a checkbox in the GUI detail panel when selecting the task

---

## Technical Design

### Approach

Add `keep_stale` as a keyword-only `bool` parameter to `should_process_task()`, defaulting
to `False` (preserving backward compatibility). The new logic slot sits between the
"force" check and the "missing output" check in the existing decision tree:

```
1. Missing input?        -> raise FileNotFoundError  (unchanged)
2. force=True?           -> return True              (unchanged, overrides keep_stale)
3. Output missing?       -> return True              (unchanged)
4. Output stale?
   a. keep_stale=True    -> touch output(s), return False   ** NEW **
   b. keep_stale=False   -> return True              (unchanged)
5. Up-to-date?           -> return False             (unchanged)
```

Touching is done via `Path.touch()` which updates `st_mtime` to the current time. This
makes the staleness check pass on subsequent runs without modifying file content.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Parameter on `should_process_task()` | Generic, reusable, zero caller changes for tasks that don't use it | Slightly wider API surface | **Chosen** |
| Wrapper function in `review_single_touches` only | No shared API change | Duplicates timestamp logic; not reusable for other manual tasks | Rejected |
| Separate "touch outputs" utility called by workflow | Decoupled from decision logic | Caller must coordinate two calls; easy to forget | Rejected |

### Architecture Changes

No new modules. Changes are limited to existing files:

- `code/src/utils/should_process_task.py` — add `keep_stale` parameter + touch logic
- `code/scripts/_3_preprocessing/_6_metadata_matching/review_single_touches.py` — accept + pass through
- `code/scripts/preprocess_workflow_kinect_manual.py` — read option from DAG config + pass through
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — add option to `review_single_touches`

---

## Implementation Plan

### Phase 1: Core Logic
**Goal:** Add `keep_stale` to `should_process_task()`.

**Tasks:**
- [x] Task 1.1 — Add `keep_stale: bool = False` keyword parameter to `should_process_task()`
- [x] Task 1.2 — After the staleness check (line 83), if stale and `keep_stale=True`, call `p.touch()` on each output and return `False`
- [x] Task 1.3 — Add `PermissionError` handling around `touch()` (mirror pattern from `clean_task_outputs`)
- [x] Task 1.4 — Add a log message distinguishing the "kept stale" case (e.g. "Task is stale but keep_stale=True — refreshing output timestamps.")

**Files Modified:**
- `code/src/utils/should_process_task.py` — add parameter, touch logic, logging

**Dependencies:** None

### Phase 2: Wiring
**Goal:** Connect the option from YAML config through to `should_process_task()`.

**Tasks:**
- [x] Task 2.1 — Add `keep_stale: bool = False` parameter to `review_single_touches()` function signature
- [x] Task 2.2 — Pass `keep_stale` through to `should_process_task()` call inside `review_single_touches()`
- [x] Task 2.3 — Add `keep_stale: bool = False` parameter to `review_single_touches_flow()` in the manual workflow
- [x] Task 2.4 — Read `keep_stale` from DAG task options alongside `force_processing` in the workflow dispatcher
- [x] Task 2.5 — Add `keep_stale: false` to `review_single_touches.options` in the manual DAG YAML

**Files Modified:**
- `code/scripts/_3_preprocessing/_6_metadata_matching/review_single_touches.py` — add parameter, pass through
- `code/scripts/preprocess_workflow_kinect_manual.py` — read from config, pass to flow and function
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — add option entry

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run with stale output + `keep_stale: false` — task processes normally (baseline)
- [ ] Run with stale output + `keep_stale: true` — output timestamp refreshed, task skipped
- [ ] Run with stale output + `keep_stale: true` + `force_processing: true` — task processes (force wins)
- [ ] Run with missing output + `keep_stale: true` — task processes (missing output always triggers)
- [ ] Run with up-to-date output + `keep_stale: true` — task skipped (no change from current behavior)
- [ ] Verify the option appears as a checkbox in the GUI detail panel for `review_single_touches`

### Edge Cases
- [ ] Output file is read-only or locked — `PermissionError` handled gracefully, processing proceeds
- [ ] Multiple output paths — all are touched when stale

---

## Documentation Plan

- [ ] No external documentation needed — the option is self-documenting via the YAML config and GUI checkbox

---

## Rollback Plan

1. Revert the single commit on the feature branch
2. No data migration — `keep_stale` defaults to `False`, so removing it restores original behavior
3. Any YAML configs with `keep_stale: true` become harmless unknown keys (ignored by code)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| User enables `keep_stale` and forgets, missing a genuinely needed re-review | Medium | Low | `force_processing` always overrides; log message makes the skip visible |
| `Path.touch()` fails on network/read-only filesystems | Low | Low | `PermissionError` catch falls through to normal processing |

---

## References

- `code/src/utils/should_process_task.py` — existing staleness logic
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — auto-renders boolean options as checkboxes
- `docs/development/plans/completed/stale-output-cleanup.md` — related prior work on staleness handling
