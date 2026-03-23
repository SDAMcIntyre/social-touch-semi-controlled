# Plan: Non-Blocking Dependency Popups

**Date:** 2026-03-21
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/non-blocking-dependency-popups`

---

## Overview

When the auto kinect preprocessing pipeline hits a session requiring manual intervention, it shows a blocking popup and then calls `sys.exit(0)`, killing the entire batch. This plan makes the popup non-blocking and informative (including the session name), so the pipeline skips the failed session and continues processing the rest of the batch.

## Problem Statement

The current `show_dependency_error_popup()` in `dependency_popup.py` uses `QMessageBox.exec_()` (a modal, blocking call) followed by `sys.exit(0)`. This has two consequences:

1. The user must click OK before anything continues — the pipeline is frozen waiting for input.
2. After clicking OK, `sys.exit(0)` terminates the entire Python process — all remaining sessions in the batch are lost.

Additionally, the popup does not display which session failed, making it harder for the user to know which session needs manual attention.

The manual batch loop (`preprocess_workflow_kinect_manual.py:380-394`) already handles errors gracefully with `try/except + continue`. The auto batch loop (`preprocess_workflow_kinect_auto.py:614-636`) does not.

## Goals

### In Scope
1. Make the dependency error popup non-blocking (no user click required to continue)
2. Display the session name in the popup so the user knows which session needs manual work
3. Remove the `sys.exit(0)` call so the pipeline continues to the next session
4. Add error resilience to the auto batch processing loop (matching the manual loop pattern)

### Out of Scope
- Changing the manual pipeline workflow
- Adding new GUI features or notification systems beyond the existing popup
- Changing which conditions trigger `PipelineDependencyError`
- Auto-retry or auto-recovery of failed sessions

## Success Criteria

- [ ] When a session raises `PipelineDependencyError`, a non-blocking popup appears showing the session name, the error message, and the required pipeline
- [ ] The pipeline immediately continues to the next session without waiting for user interaction
- [ ] The popup remains visible until the user dismisses it (does not auto-close)
- [ ] All remaining sessions in the batch are processed even if earlier sessions fail
- [ ] The pipeline monitor/report correctly marks failed sessions as "FAILURE" and subsequent sessions are still processed

---

## Technical Design

### Approach

Replace the blocking `QMessageBox.exec_()` + `sys.exit(0)` pattern with a non-modal `QMessageBox.show()` that stays visible but does not block execution. Pass the session name through `PipelineDependencyError` so the popup can display it. Add `try/except` around the auto batch loop's sequential session calls.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Non-modal `QMessageBox.show()` | Simple, reuses existing Qt infrastructure, popup stays visible | Multiple popups may stack if several sessions fail | **Chosen** — stacking is acceptable and even helpful (shows all failures at a glance) |
| Console-only logging (no popup) | Simplest change | User may miss failures buried in log output | Rejected — popups provide immediate visual feedback |
| Toast/notification system | Modern UX, auto-dismiss | Requires new infrastructure, may auto-dismiss before user reads | Rejected — over-engineering |

### Architecture Changes

- `PipelineDependencyError` gains an optional `session_name` attribute
- `show_dependency_error_popup()` becomes non-blocking and accepts a session name parameter
- `TaskExecutor.__exit__` passes the session name to the popup and no longer gets killed by `sys.exit`
- Auto batch loop gains error resilience

---

## Implementation Plan

### Phase 1: Non-blocking popup and session name propagation
**Goal:** Make the popup non-blocking, informative, and non-fatal

**Tasks:**
- [ ] Task 1.1 — Add `session_name` parameter to `PipelineDependencyError.__init__` (default `None` for backward compat)
- [ ] Task 1.2 — Rewrite `show_dependency_error_popup()`: remove `sys.exit(0)`, use `msg.show()` instead of `msg.exec_()`, include session name in the message text, keep a reference to prevent garbage collection
- [ ] Task 1.3 — Update `TaskExecutor.__init__` to accept and store `session_name`; pass it when calling `show_dependency_error_popup()` in `__exit__`; reorder `__exit__` so error bookkeeping (setting `error_msg`, updating monitor) happens before the popup call
- [ ] Task 1.4 — Update the `TaskExecutor` instantiation in `run_single_session_pipeline()` (auto workflow, line 564) to pass `block_name` as `session_name`
- [ ] Task 1.5 — Propagate `session_name` into each `PipelineDependencyError` raise site in the auto workflow (7 sites, lines 125-400) so the error object carries the session context

**Files Modified:**
- `code/src/utils/pipeline/pipeline_dependency_error.py` — Add `session_name` attribute
- `code/src/utils/pipeline/dependency_popup.py` — Non-blocking popup with session name display
- `code/src/utils/pipeline/task_executor.py` — Pass session name, reorder `__exit__` logic
- `code/scripts/preprocess_workflow_kinect_auto.py` — Pass `block_name` to `TaskExecutor`; add `session_name` to `PipelineDependencyError` raise sites

**Dependencies:** None

### Phase 2: Batch loop resilience
**Goal:** Ensure the auto batch loop continues processing after a session failure

**Tasks:**
- [ ] Task 2.1 — In `run_batch_processing()` sequential mode (line 630-636), wrap the session call in `try/except Exception` with logging and `continue`, matching the pattern in `run_batch_sequentially()` from the manual workflow

**Files Modified:**
- `code/scripts/preprocess_workflow_kinect_auto.py` — Add try/except around sequential session call

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run auto preprocessing batch with 3+ sessions where session #1 or #2 requires manual intervention (e.g., missing forearm extraction)
- [ ] Verify: non-blocking popup appears showing session name, error, and required pipeline
- [ ] Verify: pipeline immediately moves to session #2/#3 without waiting for popup dismissal
- [ ] Verify: popup remains visible on screen until user closes it
- [ ] Verify: if multiple sessions fail, multiple popups are shown (one per failure)
- [ ] Verify: pipeline monitor/report shows correct status for all sessions (failed + successful)
- [ ] Verify: manual preprocessing pipeline is unaffected (it does not use `TaskExecutor` or `dependency_popup`)

### Edge Cases
- [ ] All sessions in batch require manual intervention — pipeline completes with all failures reported
- [ ] No sessions require manual intervention — pipeline runs normally, no popups
- [ ] Non-`PipelineDependencyError` exceptions in a session — session fails, batch continues (Phase 2)

---

## Documentation Plan

- [ ] No external documentation needed — this is an internal behavioral fix

---

## Rollback Plan

1. Revert the 4 modified files to their previous state
2. No data migrations or breaking changes involved

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Multiple non-modal popups overlap on screen | Medium | Low | Acceptable — each popup shows which session failed; user can dismiss individually |
| QMessageBox garbage-collected before user sees it | Medium | Medium | Store reference in a module-level list to prevent GC |
| QApplication not initialized when popup is called | Low | Medium | Already handled: existing code creates QApplication if none exists |
