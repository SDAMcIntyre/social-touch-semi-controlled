# Plan: Cross-Pipeline Dependency Error Popups

**Created:** 2026-03-16 12:00
**Approved:** 2026-03-16
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `hotfix/cross-pipeline-dependency-popups`

---

## Overview

**What:** Add popup dialogs (QMessageBox) that inform the user when an auto pipeline task fails because a prerequisite from the manual pipeline hasn't been run yet.
**Why:** Currently, cross-pipeline dependency errors are buried in subprocess stdout — the user only sees "Failed (exit code X)" in the GUI status bar with no guidance on what to do.
**How:** Introduce a `PipelineDependencyError` exception, a popup utility, and integrate them into `TaskExecutor` so that any cross-pipeline validation failure surfaces a clear, actionable popup.

## Problem Statement

- The auto pipeline runs as a subprocess; errors are printed to stdout but never shown to the user visually
- The GUI launcher only reports exit codes — no error detail is surfaced
- Users must dig through console output to discover that they need to run a manual step first
- This affects multiple validation points across the auto pipeline (LED, forearm, hand model, stickers, etc.)

## Goals

### In Scope
1. Reusable popup mechanism for cross-pipeline dependency errors
2. Convert all existing cross-pipeline validation checks in the preprocess auto pipeline
3. Clear, actionable popup messages identifying which manual step is required

### Out of Scope
- Pre-flight validation in the GUI before launching the subprocess
- Capturing generic subprocess errors (non-dependency errors)
- Postprocess pipeline validation (can be added later with the same mechanism)

## Success Criteria

- [x] `PipelineDependencyError` exception class exists and is importable
- [x] Popup utility creates QApplication if needed and shows a blocking QMessageBox
- [x] `TaskExecutor.__exit__` detects `PipelineDependencyError` and triggers popup
- [x] All 8 cross-pipeline validation points converted to `PipelineDependencyError`
- [ ] After dismissing popup, task is still marked FAILURE and pipeline continues normally

---

## Technical Design

### Approach

Show popup directly in the subprocess when a dependency error is caught. This avoids IPC changes between the subprocess and GUI parent process. PyQt5 is already a project dependency. A `QApplication` instance is created on-the-fly if none exists (auto scripts are headless).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Popup in subprocess (chosen) | Self-contained, no IPC, minimal changes | Requires QApplication creation in headless context | **Chosen** |
| Capture subprocess stdout in GUI | Popup in parent process, cleaner architecture | Requires stdout parsing/protocol, more invasive | Rejected |
| Pre-flight validation in GUI | Catches errors before launch | Duplicates validation logic, must know all checks | Rejected |

### Architecture Changes

```
code/src/utils/pipeline/
├── __init__.py
├── pipeline_dependency_error.py  — NEW: custom exception
├── dependency_popup.py           — NEW: QMessageBox utility
├── task_executor.py              — MODIFIED: detect + show popup
├── pipeline_config_manager.py
└── ...
```

**Integration point:** `TaskExecutor.__exit__` is the single place where all task exceptions are caught. Adding an `isinstance` check there makes every task that raises `PipelineDependencyError` automatically trigger a popup — no per-task wiring needed.

---

## Implementation Plan

### Phase 1: Infrastructure
**Goal:** Create the exception class and popup utility
**Started:** —
**Completed:** —

- [x] Create `pipeline_dependency_error.py` with `PipelineDependencyError(message, required_pipeline)` exception
- [x] Create `dependency_popup.py` with `show_dependency_error_popup(error)` function
- [x] Modify `task_executor.py` to detect `PipelineDependencyError` and call popup

**Files Modified:**
- `code/src/utils/pipeline/pipeline_dependency_error.py` — New: exception class (~10 lines)
- `code/src/utils/pipeline/dependency_popup.py` — New: popup utility (~25 lines)
- `code/src/utils/pipeline/task_executor.py` — Add isinstance check in `__exit__` (3 lines)

**Dependencies:** None

### Phase 2: Convert Validation Points
**Goal:** Replace all cross-pipeline ValueError/FileNotFoundError with PipelineDependencyError
**Started:** —
**Completed:** —

- [x] Convert `generate_led_roi.py` line 57: LED ROI metadata → `PipelineDependencyError(..., "manual (prepare LED tracking)")`
- [x] Convert `preprocess_workflow_kinect_auto.py` line 124: forearm validation → `PipelineDependencyError(..., "manual (forearm extraction)")`
- [x] Convert line 135: hand model validation → `PipelineDependencyError(..., "manual (hand model assignment)")`
- [x] Convert line 154: 2D sticker tracking validation → `PipelineDependencyError(..., "manual (review 2D stickers)")`
- [x] Convert line 186: color threshold validation → `PipelineDependencyError(..., "manual (review color threshold)")`
- [x] Convert line 269: tracked hands file check → `PipelineDependencyError(..., "auto (track hands model)")`
- [x] Convert lines 272-273: hand curation check → `PipelineDependencyError(..., "manual (curate hand models)")`
- [x] Convert line 277: hand model metadata check → `PipelineDependencyError(..., "manual (hand model assignment)")`
- [x] Convert lines 364-366: single touches validation → `PipelineDependencyError(..., "manual (review single touches)")`

**Files Modified:**
- `code/scripts/_3_preprocessing/_5_led_tracking/generate_led_roi.py` — 1 raise statement
- `code/scripts/preprocess_workflow_kinect_auto.py` — 8 raise statements

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run auto pipeline on a session missing LED ROI metadata — popup appears with "manual (prepare LED tracking)" message
- [ ] Run auto pipeline on a session missing forearm extraction — popup appears
- [ ] Dismiss popup with OK — task marked FAILURE, downstream tasks skipped, pipeline continues
- [ ] Run auto pipeline on a fully prepared session — no popups, pipeline completes normally
- [ ] Verify popup renders correctly on Windows (QApplication creation works)

### Edge Cases
- [ ] Multiple sessions failing on same dependency — each gets its own popup (expected)
- [ ] Pipeline aborted while popup is showing — popup dismissed, process terminates

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal infrastructure)

---

## Rollback Plan

1. Revert commits on the hotfix branch
2. No data changes, no config changes — purely additive code

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| QApplication creation fails in CI/headless | Low | Low | Wrap popup in try/except; error still prints to console as before |
| Multiple popups annoying for batch runs | Medium | Low | Expected behavior; each is a real error requiring user attention |
| Popup blocks subprocess indefinitely if no display | Low | Medium | Windows always has a display; not a concern for this project's environment |

---

## References

- Related files: `code/src/utils/pipeline/task_executor.py`, `code/scripts/preprocess_workflow_kinect_auto.py`
