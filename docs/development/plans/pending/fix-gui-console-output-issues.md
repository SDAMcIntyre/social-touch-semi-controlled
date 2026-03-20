# Plan: Fix GUI Console Output Issues

**Date:** 2026-03-19
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/fix-gui-console-output-issues`

---

## Overview

Fix two bugs in the pipeline GUI console output: (1) dependency errors exit with code 0, making the GUI falsely report "Finished" instead of "Failed", and (2) garbled CJK characters in `PipelineMonitor` print statements that were likely intended to be emoji.

## Problem Statement

The GUI launcher runs pipeline scripts as subprocesses and uses the exit code to determine the status label shown to the user. When a `PipelineDependencyError` occurs (e.g., a manual step hasn't been completed), `dependency_popup.py` calls `sys.exit(0)`, causing the GUI to display "Finished (exit code 0)" — misleading the user into thinking the pipeline succeeded. Additionally, `pipeline_monitor.py` contains two garbled Unicode characters (`投`, `圷`) that appear as nonsense in the console output.

## Goals

### In Scope
1. Fix exit code so dependency errors correctly signal failure to the GUI
2. Replace garbled Unicode characters with proper emoji in PipelineMonitor

### Out of Scope
- Standardizing message formatting across all preprocessing modules (larger effort, separate plan)
- Refactoring logging infrastructure (multiple `basicConfig()` calls)
- Adding structured logging or Prefect logger integration

## Success Criteria

- [ ] When a `PipelineDependencyError` stops the pipeline, the GUI status bar shows "Failed (exit code 1)"
- [ ] PipelineMonitor prints display clean, readable emoji instead of garbled characters

---

## Technical Design

### Approach

Two targeted fixes in two files. No architectural changes.

**Fix 1 — Exit code:** The GUI's `_poll_process` method (`launcher_window.py:373-375`) determines the status label:
```python
label = "Finished" if retcode == 0 else "Failed"
```
Changing `sys.exit(0)` to `sys.exit(1)` in `dependency_popup.py` makes this logic work correctly for dependency failures.

**Fix 2 — Garbled characters:** The characters `投` (line 63) and `圷` (line 146) in `pipeline_monitor.py` are CJK ideographs that appear to be encoding corruption from a prior edit. Replace with contextually appropriate emoji.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Change exit code to 1 | Simple, correct semantics | None | Chosen |
| Use distinct exit codes per error type | More granular | Over-engineering for current needs | Rejected |
| Remove emoji entirely | No encoding risk | Less readable console output | Rejected |
| Replace garbled chars with emoji | Consistent with existing style | None | Chosen |

### Architecture Changes

None. Two isolated line-level fixes.

---

## Implementation Plan

### Phase 1: Fix both issues
**Goal:** Correct exit code and garbled characters

- [ ] Task 1.1 — Change `sys.exit(0)` to `sys.exit(1)` in `dependency_popup.py:34`
- [ ] Task 1.2 — Replace `投` with `📊` in `pipeline_monitor.py:63`
- [ ] Task 1.3 — Replace `圷` with `⚠️` in `pipeline_monitor.py:146`

**Files Modified:**
- `code/src/utils/pipeline/dependency_popup.py` — Change exit code from 0 to 1
- `code/src/utils/pipeline/monitoring/pipeline_monitor.py` — Replace 2 garbled characters with emoji

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run preprocess auto pipeline from GUI, trigger a `PipelineDependencyError` (skip a manual step like forearm extraction) — GUI should show "Failed (exit code 1)"
- [ ] Run a pipeline with PipelineMonitor enabled — verify console shows clean emoji on monitor status lines
- [ ] Run a pipeline to completion — verify "Finished (exit code 0)" still appears on success

### Edge Cases
- [ ] Multiple dependency errors in batch mode — each session should trigger popup and exit with code 1

---

## Documentation Plan

- [ ] No documentation changes needed (internal bug fix)

---

## Rollback Plan

1. Revert the two commits (one per file or combined)
2. No data or configuration changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `sys.exit(1)` changes behavior for scripts that check exit code | Low | Low | Only `launcher_window.py` checks exit code; behavior improves |
| Emoji rendering varies by terminal/font | Low | Low | Emoji already used extensively in existing codebase |
