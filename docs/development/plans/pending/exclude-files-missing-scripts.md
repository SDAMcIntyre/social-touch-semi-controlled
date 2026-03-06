# Plan: Add exclude_files Filtering to Remaining Workflow Scripts

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** TBD

---

## Overview

**What:** Add `exclude_files` parameter filtering to three workflow scripts that currently ignore it.
**Why:** The DAG Launcher GUI lets users uncheck specific kinect config files, but three scripts bypass the exclusion and process every YAML in the directory.
**How:** Apply the same 3-line filtering pattern already used in the five other workflow scripts.

## Problem Statement

The DAG Config Launcher GUI (Phase 5, Task 5.5 of the `dag-config-launcher-gui` plan) required patching all batch-processing scripts to honour the `exclude_files` parameter. Five scripts were patched, but three were missed:

1. `code/scripts/view_merged_neural_kinect.py` (line ~174)
2. `code/scripts/preprocess_workflow_kinect_manual.py` (line ~377)
3. `code/scripts/preprocess_workflow_kinect_visualisation.py` (line ~214)

All three follow the same pattern — they call `get_block_files()` and immediately iterate over the full list without reading or applying `exclude_files`. The result: unchecking files in the GUI has no effect when running these workflows.

## Goals

### In Scope
1. Add `exclude_files` filtering to `view_merged_neural_kinect.py`
2. Add `exclude_files` filtering to `preprocess_workflow_kinect_manual.py`
3. Add `exclude_files` filtering to `preprocess_workflow_kinect_visualisation.py`

### Out of Scope
- Changing the `exclude_files` mechanism itself (GUI, model, or DagConfigHandler)
- Adding new parameters or features to these scripts
- Refactoring the filtering into a shared utility (the 3-line pattern is simple enough)

## Success Criteria

- [ ] All three scripts read `exclude_files` from the DAG config and filter `block_files` before the processing loop
- [ ] Cherry-picking files in the GUI and running any of these workflows processes only the checked files
- [ ] No regressions — when `exclude_files` is absent or empty, all files are processed as before

---

## Technical Design

### Approach

Insert the same 3-line pattern already proven in the other five scripts:

```python
exclude_files = set(dag_handler_template.get_parameter('exclude_files', []) or [])
if exclude_files:
    block_files = [f for f in block_files if f.name not in exclude_files]
```

Each insertion goes between the `get_block_files()` call and the `for block_file in block_files:` loop.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Inline 3-line pattern (chosen) | Consistent with existing scripts, zero new abstractions | Minor duplication (8 call sites total) | Chosen |
| Extract shared `get_filtered_block_files()` helper | Removes duplication | Touches all 8 scripts, adds indirection for a trivial operation | Rejected — not worth the churn |

### Architecture Changes

None. Three small insertions into existing functions.

---

## Implementation Plan

### Phase 1: Add filtering to all three scripts
**Goal:** Every script that calls `get_block_files()` also applies `exclude_files`.

- [ ] Task 1.1 — In `view_merged_neural_kinect.py`: add exclude_files filtering after `get_block_files()` call (~line 174)
- [ ] Task 1.2 — In `preprocess_workflow_kinect_manual.py`: add exclude_files filtering after `get_block_files()` call (~line 377)
- [ ] Task 1.3 — In `preprocess_workflow_kinect_visualisation.py`: add exclude_files filtering after `get_block_files()` call (~line 214)

**Files Modified:**
- `code/scripts/view_merged_neural_kinect.py` — 3 lines added
- `code/scripts/preprocess_workflow_kinect_manual.py` — 3 lines added
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — 3 lines added

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Open the DAG Launcher GUI, select a workflow (e.g. View Merged Neural Kinect)
- [ ] Uncheck one or more config files within the selected kinect directory
- [ ] Save and Run — confirm only the checked files are processed (check console output for block names)
- [ ] Repeat for Preprocess Kinect Manual and Preprocess Kinect Visualisation workflows
- [ ] Confirm that with no `exclude_files` in the YAML (or an empty list), all files are processed as before

### Edge Cases
- [ ] `exclude_files` key absent from YAML — all files processed (no crash)
- [ ] `exclude_files` key present but empty list — all files processed
- [ ] All files excluded — no files processed, no crash (graceful empty loop)

---

## Documentation Plan

- [ ] No documentation changes needed (internal bug fix aligning with existing feature)

---

## Rollback Plan

1. Revert the 3-line additions from each script
2. No data, config, or state changes to revert

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Typo in parameter name (`exclude_files`) | Very Low | Low | Copy-paste from working script; verified against `DagConfigHandler.get_parameter` |
| `dag_handler_template` variable not available in scope | Low | Low | All three scripts already create this variable before the insertion point |

---

## References

- Root cause analysis: Claude Code session 2026-03-06
- Original feature plan: `docs/development/plans/active/dag-config-launcher-gui.md` (Phase 5, Task 5.5)
- Working pattern: `code/scripts/preprocess_workflow_kinect_auto.py:595-597`
- Affected scripts: lines 174, 377, 214 respectively
