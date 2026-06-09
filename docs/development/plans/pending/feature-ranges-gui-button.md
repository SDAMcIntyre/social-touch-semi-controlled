# Plan: Feature Ranges GUI Button for Population RF Grid

**Date:** 2026-05-13
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/neural-kinect-viewer-responsive-navigation`
**Branch:** `feature/feature-ranges-gui-button`

---

## Overview

Move the standalone script `code/scripts/inspect_touch_feature_ranges.py` into
the analysis pipeline as a reusable module, and surface it as a "Feature
Ranges..." button inside the `map_population_rf_grid` task's grid_groups section
in the DAG launcher GUI. This lets the user inspect per-column min/max/percentile
statistics directly from the panel where grid group feature bounds are configured.

## Problem Statement

The feature range inspection logic lives as a standalone CLI script
(`code/scripts/inspect_touch_feature_ranges.py`) disconnected from the GUI
workflow. When calibrating feature bounds for population RF grid groups, the user
must leave the GUI, run the script in a terminal, mentally cross-reference the
printed table with the grid group dialog, then return to the GUI to enter values.
This is error-prone and breaks the workflow.

## Goals

### In Scope
1. Extract the script's core logic into a reusable function under `analysis/touch_analytics/`
2. Create a `FeatureRangesDialog` that displays the inspection results in a scrollable, monospace table
3. Add a "Feature Ranges..." button to the grid_groups section of the task detail panel
4. Convert the original script to a thin CLI shim that delegates to the new module

### Out of Scope
- Auto-populating grid group feature bounds from the inspection results
- Writing inspection output to disk (the dialog is ephemeral; disk output can be added later)
- Adding the inspection as a separate DAG task (user chose button-in-panel over task-row)

## Success Criteria

- [ ] `run_feature_range_inspection()` is importable from `analysis.touch_analytics` and returns `(DataFrame, str)`
- [ ] CLI script `inspect_touch_feature_ranges.py --percentile 90` produces identical output to the original
- [ ] "Feature Ranges..." button appears in the grid_groups section when `map_population_rf_grid` is selected
- [ ] Clicking the button opens a dialog showing all feature statistics in a formatted table
- [ ] Dialog percentile spinner updates the table on refresh

---

## Technical Design

### Approach

Extract the script into a pure-function module (`inspection_pipeline.py`) that
takes a directory path and percentile, returns structured data. The GUI layer
creates a lightweight dialog that calls this function. The button is placed in
`_make_grid_groups_section()` alongside the existing "New Group..." button.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Button in grid_groups section | Contextual — right where bounds are edited; no YAML changes | Only visible when grid_groups option exists | **Chosen** |
| Separate DAG task row | Standard pattern; produces persistent output file | Disconnected from the grid group editing workflow; overkill for a quick inspection | Rejected |
| Embed stats in GridGroupDialog | Tightest integration | Crowds an already complex dialog; stats are cross-feature, not per-group | Rejected |

### Architecture Changes

**New modules:**
- `code/src/analysis/touch_analytics/inspection_pipeline.py` — core logic
- `code/src/utils/gui/dag_launcher/feature_ranges_dialog.py` — PyQt5 dialog

**Modified modules:**
- `code/src/analysis/touch_analytics/__init__.py` — re-export
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — button + handler
- `code/scripts/inspect_touch_feature_ranges.py` — thin shim

**Integration points:**
- `inspection_pipeline.py` uses `pandas` and `numpy` (no project imports beyond standard patterns)
- `task_detail_panel.py` imports the dialog; the handler resolves data path via `path_tools.get_project_data_root()`
- The dialog imports `run_feature_range_inspection` from the new pipeline module

---

## Implementation Plan

### Phase 1: Core Logic Extraction
**Goal:** Reusable inspection function under `analysis/touch_analytics/`

- [ ] Task 1.1 — Create `inspection_pipeline.py` with `run_feature_range_inspection(touch_features_dir, percentile) -> (DataFrame, str)`
- [ ] Task 1.2 — Move constants (`TOUCH_KEYS`, `SHARED_NON_NUMERIC`, `_SUFFIX_FAMILIES`) and helpers (`_classify_family`, `_format_report`) into the new module
- [ ] Task 1.3 — Export `run_feature_range_inspection` from `touch_analytics/__init__.py`
- [ ] Task 1.4 — Convert `code/scripts/inspect_touch_feature_ranges.py` to a thin shim importing the new function

**Files Modified:**
- `code/src/analysis/touch_analytics/inspection_pipeline.py` — **new** core logic
- `code/src/analysis/touch_analytics/__init__.py` — add import + `__all__` entry
- `code/scripts/inspect_touch_feature_ranges.py` — rewrite as thin CLI shim

**Dependencies:** None

### Phase 2: GUI Dialog + Button
**Goal:** "Feature Ranges..." button in the grid_groups section that opens a results dialog

- [ ] Task 2.1 — Create `feature_ranges_dialog.py` with `FeatureRangesDialog(parent, touch_features_dir)`
  - `QDoubleSpinBox` for percentile (50.0–99.9, default 99.0)
  - "Refresh" button to recompute with new percentile
  - Read-only `QTextEdit` with monospace font showing the formatted report
  - Error display inline in the text area (fail-fast: show full error message, not silent)
  - Guard spinbox value changes with `blockSignals` to prevent signal recursion (KB note)
- [ ] Task 2.2 — Add "Feature Ranges..." button row in `_make_grid_groups_section()` after "New Group..." row
- [ ] Task 2.3 — Add `_on_feature_ranges()` handler method on `TaskDetailPanel` that resolves the data path and opens the dialog

**Files Modified:**
- `code/src/utils/gui/dag_launcher/feature_ranges_dialog.py` — **new** dialog
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — button + handler

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `python code/scripts/inspect_touch_feature_ranges.py` — confirm output matches original
- [ ] Run `python code/scripts/inspect_touch_feature_ranges.py --percentile 90` — confirm percentile labels change
- [ ] Launch GUI → Analysis [Processing] → select `map_population_rf_grid` → confirm "Feature Ranges..." button visible in grid_groups section
- [ ] Click button → confirm dialog opens with formatted statistics table
- [ ] Change percentile in dialog spinner → click Refresh → confirm table updates
- [ ] If `touch_features/` directory doesn't exist: confirm dialog shows clear error message (not blank or crash)

### Edge Cases
- [ ] No `*_touch_summary.csv` files found — dialog should show descriptive error
- [ ] Project data root not configured — button click shows warning message box
- [ ] Very large number of features — dialog should remain scrollable and responsive

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal pipeline restructuring)
- [ ] The CLI shim's docstring updated to reference the pipeline module

---

## Rollback Plan

All changes are additive (new files) or localized edits to existing files:
1. Delete `inspection_pipeline.py` and `feature_ranges_dialog.py`
2. Revert edits to `__init__.py`, `task_detail_panel.py`, and the CLI script
3. No data migrations, config changes, or breaking API changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `get_project_data_root()` returns None in GUI context | Low | Med | Show `QMessageBox.warning` before opening dialog |
| Qt signal recursion from spinbox | Low | Low | Guard with `blockSignals(True/False)` per KB note |
| CSV read takes too long (many sessions) | Low | Low | Current script runs <1s for 12 sessions; acceptable for modal dialog |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core extraction | Small (~80 lines new + shim rewrite) | None |
| Phase 2: GUI dialog + button | Small (~120 lines new + ~15 lines edit) | Phase 1 |

---

## References

- Original script: `code/scripts/inspect_touch_feature_ranges.py`
- Grid group dialog pattern: `code/src/utils/gui/dag_launcher/grid_group_dialog.py`
- Task detail panel: `code/src/utils/gui/dag_launcher/task_detail_panel.py:581` (`_make_grid_groups_section`)
- KB note on Qt signal recursion: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
