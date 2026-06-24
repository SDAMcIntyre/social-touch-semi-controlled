# Plan: Master-Detail Task Options Panel

**Date:** 2026-03-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-20 23:57
**Branch:** `feature/master-detail-task-panel`

---

## Overview

Replace the flat QTableWidget column-per-option layout in `TaskPanel` with a master-detail pattern: a compact task list (Task Name, Enabled, Depends On) paired with a per-task detail panel that rebuilds its layout when a row is clicked. This eliminates horizontal scrolling, N/A cell spam, and semantic incoherence caused by forcing heterogeneous task options into a single column grid.

## Problem Statement

The `TaskPanel` currently renders all task options as columns in a union across all tasks. For the Analysis workflow, this produces ~25-30 columns (9 feature toggles + 3+ combinations + 4 clustering profiles + 3 comparing profiles + scalars). Most cells are greyed-out N/A placeholders because each task only uses a subset of columns. Users must scroll horizontally to find relevant options, losing sight of task names. Adding any new feature/profile/combination worsens the problem by adding another global column.

## Goals

### In Scope
1. Replace the column-per-option table with a master-detail layout in `TaskPanel`
2. Render task-specific options in a detail panel with type-appropriate widgets
3. Apply to all workflows (simple workflows get a compact detail panel)
4. Preserve round-trip YAML editing via existing `DagConfigModel` API

### Out of Scope
- Changes to `DagConfigModel` or its API
- Changes to `FeatureCombinationDialog` or `YamlEditDialog`
- Changes to `LauncherWindow` layout or external `TaskPanel` interface
- Search/filter functionality for options
- Preset/template system for option configurations

## Success Criteria

- [ ] Task table has exactly 3 columns (Task Name, Enabled, Depends On) — no horizontal scrolling
- [ ] Clicking a task row shows that task's options in the detail panel below
- [ ] All 5 option types render correctly: scalar, feature dict, profile dict, combination dict, complex
- [ ] Feature checkboxes with "..." param buttons work (e.g., mechanics_of_solids)
- [ ] Combination CRUD works (add via "+", edit features, delete via context menu)
- [ ] Profile enable/disable toggles work with "..." for method-specific params
- [ ] Saving (Ctrl+S) produces correct YAML with preserved comments
- [ ] Simple workflows (preprocess, postprocess) display correctly with minimal detail panels
- [ ] No regressions in existing functionality

---

## Technical Design

### Approach

Master-detail pattern: the table becomes a compact task list, and a new `TaskDetailPanel` widget below it rebuilds its layout per selected task. This is the standard approach used by IDE settings dialogs (VS Code, Unity Inspector) for configuring heterogeneous objects with varying option sets.

The existing detection heuristics (`_is_profile_dict`, `_is_feature_dict`, `_is_feature_combinations_dict`) are reused in the detail panel to dispatch to type-specific section renderers.

### Layout

```
+-------------------------------------------+
| Tasks (QGroupBox)                         |
| +--- QSplitter (Vertical) ---------------+|
| | Task List (QTableWidget, 3 columns)    ||
| |  Task Name | Enabled | Depends On      ||
| |  summarize_session_blocks  [x]          ||
| |  touch_feature_extraction  [x]   <sel>  ||
| |  touch_clustering          [x]   feat.. ||
| +----------------------------------------+|
| | Task Detail Panel (QScrollArea)        ||
| |  "touch_feature_extraction" header     ||
| |  [x] Force Processing                 ||
| |  Features:                             ||
| |  [x] max   [x] mean   [ ] min         ||
| |  [ ] median [ ] std   [ ] range        ||
| |  [ ] skewness  [ ] temporal            ||
| |  [ ] mechanics_of_solids [...]         ||
| +----------------------------------------+|
+-------------------------------------------+
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Master-detail panel | No scrolling, type-appropriate widgets, familiar UX pattern, simplifies table code | Cannot compare same option across tasks side-by-side | **Chosen** — side-by-side comparison is rarely useful given heterogeneous tasks |
| Accordion rows (inline expansion) | Integrated feel, options stay in table context | QTableWidget doesn't support expandable rows natively; requires row insertion/removal that breaks index mapping | Rejected — high complexity, fragile |
| Grouped columns + frozen | Incremental change, preserves table | Doesn't eliminate N/A spam or column explosion; frozen columns in QTableWidget require fiddly overlay pattern | Rejected — treats symptoms, not root cause |

### Architecture Changes

- **New class:** `TaskDetailPanel(QScrollArea)` — renders options for one task at a time
- **Refactored:** `TaskPanel` — table reduced from N dynamic columns to 3 fixed columns; `_build_column_specs()` and per-column cell rendering removed
- **Reused without changes:** `_centred_checkbox()`, `_is_profile_dict()`, `_is_feature_dict()`, `_is_feature_combinations_dict()`, `_option_header()`, `_profile_header()`, `FeatureCombinationDialog`, `YamlEditDialog`

### Architecture Constraint (from knowledge base)

**Qt `itemChanged` signal recursion** (`note-qt-itemchanged-signal-recursion.md`): When detail panel changes propagate back to table items (e.g., updating enabled state), guard mutations with `blockSignals(True/False)` to prevent infinite recursion.

---

## Implementation Plan

### Phase 1: TaskDetailPanel Widget
**Goal:** Create the detail panel that renders per-task options
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Task 1.1 — Create `TaskDetailPanel(QScrollArea)` class with `task_changed` signal
- [x] Task 1.2 — Implement `show_task(model, task_name)` that clears content and rebuilds layout
- [x] Task 1.3 — Implement `_add_scalar_section()` for bool (checkbox) and number/string (label + line edit)
- [x] Task 1.4 — Implement `_add_feature_section()` — checkbox grid with "..." param buttons for features with extra keys
- [x] Task 1.5 — Implement `_add_profile_section()` — checkbox list with "..." buttons opening `YamlEditDialog` for method params
- [x] Task 1.6 — Implement `_add_combination_section()` — checkbox + feature-list label + "+" button + right-click delete
- [x] Task 1.7 — Implement `_add_complex_section()` — blue clickable preview opening `YamlEditDialog`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — New file: `TaskDetailPanel` class (293 LOC)

**Dependencies:** None

### Phase 2: Simplify TaskPanel Table
**Goal:** Reduce the table to 3 fixed columns and wire selection to the detail panel
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Task 2.1 — Replace `_build_column_specs()` with 3 fixed columns: Task Name, Enabled, Depends On
- [x] Task 2.2 — Simplify `populate()` to only render task name, enabled checkbox, and depends_on text per row
- [x] Task 2.3 — Remove per-column cell rendering code (profile cells, feature cells, combination cells, complex cells, combination_add cells)
- [x] Task 2.4 — Remove unused handler factories that are now handled by `TaskDetailPanel`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — Rewritten: 654 → 155 LOC

**Dependencies:** Phase 1

### Phase 3: Layout Integration and Wiring
**Goal:** Integrate the detail panel into the TaskPanel layout and connect signals
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Task 3.1 — Replace `QVBoxLayout` with `QSplitter(Qt.Vertical)` containing table + detail panel
- [x] Task 3.2 — Connect `QTableWidget.currentRowChanged` → `TaskDetailPanel.show_task()`
- [x] Task 3.3 — Auto-select first row on `populate()` so detail panel is never empty
- [x] Task 3.4 — Propagate `TaskDetailPanel.task_changed` → `TaskPanel.task_changed` signal
- [x] Task 3.5 — Set initial splitter sizes (40% table / 60% detail)
- [x] Task 3.6 — Apply `blockSignals` guard where detail panel updates could trigger table signals

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — Layout restructuring and signal wiring

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI (`python code/scripts/launch_pipeline_gui.py`) and select Analysis Workflow
- [ ] Verify table has 3 columns, no horizontal scrolling needed
- [ ] Click each of the 6 tasks and verify correct options appear:
  - `summarize_session_blocks`: just `force_processing` checkbox
  - `touch_feature_extraction`: `force_processing` + 9-feature checkbox grid (MoS has "..." button)
  - `touch_clustering`: `force_processing` + feature combinations (with "+") + clustering profiles
  - `touch_comparing`: `force_processing` + combinations + clustering profiles + comparing profiles + 2 scalars
  - `analyse_ap_efficacy`: just `force_processing`
  - `map_receptive_fields`: `grouping_columns` (complex) + `monitor` + `force_processing`
- [ ] Toggle feature checkboxes, verify model updates
- [ ] Click "..." on mechanics_of_solids, edit params, verify YAML roundtrip
- [ ] Add a new feature combination via "+", verify it appears
- [ ] Right-click delete a combination, confirm dialog works
- [ ] Toggle profile enabled states, verify model updates
- [ ] Save (Ctrl+S), reload — verify YAML preserves comments and formatting
- [ ] Switch to Preprocess Kinect Auto workflow — verify compact detail panels
- [ ] Switch to other simple workflows — verify no crashes
- [ ] Run the Analysis Workflow — verify pipeline consumes saved config correctly

### Edge Cases
- [ ] Task with no options (empty `options:` dict) — detail panel shows "No options" or is empty
- [ ] Task with only `force_processing` — detail panel shows single checkbox
- [ ] Rapid row switching — detail panel rebuilds cleanly without artifacts

---

## Documentation Plan

- [ ] Update `CLAUDE.md` if any architectural patterns change (unlikely — internal refactor)
- [ ] No user-facing documentation needed (GUI improvement, same functionality)

---

## Rollback Plan

1. `task_panel.py` is the only file modified — revert via `git checkout` of that single file
2. No data migrations, no config format changes, no breaking API changes
3. `DagConfigModel`, dialogs, and `LauncherWindow` are untouched

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Detail panel layout looks cramped for tasks with many options | Medium | Low | Wrap in `QScrollArea`; use section headers and spacing |
| Signal recursion when detail panel and table interact | Medium | Medium | Apply `blockSignals` pattern per knowledge base note |
| Loss of side-by-side option comparison across tasks | Low | Low | This comparison is rarely useful given heterogeneous tasks; users can click between rows |
| Handler factory closures leak references on rapid re-selection | Low | Medium | Clear all child widgets in `show_task()` before rebuilding; use `deleteLater()` |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- Current implementation: `code/src/utils/gui/dag_launcher/task_panel.py` (654 lines)
- Config reference: `configs/analyse_workflow_dag.yaml`
