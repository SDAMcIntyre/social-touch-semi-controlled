# Plan: Editable Non-Boolean Task Parameters in DAG Launcher GUI

**Created:** 2026-03-17
**Approved:** ---
**Completed:** 2026-03-18 11:29
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

**What:** Make non-boolean task options (int, float, string, list, dict) editable in the DAG launcher GUI, which currently only supports boolean checkbox toggles.

**Why:** Users cannot modify key tuning parameters from the GUI --- they must edit YAML files directly. This affects the analysis workflow most heavily, where `extraction_profiles`, `clustering_profiles`, scalar thresholds, and list-valued options are all locked as read-only grey text.

**How:** A two-tier editing strategy: inline cell editing for simple scalars (int, float, str) via double-click, and a YAML text editor dialog for complex types (list, dict) via click-to-open.

## Problem Statement

The `TaskPanel` (`task_panel.py`) renders task options in a QTableWidget. The cell rendering logic (lines 156--177) only creates editable checkboxes for `bool` values. All other types are displayed as read-only `QTableWidgetItem` with grey foreground and no edit flags.

This means the following parameters across all workflows are inaccessible:

| Workflow | Task | Parameter | Type |
|----------|------|-----------|------|
| Analysis | `unified_touch_analysis` | `extraction_profiles` | nested dict |
| Analysis | `unified_touch_analysis` | `clustering_profiles` | nested dict |
| Analysis | `map_receptive_fields` | `grouping_columns` | list of strings |
| Visualization | `view_neural_kinect_scene` | `crop_half_size_mm` | float |
| Visualization | `view_neural_kinect_scene_transformed` | `crop_half_size_mm` | float |

Within the nested dicts, important tunable scalars like `youngs_modulus_kpa`, `min_touches_per_cluster`, `eps`, `skin_thickness_mm`, and `aggregations` are also unreachable.

## Goals

### In Scope

1. Inline editing (double-click) for scalar task options: `int`, `float`, `str`
2. YAML text editor dialog (click-to-open) for complex task options: `list`, `dict`
3. Type-safe coercion for inline edits (coerce back to original Python type, revert on failure)
4. YAML syntax validation in the dialog (reject unparseable YAML, show error inline)
5. Round-trip fidelity via ruamel.yaml (preserve comments, key order, quoting style)
6. Visual distinction between editable scalars, clickable complex values, and existing boolean checkboxes

### Out of Scope

- Semantic validation of parameter values (e.g., checking `method` against extractor/clusterer registries)
- Structured form editors for specific parameter schemas (e.g., dedicated extractor config forms)
- Adding or removing task options from the GUI (only editing existing values)
- Adding or removing profiles within `extraction_profiles` / `clustering_profiles` (users can do this in the YAML dialog, but there is no dedicated "add profile" button)

## Success Criteria

- [ ] Double-clicking a scalar cell (int/float/str) enters edit mode; committing writes to the model and marks it dirty
- [ ] Typing an invalid value for the original type (e.g., "abc" for a float) reverts the cell without writing to the model
- [ ] Clicking a complex-value cell (list/dict) opens a YAML editor dialog pre-populated with the current value
- [ ] The dialog rejects invalid YAML with a visible error message and stays open
- [ ] Accepting valid YAML in the dialog writes the parsed value to the model and updates the cell preview
- [ ] Saving after any edit preserves YAML comments, key order, and formatting (ruamel round-trip)
- [ ] Boolean checkboxes continue to work unchanged
- [ ] All existing workflows load and display correctly with the new rendering logic

---

## Technical Design

### Approach

Extend the existing `TaskPanel` cell-rendering branch to dispatch on value type rather than treating all non-booleans identically. Simple scalars become inline-editable table cells; complex types become clickable cells that open a YAML editor dialog.

This approach was chosen because it adds editing capability with minimal new code, reuses the existing `DagConfigModel.set_task_option()` API unchanged, and avoids building structured form editors that would couple the GUI to pipeline internals.

### Validation Strategy

**Syntax-only (Level 0):** The GUI validates that edited YAML parses correctly but does not validate semantic correctness (e.g., whether a `method` name exists in a registry or whether a numeric value is in a valid range). Invalid semantic values will fail at pipeline runtime with clear error messages. This keeps the GUI fully decoupled from pipeline internals --- new extractors, clusterers, or task options work in the GUI automatically without any GUI code changes.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two-tier: inline scalars + YAML dialog | Simple, flexible, no coupling to pipeline internals, handles all types | No rich UX for nested structures, no semantic validation | **Chosen** |
| Structured form per task type | Best UX, full validation | High effort, tight coupling, must update forms when pipeline changes | Rejected --- maintenance cost too high for a research project |
| Tree widget for nested dicts | Visual hierarchy, expand/collapse | Complex to implement, awkward for mixed types, still needs inline editing for leaves | Rejected --- YAML dialog is simpler and equally powerful |
| Keep read-only, edit YAML directly | Zero effort | Defeats purpose of GUI, error-prone for researchers | Rejected --- this is the current problem |

### Architecture Changes

```
code/src/utils/gui/dag_launcher/
├── launcher_window.py              # UNCHANGED
├── workflow_selector.py            # UNCHANGED
├── kinect_directory_selector.py    # UNCHANGED
├── launcher_config.py              # UNCHANGED
├── task_panel.py                   # MODIFIED — type-dispatched rendering, inline editing, click-to-edit
└── yaml_edit_dialog.py             # NEW — QDialog for YAML fragment editing
```

No changes to `DagConfigModel` --- `set_task_option(task_name, option, value: Any)` already accepts arbitrary types and persists them correctly via ruamel.yaml.

#### Existing Patterns Reused

- **Handler factory closures** (`_make_enabled_handler`, `_make_option_handler`) --- extend with `_make_scalar_handler` for inline edits
- **Centred cell widget** (`_centred_checkbox`) --- reuse for boolean cells (unchanged)
- **Signal flow**: widget change -> handler -> `set_task_option()` -> `_dirty = True` -> emit `task_changed`

---

## Implementation Plan

### Phase 1: YAML Edit Dialog
**Goal:** Create a self-contained dialog for editing complex YAML values (list, dict).

- [ ] Task 1.1 --- Create `yaml_edit_dialog.py` with `YamlEditDialog(QDialog)`:
  - Constructor: `__init__(self, task_name: str, option_key: str, current_value: Any, parent: QWidget)`
  - Serialize `current_value` to YAML string via `ruamel.yaml.YAML().dump()` into `StringIO`
  - Display in `QPlainTextEdit` with monospace font (`Consolas` with `Courier` fallback)
  - Error label (`QLabel`) below text area, hidden by default, red text on parse failure
  - On OK: parse with `YAML().load()`, show error if invalid, reject dialog (keep open)
  - On success: store parsed value, accept dialog
  - Public method `get_value() -> Any` returns the parsed ruamel round-trip object
  - Dialog title: `"Edit {option_key} --- {task_name}"`
  - Min size: 500x400, resizable

**Files Created:**
- `code/src/utils/gui/dag_launcher/yaml_edit_dialog.py`

**Dependencies:** None

### Phase 2: Complex Value Cells (Tier 2)
**Goal:** Make list/dict option cells clickable to open the YAML dialog.

- [ ] Task 2.1 --- Add `_preview_text(val: Any) -> str` helper to `task_panel.py`:
  - `list` -> `"[item1, item2, ...] (N items)"` (show first 3 items + count)
  - `dict` -> `"{key1, key2, ...} (N keys)"` (show first 3 keys + count)
- [ ] Task 2.2 --- In `populate()`, split the non-bool else branch (lines 167--171):
  - `isinstance(val, (int, float, str))` -> Tier 1 handling (Phase 3)
  - else (list/dict) -> blue-ish foreground (`#336699`), tooltip "Double-click to edit in YAML editor", store `("complex", opt_key)` in `Qt.UserRole`
- [ ] Task 2.3 --- Extend `_on_cell_clicked()`:
  - If clicked cell is in option column range and has `("complex", opt_key)` user data:
    - Retrieve current value via `self._model.get_task_option(task_name, opt_key)`
    - Open `YamlEditDialog(task_name, opt_key, current_value, self)`
    - On `QDialog.Accepted`: call `set_task_option()`, update cell preview text, emit `task_changed`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` --- new helper, modified populate, extended click handler

**Dependencies:** Phase 1

### Phase 3: Inline Scalar Editing (Tier 1)
**Goal:** Make int/float/str option cells directly editable via double-click.

- [ ] Task 3.1 --- Change edit triggers from `NoEditTriggers` to `DoubleClicked` (line 83). Non-editable cells (task name, depends-on, complex values, N/A) already lack `Qt.ItemIsEditable` flag, so they remain protected.
- [ ] Task 3.2 --- In `populate()`, for `isinstance(val, (int, float, str))` options:
  - Create `QTableWidgetItem(str(val))` with `Qt.ItemIsEditable` flag added
  - Store original typed value in `Qt.UserRole` for type coercion on commit
- [ ] Task 3.3 --- Connect `self._table.itemChanged` signal to new `_on_item_changed()` handler
- [ ] Task 3.4 --- Implement `_on_item_changed(item: QTableWidgetItem)`:
  - Skip if `self._model is None` (signals already blocked during populate)
  - Map cell position to `(task_name, opt_key)` via `_row_task[row]` and `_option_keys[col - 2]`
  - Retrieve original type from `item.data(Qt.UserRole)`
  - Coerce new text to original type: `int()`, `float()`, or `str`
  - On coercion failure: block signals, revert cell text to `str(original)`, unblock signals, return
  - On success: `set_task_option()`, update `Qt.UserRole`, emit `task_changed`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` --- edit triggers, scalar cell flags, `itemChanged` handler

**Dependencies:** Phase 2 (so that the type-dispatch in `populate()` is already in place)

---

## Testing Plan

### Manual Verification

- [ ] Launch GUI with `analyse_workflow_dag.yaml` --- verify all 4 tasks render correctly with the new type-dispatched cells
- [ ] **Scalar editing:** double-click `crop_half_size_mm` (400.0) in a visualization workflow, change to 300.0, save, reopen YAML --- verify value persisted
- [ ] **Type coercion rejection:** type "abc" in a float cell --- verify cell reverts to original value
- [ ] **Complex editing:** click `extraction_profiles` cell in `unified_touch_analysis` --- verify dialog opens with correct YAML content
- [ ] **Dialog validation:** type broken YAML in dialog, click OK --- verify error label appears and dialog stays open
- [ ] **Dialog accept:** edit a value in dialog (e.g., change `min_touches_per_cluster: 30` to `50`), click OK --- verify cell preview updates and model is dirty
- [ ] **Boolean unchanged:** toggle `force_processing` checkbox --- verify it still works as before
- [ ] **Round-trip fidelity:** save after edits, diff YAML file --- verify comments, key order, and formatting are preserved
- [ ] **All workflows:** cycle through all workflows in the GUI --- verify no crashes or rendering regressions

### Edge Cases

- [ ] Scalar cell with string value `"auto"` (e.g., `eps: auto`) --- verify it remains a string, not coerced to null/bool
- [ ] Empty dict or list option --- verify dialog opens with `{}` or `[]` and accepts it
- [ ] Very long list/dict --- verify dialog is scrollable and preview text is truncated sensibly
- [ ] Task with no options --- verify no option columns appear (existing behavior preserved)
- [ ] Multiple workflows with different option keys --- verify columns are rebuilt on workflow switch

---

## Documentation Plan

- [ ] Add inline docstrings to `YamlEditDialog` class and public methods
- [ ] Add docstring to `_preview_text()`, `_on_item_changed()` helpers
- [ ] Update CLAUDE.md memory with new `yaml_edit_dialog.py` location

---

## Rollback Plan

1. **Before deployment:**
   - Only 2 files affected: `task_panel.py` (modified) and `yaml_edit_dialog.py` (new)
   - Revert the single commit on the feature branch to restore read-only behavior

2. **Data considerations:**
   - No data migrations or file format changes
   - YAML configs remain identical in structure (only values may change from user edits)

3. **Rollback procedure:**
   - Delete `yaml_edit_dialog.py`
   - Revert `task_panel.py` to previous version
   - No downstream effects on pipeline code

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Silent type coercion (e.g., `"400"` string vs `400` int) | Medium | Medium | Store original Python type in `Qt.UserRole`, coerce back to same type on commit |
| ruamel round-trip breaks on complex edits in dialog | Low | Medium | Use `YAML()` default round-trip mode for both serialize and parse; returns `CommentedMap`/`CommentedSeq` |
| `itemChanged` fires during `populate()` | Low | Low | Already mitigated by existing `blockSignals(True)` during populate |
| User edits YAML dialog to change structure (e.g., removes required key) | Medium | Low | Syntax-only validation; pipeline will report clear error at runtime |
| Double-click on complex cell triggers both edit mode and click handler | Low | Medium | Complex cells lack `Qt.ItemIsEditable` flag, so `DoubleClicked` trigger won't activate edit mode for them |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: YAML Edit Dialog | ~1 new file, ~80 LOC | None |
| Phase 2: Complex value cells | ~40 LOC changes in task_panel.py | Phase 1 |
| Phase 3: Inline scalar editing | ~50 LOC changes in task_panel.py | Phase 2 |

---

## References

- Current task panel: `code/src/utils/gui/dag_launcher/task_panel.py`
- Config model: `code/src/utils/pipeline/dag_config_model.py`
- Launcher window: `code/src/utils/gui/dag_launcher/launcher_window.py`
- Analysis DAG config: `configs/analyse_workflow_dag.yaml`
- Visualization DAG config: `configs/merging_pipeline_neuron_to_kinect_visualisation_dag.yaml`
- Completed GUI plan: `docs/development/plans/completed/dag-config-launcher-gui.md`
