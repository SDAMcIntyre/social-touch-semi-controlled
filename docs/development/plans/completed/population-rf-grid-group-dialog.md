# Plan: Population RF Grid — Group Dialog with Extracted-Feature Selection

**Created:** 2026-05-12
**Started:** 2026-05-12
**Approved:** —
**Completed:** 2026-05-15 08:45
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/stabilise-handmesh-pose`
**Branch:** `feature/population-rf-grid-group-dialog`

---

## Overview

**What:** Replace the dialog used to create / edit entries in the
`map_population_rf_grid` task's `grid_groups` option with a dedicated
`GridGroupDialog`. The new dialog lets the user compose fully-qualified
feature names of the form `<data_type>_<aggregation_or_extractor>`
(e.g. `hand_velocity_amplitude_mean_during_iff`) and edit the four
per-feature bounds (`min`, `max`, `step`, `span`) plus the group-level
scalars (`enabled`, `neuron_mode`, `per_gesture_type`,
`vertex_threshold_ratio`, `compute_baseline`).

**Why:** Today, clicking the "+" button on `grid_groups` opens the generic
`FeatureCombinationDialog`, which only offers a flat list of 9 items
(7 aggregations + 2 extractor names) and writes the result back as a flat
`list[str]`. This both (a) prevents the user from selecting extracted
features (the explicit ask) and (b) silently destroys the per-feature
`{min, max, step, span}` bounds the pipeline requires.

**How:** Add a new `GridGroupDialog` modelled after `ClusterGroupDialog`
(data_type × feature matrix), a structural detector
`_is_grid_groups_dict`, and a new section builder in
`task_detail_panel.py`. Add `get_grid_group_spec` /
`set_grid_group_spec` to the model so the bounds dict round-trips
correctly.

## Problem Statement

1. **Cannot select extracted features.** The user wants to add
   `hand_velocity_amplitude_mean_during_iff` (data_type +
   `mean_during_iff` extractor) to a grid group via the GUI. The current
   dialog only shows `max`, `mean`, `median`, `min`, `range`, `skewness`,
   `std`, `mean_during_iff`, `mean_before_iff` as a flat list — no
   data_type prefix can be composed.
2. **Silent data loss on save.** `FeatureCombinationDialog` returns a
   flat list and `set_combination_features`
   (`dag_config_model.py:213`) coerces the `features` value to a
   `CommentedSeq`. The bounds dict (`{min, max, step, span}` per feature
   key) is overwritten without warning, leaving a YAML that
   `PopulationRFGridConfig` cannot consume.
3. **Group-level scalars are not editable in the GUI.** Fields like
   `neuron_mode`, `per_gesture_type`, `vertex_threshold_ratio`,
   `compute_baseline` currently require manual YAML editing.

## Goals

### In Scope

1. New `GridGroupDialog` that produces a valid `grid_groups` entry,
   including bounds dict and group-level scalars.
2. Structural dispatch in `task_detail_panel.py` that routes
   `grid_groups`-shaped option values to the new dialog without affecting
   other tasks that still use `FeatureCombinationDialog`.
3. Model functions (`get_grid_group_spec`, `set_grid_group_spec`) that
   round-trip the shape correctly with `ruamel.yaml` flow-style for the
   inner bounds dicts.
4. Edit-mode pre-population that parses existing `<data_type>_<feature>`
   keys back into `(data_type, feature)` pairs via greedy
   longest-prefix match against the shared data-type list.
5. Fail-fast validation: raise `ValueError` if a stored feature key
   cannot be parsed, or if numeric fields are missing / non-numeric.

### Out of Scope

- Changing the shape of the on-disk YAML (the new dialog reads and writes
  the existing schema).
- Modifying any other dialog or task. `FeatureCombinationDialog` keeps
  its current behaviour for tasks that legitimately use a flat-list
  `features` shape.
- Adding new feature extractors. The dialog reads `AGGREGATION_NAMES` and
  `EXTRACTOR_REGISTRY` as they exist today.
- Auto-suggesting reasonable defaults for `min`/`max`/`step`/`span` based
  on the underlying data distribution — defaults are static
  placeholders.

## Success Criteria

- [ ] Clicking "New Group…" or "Edit" on a `grid_groups` entry opens
      `GridGroupDialog`, not `FeatureCombinationDialog`.
- [ ] The features matrix exposes every entry in
      `AGGREGATION_NAMES ∪ EXTRACTOR_REGISTRY` per data_type.
- [ ] Selecting `hand_velocity_amplitude` × `mean_during_iff` and saving
      produces a YAML entry whose features dict contains
      `hand_velocity_amplitude_mean_during_iff: {min: ..., max: ...,
      step: ..., span: ...}`.
- [ ] Opening the existing `velocity_pressure_2d` group, making no
      changes, and saving produces a byte-for-byte identical YAML
      (modulo formatter quirks).
- [ ] Saving an invalid entry (e.g. `min ≥ max`, missing field,
      unparseable feature key) raises an error visible in the dialog
      rather than producing a broken YAML.
- [ ] All existing tests in `code/tests/` still pass.
- [ ] No regression in tasks that use the legacy
      `FeatureCombinationDialog`.

---

## Technical Design

### Approach

Add a new dialog and dispatch branch; keep the generic dialog intact.
The dispatch is **structural** (not key-name based) so that future tasks
sharing the grid-groups shape automatically use the new widget. The new
dialog mirrors the `ClusterGroupDialog` matrix UX so the user has one
mental model for two-dimensional feature selection.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Add a new `GridGroupDialog` and dispatch on a structural detector | Keeps existing `FeatureCombinationDialog` intact; output shape can be a full dict; UX matches `ClusterGroupDialog` | Two dialog classes to maintain; need a refactor to share `_DATA_TYPES` | **Chosen** |
| Extend `FeatureCombinationDialog` with a context flag | Single dialog | Two output shapes (list vs dict) widen the contract; callers branch on context; risk of breaking other tasks | Rejected |
| Open the raw YAML editor on the "+" click | Trivial to implement | No data-type × feature scaffolding; user gets no validation; defeats the point of the GUI | Rejected |
| Special-case the option key name `grid_groups` | Simple condition | Fragile to renames; structural detection is more robust | Rejected |

### Architecture Constraints

- **Qt signal recursion** — see
  [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md).
  When pre-populating widgets in edit mode, every `setChecked` /
  `setText` call may fire its `toggled` / `editingFinished` signal and
  bounce back through the same handler. Guard pre-population blocks with
  `widget.blockSignals(True/False)`.
- **Fail-fast pipeline convention** (root `CLAUDE.md`) — no silent
  defaults when parsing existing feature keys or numeric fields; raise
  loudly with a message that surfaces in the dialog.
- **YAML handling** (root `CLAUDE.md`) — use `ruamel.yaml` round-trip
  primitives. Each inner bounds dict is written as a
  `CommentedMap` with flow style so the YAML diff stays minimal.
- **CuPy import order** — not applicable; this is GUI-only code with no
  numpy / cupy entry-point.

### Architecture Changes

```
code/src/utils/gui/dag_launcher/
├── _feature_catalog.py          (NEW — shared _DATA_TYPES)
├── cluster_group_dialog.py      (MOD — import _DATA_TYPES from shared)
├── feature_combination_dialog.py (UNCHANGED)
├── grid_group_dialog.py         (NEW — GridGroupDialog,
│                                       GridGroupReadOnlyDialog)
└── task_detail_panel.py         (MOD — add _is_grid_groups_dict and
                                         _make_grid_groups_section,
                                         and the dispatch branch)

code/src/utils/pipeline/
└── dag_config_model.py          (MOD — add get_grid_group_spec,
                                         set_grid_group_spec)
```

---

## Implementation Plan

### Phase 1: Shared data-type catalog
**Goal:** Lift `_DATA_TYPES` out of `cluster_group_dialog.py` so both
dialogs read a single source of truth.

- [x] Task 1.1 — Create `_feature_catalog.py` exporting `DATA_TYPES`.
- [x] Task 1.2 — Update `cluster_group_dialog.py` to import from it;
      keep the existing `_DATA_TYPES` symbol as a backward-compat alias
      within the module so internal references don't churn.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/_feature_catalog.py` — NEW; defines
  `DATA_TYPES: list[str]` (the existing 15 entries).
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` — replace
  the inline list (lines 27-43) with the imported constant.

**Dependencies:** None

### Phase 2: Model functions
**Goal:** Give the GUI a safe way to read/write a grid group entry
without going through the destructive `set_combination_features`.

- [x] Task 2.1 — Add `get_grid_group_spec(task, opt_key, name) -> dict`
      mirroring `get_cluster_group_spec` at
      `dag_config_model.py:242`.
- [x] Task 2.2 — Add
      `set_grid_group_spec(task, opt_key, name, spec)` that overwrites
      the named entry while preserving CommentedMap ordering. Each
      `spec["features"][feature_name]` must be written as a
      flow-style `CommentedMap`.
- [x] Task 2.3 — Add unit tests for round-tripping a representative
      group through model load → set → dump.

**Files Modified:**
- `code/src/utils/pipeline/dag_config_model.py` — add the two new
  methods; do not modify `add_combination` or
  `set_combination_features`.
- `code/tests/utils/pipeline/test_dag_config_model.py` (or a new
  `test_grid_group_spec.py`) — add the round-trip tests.

**Dependencies:** None

### Phase 3: `GridGroupDialog`
**Goal:** Implement the actual dialog.

- [x] Task 3.1 — Create
      `code/src/utils/gui/dag_launcher/grid_group_dialog.py` with
      `GridGroupDialog` and `GridGroupReadOnlyDialog` classes.
- [x] Task 3.2 — Name section: `QLineEdit` with the same kebab/snake
      sanitiser used in `cluster_group_dialog.py:444`.
- [x] Task 3.3 — Group-level scalars section: `enabled` (QCheckBox),
      `neuron_mode` (QComboBox sourced from `_VALID_NEURON_MODES` at
      `rf_population_grid_pipeline.py:19`), `per_gesture_type`
      (QCheckBox), `vertex_threshold_ratio` (QLineEdit, float),
      `compute_baseline` (QCheckBox).
- [x] Task 3.4 — Features matrix: one row per `DATA_TYPES` entry, each
      with a parent checkbox + indented sub-row of checkboxes for
      `sorted(AGGREGATION_NAMES) + sorted(EXTRACTOR_REGISTRY)`. For each
      checked `(data_type, feature)` pair, render four small
      `QLineEdit`s for `min`/`max`/`step`/`span` (default
      `{0, 1, 0.1, 0.2}`). Composed key is `f"{data_type}_{feature}"`.
- [x] Task 3.5 — Edit-mode pre-population: parse each existing feature
      key by greedy longest-prefix match against `DATA_TYPES`. Wrap the
      whole pre-population in `blockSignals(True/False)` per
      [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md).
      Raise `ValueError` if a key cannot be parsed.
- [x] Task 3.6 — `_on_ok` validation: name non-empty; ≥1 feature
      checked; all four numeric fields parse as `float`; `min < max`,
      `step > 0`, `span > 0`. Errors surface in an inline red label
      like the existing dialogs.
- [x] Task 3.7 — `get_group_spec()` returns the dict in the shape
      `PopulationRFGridConfig` expects (excluding the `enabled` key
      that lives alongside `features`).

**Files Modified:**
- `code/src/utils/gui/dag_launcher/grid_group_dialog.py` — NEW.

**Dependencies:** Phase 1, Phase 2

### Phase 4: Dispatch in `task_detail_panel.py`
**Goal:** Route `grid_groups` to the new dialog.

- [x] Task 4.1 — Add `_is_grid_groups_dict(val)` next to
      `_is_cluster_groups_dict` (around line 97):
      a non-empty dict-of-dicts whose every entry has a `features`
      dict whose values are themselves dicts containing
      `min`, `max`, `step`, `span` keys with numeric values.
- [x] Task 4.2 — Add a dispatch branch at line 221, **before** the
      `_is_feature_combinations_dict` branch:
      ```python
      elif _is_grid_groups_dict(val):
          widget = self._make_grid_groups_section(key, val)
      ```
- [x] Task 4.3 — Implement `_make_grid_groups_section` modelled on
      `_make_cluster_groups_section` (line 494): per-row enabled
      `QCheckBox`, summary `QLabel`, Edit / Delete buttons,
      "New Group…" button at the bottom. Reuse
      `add_combination` / `remove_combination` for add and delete.
- [x] Task 4.4 — Wire the Edit / New handlers to `GridGroupDialog`
      and persist via `set_grid_group_spec`.
- [x] Task 4.5 — Write a one-line summary helper for the row label
      (e.g. `"2 features, neuron_mode=iff"`).

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add the
  detector, the dispatch branch, and the section builder + handler
  factories. Do not touch the existing combination branches.

**Dependencies:** Phase 1, Phase 2, Phase 3

---

## Testing Plan

### Unit Tests

- [ ] `set_grid_group_spec` round-trips a representative group with two
      features and inline-flow bounds, producing a YAML diff of zero
      bytes against a golden file.
- [ ] Greedy longest-prefix parser correctly resolves
      `hand_velocity_amplitude_mean_during_iff` →
      `("hand_velocity_amplitude", "mean_during_iff")` and rejects
      `unknown_prefix_mean` with `ValueError`.
- [ ] `_is_grid_groups_dict` returns `True` for a real `grid_groups`
      value, `False` for `cluster_groups` (has `clustering_methods`),
      `False` for a flat `feature_combinations` value, `False` for an
      empty dict.

### Integration Tests

- [ ] Load `configs/analyse_workflow_processing_dag.yaml` through
      `DagConfigModel`, call `get_grid_group_spec` →
      `set_grid_group_spec` with the same value, dump, diff: expect no
      changes.
- [ ] Build a `PopulationRFGridConfig` directly from the dict produced
      by `GridGroupDialog.get_group_spec()` and confirm
      no exceptions.

### Manual Verification

- [ ] Run `python code/scripts/launch_pipeline_gui.py`, open
      "Analysis Workflow Processing", click `map_population_rf_grid`,
      click Edit on `velocity_pressure_2d`. Confirm: name, scalar
      params, and both features (with bounds) populate correctly. Save
      with no changes; `git diff configs/analyse_workflow_processing_dag.yaml`
      shows nothing.
- [ ] Click "New Group…", name it `test_extracted_features`, select
      `hand_velocity_amplitude` × `mean_during_iff` and `pressure` ×
      `mean_during_iff`, set bounds, save. Inspect the YAML: feature
      keys are `hand_velocity_amplitude_mean_during_iff` and
      `pressure_mean_during_iff`, inline flow style.
- [ ] Run the task: `python code/scripts/analysis_workflow.py` (or
      whatever the GUI launches) and confirm the pipeline consumes the
      new group without raising.

### Edge Cases

- [ ] Edit-mode opening a group whose `features` dict is empty —
      dialog opens cleanly with no rows pre-checked, OK requires
      at least one feature.
- [ ] Editing a group whose feature key contains underscores in both
      the data_type and the feature (e.g.
      `hand_velocity_amplitude_mean_during_iff`) — greedy
      longest-prefix match resolves correctly.
- [ ] Numeric field with whitespace or trailing comma — validator
      strips, parses, or rejects.
- [ ] Cancelling the dialog leaves the YAML untouched.

---

## Documentation Plan

- [ ] Update the inline docstring of `GridGroupDialog` to reference
      `PopulationRFGridConfig` so future maintainers find the consumer.
- [ ] No changes to `README.md` or root `CLAUDE.md` — this is an
      internal GUI affordance.
- [ ] No new knowledge-base note expected unless the Qt
      `blockSignals` pattern needs reinforcing; if so, extend
      `note-qt-itemchanged-signal-recursion.md` rather than creating a
      new note.

---

## Rollback Plan

1. **Before merge:** the change is additive and lives entirely in GUI
   and model code. To revert, drop the new file and undo the
   dispatch branch + model methods.
2. **Data considerations:** no migration. The YAML format is unchanged;
   any user who already hand-edited their `grid_groups` will continue
   to load.
3. **Rollback procedure:** `git revert` the merge commit. There is no
   schema or on-disk state to restore.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Greedy longest-prefix parsing collides with a future `data_type` name that contains an aggregation suffix | Low | Med | Tests cover known collisions; raising `ValueError` on ambiguity surfaces the issue at edit time, not pipeline time. |
| Qt signals fire during pre-population and corrupt state | Med | Low | `blockSignals` guard around the population block; manual test covers this. |
| `_is_grid_groups_dict` accidentally matches some other task's option | Low | Med | Detector is strict (inner values must each have all four of `min`/`max`/`step`/`span`); grep across the YAML for any matching shape during code review. |
| Round-trip YAML formatting differs slightly from the existing inline-flow style | Med | Low | Golden-file round-trip test against `analyse_workflow_processing_dag.yaml`. |

---

## References

- Related knowledge-base notes:
  [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md)
- Key source files:
  - `code/src/utils/gui/dag_launcher/task_detail_panel.py` (lines
    97, 217-222, 494)
  - `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` (lines
    27-45, 269-547)
  - `code/src/utils/gui/dag_launcher/feature_combination_dialog.py`
  - `code/src/utils/pipeline/dag_config_model.py` (lines 181, 213, 242)
  - `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py`
    (lines 19, 22-28)
  - `code/src/analysis/touch_analytics/representation/feature_characterization/__init__.py`
    (lines 7-12)
- Target config: `configs/analyse_workflow_processing_dag.yaml`
  (the `map_population_rf_grid.options.grid_groups` block).
