# Plan: Fix Clustering GUI — Pipeline Stage Order

**Created:** 2026-04-23
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/align-clustering-pipeline-with-guidelines` (extended)

---

## Overview

**What:** Restructure the `touch_clustering` panel in the DAG launcher GUI so
that option sections are ordered and labeled to match the five-stage pipeline
defined in `docs/design/timeseries_clustering_pipeline_guidelines.md`.

**Why:** The current implementation combines Stage 3 (reduction) and Stage 5
(evaluation) into a single "Reduction & Evaluation" tabbed dialog, even though
Stage 4 (clustering) sits between them. Additionally, Stage 2 (representation)
does not expose its 2a/2b internal structure, and sections render in YAML
iteration order rather than pipeline-stage order.

**How:** Split `ClusteringConfigDialog` into two independent dialogs
(`ReductionConfigDialog`, `EvaluationConfigDialog`), add pipeline-stage
ordering and labeling logic to `TaskDetailPanel.show_task`, and wrap
`feature_combinations` inside a "Representation" group that also shows
the 2a sub-stage.

## Problem Statement

The `dag-launcher-clustering-config-widgets` plan (commit pending) added a
combined "Reduction & Evaluation" group box to the `touch_clustering` task
panel. The dialog has two tabs (Reduction, Evaluation) and a single
"Configure…" button. This deviates from the five-stage pipeline in three ways:

1. **Stage ordering violated.** The pipeline flow is
   representation (2) → reduction (3) → clustering (4) → evaluation (5).
   The GUI appends reduction + evaluation together *after* clustering_profiles,
   suggesting they are adjacent steps. They are not — clustering sits between
   them.

2. **Stage 2 internal structure invisible.** The guidelines define Stage 2 as
   two sub-stages:
   - 2a. Series-level transformations (series → series) — kinematics
     (velocity, acceleration) computed in
     `representation/series_level/kinematics.py`.
   - 2b. Feature characterization (series → scalar vector) — the extractors
     in `representation/feature_characterization/`.

   In `touch_clustering`, `feature_combinations` is the 2b output selection
   (which extracted scalar features to merge into the clustering matrix).
   The GUI labels it simply "Feature Combinations" with no indication that it
   is one half of Stage 2, and 2a is entirely absent.

3. **Stages 3 and 5 conflated.** A single dialog for two non-adjacent stages
   breaks the mental model. Users cannot independently locate "where do I
   configure scaling?" (Stage 3) vs "where do I configure stability scoring?"
   (Stage 5) — they must open the same dialog and find the right tab.

## Goals

### In Scope

1. Split `ClusteringConfigDialog` into `ReductionConfigDialog` and
   `EvaluationConfigDialog` — each a standalone `QDialog` with no tabs.
2. Order `touch_clustering` sections by pipeline stage:
   representation (2) → reduction (3) → clustering (4) → evaluation (5).
3. Wrap `feature_combinations` inside a "Representation" parent group that
   shows both 2a (informational) and 2b (`feature_combinations` content).
4. Separate "Reduction" and "Evaluation" group boxes, each with their own
   preview label and "Configure…" button.
5. Preserve all existing widget behaviour, round-trip YAML fidelity, and
   validation logic from the parent plan.

### Out of Scope

- Changes to any pipeline code (`code/src/analysis/touch_analytics/`).
- Changes to `dag_config_model.py`.
- Applying pipeline-stage ordering to tasks other than `touch_clustering`.
- Making Stage 2a configurable (it has no config knobs today).
- Re-design of existing `feature_combinations` or `clustering_profiles`
  widgets — only their ordering and parent grouping change.

## Success Criteria

- [ ] Selecting `touch_clustering` shows four stage-ordered sections:
  Representation → Reduction → Clustering Profiles → Evaluation.
- [ ] The "Representation" section shows a "2a. Series-Level Transformations"
  informational label and contains `feature_combinations` as "2b. Feature
  Characterization."
- [ ] "Reduction" has its own group box with a preview label and "Configure…"
  button that opens `ReductionConfigDialog`.
- [ ] "Evaluation" has its own group box with a preview label and "Configure…"
  button that opens `EvaluationConfigDialog`.
- [ ] No "Reduction & Evaluation" combined section exists.
- [ ] Opening either dialog and pressing OK without changes produces no diff.
- [ ] Entering invalid threshold in the Reduction dialog shows an inline error.
- [ ] No regression: `feature_combinations`, `clustering_profiles`, and
  `camera_angle_mode` still render and save correctly.

---

## Technical Design

### Approach

Replace the combined `ClusteringConfigDialog` with two focused dialogs and
add a pipeline-stage ordering mechanism to `show_task` for `touch_clustering`.

**Dialog split.** The existing `ClusteringConfigDialog` already has cleanly
separated `_build_reduction_tab` / `_build_evaluation_tab` methods and
independent `get_reduction` / `get_evaluation` getters. The split is
mechanical: each becomes its own `QDialog` subclass with the tab's content
rendered directly in a `QVBoxLayout` (no `QTabWidget`).

**Stage ordering.** `show_task` currently iterates `options.items()` (YAML
order: `feature_combinations`, `clustering_profiles`, `reduction`,
`evaluation`) and appends widgets sequentially, with `reduction`/`evaluation`
skipped and appended as a combined section at the end. The fix:

1. Define `_TOUCH_CLUSTERING_STAGE_ORDER`, a list of keys in pipeline order:
   `["feature_combinations", "reduction", "clustering_profiles", "evaluation"]`.
2. When the task is `touch_clustering`, iterate keys in that order instead
   of YAML order. Unknown keys fall through to the default dispatch at the
   end.
3. `feature_combinations` is wrapped in a `_make_representation_section()`
   that adds the 2a info label and nests the existing combination widgets
   under "2b."
4. `reduction` dispatches to `_make_reduction_section()` (preview +
   Configure → `ReductionConfigDialog`).
5. `evaluation` dispatches to `_make_evaluation_section()` (preview +
   Configure → `EvaluationConfigDialog`).

**Stage 2a label.** A read-only `QLabel` inside the Representation group
stating: *"Series-level transformations (velocity, acceleration) — configured
in touch_feature_extraction"*. This is informational only — no widgets,
no config writes.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Two separate dialogs, pipeline-ordered sections** | Matches guidelines 1:1; clean separation; minimal dialog logic | Extra group boxes (more vertical space) | **Chosen** |
| Keep combined dialog but reorder sections in panel | Smaller diff | Still conflates stages 3+5 in a single dialog; misleading | Rejected |
| Inline all reduction/evaluation widgets (no dialog) | Most discoverable | Nullable sub-maps need conditional show/hide inline; crowds the panel; rejected in parent plan for same reason | Rejected |
| Stage-number prefixes on group titles ("3. Reduction") | Very explicit mapping | Clutters titles; couples GUI labels to doc numbering | Rejected — stage order is implicit from layout |

### Architecture Constraints (from knowledge base)

- **Qt signal recursion** — `note-qt-itemchanged-signal-recursion.md`:
  Same assessment as parent plan. Dialog `_on_ok` handlers use `accepted`
  signal, no `itemChanged` recursion risk.
- **CuPy import order** — N/A (GUI-only).

### Architecture Changes

```
code/src/utils/gui/dag_launcher/
├── clustering_config_dialog.py   # MODIFIED — split into two dialog classes
├── task_detail_panel.py          # MODIFIED — stage ordering + new section builders
├── feature_combination_dialog.py # unchanged
└── yaml_edit_dialog.py           # unchanged
```

Public API changes in `clustering_config_dialog.py`:

```python
# REMOVED
class ClusteringConfigDialog(QDialog): ...

# ADDED
class ReductionConfigDialog(QDialog):
    def __init__(self, task_name: str, *, reduction_cfg: dict | None,
                 parent: QWidget | None = None) -> None: ...
    def get_reduction(self) -> dict: ...

class EvaluationConfigDialog(QDialog):
    def __init__(self, task_name: str, *, evaluation_cfg: dict | None,
                 parent: QWidget | None = None) -> None: ...
    def get_evaluation(self) -> dict: ...
```

---

## Implementation Plan

### Phase 1: Split dialog into two classes
**Goal:** Replace `ClusteringConfigDialog` with `ReductionConfigDialog` and
`EvaluationConfigDialog`, each rendering its content directly (no tabs).
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 1.1 — Rename `ClusteringConfigDialog` to `ReductionConfigDialog`.
  Remove `QTabWidget`, render reduction widgets directly in the main layout.
  Remove `_build_evaluation_tab`, `get_evaluation`, and the `evaluation_cfg`
  parameter. Keep `_on_ok` validation for threshold.
- [x] Task 1.2 — Add `EvaluationConfigDialog` in the same file. Move
  `_build_evaluation_tab` logic into its `__init__`. Add `get_evaluation`
  getter. No numeric validation needed (spin boxes handle constraints).
- [x] Task 1.3 — Update module-level `__all__` / exports. Remove the
  `ClusteringConfigDialog` name.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/clustering_config_dialog.py` — rewrite.

**Dependencies:** None.

### Phase 2: Pipeline-ordered panel layout
**Goal:** `show_task` renders `touch_clustering` sections in pipeline-stage
order with separate Reduction and Evaluation group boxes.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 2.1 — Define `_TOUCH_CLUSTERING_STAGE_ORDER` at module level:
  `["feature_combinations", "reduction", "clustering_profiles", "evaluation"]`.
- [x] Task 2.2 — In `show_task`, when task is `touch_clustering`, iterate
  options keys in the stage-order list (falling back to YAML order for
  unlisted keys like `show_interactive`).  Remove the `has_clustering_config`
  flag and the combined-section append.
- [x] Task 2.3 — Add dispatch branches for `reduction` and `evaluation`
  keys in the options loop.  `reduction` → `_make_reduction_section()`;
  `evaluation` → `_make_evaluation_section()`.
- [x] Task 2.4 — Add `_make_reduction_section()`: group box titled
  "Reduction", preview label from `_reduction_preview()`, "Configure…"
  button → `_make_reduction_handler()` → opens `ReductionConfigDialog`.
- [x] Task 2.5 — Add `_make_evaluation_section()`: group box titled
  "Evaluation", preview label from `_evaluation_preview()`, "Configure…"
  button → `_make_evaluation_handler()` → opens `EvaluationConfigDialog`.
- [x] Task 2.6 — Replace `_clustering_preview()` with two helpers:
  `_reduction_preview(cfg)` and `_evaluation_preview(cfg)`.
- [x] Task 2.7 — Remove `_make_clustering_config_section`,
  `_make_clustering_config_handler`, and the `ClusteringConfigDialog` import.
  Add imports for the two new dialog classes.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — rewrite dispatch
  loop for `touch_clustering`, add new section builders and handlers.

**Dependencies:** Phase 1.

### Phase 3: Representation group with 2a/2b structure
**Goal:** Wrap `feature_combinations` inside a "Representation" parent group
that also shows Stage 2a as an informational label.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 3.1 — Add `_make_representation_section(key, val)` that returns a
  `QGroupBox("Representation")` containing:
  - A `QLabel` for 2a: *"2a. Series-level transformations (velocity,
    acceleration) — configured in touch_feature_extraction"*, styled with
    a muted colour.
  - A `QLabel("2b. Feature Characterization")` sub-header.
  - The existing `_make_combination_section(key, val)` content (the checkbox
    rows + "+" button), embedded inside the parent group instead of its own
    `QGroupBox`.
- [x] Task 3.2 — In the `touch_clustering` dispatch, route
  `feature_combinations` through `_make_representation_section` instead of
  `_make_combination_section`.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add
  `_make_representation_section`, adjust dispatch.

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests

- [ ] `test_reduction_dialog_populates_from_cfg` — construct
  `ReductionConfigDialog` with a full config, confirm widgets match.
- [ ] `test_evaluation_dialog_populates_from_cfg` — same for
  `EvaluationConfigDialog`.
- [ ] `test_reduction_dialog_invalid_threshold_rejects` — set threshold to
  `"abc"`, confirm `accept()` not called.
- [ ] `test_get_reduction_roundtrip` — set widgets, call `get_reduction()`,
  assert expected dict shape.
- [ ] `test_get_evaluation_stability_off_returns_none` — uncheck stability,
  confirm `get_evaluation()["stability"]` is `None`.

### Integration Tests

- [ ] `test_touch_clustering_section_order` — load
  `configs/analyse_workflow_dag.yaml`, call `show_task(model,
  "touch_clustering")`, assert group boxes appear in order:
  Representation → Reduction → Clustering Profiles → Evaluation.
- [ ] `test_no_combined_reduction_evaluation_section` — confirm no widget
  titled "Reduction & Evaluation" exists.
- [ ] `test_representation_section_has_2a_label` — confirm the
  Representation group contains a label mentioning "Series-level".
- [ ] `test_reduction_dialog_save_writes_yaml` — pump reduction widgets,
  accept, verify model state.
- [ ] `test_evaluation_dialog_save_writes_yaml` — same for evaluation.

### Manual Verification

- [ ] Launch GUI, select `touch_clustering`; confirm four stage-ordered
  sections appear and no "Reduction & Evaluation" section exists.
- [ ] Click "Configure…" in Reduction; toggle scaler to `robust`, OK,
  save; inspect YAML.
- [ ] Click "Configure…" in Evaluation; disable stability, OK, save;
  confirm `stability: null` in YAML.
- [ ] Open Reduction dialog and press OK without changes; `git diff` shows
  no change.
- [ ] Open Evaluation dialog and press OK without changes; `git diff` shows
  no change.
- [ ] Confirm `feature_combinations` widgets (checkboxes, feature lists,
  right-click delete, "+" add) still work inside the Representation group.
- [ ] Confirm `clustering_profiles` widgets still work.
- [ ] Select a non-clustering task (e.g. `map_receptive_fields_simple`);
  confirm it renders normally with no regression.

### Edge Cases

- [ ] Config missing `evaluation` key entirely: panel still renders
  Evaluation section with defaults; dialog opens with defaults.
- [ ] Config missing `reduction` key entirely: same for Reduction.
- [ ] User closes either dialog via window "X": no mutation.

---

## Documentation Plan

- [ ] Docstrings on `ReductionConfigDialog` and `EvaluationConfigDialog`.
- [ ] No changes to `README.md` or `CLAUDE.md` (GUI-only, no architectural
  constraint change).

---

## Rollback Plan

1. Revert the commits from this plan. Restore the combined
   `ClusteringConfigDialog` and the `_make_clustering_config_section` code
   path.
2. No data migrations — YAML schema is unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Embedding combination widgets inside the Representation parent breaks layout or context menu handling | Medium | Medium | Phase 3 tests the "+" button, right-click delete, and feature-edit click; fallback is rendering `_make_combination_section` as a nested `QGroupBox` titled "2b. Feature Characterization" instead of inlining its content |
| Stage-order list becomes stale if new config keys are added to `touch_clustering` | Low | Low | Unlisted keys fall through to default dispatch at the end; a warning log could be added if desired |
| Two separate "Configure…" dialogs are less convenient than one tabbed dialog | Low | Low | The pipeline-correct ordering is more important than one-click access; users edit reduction and evaluation at different points in their workflow |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Split dialog | 0.25 day | None |
| Phase 2 — Pipeline-ordered panel | 0.5 day | Phase 1 |
| Phase 3 — Representation group | 0.25 day | Phase 2 |

---

## References

- **Parent plan:**
  `docs/development/plans/active/dag-launcher-clustering-config-widgets.md`
- **Alignment plan:**
  `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md`
- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md` (stages 2–5,
  lines 17–27)
- **Current implementation:**
  - `code/src/utils/gui/dag_launcher/clustering_config_dialog.py` — dialog
    to split
  - `code/src/utils/gui/dag_launcher/task_detail_panel.py:176-197` — dispatch
    loop + combined section
- **Stage 2a code:**
  `code/src/analysis/touch_analytics/representation/series_level/kinematics.py`
- **Knowledge base:**
  `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
  (reviewed — does not apply directly)
