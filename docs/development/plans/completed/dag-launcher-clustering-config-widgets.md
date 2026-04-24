# Plan: DAG Launcher — Clustering Reduction & Evaluation Widgets

**Date:** 2026-04-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/align-clustering-pipeline-with-guidelines` (extended)

---

## Overview

Add a dedicated tabbed modal dialog (`ClusteringConfigDialog`) and a
section in `TaskDetailPanel` so users can edit the new `reduction` and
`evaluation` configuration blocks of `touch_clustering` from the DAG
launcher GUI instead of hand-editing YAML.

## Problem Statement

The `feature/align-clustering-pipeline-with-guidelines` branch
(commit `67c5996`) restructured `code/src/analysis/touch_analytics/`
into a five-stage pipeline and introduced two new config blocks in
`configs/analyse_workflow_dag.yaml` under `touch_clustering.options`:

```yaml
reduction:
  scaler: standard              # standard | robust | none
  variance_filter: null         # null | {threshold: float}
  decomposition: null           # null | {method: pca, n_components: int}
evaluation:
  internal_metrics: [silhouette, davies_bouldin, calinski_harabasz]
  stability:
    method: bootstrap           # bootstrap | null
    n_rounds: 20
    subsample_fraction: 0.8
    score: adjusted_rand_index
```

The alignment plan explicitly deferred GUI updates (Out of Scope,
line 51). As a result the DAG launcher has **no typed widgets** for
these blocks — they currently fall through to
`TaskDetailPanel._make_scalar_section`, which renders them as a blue
clickable label that opens `YamlEditDialog` (raw-YAML editor). Users
cannot discoverably switch scaler, enable/disable stability scoring, or
tune bootstrap parameters without editing YAML by hand.

## Goals

### In Scope
1. A new modal `ClusteringConfigDialog` with two tabs (Reduction,
   Evaluation) exposing every field of the two YAML blocks with typed
   widgets (combo boxes, spin boxes, checkboxes).
2. Dispatch-branch + section builder in `task_detail_panel.py` so the
   `touch_clustering` task renders one combined "Reduction & Evaluation"
   group box with a "Configure…" button and preview label.
3. Round-trip YAML fidelity: a no-op open → OK → save leaves the config
   file byte-identical (comments, quoting, flow style preserved).
4. Handling of nullable sub-maps: `variance_filter`, `decomposition`,
   and `stability` must serialise as `null` when disabled by checkbox,
   and as full nested dicts when enabled.
5. Input validation on numeric fields before `accept()` — invalid
   input shows an inline error and does not mutate the model.

### Out of Scope
- Changes to any pipeline code (`code/src/analysis/touch_analytics/`).
- Changes to `dag_config_model.py` — the existing `set_task_option` API
  is sufficient.
- New backend support for scalers/decomposition methods beyond what
  `reduction/pipeline.py` already accepts.
- Re-design of existing `touch_clustering` widgets
  (`feature_combinations`, `clustering_profiles`).
- Editing reduction/evaluation for tasks other than `touch_clustering`
  (these keys only exist there today).

## Success Criteria

- [ ] Selecting `touch_clustering` in the launcher shows a single
  "Reduction & Evaluation" group box and **no** separate `reduction` or
  `evaluation` rows.
- [ ] The "Configure…" button opens `ClusteringConfigDialog` with both
  blocks' current values pre-populated.
- [ ] Toggling any field → OK → Save writes the expected YAML shape
  (including `null` for disabled sub-maps) and matches a hand-edited
  reference.
- [ ] Opening the dialog and pressing OK without changes produces no git
  diff against `configs/analyse_workflow_dag.yaml`.
- [ ] Entering invalid numeric input (e.g. `threshold: abc`) shows an
  inline red error, keeps the dialog open, and does not mutate the
  model.
- [ ] An end-to-end clustering run with scaler toggled `standard` →
  `robust` completes without error and updates
  `cluster_metadata.json` accordingly.
- [ ] No regression: existing widgets (`feature_combinations`,
  `clustering_profiles`, `camera_angle_mode`) still render and save
  correctly after touching this code.

---

## Technical Design

### Approach

One modal dialog with two tabs, launched from a single combined
"Reduction & Evaluation" group box. This mirrors the existing
`FeatureCombinationDialog` precedent
(`code/src/utils/gui/dag_launcher/feature_combination_dialog.py`) and
keeps `task_detail_panel.py` free of the nested, nullable,
conditionally-visible widget logic these blocks require.

Write-back uses the existing `DagConfigModel.set_task_option` API
(`code/src/utils/pipeline/dag_config_model.py:139`). For the
`internal_metrics` list we rebuild a `CommentedSeq` with
`seq.fa.set_flow_style()` — same pattern as
`set_combination_features` (`dag_config_model.py:215-227`). Nullable
sub-maps are written as Python `None`, which ruamel serialises as
`null`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **One modal dialog with two tabs** | Self-contained, validated on OK, keeps task panel clean, precedent exists | Extra click to discover | **Chosen** |
| Two inline collapsible group boxes | More discoverable by default | Nested nullable sub-map (`stability`) does not fit existing widget conventions; crowds the scroll area; 6+ conditional sections bloat the panel | Rejected |
| Raw YAML via existing `YamlEditDialog` fallback | Zero code | No discoverability, no type validation, users must hand-edit YAML | Rejected (status quo) |

Stability null-state representation: an **"Enable" checkbox** that
writes `stability: null` when unchecked was chosen over a
`method` dropdown with a `none` entry — the checkbox better signals
"this is an on/off scoring step" and keeps the `method: bootstrap`
string encapsulated.

### Architecture Constraints (from knowledge base)

- **Qt signal recursion** — `note-qt-itemchanged-signal-recursion.md`:
  no handlers in this plan connect to `itemChanged` on the same widget
  they mutate (our inputs are discrete `clicked` / `accepted` signals
  on buttons and dialogs), so the recursion trap does not apply
  directly. Still, when `_on_ok` populates widget state from the
  current model during re-open, read-from-model should happen in
  `__init__` (before any signal connections) — already the pattern used
  by `FeatureCombinationDialog`.
- **CuPy import order** — N/A (GUI-only, no numerical packages imported).

### Architecture Changes

New file:

```
code/src/utils/gui/dag_launcher/
├── clustering_config_dialog.py   # NEW — tabbed QDialog
├── feature_combination_dialog.py # unchanged
├── task_detail_panel.py          # one new dispatch branch, one section builder, one handler
└── yaml_edit_dialog.py           # unchanged
```

New public API:

```python
class ClusteringConfigDialog(QDialog):
    def __init__(self, task_name: str, *,
                 reduction_cfg: dict | None,
                 evaluation_cfg: dict | None,
                 parent: QWidget | None = None) -> None: ...
    def get_reduction(self) -> dict: ...
    def get_evaluation(self) -> dict: ...
```

No changes to `DagConfigModel` — `get_task_option` /
`set_task_option` already cover the needed operations.

---

## Implementation Plan

### Phase 1: Dialog scaffold
**Goal:** `ClusteringConfigDialog` renders both tabs with
widgets bound to current YAML values (no write-back yet).

- [x] Task 1.1 — Create `clustering_config_dialog.py` with `QDialog`
  subclass, `QTabWidget`, `QDialogButtonBox`, and shared error label
  (modelled on `feature_combination_dialog.py`).
- [x] Task 1.2 — Reduction tab: `QComboBox` (scaler), `QCheckBox` +
  nested `QLineEdit` (variance_filter.threshold), `QCheckBox` + nested
  `QComboBox` + `QSpinBox` (decomposition.method / n_components).
  Show/hide nested fields via the checkbox `toggled` signal.
- [x] Task 1.3 — Evaluation tab: three `QCheckBox`es
  (internal_metrics), `QCheckBox` "Stability (bootstrap)" with nested
  `QSpinBox` (n_rounds), `QDoubleSpinBox` (subsample_fraction),
  `QComboBox` (score).
- [x] Task 1.4 — `__init__` populates every widget from
  `reduction_cfg` / `evaluation_cfg` defensively (tolerate missing
  keys → widget defaults).

**Files Modified:**
- `code/src/utils/gui/dag_launcher/clustering_config_dialog.py` — NEW.

**Dependencies:** None.

### Phase 2: Validation & getters
**Goal:** `_on_ok` validates numeric input and produces the final
dicts via `get_reduction()` / `get_evaluation()`.

- [x] Task 2.1 — Implement `_on_ok`: validate threshold (float).
  Int/float spin boxes handle their own constraints. On failure call
  `_show_error`, keep dialog open.
- [x] Task 2.2 — Implement `get_reduction()` — build dict with
  `scaler`, `variance_filter` (None or `{threshold: float}`),
  `decomposition` (None or `{method, n_components}`).
- [x] Task 2.3 — Implement `get_evaluation()` — build dict with
  `internal_metrics` as flow-style `CommentedSeq`, `stability` (None or
  `{method, n_rounds, subsample_fraction, score}`).

**Files Modified:**
- `code/src/utils/gui/dag_launcher/clustering_config_dialog.py`.

**Dependencies:** Phase 1.

### Phase 3: Panel integration
**Goal:** `TaskDetailPanel` dispatches `reduction`/`evaluation` to
the combined section and wires the dialog.

- [x] Task 3.1 — In `show_task`, track `insert_idx` separately; skip
  `reduction`/`evaluation` keys (setting `has_clustering_config=True`);
  after the loop append one `_make_clustering_config_section()` widget.
- [x] Task 3.2 — Add `_make_clustering_config_section(self)` — a
  `QGroupBox` titled "Reduction & Evaluation" containing a preview
  `QLabel` (e.g. `scaler=standard · stability=bootstrap(20)`) and a
  "Configure…" `QPushButton` connected to the handler.
- [x] Task 3.3 — Add `_make_clustering_config_handler(self)` — read
  `reduction` / `evaluation` from `self._model.get_task_option`, open
  dialog, on `Accepted` call `set_task_option` twice, emit
  `task_changed`, call `show_task(...)` to refresh the preview.
- [x] Task 3.4 — Module-level helper `_clustering_preview(red, ev) -> str`
  to format the preview label consistently.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add import,
  dispatch branch, section builder, handler factory.

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests
- [ ] `test_clustering_config_dialog_populates_from_cfg` — construct
  dialog with a full config dict, confirm all widgets reflect the
  passed values.
- [ ] `test_clustering_config_dialog_nullables_unchecked_on_null` —
  pass `variance_filter=None`, `decomposition=None`, `stability.method=None`
  or missing `stability`; confirm checkboxes are unchecked and nested
  fields hidden.
- [ ] `test_get_reduction_roundtrip` — set widgets, call
  `get_reduction()`, assert the exact dict shape expected in YAML.
- [ ] `test_get_evaluation_stability_off_returns_none` — uncheck
  stability; `get_evaluation()["stability"]` must be `None`.
- [ ] `test_on_ok_invalid_threshold_rejects` — set threshold to `"abc"`,
  call `_on_ok`, confirm `accept()` was not called and error label is
  visible.

### Integration Tests
- [ ] `test_task_detail_panel_absorbs_reduction_evaluation_into_combined_section`
  — load `configs/analyse_workflow_dag.yaml`, call `show_task(model,
  "touch_clustering")`, assert exactly one "Reduction & Evaluation"
  group box and no standalone `reduction` / `evaluation` group box.
- [ ] `test_dialog_save_writes_expected_yaml` — pump widget values,
  accept dialog, call `model.save()` to a temp path, reload the file
  and compare structure.

### Manual Verification
- [ ] Launch GUI against `configs/analyse_workflow_dag.yaml`,
  confirm the new section and button appear on `touch_clustering`.
- [ ] Toggle scaler `standard` → `robust`, OK, save, inspect YAML.
- [ ] Enable variance_filter (threshold 0.01), save, disable, save;
  confirm YAML transitions between `{threshold: 0.01}` and `null`.
- [ ] Disable stability checkbox, save; confirm `stability: null`.
- [ ] `git diff configs/analyse_workflow_dag.yaml` after a no-op open →
  OK → save shows no changes.
- [ ] Run the clustering Prefect task with stability disabled and
  scaler=robust; confirm `cluster_metadata.json` reflects the changes
  and no error is raised.

### Edge Cases
- [ ] Config file missing the `evaluation` key entirely (older YAML):
  dialog opens with defaults, OK writes the full block without
  disturbing other options.
- [ ] `internal_metrics` list contains an unknown value: dialog
  tolerates it (leaves its checkbox unchecked) and preserves it on
  write-back unchanged — or (accepted behaviour) drops it. Decide
  during Phase 2.
- [ ] User closes dialog via window "X" (equivalent to Cancel): no
  mutation, no `task_changed` emission.

---

## Documentation Plan

- [ ] Update `docs/development/plans/pending/` index entry for this
  plan (N/A — pending plans are discoverable by filename).
- [ ] Docstring on `ClusteringConfigDialog` describing the two YAML
  blocks it edits and the stability-null convention.
- [ ] No changes to `README.md` or `CLAUDE.md` (GUI-only change; no
  new architectural constraint).
- [ ] No changelog entry required (internal GUI ergonomics; no user-
  facing behaviour change beyond easier config editing).

---

## Rollback Plan

1. **Before merge:** revert the two commits (dialog file + panel edit)
   with `git revert`. No data migrations.
2. **After merge but before release:** same — `git revert` on `dev`;
   no YAML schema changes, so existing configs keep working either way.
3. **No data considerations:** the YAML schema is unchanged — this
   plan only adds a typed editor for fields that already exist.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ruamel round-trip drops `null`-style or comments when assigning a freshly built dict to a key | Medium | Medium | Use `CommentedMap` for the `stability` sub-dict when enabled; use `None` (not empty dict) for the disabled state. Verify byte-identity after no-op open/save in tests. |
| Preview label drifts out of sync after partial edits | Low | Low | Always call `show_task(...)` after `Accepted` so the label is rebuilt from model state rather than patched in-place. |
| Qt signal recursion when refreshing widgets during `__init__` | Low | Medium | Populate widgets before connecting `toggled`/`currentIndexChanged` handlers; tested pattern already used by `FeatureCombinationDialog`. |
| Unknown `internal_metrics` value in config is silently dropped | Low | Low | Phase 2 decides: preserve unknown entries unchanged on write-back unless user explicitly toggled them. |
| Widget type mismatch for numeric fields on non-US locales (comma vs. dot decimal) | Low | Low | Use `QDoubleSpinBox` / `QSpinBox` (locale-aware) for numerics; only use `QLineEdit` with an explicit `QDoubleValidator` if a spin box does not fit. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Dialog scaffold | 0.5 day | None |
| Phase 2 — Validation & getters | 0.5 day | Phase 1 |
| Phase 3 — Panel integration | 0.5 day | Phase 2 |

---

## References

- **Alignment plan (parent work):**
  `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md`
- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md`
- **YAML config:** `configs/analyse_workflow_dag.yaml` (lines 135–145)
- **Reused code:**
  - `code/src/utils/pipeline/dag_config_model.py:135,139,215-227`
  - `code/src/utils/gui/dag_launcher/feature_combination_dialog.py`
  - `code/src/utils/gui/dag_launcher/task_detail_panel.py:32,157-170,477-496`
- **Knowledge base:**
  `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
  (reviewed — does not apply directly, but informs the init/connect
  ordering used by this dialog).
