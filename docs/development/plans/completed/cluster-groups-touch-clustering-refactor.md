# Plan: Cluster Groups — Touch Clustering Refactor

**Date:** 2026-04-24
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-24 17:07
**Branch:** `feature/align-clustering-pipeline-with-guidelines`

---

## Overview

Replace the coarse `feature_combinations` concept in the touch clustering pipeline with fine-grained **cluster groups**: named configurations where the user selects specific data types (columns) and chooses which aggregations to apply per type. Each cluster group carries its own clustering algorithm profiles. A popup GUI dialog enables creating and editing groups. Downstream tasks (touch comparing, RF mapping) reference groups by name and show a read-only preview of what each group contains.

## Problem Statement

The current system uses `feature_combinations` — named lists of feature extractor types (e.g. `[mean, touch_category]`) applied uniformly across all kinematic columns. This prevents per-column control: a researcher cannot say "use velocity min+max but contact_area mean only." All algorithmic choices (clusterers) are also global, forcing every combination to run against every clusterer. The result is an inflexible config that cannot express real experimental intentions, and downstream tasks duplicate the combination/clusterer config instead of referencing the authority source.

## Goals

### In Scope
1. Replace `feature_combinations` with `cluster_groups` in the clustering pipeline, config schema, and GUI
2. Each cluster group defines: a name, a dict of data types → aggregation list, and its own `clustering_methods`
3. Add `contact_location_x/y/z` to the Stage 2b extraction scope so non-mean location aggregations are available
4. Downstream tasks (`touch_comparing`, `map_receptive_fields_clustered`) reference cluster groups by name; `run_batch_analysis` forwards full group specs to them
5. New `ClusterGroupDialog` (two-tab: Features + Algorithms) for creating/editing groups in the GUI
6. Downstream task panels show a cluster group import UI with read-only detail view

### Out of Scope
- Per-group `reduction` settings (global reduction remains at task level)
- Per-group `evaluation` settings (global evaluation remains at task level)
- Migrating existing on-disk cluster output directories (old runs stay; new runs write to `<group_name>/<clusterer>/`)
- Real-time validation of Stage 2b / Stage 3–5 compatibility at GUI time
- Any changes to the clustering algorithms themselves

## Success Criteria

- [ ] A cluster group with `velocity: [min, max], contact_area: [mean]` clusters correctly and produces output at `touch_clusters/<group_name>/binning/pooled_touch_summary_clustered.csv` containing only `velocity_magnitude_min`, `velocity_magnitude_max`, `contact_area_mean` as feature columns
- [ ] A cluster group with `location: [std]` produces output including `contact_location_x_std/y_std/z_std`
- [ ] Creating/editing a group in the GUI and saving produces correct round-trip YAML
- [ ] Downstream task panels show all groups from `touch_clustering` with correct feature summaries and clickable Details
- [ ] `touch_comparing` and `map_receptive_fields_clustered` navigate to the correct on-disk path without their own `clustering_methods` config
- [ ] Old YAML using `feature_combinations` triggers a deprecation warning and still runs

---

## Technical Design

### Approach

**Column selection at clustering time from existing per-aggregation CSVs.** Stage 2b continues to produce per-aggregation CSVs (`touch_features/<agg>/<session>.csv`). The clustering pipeline loads the relevant CSVs per cluster group spec and selects only the requested `{column}_{agg}` columns before passing to reduction/clustering. This avoids re-architecting the extraction layer and keeps Stage 2b results independently cacheable.

`clustering_methods` move inside each cluster group so different groups can use different algorithms. `reduction` and `evaluation` remain at the task level (global).

Downstream tasks receive full group specs (`cluster_group_defs: dict`) from `run_batch_analysis` so they can navigate to the correct `<group_name>/<clusterer>/` subdirectory without duplicating config.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Unified all-features CSV (Stage 2b produces one combined CSV per session) | Single file, no merge at clustering time | Must re-run extraction on config change; larger I/O; breaks current caching by aggregation type | Rejected |
| Column selection at cluster time from per-aggregation CSVs | Reuses existing cache; decouples extraction from clustering config | Small per-group merge overhead (already done today) | **Chosen** |
| Global clustering_methods (shared across all groups) | Simpler config and GUI | Cannot use different algorithms per group | Rejected (user preference) |
| Per-group clustering_methods | Different algorithms per group | More complex config and GUI dialog | **Chosen** |

### Architecture Changes

**New constant module** (or additions to `clustering_pipeline.py`):
```python
DATA_TYPE_TO_COLUMNS = {
    "contact_area":   ["contact_area"],
    "contact_depth":  ["contact_depth"],
    "velocity":       ["velocity_magnitude"],
    "acceleration":   ["acceleration_magnitude"],
    "pressure":       ["geo_pressure"],
    # location: dual-mapped (see Location column handling below)
    "touch_category": [],  # binary one-hot, no aggregation
}
_TOUCH_CATEGORY_COLUMNS = ["is_tap", "is_stroke", "dir_proximal", "dir_distal"]
_LOCATION_SHARED_COLS   = ["mean_contact_x", "mean_contact_y", "mean_contact_z"]
_LOCATION_BASE_COLS     = ["contact_location_x", "contact_location_y", "contact_location_z"]
```

**Location column dual-mapping:**
- `location: [mean]` → uses `mean_contact_x/y/z` from shared columns (always present)
- `location: [std/min/max/...]` → uses `contact_location_x_std`, etc. from per-aggregation CSVs (requires Stage 2b to have extracted those aggregations)

**New config schema (authority: `touch_clustering`):**
```yaml
touch_clustering:
  options:
    reduction:                          # global
      scaler: standard
      variance_filter: null
      decomposition: null
    evaluation:                         # global
      internal_metrics: [silhouette, davies_bouldin, calinski_harabasz]
      stability: null
    cluster_groups:
      velocity_area:
        enabled: true
        features:
          velocity: [min, max]
          contact_area: [mean]
          location: [mean]
        clustering_methods:            # per-group
          binning:
            method: binning
            n_bins: 20
            bin_method: equal_width
            enabled: true
```

**Downstream config (reference only, no own clustering_methods):**
```yaml
touch_comparing:
  options:
    cluster_groups: [velocity_area, pressure_depth]   # flow-style list
    comparing_profiles:
      bias: {method: bias, measurement_col: spike_elicited, sensor_col: session_id}
```

---

## Implementation Plan

### Phase 0: Prerequisite — Fix config forwarding
**Goal:** Land the pending `fix-analysis-workflow-stage-config-forwarding` plan so `reduction` and `evaluation` blocks are forwarded through `run_clustering`. This plan touches the same function signatures.

- [x] Implement `docs/development/plans/pending/fix-analysis-workflow-stage-config-forwarding.md`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — add `reduction`/`evaluation` forwarding in `run_batch_analysis`
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — add `reduction`/`evaluation` params to `run_clustering`

**Dependencies:** None

---

### Phase 1: Stage 2b — location column extraction
**Goal:** Enable non-mean aggregations for location by adding position columns to the extraction scope.

- [x] Add `contact_location_x`, `contact_location_y`, `contact_location_z` to `KINEMATIC_SIGNALS` in `code/src/analysis/touch_analytics/touch_config.py`
- [x] Expand `touch_feature_extraction.options.features` in `configs/analyse_workflow_dag.yaml` to include all standard aggregations: `[mean, min, max, median, std, range, skewness]`

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_config.py` — add 3 location columns to `KINEMATIC_SIGNALS`
- `configs/analyse_workflow_dag.yaml` — expand `touch_feature_extraction.options.features`

**Dependencies:** None (independent)

---

### Phase 2: Backend — column selection helpers
**Goal:** Add the data-type-to-column mapping and column selection logic that cluster groups will use.

- [x] Add `DATA_TYPE_TO_COLUMNS`, `_TOUCH_CATEGORY_COLUMNS`, `_LOCATION_SHARED_COLS`, `_LOCATION_BASE_COLS` constants to `clustering_pipeline.py`
- [x] Implement `_resolve_required_feature_folders(group_spec) -> tuple[list[str], bool]` — derives which aggregation folder names to load and whether touch_category columns are needed; `location[mean]` does NOT add `mean` to folder list (uses shared cols)
- [x] Implement `_select_feature_columns(merged_df, group_spec, needs_touch_category) -> list[str]` — selects column names from the merged DataFrame per group spec; emits warning for any missing `{col}_{agg}` column

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — new module-level constants and two helper functions

**Dependencies:** Phase 1 (location columns must be in KINEMATIC_SIGNALS before CSVs are generated)

---

### Phase 3: Clustering pipeline refactor
**Goal:** Replace the `feature_combinations × clustering_methods` cross-product with per-group clustering profiles.

- [x] Rename `feature_combinations` parameter to `cluster_groups` in `run_clustering`; add backward compat shim (translate old dict shape, emit deprecation warning)
- [x] Replace global clustering_methods iteration with per-group `group_spec["clustering_methods"]` lookup inside the combination loop
- [x] Add `feature_cols_override: list[str] | None` parameter to `_cluster_combination`; when provided, use it instead of auto-discovery of numeric columns
- [x] Wire `_resolve_required_feature_folders` + `_select_feature_columns` into the group loop to compute `feature_cols_override` per group
- [x] Update `touch_clustering_flow` signature in `analysis_workflow.py`: `feature_combinations` → `cluster_groups`
- [x] Update `run_batch_analysis` dispatch: add `"cluster_groups"` branch; keep deprecated `"feature_combinations"` branch with warning

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — refactor `run_clustering`, `_cluster_combination`
- `code/scripts/analysis_workflow.py` — update `touch_clustering_flow` signature and `run_batch_analysis`

**Dependencies:** Phase 0, Phase 2

---

### Phase 4: Downstream pipeline refactor
**Goal:** Downstream tasks navigate to cluster output directories using group names + per-group clusterer list, without own `clustering_methods` config.

- [x] Refactor `run_comparing` signature: `feature_combinations: dict` → `cluster_groups: list[str]`, add `cluster_group_defs: dict`; iterate `cluster_groups` × each group's `clustering_methods`; remove own `clustering_methods` parameter
- [x] Refactor `run_cluster_rf_mapping` signature: same change
- [x] Update `touch_comparing_flow` and `map_receptive_fields_clustered_flow` signatures accordingly
- [x] Update `run_batch_analysis`: when dispatching downstream tasks, extract full group specs from `touch_clustering.options.cluster_groups` and pass as `cluster_group_defs`; remove `clustering_methods` dispatch for downstream tasks

**Files Modified:**
- `code/src/analysis/touch_analytics/comparing_pipeline.py` — refactor `run_comparing`
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — refactor `run_cluster_rf_mapping`
- `code/scripts/analysis_workflow.py` — update downstream flow signatures and `run_batch_analysis`

**Dependencies:** Phase 3

---

### Phase 5: YAML schema migration
**Goal:** Migrate `configs/analyse_workflow_dag.yaml` to the new cluster_groups schema.

- [x] Replace `touch_clustering.options.feature_combinations` with `cluster_groups` (with per-group `clustering_methods`)
- [x] Remove `touch_clustering.options.clustering_methods` (global, now per-group)
- [x] Replace `touch_comparing.options.feature_combinations` and `touch_comparing.options.clustering_methods` with `cluster_groups: [...]` flow-style list
- [x] Replace `map_receptive_fields_clustered.options.feature_combinations` and `.clustering_methods` with `cluster_groups: [...]`
- [x] Verify round-trip YAML fidelity (load → no-op save → diff = empty)

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — full schema migration for 3 tasks

**Dependencies:** Phase 3, Phase 4 (code must consume new schema before YAML is changed)

---

### Phase 6: DagConfigModel new methods
**Goal:** Expose cluster group CRUD and per-group clustering_methods CRUD for the GUI layer.

- [x] `get_cluster_group_spec(task_name, group_name) -> dict`
- [x] `set_cluster_group_spec(task_name, group_name, spec) -> None`
- [x] `get_downstream_cluster_group_names(task_name) -> list[str]` — reads the flow-style list value
- [x] `set_downstream_cluster_group_names(task_name, names: list[str]) -> None` — writes as flow-style CommentedSeq
- [x] `get_group_clustering_methods(task_name, group_name) -> dict`
- [x] `set_group_clustering_profile_enabled(task_name, group_name, profile_name, enabled) -> None`
- [x] `set_group_clustering_profile_spec(task_name, group_name, profile_name, spec) -> None`

Note: `get_cluster_group_names` and top-level group `enabled` toggle reuse existing `get_profile_names` / `set_profile_enabled` with `option_key="cluster_groups"`.

**Files Modified:**
- `code/src/utils/pipeline/dag_config_model.py` — 7 new methods

**Dependencies:** None (independent of pipeline code)

---

### Phase 7: GUI — ClusterGroupDialog + touch_clustering panel
**Goal:** Allow users to create and edit cluster groups from the GUI.

- [x] Create `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` with `ClusterGroupDialog` (two-tab QDialog)
  - **Tab "Features":** scrollable list of data types (contact_area, contact_depth, velocity, acceleration, pressure, location, touch_category); checking an aggregatable type reveals inline aggregation toggle row (mean, min, max, median, std, range, skewness); touch_category has no aggregation row
  - **Tab "Algorithms":** list of known clusterer methods with enable checkbox + "Configure" button (opens method-specific params dialog, reusing existing `ReductionConfigDialog` pattern) + preview label
  - Validation: name `[a-z0-9_]+`, unique, ≥1 data type selected, ≥1 aggregation per aggregatable type, ≥1 clusterer enabled
  - `get_group_name() -> str`, `get_group_spec() -> dict`
- [x] Update `task_detail_panel.py` for `touch_clustering`:
  - Replace `"feature_combinations"` with `"cluster_groups"` in `_TOUCH_CLUSTERING_STAGE_ORDER`
  - Add `_make_cluster_groups_section(val: dict) -> QWidget`: `QGroupBox("Cluster Groups")` with per-group rows (QCheckBox | summary QLabel | Edit button | Delete button) and "New Group…" button
  - Add `_cluster_group_summary(spec) -> str`: produces e.g. `"velocity[min,max] · area[mean]"` — shows features only (not algorithms) for compactness
  - Wire Edit button → `ClusterGroupDialog` pre-populated with existing spec; OK → `set_cluster_group_spec`
  - Wire Delete button → confirm dialog → remove group from YAML
  - Wire "New Group…" → empty `ClusterGroupDialog`; OK → `set_cluster_group_spec`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` — NEW
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — cluster groups section + `_cluster_group_summary`

**Dependencies:** Phase 5 (YAML shape must be stable), Phase 6 (DagConfigModel methods)

---

### Phase 8: GUI — downstream task panels
**Goal:** Downstream task panels show which cluster groups exist in `touch_clustering` and let users enable/disable them.

- [x] Add `ClusterGroupReadOnlyDialog` to `cluster_group_dialog.py` — shows name, enabled state, and full features breakdown (read-only); opened via "Details…" button
- [x] Update `task_detail_panel.py` else-branch dispatch: detect `key == "cluster_groups"` with a list value (as opposed to a dict); call `_make_downstream_cluster_groups_section(val)`
- [x] Implement `_make_downstream_cluster_groups_section(val: list) -> QWidget`:
  - `QGroupBox("Cluster Groups")`
  - Italic sub-label: "From touch_clustering:"
  - For each group defined in `touch_clustering.options.cluster_groups`: row with QCheckBox (checked if name in val) + summary label + "Details…" button
  - Checkbox state change → `set_downstream_cluster_group_names`
  - "Details…" → `ClusterGroupReadOnlyDialog`
  - If `touch_clustering` has no groups: show informational label

**Files Modified:**
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` — add `ClusterGroupReadOnlyDialog`
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — downstream cluster groups section

**Dependencies:** Phase 7

---

### Phase 9: Cleanup and integration test
**Goal:** Remove deprecated shims, verify full pipeline end-to-end.

- [x] Remove `feature_combinations` backward compat shim from `run_clustering` (or keep warning-only shim if old YAMLs may still exist) — keeping warning-only shim; old YAMLs may still exist
- [x] Search codebase for any remaining `feature_combinations` references outside shims; update or remove — all references are intentional shim code or orphaned unused defaults in `pipeline_shared.py`
- [ ] End-to-end test: define two cluster groups, run full pipeline (extraction → clustering → comparing → RF mapping), verify outputs at correct paths — manual verification required

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — remove/trim shim
- `code/scripts/analysis_workflow.py` — trim shim

**Dependencies:** All prior phases

---

## Testing Plan

### Unit Tests
- [ ] `_resolve_required_feature_folders` — verify location[mean] does not add 'mean' to folder list; verify touch_category sets `needs_touch_category=True`
- [ ] `_select_feature_columns` — given a merged DataFrame, verify correct column names selected per group spec; verify warning emitted for missing columns
- [ ] `get_cluster_group_spec` / `set_cluster_group_spec` — verify round-trip YAML identity
- [ ] `get_downstream_cluster_group_names` / `set_downstream_cluster_group_names` — verify flow-style list serialization

### Integration Tests
- [ ] Run clustering with a group spec `velocity: [min, max], contact_area: [mean]` → verify feature matrix fed to clusterer has exactly those 3 columns
- [ ] Run clustering with `location: [std]` → verify `contact_location_x_std` etc. appear in cluster output CSV
- [ ] Run `touch_comparing` with a group name list → verify it reads from `touch_clusters/<group_name>/` not a wrong path
- [ ] Run `map_receptive_fields_clustered` with group name list → same path verification

### Manual Verification
- [ ] Create a cluster group in the GUI → save → reopen config → group appears correctly with all features and algorithms pre-populated
- [ ] Edit a cluster group → change aggregations → save → verify YAML updated
- [ ] Delete a cluster group → YAML no longer contains that group key
- [ ] Open `touch_comparing` task panel → all groups from `touch_clustering` appear in the list with correct summaries; "Details…" shows full breakdown
- [ ] Enable/disable a group in downstream panel → YAML `cluster_groups` list updated correctly

### Edge Cases
- [ ] Cluster group with `touch_category: []` only (no kinematic features) — should cluster on binary columns only
- [ ] Cluster group requesting `velocity: [std]` when Stage 2b has not produced `std` CSVs — warning emitted, column skipped, run completes on remaining features
- [ ] Old YAML with `feature_combinations` loaded in GUI → deprecation warning logged, pipeline runs successfully
- [ ] Downstream task panel when `touch_clustering` has zero cluster groups — informational label shown, no crash
- [ ] Two groups with identical feature specs but different names — both produce output in separate subdirectories

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/` with a note on the cluster_groups schema if any non-obvious constraint is discovered during implementation
- [ ] No CLAUDE.md update needed (no new global constraint)

---

## Rollback Plan

1. The `feature_combinations` backward compat shim in `run_clustering` allows old-format YAMLs to run even after the code is updated — no hard cutover.
2. To revert the YAML: restore `configs/analyse_workflow_dag.yaml` from git (`git checkout main -- configs/analyse_workflow_dag.yaml`).
3. To revert code changes: `git revert` or `git checkout main` on the affected files. The pipeline is stateless (no DB migrations); on-disk cluster outputs are unaffected.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Stage 2b `contact_location_x/y/z` not present in series-augmented CSV (column name mismatch) | Medium | Medium | Read series_pipeline.py to confirm exact column names before adding to KINEMATIC_SIGNALS; add an explicit column-existence check in the extractor |
| Existing on-disk cluster results become unreferenced (path `<combination>/<clusterer>/` vs `<group_name>/<clusterer>/`) | Low | Low | Old results are not deleted; new results go to new paths. Document the path change. |
| `ClusterGroupDialog` Algorithms tab replicating complex per-method param UIs | Medium | Medium | Reuse existing `ReductionConfigDialog`/`EvaluationConfigDialog` pattern; for method params, open a simple dict-edit dialog as a first version |
| Downstream tasks silently skip groups if `cluster_group_defs` is not forwarded correctly | Medium | High | Add explicit assertion in `run_comparing` / `run_cluster_rf_mapping` that all referenced group names exist in `cluster_group_defs` |
| `fix-analysis-workflow-stage-config-forwarding` (Phase 0) is not landed before Phase 3 | Medium | Medium | Phase 3 modifies the same `run_clustering` signature — coordinate or co-land both changes |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| 0 — Config forwarding fix | 0.5 day | None |
| 1 — Stage 2b location columns | 0.5 day | None |
| 2 — Column selection helpers | 0.5 day | Phase 1 |
| 3 — Clustering pipeline refactor | 1 day | Phase 0, 2 |
| 4 — Downstream pipeline refactor | 0.5 day | Phase 3 |
| 5 — YAML migration | 0.5 day | Phase 3, 4 |
| 6 — DagConfigModel methods | 0.5 day | None |
| 7 — ClusterGroupDialog + panel | 2 days | Phase 5, 6 |
| 8 — Downstream panel UI | 1 day | Phase 7 |
| 9 — Cleanup + integration test | 0.5 day | All |
| **Total** | **~7.5 days** | |

---

## References

- Related active plans:
  - `docs/development/plans/active/dag-launcher-clustering-config-widgets.md`
  - `docs/development/plans/active/fix-clustering-gui-pipeline-stage-order.md`
  - `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md`
- Blocking prerequisite: `docs/development/plans/pending/fix-analysis-workflow-stage-config-forwarding.md`
- Guidelines: `docs/design/timeseries_clustering_pipeline_guidelines.md`
