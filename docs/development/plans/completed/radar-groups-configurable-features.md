# Plan: Configurable Radar Groups for Touch Feature Radar

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/touch-feature-radar-plots`
**Branch:** `feature/radar-groups-configurable-features`

---

## Overview

Replace the hardcoded 9-feature list and single `aggregation` option in `render_touch_feature_radar` with configurable named `radar_groups`, each defining which data types and aggregations to plot. This mirrors the `touch_clustering` task's `cluster_groups` pattern and adds a GUI dialog for editing.

## Problem Statement

The radar pipeline currently hardcodes `RADAR_FEATURE_COLUMNS` (9 scalar features) and uses a single `aggregation: mean_during_iff` option. Researchers cannot select which features appear on the radar, cannot mix aggregation methods across features, and cannot define multiple radar profiles (e.g., one for IFF-windowed means, another for peak values). The `touch-feature-radar-plots` plan explicitly deferred this as out of scope.

## Goals

### In Scope
1. Named radar groups in the DAG config with `enabled`, `features` dict (data type -> aggregation list)
2. Multiple aggregations per data type allowed (each becomes a separate radar axis)
3. GUI dialog for creating/editing/deleting radar groups (like `ClusterGroupDialog` without clustering methods)
4. Pipeline rewrite: dynamic column resolution, multi-folder CSV loading/merging, per-group output directories
5. Default `canonical_iff` group that preserves the current 9-feature behavior

### Out of Scope
- Cross-session radar comparison (single figure with multiple sessions)
- Interactive radar plot GUI/viewer
- Per-cluster radar plots (radar of cluster centroids)
- Radar groups as downstream references from other tasks

## Success Criteria

- [ ] DAG config accepts `radar_groups` dict with named groups, each having `enabled` + `features`
- [ ] GUI renders radar_groups section with enable/disable, summary, Edit/Delete/New buttons
- [ ] Radar group dialog shows all 15 data types with 9 aggregation checkboxes each (7 basic + 2 IFF-windowed)
- [ ] Pipeline iterates enabled groups, loads from correct aggregation folders, renders with dynamic labels
- [ ] Default `canonical_iff` group produces identical output to the old hardcoded pipeline
- [ ] Multi-column data types (e.g., `hand_velocity` -> 3 axes) expand correctly on the radar
- [ ] Output directory is `touch_feature_radar/{group_name}/{session_id}/`

---

## Technical Design

### Approach

Reuse the `cluster_groups` structural pattern: named dict of dicts, each with `enabled` and `features`. The GUI detection cascade in `task_detail_panel.py` gets a key-based check for `"radar_groups"` (like the existing `"cluster_groups"` check) to avoid falling through to the wrong handler. A new `RadarGroupDialog` is a simplified `ClusterGroupDialog` minus the clustering methods section, with an expanded aggregation list that includes `mean_during_iff` and `mean_before_iff`.

The pipeline reuses `DATA_TYPE_TO_COLUMNS` from `clustering_pipeline.py` for column expansion and follows the same multi-folder CSV merge pattern.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Named groups (like cluster_groups) | Multiple radar profiles, familiar pattern, GUI reuse | Slightly more complex than a flat dict | **Chosen** |
| Single flat features dict | Simpler config | No way to switch between profiles; inconsistent with cluster_groups pattern | Rejected |
| Reuse FeatureCombinationDialog | No new dialog code | Wrong UI — flat checklist, no per-type aggregation selection | Rejected |
| Structural detector only (no key check) | Generic | Ambiguous — radar_groups would match `_is_feature_combinations_dict` | Rejected — key-based check is more robust |

### Architecture Changes

```
code/src/utils/gui/dag_launcher/
  radar_group_dialog.py                           # NEW — edit dialog
  task_detail_panel.py                            # MODIFIED — detection + section + handlers

code/src/utils/pipeline/
  dag_config_model.py                             # MODIFIED — get/set_radar_group_spec

code/src/analysis/receptive_field_mapping/
  pipelines/rf_touch_feature_radar_pipeline.py    # MODIFIED — major rewrite

code/scripts/
  analysis_workflow_processing.py                 # MODIFIED — flow signature + stage params

configs/
  analyse_workflow_processing_dag.yaml            # MODIFIED — radar_groups replaces aggregation
```

---

## Implementation Plan

### Phase 1: Config + Model
**Goal:** New YAML structure and DagConfigModel support
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 1.1 — Replace `aggregation: mean_during_iff` with `radar_groups` dict in DAG config
- [x] Task 1.2 — Add `get_radar_group_spec` and `set_radar_group_spec` to `DagConfigModel`

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — replace options block
- `code/src/utils/pipeline/dag_config_model.py` — add two methods following `get_grid_group_spec` pattern

**Dependencies:** None

### Phase 2: Pipeline Rewrite
**Goal:** Dynamic feature resolution, multi-folder loading, group iteration
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 2.1 — Remove `RADAR_FEATURE_COLUMNS`, `RADAR_DISPLAY_LABELS`, `_SUFFIX_CANDIDATES`, `_resolve_feature_columns()`
- [x] Task 2.2 — Add `_DISPLAY_NAMES` map for human-readable axis labels
- [x] Task 2.3 — Add `_resolve_required_aggregation_folders(features_spec)` (same logic as `clustering_pipeline._resolve_required_feature_folders`)
- [x] Task 2.4 — Add `_load_and_merge_feature_csvs(db_path, agg_folders, session_id)` using existing `_find_feature_csv()` + merge on `TOUCH_ID_COLS_WITH_SESSION`
- [x] Task 2.5 — Add `_resolve_radar_columns(df, features_spec)` returning `(column_names, display_labels)` via `DATA_TYPE_TO_COLUMNS` expansion
- [x] Task 2.6 — Rewrite `run_touch_feature_radar(session_configs, radar_groups, force_processing)` with group iteration and per-group output dirs

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py` — major rewrite

**Dependencies:** Phase 1

### Phase 3: Workflow Script
**Goal:** Wire new config structure through the Prefect flow
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 3.1 — Change `render_touch_feature_radar_flow` param from `aggregation: str` to `radar_groups: dict`
- [x] Task 3.2 — Update `_build_pipeline_stages` lambda to pass `radar_groups` from DAG config

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — update flow function + stage registration

**Dependencies:** Phase 2

### Phase 4: GUI Dialog
**Goal:** Radar group editing dialog and task panel integration
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 4.1 — Create `RadarGroupDialog` with name section + features section (15 data types x 9 aggregations), no clustering methods
- [x] Task 4.2 — Add `_is_radar_groups_dict` detector function
- [x] Task 4.3 — Add `_radar_group_summary` helper
- [x] Task 4.4 — Add key-based detection case in `show_task()` after `_is_cluster_groups_dict` check
- [x] Task 4.5 — Add `_make_radar_groups_section` section builder (same pattern as `_make_cluster_groups_section`)
- [x] Task 4.6 — Add handler factories: enabled, edit, delete, new (using `RadarGroupDialog` + `DagConfigModel` methods)
- [x] Task 4.7 — Import `RadarGroupDialog` in `task_detail_panel.py`

**Files Created:**
- `code/src/utils/gui/dag_launcher/radar_group_dialog.py`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — import, detector, summary, detection case, section builder, handlers

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `_resolve_required_aggregation_folders` returns correct deduplicated folder list for mixed aggregations
- [ ] `_resolve_radar_columns` expands multi-column data types (e.g., `hand_velocity: [mean]` -> 3 columns) and raises `ValueError` for missing columns
- [ ] Display labels disambiguate when a data type has >1 aggregation (e.g., `"Area (mean)"`, `"Area (max)"`)
- [ ] `_is_radar_groups_dict` returns True for valid radar groups and False for cluster_groups, grid_groups, and feature_combinations

### Integration Tests
- [ ] Pipeline processes a single-aggregation radar group end-to-end with real session data
- [ ] Pipeline processes a mixed-aggregation radar group (features from 2+ folders) by loading and merging CSVs
- [ ] Idempotency: second run with unchanged inputs skips processing per group

### Manual Verification
- [ ] Launch GUI -> select `render_touch_feature_radar` -> verify radar_groups section shows `canonical_iff` with correct summary
- [ ] Click Edit -> verify dialog shows 15 data types with 9 aggregation checkboxes each
- [ ] Create new group, save, verify it appears in the task panel and persists in YAML
- [ ] Delete a group, verify removal from YAML
- [ ] Run pipeline -> verify output in `touch_feature_radar/{group_name}/{session_id}/`
- [ ] Verify radar axes match configured features with correct display labels

### Edge Cases
- [ ] Data type with all aggregations disabled -> validation error in dialog
- [ ] `mechanics_of_solids` bundle type (5 columns) + individual `mos_strain` -> deduplicate or allow both
- [ ] Single-folder group (all features use same aggregation) -> skip merge, load single CSV
- [ ] Group with only 2 features -> radar renders with 2 axes (degenerate but valid)

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/touch-feature-radar-plots.md` — mark "custom feature selection" as now in scope, reference this plan
- [ ] Update `code/src/analysis/CLAUDE.md` — mention radar_groups config pattern in orchestration section

---

## Rollback Plan

1. Revert `radar_group_dialog.py` (new file — just delete)
2. Revert changes to `task_detail_panel.py`, `dag_config_model.py`, `rf_touch_feature_radar_pipeline.py`, `analysis_workflow_processing.py`
3. Restore `aggregation: mean_during_iff` in DAG config
4. No data migration — output PNGs in new directory structure can be deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Multi-column data types produce cluttered radars (e.g., hand_velocity -> 3 axes per aggregation) | Medium | Low | User responsibility — config allows it, radar handles any axis count >= 3 |
| Column name collision when merging CSVs from multiple aggregation folders | Low | High | Column suffixes are unique per extractor (`_mean` vs `_mean_during_iff`); warn on duplicates during merge |
| Old YAML configs with `aggregation:` break | High | Low | Raise `ValueError` with clear migration instructions pointing to new format |
| Qt signal recursion in radar group dialog checkboxes | Low | Medium | Guard mutations with `blockSignals(True/False)` per knowledge base note |

---

## References

- Parent plan: `docs/development/plans/active/touch-feature-radar-plots.md`
- Cluster groups pattern: `configs/analyse_workflow_processing_dag.yaml` lines 286–463
- `ClusterGroupDialog`: `code/src/utils/gui/dag_launcher/cluster_group_dialog.py`
- `DATA_TYPE_TO_COLUMNS`: `code/src/analysis/touch_analytics/clustering_pipeline.py` lines 58–74
- Feature catalog: `code/src/utils/gui/dag_launcher/_feature_catalog.py`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
