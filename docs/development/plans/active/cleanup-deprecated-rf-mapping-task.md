# Plan: Remove deprecated `map_receptive_fields` DAG task

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/cluster-based-rf-mapping` (existing)

---

## Overview

**What:** Remove the old selectivity+DBSCAN `map_receptive_fields` flow from the analysis workflow and DAG config.
**Why:** `map_receptive_fields_clustered` has replaced it; the old task is already `enabled: false`. Keeping dead flow code, unused imports, and helper functions adds confusion and maintenance burden.
**How:** Delete the flow function, its private helpers, unused imports, and the DAG entry. Keep underlying library modules intact because `center_on_receptive_field.py` still depends on them.

## Problem Statement

The old `map_receptive_fields` flow groups touches by experimenter-defined metadata (`type_metadata`, `direction`) and uses selectivity+DBSCAN. The new `map_receptive_fields_clustered` flow uses data-driven cluster labels from `touch_clustering` and produces spike-count heatmaps. The old flow is disabled in the DAG but its code remains in `analysis_workflow.py` (~140 lines), along with two private helper functions and four unused imports. This dead code clutters the workflow script and creates confusion about which RF mapping path is active.

## Goals

### In Scope
1. Remove `map_receptive_fields_flow` and its registration from the analysis workflow
2. Remove private helpers used only by the old flow (`_save_group_csv`, `_load_forearm_pcd`)
3. Remove unused imports from `analysis_workflow.py`
4. Remove the `map_receptive_fields` task entry from `analyse_workflow_dag.yaml`
5. Clean up `receptive_field_mapping/__init__.py` re-exports that are no longer used outside the package

### Out of Scope
- Deleting library modules (`rf_mapping_engine.py`, `rf_mapping_config.py`, `rf_data_loader.py`, `rf_visualizer.py`) — still used by `center_on_receptive_field.py`
- Modifying `center_on_receptive_field.py`
- Changes to the `map_receptive_fields_clustered` flow
- Removing or archiving completed plan docs that reference the old flow

## Success Criteria

- [ ] `map_receptive_fields_flow` no longer exists in `analysis_workflow.py`
- [ ] `map_receptive_fields` task entry no longer exists in `analyse_workflow_dag.yaml`
- [ ] No import in `analysis_workflow.py` references `RFMappingConfig`, `RFMappingEngine`, `load_grouped_spatial_data`, or `RFVisualizer`
- [ ] `center_on_receptive_field.py` still imports and uses `GroupedSpatialData`, `SelectivityDBSCANConfig`, and `RFMappingEngine` without error
- [ ] `map_receptive_fields_clustered` flow is unaffected and runs as before

---

## Technical Design

### Approach

Straight deletion of dead code. No refactoring, no moving code between files.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Delete flow + helpers only (keep library modules) | Minimal risk, `center_on_receptive_field.py` unaffected | Old library modules remain | **Chosen** |
| Delete flow + all old library modules | Cleaner package | Breaks `center_on_receptive_field.py`, requires rewrite of that script | Rejected |
| Disable and leave code in place | Zero risk | Dead code stays, ongoing confusion | Rejected |

### Architecture Changes

No new modules. No interface changes. Pure removal.

---

## Implementation Plan

### Phase 1: Remove flow and DAG entry
**Goal:** Eliminate dead code from the workflow script and DAG config.

**Tasks:**
- [x] Task 1.1 — Remove `map_receptive_fields_flow` function (lines 211-347) from `analysis_workflow.py`
- [x] Task 1.2 — Remove `_save_group_csv` helper (lines 384-406) — only called by old flow
- [x] Task 1.3 — Remove `_load_forearm_pcd` helper (lines 409-417) — only called by old flow
- [x] Task 1.4 — Remove `("map_receptive_fields", map_receptive_fields_flow)` from `available_tasks` (line 489)
- [x] Task 1.5 — Remove unused imports: `RFMappingConfig` (line 37), `RFMappingEngine` (line 38), `load_grouped_spatial_data` (line 39), `RFVisualizer` (line 40)
- [x] Task 1.6 — Remove `map_receptive_fields` task block from `configs/analyse_workflow_dag.yaml` (lines 158-165)

**Files Modified:**
- `code/scripts/analysis_workflow.py` — remove flow, helpers, imports, registration
- `configs/analyse_workflow_dag.yaml` — remove task entry

**Dependencies:** None

### Phase 2: Clean up package exports
**Goal:** Remove re-exports from `__init__.py` that are no longer used outside the package.

**Tasks:**
- [x] Task 2.1 — Audit which `__init__.py` exports are still imported externally (by `center_on_receptive_field.py` or `rf_cluster_pipeline.py`)
- [x] Task 2.2 — Remove re-exports that have no external consumers; keep `run_cluster_rf_mapping` and anything `center_on_receptive_field.py` uses via the package

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — trim exports

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] `python -c "from analysis.receptive_field_mapping import run_cluster_rf_mapping"` succeeds
- [ ] `python -c "from analysis.receptive_field_mapping.rf_mapping_engine import RFMappingEngine"` succeeds (used by `center_on_receptive_field.py`)
- [ ] `python -c "from analysis.receptive_field_mapping.rf_mapping_config import GroupedSpatialData, SelectivityDBSCANConfig"` succeeds
- [ ] Run `map_receptive_fields_clustered` task via DAG — produces same output as before
- [ ] Grep for `map_receptive_fields_flow` in the codebase — zero hits outside `docs/`

---

## Rollback Plan

All changes are deletions in tracked files on an existing feature branch. Rollback:
1. `git checkout HEAD -- code/scripts/analysis_workflow.py configs/analyse_workflow_dag.yaml code/src/analysis/receptive_field_mapping/__init__.py`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Break `center_on_receptive_field.py` by over-deleting | Low | High | Explicit out-of-scope rule: do not delete library modules |
| Break `map_receptive_fields_clustered` flow | Low | High | Only removing the *old* flow registration; clustered flow untouched |
| Miss an import that references the old flow | Low | Low | Grep verification in testing plan |
