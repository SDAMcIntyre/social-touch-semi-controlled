# Plan: Postprocessing Pipeline Refactor

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/postprocessing-pipeline-refactor`

---

## Overview

Refactor the postprocessing pipeline so the forearm of reference is an explicit, tracked artifact flowing through every stage. Currently the forearm is resolved ad-hoc by helper functions and the PCA forearm export is a separate step from the PCA calibration itself. This refactor makes the forearm a first-class pipeline artifact from stage 0 onward, and merges the PCA calibration + forearm export into a single coherent task.

## Problem Statement

The current pipeline has two design issues:
1. **Forearm resolution is ad-hoc** — helper functions (`_resolve_session_forearm`, `_find_forearm_ply`) resolve the forearm PLY at multiple points in the code with fallback chains, making data flow implicit and error-prone.
2. **PCA calibration and forearm export are separate tasks** — `set_xyz_reference_from_gestures` computes the PCA transform and applies it to CSVs, while `export_forearm_pca_calibrated` separately applies it to the forearm PLY. These are logically one operation.

## Goals

### In Scope
1. Add a new initial task (`fetch_forearm_of_reference`) that resolves and copies the forearm PLY into a dedicated `forearm_source/` folder
2. Merge `set_xyz_reference_from_gestures` + `export_forearm_pca_calibrated` into a single `calibrate_pca_xyz` task
3. Thread the forearm PLY explicitly through the pipeline context at every stage
4. Simplify output directory naming (`blocks_deduped`, `blocks_projected` instead of `blocks_registered_deduped`, `blocks_registered_projected`)
5. Update the orchestrator, DAG config, and visualization pipeline to match

### Out of Scope
- Changing the mathematical operations of any task (transforms, DBSCAN, PCA, etc.)
- Modifying the analysis pipeline's forearm resolution (`resolve_forearm_ply`)
- Adding new visualization capabilities
- Changing the `apply_icp_registration` or `center_on_receptive_field` task logic

## Success Criteria

- [ ] Pipeline runs end-to-end on a session with the new task structure
- [ ] `forearm_source/{session_id}_forearm.ply` is produced at step 0
- [ ] `calibrate_pca_xyz` produces both CSV outputs and forearm PLY in one task
- [ ] `{session_id}_forearm.ply` appears at session root after aggregate (analysis-compatible)
- [ ] Visualization stage viewer works with the new directory structure
- [ ] No forearm resolution fallback logic remains in the orchestrator

---

## Technical Design

### Approach

Make the forearm PLY an explicit pipeline artifact by:
1. Fetching it once at the start into a canonical location
2. Passing it through the context dict at each stage
3. Each task that transforms the forearm produces a new copy in its output directory

This eliminates the need for ad-hoc resolution helpers and makes the data flow visible in the orchestrator's `pipeline_stages` list.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Explicit forearm in context (chosen) | Clear data flow, fail-fast if missing, no fallback chains | Requires rewriting orchestrator stage definitions | Chosen |
| Keep implicit resolution, just merge PCA tasks | Minimal code change | Doesn't fix the root issue of ad-hoc forearm discovery | Rejected |
| Pass forearm as a separate parallel "track" alongside CSVs | Conceptually clean separation | Over-engineering for a linear pipeline | Rejected |

### Architecture Changes

**New pipeline stage sequence:**
```
fetch_forearm_of_reference  →  forearm_source/
        ↓
apply_icp_registration      →  blocks_registered/  (CSVs only)
        ↓
deduplicate_xy              →  blocks_deduped/ + forearm_deduped/
        ↓
project_contacts_onto_forearm → blocks_projected/  (CSVs only)
        ↓
calibrate_pca_xyz           →  blocks_pca_calibrated/ + forearm_pca_calibrated/
        ↓
center_on_receptive_field   →  blocks_rf_centered/ + forearm_rf_centered/
        ↓
aggregate_session           →  {session_id}_aggregated_session.csv + {session_id}_forearm.ply
```

**Context dict keys flowing through orchestrator:**
```python
context = {
    "source_files":     [...],        # from input resolution
    "session_configs":  [...],        # from input resolution
    "source_forearm":   Path,         # from fetch_forearm_of_reference
    "registered_files": [...],        # from apply_icp_registration
    "deduped_files":    [...],        # from deduplicate_xy
    "deduped_forearm":  Path,         # from deduplicate_xy
    "projected_files":  [...],        # from project_contacts_onto_forearm
    "pca_files":        [...],        # from calibrate_pca_xyz
    "pca_forearm":      Path,         # from calibrate_pca_xyz
    "pca_json":         Path,         # from calibrate_pca_xyz
    "rf_files":         [...],        # from center_on_receptive_field
    "rf_forearm":       Path,         # from center_on_receptive_field
    "aggregated_file":  Path,         # from aggregate_session
}
```

### Constraints (from Knowledge Base)

- **Contact points serialization:** Format `[[x1 y1 z1] [x2 y2 z2]]` must be preserved through all transforms (note-spatial-alignment-pipeline).
- **Scheduled transforms:** `transform_spatial_columns_scheduled()` handles per-frame transforms within a single block CSV — must not be broken (note-forearm-icp-registration).
- **Single-forearm no-op:** Sessions with only one forearm snapshot must still work (ICP registration copies files unchanged, PCA calibration still computes from CSVs).

---

## Implementation Plan

### Phase 1: New fetch task + package exports
**Goal:** Create `fetch_forearm_of_reference` task and update package exports
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Create `code/scripts/_5_postprocessing/fetch_forearm_of_reference.py`
- [x] Update `code/scripts/_5_postprocessing/__init__.py` to export the new function

**Files Modified:**
- `code/scripts/_5_postprocessing/fetch_forearm_of_reference.py` — **Create:** task function that resolves unified/single forearm PLY and copies to `forearm_source/{session_id}_forearm.ply`
- `code/scripts/_5_postprocessing/__init__.py` — Add `fetch_forearm_of_reference` export

**Dependencies:** None

### Phase 2: Merge PCA calibration + forearm export
**Goal:** Combine `set_xyz_reference_from_gestures` and `export_forearm_pca_calibrated` into `calibrate_pca_xyz`
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Refactor `set_xyz_reference_from_gestures.py` to accept `forearm_ply_path` and `forearm_output_dir` parameters
- [x] Add forearm PCA transform logic (absorbed from `export_forearm_pca_calibrated.py`)
- [x] Update return type to include forearm output path
- [x] Rename the main function to `calibrate_pca_xyz` (keep old name as alias if needed for transition)
- [x] Delete `export_forearm_pca_calibrated.py`
- [x] Update `__init__.py` exports: add `calibrate_pca_xyz`, remove `export_forearm_pca_calibrated`

**Files Modified:**
- `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py` — Refactor: add forearm PLY handling, rename to `calibrate_pca_xyz`
- `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py` — **Delete**
- `code/scripts/_5_postprocessing/__init__.py` — Update exports

**Dependencies:** Phase 1

### Phase 3: Rewrite orchestrator
**Goal:** Update the workflow script with new stage sequence and context flow
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Remove `_resolve_session_forearm()` helper
- [x] Remove `_resolve_latest_forearm_ply()` helper
- [x] Add `fetch_forearm_of_reference_flow` wrapper
- [x] Replace `set_xyz_reference_from_gestures_flow` + `export_forearm_pca_calibrated_flow` with `calibrate_pca_xyz_flow`
- [x] Rewrite `pipeline_stages` list with new 7-stage sequence
- [x] Update context key assignments
- [x] Update imports

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — Rewrite orchestrator stages, flow wrappers, remove old helpers

**Dependencies:** Phase 2

### Phase 4: Config, aggregate, and visualization updates
**Goal:** Update all supporting files for new directory names and task structure
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Rewrite `configs/postprocess_workflow_kinect_auto_dag.yaml` with new task keys and dependencies
- [x] Update `aggregate_session_blocks` to enforce target name `{session_id}_forearm.ply` regardless of input filename
- [x] Update `postprocess_visualization.py` path resolution for renamed directories (`blocks_deduped`, `blocks_projected`)
- [x] Update `postprocessing_stage_viewer.py` stage labels and path constants

**Files Modified:**
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — New task structure
- `code/scripts/_4_merging/aggregate_blocks_session.py` — Enforce `{session_id}_forearm.ply` naming
- `code/scripts/postprocess_visualization.py` — Update directory references
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — Update stage constants

**Dependencies:** Phase 3

---

## Testing Plan

### Integration Tests
- [ ] Run full pipeline on ST13-03 block-order05 — all stages complete without error
- [ ] Verify output directory structure matches expected layout
- [ ] Verify `{session_id}_forearm.ply` at session root is loadable by `resolve_forearm_ply()`

### Manual Verification
- [ ] Check `forearm_source/` contains a valid PLY after fetch step
- [ ] Check `forearm_deduped/` PLY has fewer vertices than source (dedup worked)
- [ ] Check `forearm_pca_calibrated/` PLY is in PCA space (visual inspection via stage viewer)
- [ ] Check `forearm_rf_centered/` PLY is origin-centered (visual inspection)
- [ ] Run visualization stage viewer — all 6 stages render correctly
- [ ] Confirm analysis pipeline can still load forearm via `resolve_forearm_ply(session_dir, session_id)`

### Edge Cases
- [ ] Single-forearm session (no `_unified_registered.ply`) — fetch task falls back to single PLY
- [ ] Session with all tasks disabled except fetch — pipeline exits cleanly
- [ ] Force processing = true — all tasks re-run even when outputs exist

---

## Documentation Plan

- [ ] Update `CLAUDE.md` architecture table (postprocessing stages)
- [ ] Update `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` with new stage names and directory paths

---

## Rollback Plan

1. **Before deployment:** All changes are on a feature branch; revert = delete branch
2. **Data considerations:** No migrations. Old output directories (`blocks_registered_deduped/`, `blocks_registered_projected/`) will remain on disk from prior runs but won't be written to. No data loss.
3. **Rollback procedure:** `git merge --abort` or revert the merge commit on `dev`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Visualization path resolution breaks silently | Medium | Medium | Test stage viewer on existing data before merging |
| Analysis pipeline can't find forearm after refactor | Low | High | Aggregate step enforces `{session_id}_forearm.ply` naming — same as current |
| PCA calibration JSON path changes break downstream | Low | Medium | Keep JSON filename unchanged (`pca-xyz_transformation-matrices.json`) |
| Old output directories confuse users | Low | Low | Document in commit message that old dirs are stale artifacts |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Analysis forearm resolver: `code/src/analysis/receptive_field_mapping/rf_data_loader.py:resolve_forearm_ply()`
