# Plan: Project Contact Points onto Forearm of Reference

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/contact-point-forearm-projection`

---

## Overview

Add a postprocessing Stage 4 that snaps each session's contact points onto the PCA-calibrated forearm surface using nearest-point projection. This ensures contact points lie exactly on the forearm-of-reference geometry, eliminating small spatial discrepancies caused by mesh resolution, penetration-depth variation, and registration artifacts.

## Problem Statement

After stages 1–3, both the contact data (CSVs) and the forearm surface (PLY) are in PCA-calibrated space. However, contact points originate from per-frame signed-distance detection on the forearm mesh and may float slightly off the reference surface. When aggregating contact points across touches or sessions for receptive field mapping, these small discrepancies reduce spatial precision. Projecting all contact points onto the reference forearm unifies their positions to the canonical surface.

## Goals

### In Scope
1. Nearest-point projection of `contact_points` onto the PCA-calibrated forearm PLY
2. Updated `contact_location_x/y/z` columns (mean of projected points)
3. New output directory `sessions_contact_projected/` with projected CSVs
4. Integration into the existing postprocessing DAG as Stage 4

### Out of Scope
- Cross-session forearm unification (each session uses its own forearm-of-reference)
- Mesh-based projection (using triangle surfaces rather than vertices)
- Modifying upstream contact detection logic

## Success Criteria

- [ ] Output CSVs exist in `sessions_contact_projected/` for each processed session
- [ ] Every non-empty `contact_points` cell contains only coordinates that are actual forearm PLY vertices
- [ ] `contact_location_x/y/z` equals the mean of projected points per row
- [ ] Rows with empty contact points pass through unchanged (NaN location preserved)
- [ ] Stage is idempotent (skipped when outputs are up-to-date, unless force_processing)
- [ ] Sessions without a forearm PLY are handled gracefully (warning, skip)

---

## Technical Design

### Approach

Build a KD-tree from the PCA-calibrated forearm PLY vertices (once per session), then for each CSV row, parse the `contact_points` string, query the KD-tree for the nearest vertex per point, and replace. This is the simplest approach that guarantees all output points lie exactly on the reference surface.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Nearest-vertex (KD-tree) | Simple, fast, guarantees output on surface | Limited to vertex resolution | **Chosen** |
| Normal-ray projection | More geometrically precise | Requires good surface normals; complex | Rejected — overkill for sub-mm adjustments |
| ICP between forearms then transform | Handles large misalignments | Adds complexity; misalignments are small post-PCA | Rejected — upstream stages already align |

### Architecture Changes

No new modules or classes. One new script following the exact same pattern as Stage 3 (`export_forearm_pca_calibrated.py`).

**Reused utilities:**
- `parse_contact_points()` / `serialize_contact_points()` — `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- `should_process_task()` — `code/src/utils/should_process_task.py`
- `open3d.io.read_point_cloud()` — PLY loading
- `scipy.spatial.KDTree` — nearest-neighbor queries

---

## Implementation Plan

### Phase 1: Core Projection Script
**Goal:** Implement the projection logic as a standalone function

- [ ] Create `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`
  - `_load_forearm_vertices(ply_path)` — load PLY, return `(N, 3)` array
  - `_project_single_csv(input_csv, output_csv, kdtree, vertices)` — parse/project/serialize each row
  - `project_contacts_onto_forearm(input_files, forearm_ply_path, output_dir, force_processing)` — entry point
- [ ] Handle edge cases: None PLY path, empty contact points, empty PLY

**Files Modified:**
- `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py` — **new file**

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Wire the new stage into the orchestrator and DAG config

- [ ] Add import to `code/scripts/_5_postprocessing/__init__.py`
- [ ] Add `@flow` wrapper in `code/scripts/postprocess_workflow_kinect_auto.py`
- [ ] Add Stage 4 entry to `pipeline_stages` list (after Stage 3)
- [ ] Add `project_contacts_onto_forearm` task to `configs/postprocess_workflow_kinect_auto_dag.yaml` with `depends_on: [set_xyz_reference_from_gestures, export_forearm_pca_calibrated]`

**Files Modified:**
- `code/scripts/_5_postprocessing/__init__.py` — add import line
- `code/scripts/postprocess_workflow_kinect_auto.py` — add flow wrapper + pipeline stage
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — add task entry

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline on one session with `force_processing: true` for the new task
- [ ] Confirm CSVs appear in `sessions_contact_projected/`
- [ ] Load output CSV and forearm PLY; verify every projected point is an exact PLY vertex
- [ ] Verify `contact_location_x/y/z` = mean of projected points for a sample row
- [ ] Verify rows with `"[]"` contact_points are unchanged
- [ ] Disable the task in DAG config → confirm pipeline skips it cleanly

### Edge Cases
- [ ] Session with no forearm PLY (Stage 3 returned None) → warning logged, stage skipped
- [ ] CSV rows where all contact_points are empty → output identical to input
- [ ] Re-run with `force_processing: false` → outputs skipped (idempotent)

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal pipeline stage)
- [ ] Inline docstrings in the new script

---

## Rollback Plan

1. Remove `project_contacts_onto_forearm` from DAG config (set `enabled: false` or delete entry)
2. Remove Stage 4 from `pipeline_stages` in orchestrator
3. Delete `sessions_contact_projected/` output directories if generated
4. No upstream data is modified — rollback has zero impact on existing outputs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KD-tree query snaps to wrong vertex (far away) | Low | Med | Contact points are already near the surface; distance would be sub-mm. Could add a max-distance threshold and log warnings for outliers. |
| Large CSVs slow to process row-by-row | Low | Low | Matches existing pattern in `csv_spatial_transformer.py`. Vectorized batch possible later if needed. |
| Stage 3 returns None for some sessions | Med | Low | Handled by early return with warning log. |
