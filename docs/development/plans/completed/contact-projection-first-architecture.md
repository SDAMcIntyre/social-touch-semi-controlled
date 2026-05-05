# Plan: Contact Projection First Architecture

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-05 07:15
**Base Branch:** `feature/analysis-workflow-processing-viewer-separation`
**Branch:** `feature/contact-projection-first-architecture`

---

## Overview

Move the contact-point projection step to run **before** ICP registration and PCA
calibration, and replace the current Hungarian assignment with independent per-point
nearest-neighbour projection.  Once contact points are snapped exactly onto the
per-block forearm surface, every subsequent rigid-body transform (ICP, PCA) moves both
the forearm and its on-surface contact points identically, so the on-surface property is
preserved through all later stages without further correction.

## Problem Statement

### Current architecture

```
blocks_merged/ → ICP → PCA → export forearm PLY → project_contacts → RF center
```

Each of ICP and PCA introduces a small residual between the forearm surface and the
contact points.  By the time projection runs (after both transforms), some points have
drifted into low-density or edge regions of the PCA forearm mesh.

### Silent point-loss bug

The current multi-point projection path (`project_contacts_onto_forearm.py`, lines
94–105) builds a unique candidate pool of size C from k=M nearest-neighbor queries and
then solves a Hungarian assignment on an (M, C) cost matrix.  The inline comment claims
"C >= M" but this is **not guaranteed**: when M contact points cluster in a sparse
forearm region, `np.unique` can return C < M.  `scipy.optimize.linear_sum_assignment`
on an (M, C) matrix with M > C returns arrays of length C, so M−C contact points are
**silently dropped** — no log, no warning, no exception — violating the project's
fail-fast convention.

The dropped points create spatial "holes" in the contact-point cloud that are
visible in the PostprocessingStageViewer when switching between the PCA Calibrated
and Contact Projected stages.

### Why "project first" fixes both problems

1. At the merge stage, contact points are already forearm-mesh vertices (extracted by
   `objects_interaction_processor.py`), so they lie on (or extremely close to) the
   per-block forearm surface.  Projecting them there snaps away any tiny floating-point
   offset with minimal displacement.
2. ICP and PCA are rigid-body transforms applied identically to both the forearm cloud
   and the contact points — on-surface membership is preserved exactly after PCA.
3. No late-stage projection is needed; the "Contact Projected" viewer stage becomes
   equivalent to the PCA-calibrated stage.

## Goals

### In Scope
1. Replace the Hungarian assignment in `project_contacts_onto_forearm.py` with
   independent per-point nearest-neighbour projection (no point loss possible).
2. Add a projection step in the workflow **before** ICP, using the per-block forearm
   from the preprocessing `forearm_pointclouds/` catalogue.
3. Remove the now-redundant late-stage projection step (current step 4) from the
   workflow.
4. Update `resolve_stage_paths()` in `postprocess_visualization.py` so the viewer
   shows the correct CSV at each slot.

### Out of Scope
- Changes to the ICP or PCA calibration algorithms.
- Changing the RF-centering or aggregation steps.
- Re-labelling or removing viewer stage slots (a cosmetic follow-up can do this).
- Backfilling or migrating previously processed session data.

## Success Criteria

- [ ] Running the full postprocess pipeline on a session that previously showed holes
  produces a contact-point cloud with no holes in any of stages 2–4 of the viewer.
- [ ] `projection_stats.csv` reports `n_points_projected` equal to the total contact-
  point count from the merged CSVs (no fewer) for every block.
- [ ] The fail-fast assertion `assert len(projected) == M` in the projection function
  never triggers in normal operation.
- [ ] All existing postprocess pipeline tests pass.

---

## Technical Design

### Approach

**Projection algorithm change** — replace the entire `if M == 1 / else` block with one
unconditional call:

```python
distances, indices = kdtree.query(query)   # (M,) — one nn per contact point
projected = vertices[indices]              # (M, 3)  — all M always assigned
assert len(projected) == M
```

This satisfies the user requirement "all initial contact points must remain" and
minimises the sum of per-point displacements (each point reaches its geometrically
nearest forearm vertex independently).  The uniqueness constraint is dropped; allowing
two nearby contact points to snap to the same vertex is acceptable and matches the
downstream analysis use case.

**Pipeline reordering** — insert a new step before ICP that calls
`project_contacts_onto_forearm` once per session config, passing the per-block forearm
as the projection surface.  Each block's per-video forearm is resolved via
`ForearmCatalog.get_forearms_with_fallback()` (same lookup used by the viewer).

```
blocks_merged/ ──[new step 0.5: project onto per-block forearm]──► blocks_merged_projected/
                                                                              │
                                                                        ICP → PCA → RF center
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Project onto per-block forearm before ICP (proposed) | Eliminates drift; no holes; simple algorithm | ICP residual still leaves a tiny gap (< mesh resolution) | **Chosen** |
| Project after ICP, before PCA | Unified forearm available; smaller drift than current | Still accumulates ICP residual; per-block forearms discarded | Rejected |
| Keep current position (after PCA); fix Hungarian | Minimal code change | C < M still possible at mesh edges; root cause not addressed | Rejected |
| Project after ICP onto unified forearm, keep late-stage as verification | Belt-and-suspenders | Extra step; dual projection overhead | Not needed |

### Architecture Changes

New output directory: `blocks_merged_projected/` — holds per-block CSVs after
early projection.  This becomes the input to the ICP step (replacing `blocks_merged/`).

The existing `blocks_contact_projected/` directory still exists after the change but
now holds the same data as `blocks_pca_calibrated/` (since projection already happened).
The `project_contacts_onto_forearm_flow()` call in the workflow is removed; the function
itself is kept and reused by the new early step.

### Knowledge-Base Constraints

- **Units**: All coordinates in mm (from Kinect SDK); the projection function must not
  introduce any unit conversion.
- **KDTree vertex snapping pattern**: `rf_simple_pipeline.py` already uses independent
  per-point KDTree snapping — the new algorithm follows the same pattern.
- **ICP parameters**: Unchanged (`max_correspondence_distance=0.10`, fitness ≥ 0.9).
  Moving projection before ICP does not affect ICP itself.

---

## Implementation Plan

### Phase 1: Replace projection algorithm

**Goal:** Make `project_contacts_onto_forearm.py` loss-proof and remove the
Hungarian machinery.

**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 1.1 — In `_project_single_csv()`, delete the `if M == 1 / else` block and
  replace with:
  ```python
  distances, indices = kdtree.query(query)
  projected = vertices[indices]
  assert len(projected) == M, (
      f"Projection dropped {M - len(projected)} of {M} contact points — "
      "impossible with per-point NN."
  )
  ```
- [x] Task 1.2 — Remove now-unused imports: `cdist` from `scipy.spatial.distance` and
  `linear_sum_assignment` from `scipy.optimize`.
- [x] Task 1.3 — Update the function docstring to reflect the new algorithm (no
  uniqueness constraint; all M points guaranteed in output).

**Files Modified:**
- `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py` — algorithm
  replacement, import removal, docstring update

**Dependencies:** None

### Phase 2: Add early projection step to the workflow

**Goal:** Run projection on `blocks_merged/` CSVs before ICP, writing output to
`blocks_merged_projected/`.

**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 2.1 — Add a helper `_load_per_block_forearm_ply(config: KinectConfig) ->
  Optional[Path]` (or accept the o3d PointCloud from `ForearmCatalog`) to
  `postprocess_workflow_kinect_auto.py`.  Mirrors `_load_per_video_forearm()` from
  `postprocess_visualization.py` but returns the PLY path rather than loading into
  memory (or saves the o3d cloud to a temp path if the function requires a Path).
- [x] Task 2.2 — Insert a new pipeline stage before `apply_icp_registration`:
  - Input CSVs: `context["source_files"]` (from `blocks_merged/`)
  - Forearm: per-block forearm resolved via `ForearmCatalog`
  - Output dir: `session_output_dir / "blocks_merged_projected"`
  - Stats path: `session_output_dir / "blocks_merged_projected" / "projection_stats.csv"`
  - Store result in `context["projected_source_files"]`
- [x] Task 2.3 — Change the ICP step to consume `context["projected_source_files"]`
  instead of `context["source_files"]`.
- [x] Task 2.4 — Remove the old step-4 `project_contacts_onto_forearm` call from
  `pipeline_stages` (the step that reads `pca_data_files` and writes to
  `blocks_contact_projected/`).
- [x] Task 2.5 — Update the step-5 RF-centering call: its input was
  `context["projected_files"]` (old step-4 output); change it to consume
  `context["pca_data_files"]` directly (PCA-calibrated = already on-surface).

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — new step, step reordering,
  ICP input change, old step-4 removal, RF-center input change

**Dependencies:** Phase 1

### Phase 3: Update viewer stage paths

**Goal:** Ensure the PostprocessingStageViewer shows the correct CSV at each slot.

**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 3.1 — In `postprocess_visualization.py`, update `resolve_stage_paths()`:
  - Stage 0 "Merged (Raw)": unchanged — `blocks_merged/`
  - Stage 1 "ICP Registered": unchanged — `blocks_registered/`
  - Stage 2 "PCA Calibrated": unchanged — `blocks_pca_calibrated/`
  - Stage 3 "Contact Projected": point to `blocks_pca_calibrated/` CSV (data is
    already on-surface; the separate `blocks_contact_projected/` directory no longer
    exists) OR relabel the slot as "PCA (On-Surface)" in a follow-up
  - Stage 4 "RF Centered": unchanged — `blocks_rf_centered/`
- [x] Task 3.2 — Verify that `resolve_before_after_paths()` (used by the Before/After
  step viewer) still resolves valid paths after the removal of `blocks_contact_projected/`.
  Updated Step 3 `after_csv` and Step 4 `before_csv` to use `blocks_pca_calibrated/`
  instead of the defunct `blocks_contact_projected/`.

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — `resolve_stage_paths()` slot 3 path

**Dependencies:** Phase 2

### Phase 4: DAG config update

**Goal:** Ensure the DAG config reflects the new step order.

**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 4.1 — Open `configs/postprocess_visualization_dag.yaml` (already modified
  in working tree) and verify that task `project_contacts_onto_forearm` either refers
  to the new early step or is removed.
- [x] Task 4.2 — Update `depends_on` chains if the old step was an explicit dependency
  for downstream tasks.

**Files Modified:**
- `configs/postprocess_workflow_kinect_auto_dag.yaml`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] Test `_project_single_csv()` with M=1 point — assert output has 1 point.
- [ ] Test `_project_single_csv()` with M=5 clustered points all with the same nearest
  vertex — assert output has 5 points (no longer drops any).
- [ ] Test `_project_single_csv()` with an empty `contact_points` cell — assert
  row passes through unchanged.

### Manual Verification
- [ ] Run the full postprocess pipeline on a session that previously exhibited holes.
  In the PostprocessingStageViewer switch between "PCA Calibrated" and "Contact
  Projected" and confirm the cloud topology is identical (no new holes introduced).
- [ ] Open the stage viewer at Stage 0 "Merged (Raw)" and Stage 1 "ICP Registered"
  and confirm contact-point clouds display correctly with no regressions.
- [ ] Check `blocks_merged_projected/projection_stats.csv`: `n_points_projected` must
  equal the sum of contact-point counts in `blocks_merged/` (no fewer).

### Edge Cases
- [ ] Session with only a single block (no ICP multi-forearm registration) — projection
  still runs and produces a valid output.
- [ ] Session where the per-block forearm metadata is missing — workflow logs a warning
  and skips projection for that block (fail-fast for unexpected states, graceful skip for
  missing prerequisite data).
- [ ] Block where all frames have empty `contact_points` — projection writes through
  unchanged rows with no crash.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` if it references the stage ordering.
- [ ] No README or user-guide changes required (internal pipeline step).

---

## Rollback Plan

1. The old `blocks_contact_projected/` data (if already generated) is not deleted by
   this change — it simply stops being written.  Previously processed sessions can be
   re-run with `force_processing=True` to regenerate under the new architecture.
2. To revert: restore `postprocess_workflow_kinect_auto.py` from git, which restores
   the old step-4 call and removes the new step 0.5.  The projection function change
   (Phase 1) is backward-compatible — the simpler NN algorithm produces valid output
   for any input.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Per-block forearm metadata missing for some sessions | Low | Medium | Log warning and fall back to skipping projection for that block; ICP still runs on raw merged data |
| ICP residual leaves contact points slightly off unified forearm after registration | Medium | Low | Acceptable; gap is < mesh vertex spacing; far smaller than current accumulated error |
| `ForearmCatalog` not available in workflow import context | Low | Medium | Check imports; `ForearmCatalog` is already imported in `postprocess_visualization.py` so the pattern is established |
| `blocks_contact_projected/` path hardcoded in other scripts | Low | Low | Grep for references before removing the step |

---

## References

- Investigation notes: `.claude/plans/the-postprocessing-stage-viewer-serene-avalanche.md`
- KDTree snapping pattern: `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py`
- Per-video forearm loading: `code/scripts/postprocess_visualization.py:_load_per_video_forearm()`
- Hungarian assignment (to be removed): `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py:94-105`
