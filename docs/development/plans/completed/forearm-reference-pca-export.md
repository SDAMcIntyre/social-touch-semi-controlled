# Plan: Forearm of Reference — PCA-Calibrated PLY Export

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/forearm-reference-pca-export`

---

## Overview

**What:** After PCA calibration (stage 2 of postprocessing), apply the same full transform (Z-alignment + XY-alignment) to the session's forearm-of-reference point cloud and save it as a PLY in a dedicated output directory.
**Why:** Currently, contact data lives in PCA-calibrated space but the forearm geometry remains in ICP space. Downstream analysis (receptive field mapping, visualization) needs the forearm surface in the same coordinate frame as the calibrated contact data.
**How:** Load the unified registered PLY, apply `PCACalibrationEngine.apply_full_transform`, write the result to a dedicated `forearm_pca_calibrated/` directory.

## Problem Statement

- After postprocessing stage 2, all spatial columns (stickers, contact locations, contact points) are in PCA-calibrated coordinates.
- The forearm-of-reference PLY (`{session_id}_unified_registered.ply`) remains in ICP-registered coordinates.
- Any visualization or analysis that overlays contacts on the forearm surface must either reverse-transform the contacts or forward-transform the forearm. A forward-transformed forearm PLY eliminates this mismatch once and avoids repeated on-the-fly transformation.

## Goals

### In Scope
1. Load the unified registered forearm PLY for each session
2. Apply the full PCA calibration (both phases) to its vertices
3. Save the transformed forearm as a PLY in a dedicated output directory (`forearm_pca_calibrated/`)

### Out of Scope
- Modifying the existing PCA calibration logic
- Transforming individual per-snapshot forearm PLYs (only the unified registered one)
- Adding visualization of the transformed forearm (separate concern)
- Changing the forearm catalog or registration pipeline

## Success Criteria

- [ ] A `{session_id}_forearm_pca_calibrated.ply` file is produced in `forearm_pca_calibrated/` under the session output directory
- [ ] The PLY vertices are in the same coordinate frame as the PCA-calibrated CSVs
- [ ] The step is idempotent (skipped when outputs are up-to-date)
- [ ] Single-forearm sessions (no ICP transforms) are handled correctly — use the raw forearm PLY as input
- [ ] The step integrates into the existing postprocessing workflow without disrupting stages 1 or 2

---

## Technical Design

### Knowledge Base Relevance Check

Reviewed all notes in `docs/development/knowledge-base/README.md`:
- **note-forearm-icp-registration.md** — Directly relevant. Reuse the same path conventions (`forearm_pointclouds/{session_id}_unified_registered.ply`) and the `ForearmCatalog` lookup patterns. The canonical-key / hold-last-frame semantics do not apply here (we only need the final unified PLY, not per-snapshot transforms).
- **note-somatosensory-units-and-calculations.md** — Relevant for context: all coordinates are in mm (Kinect SDK native). The PCA transform preserves units (rotation + translation in mm space), so the output PLY remains in mm.
- **note-cupy-import-order.md** — Not applicable: this feature does not use CuPy.
- **note-open3d-scenewidget-layout.md** — Not applicable: no GUI involved.

### Approach

Add a lightweight function at the end of `set_xyz_reference_from_gestures` (or as a new sub-step called immediately after it in the orchestrator) that:

1. Resolves the forearm PLY path from `session_configs` (via `forearm_pointclouds/{session_id}_unified_registered.ply`)
2. Loads it with Open3D (`o3d.io.read_point_cloud`)
3. Extracts the vertex positions as an `(N, 3)` numpy array
4. Applies `PCACalibrationEngine.apply_full_transform(vertices, calib_result)`
5. Writes the result back as a PLY to `forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply`

The preferred integration point is a **new helper function** called from `run_single_session_postprocessing` after stage 2, using the `calib_result` (or loading it from the saved JSON). This keeps stage 2's function signature unchanged and separates concerns.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New step after stage 2 in orchestrator | Clean separation, no change to existing stage 2 API | Needs to load calibration from JSON (trivial) | Chosen |
| Embed inside `set_xyz_reference_from_gestures` | Direct access to `calib_result` | Mixes CSV transformation with PLY I/O; requires passing `session_configs` into stage 2 | Rejected |
| Transform on-the-fly in viewer/analysis | No extra file | Repeated computation, coupling to calibration data | Rejected |

### Architecture Changes

New file:
```
code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py
```

This script:
- Loads the calibration JSON (`pca-xyz_transformation-matrices.json`) to reconstruct `CalibrationResult`
- Loads the unified registered PLY
- Applies the full PCA transform to the point cloud vertices
- Saves the output PLY

Modified files:
- `code/scripts/postprocess_workflow_kinect_auto.py` — add a step 3 in `pipeline_stages` that calls the new export function after PCA calibration
- `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py` — add `CalibrationResult.from_dict()` class method (inverse of `to_dict()`)

---

## Implementation Plan

### Phase 1: Core Transform and Export
**Goal:** Implement the forearm PLY transformation and saving logic

**Tasks:**
- [ ] Task 1.1 — Add `CalibrationResult.from_dict()` class method to `calibration_pca_engine.py`
- [ ] Task 1.2 — Create `export_forearm_pca_calibrated.py` with a function that loads calibration JSON + forearm PLY, applies full transform, writes output PLY
- [ ] Task 1.3 — Handle the single-forearm fallback: if `_unified_registered.ply` does not exist, look for the single forearm PLY in the pointclouds directory

**Files Modified:**
- `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py` — add `from_dict()`
- `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py` — new file

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Wire the new step into the postprocessing orchestrator

**Tasks:**
- [ ] Task 2.1 — Add a `export_forearm_pca_calibrated` flow wrapper in `postprocess_workflow_kinect_auto.py`
- [ ] Task 2.2 — Add step 3 to `pipeline_stages` list, consuming `pca_report` (the output_dir from stage 2) and `session_configs`
- [ ] Task 2.3 — Register the new task name in the DAG config YAML so the executor recognizes it

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — new flow + pipeline stage entry
- `configs/postprocess_workflow_kinect_auto_dag.yaml` (if it exists) — add task entry

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `CalibrationResult.from_dict(calib.to_dict())` round-trips correctly — reconstructed object produces identical transforms
- [ ] `apply_full_transform` on a known small point cloud (e.g., 4 vertices of a tetrahedron) matches hand-computed expected output

### Integration Tests
- [ ] `export_forearm_pca_calibrated()` with a real unified PLY + calibration JSON produces a valid PLY with the correct vertex count
- [ ] The orchestrator runs all 3 stages end-to-end on a test session without errors

### Manual Verification
- [ ] Run postprocessing on a multi-forearm session — verify `forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply` is produced
- [ ] Open the output PLY alongside a PCA-calibrated CSV in a viewer — confirm spatial alignment (contacts should sit on the forearm surface)
- [ ] Run postprocessing on a single-forearm session — verify the fallback PLY path works
- [ ] Run postprocessing twice — verify idempotency (second run skips)

### Edge Cases
- [ ] Session with no forearm PLY at all (e.g., corrupted extraction) — should log a warning and skip, not crash
- [ ] Session where PCA calibration failed (no JSON) — step should be skipped gracefully

---

## Documentation Plan

- [ ] Update README.md with new commands/features — N/A (no user-facing CLI change)
- [ ] Update CLAUDE.md with architecture changes — N/A (internal pipeline detail)
- [ ] Create/update user guide — N/A
- [ ] Add changelog entry — N/A (no versioned release)
- [ ] Update inline code comments in the orchestrator to describe the 3-stage pipeline

---

## Rollback Plan

1. **Before deployment:**
   - Verify no downstream scripts depend on `forearm_pca_calibrated/` output
   - Check that no analysis notebooks reference the calibrated forearm PLY path

2. **Data considerations:**
   - No migrations. Output is a new directory (`forearm_pca_calibrated/`) — no existing files are modified or overwritten
   - No breaking changes to existing stage 1 or stage 2 outputs

3. **Rollback procedure:**
   - Remove `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py`
   - Remove the step 3 flow wrapper and pipeline stage entry from `postprocess_workflow_kinect_auto.py`
   - Remove task entry from DAG config YAML (if added)
   - Optionally revert `CalibrationResult.from_dict()` (harmless utility, can be kept)
   - Delete `forearm_pca_calibrated/` output directories if produced

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Open3D not available in the postprocessing environment | Low | High | Already used by ICP registration in the same pipeline; same environment |
| Forearm PLY path differs between sessions | Low | Med | Resolve path from `KinectConfig` consistently with existing ICP step |
| Large PLY files slow down the step | Low | Low | Single PLY per session; transform is a matrix multiply — negligible cost |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core Transform and Export | ~80 lines new code, ~10 lines modified | None |
| Phase 2: Pipeline Integration | ~30 lines modified | Phase 1 |

---

## References

- Related Plans: `docs/development/plans/completed/multi-snapshot-forearm-registration.md`
- Knowledge Base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Key source: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
