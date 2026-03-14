# Plan: Fix PCA Calibration Mirroring in Postprocess Visualization

**Created:** 2026-03-14
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/postprocessing-visualization-pipeline` (existing)

---

## Overview

**What:** Fix the Y-axis mirroring observed in the `PostprocessedSceneViewer` advanced mode 3D scene by enforcing proper rotations (det = +1) in the PCA calibration engine.

**Why:** The PCA calibration can produce improper rotations (reflections) due to the sign ambiguity inherent in PCA eigenvectors, causing the entire calibrated coordinate system to be mirrored.

**How:** Add determinant checks after constructing R1 and R2 in `PCACalibrationEngine.compute_calibration()`, flipping the appropriate row when det < 0. Then re-run the postprocessing pipeline to regenerate all downstream data.

## Problem Statement

The `PostprocessedSceneViewer` advanced mode shows the 3D scene mirrored on the Y axis. All spatial data (forearm PLY, contact points, stickers, hand mesh) passes through `PCACalibrationEngine`, which assembles rotation matrices R1 and R2 from `sklearn.PCA.components_` rows. These rows have **arbitrary sign** — sklearn uses the SVD sign convention where the largest-magnitude element of each component is positive, but this heuristic does not guarantee a right-handed coordinate system. When `det(R1) = -1` or `det(R2[:2,:2]) = -1`, the transform is an improper rotation (rotation + reflection), causing the observed mirroring.

Since all data flows through the same calibration, the mirroring is globally consistent — objects are spatially coherent with each other, just reflected. This makes the issue purely visual in the current viewer but would produce subtle errors in any future analysis that assumes a right-handed coordinate frame.

## Goals

### In Scope
1. Enforce `det(R1) = +1` and `det(R2[:2,:2]) = +1` in the PCA calibration engine
2. Re-run the postprocessing pipeline to regenerate calibration artifacts
3. Verify the fix in the `PostprocessedSceneViewer` advanced mode

### Out of Scope
- Modifying the `PostprocessedSceneViewer` rendering code
- Changing the camera up vector or other display-level workarounds
- Modifying the ICP registration pipeline (ICP already produces proper rotations)

## Success Criteria

- [ ] `PCACalibrationEngine.compute_calibration()` always produces R1 with `det(R1) = +1`
- [ ] `PCACalibrationEngine.compute_calibration()` always produces R2 with `det(R2[:2,:2]) = +1`
- [ ] Y-axis mirroring is no longer visible in `PostprocessedSceneViewer` advanced mode
- [ ] Hand mesh, stickers, and contact points appear in anatomically correct (non-mirrored) orientation

---

## Technical Design

### Approach

Add determinant checks immediately after R1 and R2 construction in `compute_calibration()`. When the determinant is -1, flip the sign of one row to convert the improper rotation to a proper rotation. This is the standard technique for handling PCA sign ambiguity.

**Which row to flip for R1:** Row 2 (Z axis = PC1, tapping depth). The tapping-depth axis has the most arbitrary sign — depth can be measured inward or outward — so flipping it preserves the more semantically meaningful X and Y orientations.

**Which row to flip for R2:** Row 1 (Y component in the XY 2D PCA). X is aligned with the primary stroking direction (more semantically anchored); Y is the perpendicular complement.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Fix in PCA calibration engine (source) | Fixes data at the root; all consumers get correct data; principled | Requires re-running pipeline to regenerate artifacts | **Chosen** |
| Negate Y in the viewer (display-only) | No re-run needed; immediate visual fix | Band-aid; leaves data in reflected coordinate system; must be replicated in every consumer | Rejected |
| Adjust camera up vector | Simple display tweak | Does not fix the data; only masks the symptom for one specific viewing angle | Rejected |

### Architecture Changes

No new modules or classes. One existing file is modified with 4 lines of code (2 determinant checks + 2 conditional flips).

---

## Implementation Plan

### Phase 1: Fix PCA Calibration Engine
**Goal:** Enforce proper rotations in R1 and R2

- [ ] Add `if np.linalg.det(R1) < 0: R1[2] *= -1` after R1 construction (line 50)
- [ ] Add `if np.linalg.det(R2[:2, :2]) < 0: R2[1, :2] *= -1` after R2 construction (line 67)

**Files Modified:**
- `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py` — Add 4 lines: determinant checks after R1 and R2 construction

**Dependencies:** None

### Phase 2: Re-run Postprocessing Pipeline
**Goal:** Regenerate all downstream data with correct (non-reflected) calibration

- [ ] Re-run `set_xyz_reference_from_gestures` — recomputes `pca-xyz_transformation-matrices.json`
- [ ] Re-run `export_forearm_pca_calibrated` — regenerates forearm PLY files
- [ ] Re-run `project_contacts_onto_forearm` — regenerates contact projection CSVs

**Files Modified:** None (pipeline re-execution only)

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] After Phase 1, add temporary debug print `print(f"det(R1)={np.linalg.det(R1):.1f}, det(R2_xy)={np.linalg.det(R2[:2,:2]):.1f}")` during pipeline re-run to confirm both determinants are +1
- [ ] Launch `PostprocessedSceneViewer` in advanced mode — Y-axis mirroring should be gone
- [ ] Visually confirm: hand mesh, stickers, and contact points appear in anatomically correct orientation

### Edge Cases
- [ ] Sessions where det(R1) was already +1 should be unaffected (determinant check is a no-op)
- [ ] Multiple sessions produce consistent orientation (all right-handed)

---

## Documentation Plan

- [ ] No external documentation changes needed (internal bugfix)
- [ ] Inline comment on the determinant check explains the PCA sign ambiguity rationale

---

## Rollback Plan

1. Revert the 4-line change in `calibration_pca_engine.py`
2. Re-run the postprocessing pipeline to regenerate artifacts with the old calibration
3. No data schema changes — rollback is safe and complete

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flipping Z changes the semantic meaning of "positive Z" for existing analyses | Low | Med | Z sign was arbitrary to begin with (PCA); no existing analysis depends on a specific Z direction |
| Some sessions had correct det already, others had det=-1; re-running changes only the reflected ones | Med | Low | This is the desired behavior — sessions are corrected to a consistent right-handed frame |

---

## References

- Active plan: `docs/development/plans/active/postprocessing-visualization-pipeline.md`
- PCA engine: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md` (confirms ICP produces proper rotations)
