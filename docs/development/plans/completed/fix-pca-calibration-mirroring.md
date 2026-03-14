# Plan: Fix PCA Calibration Mirroring in Postprocess Visualization

**Created:** 2026-03-14
**Author:** Basil Duvernoy
**Completed:** 2026-03-14
**Status:** Completed
**Branch:** `feature/postprocessing-visualization-pipeline` (existing)

---

## Overview

**What:** Fix the mirroring (left hands appearing as right hands) in the `PostprocessedSceneViewer` by computing PCA-aware camera settings instead of reusing kinect-space camera defaults.

**Why:** The postprocess viewer copies the neural/kinect viewer's hardcoded camera direction, but the data is in PCA-calibrated space where the axes are rotated. The camera ends up looking at the arm from the wrong side, producing a mirrored view in both simple and advanced mode.

**How:** Transform the camera direction and up vector through the PCA composite rotation (`R2 @ R1`) so the viewer looks at PCA-calibrated data from the same physical direction as the kinect viewer looks at kinect-space data.

## Problem Statement

The `PostprocessedSceneViewer` shows left hands as right hands in both simple and advanced mode. Both the forearm pointcloud (pre-baked in PCA space) and the hand mesh model (transformed at render time) appear mirrored. The same dataset viewed through the `NeuralKinectViewer` displays correctly.

A previous fix attempt enforced `det(R1) = +1` and `det(R2[:2,:2]) = +1` in the PCA calibration engine. This is mathematically correct (prevents data reflections) but did not solve the visual issue because it addresses a different problem (data integrity vs. viewing angle).

**Root cause:** Both viewers use identical hardcoded camera settings its:
```python
self.plotter.camera.position = [cx, cy, cz - 400.0]
self.plotter.camera.up       = [0.375, -0.904, -0.201]
```
These were designed for kinect camera space where +Z points away from the kinect. In PCA space, Z = tapping depth (perpendicular to arm surface), X = stroking direction, Y = cross-arm. The camera direction `[0, 0, 1]` has completely different physical meaning in each space. The result: the camera looks at the arm from the wrong side, mirroring the view.

## Goals

### In Scope
1. Compute PCA-aware camera orientation in `PostprocessedSceneViewer`
2. Pass PCA calibration to simple mode (currently only passed in advanced mode)
3. Fix the "Recenter" button to use PCA-aware camera orientation

### Out of Scope
- Modifying the PCA calibration engine (determinant fix is already in place and correct)
- Modifying the `NeuralKinectViewer` (camera settings are correct for kinect space)
- Re-running the postprocessing pipeline (no data changes needed)
- Changing the ICP registration pipeline

## Success Criteria

- [ ] Forearm pointcloud appears as a left arm in both simple and advanced mode (matching the neural/kinect viewer)
- [ ] Hand mesh (advanced mode) appears as a left hand
- [ ] "Recenter" button returns to the corrected camera angle
- [ ] Graceful fallback when `pca_calib` is `None` (uses original hardcoded camera)

---

## Technical Design

### Approach

Transform the kinect-space camera defaults through the PCA composite rotation. The PCA transform for directions is:

```
d_pca = R_composite @ d_kinect    where R_composite = R2 @ R1
```

**Derivation:** The full PCA point transform is `p_pca = ((p - mean_1) @ R1.T - [mean_2_x, mean_2_y, 0]) @ R2.T`. For directions (translation-invariant): `d @ R1.T @ R2.T`, which in column-vector form is `R2 @ R1 @ d`. NumPy's `matrix @ vector` uses column-vector convention for 1D arrays, so `R_composite @ d_kinect` is correct.

Add a helper method `_compute_camera_params()` to the viewer that:
1. Takes the kinect-space camera direction `[0, 0, 1]` and up `[0.375, -0.904, -0.201]`
2. Rotates both through `R2 @ R1` if PCA calibration is available
3. Computes camera position as `centroid - 400 * view_dir_pca`
4. Includes a safety guard for degenerate up vectors (nearly parallel to view direction)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Transform camera through PCA rotation | Exact same viewing angle as kinect viewer; no data changes; no pipeline re-run | Requires PCA calib in simple mode | **Chosen** |
| Constrain PCA axis directions (e.g. ensure `R1[2,2] > 0`) | Fixes at data level | Assumes kinect Z aligns with tapping depth — fails when arm is at an angle; requires pipeline re-run | Rejected |
| Negate Z offset in camera (`cz + 400` instead of `cz - 400`) | One-character change | Only fixes one axis; doesn't fix up vector; fragile across sessions | Rejected |
| Adjust camera up vector only | Simple | Doesn't fix viewing direction; partial fix at best | Rejected |

### Architecture Changes

No new modules or classes. One new private method added to `PostprocessedSceneViewer`. Two existing camera-setting locations updated to call it.

---

## Implementation Plan

### Phase 1: Add PCA-aware camera computation
**Goal:** Compute correct camera settings for PCA-calibrated space

- [ ] Add `_compute_camera_params()` method to `PostprocessedSceneViewer`
- [ ] Replace hardcoded camera in `_init_actors()` (line 519-523) with call to `_compute_camera_params()`
- [ ] Replace hardcoded camera in `_recenter_view()` (line 714-717) with call to `_compute_camera_params()`

**Files Modified:**
- `code/src/postprocessing/gui/postprocessed_scene_viewer.py` — Add `_compute_camera_params()` (~15 lines), update 2 camera-setting locations

**Dependencies:** None

### Phase 2: Pass PCA calibration to simple mode
**Goal:** Enable camera fix in simple mode

- [ ] Load PCA calibration JSON in `run_single_session_pipeline()` (simple mode launcher)
- [ ] Pass `pca_calib` to `PostprocessedSceneViewer` constructor in simple mode

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — Add `_load_pca_calib()` call and `pca_calib=` kwarg in simple mode launcher (~2 lines changed)

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Launch `postprocess_visualization.py` in simple mode — forearm pointcloud should appear as a left arm
- [ ] Launch in advanced mode — both forearm pointcloud and hand mesh should appear as left hands
- [ ] Click "Recenter" button — should return to the corrected (non-mirrored) camera angle
- [ ] Compare side-by-side with neural/kinect viewer on the same dataset — visual orientation should match
- [ ] Test with a different session/dataset to confirm the fix generalizes

### Edge Cases
- [ ] `pca_calib` is `None` (missing JSON) — should fall back to original hardcoded camera without error
- [ ] Session where `det(R1)` was already +1 — camera fix should still produce correct orientation
- [ ] Degenerate case where PCA-rotated up vector is nearly parallel to view direction — safety guard should pick an orthogonal alternative

---

## Documentation Plan

- [ ] No external documentation changes needed (internal bugfix)
- [ ] Inline comment on `_compute_camera_params()` explains the kinect-to-PCA camera transform rationale

---

## Rollback Plan

1. Revert the camera computation changes in `postprocessed_scene_viewer.py`
2. Remove the `pca_calib=` kwarg from the simple mode launcher
3. No data changes to revert — this fix is purely display-level

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PCA-rotated up vector nearly parallel to view direction | Very Low | Med (rendering glitch) | Safety guard: if `dot(view, up) > 0.99`, compute orthogonal up via cross product |
| PCA calibration JSON missing in simple mode | Low | Low (no fix, but no crash) | Graceful fallback: `None` → original hardcoded camera |
| ICP rotation also contributes to viewing angle mismatch | Low | Low | ICP rotations are small corrections; the kinect viewer also ignores them in camera settings |

---

## References

- Active plan: `docs/development/plans/active/postprocessing-visualization-pipeline.md`
- PCA engine: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- Postprocess viewer: `code/src/postprocessing/gui/postprocessed_scene_viewer.py`
- Neural/kinect viewer (reference): `code/src/merging/gui/neural_kinect_scene_viewer.py`
- Launch script: `code/scripts/postprocess_visualization.py`
