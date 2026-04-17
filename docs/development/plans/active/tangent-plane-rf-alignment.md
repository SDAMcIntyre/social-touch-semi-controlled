# Plan: Tangent-Plane Alignment for RF Heatmap Rendering

**Date:** 2026-04-17
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/tangent-plane-rf-alignment`

---

## Overview

Add a render-time coordinate rotation to RF heatmap visualizations so that
each session's forearm surface is displayed face-on with a consistent
orientation: surface normal pointing at the camera (Z), forearm longitudinal
axis horizontal (X). This enables meaningful visual comparison of receptive
field maps across sessions without modifying any persisted pipeline data.

## Problem Statement

After PCA calibration and RF centering, each session's data has its origin at
the RF hotspot but the forearm surface is not perpendicular to the viewing
direction. The forearm curves differently at each RF location, so the camera
must be oriented per-session to get a face-on view. Currently this is done via
`_compute_surface_normal()` + angle conversion, but the resulting views still
have inconsistent "up" and "horizontal" directions across sessions, making
cross-session comparison unreliable.

## Goals

### In Scope

1. Compute a tangent-plane rotation matrix at the RF center using the local
   surface normal and the PCA-calibrated longitudinal axis
2. Apply the rotation at render time in the matplotlib heatmap renderer
3. Apply the rotation in the PyVista camera angle picker
4. Provide a consistent face-on default camera across all sessions

### Out of Scope

- Persisting rotated data to disk (this is render-time only)
- Modifying the postprocessing pipeline stages
- Changing the RF centering algorithm
- GPU-accelerated rendering

## Success Criteria

- [ ] All RF heatmap PNGs show the forearm surface face-on (normal pointing at camera)
- [ ] The forearm longitudinal (proximal-distal) axis is consistently horizontal across sessions
- [ ] The camera angle picker starts in the aligned frame by default
- [ ] Rotation matrix is orthonormal (R @ R.T = I, det(R) = +1) for all sessions
- [ ] Graceful fallback when normal estimation fails (current behavior preserved)

---

## Technical Design

### Approach

Compute a 3x3 rotation matrix R that maps each session's RF-centered
coordinates into a canonical tangent-plane frame:

1. **Z_new** = surface normal at origin, computed via KD-tree k=50 neighbors +
   SVD (reusing `_compute_surface_normal()`). Oriented outward: flip if
   `Z_new . [0,0,1] < 0`.
2. **X_new** = PCA X-axis `[1,0,0]` projected onto tangent plane:
   `X_proj = [1,0,0] - ([1,0,0] . Z_new) * Z_new`, normalized. Fallback to
   `[0,1,0]` if degenerate.
3. **Y_new** = `Z_new x X_new` (right-hand rule).
4. **R** = rows `[X_new; Y_new; Z_new]`. Enforce `det(R) = +1`.
5. **Apply**: `points_aligned = points @ R.T`

After rotation the camera looks down -Z with Y up, giving a consistent face-on
view with the longitudinal axis horizontal.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Render-time rotation (chosen) | No data changes, reversible, simple | Must apply in each renderer | Chosen |
| New postprocessing stage | Persistent, applied once | Adds pipeline complexity, modifies data | Rejected |
| Camera-only adjustment (current) | Already exists | Inconsistent "up" direction, no true alignment | Augmented |

### Architecture Changes

New module: `tangent_plane_alignment.py` — pure math, no I/O.

Modified modules:
- `rf_cluster_visualizer.py` — rotate points before matplotlib scatter
- `rf_camera_angle_task.py` — pre-rotate scene data, add rotation to dataclass
- `rf_camera_angle_picker.py` — apply rotation to scene, set default camera

### Knowledge Base Constraints

- **note-spatial-alignment-pipeline.md**: This rotation extends the 5-space
  pipeline as a 6th render-time-only space. No persistence needed.
- **note-forearm-icp-registration.md**: Follow the pattern of logging alignment
  quality metrics. Handle single-forearm sessions with no-op path.
- **note-somatosensory-units-and-calculations.md**: All coordinates stay in mm.
  Rotation is dimensionless; axis labels unchanged.

---

## Implementation Plan

### Phase 1: Tangent-Plane Rotation Utility

**Goal:** Create the pure-math module that computes the rotation matrix.

**Started:** 2026-04-17
**Completed:** 2026-04-17

**Tasks:**
- [x] Task 1.1 — Create `tangent_plane_alignment.py` with `compute_tangent_plane_rotation()` and `align_points()`
- [x] Task 1.2 — Refactor `_compute_surface_normal()` out of `rf_cluster_visualizer.py` into the new module (re-export from visualizer for backward compat)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` — New file
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — Import `_compute_surface_normal` from new module

**Dependencies:** None

### Phase 2: Matplotlib Heatmap Integration

**Goal:** Apply tangent-plane rotation in the static heatmap renderer.

**Started:** 2026-04-17
**Completed:** 2026-04-17

**Tasks:**
- [x] Task 2.1 — In `render_forearm_heatmap()`, compute R after loading forearm vertices
- [x] Task 2.2 — Rotate `forearm_sub` and spike point coordinates before plotting
- [x] Task 2.3 — Set camera to fixed face-on view (`elev=0, azim=0`) when R is available; keep current angle-based fallback otherwise

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — Add rotation before scatter calls, update `view_init`

**Dependencies:** Phase 1

### Phase 3: Camera Angle Picker Integration

**Goal:** Pre-rotate scene data in the interactive PyVista viewer.

**Started:** 2026-04-17
**Completed:** 2026-04-17

**Tasks:**
- [x] Task 3.1 — Add `tangent_rotation: Optional[np.ndarray]` field to `SessionSceneData`
- [x] Task 3.2 — Compute and store R in `collect_session_scene_data()`; pre-rotate `forearm_points` and `contact_points`
- [x] Task 3.3 — In `RFCameraAnglePicker`, set default camera to look along -Z with Y up when tangent rotation was applied

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — Add rotation field, compute R, pre-rotate points
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py` — Update default camera orientation

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `compute_tangent_plane_rotation()` on a synthetic half-cylinder: Z_new should align with the radial direction at the query point
- [ ] R is orthonormal: `R @ R.T ≈ I` within 1e-10
- [ ] `det(R) = +1` (proper rotation, not reflection)
- [ ] `align_points()` round-trips: `align_points(points, R) @ R = points`
- [ ] Degenerate case: normal parallel to [1,0,0] falls back to [0,1,0] projection

### Manual Verification
- [ ] Run RF heatmap rendering for a known session — surface should face camera directly
- [ ] Compare two sessions of the same neuron — longitudinal axis should be horizontal in both
- [ ] Open camera angle picker — default view should be face-on with consistent orientation
- [ ] Verify axis labels still show mm units

### Edge Cases
- [ ] Forearm PLY missing — graceful fallback to current behavior
- [ ] Very few forearm vertices near RF center (k > available points)
- [ ] Surface normal nearly parallel to PCA X-axis (degenerate tangent projection)

---

## Documentation Plan

- [ ] Update knowledge base `note-spatial-alignment-pipeline.md` to document the 6th render-time coordinate space
- [ ] Add inline docstrings to new module functions

---

## Rollback Plan

Since this is render-time only with no data changes:

1. Revert the 3-4 modified files to restore previous rendering behavior
2. No data migrations or state changes to undo

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Surface normal estimation degenerate at some RF locations | Low | Low | Fallback to current camera-angle approach; log warning |
| PCA X-axis nearly parallel to surface normal | Very Low | Low | Fall back to projecting [0,1,0]; detect via norm threshold |
| Matplotlib 3D view_init(0,0) not looking down -Z as expected | Low | Med | Verify with test render; adjust elev/azim if axis convention differs |

---

## References

- Related module: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py` — determinant enforcement pattern
- Related function: `rf_cluster_visualizer.py:_compute_surface_normal()` — reused for normal estimation
- Knowledge base: `note-spatial-alignment-pipeline.md`, `note-forearm-icp-registration.md`
