# Plan: Fix RF Simple 3D Heatmap Camera Orientation Mismatch

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-12 10:42
**Base Branch:** `dev`
**Branch:** `fix/rf-simple-3d-heatmap-camera-orientation`

---

## Overview

The 3D heatmap PNG produced by `render_forearm_heatmap()` (Path A) does not match the orientation the researcher sets in the RF Camera Settings Viewer. Two bugs in the matplotlib viewing parameters cause a 90° on-screen rotation and inverted depth ordering. The fix is a 2-line change in `rf_cluster_visualizer.py`.

## Problem Statement

The RF Camera Settings Viewer (`RFCameraSettingsViewer`, PyVista) lets the researcher interactively orient the forearm and save camera parameters. Downstream, `camera_settings_to_rotation()` converts those parameters to a 3×3 rotation matrix R = [x_right; y_up; z_view_dir]. The 3D rendering path rotates all geometry by R into camera space, then renders with a hardcoded `ax.view_init(elev=90, azim=-90)`.

Two bugs in this hardcoded view cause the mismatch:

1. **Wrong azimuth (`azim=-90`):** At `elev=90` (top-down view), `azim=-90` maps the camera frame's X axis to screen-up and -Y to screen-right — a 90° clockwise rotation of the correct view. The correct value is `azim=0`, which maps X→screen-right and Y→screen-up.

2. **Inverted depth ordering:** `elev=90` places matplotlib's virtual camera at +Z looking along -Z. But R defines +Z as the view direction (camera → focal point), so the forearm's front surface (nearest to the real camera) has smaller Z values and its back surface has larger Z. matplotlib at `elev=90` treats larger Z as closer, so the **back of the forearm occludes the front**, producing garbled or inside-out rendering.

The 2D projection path (Path B) is unaffected — `project_tangent_plane` drops Z entirely, and `project_cylindrical_unwrap` uses `rotation_matrix[2]` as an angular reference, not for spatial rendering.

## Goals

### In Scope
1. Fix the on-screen axis mapping so X=right and Y=up after rotation
2. Fix depth ordering so the camera-facing surface renders in front
3. Preserve Path B (2D projections) unchanged

### Out of Scope
- Changing `camera_settings_to_rotation()` — shared by all consumers, safe as-is
- Matching PyVista's perspective projection in matplotlib — inherent engine difference
- Changing the colormap or rendering style — those are intentional design choices
- Fixing the same issue in other viewers — covered by the separate `enforce-rf-camera-settings-all-viewers` plan

## Success Criteria

- [ ] 3D heatmap PNG orientation matches the RF Camera Settings Viewer for at least 2 sessions with non-trivial camera angles
- [ ] Depth ordering is correct: front surface visible, back surface hidden
- [ ] 2D projection PNGs (`_cylindrical_unwrap`, `_tangent_plane`) are byte-identical before and after the fix
- [ ] Cluster pipeline heatmaps (`visualize_receptive_fields_clustered`) also render correctly (same code path)

---

## Technical Design

### Approach

Negate the third row of R (view direction) **locally** inside the 3D rendering path of `render_forearm_heatmap()`. This creates a display-only rotation R' where Z points from the scene toward the camera (instead of away from it), so:
- Depth sorts correctly at `elev=90` (camera-facing surface has larger Z = closer)
- XY positions are unchanged (right and up are preserved)

Combined with changing `azim=-90` to `azim=0`, this produces the correct on-screen orientation.

**Why this works mathematically:**

```
R = [x_right; y_up; z_view_dir]          — from camera_settings_to_rotation()
R' = [x_right; y_up; -z_view_dir]        — R.copy(); R[2] *= -1

pts @ R'.T gives:
  new_x = pts · x_right       (unchanged — screen right)
  new_y = pts · y_up           (unchanged — screen up)
  new_z = -(pts · z_view_dir)  (negated — closer to camera = larger Z)

view_init(elev=90, azim=0):
  screen right = +X = camera right  ✓
  screen up    = +Y = camera up     ✓
  depth        = +Z = toward camera ✓
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Negate R[2] locally + fix azim | Minimal change (2 lines); handles all 3 camera DOF (including roll); doesn't affect Path B | Requires understanding why the negation is needed | **Chosen** |
| Compute elev/azim from camera params, skip rotation | More intuitive; no rotation needed | Only 2 DOF (no roll support); different projection model; would require removing all rotation code in the 3D path | Rejected |
| Change `camera_settings_to_rotation()` to negate z_new | Fixes depth globally | Breaks `cylindrical_unwrap` which uses `R[2]` as angular reference (`camera_facing = -R[2]`) | Rejected |
| Use `elev=-90` instead of negating R[2] | No matrix modification | Screen axis mapping at negative pole is unintuitive and may flip Y; matplotlib behavior at poles is implementation-dependent | Rejected |

### Architecture Changes

No new modules, classes, or interfaces. Single file modified with a 2-line change.

---

## Implementation Plan

### Phase 1: Fix View Parameters
**Goal:** Correct the matplotlib camera orientation in the 3D rendering path
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 1.1 — At line 324 of `rf_cluster_visualizer.py`, after `R = rotation_matrix`, add Z-negation: `if R is not None: R = R.copy(); R[2] *= -1`
- [x] Task 1.2 — At line 504, change `ax.view_init(elev=90, azim=-90)` to `ax.view_init(elev=90, azim=0)`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — negate R[2] at line 324, fix azim at line 504

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Open the Camera Settings Viewer (`set_rf_camera_settings`), note the forearm orientation for a test session (e.g. the forearm's long axis direction and which surface faces the camera)
- [ ] Run `map_receptive_fields_simple` with `force_processing: true` to regenerate the PNG
- [ ] Compare the PNG orientation against the viewer — forearm should face the same direction with the same up/right alignment
- [ ] Repeat with a second session where the camera was rotated to a different (non-axis-aligned) angle
- [ ] Run `visualize_receptive_fields_clustered` with `force_processing: true` and verify the cluster heatmap PNGs also match the viewer orientation
- [ ] Verify 2D projection outputs (`_cylindrical_unwrap` PNGs) are unchanged

### Edge Cases
- [ ] Session with near-axis-aligned camera (e.g. looking almost along the forearm long axis) — depth ordering should still be correct
- [ ] Session with `rotation_matrix=None` (if camera settings are missing for a session in the simple pipeline, `load_rf_camera_rotation` raises `ValueError` — verify this still happens cleanly)

---

## Documentation Plan

- [ ] No documentation changes needed — this is a bugfix, not a feature change
- [ ] The existing docstring for `render_forearm_heatmap` already describes the camera orientation intent correctly; the code now matches it

---

## Rollback Plan

1. Revert the two changed lines in `rf_cluster_visualizer.py`
2. Re-run `map_receptive_fields_simple` and `visualize_receptive_fields_clustered` with `force_processing: true` to regenerate PNGs
3. No data, config, or external state changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `azim=0` axis mapping doesn't match the derivation (matplotlib convention differs by version) | Low | Med | Verify visually with a known camera orientation before merging; adjust azim if needed |
| Negating R[2] affects some rendering sub-step not identified in analysis | Very low | Low | The 3D path is self-contained (lines 288-552); all rotation uses flow through the local `R` variable |
| Existing PNGs become inconsistent with new outputs | Certain | None | Intended — old PNGs had wrong orientation; re-generation with `force_processing: true` updates them |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~10 minutes | None |

---

## References

- Camera rotation builder: `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` — `camera_settings_to_rotation()` (lines 8-39)
- Rotation consumer (3D path): `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — `render_forearm_heatmap()` (lines 288-552)
- Rotation consumer (2D path): `code/src/analysis/receptive_field_mapping/rf_projection.py` — `project_cylindrical_unwrap()` uses `rotation_matrix[2]` (line 117)
- Camera settings viewer: `code/src/analysis/receptive_field_mapping/gui/rf_camera_settings_viewer.py`
- Related plan: `docs/development/plans/pending/enforce-rf-camera-settings-all-viewers.md` (separate concern — threading rotation into viewers that currently lack it)

---
