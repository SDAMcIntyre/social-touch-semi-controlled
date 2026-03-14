# Plan: Fix PCA Calibration Export Color Loss

**Date:** 2026-03-14
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `fix/pca-export-color-loss`

---

## Overview

The forearm pointcloud loses its RGB color data during the PCA calibration export step. Colors are preserved through the entire upstream pipeline but dropped when `export_forearm_pca_calibrated.py` creates a new pointcloud with only transformed coordinates. The fix is to copy the original colors onto the output pointcloud before saving.

## Problem Statement

The postprocessed scene viewer renders forearm pointclouds in uniform gray instead of their original Kinect RGB colors. The viewer correctly checks for colors (`has_colors()`) and falls back to gray when none are present. The root cause is upstream: the PCA export script discards color data.

Colors survive every prior stage (Kinect capture → arm segmentation → forearm cleaning → registration → unified cloud), but are lost at the final export because the script extracts only coordinates, transforms them, and writes a new pointcloud without transferring the color attribute.

## Goals

### In Scope
1. Preserve RGB colors through the PCA calibration export step
2. Handle the edge case where the input PLY has no colors (no-op)

### Out of Scope
- Modifying the PCA calibration engine itself (it correctly operates on coordinates only)
- Fixing color in any upstream pipeline stage (colors are already preserved upstream)
- Adding color to pointclouds that never had color data

## Success Criteria

- [ ] Output `*_forearm_pca_calibrated.ply` files contain RGB color data when the input PLY has colors
- [ ] The postprocessed scene viewer renders forearm pointclouds with original Kinect colors instead of gray
- [ ] Pointclouds without color data are handled gracefully (no crash, no change in behavior)

---

## Technical Design

### Approach

Copy the color attribute from the input pointcloud to the output pointcloud after applying the coordinate transform. Colors are per-vertex RGB values from the Kinect sensor — they are independent of spatial transformations and should be transferred as-is.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Copy `pcd.colors` to `out_pcd.colors` | Minimal change, correct semantics | None | **Chosen** |
| Transform in-place on `pcd` | One fewer object allocation | Mutates input; `apply_full_transform` returns a new array anyway | Rejected |

### Architecture Changes

None. This is a 2-line bug fix in a single file.

---

## Implementation Plan

### Phase 1: Fix color preservation
**Goal:** Ensure the exported PLY retains RGB colors from the input

- [ ] Add color transfer after creating `out_pcd` in `export_forearm_pca_calibrated.py`

**Files Modified:**
- `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py` (lines 109-112) — After setting `out_pcd.points`, add a conditional copy of `pcd.colors` to `out_pcd.colors`

**Change detail:**

After line 111 (`out_pcd.points = ...`), add:
```python
if pcd.has_colors():
    out_pcd.colors = pcd.colors
```

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run the postprocessing pipeline on a session with a colored forearm PLY
- [ ] Open the output `*_forearm_pca_calibrated.ply` in the postprocessed scene viewer
- [ ] Confirm the forearm renders with original Kinect RGB colors instead of uniform gray
- [ ] Verify the PLY file size increased (color data now present)

### Edge Cases
- [ ] Input PLY without colors (e.g., synthetically generated) — should produce output without colors, same as current behavior

---

## Documentation Plan

- [ ] No documentation changes required (bug fix, no new features or API changes)

---

## Rollback Plan

1. Revert the single commit on the fix branch
2. No data migration or breaking changes — output PLY files are regenerated on each run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Input PLY has no colors | Low | Low | Guarded by `has_colors()` check |

---

## References

- **Root cause file:** `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py` lines 109-112
- **PCA engine:** `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py` (no changes needed)
- **Downstream consumer:** `code/src/postprocessing/gui/postprocessed_scene_viewer.py` lines 216-224 (already handles colors correctly)
