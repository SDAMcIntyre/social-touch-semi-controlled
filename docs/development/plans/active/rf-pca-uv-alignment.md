# Plan: PCA-aligned UV coordinates for population response field

**Date:** 2026-05-26
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-hotspot-center`
**Branch:** `feature/rf-pca-uv-alignment`

---

## Overview

Add a rotation+shift step to the population response field pipeline so that the
SLIM UV coordinate system is aligned to the RF data: U axis along the major
principal axis of the response field (most spatial spread), V along the minor
axis, centered on the geometric centroid of the response field. The alignment is
computed from the "all" gesture type and applied uniformly to all gesture types.

## Problem Statement

The SLIM UV parameterization produces an arbitrary orientation that varies across
sessions. This makes it hard to compare response field shapes and
proximal-distal center offsets across sessions because the U and V axes have no
consistent anatomical or functional meaning. Aligning the UV axes to the
response field's principal axes makes U/V offsets interpretable (e.g. "U = along
the RF's longest dimension") and centers the data for cross-session comparison.

## Goals

### In Scope
1. Compute unweighted PCA of above-threshold vertices from the "all" gesture heatmap
2. Shift all UV coordinates so the geometric centroid is at origin
3. Rotate all UV coordinates so the major PCA axis aligns with U
4. Store alignment metadata (center, rotation angle, rotation matrix) in NPZ
5. All downstream rendering and boundary detection uses the aligned UV

### Out of Scope
- Re-solving the SLIM parameterization (we post-process with a rigid transform)
- Modifying downstream pipelines (they consume the NPZ as-is)
- IFF-weighted PCA (decision: unweighted, all above-threshold vertices equal)
- Aligning individual gesture types separately

## Success Criteria

- [ ] Output PNGs show the "all" response field centered at origin with major axis horizontal
- [ ] NPZ contains `alignment_center_uv`, `alignment_rotation_deg`, `alignment_rotation_matrix`
- [ ] `forearm_uv` in NPZ is centered near (0, 0)
- [ ] Downstream tasks (`compare_rf_center_proximal_distal`, `compare_session_rf_boundaries`) run without errors
- [ ] 3D XYZ back-projected metrics (boundary perimeter, area) are unchanged

---

## Technical Design

### Approach

After computing heatmaps for all gesture types but before grid interpolation,
extract the "all" heatmap, select above-threshold vertices (heatmap > 0),
compute their geometric centroid and unweighted covariance in UV space,
eigendecompose to find the major axis, then apply a rigid shift+rotation to all
`forearm_uv` coordinates. All subsequent steps (interpolation, boundary
detection, rendering, NPZ save) use the transformed UV.

This is a rigid 2D transformation that preserves triangle topology and
barycentric coordinates, so `uv_points_to_xyz()` produces identical 3D results.
The neuron-centroid containment invariant (knowledge base:
`note-analysis-pipeline-coordinate-spaces.md`) is preserved because all vertices
are transformed identically.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Post-process rigid transform on SLIM UV | Simple, preserves mesh topology, no re-solve | Orientation only meaningful per-session | **Chosen** |
| Re-solve SLIM with alignment constraints | Could enforce cross-session consistency | Risks foldovers (knowledge base: `note-mesh-parameterization-interior-pin-foldovers.md`); complex | Rejected |
| IFF-weighted PCA | Captures intensity distribution | User preference: unweighted (spatial footprint) | Rejected |
| Align per-gesture-type separately | Each gesture in its own frame | Breaks cross-gesture comparison within a session | Rejected |

### Architecture Changes

New module:
- `code/src/analysis/receptive_field_mapping/metrics/rf_pca_alignment.py`

Modified module:
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`

No changes to downstream consumers, renderers, or boundary detection.

---

## Implementation Plan

### Phase 1: PCA alignment module
**Goal:** Provide reusable functions for computing and applying UV alignment.
**Started:** 2026-05-26
**Completed:** 2026-05-26

**Tasks:**
- [x] Task 1.1 — Create `compute_rf_pca_alignment(forearm_uv, heatmap)` that selects vertices with `heatmap > 0`, computes geometric centroid and unweighted covariance, eigendecomposes, returns `(center, rotation_matrix, angle_deg)`. Raise `ValueError` if < 3 valid vertices.
- [x] Task 1.2 — Create `apply_uv_alignment(uv, center, rotation_matrix)` returning `(uv - center) @ R.T`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/metrics/rf_pca_alignment.py` — New file with both functions

**Dependencies:** None

### Phase 2: Pipeline integration
**Goal:** Insert alignment step into the population response field pipeline.
**Started:** 2026-05-26
**Completed:** 2026-05-26

**Tasks:**
- [x] Task 2.1 — Import the new functions in `rf_population_response_field_pipeline.py`
- [x] Task 2.2 — After canonical ordering of `results` dict (line ~267) and the empty-results guard (line ~276), extract the "all" heatmap and call `compute_rf_pca_alignment`. Log center and angle.
- [x] Task 2.3 — Call `apply_uv_alignment` to transform `forearm_uv`. All subsequent code uses the transformed array.
- [x] Task 2.4 — Add alignment params to `_SessionCompositeData` dataclass (3 new fields: `alignment_center`, `alignment_rotation_matrix`, `alignment_angle_deg`)
- [x] Task 2.5 — Pass alignment params to `_save_response_fields_npz` and store as `alignment_center_uv`, `alignment_rotation_deg`, `alignment_rotation_matrix` in the NPZ

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — Insert alignment block, extend dataclass, extend NPZ save

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `compute_rf_pca_alignment` with a known point cloud (e.g. ellipse-shaped) returns expected center, angle, and rotation matrix
- [ ] `compute_rf_pca_alignment` raises `ValueError` when < 3 valid vertices
- [ ] `apply_uv_alignment` correctly shifts and rotates a simple test array
- [ ] Round-trip: `apply_uv_alignment` then inverse transform recovers original coordinates

### Integration Tests
- [ ] Run `extract_population_rf_response_field_boundaries` end-to-end with `force_processing: true`
- [ ] Verify NPZ contains alignment metadata keys
- [ ] Verify `forearm_uv` in NPZ is centered near (0, 0)

### Manual Verification
- [ ] Inspect "all" gesture PNG: response field centered, major axis horizontal
- [ ] Run `compare_rf_center_proximal_distal` and verify it completes without errors
- [ ] Run `compare_session_rf_boundaries` and verify it completes without errors
- [ ] Compare 3D boundary metrics (perimeter_xyz_mm, area_xyz_mm2) before and after — should be identical

### Edge Cases
- [ ] Session with very few above-threshold vertices (near the 3-vertex minimum)
- [ ] Session where response field is nearly circular (degenerate PCA — small eigenvalue ratio)
- [ ] Session missing the "all" gesture type in results (should not happen but guard against it)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention alignment step in population response fields section
- [ ] Add inline docstring to `compute_rf_pca_alignment` explaining the unweighted PCA choice

---

## Rollback Plan

1. Remove the alignment block from the pipeline (revert to passing raw SLIM UV)
2. Re-run with `force_processing: true` to regenerate NPZs and PNGs
3. No data migration needed — NPZ files are regenerated from source data

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PCA axis flip (sign ambiguity) | Medium | Low | Consistent convention: always pick the eigenvector direction that makes the first coordinate positive, or accept that 180-degree flips are equivalent for axis alignment |
| Near-circular RF (degenerate PCA) | Low | Low | Still produces a valid rotation; the axis choice is arbitrary but harmless when the RF has no clear elongation |
| Downstream pipeline expects original UV scale | Low | High | Verify by running all downstream tasks; the NPZ stores alignment metadata for inversion if needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~30 min | None |
| Phase 2 | ~45 min | Phase 1 |
| Testing | ~30 min | Phase 2 |

---

## References

- Knowledge base: `note-mesh-parameterization-interior-pin-foldovers.md` — post-process, don't constrain SLIM
- Knowledge base: `note-analysis-pipeline-coordinate-spaces.md` — neuron-centroid invariant
- Existing PCA pattern: `rf_inflection_boundary.py::_compute_contour_pca()` (lines 255-276)
