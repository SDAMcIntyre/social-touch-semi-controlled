# Plan: Delaunay Hole Filling and Step 2 Triangle-Quality Diagnostic

**Date:** 2026-05-21
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/slim-uv-colormapped-diagnostic-panels`
**Branch:** `feature/slim-uv-delaunay-hole-filling`

---

## Overview

**What:** Replace centroid-fan hole filling with best-fit-plane Delaunay triangulation for large
holes in the SLIM UV mesh cleaning pipeline, and add triangle-quality visualization to the
step 2 diagnostic figure.

**Why:** Session ST14-02 (10 holes, 178 fan triangles) shows a visible "starburst" artifact in the
thumb region — long elongated triangles radiating from single centroid vertices. These degrade
SLIM convergence and create distortion hotspots. The step 2 diagnostic renders a solid surface
with no wireframe, making the issue invisible without external inspection.

**How:** Use `scipy.spatial.Delaunay` on a best-fit-plane projection for holes with >4 boundary
vertices. Expand the step 2 diagnostic to a 2x2 layout with wireframe edges and face
aspect-ratio coloring.

## Problem Statement

`_fill_interior_holes()` in `slim_helpers.py` uses centroid-fan triangulation: one centroid vertex
per hole with fan triangles to every boundary vertex. For small holes (3-4 vertices) this is fine,
but for large holes (13+ boundary vertices as seen in ST14-02's thumb region) it creates triangles
with extreme aspect ratios (longest/shortest edge >> 10). These poorly-conditioned triangles:

- Concentrate SLIM distortion energy in the filled region
- Create numerically ill-conditioned Jacobians in harmonic/Tutte initialization
- Are a known cause of Tutte flip failures that trigger the expensive trim-retry loop
  (see `bug-slim-uv-non-manifold-flip.md`: ST14-02 required Tutte fallback + SLIM)

The step 2 diagnostic (`_plot_step2_cleaned_mesh`) renders the cleaned mesh as a solid
`lightsteelblue` surface with `edgecolor="none"`, making it impossible to see individual triangle
shapes or identify quality issues without opening the mesh in an external tool.

## Goals

### In Scope
1. Replace centroid-fan with Delaunay triangulation for holes with >4 boundary vertices
2. Handle non-convex hole boundaries (filter Delaunay simplices outside the polygon)
3. Insert centroid for very large holes (>20 boundary vertices) to limit triangle span
4. Add wireframe edges and face aspect-ratio coloring to the step 2 diagnostic
5. Add triangle quality statistics annotation to step 2
6. Add unit tests for the new hole-filling logic

### Out of Scope
- Changing the BPA mesh construction parameters (the upstream point cloud quality)
- Modifying steps 1, 3-6 diagnostic figures
- Changing the existing QC figures (`save_slim_qc_figures`)
- Re-meshing or subdivision of the filled region after Delaunay
- Adjusting `_TRIM_FLIP_MAX` or `_TRIM_MAX_ROUNDS` thresholds

## Success Criteria

- [ ] ST14-02 step 2 diagnostic shows no starburst artifact in the thumb region
- [ ] All 12 sessions run `precompute_forearm_slim_uv` to completion without error
- [ ] Step 2 diagnostic shows wireframe edges and aspect-ratio heatmap in 2x2 layout
- [ ] Maximum face aspect ratio drops significantly for sessions with large filled holes
- [ ] All existing tests in `test_forearm_slim_uv.py` pass without modification
- [ ] New `TestFillInteriorHoles` tests pass

---

## Technical Design

### Approach

Use `scipy.spatial.Delaunay` on a best-fit-plane 2D projection of the hole boundary vertices.
The best-fit plane is computed via SVD of the centered boundary positions: the two largest
singular vectors form the 2D basis, the smallest is the plane normal. For non-convex holes,
filter out Delaunay simplices whose centroids fall outside the boundary polygon using
`matplotlib.path.Path.contains_point`. For very large holes (N > 20), insert the boundary
centroid as an additional interior point before Delaunay to prevent triangles spanning the full
hole diameter.

Small holes (N <= 4) keep centroid-fan — Delaunay of 3-4 points is equivalent or worse.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Best-fit-plane Delaunay | Maximizes minimum angle (Delaunay criterion); scipy already available; handles any hole size | Requires SVD projection + polygon filter (~50 lines) | **Chosen** |
| Recursive centroid subdivision | Simple to implement | Still produces elongated triangles; more new vertices | Rejected |
| `trimesh.repair.fill_holes()` | Zero code to write | Black-box; no manifold guarantee; violates fail-fast philosophy | Rejected |
| Ear-clipping | Good for simple polygons | Concavity handling complex; Delaunay subsumes it | Rejected |

### Architecture Changes

No new modules or classes. Changes are localized to two existing files plus one test file.

```
surface/
  slim_helpers.py       -- new _fill_hole_delaunay(); modify _fill_interior_holes()
  slim_qc_figures.py    -- new _compute_face_aspect_ratios(); rewrite _plot_step2_cleaned_mesh()
tests/
  test_forearm_slim_uv.py -- new TestFillInteriorHoles class
```

### Knowledge Base Constraints

From `bug-slim-uv-non-manifold-flip.md`:
- The centroid-fan approach was added to fix disk topology; the replacement must maintain the
  same topological guarantee (each hole fully closed, single boundary loop preserved)
- ST14-02 had 10 holes (178 fan triangles); ST14-04 had 13 holes — both are validation targets
- Fan triangles can cause Tutte flip failures (ST18-04: 17 flips in 2 trim rounds)

From `investigation-mesh-flattening-algorithms-survey.md`:
- SLIM was chosen as the production flattener; better input triangle quality improves convergence

From `note-mesh-parameterization-interior-pin-foldovers.md`:
- Boundary-only constraints are used (no interior pins) — better-conditioned input mesh
  reduces the need for Tutte fallback

---

## Implementation Plan

### Phase 1: Delaunay hole filling
**Goal:** Replace centroid-fan with Delaunay for large holes while preserving topology guarantees.
**Started:** 2026-05-21
**Completed:** 2026-05-21

- [x] Add `from scipy.spatial import Delaunay` import to `slim_helpers.py`
- [x] Implement `_fill_hole_delaunay(V, loop, n_existing_new_verts)` (~50 lines):
  SVD best-fit-plane projection, Delaunay triangulation, non-convex polygon filter,
  optional centroid insertion for N > 20, degenerate-plane fallback
- [x] Modify `_fill_interior_holes()` loop body: dispatch to `_fill_hole_delaunay` for
  loops with > 4 boundary vertices, keep centroid-fan for smaller loops
- [x] Update logging to report method (fan/Delaunay) and triangle count per hole

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_helpers.py` — new function + modified loop

**Dependencies:** None

### Phase 2: Step 2 diagnostic upgrade
**Goal:** Make triangle quality visible in the step 2 diagnostic figure.
**Started:** 2026-05-21
**Completed:** 2026-05-21

- [x] Add `_compute_face_aspect_ratios(V, F) -> np.ndarray` (~15 lines): per-face
  longest_edge / shortest_edge
- [x] Add optional `edgecolor` and `linewidth` parameters to `_plot_mesh_3d()`
  (defaults preserve backward compatibility)
- [x] Rewrite `_plot_step2_cleaned_mesh()` from 1x2 to 2x2 layout:
  top row = wireframed mesh views; bottom-left = aspect-ratio heatmap (YlOrRd);
  bottom-right = quality statistics text
- [x] Figure size from `(10, 4.5)` to `(10, 9)`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_qc_figures.py` — new function + rewritten step 2

**Dependencies:** None (can be implemented in parallel with Phase 1)

### Phase 3: Testing
**Goal:** Verify correctness and no regressions.
**Started:** 2026-05-21
**Completed:** 2026-05-21

- [x] Add `TestFillInteriorHoles` class to `test_forearm_slim_uv.py`
- [x] Test: small hole (3-4 verts) produces centroid-fan (N new faces = N boundary verts, 1 new vertex)
- [x] Test: large hole (20+ verts) produces Delaunay triangles with bounded aspect ratio (AR < 10)
  and exactly 1 boundary loop
- [x] Test: non-convex hole boundary — no triangles outside the polygon
- [x] Run existing test suite: `pytest code/tests/test_forearm_slim_uv.py -v`

**Files Modified:**
- `code/tests/test_forearm_slim_uv.py` — new test class

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [x] `test_small_hole_uses_fan` — 3-vertex hole produces 3 new faces and 1 new vertex (centroid)
- [x] `test_large_hole_uses_delaunay` — 20+ vertex hole produces well-shaped triangles (max AR < 10),
  result has exactly 1 boundary loop, mesh vertex count is reasonable
- [x] `test_nonconvex_hole` — concave hole boundary produces no triangles outside the polygon
- [x] `test_degenerate_plane_fallback` — nearly collinear boundary vertices fall back to centroid-fan

### Integration Tests
- [x] Existing `TestFlattenSlim` tests pass unchanged
- [x] Existing `TestPrecomputeForearmSlimUv` tests pass unchanged
- [x] Existing `TestBarycentricUvLookup` and `TestUvPointsToXyz` tests pass unchanged

### Manual Verification
- [ ] Run `precompute_forearm_slim_uv` with `save_diagnostics: true, force_processing: true`
  on ST14-02 — step 2 shows no starburst; aspect-ratio heatmap shows no hotspots in thumb
- [ ] Compare step 6 distortion: conformal distortion peaks in thumb region should decrease
- [ ] Run all 12 sessions end-to-end — all pass without error

### Edge Cases
- [ ] Hole with exactly 5 boundary vertices (smallest Delaunay case) — produces valid triangulation
- [ ] Hole with >20 boundary vertices — centroid inserted, all triangles bounded
- [ ] Session with no interior holes (clean BPA mesh) — `_fill_interior_holes` is a no-op

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md` — note that
  centroid-fan was replaced by Delaunay for large holes
- [ ] Docstring on `_fill_hole_delaunay()` describing the algorithm and fallback behavior

---

## Rollback Plan

All changes are additive or internal refactors of private functions:

1. Revert `slim_helpers.py` to restore centroid-fan in `_fill_interior_holes()`
2. Revert `slim_qc_figures.py` to restore 1x2 step 2 layout
3. Revert `test_forearm_slim_uv.py` to remove new test class
4. Re-run `precompute_forearm_slim_uv` with `force_processing: true` to regenerate caches

No data migrations. No public API changes. Cached `.npz` files will be regenerated on next run
(the cache staleness check uses PLY mtime/hash, not mesh topology).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Delaunay of non-convex boundary includes exterior triangles | High | Med | Polygon filter via `matplotlib.path.Path.contains_point` on each simplex centroid |
| Best-fit plane degeneracy (collinear boundary) | Low | Low | SVD singular-value check; fall back to centroid-fan with logged warning |
| Hole filling changes break SLIM convergence | Low | High | Run all 12 sessions; compare distortion statistics before/after |
| matplotlib 3D wireframe rendering slow for large meshes | Low | Low | Use thin `linewidth=0.15` and light gray color; meshes are <15k faces |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Delaunay hole filling | ~45 min | None |
| Phase 2: Step 2 diagnostic | ~30 min | None (parallel) |
| Phase 3: Testing | ~30 min | Phase 1 |

---

## References

- Knowledge base: `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md`
- Knowledge base: `docs/development/knowledge-base/investigation-mesh-flattening-algorithms-survey.md`
- Related plans: `docs/development/plans/active/slim-uv-diagnostic-figures.md`
- Related plans: `docs/development/plans/active/slim-uv-colormapped-diagnostic-panels.md`
