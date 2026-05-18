# Plan: Mesh-Constrained RF Population Heatmap Interpolation

**Date:** 2026-05-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/visualize-population-rf-maps`
**Branch:** `feature/visualize-population-rf-maps`
**Started:** 2026-05-18
**Phase 1 Completed:** 2026-05-18
**Phase 2 Completed:** 2026-05-18
**Phase 3 Completed:** 2026-05-18

---

## Overview

Replace the population RF heatmap interpolation engine (`griddata` + cKDTree
distance mask) with a mesh-constrained two-step method: `igl.harmonic` to
diffuse sparse RF values across the SLIM mesh, then
`matplotlib.tri.CubicTriInterpolator` to rasterize onto a pixel grid using the
mesh faces. This eliminates boundary bleed-through and replaces the arbitrary
circular distance mask with the actual RF data region boundary.

## Problem Statement

The current population RF heatmap renderer (`rf_population_map_renderer.py`)
uses `scipy.interpolate.griddata(method='cubic')` to interpolate scattered 2D
sample points onto a regular 150x150 grid. After interpolation, grid cells
farther than 10 mm from the nearest sample are masked to NaN via cKDTree.

Two problems:

1. **Interpolation bleed-through:** `griddata` builds its own Delaunay
   triangulation from the valid sample points, ignoring the forearm mesh
   topology. Delaunay triangles can span across concave gaps in the contact
   region, causing the interpolant to bleed into uncontacted areas.

2. **Circular distance mask:** The cKDTree mask applies a fixed isotropic radius
   around each sample point. This is a poor approximation of the actual RF data
   boundary — it clips concave edges too aggressively in some places and extends
   too far in others.

The SLIM UV cache already stores the mesh triangulation (`V`, `F`, `uv`) that
could constrain interpolation to the correct domain, but this information is
discarded before rendering.

## Goals

### In Scope

1. Replace `griddata` + cKDTree with `igl.harmonic` + `CubicTriInterpolator` in
   both renderer functions (`render_population_rf_map`,
   `render_population_rf_composite`)
2. Thread SLIM mesh data (V, F, uv) from the cache through the pipeline to the
   renderer
3. Remove the `disjoint_mask_distance_mm` parameter from the population pipeline
   (no longer needed)

### Out of Scope

- `rf_2d_renderer.py` (cluster pipeline renderer) — same griddata problem but
  different pipeline and data shapes; address as a follow-up
- `rf_cluster_visualizer.py` / `rf_cluster_pipeline.py` — their
  `disjoint_mask_distance_mm` is untouched
- DAG config YAML changes — no `disjoint_mask_distance_mm` key exists under
  `visualize_population_rf_maps` in the current configs
- `rf_extraction_io.py` — the cluster pipeline's visualization summary uses
  `disjoint_mask_distance_mm` for idempotency; untouched

## Success Criteria

- [ ] Population RF heatmap follows the contacted region boundary (no bleed
      beyond RF data)
- [ ] Interpolation doesn't cross concave gaps in the contact region
- [ ] NaN (transparent) outside the RF data region
- [ ] Smooth (C1) interpolation within the RF region
- [ ] Below-threshold vertices form a smooth transition to zero at the RF edge
- [ ] `disjoint_mask_distance_mm` parameter removed from population pipeline
- [ ] Existing cluster pipeline `disjoint_mask_distance_mm` unaffected

---

## Technical Design

### Approach

Two-step mesh-constrained interpolation:

**Step 1 — Harmonic fill (`igl.harmonic`):** Only ~1-5% of mesh vertices have
valid RF data. `CubicTriInterpolator` requires finite values at all vertices of
unmasked faces. Rather than setting unknown vertices to 0.0 (which creates
artificial dips at interior gaps), we solve the discrete Laplace equation on the
mesh with Dirichlet constraints at contacted vertices. This smoothly propagates
known RF values to all mesh vertices, filling gaps and providing smooth boundary
transitions.

- Contacted above-threshold vertices: constrained to their RF value
- Contacted below-threshold vertices: constrained to 0.0 (smooth boundary)
- Uncontacted vertices: solved by the harmonic equation (smooth fill)

**Step 2 — Mesh-constrained rasterization (`CubicTriInterpolator`):** Build a
`matplotlib.tri.Triangulation` from the SLIM UV coordinates and mesh faces, with
a face mask that excludes faces where no vertex was contacted. The
CubicTriInterpolator (Clough-Tocher C1) evaluates on the regular pixel grid.
Points outside unmasked triangles automatically return masked (NaN) — no
distance mask needed.

**Why the mesh faces matter:** The SLIM mesh faces encode the surface
connectivity. Unlike a Delaunay triangulation of scattered points, mesh
triangles cannot span across concave gaps because the mesh topology only
connects adjacent surface regions. This is the core fix for the bleed-through
problem.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `igl.harmonic` + `CubicTriInterpolator` + face mask | Uses existing mesh and dependencies; respects RF boundary; C1 smooth | Requires vertex mapping from original to SLIM mesh | **Chosen** |
| Face-mask only (no harmonic fill) | Simpler; no diffusion step | Sparse valid faces; interior gaps produce artifacts; CubicTriInterpolator needs finite values at all unmasked vertices | Rejected — only ~1-5% of vertices are valid, so most faces would be masked |
| `griddata` + polygon mask (mesh boundary) | Minimal code change | Delaunay still crosses concavities — fixes masking but not interpolation bleed-through | Rejected — doesn't solve the root cause |
| RBF + polygon mask | Smoothest result (C-infinity) | Ignores mesh geometry; bleeds through concavities before masking; O(N^3) | Rejected — same bleed-through problem |

### Architecture Changes

No new modules. Changes are contained within the existing renderer and pipeline
files. The `_interpolate_on_mesh` helper is a private function in the renderer
module.

**Data flow change:** The pipeline currently projects all original forearm
vertices to SLIM UV via `project_to_2d(method='slim')`, discarding the mesh
faces. The new flow loads the SLIM cache directly and passes `V`, `F`, `uv` to
the renderer alongside the heatmap values mapped to SLIM vertex indices.

```
Before:
  forearm_vertices → project_to_2d(slim) → forearm_uv → griddata → distance mask

After:
  SLIM cache → (V, F, uv) ─┬→ igl.harmonic(V, F, constraints) → filled values
                            └→ Triangulation(uv, F, mask) → CubicTriInterpolator → grid
```

**Vertex mapping:** The SLIM mesh has fewer vertices than the original forearm
PLY (cleaning removes non-manifold edges, orphans, degenerate faces). Heatmap
values are transferred from original to SLIM vertices via KDTree
nearest-neighbor lookup. The cleaned mesh vertices are geometrically close
(sub-mm) to their original counterparts.

---

## Implementation Plan

### Phase 1: Renderer — replace interpolation engine

**Goal:** Replace griddata + cKDTree with harmonic fill + CubicTriInterpolator
in both renderer functions.

- [x] 1.1 — Add `_interpolate_on_mesh(forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v)` helper that performs harmonic fill, face masking, CubicTriInterpolator evaluation, and NaN conversion
- [x] 1.2 — Update `render_population_rf_map()` to accept `forearm_faces` and `forearm_V` parameters, compute grid bounds from full mesh UV, and call `_interpolate_on_mesh`
- [x] 1.3 — Update `render_population_rf_composite()` similarly, using its existing `uv_xlim`/`uv_ylim` for grid bounds
- [x] 1.4 — Remove `disjoint_mask_distance_mm` parameter from both functions
- [x] 1.5 — Replace `scipy.interpolate.griddata` and `scipy.spatial.cKDTree` imports with `igl` and `matplotlib.tri as mtri`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_map_renderer.py` — replace interpolation engine, add helper (~40 lines replaced, ~35 lines added)

**`_interpolate_on_mesh` algorithm:**

1. `contacted_mask = np.isfinite(heatmap_val)` — all contacted vertices (above + below threshold)
2. `z = np.where(heatmap_val >= 0.0, heatmap_val, 0.0)` — above-threshold keep value; below-threshold → 0.0; NaN → 0.0
3. `b = np.where(contacted_mask)[0]` — constraint vertex indices
4. `bc = z[b].reshape(-1, 1)` — constraint values
5. `filled = igl.harmonic(forearm_V, forearm_faces, b, bc, 1).ravel()` — diffuse to all vertices
6. `face_mask = ~np.any(contacted_mask[forearm_faces], axis=1)` — mask faces where NO vertex was contacted
7. Build `Triangulation(uv[:, 0], uv[:, 1], triangles=forearm_faces, mask=face_mask)`
8. `CubicTriInterpolator(tri, filled, kind='min_E')` — Clough-Tocher C1
9. Evaluate on grid → masked array → convert mask to NaN → clip negatives to 0.0
10. Raise `ValueError` if fewer than 4 contacted vertices or all-NaN result

**Dependencies:** None

### Phase 2: Pipeline — load SLIM cache and map heatmap to SLIM vertices

**Goal:** Thread SLIM mesh data from the cache to the renderer, with heatmap
values mapped from original forearm vertices to SLIM mesh vertices.

- [x] 2.1 — Replace `project_to_2d(method='slim')` call with direct `load_slim_uv_cache()` to get `cache.V`, `cache.F`, `cache.uv`
- [x] 2.2 — Build KDTree mapping from original forearm vertices to SLIM vertices: `orig_tree = KDTree(pop_data.forearm_vertices)` → `nearest_orig_for_slim = orig_tree.query(cache.V)`
- [x] 2.3 — In the render loop, map each gesture subset's `thresholded` array to SLIM vertices: `slim_heatmap = thresholded[nearest_orig_for_slim]`
- [x] 2.4 — Update `render_population_rf_map` call to pass `forearm_faces=slim_faces, forearm_V=slim_V`
- [x] 2.5 — Add `forearm_faces` and `forearm_V` fields to `_SessionCompositeData` dataclass
- [x] 2.6 — Update `render_population_rf_composite` call in composite loop to pass faces and V
- [x] 2.7 — Remove `disjoint_mask_distance_mm` from `run_population_rf_maps` function signature and docstring
- [x] 2.8 — Update imports: remove `rf_projection.project_to_2d`, add `forearm_slim_uv.load_slim_uv_cache` and `scipy.spatial.KDTree`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py` — data flow, SLIM cache loading, vertex mapping (~15 lines changed)

**Dependencies:** Phase 1

### Phase 3: Workflow — remove `disjoint_mask_distance_mm` parameter

**Goal:** Remove the parameter from the flow function and DAG config dispatcher
for the population pipeline only. The cluster pipeline keeps its own parameter.

- [x] 3.1 — Remove `disjoint_mask_distance_mm` from `visualize_population_rf_maps_flow` signature (line 233) and from the `run_population_rf_maps(...)` call (line 250)
- [x] 3.2 — Remove the population-specific dispatcher at lines 1362-1363: the `if task_name == "visualize_population_rf_maps"` block's `disjoint_mask_distance_mm` handler. The generic handler at lines 1366-1367 (used by cluster pipeline tasks) remains.

**Files Modified:**
- `code/scripts/analysis_workflow.py` — 4 lines removed

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

No new unit tests — the interpolation change is a rendering implementation
detail. The existing test suite (`test_dag_config_model.py`, etc.) does not test
rendering output.

### Manual Verification

- [ ] Delete sentinel files for 2+ test sessions to force reprocessing
- [ ] Run `visualize_population_rf_maps` task via the pipeline GUI
- [ ] Per-gesture PNGs: heatmap follows contacted region outline, not circles
- [ ] Composite PNGs: same check across multi-panel view
- [ ] Scatter panel: forearm shape in UV space looks correct (SLIM vertices vs original)
- [ ] Color scale: consistent session-wide vmax (per-gesture) and global vmax (composites)
- [ ] Below-threshold boundary: smooth transition to zero at RF edge, not a hard cutoff

### Edge Cases

- [ ] Session with very few contacts (<10 touched vertices per gesture) — should raise `ValueError` with informative message
- [ ] Session with narrow strip of contacts — should produce narrow heatmap, not a blob
- [ ] Gesture subset with zero contacts — should skip gracefully (existing logic unchanged)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — update population RF maps bullet to mention mesh-constrained interpolation instead of griddata
- [ ] No CLAUDE.md root changes needed (no new conventions or constraints)

---

## Rollback Plan

1. `git revert` the implementation commit(s)
2. No data migrations — output PNGs are regenerated from upstream NPZ data
3. Sentinel files can be deleted to force re-rendering with the old method

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SLIM mesh vertex mapping inaccurate (cleaned mesh differs significantly from original) | Low | Med | KDTree nearest-neighbor should be sub-mm; log max mapping distance and raise if > 1 mm |
| `igl.harmonic` fails on degenerate mesh | Low | High | SLIM cache already ensures clean manifold mesh via `clean_mesh()` + `flatten_slim()`; raise `ValueError` with diagnostic info |
| `CubicTriInterpolator` poor gradient estimation at face-mask boundaries | Med | Low | `kind='min_E'` minimizes bending energy; visual inspection on test sessions |
| Composite global UV limits mismatch with SLIM UV extent | Low | Med | Both scatter and interpolation now use `cache.uv`; limits computed from same array |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer | ~30 min | None |
| Phase 2: Pipeline | ~20 min | Phase 1 |
| Phase 3: Workflow | ~5 min | Phase 2 |
| Manual verification | ~15 min | Phase 3 |

---

## References

- Approved implementation plan: `.claude/plans/analyse-visualize-population-rf-ancient-sun.md`
- Related knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Parent plan: `docs/development/plans/active/visualize-population-rf-maps.md`
- matplotlib.tri API: https://matplotlib.org/stable/api/tri_api.html
- libigl harmonic: https://libigl.github.io/libigl-python-bindings/tut-chapter3/
