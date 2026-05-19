# Plan: Population RF Inflection Boundary Extraction

**Date:** 2026-05-19
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/separate-rf-metrics-from-visualization`
**Branch:** `feature/population-rf-inflection-boundary`

---

## Overview

Add inflection-point boundary detection to the population RF heatmap pipeline.
The Laplacian zero-crossing on the interpolated 2D IFF heatmap marks where the
surface transitions from concave (near the peak) to convex (on the flanks) —
the ring of steepest descent. This gives a physiologically meaningful RF
boundary, unlike the current arbitrary top-20% threshold. The boundary is drawn
on the rendered heatmaps and its geometric metrics are persisted.

## Problem Statement

The existing "hotspot" boundary in `rf_metrics.py` uses a fixed top-20%
threshold plus a convex hull. This is arbitrary — it ignores the actual shape
of the IFF response surface. For smooth bell-shaped population RF maps, the
inflection point (where the second spatial derivative changes sign) is a
natural, data-driven boundary that corresponds to the steepest descent ring
around the peak.

There is currently no gradient-based or curvature-based boundary detection
anywhere in the codebase.

## Goals

### In Scope
1. Detect the inflection boundary on per-session, per-gesture population RF
   heatmaps using Laplacian zero-crossing
2. Compute geometric metrics on the boundary: area, perimeter, circularity,
   centroid, PCA orientation, mean IFF along contour
3. Overlay the boundary contour on rendered heatmap PNGs (per-gesture and
   composites)
4. Persist boundary metrics in the sentinel JSON
5. Expose a `inflection_sigma` DAG config option to control smoothing (or
   disable the feature via `null`)

### Out of Scope
- Inflection boundaries on grid-based RF maps (per-cell) — future work
- Inflection boundaries on the cluster RF pipeline — separate feature
- Converting UV-space metrics to mm-scale (requires UV-to-mm calibration)
- Multi-peak detection or secondary inflection rings
- 3D mesh-native Laplacian (operating on UV-projected 2D grid is sufficient)

## Success Criteria

- [ ] `compute_inflection_boundary()` returns correct boundary for synthetic
      Gaussian hills (area, centroid, circularity validated)
- [ ] Returns `None` gracefully for flat, all-NaN, or edge-peak cases
- [ ] Per-gesture PNGs show green inflection contour on interpolated heatmap
      panel
- [ ] Composite PNGs show contours on each gesture panel
- [ ] Sentinel JSON includes `inflection_boundaries` dict with per-gesture
      metrics
- [ ] Setting `inflection_sigma: null` disables the feature with no rendering
      change
- [ ] All existing tests pass (no regressions)

---

## Technical Design

### Approach

Operate on the 150x150 `grid_z` array already produced by
`_interpolate_on_mesh()` in the population RF renderer:

1. **NaN-aware Gaussian smooth** — replace NaN with 0, smooth both values and
   a binary weight mask, divide to normalize near NaN edges
2. **Compute Laplacian** — `scipy.ndimage.laplace()` on the smoothed grid
3. **Re-mask** — NaN where `grid_z` was NaN, plus a 2-pixel border around NaN
   regions to prevent false zero-crossings from the finite-difference stencil
4. **Extract contours** — `skimage.measure.find_contours(laplacian, 0.0)`
5. **Select enclosing contour** — smallest-area contour that contains the
   global IFF peak (via `matplotlib.path.Path.contains_point`)
6. **Compute metrics** — polygon geometry on the UV-space contour

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Laplacian zero-crossing | Mathematically precise inflection; handles arbitrary shapes; operates on existing grid | Sensitive to noise (mitigated by Gaussian smoothing) | **Chosen** |
| Gradient magnitude ridge | More robust to noise; equivalent for bell-shaped profiles | Not precisely the inflection for non-radially-symmetric shapes; harder to extract as a single contour | Rejected |
| Level-set at fixed % of peak | Simple to implement | Arbitrary threshold (same problem as current approach) | Rejected |
| Radial profile inflection | Intuitive; works well for unimodal peaks | Requires choosing radial sampling directions; fails for non-convex shapes | Rejected |

### Architecture Changes

**New module:** `rf_inflection_boundary.py` — pure computation, no matplotlib
rendering, no I/O. Contains `InflectionBoundary` dataclass and
`compute_inflection_boundary()` public function.

**Modified modules:**
- `rf_population_map_renderer.py` — extract grid construction into
  `compute_interpolated_grid()`; add `precomputed_grid` and
  `inflection_boundary` params to renderers; add `_draw_inflection_boundary()`
- `rf_population_map_pipeline.py` — call boundary computation in per-gesture
  and composite loops; extend sentinel JSON
- `analysis_workflow.py` — forward `inflection_sigma` option
- `analyse_workflow_processing_dag.yaml` — add `inflection_sigma` option

```
rf_population_map_pipeline.py
  |
  |-- compute_interpolated_grid()          [extracted from renderer]
  |-- compute_inflection_boundary()        [new module]
  |
  v
render_population_rf_map()                 [receives precomputed grid + boundary]
  |-- _draw_inflection_boundary()          [new helper]
```

---

## Implementation Plan

### Phase 1: Core Computation
**Goal:** Implement inflection boundary detection as a standalone pure module

**Started:** 2026-05-19
**Completed:** 2026-05-19

- [x] Create `rf_inflection_boundary.py` with `InflectionBoundary` dataclass
- [x] Implement `_compute_masked_laplacian()` with NaN-aware Gaussian smoothing
      and 2-pixel NaN border masking
- [x] Implement `_find_peak_location()` — global peak ignoring NaN
- [x] Implement `_select_enclosing_contour()` — point-in-polygon selection of
      smallest contour enclosing the peak
- [x] Implement `_contour_pixels_to_uv()` — row/col to UV coordinate conversion
- [x] Implement polygon metrics: `_compute_polygon_area()` (shoelace),
      `_compute_polygon_perimeter()`, `_compute_polygon_centroid()`,
      `_compute_contour_pca()`
- [x] Implement `_sample_grid_along_contour()` — bilinear interpolation of IFF
      values along the contour path
- [x] Implement `compute_inflection_boundary()` public orchestrator
- [x] Implement `inflection_boundary_to_dict()` serialization
- [x] Create `test_rf_inflection_boundary.py` with synthetic-data tests

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_inflection_boundary.py`
- `code/tests/test_rf_inflection_boundary.py`

**Dependencies:** None

### Phase 2: Rendering Integration
**Goal:** Extract grid computation and add boundary overlay to heatmap rendering

**Started:** 2026-05-19
**Completed:** 2026-05-19

- [x] Extract grid construction from `render_population_rf_map()` into
      `compute_interpolated_grid()` returning `(grid_u, grid_v, grid_z)`
- [x] Add `precomputed_grid` optional param to `render_population_rf_map()` and
      `render_population_rf_composite()` — skip internal grid computation when
      provided
- [x] Add `inflection_boundary` param to `render_population_rf_map()`
- [x] Add `inflection_boundaries` dict param to
      `render_population_rf_composite()`
- [x] Implement `_draw_inflection_boundary(ax, contour_uv, centroid_uv)` —
      green (`#00ff88`) closed contour, linewidth 1.5, zorder 6, centroid '+'
      marker; drawn on interpolated panel only

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_map_renderer.py`

**Dependencies:** Phase 1

### Phase 3: Pipeline Integration
**Goal:** Wire boundary computation into the population RF map pipeline and
persist metrics

**Started:** 2026-05-19
**Completed:** 2026-05-19

- [x] Add `inflection_sigma: float | None = None` param to
      `run_population_rf_maps()`
- [x] In Pass 1 per-gesture loop: call `compute_interpolated_grid()`, then
      `compute_inflection_boundary()`, pass both to renderer
- [x] In Pass 2 composite loop: for `'interpolated'` composites, pre-compute
      grids and boundaries per gesture type, pass as `inflection_boundaries`
- [x] Extend `_write_sentinel()` to include `inflection_boundaries` dict
      (serialized metrics, no contour vertices)
- [x] Add `inflection_sigma: float | None = None` to
      `visualize_population_rf_maps_flow()` in `analysis_workflow.py`
- [x] Forward `inflection_sigma` from DAG options in the dispatch block
- [x] Add `inflection_sigma: 2.0` to `visualize_population_rf_maps.options` in
      DAG config YAML

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py`
- `code/scripts/analysis_workflow.py`
- `configs/analyse_workflow_processing_dag.yaml`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] Circular Gaussian hill (100x100) — boundary is approximately circular at
      ~1 sigma from peak
- [ ] Elliptical Gaussian — PCA major/minor match sigma_x, sigma_y ratios
- [ ] Circularity of circular Gaussian is close to 1.0
- [ ] Centroid matches Gaussian center within grid resolution
- [ ] Mean IFF on contour is approximately peak * e^(-0.5) for Gaussian
- [ ] Flat surface returns `None`
- [ ] All-NaN grid returns `None`
- [ ] Peak at grid edge (open contour cannot close) returns `None`
- [ ] `inflection_boundary_to_dict()` round-trip produces JSON-safe dict

### Integration Tests
- [ ] `render_population_rf_map()` with `inflection_boundary=None` produces
      identical output to current behavior
- [ ] `render_population_rf_map()` with a valid boundary produces a PNG without
      errors

### Manual Verification
- [ ] Run pipeline on a real session with `inflection_sigma: 2.0` — inspect
      green contour on per-gesture PNGs
- [ ] Verify composite PNGs show contours on each gesture panel
- [ ] Verify sentinel JSON contains `inflection_boundaries` with metrics
- [ ] Run with `inflection_sigma: null` — PNGs render as before
- [ ] Try `inflection_sigma` values 1.0, 2.0, 4.0 — contour smoothness varies
      as expected

### Edge Cases
- [ ] Session with very few touches (sparse heatmap) — boundary may not close;
      should return `None` and render without contour
- [ ] Gesture type with zero touches — skipped, no boundary computed
- [ ] Multi-modal heatmap — picks smallest contour enclosing the global peak

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add inflection boundary to the
      Population RF maps bullet
- [ ] Add inline docstring to `compute_inflection_boundary()` explaining the
      Laplacian approach and the `gaussian_sigma` parameter

---

## Rollback Plan

1. Set `inflection_sigma: null` in DAG config — disables the feature entirely
2. All changes are additive (new file, new optional params with `None`
   defaults) — removing `rf_inflection_boundary.py` and reverting the optional
   params restores previous behavior
3. No data migrations or breaking changes to existing outputs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| False contours from noise in sparse heatmaps | Medium | Low | Tunable `gaussian_sigma`; return `None` when no valid contour encloses peak |
| Non-closed contour when peak is near mesh edge | Medium | Low | `_select_enclosing_contour` filters for closed contours enclosing the peak; returns `None` otherwise |
| NaN boundary artifacts in Laplacian | Medium | Medium | Weight-normalized Gaussian smoothing + 2-pixel NaN border masking |
| Double grid computation (pipeline + renderer) | Low | Low | `precomputed_grid` param avoids redundant interpolation |
| Multiple zero-crossing rings for multi-modal RF | Low | Low | Select smallest-area ring enclosing the global peak |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core computation + tests | ~2 hours | None |
| Phase 2: Rendering integration | ~1 hour | Phase 1 |
| Phase 3: Pipeline + DAG wiring | ~1 hour | Phase 2 |

---

## References

- Related module: `code/src/analysis/receptive_field_mapping/rf_metrics.py` — existing hotspot metrics (top-20% threshold)
- Related module: `code/src/analysis/receptive_field_mapping/rf_population_map_renderer.py` — current rendering pipeline
- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md`
