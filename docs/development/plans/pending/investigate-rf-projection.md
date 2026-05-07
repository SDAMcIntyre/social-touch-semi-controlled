# Plan: Investigate RF 2D Projection Correctness

**Date:** 2026-05-07
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/reduce-population-rf-grid-metrics`
**Branch:** `feature/reduce-population-rf-grid-metrics`

---

## Overview

The population RF grid metrics pipeline computes area metrics (`threshold_area_mm2`, `hotspot_area_mm2`, `convex_hull_area_mm2`, `perimeter_mm`, `circularity`, `eccentricity`) by projecting 3D forearm vertices onto a 2D tangent plane and computing convex hulls. The projection has never been visually verified. This plan creates a diagnostic script to inspect the projection and fixes a confirmed centroid-mismatch bug discovered during code review.

## Problem Statement

Area metrics depend entirely on the quality of the 3D→2D tangent plane projection. If the tangent plane is misaligned, all area/shape metrics are wrong. Code review revealed:

1. **Centroid mismatch bug**: `compute_grid_cell_metrics()` calls `compute_rf_metrics()` (which uses a spike-weighted centroid for the tangent plane) then calls `project_to_2d()` a second time with an arithmetic-mean centroid. Boundary-shape metrics (`perimeter_mm`, `circularity`, `eccentricity`) are computed in a different tangent plane than area metrics.
2. **No visual validation exists**: the tangent plane normal, the k=50 SVD neighbors, and the projected 2D coordinates have never been inspected visually.
3. **Curvature distortion is undocumented**: the knowledge base notes ~5–10% distance compression at tangent-plane edges (quadratic in distance from center), but no session-specific measurement exists.

## Goals

### In Scope
1. Fix the centroid mismatch bug so all metrics use the same tangent plane
2. Create a diagnostic script that visualizes the projection pipeline for any grid cell
3. Compare tangent-plane vs cylindrical-unwrap projections on real data

### Out of Scope
- Implementing a new projection algorithm (exponential map, ARAP, etc.)
- Changing the default projection method used by the pipeline
- Automated regression tests for projection quality

## Success Criteria

- [ ] Centroid mismatch fixed — both `compute_rf_metrics()` and the boundary-shape projection use the same (weighted) centroid
- [ ] Diagnostic script runs on any session/gesture/cell and produces interpretable visual output
- [ ] 3D panel shows: forearm mesh, RF vertices (colored by IFF), tangent plane normal arrow, k=50 SVD neighbors
- [ ] 2D panels show: tangent-plane projection with convex hull / threshold / hotspot boundaries, and cylindrical-unwrap comparison
- [ ] Area values from both projections are annotated for direct comparison

---

## Technical Design

### Approach

Fix the bug first (small, isolated change), then build the diagnostic script. The script loads existing pipeline artifacts (forearm PLY + population RF grid NPZ) and re-runs the projection steps with visual output at each stage. Uses PyVista for interactive 3D and matplotlib for 2D panels.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Standalone diagnostic script | No pipeline changes, reuses existing functions, user controls which cells to inspect | Requires manual invocation | **Chosen** |
| Add debug mode to the pipeline itself | Integrated, runs automatically | Clutters pipeline code, heavy 3D deps in batch mode | Rejected |
| Unit test with golden-file comparison | Automated | Doesn't help the user *see* the projection; hard to define "correct" for real data | Rejected (out of scope) |

### Architecture Changes

No new modules or classes. One new script, one bug fix.

**Reusable functions (no wrappers needed):**

| Module | Function | Used for |
|--------|----------|----------|
| `rf_data_loader.py` | `load_forearm_vertices()` | Load forearm PLY mesh |
| `tangent_plane_alignment.py` | `compute_tangent_plane_rotation()`, `align_points()` | Reproduce tangent plane fit + visualize normal/neighbors |
| `rf_projection.py` | `project_to_2d()`, `project_cylindrical_unwrap()` | Compare both projection methods |
| `rf_metrics.py` | `compute_rf_metrics()`, `_compute_weighted_centroid()` | Compute area metrics + weighted centroid |
| `rf_grid_cell_metrics.py` | `compute_grid_cell_metrics()` | End-to-end metric computation (after bug fix) |

---

## Implementation Plan

### Phase 1: Fix centroid mismatch
**Goal:** Ensure all metrics in `compute_grid_cell_metrics()` use the same spike-weighted centroid for the tangent plane.

**Tasks:**
- [ ] Task 1.1 — In `compute_grid_cell_metrics()`, compute the spike-weighted centroid from `active_values` and `active_positions` before calling `project_to_2d()`
- [ ] Task 1.2 — Pass the weighted centroid (instead of `active_positions.mean(axis=0)`) to the second `project_to_2d()` call for boundary-shape metrics

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_grid_cell_metrics.py` — Replace arithmetic-mean centroid with weighted centroid at the `project_to_2d()` call (~line 428)

**Dependencies:** None

### Phase 2: Create diagnostic script
**Goal:** Build `inspect_rf_projection.py` that visualizes the full projection pipeline for a chosen grid cell.

**Tasks:**
- [ ] Task 2.1 — CLI interface: `--session`, `--gesture`, `--cell`, `--output-dir`, `--all-cells`
- [ ] Task 2.2 — Data loading: find forearm PLY and population RF grid NPZ for the specified session/gesture, extract the RF map for the specified cell
- [ ] Task 2.3 — 3D PyVista panel: forearm mesh (gray, semi-transparent), active RF vertices (colored by IFF), weighted centroid (red sphere), arithmetic centroid (blue sphere, to confirm they now coincide), tangent plane normal (arrow), k=50 SVD neighbors (highlighted)
- [ ] Task 2.4 — 2D matplotlib panel (tangent-plane): projected vertices colored by IFF, convex hull outline, threshold boundary (≥50% peak, dashed), hotspot boundary (top 20%, dotted), area values annotated
- [ ] Task 2.5 — 2D matplotlib panel (cylindrical unwrap): same data projected via `project_cylindrical_unwrap()`, same hull overlays, area values annotated for comparison
- [ ] Task 2.6 — `--all-cells` mode: iterate over all non-empty cells, save panels to `<output-dir>/<session>/<gesture>/cell_<index>/`

**Files Modified:**
- `code/scripts/inspect_rf_projection.py` — **Create** standalone diagnostic script

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run the script on a known session (e.g. ST13-01, tap, cell 5) and visually confirm the tangent plane normal is perpendicular to the forearm surface
- [ ] Confirm the k=50 neighbors are clustered around the contact region, not scattered
- [ ] Confirm the weighted and arithmetic centroids now coincide (after bug fix)
- [ ] Confirm the convex hull / threshold / hotspot boundaries look sensible on the 2D projection
- [ ] Compare tangent-plane vs cylindrical-unwrap areas — check that difference is within the expected ~5–10% range for reasonably sized RFs
- [ ] Run `--all-cells` and scan outputs for any cell with a visibly wrong tangent plane (normal pointing sideways, projected points collapsed, etc.)

### Edge Cases
- [ ] Cell with very few active vertices (<5) — does the tangent plane degrade gracefully?
- [ ] Cell near the wrist/elbow where forearm curvature is strongest — is distortion noticeably worse?
- [ ] Cell where forearm_vertices is sparse around the contact region

---

## Documentation Plan

- [ ] Docstring at top of `inspect_rf_projection.py` with usage examples
- [ ] Update `code/src/analysis/CLAUDE.md` to mention the diagnostic script under RF mapping tools

---

## Rollback Plan

1. The centroid bug fix is a single-line change — revert the one edit in `rf_grid_cell_metrics.py`
2. The diagnostic script is a new file with no integration points — delete it

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tangent plane is fundamentally inadequate for large RFs | Low | High | The script compares tangent-plane vs cylindrical-unwrap; if distortion is large, recommend switching default projection |
| PyVista not available in the user's environment | Low | Med | PyVista is already used by 6+ GUI modules in this codebase |
| k=50 SVD fit is poor on curved forearm regions | Med | Med | The script makes this visible; if confirmed, a follow-up plan can increase k or use local mesh normals |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Fix centroid mismatch | ~15 min | None |
| Phase 2: Diagnostic script | ~2 hours | Phase 1 |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` — tangent plane fitting
- `code/src/analysis/receptive_field_mapping/rf_projection.py` — projection methods
- `code/src/analysis/receptive_field_mapping/rf_metrics.py` — area metric computation
- `code/src/analysis/receptive_field_mapping/rf_grid_cell_metrics.py` — grid cell metric orchestration (bug location)
