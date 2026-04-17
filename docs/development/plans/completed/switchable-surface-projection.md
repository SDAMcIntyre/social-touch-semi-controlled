# Plan: Switchable 3D-to-2D Surface Projection Infrastructure

**Created:** 2026-04-17 18:00
**Approved:** —
**Completed:** 2026-04-17 15:34
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/switchable-surface-projection`

---

## Overview

Build a method-switchable 2D projection infrastructure for RF heatmaps so that
body surfaces (initially fingers, which are cylindrical) can be flattened onto a
2D plane for publication-quality visualisation. The projection method is selected
by a string key (`"tangent_plane"`, `"cylindrical_unwrap"`), and a new 2D
renderer consumes the projected `(u, v)` coordinates. The existing 3D rendering
path is preserved and remains the default.

## Problem Statement

RF heatmaps are currently rendered as 3D trisurf/scatter plots, which suffer
from depth ambiguity, inconsistent cross-session viewpoints, and non-standard
format for publication. For finger skin — a nearly perfect cylinder (radius
~7–12 mm) — a proper 2D unwrapping is needed, but the forearm pipeline also
benefits from a tangent-plane 2D projection. No projection or 2D rendering
infrastructure exists today; the current code only rotates the 3D scene for a
face-on camera.

## Goals

### In Scope

1. Projection registry mapping string keys to projection functions, with a
   single dispatch function `project_to_2d()`
2. `"tangent_plane"` method wrapping existing `compute_tangent_plane_rotation()`
   + `align_points()` + drop-Z
3. `"cylindrical_unwrap"` method: fit cylinder axis via PCA, unwrap to
   `(u = r*theta, v = h)` with seam placement and optional per-point radius
4. 2D renderer producing scatter + interpolated heatmap panels, matching the
   existing visual conventions (RdYlBu_r, LogNorm, black background)
5. Integration into `render_forearm_heatmap()` via a `projection_method`
   parameter (default `None` = 3D, unchanged)
6. Threading `projection_method` through both `run_simple_rf_mapping()` and
   `run_cluster_rf_mapping()`

### Out of Scope

- Exponential map, geodesic MDS, LSCM/ARAP implementations (future methods,
  but the registry is designed for them)
- DAG YAML configuration of the projection method (can be added later)
- Interactive 2D GUI (only static PNG output for now)
- Forearm-specific cylinder fitting refinements (RANSAC, etc.)

## Success Criteria

- [ ] `project_to_2d(pts, verts, centroid, method="tangent_plane")` returns
      `(N, 2)` array identical to manual tangent-plane + drop-Z
- [ ] `project_to_2d(pts, verts, centroid, method="cylindrical_unwrap")` returns
      `(N, 2)` array with `v = height` and `u = arc_length` in mm
- [ ] `project_to_2d(..., method="nonexistent")` raises `KeyError`
- [ ] `render_forearm_heatmap(..., projection_method=None)` produces identical
      output to the current 3D renderer (backward compatible)
- [ ] `render_forearm_heatmap(..., projection_method="cylindrical_unwrap")`
      produces a 2D PNG with scatter + interpolated heatmap panels
- [ ] Output filename includes the projection method to avoid overwriting 3D
      renders
- [ ] Synthetic cylinder test: known theta positions recovered within 0.01 rad

---

## Technical Design

### Approach

A flat registry (`dict[str, ProjectionFn]`) maps method names to pure
functions. Each function has the same signature: `(points_3d, forearm_vertices,
contact_centroid, **kwargs) -> np.ndarray(N, 2)`. A single entry point
`project_to_2d()` dispatches by key. The 2D renderer is decoupled from
projection — it receives `(u, v, scalar)` and renders method-agnostic plots.

The existing `render_forearm_heatmap()` gains one optional parameter. When set,
it projects the data, calls the 2D renderer, and returns early — the 3D code
path is untouched.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Registry dict + dispatch function | Simple, extensible, no base class | No interface enforcement | **Chosen** — matches the pipeline architecture vision in the knowledge base |
| ABC with `Projector` subclasses | Type-safe, encapsulates state | Over-engineered for stateless pure functions | Rejected |
| Separate 2D render function per method | Method-specific axis labels, etc. | Duplicated rendering code | Rejected — renderer takes method name for labels |

### Architecture Changes

Two new files, three modified files:

```
code/src/analysis/receptive_field_mapping/
├── rf_projection.py          # NEW — projection registry + methods
├── rf_2d_renderer.py         # NEW — 2D scatter + heatmap renderer
├── rf_cluster_visualizer.py  # MODIFY — add projection_method param
├── rf_simple_pipeline.py     # MODIFY — thread projection_method
├── rf_cluster_pipeline.py    # MODIFY — thread projection_method
└── __init__.py               # MODIFY — export project_to_2d
```

### Key Existing Code to Reuse

| Module | Function | Reuse |
|--------|----------|-------|
| `tangent_plane_alignment.py` | `compute_tangent_plane_rotation()`, `align_points()` | Wrapped by `project_tangent_plane` |
| `rf_surface_utils.py` | `load_or_build_forearm_mesh()` | Forearm vertex loading in integration path |
| `rf_cluster_visualizer.py` | LogNorm / RdYlBu_r / black-background styling (lines 91–99, 161–198) | Replicated in `rf_2d_renderer.py` |

### Data Flow

```
spike_counts_df (x, y, z, spike_count)
     │
     ▼
render_forearm_heatmap(projection_method="cylindrical_unwrap")
     │
     ├─ load forearm PLY → forearm_vertices
     ├─ compute contact_centroid
     │
     ▼
project_to_2d(spike_xyz, forearm_vertices, centroid, method=...)
     │                                        │
     ▼                                        ▼
 PROJECTION_METHODS["cylindrical_unwrap"]   PROJECTION_METHODS["tangent_plane"]
     │                                        │
     ▼                                        ▼
 uv_points (N, 2) in mm                   uv_points (N, 2) in mm
     │
     ▼
render_2d_heatmap(uv_points, counts, ..., projection_method=...)
     │
     ├─ Panel 1: scatter plot (u, v) colored by spike_count
     ├─ Panel 2: interpolated heatmap (griddata cubic)
     └─ Save PNG
```

---

## Implementation Plan

### Phase 1: Projection Module

**Goal:** Create `rf_projection.py` with the registry, tangent-plane wrapper,
and cylindrical unwrapping implementation.

**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 1.1 — Create `rf_projection.py` with `PROJECTION_METHODS` dict and
      `project_to_2d()` dispatch function
- [x] 1.2 — Implement `project_tangent_plane()` wrapping
      `compute_tangent_plane_rotation()` + `align_points()` + `[:, :2]` slice
- [x] 1.3 — Implement `fit_cylinder_axis()`: PCA on forearm vertices to extract
      axis direction, project centroid onto axis for center, compute mean radial
      distance for radius
- [x] 1.4 — Implement `project_cylindrical_unwrap()`: compute `h`, `theta`,
      `u = r * theta`, seam placement opposite RF center, optional per-point
      radius via `per_point_radius` kwarg
- [x] 1.5 — Register both methods via module-level `PROJECTION_METHODS` dict
      population

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_projection.py` — New file

**Dependencies:** None

### Phase 2: 2D Renderer

**Goal:** Create `rf_2d_renderer.py` with scatter + interpolated heatmap output.

**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 2.1 — Implement `render_2d_heatmap()` with two-panel figure: left =
      scatter, right = interpolated heatmap via `scipy.interpolate.griddata`
- [x] 2.2 — Match visual conventions from `rf_cluster_visualizer.py`: RdYlBu_r
      colormap, LogNorm, black background, white axis labels/ticks
- [x] 2.3 — Dynamic axis labels based on `projection_method`: for
      `cylindrical_unwrap` → "Longitudinal (mm)" / "Circumferential (mm)"; for
      `tangent_plane` → "u (mm)" / "v (mm)"
- [x] 2.4 — Optional forearm-vertices background as light grey scatter for
      spatial context
- [x] 2.5 — Title annotation with session_id, cluster_label, and projection
      method name

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_2d_renderer.py` — New file

**Dependencies:** Phase 1

### Phase 3: Pipeline Integration

**Goal:** Wire projection into the existing rendering and pipeline entry points.

**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 3.1 — Add `projection_method: Optional[str] = None` parameter to
      `render_forearm_heatmap()` in `rf_cluster_visualizer.py`
- [x] 3.2 — Implement early-return 2D path: when `projection_method` is set,
      load forearm vertices, call `project_to_2d()`, optionally project forearm
      vertices, call `render_2d_heatmap()`, return
- [x] 3.3 — Thread `projection_method` through `run_simple_rf_mapping()` in
      `rf_simple_pipeline.py`
- [x] 3.4 — Thread `projection_method` through `run_cluster_rf_mapping()` in
      `rf_cluster_pipeline.py`
- [x] 3.5 — Adjust output filenames: append `_{method}` before `.png` when
      projection_method is set (e.g., `ST13-01_rf_simple_cylindrical_unwrap.png`)
- [x] 3.6 — Export `project_to_2d` from `__init__.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — Add
  parameter + early-return 2D dispatch block before line 91
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — Add
  parameter to `run_simple_rf_mapping()`, pass through to render call
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — Add
  parameter to `run_cluster_rf_mapping()`, pass through to render call
- `code/src/analysis/receptive_field_mapping/__init__.py` — Add `project_to_2d`
  to imports and `__all__`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

- [ ] Synthetic cylinder: 100 points on cylinder (r=10mm, L=40mm), verify
      `project_cylindrical_unwrap` recovers known theta within 0.01 rad and h
      exactly
- [ ] Tangent-plane equivalence: verify `project_tangent_plane` output matches
      manual `compute_tangent_plane_rotation()` + `align_points()[:, :2]`
- [ ] Registry dispatch: `project_to_2d(..., method="tangent_plane")` and
      `"cylindrical_unwrap"` dispatch correctly; `"nonexistent"` raises KeyError
- [ ] Per-point radius: on a tapered cylinder (r varies 8→12mm), verify arc
      lengths differ from global-radius version
- [ ] Seam placement: verify seam (theta discontinuity) is at pi, opposite the
      contact centroid at theta=0

### Integration Tests

- [ ] `render_forearm_heatmap(..., projection_method=None)` produces identical
      PNG to current code (byte-level comparison or visual diff)
- [ ] `render_forearm_heatmap(..., projection_method="cylindrical_unwrap")`
      produces a valid 2-panel PNG without errors

### Manual Verification

- [ ] Run on a real finger session: visually inspect that cylindrical unwrap
      produces a rectangular map with RF hotspot near center
- [ ] Run on a real forearm session: visually compare tangent_plane 2D vs
      existing 3D render for spatial consistency
- [ ] Side-by-side comparison: both methods on same session, verify orientation
      and scale consistency

### Edge Cases

- [ ] Empty spike_counts_df: should log warning and return early (match existing
      behavior)
- [ ] Missing forearm PLY: should gracefully handle — project spike points only,
      no forearm background
- [ ] Single spike point: should produce a valid scatter with one dot
- [ ] All spike points at same location: griddata interpolation handles
      gracefully (no crash)

---

## Documentation Plan

- [ ] Update knowledge base note
      `note-3d-to-2d-surface-projection-algorithms.md` with implementation
      status of tangent_plane and cylindrical_unwrap
- [ ] Add inline docstrings to `rf_projection.py` explaining the registry
      pattern and how to add new methods

---

## Rollback Plan

All changes are additive:

1. The two new files (`rf_projection.py`, `rf_2d_renderer.py`) can be deleted
2. The `projection_method` parameter defaults to `None` everywhere — removing it
   restores the original signatures
3. No existing code paths are modified; the 2D path is an early-return branch
4. No data migrations, no config changes, no database modifications
5. Revert: `git revert <merge-commit>` on dev

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PCA axis not aligned to cylinder for non-PCA-calibrated data | Low | Med | `fit_cylinder_axis` computes its own PCA from forearm_vertices, does not depend on prior calibration |
| Seam artifact visible in heatmap near theta=pi | Low | Low | Seam placed opposite RF center; for fingers the Kinect only captures ~180 deg, so seam falls in unobserved region |
| griddata interpolation artifacts with sparse spike data | Med | Low | Fallback to scatter-only if fewer than 4 data points (minimum for cubic interpolation) |
| Matplotlib Agg backend not rendering 2D properly in CI | Low | Low | 2D figures use standard 2D axes, no 3D projection — simpler than current code |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Projection Module | ~1 hour | None |
| Phase 2: 2D Renderer | ~1 hour | Phase 1 |
| Phase 3: Pipeline Integration | ~30 min | Phase 2 |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Completed plan: `docs/development/plans/completed/rf-surface-rendering.md`
- Completed plan: `docs/development/plans/completed/tangent-plane-rf-alignment.md`
