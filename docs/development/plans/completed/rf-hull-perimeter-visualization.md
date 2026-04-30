# Plan: RF Cluster Gallery — Perimeter Visualization for Hull Contacts

**Date:** 2026-04-28
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/gmm-gallery-feature-ranges`

---

## Overview

Replace the 3D convex hull wireframe rendering for neuron and cluster contact
points in the RF Cluster Gallery Viewer with **2D alpha-shape perimeters**
computed in the tangent plane. Spatially disconnected groups of contacts are
separated into independent blobs via KDTree connected components before
perimeter estimation, so receptive fields that span multiple distinct regions
render as separate closed loops. A three-way display mode selector
(*Perimeter* / *Point cloud* / *Off*) replaces the current binary toggle.

## Problem Statement

The current implementation wraps all contact points in a 3D `ConvexHull`
(scipy). This over-estimates receptive field extent and cannot represent
receptive fields that span two or more disconnected patches on the forearm.
There is also no way to visualise the raw contact-point scatter independently
of the forearm mesh. The user needs a boundary visualisation that follows the
true outer edge of each spatial region.

## Goals

### In Scope
1. Replace 3D convex hull wireframes with 2D alpha-shape perimeters
   (one closed loop per spatial blob, rendered in the tangent plane)
2. Separate contact clouds into blobs by KDTree connected components with a
   configurable distance threshold (`blob_sep_mm`)
3. Expose `Alpha radius (mm)` as a tunable control for perimeter tightness
4. Add a three-way display mode: **Perimeter** / **Point cloud** / **Off**
5. Remove the obsolete `Hull wireframes` checkbox and `Max hull edge` spinbox

### Out of Scope
- Changing `rf_2d_renderer._draw_hull_2d` (separate matplotlib renderer)
- Changing `rf_metrics._compute_convex_hull_area` (quantitative metric, not
  visualisation)
- Exporting or saving perimeters to disk
- GPU acceleration

## Success Criteria

- [ ] Gallery renders alpha-shape perimeters for both neuron and cluster
      contact layers
- [ ] Two spatially separated contact blobs produce two separate closed
      perimeter loops (verified visually)
- [ ] Display mode combo switches between Perimeter / Point cloud / Off and
      triggers an immediate scene rebuild
- [ ] `Blob sep` spinbox correctly groups/separates blobs as its value changes
- [ ] `Alpha radius` spinbox controls perimeter tightness (smaller → tighter,
      larger → approaches convex hull)
- [ ] Old `show_hull_wireframes` checkbox and `Max hull edge` spinbox are gone

---

## Technical Design

### Approach

After the existing tangent-plane rotation is applied to 3D contact points,
x/y becomes the 2D tangent-plane projection and z is approximately constant.
The algorithm operates in this 2D space:

**Step 1 — Blob separation (KDTree connected components)**  
Build a `scipy.spatial.KDTree` on `pts[:, :2]`. Query all pairs within
`blob_sep_mm`. Build a sparse adjacency matrix and call
`scipy.sparse.csgraph.connected_components` to label each point.  
*No sklearn dependency — scipy is already used throughout the RF mapping code.*

**Step 2 — Per-blob alpha shape (Delaunay circumradius filter)**  
For each blob with ≥ 3 points:
- Run `scipy.spatial.Delaunay` on the 2D blob points.
- Keep triangles whose circumradius `R = abc / (4·Area) < alpha_mm`.
- Boundary edges = edges shared by **exactly one** kept triangle.

The alpha-shape algorithm is from Edelsbrunner et al. (1983) and is
implemented from first principles using scipy — no external library required.

**Step 3 — 3D reconstruction**  
For each blob, prepend `mean(blob_z)` as the z coordinate for all boundary
vertices, then accumulate all blob boundaries into a single `pv.PolyData`
wireframe per layer (neuron / cluster).

**Parameter intuition:**

| Scenario | Behaviour |
|----------|-----------|
| `alpha_mm` very large | Approaches convex hull |
| `alpha_mm` ≈ typical inter-point spacing | Tight concave outline |
| `alpha_mm` too small | No valid triangles → no perimeter drawn |
| `blob_sep_mm` large | All points merged into one blob |
| `blob_sep_mm` small | Many small blobs; single-point blobs silently skipped |

Defaults: `blob_sep_mm = 20 mm`, `alpha_mm = 15 mm`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| 2D alpha shape via Delaunay | Pure scipy; handles concavities + multiple blobs | Requires manual edge-tracing | **Chosen** |
| Convex hull per blob | Simpler | Still over-estimates; user wants true perimeter | Rejected |
| 3D alpha shape (tetrahedra) | True 3D | Much more complex; overkill for near-planar data | Rejected |
| `alphashape` library | Simpler API | Not in project deps; adds install risk | Rejected |
| DBSCAN blob separation | Standard | Requires sklearn; scipy KDTree+csgraph is equivalent | Rejected |

### Architecture Changes

**Single file modified:**
`code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

| Change | Detail |
|--------|--------|
| Imports | Replace `ConvexHull` with `Delaunay, QhullError`; add `csr_matrix` (scipy.sparse) and `connected_components` (scipy.sparse.csgraph) |
| Remove | `_build_convex_hull_mesh()` — private, no callers outside this file |
| Add | `_alpha_shape_edges(pts_2d, alpha_mm) → set[tuple[int,int]]` |
| Add | `_separate_blobs(pts_2d, max_dist_mm) → np.ndarray` |
| Add | `_build_perimeter_mesh(points, blob_sep_mm, alpha_mm) → Optional[pv.PolyData]` |
| Modify | `_GallerySettings` — remove 2 fields, add 3 fields |
| Modify | `_build_scene()` hull block (lines 693–712) |
| Modify | `_build_settings_panel()` hull UI block (lines 375–412) |

---

## Implementation Plan

### Phase 1: Core Algorithm
**Goal:** Implement the three new module-level private functions.

- [x] Replace `from scipy.spatial import ConvexHull` with
      `from scipy.spatial import Delaunay, QhullError`
- [x] Add `from scipy.sparse import csr_matrix` and
      `from scipy.sparse.csgraph import connected_components`
- [x] Remove `_build_convex_hull_mesh()` entirely
- [x] Implement `_separate_blobs(pts_2d, max_dist_mm) → np.ndarray`  
      Build KDTree on pts_2d; query_pairs(max_dist_mm); build csr_matrix
      adjacency; connected_components → label array
- [x] Implement `_alpha_shape_edges(pts_2d, alpha_mm) → set[tuple[int,int]]`  
      Delaunay triangulation; per-simplex circumradius filter (guard area ≤ 0);
      edge_count dict; return edges with count == 1
- [x] Implement `_build_perimeter_mesh(points, blob_sep_mm, alpha_mm) → Optional[pv.PolyData]`  
      Guard (None / len < 3); project to 2D; `_separate_blobs`; per-blob
      `_alpha_shape_edges`; 3D reconstruction with mean-z; accumulate into
      single PolyData; return None if empty

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

**Dependencies:** None

### Phase 2: Settings & Scene Rendering
**Goal:** Wire new settings into `_GallerySettings` and `_build_scene()`.

- [x] Update `_GallerySettings` dataclass:
  - Remove: `show_hull_wireframes: bool = True`, `hull_max_edge_mm: float = 15.0`
  - Add: `hull_display_mode: str = "perimeter"` (`"perimeter"` | `"pointcloud"` | `"off"`)
  - Add: `hull_blob_sep_mm: float = 20.0`
  - Add: `hull_alpha_mm: float = 15.0`
- [x] Replace hull block in `_build_scene()` (lines 693–712) with:
  - Iterate over both contact layers (neuron + cluster) in a loop
  - Apply tangent rotation if present
  - Branch on `hull_display_mode`:
    - `"perimeter"` → `_build_perimeter_mesh(...)` → `add_mesh(style="wireframe")`
    - `"pointcloud"` → `pv.PolyData(pts)` → `add_mesh(color=..., point_size=s.contact_size)`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

**Dependencies:** Phase 1

### Phase 3: GUI Control Panel
**Goal:** Replace old hull UI controls with the new three-way selector and parameter spinboxes.

- [x] Remove `_hull_cb` (checkbox) and `_hull_edge_spin` (spinbox) from
      `_build_settings_panel()`
- [x] Add `QComboBox` with items `["Perimeter", "Point cloud", "Off"]`
      connected to `hull_display_mode` via `_set_and_rebuild`
- [x] Add `QDoubleSpinBox` "Blob sep" (range 1–200 mm, step 5, suffix " mm")
      connected to `hull_blob_sep_mm`
- [x] Add `QDoubleSpinBox` "Alpha radius" (range 1–100 mm, step 1, suffix " mm")
      connected to `hull_alpha_mm`
- [x] Keep existing neuron and cluster hull color buttons unchanged

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `_alpha_shape_edges` — 3×3 uniform grid, alpha just large enough: verify
      only outer-boundary edges returned, no interior edges
- [ ] `_alpha_shape_edges` — 3 collinear points: verify empty set returned
      (degenerate triangle area guard)
- [ ] `_separate_blobs` — two clusters with gap >> threshold: verify two
      distinct integer labels
- [ ] `_build_perimeter_mesh` — None input → returns None
- [ ] `_build_perimeter_mesh` — 2-point input → returns None (below min-3 guard)
- [ ] `_build_perimeter_mesh` — two well-separated blobs → PolyData.lines
      non-empty and spans both blobs

### Manual Verification
- [ ] Open Gallery Viewer on a session with dense contacts — perimeter mode
      shows closed loop(s) around contact points
- [ ] Increase blob separation → separate loops merge into one
- [ ] Decrease blob separation → one loop splits into multiple
- [ ] Switch to "Point cloud" → neuron and cluster contacts appear as coloured
      dots with no forearm mesh dependency
- [ ] Switch to "Off" → no hull overlay visible at all
- [ ] Decrease alpha radius → perimeter wraps more tightly / becomes concave
- [ ] Session with < 3 contacts → no perimeter, no exception, scene renders

### Edge Cases
- [ ] All contact points identical (degenerate cloud) → no perimeter, no crash
- [ ] Single contact point → no perimeter, no crash
- [ ] `tangent_rotation` is None → perimeter computed using raw 3D coordinates
      projected via `[:, :2]`

---

## Documentation Plan

- [ ] No CLAUDE.md update required — the change is visualisation-only, not
      architectural
- [ ] Plan moves to `active/` when implementation branch is opened, then
      `completed/` when shipped

---

## Rollback Plan

All changes are confined to one file:

```bash
git checkout HEAD -- code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py
```

No data migration, no schema changes, no downstream effects.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Degenerate Delaunay triangle — sqrt of negative in circumradius | Med | High | Guard `if area <= 0: continue` before circumradius computation |
| `scipy.sparse.csgraph` unavailable | Low | High | Ships with all scipy ≥ 0.9; already in requirements |
| Empty perimeter when `alpha_mm` too small | Med | Low | Falls through gracefully — no mesh added, scene still renders |
| Lambda closure captures wrong variable in combo signal | Med | Med | Use dict lookup inside lambda body (no loop-variable closure) |
| KDTree slow for very large point clouds | Low | Low | Contact clouds are typically < 500 pts; overhead negligible |

---

## References

- Edelsbrunner, H., Kirkpatrick, D., Mussig, R. (1983). "On the shape of a set
  of points in the plane." *IEEE Trans. Inf. Theory*, 29(4), 551–559.
- Current hull rendering: `rf_cluster_gallery_viewer.py` lines 78–116, 693–712
- Current hull UI: `rf_cluster_gallery_viewer.py` lines 375–412
- Existing KDTree usage: `rf_surface_utils.py:11`, `tangent_plane_alignment.py:14`
