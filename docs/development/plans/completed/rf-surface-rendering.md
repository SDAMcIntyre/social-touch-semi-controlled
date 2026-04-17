# Plan: Surface-Based RF Heatmap Rendering

**Date:** 2026-04-17
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/rf-surface-rendering`

---

## Overview

**What:** Replace dot-based scatter plots in RF (receptive field) visualizations with smooth-shaded triangle mesh surfaces, using the existing 2.5D Delaunay triangulation from `define_forearm_mesh.py`.

**Why:** At certain zoom levels and camera angles, gaps between point cloud dots make RF heatmaps hard to read. The data comes from Kinect depth at ~5mm spacing — a regular grid well-suited to surface reconstruction.

**How:** A new utility module provides mesh loading (with disk cache), scalar-to-vertex mapping via KDTree, and format conversion. Each of the three rendering paths (PyVista, Matplotlib, Open3D) gains surface mode with fallback to scatter.

## Problem Statement

- RF heatmaps render forearm point clouds and spike-count overlays as scatter plots with fixed dot sizes (s=2–20)
- At wide zoom or oblique angles, gaps between dots create visual noise and make it hard to identify spatial patterns
- The forearm surface after tangent-plane rotation is a clean height field — an ideal candidate for 2.5D Delaunay surface reconstruction, which is already implemented in the preprocessing pipeline
- The existing mesh generation code (`define_forearm_mesh.py`) is not connected to the RF visualization pipeline

## Goals

### In Scope
1. Surface rendering for the forearm background in all three RF visualization paths
2. Scalar mapping of spike counts / selectivity scores onto mesh vertices
3. Single-mesh rendering (forearm + heatmap overlay as one surface with a scalar array)
4. On-the-fly mesh generation with disk cache for RF-centered forearm PLYs
5. Scatter fallback when mesh generation fails
6. Surface/scatter toggle in the PyVista interactive viewer

### Out of Scope
- 2D flattened RF heatmaps (covered by the separate `note-3d-to-2d-surface-projection-algorithms.md` knowledge base entry)
- Mesh generation changes to the preprocessing pipeline itself
- Surface rendering for the general `SceneViewer` (preprocessing GUI)
- Advanced surface reconstruction methods (Poisson, ball pivoting) — Delaunay 2.5D is sufficient

## Success Criteria

- [ ] PyVista viewer (`rf_camera_angle_picker`) renders forearm as a smooth-shaded surface by default
- [ ] Selectivity overlay appears as a colored region on the surface, not floating dots
- [ ] Surface/scatter toggle works in the PyVista settings panel
- [ ] Matplotlib path (`rf_cluster_visualizer`) produces PNG with solid surface + heatmap overlay
- [ ] Mesh cache `.obj` file appears alongside PLY after first run; second run loads from cache
- [ ] Sessions with no contact data render as a plain grey surface
- [ ] Degenerate PLY files fall back to scatter rendering without errors

---

## Technical Design

### Approach

Reuse the existing `define_forearm_mesh()` function (2.5D Delaunay on XY, preserve Z) to generate triangle meshes from RF-centered forearm PLYs. A new utility module (`rf_surface_utils.py`) handles mesh loading/caching, scalar mapping, and format conversion for each rendering backend. Each rendering path wraps mesh usage in a try/except to guarantee scatter fallback.

The key insight: after tangent-plane rotation, the forearm is always a single-valued height field over XY, making 2.5D Delaunay mathematically correct and the simplest option.

For the heatmap overlay, instead of rendering two overlapping geometries (grey forearm + colored contacts), use a **single mesh with a per-vertex scalar array**. Unassigned vertices get NaN and render as the base forearm color (PyVista `nan_color`, matplotlib face fallback). This eliminates z-fighting.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| 2.5D Delaunay (existing) | Already implemented; correct for height fields; preserves original vertices | Fails if surface folds over XY | **Chosen** — forearm after tangent rotation is always a height field |
| Screened Poisson | Smooth, watertight output | Creates extra geometry beyond point cloud boundary; loses vertex correspondence; needs density trimming | Rejected — overcomplicated |
| Ball Pivoting | Preserves vertices; no extra geometry | Sensitive to radius tuning; slower; can leave holes with uneven spacing | Rejected — less robust than Delaunay for regular grids |
| Alpha Shapes | Simple single parameter | Tends to close surface into a volume; alpha parameter is finicky | Rejected |

### Knowledge Base Constraints

- **`note-3d-to-2d-surface-projection-algorithms.md`** — Confirms the mesh is available from `define_forearm_mesh.py` and that tangent-plane rotation produces a clean height field suitable for 2.5D Delaunay.
- **`note-open3d-filament-opengl-failure.md`** — Open3D visualization can fail on some Windows machines due to driver issues; this motivates prioritizing the PyVista path (more portable).
- **`note-cupy-import-order.md`** — Any new module imported from preprocessing must respect the CuPy-before-preprocessing import order.

### Architecture Changes

**New module:**
```
code/src/analysis/receptive_field_mapping/
├── rf_surface_utils.py        (NEW — mesh loading, scalar mapping, format conversion)
├── rf_cluster_visualizer.py   (MODIFY — matplotlib surface rendering)
├── rf_visualizer.py           (MODIFY — Open3D surface rendering)
├── rf_camera_angle_task.py    (MODIFY — SessionSceneData gains forearm_mesh field)
└── gui/
    └── rf_camera_angle_picker.py  (MODIFY — PyVista surface rendering + toggle)
```

**`rf_surface_utils.py` API (4 functions):**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `load_or_build_forearm_mesh` | `(forearm_ply_path: Path, mesh_cache_path: Path = None) -> Optional[trimesh.Trimesh]` | Load cached mesh or generate via `define_forearm_mesh()` |
| `map_scalars_to_mesh` | `(mesh: trimesh.Trimesh, contact_points: ndarray, scalar_values: ndarray, radius: float = 5.0) -> ndarray` | KDTree nearest-neighbor scalar assignment; returns per-vertex array with NaN for unassigned |
| `mesh_to_pyvista` | `(mesh: trimesh.Trimesh) -> pv.PolyData` | Convert trimesh faces+vertices to VTK-format PolyData |
| `apply_rotation_to_mesh` | `(mesh: trimesh.Trimesh, R: ndarray) -> trimesh.Trimesh` | Rotate vertices via `align_points()`, fix normals |

**Mesh cache naming convention:**
```
forearm_rf_centered/{session_id}_forearm.ply          (existing)
forearm_rf_centered/{session_id}_forearm_mesh.obj     (new, auto-generated)
```

---

## Implementation Plan

### Phase 1: Foundation — `rf_surface_utils.py`
**Goal:** Create the shared utility module with mesh loading, scalar mapping, and format conversion.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] Task 1.1 — Create `rf_surface_utils.py` with `load_or_build_forearm_mesh()`: check cache mtime vs PLY mtime, load from cache or call `define_forearm_mesh()`, return `None` on failure
- [x] Task 1.2 — Implement `map_scalars_to_mesh()`: build KDTree from mesh vertices, for each contact point find vertices within radius, assign scalar (max if conflict), return per-vertex array with NaN for unassigned
- [x] Task 1.3 — Implement `mesh_to_pyvista()`: convert trimesh vertices + faces to `pv.PolyData` with VTK face format
- [x] Task 1.4 — Implement `apply_rotation_to_mesh()`: apply rotation matrix to vertices via `align_points()`, call `mesh.fix_normals()`
- [x] Task 1.5 — Verify import path for `define_forearm_mesh` (lives in `code/scripts/`, not `code/src/`); if sys.path doesn't include it, extract the core triangulation logic into a callable function or use an absolute import path

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_surface_utils.py` — New file (~80–100 lines)

**Dependencies:** None

### Phase 2: PyVista Path (highest priority)
**Goal:** Surface rendering in the interactive RF camera angle picker with scatter fallback and UI toggle.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] Task 2.1 — Add `forearm_mesh: Optional[trimesh.Trimesh] = None` to `SessionSceneData` dataclass
- [x] Task 2.2 — In `collect_session_scene_data()`: after loading forearm PLY (line ~143), call `load_or_build_forearm_mesh()` with cache path; if `apply_tangent_rotation`, apply rotation to mesh
- [x] Task 2.3 — In `_build_scene()`: when `data.forearm_mesh` is available, convert to PyVista, map scalars, render as `plotter.add_mesh(mesh_pv, scalars="selectivity", cmap=..., nan_color=forearm_color, smooth_shading=True)`
- [x] Task 2.4 — Add `render_as_surface: bool = True` to `_ViewerSettings`; add checkbox in Forearm settings group; disable `forearm_size`/`forearm_spheres` widgets when surface mode is active
- [x] Task 2.5 — Implement scatter fallback: when mesh is `None` or toggle is off, run existing point cloud code unchanged

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — `SessionSceneData` + mesh loading in `collect_session_scene_data()`
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py` — Surface rendering in `_build_scene()`, settings toggle

**Dependencies:** Phase 1

### Phase 3: Matplotlib Path
**Goal:** Surface rendering in static PNG heatmaps produced by `rf_cluster_visualizer.py`.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] Task 3.1 — In `render_forearm_heatmap()`: after loading forearm PLY, attempt mesh loading via `load_or_build_forearm_mesh()`
- [x] Task 3.2 — Replace `ax.scatter` for forearm background with `ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, color='lightgrey', shade=True, edgecolor='none')`
- [x] Task 3.3 — Map spike count scalars to mesh vertices via `map_scalars_to_mesh()`; render heatmap overlay as `plot_trisurf()` with facecolors from colormap
- [x] Task 3.4 — Retain `LogNorm` colorbar logic for spike counts
- [x] Task 3.5 — Preserve scatter fallback when mesh is `None`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — Surface rendering in `render_forearm_heatmap()`

**Dependencies:** Phase 1

### Phase 4: Open3D Path (lowest priority)
**Goal:** Surface rendering in the Open3D split-screen viewer and static 2D projection PNGs.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] Task 4.1 — In `visualize_rf_map()`: if forearm can be meshed, pass as `o3d.geometry.TriangleMesh` (via existing `trimesh_to_open3d()`) instead of point cloud
- [x] Task 4.2 — In `save_rf_map_image()`: replace `ax.scatter` forearm background with `ax.triplot()` or `plot_trisurf()` for 2D projections
- [x] Task 4.3 — Scatter fallback for both methods

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_visualizer.py` — Surface rendering in both `visualize_rf_map()` and `save_rf_map_image()`

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `map_scalars_to_mesh` with no contact points — returns all-NaN array
- [ ] `map_scalars_to_mesh` with all contacts on one vertex — max scalar wins
- [ ] `map_scalars_to_mesh` with contacts outside radius — those vertices stay NaN
- [ ] `load_or_build_forearm_mesh` with nonexistent PLY — returns `None`
- [ ] `load_or_build_forearm_mesh` with valid PLY — returns mesh with correct vertex count
- [ ] `mesh_to_pyvista` round-trip — vertex count and face count preserved

### Integration Tests
- [ ] Full pipeline: PLY → mesh generation → scalar mapping → PyVista rendering (no crash)
- [ ] Full pipeline: PLY → mesh generation → scalar mapping → matplotlib PNG output

### Manual Verification
- [ ] Run `rf_camera_angle_picker` on a session with RF centering — forearm renders as smooth surface
- [ ] Selectivity overlay appears as colored region on the mesh surface
- [ ] Toggle surface/scatter in settings panel — both modes render correctly
- [ ] Run `render_forearm_heatmap()` — PNG shows solid surface with heatmap
- [ ] Mesh cache `.obj` appears after first run; second run is faster
- [ ] Open3D viewer shows mesh in left panel (Phase 4)

### Edge Cases
- [ ] Session with no contact points — plain grey surface, no scalars
- [ ] Degenerate PLY (too few points for Delaunay) — scatter fallback, no crash
- [ ] Very sparse contacts (<10 points) — scalars map correctly to small vertex region
- [ ] PLY with no normals — mesh generation still works (Delaunay doesn't need normals)

---

## Documentation Plan

- [ ] Add inline docstrings to all 4 functions in `rf_surface_utils.py`
- [ ] Update `rf_camera_angle_picker.py` docstring to mention surface mode
- [ ] No README or CLAUDE.md changes needed (internal visualization feature)

---

## Rollback Plan

1. **Before deployment:** All changes are additive (new module + optional code paths). The scatter rendering code is preserved behind fallbacks.
2. **Rollback procedure:** Revert the feature branch commits. No data migrations, no config changes, no breaking API changes.
3. **Partial rollback:** Set `render_as_surface = False` in `_ViewerSettings` default to disable surface mode without reverting code.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `define_forearm_mesh` import path issues (`code/scripts/` not on sys.path in all contexts) | Medium | Medium | Verify sys.path at runtime; if needed, extract core triangulation (~30 lines of scipy/trimesh) into `rf_surface_utils.py` directly |
| Mesh winding incorrect after tangent-plane rotation | Low | Medium | Call `mesh.fix_normals()` after rotation; existing auto-correction in `define_forearm_mesh` handles this |
| `plot_trisurf` slow for large meshes | Low | Low | Forearm at 5mm spacing = ~5k–10k vertices, ~10k–20k faces — well within matplotlib's comfort zone |
| PyVista `nan_color` not rendering as expected | Low | Medium | Test with current pyvista 0.46.1; fallback: use two separate meshes (grey base + colored overlay) if needed |
| CuPy import order violation in new module | Low | High | `rf_surface_utils.py` imports trimesh/scipy, not preprocessing directly; CuPy constraint applies only if preprocessing tree is imported |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Foundation | ~2 hours | None |
| Phase 2: PyVista path | ~3 hours | Phase 1 |
| Phase 3: Matplotlib path | ~2 hours | Phase 1 |
| Phase 4: Open3D path | ~1 hour | Phase 1 |

---

## References

- Existing mesh generation: `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py`
- Existing trimesh→Open3D helper: `trimesh_to_open3d()` in same file
- Tangent-plane alignment: `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py`
- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Active related plan: `docs/development/plans/active/tangent-plane-rf-alignment.md`
