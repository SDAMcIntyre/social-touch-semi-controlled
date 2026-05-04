# Plan: RF Feature-Space Explorer — Layout, Bug Fix & Sidecar Cache

**Date:** 2026-05-03
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 14:35
**Base Branch:** `feature/rf-explorer-independent-dag-flow`
**Branch:** `feature/rf-explorer-layout-bugfix-cache`

---

## Overview

**What:** Three improvements to the RF Feature-Space Explorer GUI: a new
full-width point-cloud layout with an overlaid scatter, a fix for the
RF-disappears-on-checkbox-toggle bug, and a sidecar cache for fast session
reloads.

**Why:** After hands-on use, the side-by-side layout wastes screen space, the
Delaunay surface obscures individual vertex responses, a rendering bug makes
the tool unusable after any checkbox interaction, and multi-second load times
slow down interactive exploration.

**How:** Restructure the Qt layout from a horizontal splitter to a grid overlay,
switch from Delaunay mesh to raw point cloud, rebuild the PyVista actor on
every filter update (fixing the stale VTK mapper), and add a `.npz` sidecar
that skips CSV parsing and geometry computation on cache hit.

## Problem Statement

The RF Feature-Space Explorer shipped on `feature/rf-feature-space-explorer` as
a minimal interactive tool. Three issues emerged from use:

1. **Layout inefficiency:** The 40/60 horizontal split gives too much space to
   the scatter plot and not enough to the 3D view. The Delaunay surface
   rendering fills gaps between vertices, hiding the actual per-vertex spike
   density pattern. Extreme pressure/velocity outliers (top/bottom 1%) compress
   the scatter's useful range.

2. **RF display bug:** After toggling any gesture-type checkbox, the RF heatmap
   disappears permanently — even when restoring the original checkbox state.
   This makes the gesture-type filtering feature effectively broken.

3. **Slow loading:** Each session load re-reads a large CSV, builds a Delaunay
   mesh, computes tangent-plane rotation, constructs a KDTree, and queries
   nearest vertices. This takes several seconds per session, discouraging
   rapid session switching.

## Goals

### In Scope

1. Full-width 3D point cloud view with the pressure-velocity scatter overlaid
   at bottom-left (~350x280px, semi-transparent background)
2. Forearm rendered as a point cloud (spheres at vertex positions) instead of a
   Delaunay-triangulated surface
3. Scatter axis limits trimmed to the 1st-99th percentile of each dimension
4. Fix: RF heatmap correctly updates (appears/disappears/reappears) when
   gesture-type checkboxes are toggled in any order
5. Sidecar `.npz` cache alongside the series CSV for near-instant reloads,
   with mtime-based freshness checking

### Out of Scope

- Configurable point size, colormap, or background color (future enhancement)
- Camera persistence across sessions
- Cross-session point cloud pooling
- Export or screenshot functionality

## Success Criteria

- [ ] 3D view fills the window; scatter is a small overlay at bottom-left
- [ ] Forearm renders as individual point-spheres, not a triangulated surface
- [ ] Scatter axes exclude the 1% extreme tails on each dimension
- [ ] Unchecking all gesture types clears the RF; re-checking restores it
- [ ] Toggling checkboxes in any sequence always shows the correct RF
- [ ] First session load creates `*_explorer_cache.npz`; second load is
      near-instant (logs confirm cache hit)
- [ ] Touching the series CSV or forearm PLY invalidates the cache (logs
      confirm cache miss + rebuild)

---

## Technical Design

### Approach

**Layout:** Replace the `QSplitter(Horizontal)` with a `QGridLayout` where both
the PyVista interactor and the matplotlib canvas occupy the same cell (0, 0).
The 3D view fills the cell; the scatter canvas is pinned to
`Qt.AlignBottom | Qt.AlignLeft` with a fixed size. This is the standard Qt
idiom for overlays — no manual `resizeEvent` overrides needed.

**Point cloud:** Remove Delaunay mesh construction from `load_explorer_data()`
entirely. Store raw rotated vertices in `ExplorerSessionData.forearm_vertices`.
In the GUI, create `pv.PolyData(vertices)` and render with
`render_points_as_spheres=True, point_size=5`. The gallery viewer
(`rf_cluster_gallery_viewer.py:1006-1036`) already uses this pattern.

**Bug fix:** The root cause is that `_apply_filter_update()` mutates the scalar
array in-place (`mesh["spike_density"] = ...`) and calls `plotter.render()`.
When scalars become all-zeros (all types unchecked), the VTK mapper sets its
internal range to [0, 0]. On re-check, the stale range clamps all values to the
same color (black on "hot" = invisible). Fix: rebuild the `pv.PolyData` and
call `add_mesh(name="forearm")` on each update — PyVista replaces actors by
name, forcing a fresh mapper with auto-computed scalar range.

**Sidecar cache:** Follow the `load_forearm_vertices()` pattern in
`rf_data_loader.py:29-63`: check if `*_explorer_cache.npz` exists and its
mtime >= max(series CSV mtime, forearm PLY mtime). On hit, load arrays
directly. On miss, compute + save. Store `gesture_types` as categorical codes +
labels to avoid pickle in the npz format.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Layout: absolute positioning** for scatter overlay | Full control over placement | Requires `resizeEvent` override, fragile across DPI settings | Rejected — QGridLayout overlay is the standard Qt idiom |
| **Layout: keep splitter**, just change ratio | Minimal code change | Still wastes space; doesn't achieve "hover" effect | Rejected — doesn't match the desired layout |
| **Bug fix: call `mesh.Modified()`** after scalar update | Minimal change (one line) | Undocumented VTK behavior, may not update scalar bar range | Rejected — rebuilding PolyData is reliable and sub-ms |
| **Bug fix: set explicit `clim`** on `add_mesh` | Keeps the mesh object stable | Must track global min/max across all filter states; edge case when all zeros | Rejected — fragile, requires bookkeeping |
| **Cache: pickle** the entire `ExplorerData` | Simple, one-call save/load | Pickle is version-fragile, stores PyVista objects poorly | Rejected — `npz` with categorical encoding is robust |
| **Cache: parquet** for the DataFrame portion | Good compression, pandas-native | `ExplorerData` is arrays not a DataFrame; would need two files (parquet + npz) | Rejected — single `.npz` is simpler |
| **Keep Delaunay mesh** but also support point cloud | Flexibility for future surface mode | Extra computation and data model complexity for an unused feature | Rejected — point cloud is the requested rendering |

### Architecture Changes

**Modified dataclass** (`rf_explorer_data.py`):

```python
# Before
@dataclass
class ExplorerSessionData:
    forearm_mesh: pv.PolyData       # Delaunay mesh
    vertices: np.ndarray            # rotated (N, 3)
    tangent_rotation: np.ndarray    # (3, 3)

# After
@dataclass
class ExplorerSessionData:
    forearm_vertices: np.ndarray    # rotated (N, 3)
    tangent_rotation: np.ndarray    # (3, 3)
```

`ExplorerData` fields are unchanged — public API preserved.

**New private functions** (`rf_explorer_data.py`):

| Function | Purpose |
|----------|---------|
| `_explorer_cache_path(series_csv_path)` | Returns `{stem}_explorer_cache.npz` path |
| `_save_explorer_cache(series_csv_path, data)` | Saves ExplorerData arrays to `.npz` |
| `_load_explorer_cache(series_csv_path, forearm_ply_path)` | Loads from `.npz` if fresh, else returns `None` |

**Removed imports** (`rf_explorer_data.py`): `pyvista`, `build_delaunay_mesh`,
`mesh_to_pyvista`. The `trimesh` inline import (line 81) is also removed.

**GUI layout change** (`rf_feature_space_explorer.py`): `QSplitter` replaced by
`QGridLayout`. `self._mesh` renamed to `self._cloud`. `_render_3d()` and
`_apply_filter_update()` both build fresh `pv.PolyData` + `add_mesh()`.

**Reused existing code:**

| Function | File | Usage |
|----------|------|-------|
| `load_forearm_vertices()` | `rf_data_loader.py` | PLY loading with .npy sidecar (unchanged) |
| `compute_tangent_plane_rotation()` | `tangent_plane_alignment.py` | Face-on rotation (unchanged) |
| Point cloud rendering pattern | `rf_cluster_gallery_viewer.py:1006-1036` | Template for `pv.PolyData` + `add_mesh` |
| Sidecar mtime pattern | `rf_data_loader.py:29-63` | Template for cache freshness check |

---

## Implementation Plan

### Phase 1: Data Model — Remove Delaunay, Simplify ExplorerSessionData
**Goal:** Eliminate mesh construction from the data loading pipeline and
simplify the session data to raw vertices.

**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 1.1 — Update `ExplorerSessionData` dataclass: replace
  `forearm_mesh: pv.PolyData` + `vertices: np.ndarray` with single
  `forearm_vertices: np.ndarray`
- [x] Task 1.2 — Strip mesh-building from `load_explorer_data()`: remove
  `build_delaunay_mesh()`, `trimesh.Trimesh` construction, `mesh_to_pyvista()`
  calls, and `max_edge_mm` parameter. Keep tangent rotation, vertex rotation,
  KDTree query, and distance filter
- [x] Task 1.3 — Remove unused imports: `pyvista`, `build_delaunay_mesh`,
  `mesh_to_pyvista`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` — dataclass
  and loader function

**Dependencies:** None

### Phase 2: Sidecar Cache for Fast Loading
**Goal:** Add mtime-based `.npz` cache that skips CSV parsing and geometry
computation on cache hit.

**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 2.1 — Add `_explorer_cache_path()` helper returning
  `{stem}_explorer_cache.npz`
- [x] Task 2.2 — Add `_save_explorer_cache()`: encode gesture_types as
  categorical codes + labels, save all arrays via `np.savez_compressed`, wrap
  in try/except with `logger.warning` on failure
- [x] Task 2.3 — Add `_load_explorer_cache()`: check cache existence, validate
  mtime >= max(series CSV, forearm PLY), load arrays, reconstruct
  gesture_types from codes, validate shapes (raise `ValueError` on mismatch),
  return `ExplorerData` or `None`
- [x] Task 2.4 — Wire cache into `load_explorer_data()`: check cache at top,
  save cache at bottom

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` — three new
  private functions + two integration points in `load_explorer_data()`

**Dependencies:** Phase 1 (cache stores the new data shape)

### Phase 3: Layout — Full-Width 3D with Overlaid Scatter
**Goal:** Replace the side-by-side splitter layout with a full-width 3D view
and a small overlaid scatter at bottom-left.

**Completed:** 2026-05-03

- [x] Task 3.1 — Rewrite `_build_ui()`: replace `QSplitter` with
  `QGridLayout`, place 3D interactor and scatter canvas in same cell (0,0),
  scatter pinned `Qt.AlignBottom | Qt.AlignLeft` with `setFixedSize(350, 280)`,
  semi-transparent figure background
- [x] Task 3.2 — Update imports: add `QGridLayout`, remove `QSplitter`
- [x] Task 3.3 — Add 1st-99th percentile axis trimming in `_draw_scatter()`:
  `set_xlim` and `set_ylim` after plotting all series

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `_build_ui()`, `_draw_scatter()`, imports

**Dependencies:** None (parallel with Phases 1-2, same branch)

### Phase 4: Point Cloud Rendering
**Goal:** Render the forearm as individual vertex spheres instead of a
Delaunay surface.

**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 4.1 — Rename `self._mesh` to `self._cloud` in `__init__`,
  `_load_session`, and all references
- [x] Task 4.2 — Rewrite `_render_3d()`: create `pv.PolyData(vertices)`,
  assign spike_density scalar, `add_mesh` with `render_points_as_spheres=True,
  point_size=5`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `_render_3d()`, `__init__`, `_load_session`

**Dependencies:** Phase 1 (uses `forearm_vertices` instead of `forearm_mesh`)

### Phase 5: Bug Fix — RF Disappears After Checkbox Toggle
**Goal:** Fix the stale VTK mapper scalar range that causes the RF heatmap to
disappear permanently after any checkbox toggle.

**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 5.1 — Rewrite `_apply_filter_update()`: rebuild `pv.PolyData` with
  updated scalars, call `add_mesh(name="forearm")` instead of mutating the
  existing scalar array in-place

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `_apply_filter_update()`

**Dependencies:** Phase 4 (uses `self._cloud` and `forearm_vertices`)

---

## Testing Plan

### Manual Verification
- [ ] Launch explorer from DAG (`explore_rf_feature_space` task)
- [ ] 3D view fills the window; scatter is a small overlay at bottom-left
- [ ] Forearm renders as point spheres, not triangulated surface
- [ ] Scatter axes trim the 1% tails; main distribution fills the plot
- [ ] Drag filter rectangle — 3D heatmap updates in real time
- [ ] Uncheck all gesture types — RF clears (forearm visible, no spike color)
- [ ] Re-check all gesture types — RF reappears matching initial state
- [ ] Toggle individual gesture types in various orders — RF always correct
- [ ] Switch sessions — view resets correctly
- [ ] First load creates `*_explorer_cache.npz` alongside series CSV
- [ ] Second load is near-instant (check log for "loading explorer cache")
- [ ] Touch the series CSV, re-launch — cache is rebuilt

### Edge Cases
- [ ] Session with very few valid frames (<10) — renders without crash
- [ ] All checkboxes unchecked then rectangle dragged — no crash, empty RF
- [ ] Rectangle moved fully outside data range — empty heatmap, no crash
- [ ] Session with no spikes at all — point cloud visible, uniform color
- [ ] Corrupt or truncated `.npz` sidecar — detected, recomputed from source

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
  with new layout structure, point cloud rendering, and overlay details
- [ ] Update `docs/development/knowledge-base/note-rf-gui-comparison-gallery-vs-explorer.md`
  to reflect the layout and rendering changes

---

## Rollback Plan

All changes are confined to two files plus generated cache artifacts:

1. Revert changes to `rf_explorer_data.py` and
   `gui/rf_feature_space_explorer.py`
2. Delete any `*_explorer_cache.npz` files created by the sidecar cache

No migrations, no config changes, no breaking API changes. The
`launch_feature_space_explorer()` call site in `rf_cluster_pipeline.py` is
unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Matplotlib blitting breaks with transparent figure background | Medium | Low | Test on Windows; fallback to opaque background with margin |
| Mouse events on scatter leak to 3D plotter underneath | Low | Medium | Qt routes events to topmost widget by default; verify drag-rect works on overlay |
| Rebuilding PolyData on every filter update introduces lag | Low | Low | Sub-ms for 3k-10k vertices; 30ms debounce timer already throttles |
| Cache format becomes stale after code changes | Low | Low | User deletes `.npz` manually; could add a version field later |
| Semi-transparent scatter unreadable over bright heatmap areas | Medium | Low | Dark axes facecolor (0.1, 0.1, 0.1, 0.7) provides contrast |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data model | ~30 min | None |
| Phase 2: Sidecar cache | ~1 hour | Phase 1 |
| Phase 3: Layout overhaul | ~1.5 hours | None |
| Phase 4: Point cloud rendering | ~30 min | Phase 1 |
| Phase 5: Bug fix | ~30 min | Phase 4 |

---

## References

- Existing plan: `docs/development/plans/active/rf-feature-space-explorer.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-gui-comparison-gallery-vs-explorer.md`
- Gallery viewer point cloud pattern: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py:1006-1036`
- Sidecar cache pattern: `code/src/analysis/receptive_field_mapping/rf_data_loader.py:29-63`
- should_process_task: `code/src/utils/should_process_task.py`
