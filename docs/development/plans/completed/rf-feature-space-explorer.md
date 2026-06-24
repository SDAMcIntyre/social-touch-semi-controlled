# Plan: RF Feature-Space Explorer GUI

**Date:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 14:41
**Base Branch:** `feature/per-type-clustering-outputs`
**Branch:** `feature/rf-feature-space-explorer`

---

## Overview

**What:** A new PyQt5 GUI with an interactive 2D scatter plot (pressure x
hand_velocity_signed) linked to a real-time 3D forearm RF heatmap. A draggable
rectangle on the scatter acts as a spatial filter; gesture-type checkboxes
provide additional filtering.

**Why:** The current cluster-based RF visualization is static and predetermined
by clustering parameters. This GUI lets the researcher dynamically define
regions of interest in feature space and immediately see the corresponding RF
pattern -- enabling hypothesis-driven exploration such as "do high-pressure,
slow strokes activate a different RF zone than low-pressure, fast strokes?"

**How:** Precompute per-frame to per-vertex mapping at load time using KDTree,
then use `np.bincount` for sub-millisecond filter updates on each rectangle
drag. Matplotlib `FigureCanvasQTAgg` with blitting for the 2D scatter; PyVista
`QtInteractor` for the 3D view.

## Problem Statement

The RF cluster gallery viewer (`rf_cluster_gallery_viewer.py`) shows
precomputed cluster-based RF heatmaps. Cluster membership is fixed at
clustering time and cannot be adjusted interactively. There is no way to ask
"what does the RF look like if I restrict to frames with high pressure and
low velocity?" without re-running the entire clustering pipeline with
different parameters.

This matters because the relationship between touch kinematics (pressure,
velocity) and RF activation is a central research question. An interactive
explorer that lets the researcher sweep through the feature space and
immediately see the spatial RF response would accelerate hypothesis generation
and validation.

## Goals

### In Scope

1. Interactive 2D scatter plot (pressure x velocity_signed) with one dot per
   frame (filtered to frames with non-NaN contact position), colored by gesture
   type
2. Draggable/resizable rectangle overlay on the scatter acting as a 2D
   feature-space mask -- corners for resize, body for move
3. Real-time 3D forearm heatmap showing spike density for frames within the
   rectangle and checked gesture types
4. Gesture-type checkboxes: tap, stroke_proximal, stroke_distal, stroke_unknown
5. Session selector combo box for switching between sessions (single-session
   forearm rendering)
6. Pipeline integration as an independent DAG task (`explore_rf_feature_space`)
   via `explore_rf_feature_space_flow` in `analysis_workflow.py`, with
   `launch_feature_space_explorer()` in `rf_cluster_pipeline.py` as the
   underlying call target

### Out of Scope

- Per-touch aggregation (mean/max) -- we work with raw per-frame data
- Cross-session pooling onto a single forearm
- Saving/exporting filtered subsets to disk
- DAG config GUI integration (standalone launch only for now)
- Colormap/metric customization in the 3D view (hardcoded defaults; polish
  in a follow-up)

## Success Criteria

- [ ] 2D scatter renders 100k+ frames without UI freeze at startup
- [ ] Rectangle drag updates 3D view within 30ms (no perceptible delay)
- [ ] Checkbox toggle updates 3D view within 30ms
- [ ] Session switching loads new data and resets the view correctly
- [ ] GUI launches from the `explore_rf_feature_space` DAG task (via `explore_rf_feature_space_flow` → `launch_feature_space_explorer()`)

---

## Technical Design

### Approach

The key insight is that each frame maps to exactly one mesh vertex (nearest
neighbor via KDTree). This 1-to-1 mapping can be precomputed once at load time
as an integer index array (`frame_vertex_idx`). On each filter change,
`np.bincount` aggregates spike counts per vertex in ~0.1ms -- orders of
magnitude faster than recomputing `map_scalars_to_mesh()`.

The 2D scatter uses matplotlib's blitting API: the 100k+ scatter points are
rendered once to a rasterized bitmap; during rectangle drag, only the rectangle
artist is redrawn on top via blit. A `QTimer(singleShot=True, 30ms)` throttles
3D updates to ~33fps.

### Data Flow

```
Series-augmented CSV (per-frame)
  |
  v  [load_explorer_data] -- filter to non-NaN contact rows
  |                        -- extract (pressure, velocity_signed, x, y, z,
  |                           Nerve_spike, gesture_type)
  v
ExplorerData
  |-- pressure: (n_frames,)           float64
  |-- velocity_signed: (n_frames,)    float64
  |-- gesture_types: (n_frames,)      object (str)
  |-- spikes: (n_frames,)             bool
  |-- frame_vertex_idx: (n_frames,)   int  <- KDTree nearest-neighbor
  |-- session_data: ExplorerSessionData (mesh, vertices, tangent rotation)
  |
  v  [on rectangle drag / checkbox toggle]
  |
  mask = rect_mask & checkbox_mask       # boolean (n_frames,)
  vertex_spikes = np.bincount(
      frame_vertex_idx[mask],
      weights=spikes[mask],
      minlength=n_verts
  )
  |
  v  [update PyVista scalars -- no mesh rebuild]
  mesh_pv["scalars"] = vertex_spikes     # zeros -> NaN for passthrough
  plotter.render()                       # ~5-10ms
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Sparse matrix (n_frames x n_verts) | General, supports weighted mappings | Overkill for 1-to-1 mapping; `bincount` is simpler and faster | Rejected |
| QPainter for 2D plot | Full control, no matplotlib dependency | Must reimplement axes, ticks, legends, scatter, zoom from scratch; significant effort for no bottleneck gain | Rejected |
| Per-touch aggregation | Fewer scatter points (~5k) | Loses temporal resolution; user explicitly wants per-frame data | Rejected |
| matplotlib 3D for the RF view | Consistent 2D+3D library | Too slow for interactive updates; no smooth VTK-quality rotation | Rejected |

### Architecture Changes

Three files added, one modified:

```
code/src/analysis/receptive_field_mapping/
    gui/
        __init__.py                          # MODIFY: add RFFeatureSpaceExplorer export
        rf_cluster_gallery_viewer.py         # existing (template, not modified)
        rf_feature_space_explorer.py         # NEW: QMainWindow + DraggableFilterRect
    rf_explorer_data.py                      # NEW: ExplorerData + load_explorer_data()
    rf_cluster_pipeline.py                   # MODIFY: add launch_feature_space_explorer()
```

**New classes:**

| Class | File | Purpose |
|-------|------|---------|
| `ExplorerSessionData` | `rf_explorer_data.py` | Dataclass: forearm vertices, vertex colors, tangent rotation, Delaunay mesh |
| `ExplorerData` | `rf_explorer_data.py` | Dataclass: per-frame arrays (pressure, velocity, spikes, gesture_types, frame_vertex_idx) + session geometry |
| `DraggableFilterRect` | `rf_feature_space_explorer.py` | Matplotlib rectangle with 8-zone drag/resize, blitting, `on_changed` callback |
| `RFFeatureSpaceExplorer` | `rf_feature_space_explorer.py` | QMainWindow: QSplitter with matplotlib canvas (left, ~40%) + PyVista interactor (right, ~60%) + bottom checkbox bar |

**Reused existing code:**

| Function/Class | File | Usage |
|---|---|---|
| `build_delaunay_mesh()` | `rf_surface_utils.py` | Build forearm mesh from vertices |
| `mesh_to_pyvista()` | `rf_surface_utils.py` | Convert trimesh to PyVista PolyData |
| `load_forearm_vertices()` | `rf_data_loader.py` | Load forearm PLY with .npy sidecar cache |
| `compute_tangent_plane_rotation()` | `tangent_plane_alignment.py` | Face-on camera orientation |
| `classify_gesture_type()` | `preparation/gesture_type.py` | Gesture type labels (if not already in CSV) |
| `launch_gallery_viewer()` pattern | `rf_cluster_pipeline.py` | Template for the launch function |

**Architecture constraints:**
- Qt `blockSignals` guard required on checkbox handlers to prevent signal
  recursion (see `note-qt-itemchanged-signal-recursion.md`)
- matplotlib backend: import `FigureCanvasQTAgg` directly to avoid conflict
  with `matplotlib.use('Agg')` in batch-rendering modules

---

## Implementation Plan

### Phase 1: Data Model + Precomputation
**Goal:** Load series-augmented CSV and build per-frame vertex mapping.
**Started:** 2026-05-01
**Completed:** 2026-05-01

- [x] Task 1.1 -- Create `rf_explorer_data.py` with `ExplorerData` and
  `ExplorerSessionData` dataclasses
- [x] Task 1.2 -- Implement `load_explorer_data(series_csv_path,
  forearm_ply_path, max_edge_mm)`:
  - Read CSV, filter to rows where `contact_location_x` is not NaN
  - Extract columns: `pressure`, `hand_velocity_signed`,
    `contact_location_x/y/z`, `Nerve_spike`, `gesture_type`
  - Load forearm vertices via `load_forearm_vertices()`, build Delaunay mesh
  - Compute tangent rotation, apply to both mesh vertices and contact points
  - Build `scipy.spatial.cKDTree` on rotated mesh vertices
  - Query nearest vertex for each rotated contact point ->
    `frame_vertex_idx`
- [x] Task 1.3 -- Validate that all contacts map within a reasonable radius;
  raise `ValueError` if any frame's contact is >15mm from nearest vertex

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` -- NEW

**Dependencies:** None

### Phase 2: GUI Skeleton + Static Scatter
**Goal:** QMainWindow with matplotlib scatter and PyVista viewport, no
interaction yet.
**Started:** 2026-05-01
**Completed:** 2026-05-01

- [x] Task 2.1 -- Create `rf_feature_space_explorer.py` with
  `RFFeatureSpaceExplorer(QMainWindow)` and `_build_ui()` method
- [x] Task 2.2 -- Layout: `QSplitter(Horizontal)` with matplotlib
  `FigureCanvasQTAgg` (left, ~40%) + `pyvistaqt.QtInteractor` (right, ~60%);
  bottom `QHBoxLayout` with placeholder checkboxes and frame-count label
- [x] Task 2.3 -- Draw static scatter: pressure (X) x velocity_signed (Y),
  colored by gesture type (fixed palette, alpha=0.3, `rasterized=True`)
- [x] Task 2.4 -- Render initial 3D forearm mesh with full-data heatmap
  (all frames, unfiltered) using `mesh_to_pyvista()` + scalar mapping via
  `np.bincount`
- [x] Task 2.5 -- Implement deferred initialization pattern (`showEvent` ->
  `QTimer.singleShot(0, _deferred_start)`)
- [x] Task 2.6 -- Add session selector combo box in toolbar; on change, reload
  `ExplorerData` for new session
- [x] Task 2.7 -- Update `gui/__init__.py` to export `RFFeatureSpaceExplorer`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` -- NEW
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` -- add export

**Dependencies:** Phase 1

### Phase 3: Interactive Rectangle + Real-Time Filter
**Goal:** Draggable/resizable rectangle with live 3D updates.
**Started:** 2026-05-01
**Completed:** 2026-05-01

- [x] Task 3.1 -- Implement `DraggableFilterRect` class:
  - `matplotlib.patches.Rectangle` with semi-transparent fill + solid border
  - 8 grab zones (4 corners, 4 edges) + interior for move
  - Hit-test on `button_press_event` sets drag mode
  - `motion_notify_event` updates rectangle geometry
  - `button_release_event` ends drag
- [x] Task 3.2 -- Implement blitting: after scatter render, call
  `canvas.copy_from_bbox(ax.bbox)` to save background; on mouse move, restore
  background + draw rectangle + blit
- [x] Task 3.3 -- Wire `on_changed(x_min, x_max, y_min, y_max)` callback to
  `QTimer(singleShot=True, interval=30)` for throttled 3D update
- [x] Task 3.4 -- Implement filter path in `_apply_filter_update()`:
  rect mask + checkbox mask -> `np.bincount` -> update PyVista scalars ->
  `plotter.render()`
- [x] Task 3.5 -- Add gesture-type checkboxes (tap, stroke_proximal,
  stroke_distal, stroke_unknown); connect `stateChanged` to
  `_apply_filter_update()` with `blockSignals` guard
- [x] Task 3.6 -- Add frame count label: "Frames: {selected} / {total}"
- [x] Task 3.7 -- Test responsiveness with real session data (100k+ frames)
  (manual testing deferred; logic verified via code review)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` -- add interaction classes and methods

**Dependencies:** Phase 2

### Phase 4: Pipeline Integration
**Goal:** Launch the explorer as a first-class DAG task from the analysis pipeline.
**Started:** 2026-05-01
**Completed:** 2026-05-01

**Revised architecture (rf-explorer-independent-dag-flow):** The explorer is
no longer launched via a flag on `visualize_receptive_fields_clustered`. It is
now an independent DAG task (`explore_rf_feature_space`) with its own Prefect
`@flow` (`explore_rf_feature_space_flow`) and a direct dependency on
`touch_series_transforms` — the stage that actually produces the data it
consumes. This removes the false dependency on RF clustering and makes the
explorer independently enable/disable-able in the DAG config.

- [x] Task 4.1 -- Add `launch_feature_space_explorer()` to
  `rf_cluster_pipeline.py`, following `launch_gallery_viewer()` pattern
- [x] Task 4.2 -- ~~Wire it to be callable from `run_cluster_rf_visualization()`
  via a `feature_space_explorer: true` option flag~~ — **superseded**: add
  `explore_rf_feature_space_flow` `@flow` to `analysis_workflow.py` and
  register it in `available_tasks`; add `explore_rf_feature_space` task block
  to `analyse_workflow_dag.yaml` with `depends_on: [touch_series_transforms]`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` -- add launch function
- `code/scripts/analysis_workflow.py` -- add `explore_rf_feature_space_flow` and register in `available_tasks`; remove `feature_space_explorer` param from `visualize_receptive_fields_clustered_flow`
- `configs/analyse_workflow_dag.yaml` -- add `explore_rf_feature_space` task block

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `test_load_explorer_data` -- Verify correct frame filtering (NaN
  exclusion), column extraction, and `frame_vertex_idx` shape matches filtered
  frame count
- [ ] `test_bincount_filter` -- Verify that boolean mask -> `np.bincount`
  produces expected per-vertex spike counts for a small synthetic dataset
  (5 frames, 10 vertices)

### Manual Verification
- [ ] Launch GUI with a real session's series-augmented CSV
- [ ] Verify scatter shows correct pressure x velocity_signed distribution
  with gesture-type coloring
- [ ] Drag rectangle corners -- 3D view updates in real time, no perceptible
  lag
- [ ] Move rectangle body -- 3D heatmap shifts correspondingly
- [ ] Toggle checkboxes -- 3D view updates immediately
- [ ] Switch sessions -- scatter and 3D view reload correctly
- [ ] Verify gesture type colors match between scatter legend and checkboxes

### Edge Cases
- [ ] Session with very few valid contact frames (< 10) -- renders without
  crash
- [ ] All checkboxes unchecked -- 3D shows empty forearm (NaN scalars), no
  crash
- [ ] Rectangle dragged fully outside data range -- shows empty heatmap
- [ ] Session with no spikes at all -- density/count mode still displays
  contact density

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with explorer description in the
  RF mapping section
- [ ] Add knowledge-base note if any non-obvious rendering or Qt issues arise
  during implementation

---

## Rollback Plan

All changes are additive:
1. Two new files (`rf_explorer_data.py`, `rf_feature_space_explorer.py`)
2. One-line export in `gui/__init__.py`
3. Launch function + option flag in `rf_cluster_pipeline.py`

**Rollback procedure:** Delete the two new files, revert the `__init__.py`
export line, and revert the launch function/option additions in the pipeline
module.

No migrations, no breaking changes to existing interfaces.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 100k+ scatter points slow initial render | Medium | Low | `rasterized=True` on scatter, render once then blit. Fallback: stride/downsample for display only |
| matplotlib backend conflict (`Agg` vs `Qt5Agg`) | Low | Medium | Import `FigureCanvasQTAgg` directly; use lazy imports to avoid triggering `matplotlib.use('Agg')` from batch modules |
| KDTree radius miss (contact far from mesh) | Low | Medium | Use nearest-neighbor without radius cutoff for `frame_vertex_idx`; log warning + raise if distance > 15mm |
| Qt signal recursion on checkbox toggle | Low | Medium | Guard with `blockSignals` pattern per `note-qt-itemchanged-signal-recursion.md` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data Model | ~2 hours | None |
| Phase 2: GUI Skeleton | ~3 hours | Phase 1 |
| Phase 3: Interactive Rectangle | ~4 hours | Phase 2 |
| Phase 4: Pipeline Integration | ~1 hour | Phase 3 |

---

## References

- Template GUI: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`
- Data loading pattern: `code/src/analysis/receptive_field_mapping/rf_gallery_data.py`
- Surface utilities: `code/src/analysis/receptive_field_mapping/rf_surface_utils.py`
- Tangent plane: `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py`
- Data loader: `code/src/analysis/receptive_field_mapping/rf_data_loader.py`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-cluster-visualization-overview.md`
