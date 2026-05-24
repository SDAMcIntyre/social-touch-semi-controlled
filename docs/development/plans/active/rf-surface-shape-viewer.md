# Plan: RF Surface Shape Viewer

**Date:** 2026-05-19
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/population-rf-vertex-data-export`
**Branch:** `feature/rf-surface-shape-viewer`

---

## Overview

**What:** An interactive 3D surface viewer for population receptive field data,
where the Z axis encodes mean IFF to expose RF topography (peaks, shoulders,
asymmetry).
**Why:** The existing 2D heatmap renders flatten the RF response into colour
alone; a 3D surface lets the researcher rotate and inspect the spatial shape of
the RF, revealing features that flat colour cannot convey.
**How:** Load the per-session `_rf_population_vertex_data.npz` (already
produced by `visualize_population_rf_maps`), build a
`pv.StructuredGrid(grid_u, grid_v, grid_z)` surface in PyVista, and present it
in a `QMainWindow` with session and gesture-type dropdown navigation.

## Problem Statement

Population RF maps are currently rendered as 2D heatmaps (PNGs) using
matplotlib. While these are useful for publication-quality figures, they
collapse the IFF response axis into colour, making it difficult to:

- Judge the relative height and steepness of the RF peak.
- Detect asymmetry or secondary shoulders in the response profile.
- Compare RF shapes across gesture types by rotating the view.

An interactive 3D surface viewer addresses all three by mapping mean IFF to the
vertical axis, producing a topographic landscape that can be freely rotated and
zoomed.

## Goals

### In Scope
1. 3D surface rendering of the interpolated RF grid (150x150) with Z = mean IFF.
2. Session dropdown to navigate between sessions (one NPZ per session).
3. Gesture-type dropdown that repopulates per session (e.g. `all`, `tap`,
   `stroke_proximal`, `stroke_distal`).
4. `jet` colormap with scalar bar, consistent with existing RF population maps.
5. Axis labels: U (mm), V (mm), Mean IFF (Hz).
6. DAG integration as a viewer task (`explore_rf_surface`).

### Out of Scope
- Overlaying the inflection boundary contour on the 3D surface (follow-up).
- Side-by-side comparison of multiple gesture types in split viewports.
- Exporting the 3D surface as a mesh file (OBJ/STL).
- Camera persistence across sessions (can be added later).

## Success Criteria

- [ ] Viewer launches from the DAG pipeline and displays a 3D surface for each
      session that has a `_rf_population_vertex_data.npz`.
- [ ] Session dropdown lists all loaded sessions; switching sessions loads the
      new NPZ and re-renders.
- [ ] Gesture-type dropdown repopulates on session change; switching gesture
      renders the corresponding surface.
- [ ] NaN regions (outside the forearm mesh) are not rendered — only the
      contacted surface is visible.
- [ ] Surface is coloured by mean IFF with a `jet` colormap and scalar bar.
- [ ] Viewer raises `ValueError` if an NPZ file is missing or has missing keys.

---

## Technical Design

### Approach

Build `RFSurfaceViewer(QMainWindow)` following the established PyQt5 + PyVista
viewer pattern used by `TouchPopulationExplorer` and
`RFClusterGalleryViewer`. Use `pv.StructuredGrid` for the surface — it
natively supports regular grids (the 150x150 meshgrid) and renders NaN
vertices as invisible, which handles the outside-mesh masking without explicit
cell removal.

Data is loaded from the already-exported NPZ files; no recomputation of
heatmaps or interpolation is needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `pv.StructuredGrid` from meshgrid | Direct fit to 150x150 regular grid; NaN vertices invisible; supports scalars + colormap | Cannot represent irregular meshes | **Chosen** — grid data is regular |
| `pv.PolyData` triangulated surface from per-vertex heatmap | Uses the actual forearm mesh topology | Requires re-triangulating the UV-space mesh; more complex; the vertex heatmap has NaN/-1 sentinels to filter | Rejected |
| Matplotlib `plot_surface` | No PyVista dependency | Non-interactive (no free rotation); no consistency with other viewers | Rejected |

### Architecture Changes

**New module:**
```
code/src/analysis/receptive_field_mapping/gui/rf_surface_viewer.py
```

Contains:
- `_load_vertex_data(npz_path: Path) -> dict` — loads NPZ, validates keys,
  returns `{session_id, gesture_types, grids: {gtype: (grid_u, grid_v, grid_z)}}`.
- `RFSurfaceViewer(QMainWindow)` — the viewer class.

**Modified modules:**
- `gui/__init__.py` — add `RFSurfaceViewer` to `__all__`.
- `rf_cluster_pipeline.py` — add `launch_rf_surface_viewer()` launcher.
- `analysis_workflow.py` — add `explore_rf_surface_flow` + registration.
- `analyse_workflow_viewers_dag.yaml` — add `explore_rf_surface` task.

**Integration points:**
- NPZ files produced by `rf_population_map_pipeline.py::_save_vertex_data_npz`
  — the viewer is a pure consumer of these files.
- Session resolution via `_resolve_explorer_session_paths()` (already used by
  `launch_touch_population_explorer`).
- `QApplication` lifecycle follows the `app.exec_()` pattern from existing
  launchers.

---

## Implementation Plan

### Phase 1: Viewer class + data loading
**Started:** 2026-05-19
**Completed:** 2026-05-19
**Goal:** Build the `RFSurfaceViewer` class with NPZ loading, StructuredGrid
construction, and two-tier dropdown navigation.

- [x] Task 1.1 — Implement `_load_vertex_data()` helper: open NPZ, validate
      required keys (`gesture_types`, `grid_u_{g}`, `grid_v_{g}`, `grid_z_{g}`
      for each gesture type), return structured dict. Raise `ValueError` on
      missing keys.
- [x] Task 1.2 — Implement `RFSurfaceViewer.__init__` and `_build_ui()`:
      QToolBar with Session and Gesture Type `QComboBox` dropdowns, central
      `QtInteractor` plotter.
- [x] Task 1.3 — Implement `_build_surface(grid_u, grid_v, grid_z)`:
      construct `pv.StructuredGrid`, assign IFF scalars, add to plotter with
      `jet` colormap and scalar bar.
- [x] Task 1.4 — Implement `_on_session_changed` / `_on_gesture_changed`
      cascade: block signals during gesture combo repopulation, call
      `plotter.clear()` + `_build_surface()` on change.
- [x] Task 1.5 — Implement deferred init (`showEvent` →
      `QTimer.singleShot(0, _deferred_start)`) and `closeEvent`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_surface_viewer.py` — **new file** (~200 lines)

**Dependencies:** None

### Phase 2: Pipeline integration
**Started:** 2026-05-19
**Completed:** 2026-05-19
**Goal:** Wire the viewer into the DAG and analysis workflow so it can be
launched from the pipeline GUI.

- [x] Task 2.1 — Add `launch_rf_surface_viewer(input_items, neuron_mode)` to
      `rf_cluster_pipeline.py`: resolve session paths, locate NPZ files
      (raise if none found), load data, create `RFSurfaceViewer`, call
      `app.exec_()`.
- [x] Task 2.2 — Add `explore_rf_surface_flow` to `analysis_workflow.py`:
      `@flow` function delegating to `launch_rf_surface_viewer`.
- [x] Task 2.3 — Register `explore_rf_surface_flow` in the `available_tasks`
      list and add `neuron_mode` option forwarding in the task dispatch block.
- [x] Task 2.4 — Add `explore_rf_surface` entry to
      `configs/analyse_workflow_viewers_dag.yaml` with `category: viewer`,
      `enabled: false`, `neuron_mode: iff`, `depends_on: []`.
- [x] Task 2.5 — Add `RFSurfaceViewer` to `gui/__init__.py` imports and
      `__all__`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add launcher (~50 lines)
- `code/scripts/analysis_workflow.py` — add flow + registration (~15 lines)
- `configs/analyse_workflow_viewers_dag.yaml` — add task entry (~7 lines)
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — add import + `__all__` entry

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Enable `explore_rf_surface: enabled: true` in the viewer DAG config,
      run the analysis workflow — viewer launches and displays a 3D surface.
- [ ] Switch sessions via dropdown — surface updates to the new session's data.
- [ ] Switch gesture types — surface updates to the selected gesture.
- [ ] Rotate, zoom, pan the 3D view — interaction is responsive.
- [ ] NaN regions (outside mesh outline) are invisible — only the RF region
      renders.
- [ ] Scalar bar shows correct IFF range and `jet` colourmap.

### Edge Cases
- [ ] Session with only one gesture type — gesture combo shows one entry,
      no crash.
- [ ] Session with missing NPZ — `ValueError` raised with clear message
      (not a silent empty window).
- [ ] All grid values NaN for a gesture — surface is empty, no crash.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `explore_rf_surface` to the
      viewer list and the DAG task table.

---

## Rollback Plan

1. Revert the commits on the feature branch.
2. Remove the `explore_rf_surface` entry from the DAG YAML.
3. No data migration — the viewer is read-only and produces no output files.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NaN handling in StructuredGrid produces visual artifacts (holes, spikes) | Med | Low | Test with real data; fall back to `threshold` filter to remove NaN cells explicitly if needed |
| 150x150 grid too coarse for smooth 3D surface | Low | Low | Acceptable for shape inspection; grid resolution can be increased in the upstream pipeline later |
| Camera resets on gesture change is jarring | Low | Low | Acceptable for initial version; camera preservation can be added as a follow-up by saving/restoring `plotter.camera_position` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~200 lines, ~1 hour | None |
| Phase 2 | ~70 lines, ~30 min | Phase 1 |

---

## References

- Related Plans: `docs/development/plans/pending/population-rf-vertex-data-export.md` (predecessor — produced the NPZ data)
- Key data source: `rf_population_map_pipeline.py::_save_vertex_data_npz` (NPZ schema)
- Pattern reference: `gui/touch_population_explorer.py` (session dropdown cascade)
- Knowledge base: `note-qt-itemchanged-signal-recursion.md` (signal blocking pattern)
