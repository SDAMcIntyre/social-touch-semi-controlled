> **Superseded** by [`postprocess-before-after-step-viewer`](../active/postprocess-before-after-step-viewer.md) (2026-04-13).
> That plan generalises the dual-plotter approach to all 4 postprocessing steps, not just projection.

---

# Plan: Projection Comparison Viewer

**Created:** 2026-04-13 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/projection-comparison-viewer`

---

## Overview

**What:** A new side-by-side 3D viewer that displays pre-projection and post-projection contact points on the forearm surface for each neuron/session block.
**Why:** The postprocessing pipeline's stage 4 (contact projection) snaps PCA-calibrated contact points to the nearest forearm vertex via KD-tree, but there is currently no way to visually inspect the before/after effect of this projection step.
**How:** A new `ProjectionComparisonViewer` class (PyQt5 + pyvistaqt) with two synchronized `QtInteractor` plotters, integrated into the existing postprocess visualization workflow as a new DAG task.

## Problem Statement

- The contact-projection step (stage 4) eliminates small spatial discrepancies by snapping every contact point to the nearest forearm vertex. This is a critical geometric operation.
- There is no visual tool to inspect how far contact points move during projection, or to spot anomalies where projection produces unexpected results (e.g. points snapping to the wrong forearm region).
- Researchers need to verify per-neuron that the projection is behaving correctly before downstream analysis (RF mapping, clustering) consumes the projected data.

## Goals

### In Scope
1. Side-by-side 3D viewer with synchronized camera control (rotate/zoom one side, the other follows)
2. Per-frame animation of contact points (same frame slider pattern as `PostprocessedSceneViewer`)
3. Forearm PLY displayed on both sides as static reference surface
4. Distinct contact point colours (cyan = pre-projection, red = post-projection)
5. Integration as a DAG task (`view_projection_comparison`) in the postprocess visualization workflow
6. NeuralDataPanel when `Nerve_freq` column is available
7. Visibility and point-size controls for forearm and contact points on each side

### Out of Scope
- Displacement statistics overlay (projection distances are already in `projection_stats.csv`)
- Comparison across sessions (this viewer is per-block, matching existing viewer pattern)
- Exporting comparison frames as images/video
- Modifications to the projection algorithm itself

## Success Criteria

- [ ] Viewer opens with two side-by-side 3D viewports for a given session block
- [ ] Left viewport shows pre-projection contact points (PCA-calibrated, not surface-snapped)
- [ ] Right viewport shows post-projection contact points (surface-snapped)
- [ ] Both viewports display the same forearm PLY as background
- [ ] Camera interaction in one viewport is mirrored in the other
- [ ] Frame slider animates both viewports simultaneously
- [ ] Viewer integrates into the DAG workflow and can be enabled/disabled via YAML config
- [ ] Viewer closes cleanly, allowing batch processing to continue to the next block

---

## Technical Design

### Approach

Create a new `ProjectionComparisonViewer(QMainWindow)` class following the same architecture as the existing `PostprocessedSceneViewer`. The viewer embeds two `pyvistaqt.QtInteractor` widgets side by side, each with its own static forearm PLY and dynamic contact-point PolyData. Camera synchronization uses VTK's `AddObserver` on interactor events with a re-entrancy guard to prevent infinite loops.

This approach was chosen because:
- PyQt5 + pyvistaqt is the established rendering stack in this project
- `QtInteractor` is a standard QWidget — two can coexist in a single `QHBoxLayout`
- VTK observer-based camera sync is efficient (event-driven, not polling)
- The existing viewer's actor-management pattern (register once, mutate in-place via `DeepCopy`/`mesh.points`) is proven and avoids VTK scene rebuild overhead

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two pyvistaqt QtInteractors in one QMainWindow | Proven Qt widget embedding; event-driven camera sync; consistent with existing codebase | Slightly more complex than single-plotter approaches | **Chosen** |
| Single plotter with two `pv.Plotter` sub-renderers | Built-in PyVista `shape=(1,2)` support | Camera sync is manual and less intuitive; subplot API is less flexible for Qt controls | Rejected — harder to integrate per-side visibility controls |
| Open3D split-screen (`visualize_point_cloud_comparison`) | Already exists in codebase | Static only (no per-frame animation); Open3D GUI doesn't integrate well with PyQt5 neural panel; inconsistent rendering stack | Rejected — no frame animation support |
| Matplotlib 3D side-by-side | Simple to implement | Too slow for interactive rotation; no smooth per-frame animation at 30fps | Rejected — performance |

### Architecture Changes

New module in existing GUI package:

```
code/src/postprocessing/gui/
├── __init__.py                          — add ProjectionComparisonViewer export
├── postprocessed_scene_viewer.py        — existing (unchanged)
└── projection_comparison_viewer.py      — NEW (~500 lines)
```

Integration points:
- `code/scripts/postprocess_visualization.py` — new path key + launcher function + batch call
- `configs/postprocess_visualization_dag.yaml` — new task definition

Reused internal imports (same as `PostprocessedSceneViewer`):
- `postprocessing.xyz_reference_from_gestures.calibration_pca_engine.CalibrationResult`
- `merging.gui.neural_kinect_scene_viewer.NeuralDataPanel`

### Key Data Paths

| Data | Source Directory | Stage |
|------|-----------------|-------|
| Pre-projection CSV | `blocks_pca_calibrated/` | Stage 2 output |
| Post-projection CSV | `blocks_contact_projected/` | Stage 4 output |
| Forearm PLY | `forearm_pca_calibrated/` | Stage 3 output |

Both CSVs share the same filename: `{session_id}_semicontrolled_{block_id}_merged_data_pca-xyz.csv`

---

## Implementation Plan

### Phase 1: Config + Path Resolution
**Goal:** Wire up the new task in the DAG config and resolve the pre-projection CSV path.
**Started:** —
**Completed:** —

- [ ] Add `view_projection_comparison` task to `configs/postprocess_visualization_dag.yaml` (disabled by default)
- [ ] Add `pca_calibrated_csv` key to `resolve_postprocessed_paths()` in `postprocess_visualization.py`

**Files Modified:**
- `configs/postprocess_visualization_dag.yaml` — add new task block
- `code/scripts/postprocess_visualization.py` — add one path key to return dict

**Dependencies:** None

### Phase 2: Core Viewer Class
**Goal:** Implement the dual-plotter viewer with camera sync, frame animation, and controls.
**Started:** —
**Completed:** —

- [ ] Create `ProjectionComparisonViewer(QMainWindow)` in `projection_comparison_viewer.py`
- [ ] Implement data loading: load both CSVs, filter kinect rows, pre-parse contact points per frame
- [ ] Implement forearm PLY loading (Open3D -> PyVista, same pattern as existing viewer)
- [ ] Build UI: two `QtInteractor` widgets in `QHBoxLayout` with labels, right-panel controls, frame slider, neural panel
- [ ] Implement `_init_actors()`: register forearm PLY + dynamic contact PolyData on each plotter
- [ ] Implement `_update_frame()`: DeepCopy contact points, render both plotters, update cursor
- [ ] Implement VTK observer camera sync with `_syncing` re-entrancy guard (InteractionEvent, EndInteractionEvent, MouseWheelForwardEvent, MouseWheelBackwardEvent)
- [ ] Implement PCA-aware camera positioning (`_compute_camera_params`)
- [ ] Implement deferred render pattern (`showEvent` + `QTimer.singleShot`)
- [ ] Implement drag-throttle timer (80ms) and play timer (33ms, ~30fps)

**Files Modified:**
- `code/src/postprocessing/gui/projection_comparison_viewer.py` — new file (~500 lines)

**Dependencies:** Phase 1

### Phase 3: Integration
**Goal:** Connect the viewer to the workflow script and export from the package.
**Started:** —
**Completed:** —

- [ ] Add `ProjectionComparisonViewer` to `postprocessing/gui/__init__.py` exports
- [ ] Add `run_single_session_pipeline_projection_comparison()` launcher in `postprocess_visualization.py`
- [ ] Add launcher call in `run_batch_sequentially()` loop
- [ ] Import `ProjectionComparisonViewer` at top of `postprocess_visualization.py`

**Files Modified:**
- `code/src/postprocessing/gui/__init__.py` — add export
- `code/scripts/postprocess_visualization.py` — add launcher function + batch call + import

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Enable `view_projection_comparison` in DAG config, run `postprocess_visualization.py` — viewer opens
- [ ] Left viewport shows cyan contact points floating near (but not exactly on) the forearm surface
- [ ] Right viewport shows red contact points snapped to the forearm surface vertices
- [ ] Frame slider animates both viewports simultaneously
- [ ] Rotating/zooming in left viewport mirrors to right viewport (and vice versa)
- [ ] Recenter button resets both cameras to the PCA-aware default view
- [ ] Play/Pause button animates at ~30fps and can be stopped
- [ ] NeuralDataPanel cursor tracks frame position (when `Nerve_freq` column present)
- [ ] Visibility checkboxes toggle forearm/contacts on each side independently
- [ ] Point-size sliders adjust rendering on the correct plotter
- [ ] Closing the viewer allows the batch to proceed to the next block

### Edge Cases
- [ ] Session with no contact points (empty `contact_points` column) — viewer shows only forearm
- [ ] Missing pre-projection CSV — viewer skips with a warning message
- [ ] Missing forearm PLY — viewer skips with a warning message
- [ ] Single-frame block — slider is at 0, no animation possible, viewer still renders

---

## Documentation Plan

- [ ] Update inline docstrings in new module
- [ ] Update `postprocess_visualization_dag.yaml` comments with new task description

---

## Rollback Plan

All changes are additive (new file + small edits to existing files). Rollback:
1. Delete `code/src/postprocessing/gui/projection_comparison_viewer.py`
2. Revert edits to `__init__.py`, `postprocess_visualization.py`, and DAG config
3. No data migrations or breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Two QtInteractors cause VTK context conflicts | Low | High | Proven by PyVista docs — multiple QtInteractors in one window is supported; test early |
| Camera sync infinite loop via VTK observers | Medium | Medium | Re-entrancy guard (`_syncing` boolean); same pattern used in Open3D comparison viewer |
| Pre/post CSV frame count mismatch | Low | Low | Use `min()` of both kinect row counts; should not happen in practice (same block, same pipeline) |
| Memory overhead from loading two CSVs per block | Low | Low | Typical block CSVs are small (few thousand rows); negligible compared to forearm PLY |

---

## References

- Existing viewer: `code/src/postprocessing/gui/postprocessed_scene_viewer.py`
- Projection step: `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`
- Open3D comparison: `code/src/utils/gui/visualize_point_cloud_comparison.py`
- Knowledge base: `docs/development/knowledge-base/note-open3d-scenewidget-layout.md` (layout patterns)
