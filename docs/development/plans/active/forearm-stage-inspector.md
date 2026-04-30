# Plan: Forearm Pointcloud Stage Inspector

**Date:** 2026-04-28
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/gmm-clusterer`

---

## Overview

**What:** A single-window PyQt5 + PyVista 3D viewer that lets the user inspect a forearm pointcloud at each postprocessing stage, switch between sessions via dropdown, and optionally overlay a 2.5D Delaunay triangulation mesh.

**Why:** The analysis pipeline previously used 2.5D Delaunay triangulation for forearm meshing but the results were incorrect. There is no tool to visually inspect the forearm geometry at each postprocessing stage — researchers need to see vertex colors, point distribution, and how the Delaunay mesh looks at each coordinate-space transform to diagnose issues.

**How:** A new `ForearmStageInspector(QMainWindow)` class with two `QComboBox` dropdowns (session, stage), one `QCheckBox` (Delaunay toggle), and a single `pyvistaqt.QtInteractor`. Integrated into the existing `postprocess_visualization.py` workflow as a new DAG task.

## Problem Statement

The postprocessing pipeline transforms forearm pointclouds through three stages (Raw/Unified → PCA-Calibrated → RF-Centered), but there is no tool to inspect the forearm geometry at each stage in a single viewer. The 2.5D Delaunay triangulation previously used in the analysis pipeline produced incorrect meshes, and there is no way to visually compare how the triangulation behaves at each stage to understand why.

## Goals

### In Scope
1. Single-window 3D viewer with session and stage dropdown selectors
2. Display forearm pointclouds with their PLY vertex colors (not uniform color)
3. ON/OFF checkbox to toggle 2.5D Delaunay triangulation mesh overlay
4. Integration as a DAG task in the postprocess visualization pipeline

### Out of Scope
- Contact point overlay or neural data panels
- Modifying any processing algorithm
- BPA mesh comparison (only 2.5D Delaunay)
- Per-video (per-block) forearm inspection — this viewer is session-level

## Success Criteria

- [ ] Viewer opens with two dropdowns (session, stage) and a Delaunay checkbox
- [ ] Session dropdown lists all sessions from the DAG config
- [ ] Stage dropdown shows: Raw (Unified/Registered), PCA-Calibrated, RF-Centered
- [ ] Forearm pointcloud renders with PLY vertex colors
- [ ] Delaunay ON: semi-transparent mesh overlay with vertex colors appears
- [ ] Delaunay OFF: only the pointcloud is shown
- [ ] Missing stage PLY shows "No PLY found" message (no crash)
- [ ] Camera is preserved when switching stages within the same session

---

## Technical Design

### Forearm PLY Stages and Paths

| Stage | Path | Source |
|-------|------|--------|
| Raw (Unified/Registered) | `{session_processed_output_dir}/forearm_pointclouds/{session_id}_unified_registered.ply` (fallback: first `*.ply`) | `_load_unified_forearm()` in `postprocess_visualization.py` |
| PCA-Calibrated | `{session_merged_output_dir}/forearm_pca_calibrated/{session_id}_forearm.ply` | Postprocessing stage 3 output |
| RF-Centered | `{session_merged_output_dir}/forearm_rf_centered/{session_id}_forearm.ply` | Postprocessing stage 5 output |

### Approach

**Single-window with lazy loading.** Unlike `BeforeAfterStepViewer` (one window per block), this viewer opens once and lets the user navigate via dropdowns. All session stage paths are pre-discovered at startup, but PLY data is loaded lazily on selection change and cached.

**`plotter.clear()` + rebuild pattern** (same as `RFClusterGalleryViewer`). Stage/session switches are infrequent (not 30fps animation), so clear-and-rebuild is simpler than managing named actors with `DeepCopy`.

**Color handling.** PLY vertex colors are loaded via Open3D (`pcd.has_colors()` → `np.asarray(pcd.colors)`), converted to uint8 RGB, and passed to PyVista as `scalars='colors', rgb=True`. When the Delaunay mesh is rendered, the same vertex colors are propagated to the mesh (vertex indices are preserved by the Delaunay triangulation).

**Delaunay triangulation.** Reuse the exact pattern from `define_forearm_mesh.py`:
```python
from scipy.spatial import Delaunay
xy = points[:, 0:2]
tri = Delaunay(xy)
mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
if np.mean(mesh.face_normals[:, 2]) < 0:
    mesh.invert()
mesh.fix_normals()
```
Converted to PyVista via `mesh_to_pyvista()` pattern from `rf_surface_utils.py`. Rendered semi-transparent (`opacity=0.4`) so the pointcloud remains visible underneath.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single-window with dropdown navigation | One window for all sessions/stages; instant switching | Must pre-discover all paths | **Chosen** |
| One window per session (sequential, like BeforeAfterStepViewer) | Simpler launcher | Cannot compare across sessions; tedious for 15+ sessions | Rejected |
| Open3D visualizer (like `define_forearm_mesh.py`) | No PyQt5 dependency | No dropdown widgets; inconsistent with existing postprocessing viewers | Rejected |

### Architecture Changes

New file:
```
code/src/postprocessing/gui/
  forearm_stage_inspector.py    — ForearmStageInspector(QMainWindow) + helpers (~300 lines)
```

Modified files:
- `code/src/postprocessing/gui/__init__.py` — add `ForearmStageInspector` export
- `code/scripts/postprocess_visualization.py` — add launcher function + batch integration
- `configs/postprocess_visualization_dag.yaml` — add task entry

### Reused Patterns
- **PLY loading:** `o3d.io.read_point_cloud()` + color extraction — used in `export_forearm_pca_calibrated.py`, `rf_surface_utils.py`
- **PyVista embedding:** `pyvistaqt.QtInteractor` — used in `rf_cluster_gallery_viewer.py`, `before_after_step_viewer.py`
- **Dropdown widgets:** `QComboBox` with `currentIndexChanged` signal — used in `rf_cluster_gallery_viewer.py`
- **Forearm PLY resolution:** `_load_unified_forearm()` in `postprocess_visualization.py`
- **Delaunay triangulation:** `define_forearm_mesh.py` (project to XY, `scipy.spatial.Delaunay`, normal correction)
- **Mesh conversion:** `mesh_to_pyvista()` in `rf_surface_utils.py`
- **Deferred render:** `showEvent` + `QTimer.singleShot` — used in `before_after_step_viewer.py`
- **Session grouping:** `defaultdict(list)` pattern from `postprocess_workflow_kinect_auto.py`

### Knowledge Base Check
- **`note-qt-itemchanged-signal-recursion.md`** — Not directly applicable (no `QTreeWidget`), but the principle of guarding against recursive signal emission applies: avoid calling `setCurrentIndex` from within a `currentIndexChanged` handler.
- No other notes are applicable to this feature.

### Class Skeleton

```
ForearmStageInspector(QMainWindow)
    __init__(stage_index: Dict[str, ForearmStagePaths], parent=None)

    # UI
    _build_ui() -> None
    _session_combo: QComboBox
    _stage_combo: QComboBox
    _delaunay_checkbox: QCheckBox
    _status_label: QLabel
    plotter: QtInteractor

    # State
    _stage_index: Dict[str, ForearmStagePaths]
    _ply_cache: Dict[Path, pv.PolyData]
    _mesh_cache: Dict[Path, pv.PolyData]
    _session_cameras: Dict[str, dict]
    _initial_render_done: bool

    # Slots
    _on_session_changed(index: int) -> None
    _on_stage_changed(index: int) -> None
    _on_delaunay_toggled(state: int) -> None

    # Scene
    _rebuild_scene() -> None
    _get_current_ply_path() -> Optional[Path]
    _load_or_cache_ply(path: Path) -> Optional[pv.PolyData]
    _compute_or_cache_delaunay(path: Path, cloud: pv.PolyData) -> Optional[pv.PolyData]

    # Qt lifecycle
    showEvent(event) -> None
    closeEvent(event) -> None
```

---

## Implementation Plan

### Phase 1: Data model and helpers
**Goal:** Create the data structures and PLY loading/meshing utility functions.
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 1.1 — Create `code/src/postprocessing/gui/forearm_stage_inspector.py` with `ForearmStagePaths` dataclass (fields: `session_id`, `raw_ply`, `pca_calibrated_ply`, `rf_centered_ply` — all `Optional[Path]`) and `STAGES` constant list
- [x] Task 1.2 — Implement `resolve_all_session_stage_paths(session_map: Dict[str, List[KinectConfig]]) -> Dict[str, ForearmStagePaths]` — iterates sessions, resolves each stage's PLY path using the `_load_unified_forearm` pattern for raw and direct path lookups for PCA/RF
- [x] Task 1.3 — Implement `load_forearm_ply(ply_path: Path) -> pv.PolyData` — loads PLY via Open3D, extracts points + vertex colors (fallback to grey if no colors), returns `pv.PolyData` with `'colors'` array
- [x] Task 1.4 — Implement `compute_delaunay_mesh(cloud: pv.PolyData) -> pv.PolyData` — 2.5D Delaunay on XY, trimesh normal fix, convert via `mesh_to_pyvista()` pattern, transfer vertex colors

**Files Modified:**
- `code/src/postprocessing/gui/forearm_stage_inspector.py` — **new** (data model + helpers)

**Dependencies:** None

### Phase 2: ForearmStageInspector viewer class
**Goal:** Build the interactive PyQt5 + PyVista viewer window.
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 2.1 — Implement `ForearmStageInspector.__init__(stage_index, parent)` — store state, init caches (`_ply_cache`, `_mesh_cache`, `_session_cameras`), call `_build_ui()`
- [x] Task 2.2 — Implement `_build_ui()` — toolbar row with session `QComboBox`, stage `QComboBox`, Delaunay `QCheckBox`, Close button; central `QtInteractor`; bottom status `QLabel`
- [x] Task 2.3 — Implement `_on_session_changed()`, `_on_stage_changed()`, `_on_delaunay_toggled()` slots — all call `_rebuild_scene()`
- [x] Task 2.4 — Implement `_rebuild_scene()`:
  1. Save current camera if a scene was previously shown
  2. `plotter.clear()` and `plotter.set_background('black')`
  3. Resolve PLY path for current (session, stage) from the index
  4. If path is None or does not exist: show text overlay "No PLY available for this stage" and return
  5. Load PLY (or cache hit)
  6. `plotter.add_points(cloud, scalars='colors', rgb=True, point_size=5, render_points_as_spheres=True, name='forearm')`
  7. If Delaunay checked: compute (or cache hit) mesh, `plotter.add_mesh(mesh_pv, scalars='colors', rgb=True, opacity=0.4, name='delaunay_mesh')`
  8. Restore camera if saved for this session; otherwise `plotter.reset_camera()`
  9. Update status label with PLY path
- [x] Task 2.5 — Implement `showEvent` (deferred initial render) and `closeEvent` (release OpenGL context via `plotter.close()`)

**Files Modified:**
- `code/src/postprocessing/gui/forearm_stage_inspector.py` — add viewer class

**Dependencies:** Phase 1

### Phase 3: Pipeline integration
**Goal:** Wire the viewer into the postprocess visualization workflow and DAG config.
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 3.1 — Add `ForearmStageInspector` export to `code/src/postprocessing/gui/__init__.py`
- [x] Task 3.2 — Add `run_forearm_stage_inspector(session_map, dag_handler)` function to `code/scripts/postprocess_visualization.py` — checks `can_run("view_forearm_stage_inspector")`, calls `resolve_all_session_stage_paths`, creates `QApplication` + `ForearmStageInspector`, shows + `exec_()`
- [x] Task 3.3 — Modify `run_batch_sequentially()` in `postprocess_visualization.py` to group configs by `session_id` into a `session_map` and call `run_forearm_stage_inspector()` before the per-block loop
- [x] Task 3.4 — Add `view_forearm_stage_inspector` task to `configs/postprocess_visualization_dag.yaml`

**Files Modified:**
- `code/src/postprocessing/gui/__init__.py` — add export
- `code/scripts/postprocess_visualization.py` — add launcher + batch integration
- `configs/postprocess_visualization_dag.yaml` — add task entry

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Enable `view_forearm_stage_inspector` in DAG config, run `postprocess_visualization.py`
- [ ] Session dropdown lists all sessions from `kinect_configs`
- [ ] Stage dropdown shows all three stages
- [ ] Switching sessions loads the correct forearm PLY for each stage
- [ ] Switching stages updates the 3D view with the correct coordinate space
- [ ] PLY vertex colors are displayed (visually distinct from uniform grey)
- [ ] Delaunay checkbox ON: mesh overlay appears with vertex colors, semi-transparent
- [ ] Delaunay checkbox OFF: mesh disappears, only pointcloud remains
- [ ] Missing stage PLY shows "No PLY found" text overlay
- [ ] Camera is preserved when switching stages within the same session
- [ ] Viewer closes cleanly (no OpenGL/VTK errors)

### Edge Cases
- [ ] Session with no forearm PLYs at any stage — all stages show "No PLY found"
- [ ] Session with only raw PLY (PCA/RF stages not yet run) — only raw stage shows data
- [ ] PLY file without vertex colors — renders in fallback grey
- [ ] `session_merged_output_dir` is None — PCA and RF stages resolve to None gracefully

---

## Documentation Plan

- [ ] Inline docstrings in `forearm_stage_inspector.py`
- [ ] Update `postprocess_visualization_dag.yaml` header comment with new task

---

## Rollback Plan

All changes are additive:
1. Delete `code/src/postprocessing/gui/forearm_stage_inspector.py`
2. Revert edits to `__init__.py`, `postprocess_visualization.py`, and DAG config
3. No data migrations or breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large PLY files cause slow loading on stage switch | Medium | Medium | LRU cache keyed by path; typical forearm clouds are 10k-50k points |
| Delaunay triangulation slow on large clouds | Low | Low | Cache computed meshes per (session, stage) key |
| `session_merged_output_dir` is None for some configs | Medium | Low | Check for None in path resolution; show "No data" gracefully |
| PLY files without vertex colors | Low | Low | Fallback to uniform grey (160, 160, 160) |
| OpenGL context issues after other viewers | Low | Medium | Use same `closeEvent` / `plotter.close()` pattern from BeforeAfterStepViewer |

---

## References

- Delaunay pattern: `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py`
- Unified forearm resolution: `code/scripts/postprocess_visualization.py:_load_unified_forearm()`
- PyVista viewer pattern: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`
- Mesh conversion: `code/src/analysis/receptive_field_mapping/rf_surface_utils.py:mesh_to_pyvista()`
- Existing viewer plan: `docs/development/plans/completed/postprocess-before-after-step-viewer.md`
- Visualization workflow: `code/scripts/postprocess_visualization.py`
