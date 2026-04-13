# Plan: Pick RF Camera Angle — Multi-Session Viewer

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/pick-rf-camera-angle`

---

## Overview

**What:** A multi-session interactive viewer for picking and persisting camera angles on RF-centered forearms with selectivity heatmaps and full rendering controls.

**Why:** Downstream visualization and figure generation need a consistent, researcher-chosen camera angle per session. The researcher needs to compare sessions side-by-side (via dropdown switching) and fine-tune rendering parameters (colors, sizes, opacity) to get the best visual output. These rendering settings should be global so the researcher can establish a consistent visual style across all sessions.

**How:** After all sessions complete RF centering, a single PyQt5+pyvistaqt GUI launches showing all valid sessions in a dropdown. The viewer displays the 3D forearm + selectivity heatmap with a settings panel for full rendering control. Camera parameters are saved per-session to `camera_params.json`.

## Problem Statement

After RF centering, all sessions share a common coordinate origin, but there is no persisted camera viewpoint. Each session's forearm geometry and contact distribution is different, so a single hardcoded angle does not work. The researcher needs to:
1. Interactively choose the best viewing angle per session
2. Control how the point clouds are rendered (colors, sizes, opacity, background)
3. Switch between sessions quickly while preserving visual settings
4. Save each session's camera angle for reproducible downstream rendering

The current single-session picker (Phase 1-2 already implemented) forces the researcher to set up rendering preferences from scratch for each session and offers no control over visual parameters.

## Goals

### In Scope

1. Multi-session viewer with QComboBox dropdown to switch between available sessions
2. Extended settings panel: background color, forearm color/size/opacity/spheres, contact colormap/size/opacity/spheres, scalar bar toggle, axes toggle
3. Global settings that persist when switching sessions
4. Per-session camera parameters saved to individual `camera_params.json` files
5. Post-batch execution — viewer launches once after all sessions complete, not per-session
6. Session status indicators in dropdown (saved / modified / untouched)
7. Graceful handling of sessions with failed RF centering (excluded from dropdown)

### Out of Scope

- Automatic camera angle selection (this is explicitly interactive)
- Consuming `camera_params.json` in downstream tasks (future work)
- Persisting rendering settings to disk (settings are in-memory only, reset on next launch)
- Side-by-side multi-session viewing (only one session visible at a time)

## Success Criteria

- [ ] Viewer launches once after all sessions complete RF centering
- [ ] Dropdown lists all sessions with valid RF centering from DAG config
- [ ] Switching sessions updates the 3D view while preserving all rendering settings
- [ ] All rendering controls work: background color, forearm color/size/opacity/spheres, contact colormap/size/opacity/spheres, scalar bar, axes
- [ ] "Save Camera" writes valid `camera_params.json` for the active session
- [ ] "Save All & Close" writes all modified cameras and closes
- [ ] Dropdown shows status indicators (saved/modified/untouched)
- [ ] Per-session camera is restored when switching back to a previously viewed session
- [ ] Disabling `pick_rf_camera_angle` in DAG does not block `aggregate_session`

---

## Technical Design

### Approach

Rewrite the existing `RFCameraAnglePicker` as a multi-session viewer. A new `collect_session_scene_data()` function loads forearm PLYs, computes selectivity overlays, and packages data for all valid sessions. The viewer receives a `Dict[str, SessionSceneData]` and manages session switching via `plotter.clear()` + `_build_scene()` (fast for static point clouds). Rendering settings live in a `_ViewerSettings` dataclass and are applied on every scene rebuild.

The task moves from a per-session pipeline stage to a post-batch step in `run_batch_postprocessing()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PyVista + pyvistaqt (QtInteractor) | Proven pattern in codebase, interactive, VTK-powered | Heavier than matplotlib | **Chosen** — matches existing viewers |
| Pre-build all session actors, toggle visibility | No rebuild delay | High GPU memory with many sessions | Rejected — clear+rebuild is fast enough |
| Per-session viewer with settings file | Simpler code | Poor UX, no quick comparison | Rejected — user wants multi-session |

### Architecture Changes

**Rewritten modules:**
- `code/src/postprocessing/gui/rf_camera_angle_picker.py` — multi-session viewer with settings panel
- `code/scripts/_5_postprocessing/pick_rf_camera_angle.py` — batch data collection + single launch

**Modified modules:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — move task from per-session to post-batch

**New data structures:**
- `SessionSceneData` dataclass — per-session scene data bundle
- `_ViewerSettings` dataclass — global rendering settings

**Reused utilities (unchanged):**
- `parse_contact_points()` from `preprocessing.forearm_extraction.registration.csv_spatial_transformer`
- `RFMappingEngine.compute_selectivity()` from `analysis.receptive_field_mapping.rf_mapping_engine`
- `_compute_surface_normal()` from `analysis.receptive_field_mapping.rf_cluster_visualizer`

**Reference patterns:**
- `PostprocessedSceneViewer` (`code/src/postprocessing/gui/postprocessed_scene_viewer.py`) — right panel with sliders/toggles, deferred render, bounding proxy
- `HandModelSelectorGUI` (`code/src/preprocessing/motion_analysis/hand_tracking/gui/hand_model_selector.py`) — QComboBox + `plotter.clear()` + rebuild on switch

### Architecture Constraints (from Knowledge Base)

- **CuPy import order:** The task module imports from `preprocessing.*`, so `cupy` must be imported first (see `note-cupy-import-order.md`)
- **Widget layout:** Keep pyvistaqt renderer at top level, controls in separate panel (see `note-open3d-scenewidget-layout.md`)
- **Deferred render:** Must use `showEvent` + `QTimer.singleShot(0, ...)` + bounding proxy pattern (see `bug-neural-kinect-viewer-initial-render.md`)
- **Units:** All coordinates in mm (see `note-somatosensory-units-and-calculations.md`)

### GUI Layout

```
+------------------------------------------------------------------+
| [Session: v QComboBox       ] [ Save Camera ] [ Save All & Close ]|
+------------------------------------------------------------------+
| 3D Plotter (stretch=4)          | Settings Panel (260px, scroll)  |
|                                  | -- Background ---------------  |
|                                  | Color: [picker button]         |
|                                  | -- Forearm -------------------  |
|                                  | Color: [picker button]         |
|                                  | Point size: [====slider====]   |
|                                  | Opacity:    [====slider====]   |
|                                  | [x] Render as spheres          |
|                                  | -- Contact Points ------------  |
|                                  | Colormap: [v QComboBox]        |
|                                  | Point size: [====slider====]   |
|                                  | Opacity:    [====slider====]   |
|                                  | [x] Render as spheres          |
|                                  | -- Display ------------------  |
|                                  | [x] Scalar bar                 |
|                                  | [x] Axes                       |
+------------------------------------------------------------------+
```

### Session Switching Behavior

On dropdown change (`_on_session_changed`):
1. Store current camera in `_modified_cameras[old_session_id]` (in-memory)
2. `plotter.clear()` + `_build_scene()` with new session data and current `_settings`
3. Restore camera: check `_modified_cameras` first, then `saved_camera_params`, then initial_normal-based default

### Output Format

`camera_params.json` per session (unchanged):
```json
{
  "camera_position": [x, y, z],
  "focal_point": [x, y, z],
  "up_vector": [x, y, z],
  "view_angle": 30.0
}
```

### DAG Dependency Structure

```
center_on_receptive_field (per-session)
├── aggregate_session     (per-session, independent of picker)
└── pick_rf_camera_angle  (post-batch, all sessions at once)
```

---

## Implementation Plan

### Phase 1: Batch Data Collection

**Goal:** Rewrite the task module to collect data for all sessions and launch the viewer once.

- [x] Define `SessionSceneData` dataclass in `pick_rf_camera_angle.py`
- [x] Add `collect_session_scene_data(session_output_dirs: Dict[str, Path]) -> Dict[str, SessionSceneData]` — iterates sessions, checks RF status, loads forearm PLY, computes selectivity, computes surface normal, loads existing camera_params.json
- [x] Add `pick_rf_camera_angle_batch(session_output_dirs, *, force_processing)` — calls collect, filters, launches GUI once
- [x] Keep old `pick_rf_camera_angle()` as backward-compat wrapper

**Files Modified:**
- `code/scripts/_5_postprocessing/pick_rf_camera_angle.py` — rewrite with batch collection

**Dependencies:** None

### Phase 2: Multi-Session GUI with Settings

**Goal:** Rewrite the GUI as a multi-session viewer with rendering controls.

- [x] Rewrite `RFCameraAnglePicker.__init__` to accept `Dict[str, SessionSceneData]`
- [x] Define `_ViewerSettings` dataclass with defaults
- [x] `_build_ui()` — top toolbar (QComboBox + Save Camera + Save All & Close) + horizontal split (plotter + scrollable settings panel)
- [x] `_build_settings_panel()` — QGroupBoxes for Background, Forearm, Contact Points, Display with color pickers (QPushButton + QColorDialog), sliders, checkboxes, colormap QComboBox
- [x] `_on_session_changed(index)` — store camera, clear, rebuild, restore camera
- [x] `_build_scene()` — uses `self._settings` for rendering, `self._sessions[current]` for data; includes bounding proxy
- [x] `_apply_setting_change()` — preserve camera, clear, rebuild (triggered by any settings widget)
- [x] `_on_save_camera()` — write current camera to disk, update combo indicator
- [x] `_on_save_all_and_close()` — write all modified cameras, close
- [x] `_update_combo_display()` — status indicators: "(saved)" / "(modified)" / ""
- [x] Preserve deferred render pattern (showEvent -> QTimer.singleShot -> _deferred_start)

**Files Modified:**
- `code/src/postprocessing/gui/rf_camera_angle_picker.py` — rewrite

**Dependencies:** Phase 1

### Phase 3: Pipeline Integration

**Goal:** Move pick_rf_camera_angle from per-session to post-batch execution.

- [x] Remove `pick_rf_camera_angle` stage from `pipeline_stages` in `run_single_session_postprocessing()`
- [x] Add post-batch call in `run_batch_postprocessing()` after session loop: check DAG enabled, collect session_output_dirs from session_map, call `pick_rf_camera_angle_batch()`
- [x] Update `pick_rf_camera_angle_flow` wrapper to use `pick_rf_camera_angle_batch` (removed flow wrapper, direct import of batch function)
- [x] Add comment in DAG YAML clarifying post-batch execution

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — remove per-session stage, add post-batch call
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — update comment

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification

- [ ] Run postprocess workflow with multiple sessions — viewer launches once after all complete
- [ ] Dropdown lists all sessions with valid RF centering, excludes failed ones
- [ ] Switch sessions — 3D view updates, rendering settings persist
- [ ] Change each setting — scene updates correctly: background color, forearm color/size/opacity/spheres, contact colormap/size/opacity/spheres, scalar bar, axes
- [ ] "Save Camera" writes `camera_params.json` for active session, combo shows "(saved)"
- [ ] "Save All & Close" writes all modified cameras, window closes
- [ ] Switch to previously viewed session — camera position restored from in-memory state
- [ ] Set `enabled: false` for `pick_rf_camera_angle` in DAG — `aggregate_session` still runs, no viewer

### Edge Cases

- [ ] Only one session with valid RF centering — dropdown has single entry, viewer still works
- [ ] No sessions with valid RF centering — no viewer launched, logs info message
- [ ] Session with zero contact points — forearm shown without heatmap, settings still apply
- [ ] Close via X button (not Save All) — no camera_params.json written for unsaved sessions
- [ ] Existing camera_params.json from previous run — loaded and applied on session select

---

## Documentation Plan

- [ ] Inline docstrings for `pick_rf_camera_angle_batch()`, `SessionSceneData`, `RFCameraAnglePicker`
- [ ] Comment in DAG YAML explaining post-batch, optional nature of the task

---

## Rollback Plan

1. Revert `rf_camera_angle_picker.py` and `pick_rf_camera_angle.py` to single-session versions (from git history)
2. Restore per-session pipeline stage in `postprocess_workflow_kinect_auto.py`
3. Remove post-batch call from `run_batch_postprocessing()`
4. No data migration needed — `camera_params.json` files are standalone

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyVista/VTK blank viewport on first render | Med | Low | Preserve proven deferred-render pattern from PostprocessedSceneViewer |
| Memory usage with many sessions loaded | Low | Low | ~36 MB for 15 sessions at 100K points each — acceptable |
| `plotter.clear()` + rebuild flicker | Med | Low | Rebuild is <100ms for static point clouds; camera restore eliminates jarring resets |
| QColorDialog blocks event loop | Low | Low | Modal dialog is expected UX for color picking |
| GPU driver issues on Windows | Low | Med | Document as known environment issue (per KB note) |
