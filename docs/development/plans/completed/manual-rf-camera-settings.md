# Plan: Manual RF Camera Settings

**Date:** 2026-05-08
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-11 07:31
**Base Branch:** `dev`
**Branch:** `feature/manual-rf-camera-settings`

---

## Overview

**What:** Replace automatic PCA-based tangent plane rotation with a user-controlled GUI for setting camera orientation per session, and thread the saved rotation matrix through all downstream RF rendering and projection code.

**Why:** The current `compute_tangent_plane_rotation()` derives its rotation from whichever ~180 degrees of the forearm the Kinect happened to capture. This makes all 2D-projected metrics (hull area, ellipse shape, Gaussian fit) and heatmap renderings sensitive to the physical camera setup — a confound for cross-session comparisons.

**How:** A new PyQt5 viewer lets the researcher interactively orient the forearm per session and save camera settings. The saved orientation becomes the single source of truth for rotation matrix R, replacing automatic PCA everywhere.

## Problem Statement

`compute_tangent_plane_rotation()` fits a surface normal via SVD on local forearm vertices near the contact centroid. The Kinect captures only one side of the forearm (~180 degrees), so:

1. Different physical camera positions capture different forearm surface regions, producing different SVD normals.
2. The rotation matrix R derived from this normal varies across sessions, making 2D-projected metrics incomparable.
3. Heatmap renderings orient differently even for the same forearm region.
4. The existing `pick_rf_camera_angle_batch()` auto-assign produces `camera_params.json` files that are never consumed by any rendering or metric code — a dead artifact.

## Goals

### In Scope
1. New GUI task (`set_rf_camera_settings`) for interactively setting camera orientation per session
2. Persistent camera settings saved to `4_analysed/rf_camera_settings/rf_camera_settings.json`
3. Derive rotation matrix R from saved camera settings instead of PCA
4. Thread R through all downstream RF projection, metrics, and rendering code
5. Remove auto-assign (`rf_camera_angle_task.py`) and auto-computed tangent plane rotation from production paths

### Out of Scope
- Changing the 2D projection algorithm itself (tangent plane drop-Z remains the default)
- Modifying `project_cylindrical_unwrap()` (it uses its own PCA axis, not the tangent plane R)
- Adding a new projection method
- Migrating old `camera_params.json` files

## Success Criteria

- [ ] GUI launches, displays forearm mesh with all contact points overlaid per session
- [ ] Session dropdown shows green background for sessions with saved camera settings
- [ ] "Save Camera Settings" button persists to JSON; overwrite shows warning popup
- [ ] Reopening the GUI restores saved camera orientation per session
- [ ] `visualize_receptive_fields_clustered` uses saved R for heatmap rendering
- [ ] `reduce_population_rf_grid` uses saved R for metric projection
- [ ] Running any downstream task without saved camera settings raises a clear `ValueError`
- [ ] `rf_camera_angle_task.py` deleted; `camera_angle_mode` parameter removed from flows

---

## Technical Design

### Approach

The camera settings define a viewing coordinate frame. The rotation matrix R is derived from the saved camera position, focal point, and up vector:

```
view_dir = normalize(focal_point - camera_position)    -> Z axis
right    = normalize(cross(up_vector, view_dir))       -> X axis
up_corr  = cross(view_dir, right)                      -> Y axis
R        = stack([right, up_corr, view_dir])            -> 3x3 orthonormal
```

This R has the same semantics as `compute_tangent_plane_rotation()` output: `align_points(pts, R)` rotates into a frame where Z is the view direction and XY is the projection plane.

The new GUI is a standalone `RFCameraSettingsViewer(QMainWindow)` — simpler than `TouchPopulationExplorer` (no scatter plot, no filter rectangle, no gesture checkboxes). It reuses the same data loading (`PopulationData`) and session switching patterns.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Manual GUI with per-session save | User controls orientation; reproducible; decouples from camera placement | Requires running GUI before downstream tasks | **Chosen** |
| Fix PCA to be camera-invariant | No GUI needed | Fundamentally impossible — PCA depends on which vertices are captured | Rejected |
| Inherit from `TouchPopulationExplorer` | Code reuse | Parent is heavily specialized (scatter, filters, thresholds); disabling 80% of features is messier than copying the few needed patterns | Rejected |
| Auto-assign with better heuristics | No user interaction needed | Still camera-dependent; just hides the problem | Rejected |

### Architecture Changes

**New file:**
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_settings_viewer.py` — `RFCameraSettingsViewer(QMainWindow)`

**Output file format** (single JSON, keyed by session_id):
```json
{
  "ST13-01_semicontrolled": {
    "camera_position": [152.3, -45.2, 450.8],
    "focal_point": [152.3, -45.2, 50.8],
    "up_vector": [0.0, 1.0, 0.0],
    "view_angle": 30.0
  }
}
```

**R propagation:** All functions that currently call `compute_tangent_plane_rotation()` internally gain a keyword-only `rotation_matrix` parameter. Callers load R from the camera settings file and pass it explicitly. When `rotation_matrix` is `None`, `project_tangent_plane()` raises `ValueError` (fail-fast).

### Knowledge Base

- **note-3d-to-2d-surface-projection-algorithms.md** — confirms projection pipeline is method-agnostic once R is provided externally; documents ~5-10% distortion at tangent plane edges (acceptable for 20-60 mm patches).
- **note-rf-cluster-gallery-gui-components.md** — provides the `_capture_camera()` / `_restore_camera()` patterns and deferred PyVista init via `QTimer.singleShot(0, ...)`.
- **note-rf-feature-space-explorer-gui-components.md** — provides session-switching pattern with camera state preservation.
- **note-qt-itemchanged-signal-recursion.md** — must use `blockSignals()` when updating dropdown backgrounds programmatically.

---

## Implementation Plan

### Phase 1: Foundation
**Goal:** Add `camera_settings_to_rotation()` and save/load utilities — no behavior change.
**Started:** 2026-05-08
**Completed:** 2026-05-08

- [x] Task 1.1 — Add `camera_settings_to_rotation(camera_settings: dict) -> np.ndarray` to `tangent_plane_alignment.py`
- [x] Task 1.2 — Add `load_rf_camera_settings(output_dir)`, `save_rf_camera_settings(output_dir, cameras)`, and `load_rf_camera_rotation(output_dir, session_id)` to `rf_extraction_io.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` — new function
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — three new functions

**Dependencies:** None

### Phase 2: New GUI
**Goal:** Create the camera settings viewer and wire it into the DAG — additive, no breaking changes.
**Started:** 2026-05-08
**Completed:** 2026-05-08

- [x] Task 2.1 — Create `RFCameraSettingsViewer(QMainWindow)` in `gui/rf_camera_settings_viewer.py` with: session combo (green background for saved sessions), "Save Camera Settings" button (with overwrite warning), PyVista interactor showing forearm mesh + contact point heatmap
- [x] Task 2.2 — Add `launch_rf_camera_settings_viewer()` to `rf_cluster_pipeline.py` following the `launch_touch_population_explorer()` pattern
- [x] Task 2.3 — Register `RFCameraSettingsViewer` in `gui/__init__.py` and `launch_rf_camera_settings_viewer` in `receptive_field_mapping/__init__.py`
- [x] Task 2.4 — Add `set_rf_camera_settings_flow` to `analysis_workflow.py` with category `"viewer"`, positioned after `touch_series_transforms` and before visualization tasks

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_settings_viewer.py` — **new file**
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — register new viewer
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — new launcher function
- `code/src/analysis/receptive_field_mapping/__init__.py` — export new launcher
- `code/scripts/analysis_workflow.py` — new flow + task registration

**Dependencies:** Phase 1

### Phase 3: Thread R through pipeline
**Goal:** All downstream tasks load R from saved camera settings instead of computing it. Breaking change: requires camera settings to exist.
**Started:** 2026-05-08
**Completed:** 2026-05-08

- [x] Task 3.1 — Add `rotation_matrix` keyword parameter to `project_tangent_plane()` and `project_to_2d()` in `rf_projection.py`; raise `ValueError` when None
- [x] Task 3.2 — Add `rotation_matrix` keyword parameter to `compute_rf_metrics()` in `rf_metrics.py`, pass through to `project_to_2d()`
- [x] Task 3.3 — Add `rotation_matrix` keyword parameter to `compute_grid_cell_metrics()` in `rf_grid_cell_metrics.py`, pass through to both `compute_rf_metrics()` and `project_to_2d()`
- [x] Task 3.4 — Add `rotation_matrix` keyword parameter to `render_forearm_heatmap()` in `rf_cluster_visualizer.py`, replace auto-computation at line 344, pass R to all `project_to_2d()` calls
- [x] Task 3.5 — Update `run_cluster_rf_visualization()` in `rf_cluster_pipeline.py`: load camera settings at start, derive R per session, pass to `compute_rf_metrics()` and `render_forearm_heatmap()`
- [x] Task 3.6 — Update `run_simple_rf_mapping()` in `rf_simple_pipeline.py`: load camera settings, pass R to `render_forearm_heatmap()`
- [x] Task 3.7 — Update `rf_population_grid_metrics_pipeline.py`: load camera settings, pass R to `compute_grid_cell_metrics()`
- [x] Task 3.8 — Update `load_gallery_data()` in `rf_gallery_data.py`: replace `compute_tangent_plane_rotation()` call with `camera_settings_to_rotation()` from loaded settings; `GalleryCell.tangent_rotation` populated from camera settings

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_projection.py` — add `rotation_matrix` param
- `code/src/analysis/receptive_field_mapping/rf_metrics.py` — add `rotation_matrix` param
- `code/src/analysis/receptive_field_mapping/rf_grid_cell_metrics.py` — add `rotation_matrix` param
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — add `rotation_matrix` param, remove auto-computation
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — load and pass R
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — load and pass R
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` — load and pass R
- `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` — replace PCA with camera-settings R

**Dependencies:** Phase 2

### Phase 4: Cleanup
**Goal:** Remove dead code.
**Started:** 2026-05-08
**Completed:** 2026-05-08

- [x] Task 4.1 — Delete `rf_camera_angle_task.py`; remove imports from `__init__.py` and `analysis_workflow.py`
- [x] Task 4.2 — Remove `pick_rf_camera_angle_batch` calls from `map_receptive_fields_clustered_flow` (line 550) and `visualize_receptive_fields_clustered_flow` (line 638)
- [x] Task 4.3 — Remove `camera_angle_mode` parameter from flows and from the dispatch block in `run_batch_analysis()` (lines 1063-1081)
- [x] Task 4.4 — Remove `compute_tangent_plane_rotation()` and `_compute_surface_normal()` from `tangent_plane_alignment.py`; keep `align_points()` and `camera_settings_to_rotation()`
- [x] Task 4.5 — Remove `_normal_to_view_angles()` and `_compute_surface_normal` import from `rf_cluster_visualizer.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — **delete**
- `code/src/analysis/receptive_field_mapping/__init__.py` — remove dead imports
- `code/scripts/analysis_workflow.py` — remove calls and parameter handling
- `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` — remove dead functions
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — remove dead imports/functions

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `camera_settings_to_rotation()` produces orthonormal R (det = +1, R @ R.T = I)
- [ ] `camera_settings_to_rotation()` raises when up vector is parallel to view direction
- [ ] `load_rf_camera_settings()` returns `{}` when file absent
- [ ] `load_rf_camera_settings()` raises `ValueError` on corrupt JSON
- [ ] `load_rf_camera_rotation()` raises `ValueError` when session key is missing
- [ ] Round-trip: save then load returns the same dict
- [ ] `project_tangent_plane()` with `rotation_matrix=None` raises `ValueError`
- [ ] `project_tangent_plane()` with valid `rotation_matrix` returns correct (N, 2) array

### Manual Verification
- [ ] Launch `set_rf_camera_settings`, navigate sessions, adjust camera, save — verify JSON written
- [ ] Overwrite existing setting — verify warning popup appears
- [ ] Session with saved settings shows green in dropdown
- [ ] Close and reopen viewer — verify camera restores to saved position
- [ ] Run `visualize_receptive_fields_clustered` — verify heatmaps render with user-chosen orientation
- [ ] Run `reduce_population_rf_grid` — verify metrics CSVs produced with user-defined R
- [ ] Delete settings file, run visualization — verify clear `ValueError` with actionable message

### Edge Cases
- [ ] Session with no forearm PLY — viewer raises at launch (fail-fast, existing behavior)
- [ ] First launch with no existing `rf_camera_settings.json` — all sessions show default camera, no green backgrounds
- [ ] Gallery viewer still works — `load_gallery_data()` correctly loads camera-derived rotation

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` — replace camera angle auto-assign description with new GUI task; update RF mapping section
- [ ] No user guide needed — the GUI is self-explanatory (one button, one dropdown)

---

## Rollback Plan

1. **Phase 4 is fully reversible:** restore `rf_camera_angle_task.py` from git history, re-add imports
2. **Phase 3 is reversible:** remove `rotation_matrix` parameters, restore `compute_tangent_plane_rotation()` calls in each caller
3. **Phases 1-2 are additive:** the new GUI and utilities can be deleted with no impact on existing code
4. **No data migration:** `rf_camera_settings.json` is a new file; old `camera_params.json` files are already dead artifacts

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| User forgets to run camera settings before downstream tasks | Medium | Low | Clear `ValueError` message: "Run 'set_rf_camera_settings' first" |
| Camera orientation produces poor tangent plane for some sessions | Low | Medium | User can re-run the GUI and re-save; no pipeline state to clean up |
| `PopulationData` loading is slow for many sessions | Low | Low | Same thread-pool pattern as `TouchPopulationExplorer`; caches warm after first load |
| Gallery viewer breaks if camera settings file is missing | Medium | Medium | `load_gallery_data()` raises with actionable message; gallery can only run after camera settings task |
| Existing `session_cameras.json` (gallery convenience) conflicts with new `rf_camera_settings.json` | Low | Low | Different files, different purposes; gallery's in-viewer camera memory is independent from pipeline's authoritative R source |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Foundation | ~1 hour | None |
| Phase 2: New GUI | ~3 hours | Phase 1 |
| Phase 3: Thread R | ~2 hours | Phase 2 |
| Phase 4: Cleanup | ~1 hour | Phase 3 |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-cluster-gallery-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- Related plan: `docs/development/plans/pending/rf-gallery-persistent-camera-settings.md` (gallery convenience, separate concern)
- Related plan: `docs/development/plans/pending/investigate-rf-projection.md` (projection diagnostic, complementary)
