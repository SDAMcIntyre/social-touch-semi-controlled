# Plan: Postprocessing Before/After Step Viewer

**Date:** 2026-04-13
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/postprocess-before-after-step-viewer`

---

## Overview

**What:** A reusable side-by-side 3D viewer that displays the before and after state of contact points and the forearm pointcloud for each of the 4 postprocessing spatial transform steps.
**Why:** There is no way to visually verify that contact points are correctly located relative to the forearm at each postprocessing stage. Researchers need to confirm that each transform (ICP registration, PCA calibration, forearm projection, RF centering) produces correctly aligned spatial data before downstream analysis.
**How:** A single `BeforeAfterStepViewer(QMainWindow)` class with two synchronized PyVista `QtInteractor` plotters, integrated into the existing `postprocess_visualization.py` workflow as a new DAG task that sequentially opens one viewer window per step.

**Supersedes:** `docs/development/plans/archived/projection-comparison-viewer.md` — that plan covered only step 3 (projection); this one generalises the same dual-plotter approach to all 4 steps.

## Problem Statement

The postprocessing pipeline applies 4 sequential spatial transforms to contact-point data. Each step reads from one directory and writes to another, but the intermediate results can only be inspected by manually loading CSVs. There is no visual tool to compare the before/after alignment of contact points relative to the forearm at any individual step, making it difficult to:
- Verify that ICP registration correctly aligns multi-forearm sessions
- Confirm that PCA calibration orients both contacts and forearm as expected
- Spot anomalies where projection snaps points to the wrong forearm region
- Check that RF centering translates the data to the correct origin

## Goals

### In Scope
1. Side-by-side 3D viewer with synchronized camera control (rotate/zoom one side, the other follows)
2. Per-frame animation of contact points (same frame slider pattern as `PostprocessedSceneViewer`)
3. Forearm pointcloud displayed on both sides at every step, in the correct coordinate space for that side
4. Distinct contact point colours: cyan (before) / red (after)
5. Integration as a single DAG task (`view_before_after_steps`) that sequentially opens one window per step
6. `NeuralDataPanel` when `Nerve_freq` column is present
7. Visibility and point-size controls for forearm and contact points on each side

### Out of Scope
- Displacement statistics overlay (projection distances are already in `projection_stats.csv`)
- Comparison across sessions (this viewer is per-block, matching existing viewer pattern)
- Exporting comparison frames as images/video
- Modifications to any processing algorithm

## Success Criteria

- [ ] Viewer opens with two side-by-side 3D viewports for a given session block
- [ ] Left viewport shows pre-transform contact points with forearm in matching coordinate space
- [ ] Right viewport shows post-transform contact points with forearm in matching coordinate space
- [ ] Camera interaction in one viewport is mirrored in the other
- [ ] Frame slider animates both viewports simultaneously
- [ ] All 4 steps are shown sequentially (close one window, next opens)
- [ ] Steps with missing intermediate data are skipped with a warning
- [ ] Viewer integrates into the DAG workflow and can be enabled/disabled via YAML config

---

## Technical Design

### Forearm Pointcloud Mapping Per Step

Each side must show the forearm in the same coordinate space as its contact points.

| Step | Before contacts | Before forearm | After contacts | After forearm |
|------|----------------|---------------|----------------|--------------|
| 1. ICP Registration | `blocks_merged/` | Per-video forearm via `ForearmCatalog` + `get_forearms_with_fallback()` — Kinect native space | `blocks_registered/` | Unified registered PLY `forearm_pointclouds/{session_id}_unified_registered.ply` (or single-forearm fallback) |
| 2. PCA Calibration | `blocks_registered/` | Unified registered PLY (same as ICP after) | `blocks_pca_calibrated/` | PCA-calibrated PLY `forearm_pca_calibrated/{session_id}_forearm.ply` |
| 3. Forearm Projection | `blocks_pca_calibrated/` | PCA-calibrated PLY | `blocks_contact_projected/` | PCA-calibrated PLY (same — projection only snaps contacts) |
| 4. RF Centering | `blocks_contact_projected/` | PCA-calibrated PLY | `blocks_rf_centered/` | RF-centered PLY `forearm_rf_centered/{session_id}_forearm.ply` |

**Contact CSV filename patterns:**
- Steps 1-2 (before): `{session}_semicontrolled_{block}_merged_data.csv`
- Step 2 (after) onwards: `{session}_semicontrolled_{block}_merged_data_pca-xyz.csv`

**Per-video forearm loading (step 1 before):**
Following the pattern from `compute_somatosensory_characteristics.py`:
1. `ForearmFrameParametersFileHandler.load(forearm_metadata_path)` → forearm params
2. `ForearmCatalog(forearm_params, forearm_pointcloud_dir)` → catalog
3. `get_forearms_with_fallback(catalog, video_filename)` → dict of `{frame_id: Open3D pointcloud}`
4. Use key `0` (reference forearm) as the static pointcloud for the "before" side

**Unified/single-forearm loading (step 1 after, step 2 before):**
Following the pattern from `export_forearm_pca_calibrated.py:_find_forearm_ply()`:
1. Try `forearm_pointclouds/{session_id}_unified_registered.ply`
2. Fall back to any `*.ply` in `forearm_pointclouds/` (single-forearm session)

### Approach

Create a single reusable `BeforeAfterStepViewer(QMainWindow)` that accepts before/after forearm geometry (as Open3D pointcloud or PLY path) and before/after CSV paths. The viewer embeds two `pyvistaqt.QtInteractor` widgets side by side. Camera synchronization uses VTK's `AddObserver` with a re-entrancy guard (`_syncing` boolean).

A new DAG task `view_before_after_steps` iterates over the 4 steps, resolves the correct forearm for each side, and opens one viewer per step sequentially.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two pyvistaqt QtInteractors in one QMainWindow | Proven Qt widget embedding; event-driven camera sync; consistent with existing codebase | Slightly more complex than single-plotter | **Chosen** |
| Single plotter with `pv.Plotter(shape=(1,2))` sub-renderers | Built-in PyVista subplot support | Camera sync is manual; subplot API is less flexible for per-side Qt controls | Rejected |
| Matplotlib 3D side-by-side | Simple to implement | Too slow for interactive rotation with forearm meshes | Rejected |
| Separate viewer class per step | Could specialise per step | Massive duplication; all 4 steps need the same UI | Rejected |

### Architecture Changes

New module in existing GUI package:

```
code/src/postprocessing/gui/
  __init__.py                          — add BeforeAfterStepViewer export
  postprocessed_scene_viewer.py        — existing (unchanged)
  before_after_step_viewer.py          — NEW (~500 lines)
```

Integration points:
- `code/scripts/postprocess_visualization.py` — new path resolver + forearm loader + launcher function + batch call
- `configs/postprocess_visualization_dag.yaml` — new task definition

Reused imports:
- `preprocessing.forearm_extraction.ForearmFrameParametersFileHandler` — load forearm metadata
- `preprocessing.forearm_extraction.ForearmCatalog` — per-video forearm lookup
- `preprocessing.forearm_extraction.get_forearms_with_fallback` — reference forearm with fallback
- `postprocessing.xyz_reference_from_gestures.calibration_pca_engine.CalibrationResult` — (if needed)
- `merging.gui.neural_kinect_scene_viewer.NeuralDataPanel` — neural data panel

---

## Implementation Plan

### Phase 1: Core Viewer Class
**Goal:** Create the reusable dual-plotter viewer.

- [x] Create `BeforeAfterStepViewer(QMainWindow)` in `before_after_step_viewer.py`
- [x] Constructor accepts: `before_csv_path`, `after_csv_path`, `before_forearm` (Open3D pointcloud or PLY path), `after_forearm` (Open3D pointcloud or PLY path), `step_label`, `recording_name`
- [x] Data loading: both CSVs, filter kinect rows (`time_kinect` not NaN), pre-parse `contact_points` per frame
- [x] Forearm loading: accept either Open3D geometry directly or PLY path, convert to PyVista `PolyData`
- [x] Build UI layout: two `QtInteractor` widgets in `QHBoxLayout` with "Before"/"After" labels, right-panel scroll area with visibility checkboxes + point-size sliders, frame slider + play/pause + recenter
- [x] Implement `_init_actors()`: register forearm + dynamic contact `PolyData` on each plotter
- [x] Implement `_update_frame()`: `DeepCopy` contact points, render both plotters, update neural panel cursor
- [x] Implement VTK observer camera sync with `_syncing` re-entrancy guard
- [x] Implement deferred render pattern (`showEvent` + `QTimer.singleShot`)
- [x] Implement play timer (33ms ~30fps) and drag-throttle timer (80ms)
- [x] Reuse `_parse_contact_points_cell` from `postprocessed_scene_viewer.py:106-128`

**Files modified:**
- `code/src/postprocessing/gui/before_after_step_viewer.py` — **new** (~500 lines)

**Dependencies:** None

### Phase 2: Path Resolution + Forearm Loading + Launcher
**Goal:** Wire up per-step forearm resolution and the sequential launcher.

- [x] Add `_load_per_video_forearm(config)` to `postprocess_visualization.py`:
  - Load forearm metadata from `session_processed_output_dir/forearm_pointclouds/{session_id}_arm_roi_metadata.json`
  - Build `ForearmCatalog` from loaded params + pointcloud dir
  - Call `get_forearms_with_fallback(catalog, config.source_video.name)`
  - Return the reference forearm (key `0`) as Open3D pointcloud
- [x] Add `_load_unified_forearm(config)` — loads `{session_id}_unified_registered.ply` with fallback to single forearm (reuse `_find_forearm_ply` logic from `export_forearm_pca_calibrated.py`)
- [x] Add `resolve_before_after_steps(config)` — returns list of 4 step descriptors, each with: `step_label`, `before_csv`, `after_csv`, `before_forearm` (Open3D geometry or PLY path), `after_forearm` (Open3D geometry or PLY path)
- [x] Add `run_single_session_pipeline_before_after(config, dag_handler)` launcher — checks `can_run("view_before_after_steps")`, iterates steps, skips with warning if CSVs or forearm missing, launches viewer per step
- [x] Add call in `run_batch_sequentially()` loop after existing simple/advanced calls
- [x] Add imports for `ForearmFrameParametersFileHandler`, `ForearmCatalog`, `get_forearms_with_fallback`, `BeforeAfterStepViewer`

**Files modified:**
- `code/scripts/postprocess_visualization.py`

**Dependencies:** Phase 1

### Phase 3: Package Export + DAG Config
**Goal:** Register the viewer and add the DAG task.

- [x] Add `BeforeAfterStepViewer` to `code/src/postprocessing/gui/__init__.py` exports and `__all__`
- [x] Add `view_before_after_steps` task to `configs/postprocess_visualization_dag.yaml` (disabled by default)

**Files modified:**
- `code/src/postprocessing/gui/__init__.py` — add export
- `configs/postprocess_visualization_dag.yaml` — add task block

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Enable `view_before_after_steps` in DAG config, run `postprocess_visualization.py`
- [ ] Step 1 (ICP): left shows per-video forearm + pre-registration contacts; right shows unified forearm + registered contacts — verify contacts align to forearm on the right but may be offset on the left (multi-forearm) or identical (single-forearm)
- [ ] Step 2 (PCA): left shows unified forearm + registered contacts; right shows PCA-calibrated forearm + PCA contacts — both sides should show contacts on the forearm surface, but in different orientations
- [ ] Step 3 (Projection): left shows PCA contacts floating near surface; right shows projected contacts snapped to surface — same PCA forearm on both sides
- [ ] Step 4 (RF Centering): left shows projected contacts on PCA forearm; right shows RF-centered contacts on RF-centered forearm — forearm should shift to center on receptive field
- [ ] Frame slider animates both viewports simultaneously
- [ ] Camera sync works bidirectionally
- [ ] Recenter, play/pause, visibility toggles, point-size sliders all functional
- [ ] Closing each viewer window advances to the next step

### Edge Cases
- [ ] Session with missing intermediate directory — step is skipped with warning
- [ ] Single-forearm session (no unified PLY) — fallback to single forearm PLY for ICP step
- [ ] Session with no contact points in a frame — viewer shows only forearm
- [ ] Session that only completed up to step 2 — steps 3 and 4 are skipped
- [ ] Missing forearm metadata JSON — step 1 skipped with warning

---

## Documentation Plan

- [ ] Inline docstrings in `before_after_step_viewer.py`
- [ ] Update `postprocess_visualization_dag.yaml` header comment with new task description

---

## Rollback Plan

All changes are additive (one new file + small edits to existing files). Rollback:
1. Delete `code/src/postprocessing/gui/before_after_step_viewer.py`
2. Revert edits to `__init__.py`, `postprocess_visualization.py`, and DAG config
3. No data migrations or breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Two QtInteractors cause VTK context conflicts | Low | High | Proven by PyVista docs — multiple QtInteractors in one window is supported |
| Camera sync infinite loop via VTK observers | Medium | Medium | Re-entrancy guard (`_syncing` boolean); validated by knowledge base `note-qt-itemchanged-signal-recursion.md` |
| Per-video forearm loading fails (missing metadata) | Medium | Low | Step 1 skipped with warning; remaining steps unaffected |
| Step 2 filename mismatch (raw vs `_pca-xyz`) | Low | Medium | Path resolver computes both name patterns explicitly |
| Missing intermediate directories | Medium | Low | Each step checked independently and skipped with warning |

---

## References

- Per-video forearm pattern: `code/scripts/_3_preprocessing/_4_somatosensory_quantification/compute_somatosensory_characteristics.py:82-85`
- Unified forearm fallback: `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py:27-43`
- Existing viewer: `code/src/postprocessing/gui/postprocessed_scene_viewer.py`
- Visualization workflow: `code/scripts/postprocess_visualization.py`
- ForearmCatalog: `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
- Knowledge base: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
