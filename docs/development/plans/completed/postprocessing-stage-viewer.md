# Plan: Postprocessing Stage Viewer

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `dev`
**Branch:** `feature/postprocessing-stage-viewer`

---

## Overview

**What:** A new single-window PyQt5 viewer that displays postprocessed data
(forearm PLY, contact points, neural panel) with a dropdown menu to switch
between the 5 postprocessing stages interactively.

**Why:** Currently, inspecting data after each postprocessing step requires
either opening `BeforeAfterStepViewer` (one window per step, sequentially) or
`PostprocessedSceneViewer` (only the final PCA-calibrated stage). Neither
allows fluid, interactive comparison across stages in a single window.

**How:** Create a new `PostprocessingStageViewer` class (standalone
`QMainWindow`) with a QComboBox stage dropdown. On stage change, reload the
CSV and forearm PLY, rebuild VTK actors, and recreate the neural panel. Follow
established viewer design invariants from the codebase.

## Problem Statement

The postprocessing pipeline transforms spatial data through 5 coordinate
spaces (Kinect native -> ICP-registered -> PCA-calibrated -> Projected ->
RF-centered). Each step produces intermediate CSV and forearm PLY outputs.

The existing visualization tools have gaps:
- **`PostprocessedSceneViewer`** shows only the Contact Projected stage (with
  optional hand mesh). No way to view earlier or later stages.
- **`BeforeAfterStepViewer`** opens one side-by-side window per step
  sequentially. Cumbersome for interactive exploration and requires closing
  each window to see the next step.
- **`ForearmStageInspector`** has dropdowns but only inspects the forearm
  surface itself — no CSV contact points, no neural panel.

Users need a single viewer — similar to the Merging `NeuralKinectViewer` in
richness — where they can interactively switch stages via a dropdown to verify
the effect of each postprocessing step on spatial data.

## Goals

### In Scope

1. New `PostprocessingStageViewer` class with stage dropdown (QComboBox)
2. Display at each stage: forearm PLY (static), contact points (per frame),
   NeuralDataPanel (Nerve_freq, contact_depth, contact_area)
3. Frame slider, play/pause, recenter camera — matching existing viewer UX
4. Camera reset on coordinate-frame boundary crossing (camera-space vs PCA-space)
5. New DAG task `view_postprocessing_stages` and script launcher function
6. Path resolver mapping each of 5 stages to CSV + forearm + coordinate frame

### Out of Scope

- Sticker spheres and hand mesh overlay (available in `PostprocessedSceneViewer` advanced mode)
- Side-by-side comparison (available in `BeforeAfterStepViewer`)
- Session-level multi-session dropdown (available in `ForearmStageInspector`)
- GPU-accelerated point cloud cropping (not needed — no raw MKV data)

## Success Criteria

- [ ] Viewer opens with a dropdown showing 5 stage labels
- [ ] Selecting a stage reloads forearm, contact points, and neural panel
- [ ] Frame slider navigates within a stage; position preserved across stage switches
- [ ] Camera resets when switching between camera-space and PCA-space stages
- [ ] Stages with missing CSV are handled gracefully (disabled or "no data" message)
- [ ] Play/pause animation works within each stage
- [ ] DAG task integrates with the existing postprocess visualization workflow

---

## Technical Design

### Approach

Create a new standalone `PostprocessingStageViewer(QMainWindow)` that:
1. Accepts a list of `StagePaths` dataclass instances (one per stage)
2. Loads data for the initial stage in the constructor
3. On dropdown change, reloads CSV + forearm, clears and reinits VTK actors,
   recreates the NeuralDataPanel, and restores the frame position
4. Follows all established VTK design invariants (never `plotter.clear()` in
   `_update_frame()`, in-place `DeepCopy` for contact points,
   `ResetCameraClippingRange()` before every render)

The QComboBox dropdown pattern is taken directly from `ForearmStageInspector`,
which already implements session + stage selection with `currentIndexChanged`
signal and `blockSignals()` guards.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone viewer class | Clean separation, no risk of breaking existing viewers, follows project pattern | Some code duplication with PostprocessedSceneViewer | **Chosen** |
| Extend PostprocessedSceneViewer with dropdown | Code reuse | Constructor tightly coupled to single stage, advanced-mode ICP/PCA transform chain doesn't generalize, high refactoring risk | Rejected |
| Add stage tabs to BeforeAfterStepViewer | Reuses existing dual-plotter | Wrong UX model (tabs vs dropdown), would need major restructure from side-by-side to single-panel | Rejected |

### Architecture Changes

```
code/src/postprocessing/gui/
├── __init__.py                         # MODIFY — add PostprocessingStageViewer export
├── postprocessed_scene_viewer.py       # UNCHANGED — template reference
├── before_after_step_viewer.py         # UNCHANGED
├── forearm_stage_inspector.py          # UNCHANGED — dropdown pattern reference
└── postprocessing_stage_viewer.py      # NEW — ~400-500 lines
```

**Data model:**
```python
@dataclass
class StagePaths:
    stage_label: str                              # e.g. "Merged (Raw)"
    csv_path: Optional[Path]                      # Stage's CSV file
    forearm: Optional[Union[Path, o3d.PointCloud]] # PLY path or preloaded geometry
    coordinate_frame: str                         # "camera" or "pca"
```

**Stage-to-data mapping:**

| Stage | CSV directory | CSV name pattern | Forearm source | Coord frame |
|-------|--------------|-----------------|----------------|-------------|
| 0: Merged (Raw) | `blocks_merged/` | `*_merged_data.csv` | Per-video forearm from catalog | camera |
| 1: ICP Registered | `blocks_registered/` | `*_merged_data.csv` | Unified registered PLY | camera |
| 2: PCA Calibrated | `blocks_pca_calibrated/` | `*_merged_data_pca-xyz.csv` | `forearm_pca_calibrated/*.ply` | pca |
| 3: Contact Projected | `blocks_contact_projected/` | `*_merged_data_pca-xyz.csv` | `forearm_pca_calibrated/*.ply` | pca |
| 4: RF Centered | `blocks_rf_centered/` | `*_merged_data_pca-xyz.csv` | `forearm_rf_centered/*.ply` | pca |

**Knowledge base constraints applied:**
- CuPy import order: script already has early-init pattern (no change needed)
- Qt `itemChanged` signal recursion: use `blockSignals()` when programmatically
  setting the dropdown index during stage switching

**Reusable code (no re-implementation):**
- `NeuralDataPanel` from `merging.gui.neural_kinect_scene_viewer` — imported directly
- `_load_per_video_forearm()` and `_load_unified_forearm()` from
  `postprocess_visualization.py` — called for camera-space stages
- `resolve_before_after_paths()` — reference for path patterns

---

## Implementation Plan

### Phase 1: New viewer class
**Goal:** Create `PostprocessingStageViewer` with full stage-switching capability
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 1.1 — Define `StagePaths` dataclass with `stage_label`, `csv_path`, `forearm`, `coordinate_frame` fields
- [x] 1.2 — Define stage label constants list: "Merged (Raw)", "ICP Registered", "PCA Calibrated", "Contact Projected", "RF Centered"
- [x] 1.3 — Copy `_parse_contact_points_cell()` utility (follows established duplication pattern across 3 existing viewers)
- [x] 1.4 — Implement `PostprocessingStageViewer.__init__()`: accept `stage_paths`, `recording_name`, `initial_stage`; call `_load_stage_data()`, `_build_ui()`, `_init_actors()`; set up play/drag timers
- [x] 1.5 — Implement `_load_stage_data(stage_idx)`: read CSV, partition into `_full_df`/`_kinect_df`, pre-parse contact points, load forearm PLY (handle Path vs Open3D PointCloud vs None), compute contact centroid, set `_total_frames`
- [x] 1.6 — Implement `_build_ui()`: stage QComboBox in top bar, QtInteractor plotter (stretch=4), right scrollable panel (220px, visibility checkboxes + point-size sliders), frame control bar (slider + label + recenter + play/pause), NeuralDataPanel at bottom
- [x] 1.7 — Implement `_init_actors()`: register forearm point cloud, contact points (empty PolyData seed for DeepCopy), text label, axes, bounding proxy. Camera setup based on coordinate frame
- [x] 1.8 — Implement `_update_frame(frame_idx)`: DeepCopy contact points into mesh, toggle forearm visibility, `ResetCameraClippingRange()` + `render()`, update neural panel cursor, update frame label
- [x] 1.9 — Implement `_on_stage_changed(index)`: stop playback, save frame index, detect coordinate-frame change, `_load_stage_data()`, `plotter.clear()` + `_init_actors()`, reset camera if coordinate frame changed, recreate NeuralDataPanel, restore clamped frame index, call `_update_frame()`
- [x] 1.10 — Implement Qt lifecycle methods: `showEvent`/`_deferred_start`, `closeEvent`, slider handlers with 80ms drag throttle, play/pause timer, visibility/point-size toggle handlers, `_recenter_view()`
- [x] 1.11 — Add `PostprocessingStageViewer` to `code/src/postprocessing/gui/__init__.py` imports and `__all__`

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — NEW, ~400-500 lines
- `code/src/postprocessing/gui/__init__.py` — Add import + `__all__` entry

**Dependencies:** None

### Phase 2: Path resolver and script integration
**Goal:** Wire the viewer into the postprocess visualization workflow
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 2.1 — Add import of `StagePaths` and `PostprocessingStageViewer` to `postprocess_visualization.py`
- [x] 2.2 — Implement `resolve_stage_paths(config: KinectConfig) -> List[StagePaths]`: map each of 5 stages to a `StagePaths` instance using existing `_load_per_video_forearm()` / `_load_unified_forearm()` helpers for camera-space stages, and directory path resolution for PCA-space stages
- [x] 2.3 — Implement `run_single_session_pipeline_stage_viewer(config, dag_handler)` launcher: check `can_run("view_postprocessing_stages")`, call `resolve_stage_paths()`, validate at least one stage has a CSV, create QApplication + viewer, `exec_()`, `processEvents()`, `mark_completed()`
- [x] 2.4 — Add call to `run_single_session_pipeline_stage_viewer()` in the per-block loop of `run_batch_sequentially()`

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — Add resolver function, launcher function, batch integration (~80-100 lines)

**Dependencies:** Phase 1

### Phase 3: DAG configuration
**Goal:** Register the new task in the postprocess visualization DAG
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 3.1 — Add `view_postprocessing_stages` task entry to DAG config with description, `enabled: false`, `options: {}`, `depends_on: []`

**Files Modified:**
- `configs/postprocess_visualization_dag.yaml` — Add task entry (~6 lines)

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

No unit tests for this feature — GUI viewers in this project are not unit-tested
(no existing test pattern for PyQt5+PyVista viewers). All existing viewers
(`NeuralKinectViewer`, `PostprocessedSceneViewer`, `BeforeAfterStepViewer`,
`ForearmStageInspector`) rely on manual verification.

### Manual Verification

- [ ] Enable `view_postprocessing_stages: enabled: true` in DAG config and run the script
- [ ] Viewer opens with stage dropdown showing all 5 stage labels
- [ ] Default stage (first available with data) loads correctly — forearm visible, contact points rendered, neural panel populated
- [ ] Switch from stage 0 (Merged) to stage 1 (ICP Registered) — data updates, camera preserved (both camera-space)
- [ ] Switch from stage 1 to stage 2 (PCA Calibrated) — data updates, camera resets (coordinate frame boundary)
- [ ] Switch from stage 3 to stage 4 (RF Centered) — data updates, camera preserved (both PCA-space)
- [ ] Frame slider navigates frames within a stage
- [ ] Frame position preserved when switching stages (clamped to new total)
- [ ] Play/pause animation works
- [ ] Recenter button resets camera focal point
- [ ] Visibility checkboxes toggle forearm and contact points independently
- [ ] Point-size sliders affect forearm and contact point rendering

### Edge Cases

- [ ] Stage with missing CSV — dropdown entry present but shows "No data available" message, frame controls disabled
- [ ] Stage with missing forearm PLY — contact points still render, forearm area empty
- [ ] All stages missing except one — viewer opens on the available stage
- [ ] Switching stages rapidly — no crashes or VTK context errors

---

## Documentation Plan

- [ ] No README changes needed (internal visualization tool)
- [ ] No CLAUDE.md changes needed (no architectural pattern changes)
- [ ] No user guide needed (internal tool, discoverable via DAG config)

---

## Rollback Plan

1. Delete `code/src/postprocessing/gui/postprocessing_stage_viewer.py`
2. Revert changes to `__init__.py`, `postprocess_visualization.py`, DAG config
3. All changes are additive — no existing functionality is modified or removed

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NeuralDataPanel recreation causes memory leak on repeated stage switches | Low | Med | Call `deleteLater()` on old panel before creating new one; test with 10+ rapid switches |
| VTK render context not properly released during `plotter.clear()` on stage switch | Low | High | Follow existing `processEvents()` pattern; add double-pump if needed |
| Large CSV reload causes UI freeze during stage switch | Med | Low | CSVs are typically 50-200k rows; `pd.read_csv` handles this in <1s. Could add progress cursor if needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Viewer class | ~400-500 lines new code | None |
| Phase 2: Script integration | ~80-100 lines modified | Phase 1 |
| Phase 3: DAG config | ~6 lines | Phase 2 |

---

## References

- Related viewers: `PostprocessedSceneViewer`, `ForearmStageInspector`, `NeuralKinectViewer`
- Spatial pipeline note: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`
- CuPy import order: `docs/development/knowledge-base/note-cupy-import-order.md`
- Qt signal recursion: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`

---
