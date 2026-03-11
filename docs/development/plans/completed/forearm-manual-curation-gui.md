# Plan: Manual Point Cloud Curation GUI

**Date:** 2026-03-05
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/forearm-manual-curation`

---

## Overview

Add a manual curation GUI step (step 1.5) between forearm extraction and cleaning in the forearm extraction pipeline. A PyQt5 + PyVista interactive window displays the raw extracted point cloud with its real colors and lets the operator box-select and remove unwanted points (table edges, clothing fragments, mis-segmented regions) before the cloud proceeds to automated cleaning.

## Problem Statement

After extraction, some point clouds contain artifacts that automated cleaning (duplicate-Z removal) cannot reliably handle. The operator has no way to intervene between extraction and cleaning without manual scripting. Artifacts survive into normals and mesh reconstruction, degrading quality and requiring costly re-runs of the entire pipeline.

## Goals

### In Scope
1. New `ForearmCurationGUI` class (PyQt5 `QMainWindow` + `pyvistaqt.QtInteractor`) for interactive point removal
2. New script-level function `curate_forearm_pointcloud()` with skip-if-exists logic
3. Integration into `execute_frame_batch()` as DAG task `curate_forearm` (step 1.5)
4. Persistence of curated PLY and curation metadata JSON (removed indices)
5. Re-editing support: loading previously saved removed indices when reopening

### Out of Scope
- Automated outlier detection or point-removal suggestions
- Undo/redo stack within the GUI (reset-all is sufficient for V1)
- Integration with the automated kinect workflow (`preprocess_workflow_kinect_auto.py`)
- GPU-accelerated rendering

## Success Criteria

- [ ] `curate_forearm` task appears in the DAG config; `clean_forearm` depends on it
- [ ] GUI opens with the raw point cloud displayed using its real RGB colors
- [ ] User can box-select points for removal; overlay shows removed points in red
- [ ] "Validate & Export" saves `_curated.ply` and `_curation_metadata.json`
- [ ] Clean step reads from `_curated.ply` instead of raw PLY when curation output exists
- [ ] Skip logic: if curated outputs exist and `force_processing` is false, the GUI does not open
- [ ] Re-opening the GUI with existing metadata pre-highlights previously removed points

---

## Technical Design

### Approach

Mirror the architecture of `HandMaskSelectorGUI` but adapted for point clouds (no mesh faces):

- **Input is `o3d.geometry.PointCloud`** (not a mesh). Converted to `pv.PolyData` points-only.
- **Real RGB colors** from the point cloud are used for the base layer (not a flat color).
- **Default mode is REMOVE** (checkbox checked by default), since the user's primary action is removing bad points.
- **No wireframe layer** -- point clouds have no faces to render.
- Same frustum (box) picking mechanism via `enable_cell_picking(through=True)` with `orig_ids` point data for index mapping.
- Same `excluded_mask: np.ndarray` (bool) state management pattern.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PyQt5 + PyVista (pyvistaqt) | Proven pattern (`HandMaskSelectorGUI`), frustum picking, real-time overlay, batch ops | Adds pyvistaqt to forearm pipeline (already a project dependency) | **Chosen** |
| Open3D native GUI | Same lib as rest of pipeline | No frustum picking; SceneWidget layout issues (see `note-open3d-scenewidget-layout.md`); no built-in selection mode | Rejected |
| Standalone PyVista (no Qt) | Simpler setup | No side panel for controls, harder to integrate buttons/signals | Rejected |

### Knowledge Base Relevance

- `note-open3d-scenewidget-layout.md` -- Validates avoiding Open3D's native GUI for interactive tools.
- `note-cupy-import-order.md` -- Not applicable; no CuPy usage.
- `note-forearm-icp-registration.md` -- Not directly applicable; curation happens before registration.

### Architecture Changes

New module:

```
code/src/preprocessing/forearm_extraction/curation/
  __init__.py                        -- exports
  forearm_curation_gui.py            -- ForearmCurationGUI (QMainWindow + QtInteractor)
  curation_metadata_filehandler.py   -- JSON load/save for curation metadata
```

New script function:

```
code/scripts/_3_preprocessing/_3_forearm_extraction/
  curate_forearm_pointcloud.py       -- orchestration function
```

Data flow change:

```
Before:  raw.ply --> [Clean] --> cleaned.ply
After:   raw.ply --> [Curate GUI] --> curated.ply + curation_metadata.json --> [Clean] --> cleaned.ply
```

Curation metadata schema:

```json
{
  "removed_point_indices": [12, 45, 67],
  "total_points_original": 15000,
  "total_points_remaining": 14997,
  "timestamp": "2026-03-05T14:30:00"
}
```

---

## Implementation Plan

### Phase 1: Data Layer
**Goal:** Curation metadata filehandler for JSON persistence of removed indices.

- [ ] Create `curation/__init__.py` with exports
- [ ] Create `curation/curation_metadata_filehandler.py` -- `CurationMetadataFileHandler` with static `save(data, path)` and `load(path)` methods (pattern: `ForearmSegmentationParamsFileHandler`)

**Files Created:**
- `code/src/preprocessing/forearm_extraction/curation/__init__.py`
- `code/src/preprocessing/forearm_extraction/curation/curation_metadata_filehandler.py`

**Dependencies:** None

### Phase 2: GUI
**Goal:** Interactive point cloud curation GUI modeled on `HandMaskSelectorGUI`.

- [ ] Create `curation/forearm_curation_gui.py` -- `ForearmCurationGUI(QMainWindow)`
- [ ] Constructor: `(point_cloud: o3d.geometry.PointCloud, existing_removed_indices: Optional[List[int]] = None)`
- [ ] Convert o3d PointCloud to `pv.PolyData` (points only); use real RGB from `pcd.colors`
- [ ] `excluded_mask: np.ndarray` (bool) -- `True` = removed. Pre-set from `existing_removed_indices`
- [ ] `point_data["orig_ids"]` for pick-callback index mapping
- [ ] Layout: `QtInteractor` (left, stretch=3) + controls panel (right, 300px)
- [ ] Base layer: real colors, `point_size=5`, `pickable=True`
- [ ] Overlay layer: removed points as red, `point_size=10`, `pickable=False`
- [ ] Default filter mode: REMOVE (checkbox checked by default)
- [ ] R key toggles selection mode (frustum picking via `enable_cell_picking(through=True)`)
- [ ] Batch ops: "Keep All" (reset mask) and "Remove All" (mask all)
- [ ] Signal: `curation_validated = pyqtSignal(list)` -- emits removed indices on "Validate & Export"
- [ ] Update `curation/__init__.py` exports

**Files Created:**
- `code/src/preprocessing/forearm_extraction/curation/forearm_curation_gui.py`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/curation/__init__.py` -- add export

**Dependencies:** Phase 1

### Phase 3: Script Function
**Goal:** Orchestration function with skip-if-exists logic.

- [ ] Create `curate_forearm_pointcloud.py` in scripts directory
- [ ] `curate_forearm_pointcloud(input_ply_path, output_ply_path, output_metadata_path, *, force_processing=False)`
- [ ] Use `should_process_task()` for skip check (same pattern as `clean_forearm_pointcloud`)
- [ ] Load raw PLY via `o3d.io.read_point_cloud()`
- [ ] Load existing removed indices from metadata if file exists (re-editing support)
- [ ] Create `QApplication` if none exists (`QApplication.instance() or QApplication(sys.argv)`)
- [ ] Instantiate GUI, connect `curation_validated` signal, `gui.show()` + `app.exec_()`
- [ ] On validation: `pcd.select_by_index(kept_indices)` to save curated PLY + metadata via `CurationMetadataFileHandler`
- [ ] On close without validation: skip saving, print warning

**Files Created:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/curate_forearm_pointcloud.py`

**Dependencies:** Phase 1, Phase 2

### Phase 4: Pipeline Integration
**Goal:** Wire curation into pipeline, DAG config, and FrameBatch.

- [ ] Add `curated_ply: Path` and `curated_meta: Path` to `FrameBatch` dataclass
- [ ] Populate new paths in `plan_frame_batch()`: `{base}_curated.ply`, `{base}_curation_metadata.json`
- [ ] Insert step 1.5 in `execute_frame_batch()` using `TaskExecutor('curate_forearm', ...)`
- [ ] Update clean step input: `batch.curated_ply if batch.curated_ply.exists() else batch.raw_ply`
- [ ] Update `run_session()` task simulation loop to include `'curate_forearm'`
- [ ] Update step numbering in print messages: `[1/4]...[4/4]` to `[1/5]...[5/5]`
- [ ] Add export to `_3_forearm_extraction/__init__.py`
- [ ] Add exports to `forearm_extraction/__init__.py`
- [ ] Update DAG config: add `curate_forearm` task, change `clean_forearm.depends_on` to `[curate_forearm]`

**Files Modified:**
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` -- FrameBatch, plan_frame_batch, execute_frame_batch, run_session, imports
- `code/scripts/_3_preprocessing/_3_forearm_extraction/__init__.py` -- add export
- `code/src/preprocessing/forearm_extraction/__init__.py` -- add exports
- `configs/preprocess_forearm_manual_dag.yaml` -- add task, update dependency

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `CurationMetadataFileHandler.save()` produces valid JSON with correct schema
- [ ] `CurationMetadataFileHandler.load()` round-trips saved data correctly
- [ ] `CurationMetadataFileHandler.load()` returns `None` or empty for missing file

### Integration Tests
- [ ] Full pipeline run with curation enabled: raw.ply to curated.ply to cleaned.ply chain works
- [ ] Pipeline with curation disabled (`enabled: false`): clean step falls back to raw.ply
- [ ] Re-run with `force_processing: false` skips GUI when curated outputs exist

### Manual Verification
- [ ] GUI opens with correct point cloud colors (not flat color)
- [ ] Box selection in REMOVE mode highlights selected points as red large spheres
- [ ] Switching to KEEP mode and selecting red points restores them
- [ ] "Remove All" masks all points; "Keep All" clears the mask
- [ ] R key toggles between navigation and selection modes
- [ ] "Validate & Export" closes window and saves both files
- [ ] Re-opening GUI on same frame pre-highlights previously removed points
- [ ] Closing window via X button (without validate) does not produce output files

### Edge Cases
- [ ] Empty point cloud (0 points) -- GUI shows empty viewport, allows immediate validation
- [ ] Point cloud with no colors -- falls back to a default color (light grey)
- [ ] All points removed -- curated PLY is empty, metadata reflects 0 remaining
- [ ] `existing_removed_indices` with out-of-range values -- silently filtered

---

## Documentation Plan

- [ ] Add inline docstrings to all new public functions and classes
- [ ] Update `execute_frame_batch()` docstring from "four processing steps" to "five"
- [ ] Update `docs/development/knowledge-base/README.md` if engineering notes emerge

---

## Rollback Plan

1. Revert changes to `preprocess_pipeline_extract_forearm_manual.py` (restore 4-step batch)
2. Revert DAG config: remove `curate_forearm` task, restore `clean_forearm.depends_on: [extract_forearm]`
3. No data migrations needed: curated PLY and metadata files are additive artifacts; existing raw.ply files are never modified
4. If curation step is removed, clean_forearm naturally falls back to raw.ply

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `QApplication` conflict with tkinter dialogs in same pipeline run | Medium | High | Check `QApplication.instance()` first; tkinter dialogs run in separate pipeline phases, not concurrently with Qt |
| PyVista frustum picking on points-only PolyData returns unexpected data | Low | High | Test with `enable_cell_picking` on points-only PolyData early; fall back to `enable_rubber_band_selection` if needed |
| Large point clouds (100K+ points) cause slow overlay refresh | Medium | Medium | Profile `_refresh_visuals` early; consider scalar coloring instead of separate actor if slow |
| User closes GUI without validating; pipeline proceeds with missing curated.ply | Low | Medium | Clean step has fallback: `curated_ply if exists else raw_ply`; log warning when fallback triggered |

---

## References

- Inspiration GUI: `code/src/preprocessing/motion_analysis/hand_tracking/gui/hand_mask_selector_gui.py`
- Pipeline script: `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
- Clean step (pattern reference): `code/scripts/_3_preprocessing/_3_forearm_extraction/clean_forearm_pointcloud.py`
- FileHandler pattern: `code/src/preprocessing/forearm_extraction/data_access/forearm_segmentation_parameters_filehandler.py`
- Skip-if-exists utility: `code/src/utils/should_process_task.py`
- DAG config: `configs/preprocess_forearm_manual_dag.yaml`
- Related active plan: `docs/development/plans/active/multi-snapshot-forearm-registration.md`
