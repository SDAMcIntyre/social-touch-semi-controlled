# Plan: Relocate RF Camera Angle Picker to Analysis Workflow

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** Superseded by pick-rf-camera-angle (picker built directly in analysis package)
**Branch:** `feature/pick-rf-camera-angle` (existing)

---

## Overview

**What:** Move the `pick_rf_camera_angle` interactive viewer from the postprocessing workflow into the `map_receptive_fields_clustered` flow in the analysis workflow.

**Why:** The camera angle picker is conceptually part of the RF mapping pipeline — it visualises RF-centered forearms with selectivity heatmaps. Placing it inside the RF cluster mapping flow keeps related functionality together and simplifies the postprocessing workflow.

**How:** Relocate the task module and GUI into `analysis/receptive_field_mapping/`, call `pick_rf_camera_angle_batch()` at the end of `map_receptive_fields_clustered_flow`, and remove all traces from the postprocessing workflow.

## Problem Statement

The camera angle picker currently lives in the postprocessing workflow as a post-batch step, but it operates on RF-centered data and serves the RF mapping pipeline. This split makes the data flow harder to follow and couples the postprocessing workflow to an analysis concern.

## Goals

### In Scope

1. Relocate task module and GUI from postprocessing packages to `analysis/receptive_field_mapping/`
2. Integrate the picker call inside `map_receptive_fields_clustered_flow` (after `run_cluster_rf_mapping()`)
3. Auto-detect behavior: GUI launches only when `camera_params.json` is missing for any session
4. DAG option `pick_camera_angle_force` to force-open the GUI even when all camera params exist
5. Remove all `pick_rf_camera_angle` traces from the postprocessing workflow and its DAG config
6. Remove legacy single-session wrapper `pick_rf_camera_angle()` (no callers remain)

### Out of Scope

- Consuming `camera_params.json` in `render_forearm_heatmap` (future work)
- Any GUI or functional changes to the picker itself
- Changes to `camera_params.json` output format

## Success Criteria

- [ ] `pick_rf_camera_angle_batch` is called from inside `map_receptive_fields_clustered_flow`
- [ ] GUI auto-launches when any session lacks `camera_params.json`
- [ ] GUI is skipped when all sessions have `camera_params.json` and `pick_camera_angle_force: false`
- [ ] `pick_camera_angle_force: true` forces the GUI to open regardless
- [ ] Postprocessing workflow runs without any reference to `pick_rf_camera_angle`
- [ ] No import errors — relocated modules resolve correctly

---

## Technical Design

### Approach

Relocate the two existing modules (task logic + GUI) into the `analysis/receptive_field_mapping/` package. Add a deferred call to `pick_rf_camera_angle_batch()` at the end of `map_receptive_fields_clustered_flow`. The existing auto-detect logic in `pick_rf_camera_angle_batch` already handles the conditional launch — it filters out sessions with existing `camera_params.json` unless `force_processing=True`, and skips the GUI entirely if no sessions remain.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Inside `map_receptive_fields_clustered_flow` | Tight coupling with RF mapping, no separate DAG task | Slightly larger flow function | **Chosen** — user preference |
| Separate post-batch step in analysis workflow | Independent toggle, mirrors postprocessing pattern | Analysis workflow has no post-batch pattern; adds complexity | Rejected |
| Keep in postprocessing, call from analysis | No file moves | Conceptual mismatch remains | Rejected |

### Architecture Changes

**File relocations:**

| Current path | New path |
|---|---|
| `code/scripts/_5_postprocessing/pick_rf_camera_angle.py` | `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` |
| `code/src/postprocessing/gui/rf_camera_angle_picker.py` | `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py` |

**New files:**
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — new subpackage

**Modified files:**
- `code/scripts/analysis_workflow.py` — add picker call + new parameter
- `configs/analyse_workflow_dag.yaml` — add `pick_camera_angle_force` option
- `code/src/analysis/receptive_field_mapping/__init__.py` — export new symbols
- `code/scripts/postprocess_workflow_kinect_auto.py` — remove post-batch call + import
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — remove task entry
- `code/scripts/_5_postprocessing/__init__.py` — remove export
- `code/src/postprocessing/gui/__init__.py` — remove export

**Deleted files:**
- `code/scripts/_5_postprocessing/pick_rf_camera_angle.py`
- `code/src/postprocessing/gui/rf_camera_angle_picker.py`

### Option Plumbing

```
analyse_workflow_dag.yaml
  map_receptive_fields_clustered.options.pick_camera_angle_force: false
      |
      v
run_batch_analysis()  [kwargs construction]
  if "pick_camera_angle_force" in options:
      kwargs["pick_camera_angle_force"] = options[...]
      |
      v
map_receptive_fields_clustered_flow(pick_camera_angle_force=False)
      |
      v
pick_rf_camera_angle_batch(force_processing=pick_camera_angle_force)
  -> auto-detect: filters sessions without camera_params.json
  -> if no sessions remain, skips GUI
  -> if force, shows all sessions
```

### Architecture Constraints (from Knowledge Base)

- **CuPy import order:** The relocated task module imports from `preprocessing.*`, so `cupy` must be imported first at the top of `rf_camera_angle_task.py`
- **Deferred import of GUI:** PyQt5/pyvistaqt are heavy; the import inside `pick_rf_camera_angle_batch` stays deferred (only when GUI actually launches)

---

## Implementation Plan

### Phase 1: File Relocation
**Goal:** Move task module and GUI to `analysis/receptive_field_mapping/`

- [x] Create `code/src/analysis/receptive_field_mapping/gui/__init__.py`
- [x] Move `rf_camera_angle_picker.py` to `analysis/receptive_field_mapping/gui/` (no content changes)
- [x] Move `pick_rf_camera_angle.py` to `rf_camera_angle_task.py` under `receptive_field_mapping/`:
  - Update GUI import: `from analysis.receptive_field_mapping.gui.rf_camera_angle_picker import RFCameraAnglePicker`
  - Remove legacy `pick_rf_camera_angle()` wrapper
  - Keep CuPy import guard
- [x] Update `code/src/analysis/receptive_field_mapping/__init__.py` — export `pick_rf_camera_angle_batch`, `SessionSceneData`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — new
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py` — relocated
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — relocated + updated imports
- `code/src/analysis/receptive_field_mapping/__init__.py` — updated exports

**Dependencies:** None

### Phase 2: Analysis Workflow Integration
**Goal:** Call the picker from inside `map_receptive_fields_clustered_flow`

- [x] Add `pick_camera_angle_force: bool = False` parameter to `map_receptive_fields_clustered_flow`
- [x] After `run_cluster_rf_mapping()`, add deferred import and call to `pick_rf_camera_angle_batch`:
  ```python
  from analysis.receptive_field_mapping.rf_camera_angle_task import pick_rf_camera_angle_batch
  from analysis.touch_analytics.pipeline_shared import session_id_from_path

  session_output_dirs = {
      session_id_from_path(csv_path): csv_path.parent
      for csv_path, _ in input_items
  }
  pick_rf_camera_angle_batch(session_output_dirs, force_processing=pick_camera_angle_force)
  ```
- [x] In `run_batch_analysis()` kwargs construction, add passthrough:
  ```python
  if "pick_camera_angle_force" in options:
      kwargs["pick_camera_angle_force"] = options["pick_camera_angle_force"]
  ```
- [x] Add `pick_camera_angle_force: false` to `map_receptive_fields_clustered.options` in `analyse_workflow_dag.yaml`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — flow function + kwargs
- `configs/analyse_workflow_dag.yaml` — new option

**Dependencies:** Phase 1

### Phase 3: Remove from Postprocessing
**Goal:** Clean up all postprocessing references

- [x] Remove `pick_rf_camera_angle_batch` import from `postprocess_workflow_kinect_auto.py`
- [x] Remove post-batch block (lines 370-381) from `run_batch_postprocessing()`
- [x] Remove `pick_rf_camera_angle` task entry from `postprocess_workflow_kinect_auto_dag.yaml`
- [x] Remove `pick_rf_camera_angle` import from `code/scripts/_5_postprocessing/__init__.py`
- [x] Remove `RFCameraAnglePicker` import from `code/src/postprocessing/gui/__init__.py`
- [x] Delete `code/scripts/_5_postprocessing/pick_rf_camera_angle.py`
- [x] Delete `code/src/postprocessing/gui/rf_camera_angle_picker.py`

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — remove import + post-batch block
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — remove task entry
- `code/scripts/_5_postprocessing/__init__.py` — remove export
- `code/src/postprocessing/gui/__init__.py` — remove export
- Delete 2 old source files

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification

- [ ] Run analysis workflow with `pick_camera_angle_force: false` and no existing `camera_params.json` — GUI launches after RF cluster mapping
- [ ] Run again with all `camera_params.json` present — GUI is skipped
- [ ] Set `pick_camera_angle_force: true` — GUI launches regardless
- [ ] Run postprocessing workflow — completes without any reference to `pick_rf_camera_angle`
- [ ] Verify import: `from analysis.receptive_field_mapping.rf_camera_angle_task import pick_rf_camera_angle_batch` works

### Edge Cases

- [ ] `map_receptive_fields_clustered` disabled in DAG — no picker, no error
- [ ] No sessions with valid RF centering — picker skipped with log message
- [ ] Only one valid session — picker launches with single dropdown entry

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/pick-rf-camera-angle.md` — reference this relocation plan
- [ ] Inline docstring on `pick_camera_angle_force` parameter in flow function
- [ ] Comment in `analyse_workflow_dag.yaml` explaining the option

---

## Rollback Plan

1. Restore files from git history to their original postprocessing locations
2. Re-add post-batch call in `postprocess_workflow_kinect_auto.py`
3. Re-add DAG entry in `postprocess_workflow_kinect_auto_dag.yaml`
4. Revert `analysis_workflow.py` and `analyse_workflow_dag.yaml` changes
5. No data migration needed — `camera_params.json` files are per-session, location-independent

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Import path breakage after relocation | Med | Low | Verify imports before deleting old files |
| CuPy guard missed in relocated file | Low | Med | Copy file first, verify guard is present |
| PyQt5 blocking Prefect flow thread | Low | Low | Existing behavior — already works in postprocessing |
| Postprocessing workflow breaks if old references remain | Low | Med | Grep for all `pick_rf_camera_angle` references after cleanup |
