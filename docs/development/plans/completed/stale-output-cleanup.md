# Plan: Stale Output Cleanup

**Date:** 2026-03-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-20 14:07
**Branch:** feature/stale-output-cleanup

---

## Overview

When a pipeline task fails after upstream inputs change, its old outputs from a previous successful run remain on disk, causing downstream tasks to silently consume stale files. This plan introduces a centralized `clean_task_outputs` utility function — placed alongside `should_process_task` — that each process function calls to delete its output files before processing begins.

## Problem Statement

The pipeline processes data in sequential stages (A -> B -> C). When Task A's outputs change and Task B is re-run but fails on some sessions, Task B's old outputs persist. Task C (or the analysis pipeline, which discovers inputs via directory globbing) silently picks up these stale outputs as if they were current.

This is particularly dangerous because:
- The analysis pipeline discovers inputs via `glob("*_touch_summary.csv")`, finding ALL files including stale ones
- The pipeline runs per-block/per-session in batch — some blocks may succeed while others fail, leaving a mix of current and stale outputs
- There is no warning or error — the pipeline appears to succeed

## Goals

### In Scope
1. Create a centralized `clean_task_outputs()` function in `code/src/utils/should_process_task.py`
2. Add cleanup calls to process functions called by **automatic pipelines only**, after `should_process_task` returns `True` and before actual processing
3. Protect incremental/living-output functions by excluding them from cleanup

### Out of Scope
- **Manual pipelines** — `preprocess_workflow_kinect_manual.py`, `preprocess_pipeline_extract_forearm_manual.py` and all their GUI-based process functions
- **Visualisation pipelines** — `preprocess_workflow_kinect_visualisation.py`, `merging_pipeline_neuron_to_kinect_visualisation.py`
- Changes to pipeline orchestrators — cleanup is the process function's responsibility
- Changes to `should_process_task` logic itself
- Changes to individual task function signatures
- Manifest/registry system for tracking pipeline run provenance

## Success Criteria

- [ ] When an auto-pipeline task fails, its output files are absent (not stale from a previous run)
- [ ] `should_process_task()` correctly triggers reprocessing when outputs are missing after cleanup
- [ ] `track_objects_in_video` (incremental) still loads and extends existing tracked data
- [ ] Manual pipeline functions have NO `clean_task_outputs` calls
- [ ] Happy-path execution produces identical results to current behavior
- [ ] The cleanup function is importable from the same module as `should_process_task`

---

## Technical Design

### Approach

**Process-function-level pre-run cleanup with a shared utility.** Each process function calls `clean_task_outputs(output_paths)` immediately after `should_process_task` returns `True`, using the same output paths it already passes to `should_process_task`. This deletes stale files so that if the task fails, no old outputs remain for downstream consumption.

This works because:
- `should_process_task()` already handles "output missing" -> returns `True` (reprocessing required)
- Downstream tasks that glob directories will simply not find the deleted file
- Each process function controls whether it calls cleanup (opt-in per function)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Process-function-level cleanup, auto pipelines only | One definition, each function controls its own cleanup, manual pipelines untouched | Must add one line to ~30 functions | **Chosen** |
| Orchestrator-level `pure_outputs` lambdas (previous plan) | Centralized in stage definitions | Wrong layer — puts cleanup responsibility in orchestrator, requires enumerating outputs per stage in a separate place from where they're used | Rejected (rolled back 2026-03-20) |
| Failure sentinel files | No file deletion needed | Requires downstream awareness of sentinels, couples tasks | Rejected |
| Output quarantine (move to `.quarantine/`) | Preserves files for recovery | Complex state machine, quarantine dir management | Rejected |

### Architecture Changes

No new modules or classes. Changes are localized to:

1. **`code/src/utils/should_process_task.py`** — Extract `_normalize_to_paths` to module level (shared by both functions), add `clean_task_outputs()` function.

2. **Auto-pipeline process functions only (~30 files)** — Add one import symbol and one function call line after the `should_process_task` check. Manual and visualisation pipeline functions are not modified.

### Function Design

```python
def clean_task_outputs(output_paths: PathInput) -> None:
    """Delete output files before processing to prevent stale artifacts.

    Call immediately after should_process_task() returns True,
    before actual processing begins.
    """
    paths = _normalize_to_paths(output_paths)
    for p in paths:
        if p is None:
            continue
        if p.exists() and p.is_file():
            try:
                p.unlink()
                print(f"  Cleaned stale output: '{p.name}'")
            except PermissionError:
                print(f"  Could not delete '{p}' (file locked). Proceeding anyway.")
```

Key design choices:
- Same `PathInput` type as `should_process_task` — accepts single path or list
- Filters `None` entries (some functions have optional output paths like `output_metadata_path=None`)
- Only deletes files, not directories
- `PermissionError` caught — on Windows files can be locked; stale output is better than a crash

### Calling Pattern

```python
# Before
from utils.should_process_task import should_process_task

if not should_process_task(output_paths=output_path, input_paths=input_path, force=force_processing):
    return output_path
# ... processing ...

# After
from utils.should_process_task import should_process_task, clean_task_outputs

if not should_process_task(output_paths=output_path, input_paths=input_path, force=force_processing):
    return output_path
clean_task_outputs(output_path)
# ... processing ...
```

### Excluded Functions

| Function | File | Reason |
|----------|------|--------|
| `track_objects_in_video` | `_1_sticker_tracking/track_handstickers_roi.py` | Loads existing output incrementally, merges new tracking data |

### Excluded Pipelines (manual + visualisation)

All process functions called exclusively by these pipelines are **not modified**:

| Pipeline | Orchestrator | Functions excluded |
|----------|-------------|-------------------|
| Manual preprocessing | `preprocess_workflow_kinect_manual.py` | `define_LED_roi`, `review_tracked_objects_in_video`, `define_handstickers_colorspaces_from_roi`, `define_handstickers_color_threshold`, `assign_stickers_location`, `define_hand_mask`, `define_trial_chunks`, `curate_hamer_hand_models`, `review_single_touches` |
| Forearm manual | `preprocess_pipeline_extract_forearm_manual.py` | `extract_participant_forearm`, `curate_forearm_pointcloud`, `clean_forearm_pointcloud`, `define_normals`, `define_forearm_mesh`, `apply_registration_transform` |
| Visualisation | `preprocess_workflow_kinect_visualisation.py`, `merging_pipeline_neuron_to_kinect_visualisation.py` | All visualisation-only functions |

### Knowledge Base Relevance Check

Reviewed all notes in `docs/development/knowledge-base/README.md`. No applicable notes — this feature concerns file lifecycle management, not framework quirks or import constraints.

---

## Implementation Plan

### Phase 1: Core Function and Normalizer Refactor
**Goal:** Add the `clean_task_outputs` function and make the path normalizer shared.

**Tasks:**
- [x] Task 1.1 — Extract `_normalize_to_paths` from inside `should_process_task` to module-level scope
- [x] Task 1.2 — Update `should_process_task` to call the module-level `_normalize_to_paths` (no behavior change)
- [x] Task 1.3 — Add `clean_task_outputs(output_paths: PathInput) -> None` function

**Files Modified:**
- `code/src/utils/should_process_task.py` — refactor normalizer, add new function

**Dependencies:** None

### Phase 2: Auto-Pipeline Preprocessing Functions
**Goal:** Add cleanup calls to process functions called by automatic pipelines only.

**Tasks:**
- [x] Task 2.1 — Primary processing (2 files):
  - `_2_primary_processing/_2_generate_rgb_depth_video/generate_mkv_stream_analysis.py`
  - `_2_primary_processing/_2_generate_rgb_depth_video/extract_color_to_mp4.py`
- [x] Task 2.2 — Auto preprocessing sticker tracking (8 files, excluding `track_handstickers_roi.py` and `fit_ellipses_on_correlation_videos.py`):
  - `consolidate_2d_tracking_data.py`, `create_color_correlation_videos.py`, `create_standardized_roi_videos.py`, `adjust_ellipse_centers_to_global_frame.py`, `standardize_handstickers_roi.py`, `extract_stickers_xyz_positions.py`, `correct_xyz_stickers_motion.py`
  - `fit_ellipses_on_correlation_videos.py` — **excluded**: loads existing output at startup and merges results incrementally (same pattern as `track_objects_in_video`); cleanup would delete accumulated data for unprocessed objects
- [x] Task 2.3 — Auto preprocessing hand tracking (2 files):
  - `track_hands_on_video.py`, `generate_3d_hand_in_motion.py`
- [x] Task 2.4 — Auto preprocessing LED tracking (3 files):
  - `generate_led_roi.py`, `track_LED_state_changes.py`, `validate_and_correct_LED_blinking.py`
- [x] Task 2.5 — Auto preprocessing somatosensory, metadata, unification (5 files):
  - `compute_somatosensory_characteristics.py`, `calculate_trial_id.py`, `generate_stimuli_metadata.py`, `find_single_touches.py`, `unify_datasets.py`

**Files Modified:** 21 files in `code/scripts/_2_primary_processing/` and `code/scripts/_3_preprocessing/`

**NOT modified (manual/visualisation-only):** `define_LED_roi.py`, `define_hand_mask.py`, `assign_stickers_location.py`, `curate_hamer_hand_models.py`, `define_trial_chunks.py`, `review_single_touches.py`, `review_tracked_objects_in_video.py`, `define_handstickers_colorspaces_from_roi.py`, `define_handstickers_color_threshold.py`, `extract_participant_forearm.py`, `curate_forearm_pointcloud.py`, `clean_forearm_pointcloud.py`, `define_normals.py`, `define_forearm_mesh.py`, `apply_registration_transform.py`, `unify_contact_and_led.py`, `track_handstickers_roi.py`

**Dependencies:** Phase 1

### Phase 3: Merging, Postprocessing, and Analysis Functions
**Goal:** Add cleanup calls to remaining auto-pipeline functions.

**Tasks:**
- [x] Task 3.1 — Merging (3 files):
  - `_4_merging/merge_neural_and_kinect_data.py`, `filter_merged_by_neural_quality.py`, `aggregate_blocks_session.py`
- [x] Task 3.2 — Postprocessing (4 files):
  - `_5_postprocessing/set_xyz_reference_from_gestures.py`, `project_contacts_onto_forearm.py`, `apply_icp_registration.py`, `export_forearm_pca_calibrated.py`
- [x] Task 3.3 — Analysis (5 files):
  - `code/src/analysis/touch_analytics/extraction_pipeline.py`, `clustering_pipeline.py`, `session_summary.py`, `touch_analysis.py`, `matrix_generation.py`

**Files Modified:** 12 files in `code/scripts/_4_merging/`, `code/scripts/_5_postprocessing/`, `code/src/analysis/touch_analytics/`

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] **Failure cleanup:** Run auto preprocessing on a session, force a middle task to fail -> confirm its output files are absent after the run
- [ ] **Analysis cleanup:** Run extraction on a session that will fail -> confirm `_touch_summary.csv` is removed -> confirm clustering skips it
- [ ] **Happy path:** Run full auto pipeline on a session that succeeds -> confirm all outputs are produced correctly (no regression)
- [ ] **Idempotency:** Run the same successful pipeline twice -> confirm `should_process_task` correctly skips up-to-date tasks
- [ ] **Incremental preservation:** Run `track_objects_in_video` -> confirm it still loads and extends existing tracked data
- [ ] **Manual pipeline unaffected:** Run forearm manual pipeline -> confirm no cleanup calls are present, existing outputs preserved between interactive steps

### Edge Cases
- [ ] Task with no outputs (e.g., validation tasks with empty `outputs: []`) -> cleanup is a no-op
- [ ] Function with `output_metadata_path=None` -> `None` is filtered, no error
- [ ] `force_processing: true` combined with cleanup -> outputs are deleted then recreated
- [ ] Output file locked by another process -> `PermissionError` caught, task proceeds

---

## Documentation Plan

- [ ] Add knowledge base note explaining the `clean_task_outputs` pattern and when to exclude a function

---

## Rollback Plan

All changes are additive (one new function, one new import + call per file). To rollback:

1. Remove `clean_task_outputs` calls from all process functions
2. Revert import lines to remove `clean_task_outputs`
3. Optionally revert the normalizer extraction (purely cosmetic)

No data migrations or breaking changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accidentally adding cleanup to a manual-only function | Low | High — deletes interactive work between GUI steps | Cross-reference each file against the excluded-functions list before modifying |
| Accidentally adding cleanup to an incremental function | Low | High — loses accumulated state | Audit each function: verify it does NOT load its output before writing. Only known case is `track_objects_in_video`. |
| Windows `PermissionError` on locked files | Low | Low — stale output persists | `try/except PermissionError` with warning. Task proceeds normally. |
| `None` in output paths causes crash | Med | Med — task crashes | Filter `None` values before iterating in `clean_task_outputs`. |
