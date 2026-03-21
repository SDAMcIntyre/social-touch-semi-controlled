# Plan: Hand Tracking ROI Cropping

**Date:** 2026-03-21
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-21 19:02
**Branch:** `feature/hand-tracking-roi`

---

## Overview

**What:** Add an optional per-video ROI cropping mechanism to the hand tracking pipeline, with a new GUI task in the manual pipeline for defining the crop region and auto pipeline integration that uses it.

**Why:** The HaMeR API sometimes fails to correctly identify the target hand when processing the full video frame — particularly in sessions with cluttered scenes or unusual subject positioning. Cropping to a region of interest narrows the search space and improves detection reliability.

**How:** A sidecar JSON file defines static crop bounds per video. A new manual pipeline GUI task creates this file. The auto pipeline reads it (if present), creates a temporary cropped video, and uploads that to the API instead of the full frame.

## Problem Statement

- The HaMeR API processes the entire video frame, which can cause incorrect person/hand detection in some sessions
- There is no mechanism to constrain the API's input to a specific region of the frame
- The user currently has no way to specify per-video ROI preferences within the pipeline workflow

## Goals

### In Scope
1. Sidecar ROI file format (`*_handmodel_roi.json`) stored in `kinematics_analysis/`
2. New standalone GUI task in the manual pipeline (`define_hand_tracking_roi`) to define a static ROI rectangle on a video frame
3. Auto pipeline reads the ROI file (if it exists) and creates a temporary cropped `.mp4` before API upload
4. Graceful fallback: if no ROI file exists, the pipeline processes the full video as before

### Out of Scope
- Per-frame dynamic ROI (only static crop — same bounds for all frames)
- ROI for frame-by-frame mode (`use_video_api=False`) — only batch video mode gets the temp video crop
- Automatic ROI detection or suggestion algorithms
- Coordinate realignment of API output (not needed — HaMeR returns 3D mesh geometry, not pixel-space coordinates)

## Success Criteria

- [ ] A `*_handmodel_roi.json` sidecar file can be created via the manual pipeline GUI
- [ ] The auto pipeline detects and uses the ROI file when present
- [ ] The auto pipeline works unchanged when no ROI file exists (no regression)
- [ ] A temporary cropped video is created, uploaded to the API, and cleaned up
- [ ] The downstream `generate_3d_hand_in_motion` step works identically with ROI-cropped tracking results

---

## Technical Design

### Approach

**Sidecar file format** — `{video_stem}_handmodel_roi.json` in `kinematics_analysis/`:
```json
{
  "x_min": 512,
  "x_max": 1024,
  "y_min": null,
  "y_max": null
}
```
`null` values mean "use full extent" for that axis. This keeps the format minimal and self-documenting.

**No coordinate realignment needed** — HaMeR returns 3D hand mesh reconstructions (`vertices_planar_z0`, `vertices_3d`) in a canonical space relative to the detected hand, not in pixel coordinates. Cropping the input only affects which region the API searches for a hand — it does not shift the output coordinate system.

**GUI task** — New `define_hand_tracking_roi` task in the manual pipeline. A tkinter window displays a video frame and lets the user draw a rectangle to define the crop region. Follows the same pattern as `define_led_roi` and `define_hand_mask`.

**Auto pipeline integration** — In `track_hands_on_video.py`, before calling the HaMeR API:
1. Check for ROI sidecar file (derived from video stem + output directory)
2. If found, use OpenCV `VideoCapture`/`VideoWriter` to write a temporary cropped `.mp4`
3. Upload the cropped video instead of the original
4. Clean up the temp file after processing (using existing `tempfile.TemporaryDirectory()` pattern already at line 211)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Sidecar JSON file | Per-video, no model changes, discoverable, pipeline works without it | Extra file to manage | **Chosen** |
| DAG config option | Simple to add | Global — applies to all videos equally | Rejected |
| KinectConfig model field | Truly per-session | Requires model changes, migration of existing configs | Rejected |
| Force frame-by-frame mode when ROI set | No temp video I/O | Slower API calls, different processing path | Rejected — temp video simpler |

### Architecture Changes

New files:
```
code/src/preprocessing/motion_analysis/hand_tracking/gui/
    roi_definition_gui.py          -- Tkinter GUI for drawing ROI rectangle on video frame

code/scripts/_3_preprocessing/_2_hand_tracking/
    define_hand_tracking_roi.py    -- Entry point: loads video, launches GUI, saves ROI JSON
```

Modified files:
```
code/scripts/_3_preprocessing/_2_hand_tracking/
    track_hands_on_video.py        -- ROI loading, temp video cropping, integration
    __init__.py                    -- Export new function

code/scripts/
    preprocess_workflow_kinect_manual.py   -- Add define_hand_tracking_roi flow + wiring
    preprocess_workflow_kinect_auto.py     -- Pass ROI path to track_hands_on_video

configs/
    preprocess_workflow_kinect_manual_dag.yaml  -- Add task entry
```

---

## Implementation Plan

### Phase 1: ROI GUI and Sidecar File
**Goal:** Create the GUI for defining ROI and the sidecar file format

- [x] Task 1.1 — Create `roi_definition_gui.py`: tkinter GUI that displays a video frame (navigable), lets user draw a rectangle, returns `(x_min, x_max, y_min, y_max)`. Uses `VideoMP4Manager` for frame access.
- [x] Task 1.2 — Create `define_hand_tracking_roi.py`: entry point that loads video, launches GUI, saves ROI to `{video_stem}_handmodel_roi.json`. Pattern follows `define_hand_mask.py` / `define_led_roi.py`. Uses `should_process_task()` for idempotency.
- [x] Task 1.3 — Export `define_hand_tracking_roi` from `__init__.py`

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/gui/roi_definition_gui.py` — NEW
- `code/scripts/_3_preprocessing/_2_hand_tracking/define_hand_tracking_roi.py` — NEW
- `code/scripts/_3_preprocessing/_2_hand_tracking/__init__.py` — Add export

**Dependencies:** None

### Phase 2: Auto Pipeline ROI Integration
**Goal:** Make `track_hands_on_video.py` ROI-aware with temp cropped video creation

- [x] Task 2.1 — Add `_load_roi_config(roi_path: Path) -> Optional[dict]` helper that reads the sidecar JSON; returns `None` if file doesn't exist
- [x] Task 2.2 — Add `_create_cropped_video(source_path: Path, roi: dict, temp_dir: Path) -> Path` helper that uses OpenCV `VideoCapture`/`VideoWriter` to write a cropped `.mp4`. Clamps ROI bounds to actual frame dimensions. Resolves `null` values to full extent.
- [x] Task 2.3 — Modify `HandTrackingPipeline._process_batch_mode()` to accept an optional `roi_video_path` parameter — if provided, upload the cropped video instead of the original
- [x] Task 2.4 — Modify `HandTrackingPipeline.execute()` to: look for ROI file, create temp cropped video if found, pass it to `_process_batch_mode()`, clean up after
- [x] Task 2.5 — Modify `track_hands_on_video()` entry point to accept an optional `roi_path` parameter (defaults to looking in the output directory based on video stem)

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/track_hands_on_video.py` — Add ROI loading, cropping, and integration

**Dependencies:** None (independent of Phase 1)

### Phase 3: Pipeline Wiring
**Goal:** Wire the GUI task into the manual pipeline and pass ROI path through the auto pipeline

- [x] Task 3.1 — Add `define_hand_tracking_roi` task to `configs/preprocess_workflow_kinect_manual_dag.yaml` (no dependencies, should appear before `curate_hamer_hand_models`)
- [x] Task 3.2 — Add `define_hand_tracking_roi` flow function and pipeline stage in `preprocess_workflow_kinect_manual.py`
- [x] Task 3.3 — In `preprocess_workflow_kinect_auto.py`, modify `track_hands_model_flow()` to construct the expected ROI sidecar path and pass it to `track_hands_on_video()` (the function gracefully handles missing files)

**Files Modified:**
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — Add task entry
- `code/scripts/preprocess_workflow_kinect_manual.py` — Add flow function and pipeline wiring
- `code/scripts/preprocess_workflow_kinect_auto.py` — Pass ROI path to `track_hands_on_video()`

**Dependencies:** Phase 1, Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run the manual pipeline `define_hand_tracking_roi` step — verify GUI opens, rectangle can be drawn, JSON file is saved correctly with expected format
- [ ] Run the auto pipeline `track_hands_model` on a video WITH an ROI file — verify temp cropped video is created, uploaded, and cleaned up; results pickle is produced
- [ ] Run the auto pipeline `track_hands_model` on a video WITHOUT an ROI file — verify it works exactly as before (no regression)
- [ ] Verify downstream `generate_3d_hand_in_motion` produces valid output from ROI-cropped tracking results

### Edge Cases
- [ ] ROI file with `null` y values — should crop only x axis, keep full y range
- [ ] ROI file with all `null` values — should behave as if no ROI file exists (skip cropping)
- [ ] ROI bounds that exceed video dimensions — should clamp to actual frame size

---

## Documentation Plan

- [ ] Add inline comments in `track_hands_on_video.py` explaining the ROI cropping flow
- [ ] Document the sidecar JSON format in `define_hand_tracking_roi.py` docstring

---

## Rollback Plan

1. Remove the ROI sidecar check from `track_hands_on_video.py` — pipeline reverts to full-video processing
2. Remove the manual pipeline task entries — no side effects on other tasks
3. Delete any `*_handmodel_roi.json` files — they are standalone sidecar files not referenced by other outputs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OpenCV codec issues when writing temp cropped video | Med | Med | Use same codec/fps as source video; fall back to MJPEG if needed |
| Temp video disk space on large files | Low | Low | Write to `tempfile.TemporaryDirectory()` which auto-cleans; crop reduces file size vs original |
| ROI too small — API fails to detect hand | Low | Low | User can delete ROI file and re-run with full frame |

---

## References

- Existing sidecar pattern: `*_roi_metadata.json` for sticker tracking, `.SUCCESS` flags
- Similar GUI tasks: `define_led_roi.py`, `define_hand_mask.py`
- HaMeR API client: `code/src/preprocessing/motion_analysis/hand_tracking/hamer_liu_client/hamer_client_api.py`
- API response structure: `hands[].vertices_planar_z0` (3D mesh, not pixel coords)
