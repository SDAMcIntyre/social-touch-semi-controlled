# Plan: Hand Tracking ROI Coordinate Normalization

**Date:** 2026-03-22
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/hand-tracking-roi-fix`
**Completed:** 2026-03-23 10:05

---

## Overview

When a Region of Interest (ROI) is configured, the hand tracking pipeline crops the video before uploading it to the HaMeR API. The API returns pixel coordinates relative to the cropped frame, but downstream consumers expect coordinates in the original full-frame space. This plan adds a post-processing offset to restore correct global coordinates.

## Problem Statement

`track_hands_on_video.py` correctly crops the video using the ROI bounds (`x_min`, `y_min`, `x_max`, `y_max`) before sending it to the HaMeR API. However, the returned pixel data — `person_bounding_box_xyxy` and `vertices_pixel` — is in the cropped frame's coordinate system. Downstream code (e.g., `HandTrackingDataManager.get_pixelwise_hand_geometry()`) renders these coordinates on the original full-size frame, resulting in misaligned hand mesh overlays.

## Goals

### In Scope
1. Offset `person_bounding_box_xyxy` and `vertices_pixel` by `(x_min, y_min)` when ROI cropping was applied
2. Gracefully skip error frames (no `hands` key, e.g., `{'error': 'No person detected', ...}`)

### Out of Scope
- Modifying the frame-by-frame processing mode (ROI cropping only applies to batch video mode)
- Changing the `_process_batch_mode` mapping logic
- Downstream consumer changes (the fix is applied before serialization)

## Success Criteria

- [ ] `vertices_pixel` values in the output pickle are in global frame coordinates when ROI is used
- [ ] `person_bounding_box_xyxy` values are correctly offset
- [ ] Error frames (no person detected) are not affected
- [ ] Processing without ROI works unchanged (no regression)

---

## Technical Design

### Approach

Add a module-level helper `_offset_roi_coordinates(results_map, roi)` that mutates the API response dicts in-place, adding the ROI origin offset to all pixel coordinate fields. Call it in `execute()` after processing but before assembly/serialization, only when cropping actually occurred (`roi_video_path is not None`).

### Architecture Changes

No new modules or classes. Single function addition + one call site in `execute()`.

**API response structure in `results_map` values:**

Success frame:
```python
{"frame_index": 90, "hands": [{"person_bounding_box_xyxy": [x1,y1,x2,y2], "vertices_pixel": [[x,y], ...], ...}]}
```

Error frame:
```python
{"error": "No person detected", "frame_index": 0}
```

---

## Implementation Plan

### Phase 1: Add coordinate offset helper and wire it in
**Goal:** Correct all pixel coordinates in the API response before serialization

- [x] Add `_offset_roi_coordinates(results_map, roi)` helper near existing ROI helpers (~line 133)
- [x] Call the helper in `execute()` after the processing block, conditioned on `roi_video_path is not None`

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/track_hands_on_video.py` — add helper function + call in `execute()`

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline on a video with ROI configured; verify `vertices_pixel` coords fall within full frame bounds
- [ ] Run pipeline on a video without ROI; verify output is unchanged
- [ ] Inspect a pickle output where some frames have `error` key — confirm no crash

---

## Rollback Plan

1. Revert the single commit on `feature/hand-tracking-roi-fix`
2. No data migration needed — re-run the pipeline to regenerate pickle files

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| API response structure differs from expected | Low | Med | Helper uses `.get()` with safe defaults; skips missing keys |
