# Plan: Hand Tracking ROI Improvements

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** Implemented (pending commit)
**Branch:** `feature/hand-tracking-roi-improvements`

---

## Overview

Add a `roi_mode` option (`"auto"` / `"manual"`) to the hand tracking ROI definition step so the auto pipeline can write a full-frame ROI without opening any GUI. Also: refactor the manual ROI GUI to use the shared `VideoFrameSelector` widget, add coordinate offset correction when tracking on a cropped ROI, wire `keep_stale` into `track_hands_on_video`, and support auto/manual string toggles in the DAG launcher GUI.

## Problem Statement

1. The `define_hand_tracking_roi` step always opens an interactive GUI — unusable in unattended/auto pipelines.
2. After ROI-cropped tracking, pixel coordinates are in cropped-frame space — downstream consumers get wrong positions.
3. The ROI GUI uses a custom frame-selection loop (`VideoMP4Manager`) instead of the shared `VideoFrameSelector` widget, creating maintenance divergence.
4. `track_hands_on_video` lacks `keep_stale` support, forcing unnecessary reprocessing when only upstream inputs changed.
5. The DAG launcher GUI has no widget for `"auto"`/`"manual"` string options — they render as raw text fields.

## Goals

### In Scope
1. Add `roi_mode` parameter to `define_hand_tracking_roi` — `"auto"` writes full-frame ROI, `"manual"` opens the GUI
2. Refactor `roi_definition_gui.py` to use tkinter `VideoFrameSelector` instead of `VideoMP4Manager`
3. Add `_offset_roi_coordinates()` to shift cropped-frame pixel coords back to full-frame space
4. Wire `keep_stale` into `track_hands_on_video` entry point and `should_process_task` call
5. Add auto/manual checkbox toggle widget to `task_detail_panel.py`

### Out of Scope
- Adding `define_hand_tracking_roi` as a task in the auto pipeline DAG
- Smart/content-aware auto ROI detection (e.g., hand detection to crop)
- Adding `keep_stale` to `define_hand_tracking_roi` itself
- Changes to YAML config values (session lists, enabled flags) — those are working-state tweaks

## Success Criteria

- [x] `define_hand_tracking_roi(..., roi_mode="auto")` writes a full-frame ROI JSON without any GUI
- [x] `define_hand_tracking_roi(..., roi_mode="manual")` opens `VideoFrameSelector` then `FrameROISquare` flow
- [x] After ROI-cropped tracking, hand landmark coordinates are in full-frame pixel space
- [x] `track_hands_on_video(..., keep_stale=True)` touches stale outputs instead of reprocessing
- [x] ROI file changes trigger reprocessing (ROI path included in `input_paths`)
- [x] DAG launcher shows an "Auto" checkbox for any task option with value `"auto"` or `"manual"`

---

## Technical Design

### Approach

Minimal, targeted changes to existing files. The `roi_mode` parameter flows from DAG YAML → workflow script → `define_hand_tracking_roi()`. The GUI refactor replaces the custom `VideoMP4Manager` frame selector with the shared `VideoFrameSelector` Toplevel (already used elsewhere). Coordinate offset is a post-processing step applied after the batch API returns results.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `roi_mode: auto` writes full-frame JSON | Simple, no new dependencies, downstream code unchanged | No intelligent cropping | **Chosen** — sufficient for unattended runs |
| Skip ROI entirely in auto mode | Even simpler | `track_hands_on_video` already has ROI-aware code paths; skipping diverges | Rejected |
| Content-aware auto-crop (hand detection) | Better cropping | Complex, adds ML dependency, out of scope | Rejected — future enhancement |

### Architecture Changes

No new modules or classes. Changes are within existing files:

- `define_hand_tracking_roi.py` — new `roi_mode` parameter with auto/manual branching
- `roi_definition_gui.py` — swap `VideoMP4Manager` for `VideoFrameSelector` (tkinter Toplevel)
- `track_hands_on_video.py` — new `_offset_roi_coordinates()` function, `keep_stale` parameter
- `task_detail_panel.py` — new dispatch branch for `{"auto", "manual"}` string values

---

## Implementation Plan

### Phase 1: ROI auto mode and GUI refactor
**Goal:** Enable non-interactive ROI creation and modernize the manual GUI

**Tasks:**
- [x] Task 1.1 — Add `roi_mode: str = "manual"` parameter to `define_hand_tracking_roi()`; when `"auto"`, read video dimensions via `cv2.VideoCapture` and write full-frame `{x_min, x_max, y_min, y_max}` JSON
- [x] Task 1.2 — Refactor `select_roi_on_video()` in `roi_definition_gui.py`: replace `VideoMP4Manager` with `cv2.VideoCapture` + tkinter `VideoFrameSelector` Toplevel for frame selection; keep `FrameROISquare` for rectangle drawing
- [x] Task 1.3 — Pass `roi_mode` from DAG options through the manual workflow's `define_hand_tracking_roi_flow`

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/define_hand_tracking_roi.py` — auto/manual branching, `cv2` import
- `code/src/preprocessing/motion_analysis/hand_tracking/gui/roi_definition_gui.py` — `VideoFrameSelector` integration, remove `VideoMP4Manager` dependency
- `code/scripts/preprocess_workflow_kinect_manual.py` — pass `roi_mode` from options to flow

**Dependencies:** None

### Phase 2: Coordinate offset correction and keep_stale
**Goal:** Make ROI-cropped tracking produce correct full-frame coordinates; add keep_stale support

**Tasks:**
- [x] Task 2.1 — Add `_offset_roi_coordinates(results_map, roi)` to `track_hands_on_video.py` that shifts all pixel coordinates by `(x_min, y_min)` from the ROI dict
- [x] Task 2.2 — Call `_offset_roi_coordinates()` after batch processing when both `roi_video_path` and `roi` are not None
- [x] Task 2.3 — Add `keep_stale: bool = False` parameter to `track_hands_on_video()` entry point; pass to `should_process_task()`
- [x] Task 2.4 — Include `roi_path` in `input_paths` for `should_process_task()` so ROI changes trigger reprocessing

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/track_hands_on_video.py` — offset function, keep_stale wiring, roi_path in input_paths

**Dependencies:** None (independent of Phase 1)

### Phase 3: DAG launcher GUI toggle
**Goal:** Render auto/manual string options as a checkbox in the GUI

**Tasks:**
- [x] Task 3.1 — In `_make_scalar_section()`, add dispatch branch: if `isinstance(val, str) and val in {"auto", "manual"}` → render `QCheckBox("Auto")` with checked state = `val == "auto"`
- [x] Task 3.2 — Add `_make_mode_toggle_handler()` that writes `"auto"` or `"manual"` string back to the model via `set_task_option()`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — new dispatch branch and handler

**Dependencies:** None (independent of Phase 1 & 2)

---

## Testing Plan

### Manual Verification
- [ ] Run manual preprocessing with `roi_mode: manual` — confirm `VideoFrameSelector` opens, frame selection works, `FrameROISquare` draws ROI, JSON saved
- [ ] Run manual preprocessing with `roi_mode: auto` — confirm no GUI opens, full-frame JSON written
- [ ] Run hand tracking on a video with an ROI sidecar — confirm output coordinates are in full-frame space (not cropped space)
- [ ] Set `keep_stale: true` in DAG config, touch an upstream file — confirm `track_hands_on_video` skips reprocessing and touches the output
- [ ] Open DAG launcher GUI, check that `roi_mode` option shows an "Auto" checkbox that toggles between `"auto"` and `"manual"`

### Edge Cases
- [ ] Video with no existing ROI sidecar in manual mode — GUI opens fresh (no pre-populated rectangle)
- [ ] `roi_mode: auto` on a video where ROI JSON already exists and `force_processing: false` — should skip (already up-to-date)
- [ ] Full-frame ROI (auto mode) — `_offset_roi_coordinates` adds (0, 0) offset, effectively a no-op

---

## Documentation Plan

- [ ] No external documentation needed — internal pipeline behavior change

---

## Rollback Plan

1. Revert uncommitted changes in the 4 source files
2. No data migrations or config schema changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `VideoFrameSelector` tkinter root conflicts with PyQt5 in same process | Low | Medium | ROI definition runs in a separate Prefect flow; tkinter root is created/destroyed within the function scope |
| `_offset_roi_coordinates` applied when it shouldn't be (no cropping) | Low | Medium | Guard: only called when both `roi_video_path is not None` and `roi is not None` |
| Multiple auto/manual options in a single task confused by generic checkbox label | Low | Low | Currently only `roi_mode` uses this pattern; label shows "Auto" which is self-explanatory in context |

---

## References

- Existing `VideoFrameSelector`: `code/src/preprocessing/common/gui/video_frame_selector.py`
- Existing `FrameROISquare`: `code/src/preprocessing/common/gui/frame_roi_square.py`
- `should_process_task` with `keep_stale`: `code/src/utils/should_process_task.py`
- Manual workflow ROI call site: `code/scripts/preprocess_workflow_kinect_manual.py:293-304`
