# Plan: Hand Tracking ROI — Skip-Logic Fix, GUI Alignment & Auto Mode

**Date:** 2026-03-21
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/hand-tracking-roi-fix`
**Completed:** 2026-03-23 10:05

---

## Overview

**What:** Fix three issues in the recently shipped hand-tracking-roi feature: (1) ROI file changes not triggering reprocessing, (2) the ROI GUI using a custom OpenCV frame selector instead of the shared `VideoFrameSelector` widget, (3) add an "auto" mode that writes a full-frame ROI without requiring manual drawing, and (4) wire `keep_stale` to `track_hands_model` so existing tracking results can be accepted even when the ROI file is newer.

**Why:** Issue 1 is a bug — defining or updating an ROI has no effect when tracking output already exists. Issue 2 is a consistency gap — the LED ROI uses the shared Tkinter `VideoFrameSelector`, the hand tracking ROI does not. Issue 3 is a workflow gap — most sessions need no cropping, but there is no way to mark them as "ROI assessed" without manually drawing a full-frame rectangle. Issue 4 complements the skip-logic fix — once ROI is an input dependency, `keep_stale` lets you accept existing tracking output without a costly HaMeR API re-run.

**How:** Add the ROI sidecar file to `should_process_task()` input list. Replace `_select_frame()` with `VideoFrameSelector`. Add a `roi_mode` DAG config option (`auto` | `manual`). Wire `keep_stale` to `track_hands_model` in the auto pipeline.

## Problem Statement

1. **Skip-logic bug:** `track_hands_on_video()` calls `should_process_task(input_paths=[rgb_video_path], ...)`. The ROI sidecar file (`*_handmodel_roi.json`) is not an input. When tracking output already exists (common — most sessions are already processed), creating or modifying the ROI file does not trigger reprocessing. The ROI is silently ignored.

2. **GUI inconsistency:** The hand tracking ROI GUI (`roi_definition_gui.py:_select_frame()`) uses raw OpenCV `imshow` with keyboard-only navigation (a/d/q/e keys). The LED ROI workflow uses `VideoFrameSelector` — a Tkinter widget with a slider, direct frame number entry, and arrow key support. `VideoFrameSelector` lives in `preprocessing/common/gui/` (shared, not LED-specific) and should be reused.

3. **No "auto" mode:** Most sessions need no ROI cropping (full frame is correct). Currently the only way to produce an ROI file is manual GUI drawing. An "auto" option that writes full-frame bounds to the sidecar JSON would let the user batch-mark sessions as assessed without interactive drawing.

## Goals

### In Scope
1. Make ROI file changes trigger hand tracking reprocessing
2. Replace custom frame selector with shared `VideoFrameSelector`
3. Add `roi_mode` DAG config option (`auto` | `manual`) to `define_hand_tracking_roi` task
4. Wire `keep_stale` option to `track_hands_model` in the auto pipeline

### Out of Scope
- Moving LED's `UserInterface` class to `preprocessing/common/` (not needed — minimal Tkinter root setup is simpler)
- Changes to the ROI drawing step (`FrameROISquare` — already shared, works correctly)
- Changes to `should_process_task()` itself
- Automatic ROI detection heuristics (the "auto" mode writes full-frame bounds, not a smart crop)

## Success Criteria

- [ ] After defining/modifying an ROI, the next auto pipeline run reprocesses `track_hands_model` (not skips)
- [ ] When no ROI file exists, behavior is unchanged (no regression)
- [ ] `define_hand_tracking_roi` with `roi_mode: manual` opens a Tkinter slider-based frame selector (not the old OpenCV window)
- [ ] ROI drawing step still works correctly after frame selection
- [ ] `define_hand_tracking_roi` with `roi_mode: auto` writes full-frame ROI JSON without opening any GUI
- [ ] `track_hands_model` with `keep_stale: true` accepts stale output (touches timestamps) instead of reprocessing

---

## Technical Design

### Approach

**Fix 1 — Input dependency:** Add `roi_path` to the `input_paths` list when the file exists. This is a 3-line change. When `roi_path` is `None` or the file doesn't exist, behavior is identical to current code. When it exists and is newer than the output, `should_process_task()` returns `True`.

**Fix 2 — Shared frame selector:** Replace `_select_frame()` (60 lines of OpenCV code) with `VideoFrameSelector`. The selector accepts a `cv2.VideoCapture` object via duck-typing, so we open the video with `cv2.VideoCapture` (instead of `VideoMP4Manager`), create a temporary off-screen `tk.Tk()` root, run the selector, destroy the root, then proceed to `FrameROISquare`. This follows the same pattern as LED's `UserInterface.__init__()` (root at `+10000,+10000`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add ROI to `input_paths` | Minimal change, uses existing mechanism | None | **Chosen** |
| Always force-reprocess when ROI exists | Simple | Wasteful — reprocesses even when ROI hasn't changed | Rejected |
| Import LED's `UserInterface` for frame selection | Full reuse | Cross-domain import (`led_analysis` → `motion_analysis`), bundles unneeded methods | Rejected |
| Create Tkinter root inline (4 lines) | Minimal, self-contained | Slight duplication of root lifecycle pattern | **Chosen** |
| Move `UserInterface` to `preprocessing/common/` | Maximum reuse | Over-scoped for this fix, `UserInterface` has LED-specific methods | Rejected |
| DAG config `roi_mode` option | Batch-friendly, no per-session interaction | Requires reading option in pipeline wiring | **Chosen** |
| GUI prompt at runtime | No config change | Requires interaction every time | Rejected |

### Architecture Changes

No new files. Five existing files modified, two configs updated:

```
code/scripts/_3_preprocessing/_2_hand_tracking/
    track_hands_on_video.py           -- Add ROI path to input_paths + keep_stale parameter
    define_hand_tracking_roi.py       -- Add roi_mode parameter, auto-mode logic

code/src/preprocessing/motion_analysis/hand_tracking/gui/
    roi_definition_gui.py             -- Replace _select_frame() with VideoFrameSelector

code/scripts/
    preprocess_workflow_kinect_manual.py  -- Pass roi_mode from DAG options
    preprocess_workflow_kinect_auto.py    -- Pass keep_stale to track_hands_model_flow

configs/
    preprocess_workflow_kinect_manual_dag.yaml  -- Add roi_mode option
    preprocess_workflow_kinect_auto_dag.yaml    -- Add keep_stale option to track_hands_model
```

---

## Implementation Plan

### Phase 1: Fix skip-logic bug
**Goal:** ROI file changes trigger reprocessing

- [x] Task 1.1 — In `track_hands_on_video()`, build `input_paths` list that conditionally includes `roi_path` when the file exists, before calling `should_process_task()`

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/track_hands_on_video.py` (lines 346-353) — Add ROI to input_paths

**Dependencies:** None

### Phase 2: Replace frame selector GUI
**Goal:** Use shared `VideoFrameSelector` instead of custom OpenCV navigator

- [x] Task 2.1 — Update imports: drop `VideoMP4Manager`, add `VideoFrameSelector` (from `preprocessing.common`) and `tkinter`
- [x] Task 2.2 — Delete `_select_frame()` function (lines 13-61)
- [x] Task 2.3 — Rewrite frame selection step in `select_roi_on_video()`: open video with `cv2.VideoCapture`, create off-screen `tk.Tk()` root, run `VideoFrameSelector.select_frame()`, destroy root in `finally`, read selected frame, release capture
- [x] Task 2.4 — Keep ROI drawing step (`FrameROISquare`) unchanged

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/gui/roi_definition_gui.py` — Rewrite frame selection, ~60 lines removed, ~30 lines added

**Dependencies:** None (independent of Phase 1)

### Phase 3: Add "auto" ROI mode
**Goal:** Allow batch creation of full-frame ROI files without GUI interaction

- [x] Task 3.1 — Add `roi_mode: manual` option to `define_hand_tracking_roi` in `configs/preprocess_workflow_kinect_manual_dag.yaml`
- [x] Task 3.2 — Add `roi_mode: str = "manual"` parameter to `define_hand_tracking_roi()` in `define_hand_tracking_roi.py`. When `"auto"`: read video dimensions via `cv2.VideoCapture`, write `{"x_min": 0, "x_max": width, "y_min": 0, "y_max": height}` to the sidecar JSON, skip GUI. When `"manual"`: existing behavior.
- [x] Task 3.3 — Add `roi_mode` parameter to `define_hand_tracking_roi_flow()` in `preprocess_workflow_kinect_manual.py`
- [x] Task 3.4 — Read `roi_mode` from DAG options in `run_single_session_pipeline()` and pass it through the flow

**Files Modified:**
- `configs/preprocess_workflow_kinect_manual_dag.yaml` — Add `roi_mode: manual` option
- `code/scripts/_3_preprocessing/_2_hand_tracking/define_hand_tracking_roi.py` — Add `roi_mode` parameter and auto-mode logic
- `code/scripts/preprocess_workflow_kinect_manual.py` — Pass `roi_mode` from DAG options to flow and entry point

**Dependencies:** None (independent of Phase 1 and 2)

### Phase 4: Wire `keep_stale` to `track_hands_model`
**Goal:** Allow accepting existing tracking output as-is when ROI changes make it "stale"

- [x] Task 4.1 — Add `keep_stale: false` option to `track_hands_model` in `configs/preprocess_workflow_kinect_auto_dag.yaml`
- [x] Task 4.2 — Add `keep_stale: bool = False` parameter to `track_hands_on_video()` in `track_hands_on_video.py` and pass it to `should_process_task()`
- [x] Task 4.3 — Add `keep_stale: bool = False` parameter to `track_hands_model_flow()` in `preprocess_workflow_kinect_auto.py` and pass it to `track_hands_on_video()`
- [x] Task 4.4 — Read `keep_stale` from DAG task options in the auto pipeline dispatcher and pass it to `track_hands_model_flow()`

**Files Modified:**
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — Add `keep_stale: false` option to `track_hands_model`
- `code/scripts/_3_preprocessing/_2_hand_tracking/track_hands_on_video.py` — Add `keep_stale` parameter, pass to `should_process_task()`
- `code/scripts/preprocess_workflow_kinect_auto.py` — Add `keep_stale` parameter to flow, read from DAG options

**Dependencies:** Phase 1 (the `keep_stale` parameter is meaningful only when ROI is an input dependency)

---

## Testing Plan

### Manual Verification
- [ ] Run auto pipeline on a session with existing `*_tracked_hands.pkl` but no ROI file — confirm it skips (unchanged behavior)
- [ ] Define an ROI via manual pipeline with `roi_mode: manual` — confirm Tkinter slider frame selector appears
- [ ] Draw and save ROI — confirm JSON file is written
- [ ] Re-run auto pipeline — confirm `track_hands_model` reprocesses (not skips)
- [ ] Modify the ROI JSON — re-run auto — confirm reprocessing again
- [ ] Run auto pipeline on a session with no ROI file — confirm no regression
- [ ] Set `roi_mode: auto` in DAG config — run manual pipeline — confirm full-frame ROI JSON written without GUI
- [ ] Set `keep_stale: true` on `track_hands_model` — run auto pipeline with stale output — confirm output timestamps refreshed, task skipped

### Edge Cases
- [ ] `roi_path` is `None` (no ROI path provided) — should behave exactly as before
- [ ] ROI file exists but is older than output — should skip (output is up to date)
- [ ] User cancels frame selection in new GUI — should return `None`, no file written
- [ ] Auto mode on video that already has a manually-drawn ROI + `force_processing: false` — should skip (output already up to date)
- [ ] `keep_stale: true` + `force_processing: true` — force wins, task reprocesses
- [ ] `keep_stale: true` with missing output — task processes (missing output always triggers)

---

## Documentation Plan

- No external documentation changes needed (internal pipeline fix)

---

## Rollback Plan

1. Revert the `input_paths` change in `track_hands_on_video.py` — restores original skip behavior
2. Revert `roi_definition_gui.py` — restores OpenCV frame selector
3. Revert `define_hand_tracking_roi.py` and manual DAG config — removes auto mode
4. Revert `keep_stale` wiring in `track_hands_on_video.py` and auto DAG config — removes keep_stale for hand tracking
5. All four changes are independent and can be rolled back separately

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tkinter root not destroyed on error | Low | Med | `try/finally` block guarantees `root.destroy()` |
| Tkinter + OpenCV `highgui` conflict | Low | Med | Root is destroyed before `FrameROISquare` opens; LED code uses same pattern successfully |
| `VideoFrameSelector` API mismatch with `cv2.VideoCapture` | Low | Low | Already duck-typed for `VideoCapture` (checks `hasattr(source, 'get')`) — confirmed in source |

---

## References

- Original feature plan: `docs/development/plans/completed/hand-tracking-roi-cropping.md`
- Shared GUI widget: `code/src/preprocessing/common/gui/video_frame_selector.py`
- LED ROI pattern: `code/src/preprocessing/led_analysis/gui/user_interface.py`
- Skip logic: `code/src/utils/should_process_task.py`
