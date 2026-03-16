# Plan: Fix Averaged Forearm Pointcloud Color Loss

**Date:** 2026-03-16
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `fix/averaged-forearm-color-loss`

---

## Overview

The multi-frame averaging path in forearm extraction produces white-colored pointclouds because `FrameDepthAverager` unreliably re-seeks backward in the MKV to fetch the color image. This fix passes the already-loaded color image directly to the averager, eliminating the re-seek. A secondary defensive fix normalizes non-MJPG color formats to 3-channel BGR in `KinectFrame.color`.

## Problem Statement

When extracting forearms using multi-frame depth averaging (introduced in `343ef35`, Feb 20), the `FrameDepthAverager` iterates through all requested frames for depth accumulation, advancing the MKV reader forward. It then tries to **re-seek backward** to `color_frame_id` to fetch color data. This backward seek via pyk4a's timestamp-based `playback.seek()` + `get_next_capture()` is unreliable — it can return a capture with valid depth but `capture.color == None`, triggering the silent white fallback (`np.ones((N, 3))`).

The result: all points in the extracted pointcloud are white (RGB 1.0, 1.0, 1.0). This breaks the downstream HSV Skin Color Filter (white has S=0, which is below the `s_lo=0.1` threshold), making forearm segmentation impossible.

The representative frame is already loaded earlier in `extract_participant_forearm.py` (line 241) with valid color data, but this color is never passed to the averager.

## Goals

### In Scope
1. Eliminate the unreliable color frame re-seek in `FrameDepthAverager`
2. Pass pre-loaded color image from the caller to the averager
3. Defensively handle non-MJPG color formats (4-channel BGRA) in `KinectFrame.color`
4. Add a shape guard in `generate_o3d_point_cloud` to prevent silent corruption

### Out of Scope
- Changing the depth averaging algorithm itself
- Modifying the single-frame extraction path (already works correctly)
- Fixing pyk4a's underlying seek reliability (upstream library)
- Changes to the interactive segmentation GUI

## Success Criteria

- [ ] Re-running `extract_forearm` with averaged extraction on a previously-white session produces a pointcloud with real skin-tone colors
- [ ] The HSV skin color filter retains points (non-empty output after filtering)
- [ ] Single-frame extraction continues to work correctly (no regression)
- [ ] Non-MJPG color formats (BGRA32) produce 3-channel BGR output in `KinectFrame.color`

---

## Technical Design

### Approach

Pass the representative frame's color image (already loaded at line 241 of `extract_participant_forearm.py`) directly to `FrameDepthAverager.average()` instead of a `color_frame_id` integer. This eliminates the problematic backward re-seek entirely. The color image is loaded once from a reliably positioned reader, before the averaging loop moves it.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pass pre-loaded color image to averager | Simple, eliminates re-seek entirely, zero risk of seek failure | Slightly changes the API signature | **Chosen** |
| Cache color during the averaging loop (read color from first frame in the depth loop) | No API change | Color comes from first frame, not representative; adds complexity inside the loop | Rejected |
| Retry seek with fallback to nearby frames | Doesn't change API | Complex retry logic, still relies on unreliable seeking | Rejected |
| Switch pyk4a seek strategy to 'sequential' | Might be more reliable | Much slower for backward seeks, and doesn't guarantee color availability | Rejected |

### Architecture Changes

No new modules. Minimal API change to `FrameDepthAverager.average()`:
- Parameter `color_frame_id: int` replaced by `color_image: np.ndarray | None`
- Re-seek logic (lines 80-91) removed
- Caller passes `frame.color` instead of `video_config.representative_frame_id`

---

## Implementation Plan

### Phase 1: Fix averaged extraction color path
**Goal:** Eliminate the re-seek and pass pre-loaded color image

- [ ] Task 1.1 — Change `FrameDepthAverager.average()` signature: replace `color_frame_id: int` with `color_image: np.ndarray | None`
- [ ] Task 1.2 — Remove re-seek logic (lines 80-91 in `frame_depth_averager.py`) and use the passed `color_image` directly
- [ ] Task 1.3 — Update caller in `extract_participant_forearm.py` to pass `frame.color`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/depth_averaging/frame_depth_averager.py` — Replace `color_frame_id` param with `color_image`, remove re-seek block
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` — Pass `frame.color` to `FrameDepthAverager.average()`

**Dependencies:** None

### Phase 2: Defensive fixes in KinectFrame
**Goal:** Prevent future color format issues

- [ ] Task 2.1 — Fix `KinectFrame.color` property to always return BGR (H,W,3) for non-MJPG formats (drop alpha channel from BGRA)
- [ ] Task 2.2 — Add shape guard in `generate_o3d_point_cloud()` to validate (N,3) before `Vector3dVector`

**Files Modified:**
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` — Fix `color` property (lines 31-41), add guard after line 71

**Dependencies:** None (independent of Phase 1)

---

## Testing Plan

### Manual Verification
- [ ] Run `extract_forearm` with averaged extraction on a session that previously produced white pointclouds — confirm colored output in the segmentation GUI
- [ ] Run `extract_forearm` with averaged extraction on a different session — confirm colored output (no session-specific regression)
- [ ] Run `extract_forearm` with single-frame extraction — confirm it still works correctly
- [ ] Confirm the HSV skin color filter retains a non-empty point set after filtering

### Edge Cases
- [ ] Session where `representative_frame_id` is the first frame in the averaging range
- [ ] Session where `representative_frame_id` is the last frame in the averaging range
- [ ] Session with a single frame (not averaged) — verify `is_averaged == False` path is unaffected

---

## Documentation Plan

- [ ] Remove the idea file `docs/development/plans/ideas/forearm-pointcloud-white-color-investigation.md` (superseded by this plan)

---

## Rollback Plan

1. Revert the two changed files to their previous state (`git revert` the fix commit)
2. No data migrations or breaking changes — the fix only affects runtime behavior
3. Old saved segmentation parameter files are unaffected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `frame.color` at line 241 is also `None` (lazy loading hasn't triggered, or capture is invalid) | Low | High (still white) | `frame.color` accesses the property which triggers lazy loading; if still `None`, the white fallback in `FrameDepthAverager` handles it gracefully with a warning |
| BGRA fix (Phase 2) breaks a recording with an unusual color format | Low | Med | The fix only changes the non-MJPG path; MJPG recordings (all current data) are unaffected. Logging added for unexpected formats |

---

## References

- Introducing commit: `343ef35` — `feat(forearm-extraction): implement frame averaging pipeline`
- Related fix: `965ddff` — `fix(postprocessing): preserve RGB colors in PCA-calibrated forearm export`
- Idea note: `docs/development/plans/ideas/forearm-pointcloud-white-color-investigation.md`
