# Plan: Fix Black OpenCV ROI Window (tkinter Conflict + Color Mismatch)

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/dag-workflow-category-grouping` (or new branch)

---

## Overview

The OpenCV ROI drawing window (`FrameROIRotatable`) displays a black window instead of the video frame when launched through the forearm extraction pipeline. Two bugs are identified: a duplicate `tk.Tk()` instance that can corrupt the X11 display connection on WSL2, and a BGR/RGB color format mismatch that causes incorrect channel rendering.

## Problem Statement

When the user runs the forearm extraction pipeline (via DAG launcher or directly), the annotation workflow has two interactive steps:
1. **Frame-group selector** (tkinter `MultiVideoFramesSelector`) -- works correctly.
2. **ROI drawing window** (OpenCV `FrameROIRotatable`) -- displays a black window.

The ROI window was previously functional. The black rendering is caused by `FrameROISquare._setup_window()` creating a second `tk.Tk()` Tcl interpreter while the singleton from `define_extraction_parameters.py` is still alive. On WSL2 (WSLg), destroying the second interpreter can corrupt the shared X11 display connection, causing `cv2.imshow()` to render black.

A secondary bug exists: `VideoMP4Manager` returns BGR frames by default, but they are passed to `FrameROIRotatable` with `is_rgb=True`, causing a double channel-swap (BGR treated as RGB, then converted to BGR again = R/B channels swapped).

## Goals

### In Scope
1. Eliminate the duplicate `tk.Tk()` creation in `FrameROISquare._setup_window()`
2. Fix the BGR/RGB color format mismatch in the ROI display path

### Out of Scope
- Refactoring `FrameROISquare` / `FrameROIRotatable` architecture
- Changing `VideoMP4Manager` default color format
- Modifying the tkinter `MultiVideoFramesSelector` (it works correctly)

## Success Criteria

- [ ] The ROI drawing window displays the video frame (not black) when launched from the pipeline
- [ ] The frame colors are correct (no R/B channel swap)
- [ ] The ROI window still centers on screen when possible
- [ ] No regression in `FrameROISquare` when used standalone (without a pre-existing tkinter root)

---

## Technical Design

### Approach

1. **Fix tkinter conflict:** Replace the temporary `tk.Tk()` in `_setup_window()` with a safe screen-dimension query that checks for an existing tkinter root first, falling back to creating one only when none exists.
2. **Fix color mismatch:** Pass `is_rgb=False` to `FrameROIRotatable` since `VideoMP4Manager` returns BGR frames by default.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse existing `tk.Tk()` or create only if none exists | Safe, minimal change, works standalone too | Slightly more code in `_setup_window` | **Chosen** |
| Remove window centering entirely | Simplest fix | Loses screen centering; user experience regression | Rejected |
| Use `ctypes.windll` for screen dimensions | Avoids tkinter entirely | Windows-only; fails on native Linux | Rejected |
| Change `VideoMP4Manager` default to RGB | Fixes mismatch at source | Breaks all existing BGR consumers | Rejected |
| Pass `is_rgb=False` to `FrameROIRotatable` | Minimal, correct fix at the call site | None | **Chosen** |

### Architecture Changes

No new modules or classes. Two surgical fixes in existing files.

---

## Implementation Plan

### Phase 1: Fix duplicate `tk.Tk()` in `_setup_window()`
**Goal:** Prevent X11 display corruption by avoiding duplicate Tcl interpreters.

**Tasks:**
- [ ] Task 1.1 -- Modify `_setup_window()` to check for an existing tkinter root (`tk._default_root`) before creating a new one; reuse it if available, otherwise create and destroy a temporary one as before.

**Files Modified:**
- `code/src/preprocessing/common/gui/frame_roi_square.py` -- `_setup_window()` method (lines 106-134): replace the unconditional `tk.Tk()` creation with a safe check.

**Dependencies:** None

### Phase 2: Fix BGR/RGB color mismatch
**Goal:** Ensure the ROI window displays correct colors.

**Tasks:**
- [ ] Task 2.1 -- Change `is_rgb=True` to `is_rgb=False` in `_select_roi_for_group()` since `VideoMP4Manager` returns BGR frames.

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py` -- `_select_roi_for_group()` (line 219): change `is_rgb=True` to `is_rgb=False`.

**Dependencies:** None (independent of Phase 1)

---

## Testing Plan

### Manual Verification
- [ ] Run `define_extraction_parameters.py` standalone (with sample videos) -- ROI window should display the frame with correct colors
- [ ] Run `preprocess_pipeline_extract_forearm_manual.py` -- both the frame-group selector (step 1) and ROI window (step 2) should display correctly
- [ ] Run via DAG launcher (`launch_dag_config_gui.py`) -- same verification as above
- [ ] Run `FrameROIRotatable` standalone (its `__main__` block at bottom of file) -- should still work correctly without a pre-existing tkinter root

### Edge Cases
- [ ] No pre-existing tkinter root (standalone `FrameROIRotatable` usage) -- should create temporary root for screen dimensions, then destroy it
- [ ] Pre-existing tkinter root (pipeline usage via `define_extraction_parameters.py`) -- should reuse existing root, no second interpreter created

---

## Documentation Plan

- [ ] Consider adding a knowledge-base note about tkinter `tk.Tk()` singleton conflicts with OpenCV on WSL2 if the fix proves effective

---

## Rollback Plan

1. Revert the two modified files to their previous state (`git checkout HEAD -- <file>`)
2. No data migrations or breaking changes involved

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `tk._default_root` API is private/unstable | Low | Low | It's a well-known tkinter internal used widely; fallback to creating a new root if unavailable |
| Black window has a different root cause (e.g., WSLg update) | Med | Med | Phase 1 fix is still correct regardless; if black persists, investigate WSLg/OpenCV backend separately |
| Changing `is_rgb` breaks ROI coordinate mapping | Low | Low | `is_rgb` only affects color conversion, not geometry; ROI coordinates are in original image space |
