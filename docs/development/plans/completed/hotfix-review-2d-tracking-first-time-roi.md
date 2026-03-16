# Plan: Hotfix — Enforce ROI Drawing on First-Time 2D Tracking Review

**Date:** 2026-03-16
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `hotfix/review-2d-tracking-first-time-roi`

---

## Overview

When the review 2D tracking GUI opens for the first time (no metadata file exists), users can click "Finish as Valid" or "Rerun Auto-Processing" to bypass the mandatory ROI drawing step. This hotfix hides those buttons when no metadata exists, forcing users through the correct workflow: mark frames → draw ROI → proceed.

## Problem Statement

In `review_tracked_objects_in_video()`, when `metadata_path` does not exist, a blank `ROIAnnotationManager` is created and the review GUI opens with all buttons enabled. The user can:

1. Click **"Finish as Valid"** → marks all objects as `COMPLETED` without any ROI ever drawn
2. Click **"Rerun Auto-Processing"** → triggers `PROCEED` status with an empty labeling dict, producing no meaningful output

Both paths bypass the essential first step of drawing ROI around tracked objects.

## Goals

### In Scope
1. Hide "Finish as Valid" and "Rerun Auto-Processing" buttons when no metadata file exists
2. Ensure the only available workflow path for first-time users is: mark frames for labeling → Proceed with Marked → draw ROI interactively

### Out of Scope
- Changing the button logic when metadata already exists (returning user flow)
- Adding new UI elements or validation dialogs
- Refactoring the review GUI architecture

## Success Criteria

- [ ] First-time review (no metadata file): "Finish as Valid" and "Rerun" buttons are not visible
- [ ] Returning review (metadata file exists): both buttons appear and function as before
- [ ] No changes to the `TrackerReviewGUI` or `TrackerReviewOrchestrator` classes

---

## Technical Design

### Approach

Pass a `has_metadata` flag from `review_tracked_objects_in_video()` through `review_tracking()` to `TrackerReviewGUI`, mapping it to the existing `show_valid_button` and `show_rerun_button` constructor parameters. No new UI logic is needed — the GUI already supports hiding these buttons when the flags are `False`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pass `has_metadata` flag to hide buttons | Minimal change (2 functions), uses existing GUI flags | None significant | **Chosen** |
| Disable buttons instead of hiding | User sees buttons exist but can't click | Confusing UX — user wonders why buttons are greyed out | Rejected |
| Add validation popup on click | No parameter plumbing needed | User still sees misleading buttons; adds unnecessary code | Rejected |

### Architecture Changes

None. The existing `show_valid_button` / `show_rerun_button` parameters in `TrackerReviewGUI.__init__()` already control button visibility (lines 245–256 of `review_tracking_gui.py`). This hotfix simply passes the correct values.

---

## Implementation Plan

### Phase 1: Pass metadata-existence flag through the call chain
**Goal:** Hide "Finish as Valid" and "Rerun" when no metadata file exists

**Tasks:**
- [ ] Task 1.1 — In `review_tracked_objects_in_video()`, capture `has_metadata = metadata_path.exists()` before the existing if/else block
- [ ] Task 1.2 — Add `has_metadata: bool` parameter to `review_tracking()` function signature
- [ ] Task 1.3 — In `review_tracking()`, pass `show_valid_button=has_metadata` and `show_rerun_button=has_metadata` to `TrackerReviewGUI()`
- [ ] Task 1.4 — In `review_tracked_objects_in_video()`, pass `has_metadata` to the `review_tracking()` call

**Files Modified:**
- `code/scripts/_3_preprocessing/_1_sticker_tracking/review_tracked_handstickers_roi.py`
  - `review_tracked_objects_in_video()` — add `has_metadata` variable, pass to `review_tracking()`
  - `review_tracking()` — add parameter, forward to GUI constructor

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run on a video with **no** metadata file → confirm "Finish as Valid" and "Rerun" buttons are absent; only Label/Delete/Ignore marks and "Proceed with Marked" are available
- [ ] Run on a video with an **existing** metadata file → confirm both buttons appear and work as before
- [ ] First-time flow: mark frames → click "Proceed with Marked" → draw ROI → verify metadata file is created

### Edge Cases
- [ ] Video with no tracked data file AND no metadata file (completely fresh) — buttons should be hidden
- [ ] Video with tracked data file but no metadata file — buttons should be hidden

---

## Documentation Plan

- [ ] No documentation changes needed (internal bugfix, no user-facing docs)

---

## Rollback Plan

1. Revert the single commit on the hotfix branch
2. No data changes, no migrations — purely UI logic

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing returning-user flow | Low | High | `has_metadata` is `True` when file exists, preserving current behavior |
| Forgetting to pass flag in future callers of `review_tracking()` | Low | Med | Default `has_metadata=True` so existing callers are unaffected |

---

## Knowledge Base

No applicable notes found in `docs/development/knowledge-base/`.

---

## References

- `code/scripts/_3_preprocessing/_1_sticker_tracking/review_tracked_handstickers_roi.py` — entry script
- `code/src/preprocessing/stickers_analysis/roi/gui/review_tracking_gui.py` — GUI (already supports `show_valid_button` / `show_rerun_button`)
- `code/src/preprocessing/stickers_analysis/roi/core/review_tracking_orchestrator.py` — controller (no changes needed)
