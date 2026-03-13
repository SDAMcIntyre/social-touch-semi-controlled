# Plan: Viewer Frame Offset Fix

**Created:** 2026-03-11 00:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `fix/viewer-frame-offset`

---

## Overview

The Neural Kinect Viewer displays a two-layer frame synchronization bug: the kinect RGB-D point cloud, the overlay data (stickers, contact points, hand mesh), and the 1D timeseries panel (Depth, Area) are each offset by 1 frame relative to each other. This plan investigates the root causes via targeted diagnostics, then applies minimal viewer-side fixes.

## Problem Statement

When stepping frame-by-frame in the Neural Kinect Viewer:
- At slider position **t**: the kinect point cloud shows physical time t
- At slider position **t+1**: the sticker/contact/handmesh overlays show the data matching the point cloud at t
- At slider position **t+2**: the 1D timeseries (Depth, Area) shows the values matching the overlays at t+1

This means two independent 1-frame offsets stack, so the timeseries is 2 frames behind the point cloud. The bug undermines visual correlation of neural activity with spatial contact data — the primary purpose of this viewer.

Observed in one session so far; unknown whether it affects all sessions.

## Goals

### In Scope
1. Diagnose the root cause of both frame offsets with targeted diagnostic code
2. Fix the timeseries cursor offset (Offset B) in the viewer
3. Fix or provide a workaround for the overlay-vs-point-cloud offset (Offset A)
4. Document the root cause in a knowledge-base note

### Out of Scope
- Rewriting the merge pipeline (`merge_neural_and_kinect_data.py`)
- Re-running preprocessing for existing sessions
- Addressing the hardware-level RGB-D lag noted in `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md` (related but separate)

## Success Criteria

- [ ] At any contact event, the point cloud, overlay data, and timeseries all show consistent contact/no-contact at the same slider frame
- [ ] Stepping frame-by-frame across a contact onset shows simultaneous transitions in all three representations
- [ ] No regressions: viewer launches and operates at the same performance

---

## Technical Design

### Approach

Two separate offsets require two separate fixes, both in the viewer:

**Offset A (overlay vs point cloud, 1 frame):** Likely caused by a discrepancy between how frames are indexed during preprocessing (`KinectMKV.__iter__` which uses `read()` and skips empty frames) versus how they are indexed during viewer playback (`KinectPointCloudView.__getitem__` which uses `read_once()` and does not skip). If the first MKV frame is empty, all preprocessing outputs are shifted by 1. Diagnosed first, then fixed with an index offset applied when loading overlay data.

**Offset B (timeseries vs overlay, 1 frame):** Caused by the timeseries cursor position being computed via float scaling (`int(frame_idx * _neural_scale)`) which can land before the actual kinect row in the merged CSV. With `.ffill()`, this displays the previous frame's depth/area. Fixed by replacing the float-scaled cursor with an exact kinect-row-index lookup table.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Viewer-side fixes** (lookup table + index offset) | No pipeline re-run, minimal code change, immediately testable | Does not fix root cause in preprocessing | Chosen for Offset B; chosen as interim for Offset A |
| **Fix `__iter__` to not skip frames** | Fixes root cause for Offset A | Requires re-running preprocessing for all sessions; may break other code relying on skip behavior | Rejected for now; noted as future improvement |
| **Use `round()` instead of `int()` for cursor** | 1-character change | Only fixes Offset B partially — fails when `_total_frames != len(kinect_csv)` | Rejected |
| **Fix in merge script** | Source-level fix for Offset B | Merge script doesn't know MKV frame count; requires pipeline re-run | Rejected |

### Architecture Changes

No new modules. All changes are in `code/src/merging/gui/neural_kinect_scene_viewer.py`:
- New attribute `_kinect_sample_indices: list[int]` for exact cursor positioning
- New method `NeuralDataPanel.update_cursor_at(sample_idx: int)` replacing the scale-factor-based cursor
- Diagnostic prints (temporary, removed in final phase)

### Knowledge-Base Constraints

- `bug-neural-kinect-viewer-initial-render.md` — documents VTK clipping issues in the same viewer; no direct overlap but confirms the viewer has subtle state-dependent rendering issues
- `note-somatosensory-units-and-calculations.md` — contact depth/area units (mm, mm^2); no temporal concerns
- `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md` — documents a known ~1-frame RGB-D lag in certain sessions (ST13-02 blocks 08-11, ST13-03 blocks 06-07). This may be the same phenomenon as Offset A, or a compounding hardware issue. Diagnostics will help differentiate.

---

## Implementation Plan

### Phase 1: Diagnostic verification
**Goal:** Confirm the root cause of both offsets before writing any fix.
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 1.1 — Add init-time diagnostics comparing frame counts across data sources (`_total_frames`, sticker array length, hand mesh length, contact points length, kinect rows in merged CSV)
- [ ] Task 1.2 — Add cursor-position diagnostics comparing `int(k * _neural_scale)` vs actual kinect row index for the first 20 frames
- [ ] Task 1.3 — Run the viewer on the session where the offset was observed and capture diagnostic output
- [ ] Task 1.4 — Analyze output to confirm/refute hypotheses A1-A3 and B1

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — ~25 lines of temporary print statements

**Dependencies:** None

### Phase 2: Fix Offset B (timeseries cursor)
**Goal:** Eliminate the 1-frame offset between overlay data and the timeseries panel.
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 2.1 — Build `_kinect_sample_indices` lookup table at init (from `merged_df['time_kinect'].notna()` row indices)
- [ ] Task 2.2 — Add `NeuralDataPanel.update_cursor_at(sample_idx: int)` method
- [ ] Task 2.3 — Update `_update_frame()` to use exact kinect row index instead of float-scaled cursor
- [ ] Task 2.4 — Verify fix: timeseries and overlay data now align at contact events

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py`
  - After line 594: add `_kinect_sample_indices` init (~6 lines)
  - `NeuralDataPanel`: add `update_cursor_at()` (~8 lines)
  - `_update_frame()` line 1254-1255: change cursor call (~4 lines)

**Dependencies:** Phase 1 (diagnostics confirm B1)

### Phase 3: Fix Offset A (overlay vs point cloud)
**Goal:** Eliminate the 1-frame offset between the kinect point cloud and overlay data.
**Started:** —
**Completed:** —

The exact fix depends on Phase 1 diagnostic results. The most likely fix:

**Tasks:**
- [ ] Task 3.1 — Based on diagnostics, determine the offset direction and magnitude
- [ ] Task 3.2 — Apply index offset when accessing overlay data in `_update_frame()`: shift `frame_idx` by the diagnosed offset when indexing into `_contact_pts_by_frame`, `_stickers_xyz_dict`, and `_hand_manager`
- [ ] Task 3.3 — Verify fix: point cloud and overlay data now align at contact events

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — overlay access lines in `_update_frame()` (~6 lines changed)

**Dependencies:** Phase 1 (diagnostics confirm offset direction and cause)

### Phase 4: Clean up and document
**Goal:** Remove diagnostics, clean dead code, document findings.
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 4.1 — Remove all `[DIAG-*]` print statements
- [ ] Task 4.2 — Remove unused `_neural_scale` and old `update_cursor()` if fully replaced
- [ ] Task 4.3 — Write knowledge-base note: `docs/development/knowledge-base/bug-viewer-frame-offset.md`
- [ ] Task 4.4 — Cross-reference from `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — remove ~25 lines of diagnostics
- `docs/development/knowledge-base/bug-viewer-frame-offset.md` — new file
- `docs/development/knowledge-base/README.md` — add index entry

**Dependencies:** Phases 2 and 3

---

## Testing Plan

### Manual Verification
- [ ] Open viewer on the session where the offset was originally observed
- [ ] Navigate to a contact event onset; confirm point cloud, overlays, and timeseries all transition to contact simultaneously
- [ ] Step backward through the same event; confirm all three transition to no-contact simultaneously
- [ ] Navigate to frame 0 — no IndexError, all data renders
- [ ] Navigate to last frame — no IndexError, all data renders
- [ ] If possible, test on a second session to check generality

### Edge Cases
- [ ] Session where `merged_csv_path=None` — viewer should still work (no timeseries panel)
- [ ] Session with very short recordings (< 10 frames)
- [ ] Frame where contact appears and disappears in consecutive frames

---

## Documentation Plan

- [ ] Create `docs/development/knowledge-base/bug-viewer-frame-offset.md` documenting root cause and resolution
- [ ] Update `docs/development/knowledge-base/README.md` with new entry
- [ ] Add cross-reference note to `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`

---

## Rollback Plan

All changes are in a single file (`neural_kinect_scene_viewer.py`). Revert the commit to restore previous behavior. No data migrations or external state changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Offset A is session-specific (only some MKVs have empty first frame) | Medium | Fix may over-correct for sessions without the issue | Diagnostics will reveal; if variable, make offset configurable |
| Offset A is actually a hardware-level lag, not a software bug | Medium | Viewer-side offset is a workaround, not a fix | Cross-reference with RGB-D frame lag idea; document as known limitation |
| `_kinect_sample_indices` has fewer entries than `_total_frames` | Medium | IndexError on high frame indices | Bounds check with fallback to float scaling |
| Fixing overlay offset breaks alignment for sessions that were correct | Low | Regression | Test on multiple sessions before merging |

---

## References

- Related Idea: `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
- Related Bug: `docs/development/knowledge-base/bug-neural-kinect-viewer-initial-render.md`
- Main file: `code/src/merging/gui/neural_kinect_scene_viewer.py`
- Merge script: `code/scripts/_4_merging/merge_neural_and_kinect_data.py`
- MKV reader: `code/src/preprocessing/common/data_access/kinect_mkv_manager.py`
