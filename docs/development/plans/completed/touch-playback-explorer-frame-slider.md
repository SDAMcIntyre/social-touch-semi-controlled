# Plan: Touch Playback Explorer — Frame Slider

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/touch-playback-explorer`
**Branch:** `feature/touch-playback-explorer` (same branch — enhancement to in-progress feature)

---

## Overview

Add a full-width horizontal frame slider to the Touch Playback Explorer GUI,
positioned between the toolbar and the two 3D views. The slider lets the user
scrub to any specific frame for inspection, complementing the existing
timer-based animation playback.

## Problem Statement

The current GUI only supports sequential animation via Play / Play All / Stop
buttons. There is no way to jump directly to a specific frame — the user must
play through all preceding frames or restart. For inspection and debugging of
individual touch events, direct frame access is essential.

## Goals

### In Scope
1. Add a `QSlider(Qt.Horizontal)` spanning the full window width between the
   toolbar and the 3D views
2. Slider tracks the current frame during animation playback
3. Manual slider interaction stops animation, resets heatmap accumulators, and
   renders the selected frame
4. Slider range updates automatically when a new touch is loaded

### Out of Scope
- Heatmap replay from frame 0 to the scrubbed position (would cause lag)
- Keyboard frame stepping (separate enhancement)
- Slider tick marks or frame thumbnails

## Success Criteria

- [ ] Slider appears between the toolbar and the 3D views, spanning full width
- [ ] During animation, slider position tracks the current frame smoothly
- [ ] Dragging the slider stops animation and shows the selected frame
- [ ] Changing touch/trial/session updates slider range and resets to frame 0
- [ ] Play All mode transitions update slider range correctly
- [ ] Slider is disabled when no touch is loaded

---

## Technical Design

### Approach

Restructure the central widget layout from `QHBoxLayout` (flat, splitter-only)
to `QVBoxLayout` (slider row on top, splitter below). The slider row is a
minimal `QHBoxLayout` with a label, the slider, and a value label. The slider
connects to a new `_on_slider_changed` handler; animation code updates the
slider via `blockSignals` to avoid signal recursion.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| QVBoxLayout with slider row above splitter | Full width, clean separation from 3D views, avoids VTK event capture issues | Minor layout restructure | **Chosen** |
| Add slider to existing QToolBar | No layout change needed | Toolbar is cramped; slider would be narrow, hard to use | Rejected |
| QDockWidget with slider | Flexible, floatable | Overkill for a single slider; adds visual complexity | Rejected |
| Replay heatmap 0..N on scrub | Correct cumulative heatmap | Noticeable lag on every drag; poor UX | Rejected — single-frame display on scrub instead |

### Architecture Changes

No new modules or classes. Single method added (`_on_slider_changed`), three
methods modified (`_build_ui`, `_load_touch`, `_render_frame`).

**Knowledge base constraints applied:**
- Slider placed in main `QVBoxLayout`, not overlaid on `QtInteractor` cells
  (ref: `note-rf-explorer-post-layout-bugfix-status.md` — VTK mouse capture)
- `blockSignals` guard on slider updates from animation
  (ref: `note-qt-itemchanged-signal-recursion.md`)

---

## Implementation Plan

### Phase 1: Layout and Slider Widget
**Goal:** Insert the slider row into the UI between toolbar and 3D views.

- [x] Add `QSlider` and `QVBoxLayout` to PyQt5 imports (line 19-30)
- [x] In `_build_ui`: change root layout from `QHBoxLayout` to `QVBoxLayout`
- [x] Build slider row: `QLabel("Frame:")` + `QSlider(Qt.Horizontal)` +
      `QLabel("0 / 0")` as `self._slider_value_label`
- [x] Slider starts disabled with range `[0, 0]`
- [x] Add slider row to root layout before the splitter

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
  — imports, `_build_ui` method

**Dependencies:** None

### Phase 2: Signal Wiring and Sync
**Goal:** Wire slider interaction and keep it in sync with animation.

- [x] Connect `self._frame_slider.valueChanged` to new `_on_slider_changed`
- [x] `_on_slider_changed`: guard → `_stop()` → reset accumulators →
      `_reset_heatmap()` → `_render_frame(value)`
- [x] In `_load_touch`: update slider range with `blockSignals` —
      `setMaximum(max(0, n_frames - 1))`, `setValue(0)`, `setEnabled(n_frames > 0)`,
      update `_slider_value_label`
- [x] In `_render_frame`: update slider position with `blockSignals` —
      `setValue(frame_idx)`, update `_slider_value_label` text

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
  — new `_on_slider_changed` method, modifications to `_load_touch` and
  `_render_frame`

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Launch explorer — slider appears between toolbar and 3D views, full width
- [ ] Select a touch — slider range updates, shows "1 / N"
- [ ] Click Play — slider tracks frame position during animation
- [ ] Drag slider during playback — animation stops, selected frame renders
- [ ] Change trial/session — slider resets to 0 with correct range
- [ ] Play All — slider tracks across touch transitions, range updates per touch
- [ ] Touch with 1 frame — slider locked at 0, enabled but undraggable
- [ ] No touch loaded — slider disabled

### Edge Cases
- [ ] Touch with 0 frames — slider disabled, range [0, 0]
- [ ] Rapid slider dragging — no crashes, last position renders correctly
- [ ] Play All with single touch in trial — no queue issues after slider scrub

---

## Documentation Plan

- [ ] No external docs needed — this is a UI control within an existing GUI

---

## Rollback Plan

Single file, single branch. Revert the commit that adds the slider.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Slider `valueChanged` triggers during `blockSignals` update | Low | Med — recursive rendering | Standard `blockSignals(True/False)` guard, same pattern as combo cascade |
| VTK event capture steals slider mouse events | Low | High — slider unusable | Slider in separate layout row, not overlaid on QtInteractor (confirmed safe by KB) |
| Heatmap shows single-frame data instead of cumulative on scrub | Expected | Low — acceptable for inspection | Document behavior; cumulative replay rejected for performance reasons |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Layout & widget | ~30 min | None |
| Phase 2: Signal wiring & sync | ~30 min | Phase 1 |
| Manual testing | ~15 min | Phase 2 |
