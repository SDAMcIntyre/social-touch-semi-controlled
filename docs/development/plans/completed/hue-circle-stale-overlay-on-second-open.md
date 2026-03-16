# Plan: Fix Hue Circle Showing Stale Overlay on Second Window Open

**Date:** 2026-02-22
**Author:** Claude Code
**Status:** Completed
**Branch:** `dev`

---

## Overview

When the arm segmentation interactive window is opened for a second frame, the hue circle displays the arc from the *previous* window instead of the arc corresponding to the new initial H range. The fix adds a single `_refresh()` call inside `on_layout`, mirroring the fix already applied to `BUTTON_DOWN` mouse events.

## Problem Statement

Each call to `_display_pointcloud` builds a fresh `SceneWidget` for the hue circle and sets its background with the correct initial overlay via `set_background`. However, Open3D invalidates `set_background` whenever the `SceneWidget` undergoes a resize or internal scene reset. The `on_layout` callback explicitly resizes the widget (`hue_scene_widget.frame = ...`) but contains no `_refresh()` call afterwards. As a result, after layout the widget's background reverts to stale GPU state from the previous window's render — displaying the previous H-range arc instead of the new one.

The same invalidation on `BUTTON_DOWN` was already identified and fixed (see `bug-hue-circle-drag-not-working.md`), but the layout path was missed.

## Goals

### In Scope
1. Re-apply the hue wheel overlay immediately after `on_layout` assigns `hue_scene_widget.frame`.
2. Expose the `_refresh` closure from `_make_hue_range_circle` so the `_display_pointcloud` layout callback can invoke it.

### Out of Scope
- Changes to any other widget or processing step.
- Refactoring `_make_hue_range_circle`'s return signature beyond the minimal addition.
- Fixing any other potential stale-render paths (e.g., window resize after the first layout).

## Success Criteria

- [ ] Opening the segmentation window for a second frame shows the hue circle arc that matches the new initial H range, not the arc from the previous window.
- [ ] Existing drag, click, and text-edit interactions on the hue circle continue to work correctly.
- [ ] No regressions in the downsampling or clustering interactive steps.

---

## Technical Design

### Approach

`_make_hue_range_circle` already defines a `_refresh` closure (used internally by `_on_mouse`). Expose it via the returned fragments dict under the key `"refresh"`. In `_display_pointcloud`, store this callable as `hue_refresh_fn` and call it at the end of the `on_layout` branch that handles `hue_scene_widget`, after `hue_scene_widget.frame` is set.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Call `_refresh` in `on_layout` (chosen) | Minimal change; matches existing pattern from `BUTTON_DOWN` fix | None | Chosen |
| Call `set_background` again inside `_make_hue_range_circle` using `scene_widget.enable_scene_caching` tricks | Self-contained | No hook into layout events from inside the factory | Rejected |
| Re-create the `SceneWidget` on every layout | Guaranteed fresh state | Major structural change; would reset drag state mid-session | Rejected |

### Architecture Changes

- `_make_hue_range_circle`: add `"refresh"` key to the returned dict (value: the existing `_refresh` closure). No other signature change.
- `_display_pointcloud`: unpack `hue_elements["refresh"]` → `hue_refresh_fn`; call `hue_refresh_fn()` inside `on_layout` after `hue_scene_widget.frame` is set.

---

## Implementation Plan

### Phase 1: Expose `_refresh` from `_make_hue_range_circle`

**Goal:** Make the refresh closure available to the caller.

- [x] Task 1.1 — In `_make_hue_range_circle`, add `"refresh": _refresh` to the returned dict alongside `"top"`, `"scene"`, `"bottom"`.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — add one key to the return dict (~line 672)

**Dependencies:** None

### Phase 2: Call `_refresh` in `on_layout`

**Goal:** Ensure the correct overlay is re-applied every time the layout positions the hue widget.

- [x] Task 2.1 — In `_display_pointcloud`, unpack `hue_elements["refresh"]` and store it as `hue_refresh_fn` (alongside `hue_scene_widget`).
- [x] Task 2.2 — At the end of the `if hue_scene_widget is not None:` branch in `on_layout`, after `panel_bottom.frame = ...`, call `hue_refresh_fn()`.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — two lines changed in `_display_pointcloud` (~lines 845, 1056)

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run the forearm extraction pipeline on two consecutive frames in interactive mode; confirm the hue circle for the second frame shows the default (or configured) arc, not the arc from the first frame.
- [ ] Drag both handles on the hue circle in the first and second windows; confirm drag still works.
- [ ] Edit the H start/end text boxes; confirm the overlay updates in real time.
- [ ] Click "Process" then "Continue" without touching the hue circle; confirm params carry through correctly.

### Edge Cases
- [ ] Window resize mid-session: confirm the overlay re-applies after resize (secondary benefit of the fix).
- [ ] Single-frame session (only one window opened): confirm no regression.

---

## Documentation Plan

- [x] Add `"refresh"` key to the docstring of `_make_hue_range_circle` return value description.

---

## Rollback Plan

The change is two lines in one file. Revert by removing the `"refresh"` entry from the return dict and the `hue_refresh_fn()` call in `on_layout`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_refresh` called before the SceneWidget's renderer is ready | Low | Medium | `_refresh` already safely calls `set_background`; same guard is already relied upon for BUTTON_DOWN |
| On some platforms `on_layout` is called many times (per-frame) | Low | Low | `_refresh` is cheap (numpy array copy + `set_background`); no perceptible cost |

---

## References

- Related knowledge-base entry: `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md`
- Affected file: `code/src/preprocessing/forearm_extraction/arm_segmentation.py`
- `_make_hue_range_circle` return dict: ~line 672
- `on_layout` hue branch: ~line 1034
