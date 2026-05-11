# Plan: Touch Population Explorer Improvements

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements`

---

## Overview

Four targeted improvements to `TouchPopulationExplorer`: make the filter rectangle visible (it is currently white-on-white and undetectable), change the default scatter axes to the IFF-windowed velocity and pressure features, expose width/height spinboxes so the rectangle dimensions can be set precisely, and add a single-touch selection mode where clicking a scatter dot shows only that touch's contact-point heatmap on the 3D forearm.

## Problem Statement

1. **Invisible rectangle** — `DraggableFilterRect` uses `edgecolor="white"` and `facecolor="white"` on a white scatter background. The selection box is completely undetectable, so users cannot tell what data is being filtered or where the rectangle boundary lies.

2. **Wrong default axes** — The scatter currently pre-selects the first two feature columns in alphabetical order. The most informative default pair is `hand_velocity_amplitude_mean_during_iff` (X) and `pressure_mean_during_iff` (Y), which map directly to the IFF-windowed kinematic and mechanical dimensions.

3. **No rectangle dimension control** — The filter rectangle can only be resized by dragging edges/corners on the scatter. There is no way to enter a precise width or height numerically, making it hard to reproduce a specific filter region.

4. **No per-touch inspection** — Users can filter a region of the scatter, but cannot inspect a single touch event in isolation. There is no way to select one dot and see only that touch's contact-point footprint on the forearm.

## Goals

### In Scope
1. Make the `DraggableFilterRect` patch clearly visible with a contrasting edge and fill.
2. Add a `disconnect()` method to `DraggableFilterRect` for clean event-handler teardown.
3. Set default X/Y axes to the two IFF-windowed features, with index-based fallback.
4. Add "W:" and "H:" `QDoubleSpinBox` widgets below the scatter that display and control the rectangle width and height; dragging the rectangle updates the spinboxes, and changing a spinbox updates the rectangle.
5. Add a "Single touch" toggle button that replaces the filter rectangle with a click-to-select interaction.
6. In single-touch mode: nearest-dot click selection, black ring highlight on scatter, heatmap of the one selected touch in the 3D view, informative label.
7. Properly restore region-filter mode (including rectangle) when the toggle is switched off or a session is changed.

### Out of Scope
- Animating or replaying the selected touch (that is the job of `TouchPlaybackExplorer`).
- Keyboard navigation between touches.
- Exporting the selected touch data to a file.
- Changing the rectangle styling in `RFFeatureSpaceExplorer` scatter (it will change as a side-effect of the shared class, which is acceptable).

## Success Criteria

- [ ] On startup with Stage 3 features, the scatter shows a clearly visible blue dashed rectangle.
- [ ] Default X combo pre-selects `hand_velocity_amplitude_mean_during_iff`; Y combo pre-selects `pressure_mean_during_iff`. If either is absent, falls back without error.
- [ ] Pressing "Single touch" removes the rectangle; scatter and 3D view remain visible.
- [ ] Clicking a scatter dot in single-touch mode draws a black ring around it and updates the 3D heatmap to show only that touch's contact points.
- [ ] Clicking a different dot moves the ring and updates the 3D view.
- [ ] Pressing "Single touch" again restores the filter rectangle and normal region-filter behaviour.
- [ ] Changing session while in single-touch mode resets to region mode automatically.
- [ ] "W:" and "H:" spinboxes appear below the scatter and reflect the rectangle dimensions on startup.
- [ ] Dragging the rectangle updates both spinboxes live.
- [ ] Editing a spinbox repositions/resizes the rectangle (centre held constant, dimensions changed).
- [ ] Spinboxes are hidden / disabled in single-touch mode.
- [ ] No orphaned matplotlib event handlers after switching modes (verified by mode cycling without crash).

---

## Technical Design

### Approach

All changes are confined to two files. `DraggableFilterRect` (in `rf_feature_space_explorer.py`) is extended with CID storage and a `disconnect()` method, and its patch styling is changed to a visible blue dashed box. `TouchPopulationExplorer` (in `touch_population_explorer.py`) gains the default-axis constants, state variables, toolbar button, and three new methods.

The single-touch mode lifecycle:
- **Enter**: call `_rect.disconnect()`, set `_rect = None`, register a new `button_press_event` handler via `mpl_connect`, store its CID.
- **Click**: find nearest visible dot by normalised-axis Euclidean distance, store index, redraw scatter with ring overlay, call `_apply_single_touch_display()`.
- **Exit**: call `mpl_disconnect` on the stored CID, clear selection, redraw scatter, call `_init_filter_rect()`.

The knowledge base flags the `blockSignals` pattern for combo initialisation (already in use) and the requirement to store mpl CIDs for clean `mpl_disconnect` — both addressed directly.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| CID storage + `disconnect()` on `DraggableFilterRect` | Clean teardown; no orphaned handlers | Requires modifying shared class | **Chosen** |
| Set `_patch.set_visible(False)` instead of `disconnect()` | Fewer changes | Event handlers remain registered; clicks still routed to rect | Rejected |
| Dark scatter background so white rect is visible | No code change needed | Background clash with legend/labels; aesthetic regression | Rejected |
| Separate `SingleTouchRect` subclass | No shared-class modification | Unnecessary indirection for a simple patch-style change | Rejected |

### Architecture Changes

- `DraggableFilterRect` gains `_cid_press`, `_cid_motion`, `_cid_release` attributes and a `disconnect()` public method. Patch styling changes (`edgecolor`, `facecolor`, `linestyle`).
- `TouchPopulationExplorer` gains:
  - Two `QDoubleSpinBox` widgets (`_w_spinbox`, `_h_spinbox`) in a bar below the scatter canvas, styled identically to those in `RFFeatureSpaceExplorer`.
  - A `_controls_updating: bool` guard flag to prevent re-entrant signal loops between the spinboxes and the rect (knowledge-base `blockSignals` pattern).
  - `_update_rect_from_controls()` — reads spinboxes, computes new bounds centred on the existing rect centre, calls `_rect.set_bounds()`.
  - `_update_controls_from_rect(x_min, x_max, y_min, y_max)` — updates spinbox values from rect geometry.
  - Three instance variables (`_single_touch_mode`, `_selected_touch_idx`, `_single_touch_cid`), one toolbar button (`_single_touch_btn`), and three new methods (`_set_single_touch_mode`, `_on_scatter_click_single_touch`, `_apply_single_touch_display`).
  - Existing methods `_populate_axis_combos`, `_draw_scatter`, `_on_rect_changed`, `_on_axis_changed`, `_on_checkbox_changed`, and `_load_session` are modified.
- No new modules. No DAG config or pipeline changes.

---

## Implementation Plan

### Phase 1: `DraggableFilterRect` — visible patch + `disconnect()`

**Started:** 2026-05-05
**Completed:** 2026-05-05

**Goal:** Make the selection rectangle visible and enable clean teardown.

- [x] In `__init__`, save the three `mpl_connect` return values as `self._cid_press`, `self._cid_motion`, `self._cid_release`.
- [x] Add `disconnect()` method: calls `mpl_disconnect` on the three CIDs and calls `self._patch.remove()`.
- [x] Change patch construction: `edgecolor="royalblue"`, `facecolor="royalblue"`, `alpha=0.10`, `linestyle="--"`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` — `DraggableFilterRect.__init__` and new `disconnect()` method

**Dependencies:** None

### Phase 2: Default axes

**Goal:** Pre-select the two IFF-windowed features on startup and session change.

**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Add module-level constants `_DEFAULT_X_FEATURE` and `_DEFAULT_Y_FEATURE` above the class.
- [x] Rewrite the index-selection logic in `_populate_axis_combos()` to prefer the named features.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — constants + `_populate_axis_combos`

**Dependencies:** None (independent of Phase 1)

### Phase 3: Width/height spinboxes

**Started:** 2026-05-05
**Completed:** 2026-05-05

**Goal:** Let the user type precise rectangle dimensions; keep spinboxes and rectangle in sync.

- [x] Add `QDoubleSpinBox` and `QHBoxLayout` to existing imports.
- [x] Add `_controls_updating: bool = False` to `__init__`.
- [x] Build a horizontal bar (`_build_rect_controls()`) containing "W:" `_w_spinbox` and "H:" `_h_spinbox`:
  - Range: `0.001` to `1e9`; single step: `0.1`; decimals: `4`.
  - `valueChanged` connected to `_update_rect_from_controls()`.
  - Insert the bar into `bottom_layout` between the canvas and the right panel (or below the canvas in a sub-`QVBoxLayout`).
- [x] Implement `_update_rect_from_controls()`: guard with `_controls_updating`, read spinbox values, compute centre of current rect, call `_rect.set_bounds(cx - w/2, cx + w/2, cy - h/2, cy + h/2)`.
- [x] Implement `_update_controls_from_rect(x_min, x_max, y_min, y_max)`: set `_controls_updating = True`, set `_w_spinbox.setValue(x_max - x_min)` and `_h_spinbox.setValue(y_max - y_min)`, set `_controls_updating = False`.
- [x] Call `_update_controls_from_rect` at the end of `_init_filter_rect()` so spinboxes are populated on startup.
- [x] Call `_update_controls_from_rect` inside `_on_rect_changed()` (before `draw_idle`) so dragging keeps spinboxes live.
- [x] Hide spinboxes (`setVisible(False)`) when entering single-touch mode; restore (`setVisible(True)`) on exit.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — imports, `__init__`, `_build_ui` / new `_build_rect_controls`, `_on_rect_changed`, `_init_filter_rect`, new `_update_rect_from_controls`, new `_update_controls_from_rect`

**Dependencies:** Phase 1 (spinboxes call `_rect.set_bounds()` which exists in Phase 1)

### Phase 4: Single-touch mode

**Started:** 2026-05-05
**Completed:** 2026-05-05

**Goal:** Add the toggle button and the full single-touch interaction.

- [x] Add `QPushButton` to existing imports.
- [x] Add `_single_touch_mode`, `_selected_touch_idx`, `_single_touch_cid` to `__init__`.
- [x] Add "Single touch" checkable `QPushButton` to `_build_toolbar()`; connect `toggled` to `_set_single_touch_mode`.
- [x] Implement `_set_single_touch_mode(enabled)`: on enable, call `_rect.disconnect()` + `mpl_connect` for click handler; on disable, `mpl_disconnect` + `_init_filter_rect()`.
- [x] Implement `_on_scatter_click_single_touch(event)`: find nearest visible dot via normalised-axis distance, update `_selected_touch_idx`, redraw scatter, call `_apply_single_touch_display()`.
- [x] Implement `_apply_single_touch_display()`: build single-touch `cp_mask`, call `_compute_heatmap`, push to PyVista cloud, update label.
- [x] In `_draw_scatter()`: after the legend, if `_single_touch_mode` and a touch is selected, overlay a black ring at the selected dot's coordinates.
- [x] Update `_on_axis_changed()`: call `_rect.disconnect()` before clearing `_rect`; skip `_init_filter_rect()` in single-touch mode.
- [x] Update `_on_checkbox_changed()`: route to `_apply_single_touch_display()` in single-touch mode.
- [x] Update `_load_session()`: disconnect single-touch handler and reset mode before the existing reset logic.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — imports, `__init__`, `_build_toolbar`, `_draw_scatter`, `_on_axis_changed`, `_on_checkbox_changed`, `_load_session`, and three new methods

**Dependencies:** Phase 1 (requires `disconnect()`) and Phase 3 (spinbox visibility toggle)

---

## Testing Plan

### Unit Tests

No new unit tests required — the GUI components are not covered by the existing automated test suite, which focuses on DAG config models and data loaders.

### Manual Verification

- [ ] Launch the explorer via the pipeline GUI with a session that has Stage 3 features extracted.
- [ ] Confirm blue dashed rectangle is visible on the scatter on startup.
- [ ] Confirm X axis defaults to `hand_velocity_amplitude_mean_during_iff`, Y to `pressure_mean_during_iff`.
- [ ] Drag rectangle corners and edges — 3D heatmap updates; rectangle remains visible; "W:" and "H:" spinboxes update live.
- [ ] Type a new value in "W:" spinbox — rectangle width changes, centre held constant, 3D updates.
- [ ] Type a new value in "H:" spinbox — rectangle height changes, centre held constant, 3D updates.
- [ ] Click "Single touch" — button depresses, rectangle disappears, spinboxes hidden, label reads "Click a dot to select a touch".
- [ ] Click a scatter dot — black ring appears, 3D shows only that touch.
- [ ] Click another dot — ring moves, 3D updates.
- [ ] Change X axis while in single-touch mode — scatter redraws, ring remains on the same touch (new coordinates).
- [ ] Uncheck a gesture type while in single-touch mode — scatter redraws without that type; if selected touch belongs to that type, 3D clears gracefully.
- [ ] Click "Single touch" again — mode exits, rectangle reappears, spinboxes reappear with correct values, filtering resumes.
- [ ] Change session while in single-touch mode — mode resets to region mode; button appears unpressed.
- [ ] Load a session without Stage 3 features — "no features" message shown; single-touch button is present but entering the mode shows empty scatter gracefully.

### Edge Cases

- [ ] Session with a single feature (X and Y would map to the same feature) — no crash.
- [ ] Session where neither default feature name exists — falls back to index 0 / 1.
- [ ] Click in single-touch mode with all gesture types unchecked — no visible dots, no crash, no selection change.
- [ ] Rapid mode toggling (on/off/on/off) — no orphaned event handlers, no duplicate rect patches.

---

## Documentation Plan

- [ ] No CLAUDE.md update needed (no architectural pattern change).
- [ ] No README update needed (internal GUI enhancement).

---

## Rollback Plan

All changes are in two GUI files with no pipeline or data-format impact. Rollback is `git revert` on the feature branch commit, or restoring the two files from `feature/touch-population-explorer`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `mpl_disconnect` called with stale CID after figure is closed | Low | RuntimeWarning | Guard `_apply_single_touch_display` and `disconnect()` with try/except pass (existing pattern in `_deferred_start`) |
| `_patch.remove()` raises if patch was never added to axes | Low | Crash on edge path | Wrap in try/except in `disconnect()` |
| `_DEFAULT_X_FEATURE` / `_DEFAULT_Y_FEATURE` absent in a session | Med | Wrong default (but silent) | `index()` guarded by `in` check with index fallback — already in plan |
| Blue dashed rect clashes with scatter aesthetics | Low | Cosmetic only | Easily tuned by adjusting `edgecolor`/`alpha`/`linestyle` constants |
