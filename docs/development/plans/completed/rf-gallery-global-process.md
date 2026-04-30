# Plan: RF Gallery — Global Deferred Processing

**Date:** 2026-04-28
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/gmm-gallery-feature-ranges`

---

## Overview

Replace the single threshold-specific "Process" button in the RF cluster gallery viewer with a global deferred-processing model. All settings controls stop triggering automatic scene rebuilds; changes are staged until the user explicitly applies them via one of two new buttons: **"Process"** (current cell only, for fast preview) and **"Process All"** (pre-builds all visible sessions).

## Problem Statement

The gallery viewer currently rebuilds the full 3D scene on every single control interaction (~17 auto-connected signals). This makes settings exploration slow and noisy — every intermediate drag or click causes an expensive PyVista redraw. The one existing "Process" button only covers the Delaunay threshold, not the other settings. There is no way to adjust multiple settings and apply them all at once, nor to preview on the current cell before committing to all sessions.

## Goals

### In Scope
1. Remove all auto-trigger scene rebuilds from settings controls (sliders, spinboxes, checkboxes, combos, color pickers).
2. Add **"Process"** button — applies all staged settings to the current visible cell only.
3. Make the existing threshold-specific "Process" button global by merging it into the two new buttons, which handle all settings (including threshold) together.
4. Add **"Process All"** button — saves current session threshold, pre-builds Delaunay meshes for all visible sessions using their per-session stored thresholds, then rebuilds the current view.

### Out of Scope
- Visual "dirty" indicator on settings that have unsaved changes.
- Applying a uniform threshold to all sessions at once (threshold remains intentionally per-session).
- Background/async processing of all-session Delaunay builds.

## Success Criteria

- [ ] Changing any control (slider, checkbox, spinbox, color picker, combo) does not trigger a scene rebuild.
- [ ] Clicking **"Process"** rebuilds the current cell with all currently staged settings.
- [ ] Clicking **"Process All"** pre-builds Delaunay meshes for all sidebar-visible sessions and rebuilds the current cell.
- [ ] Navigating to a new cell always renders it immediately with the current `_settings` (no manual process needed for navigation).
- [ ] The old threshold-specific "Process" button is gone; its behaviour is subsumed by the new buttons.
- [ ] Per-session thresholds and camera state remain correct across navigation.

---

## Technical Design

### Approach

Add a `_stage_setting()` method that sets an attribute on `_settings` without calling `_apply_setting_change()`. Rewire all auto-trigger signal connections to call `_stage_setting()` instead of `_set_and_rebuild()`. Update `_pick_color()` and `_on_cmap_changed()` similarly. Remove the threshold-specific `_threshold_apply_btn`. Add two new QPushButton widgets in a fixed bar anchored at the bottom of the right settings panel (outside the scroll area).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Stage all settings, two explicit process buttons | Clean UX; preview vs. commit model | One more button to explain | **Chosen** |
| Keep auto-triggers but add debounce | Less code change | Still rebuilds on every interaction; complexity of debounce timers | Rejected |
| Single "Process All" button only | Simpler | No cheap preview; must wait for all-session build each time | Rejected |

### Architecture Changes

Single file change only:

- **`rf_cluster_gallery_viewer.py`** — the following modifications:
  - Add `_stage_setting(attr, value)` method (sets attribute, no rebuild).
  - Rewire ~15 signal connections in `_build_right_settings_panel()` from `_set_and_rebuild` to `_stage_setting`.
  - Update `_pick_color()` to stage instead of rebuild.
  - Update `_on_cmap_changed()` to stage instead of calling `_apply_setting_change()`.
  - Remove `_threshold_apply_btn` widget and connection.
  - Remove `_on_threshold_changed()` method.
  - Add `_process_btn_bar` widget (fixed, below scroll area) with two `QPushButton` widgets.
  - Add `_on_process_current()` — saves current session threshold from spinbox, calls `_apply_setting_change()`, saves thresholds.
  - Add `_on_process_all()` — saves current session threshold, iterates `_visible_keys()` to warm Delaunay cache for all visible sessions, calls `_apply_setting_change()`, saves thresholds.

`_set_and_rebuild()` is retained (no external callers to break) but left unused — it can be cleaned up later.

---

## Implementation Plan

### Phase 1: Decouple settings from auto-rebuild

**Started:** 2026-04-28
**Completed:** 2026-04-28

**Goal:** All controls become purely staging — no immediate scene side-effects.

- [x] Add `_stage_setting(self, attr: str, value) -> None` method (one line: `setattr(self._settings, attr, value)`).
- [x] In `_build_right_settings_panel()`, replace all 15 `_set_and_rebuild` lambdas with `_stage_setting` lambdas.
- [x] In `_pick_color()`, replace `self._set_and_rebuild(attr, color.name())` with `setattr(self._settings, attr, color.name())` (keep the button stylesheet update; remove the rebuild).
- [x] In `_on_cmap_changed()`, remove the `self._apply_setting_change()` call (keep the two attribute assignments).
- [x] Remove `_threshold_apply_btn` widget, its layout row, and its `clicked` connection.
- [x] Remove `_on_threshold_changed()` method.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — signal rewiring and method removal.

**Dependencies:** None

---

### Phase 2: Add the two Process buttons

**Started:** 2026-04-28
**Completed:** 2026-04-28

**Goal:** Expose "Process" and "Process All" as explicit action buttons, always visible.

- [x] In `_build_right_settings_panel()`, after assembling the scroll area, create a `QWidget` button bar:
  ```python
  btn_bar = QWidget()
  btn_layout = QHBoxLayout(btn_bar)
  btn_layout.setContentsMargins(8, 6, 8, 6)
  self._process_btn = QPushButton("Process")
  self._process_all_btn = QPushButton("Process All")
  btn_layout.addWidget(self._process_btn)
  btn_layout.addWidget(self._process_all_btn)
  self._process_btn.clicked.connect(self._on_process_current)
  self._process_all_btn.clicked.connect(self._on_process_all)
  ```
- [x] Add `btn_bar` to the right panel's outer (non-scroll) layout after the scroll area.
- [x] Implement `_on_process_current()`:
  ```python
  def _on_process_current(self) -> None:
      if self._current_cell is None:
          return
      self._session_thresholds[self._current_cell.session_id] = self._threshold_spin.value()
      self._apply_setting_change()
      self._save_thresholds()
  ```
- [x] Implement `_on_process_all()`:
  ```python
  def _on_process_all(self) -> None:
      if self._current_cell is None:
          return
      self._session_thresholds[self._current_cell.session_id] = self._threshold_spin.value()
      for key in self._visible_keys():
          self._get_delaunay_mesh(self._gallery_data.cells[key])
      self._apply_setting_change()
      self._save_thresholds()
  ```

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — button bar and handler methods.

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Launch the gallery viewer with a loaded dataset. Drag several sliders and toggle checkboxes — confirm the 3D view does **not** update.
- [ ] Click **"Process"** — confirm the view updates to reflect all staged changes.
- [ ] Change colormap and opacity, click **"Process"** again — confirm incremental updates work.
- [ ] Navigate to another cell — confirm it renders immediately with current `_settings` and no process button required.
- [ ] Navigate back — confirm threshold spinbox shows the correct stored value for the original session.
- [ ] Change threshold spinbox, click **"Process"** — confirm Delaunay mesh is rebuilt for current session only.
- [ ] Click **"Process All"** — confirm current view rebuilds, then navigate quickly through sidebar cells and confirm no rebuild delay (Delaunay cache pre-warmed).
- [ ] Use color picker — confirm dialog opens, new color is visible on button, but view does not update until "Process" is clicked.
- [ ] Confirm "Max edge" spinbox row in the Forearm section no longer has an inline "Process" button.

### Edge Cases

- [ ] Click **"Process"** or **"Process All"** with no cell loaded — confirm no crash (guard `if self._current_cell is None: return`).
- [ ] Switch mode (By Cluster ↔ By Session) — confirm `_visible_keys()` returns the correct set for the new mode before "Process All" is clicked.

---

## Documentation Plan

- [ ] No README or CLAUDE.md updates required — this is an internal GUI behaviour change.

---

## Rollback Plan

All changes are in a single file (`rf_cluster_gallery_viewer.py`). To revert:

1. `git checkout HEAD -- code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

No data migrations, no config changes, no other files touched.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A missed signal connection still auto-triggers rebuild | Low | Low | Search for remaining `_set_and_rebuild` references after rewiring and confirm count is 0 |
| `_on_process_all()` blocks the UI if many sessions have large meshes | Low | Medium | Delaunay builds are fast for typical forearm point counts; acceptable for now. Async build is out-of-scope. |
| `_pick_color()` colour change appears on button but not in view — user confusion | Low | Low | Expected UX: button colour updates immediately as visual confirmation; scene updates on Process |
