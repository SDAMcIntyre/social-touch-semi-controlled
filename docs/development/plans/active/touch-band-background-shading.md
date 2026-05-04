# Plan: Touch-Band Background Shading on 2D Panels

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-playback-explorer`
**Branch:** `feature/touch-playback-explorer`

---

## Overview

**What:** Add alternating semi-transparent red/green background bands per `single_touch_id` to all 2D time-series panels in the merging and postprocessing visualization pipeline.
**Why:** The NeuralDataPanel plots (Nerve_freq, contact_depth, contact_area) show continuous data with no visual indication of where individual touches start and end. Background shading makes touch segmentation immediately visible.
**How:** Use `matplotlib.axes.Axes.axvspan()` to draw low-alpha colored bands for each contiguous non-zero `single_touch_id` block, with a toggle checkbox to show/hide them.

## Problem Statement

The 2D time-series panels in the merging/postprocessing viewers display neural and kinematic signals as continuous lines. There is no visual cue indicating which portions of the signal correspond to which individual touch. The `single_touch_id` column exists in the merged CSV but is not surfaced in the visualization — the user must mentally cross-reference the data.

## Goals

### In Scope
1. Add alternating red/green semi-transparent background bands to `NeuralDataPanel` for each `single_touch_id`
2. Add a "Touch bands" toggle checkbox to show/hide the bands
3. Guard gracefully when `single_touch_id` column is absent from the merged CSV
4. Add the same capability to `TimeSeriesPanel` via a generic `set_touch_boundaries()` API

### Out of Scope
- Interactive touch selection (clicking a band to select a touch)
- Touch-ID labels or annotations on the bands
- Color customization beyond red/green alternating
- Modifying the analysis pipeline 2D plots (feature-space renderers, RF heatmaps)

## Success Criteria

- [ ] NeuralDataPanel displays alternating red/green bands for each `single_touch_id`
- [ ] Bands are visible against the dark `#0d0d1a` subplot background
- [ ] Data lines and cursor remain clearly readable over the bands
- [ ] "Touch bands" checkbox toggles band visibility without lag
- [ ] Works in all three viewers: NeuralKinectViewer, BeforeAfterStepViewer, PostprocessedSceneViewer
- [ ] No error when `single_touch_id` column is missing (bands simply don't appear)
- [ ] TimeSeriesPanel supports optional touch boundaries via `set_touch_boundaries()`

---

## Technical Design

### Approach

Add `axvspan()` calls during axes setup for each contiguous non-zero `single_touch_id` range. Span artists are persistent (created once, not per-frame). A checkbox controls visibility via `artist.set_visible()`. A shared utility function `extract_touch_boundaries()` computes the boundary list from the `single_touch_id` array.

### Existing Pattern — `adjust_chunks_viewer.py:185`

```python
color = 'red' if idx % 2 == 0 else 'green'
span = ax.axvspan(start, end, color=color, alpha=0.8, zorder=1)
```

This plan follows the same pattern with lower alpha (~0.12) since bands are background context, not primary data.

### `single_touch_id` data structure

- Column in merged CSV: values `0` (no touch / gap), `1`, `2`, `3`, ... (sequential touch IDs)
- Each non-zero value forms a contiguous block of rows = one touch
- Rows are at neural sampling rate (~1 kHz), so sample indices map directly to the NeuralDataPanel x-axis
- Generated in preprocessing stage 6 (`find_single_touches.py`), propagated through unification into merged CSV

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `axvspan` per touch | Simple, persistent, works with zoom | Many artists for long recordings | **Chosen** — 60-180 patches is fine for matplotlib |
| Fill-between with color array | Single artist per axis | Complex to set up, harder to toggle | Rejected |
| Custom background image | Fast rendering | Complex setup, poor zoom behavior | Rejected |

### Architecture Changes

No new modules. Two existing files are modified:

- `code/src/merging/gui/neural_kinect_scene_viewer.py` — Add `extract_touch_boundaries()` function and modify `NeuralDataPanel`
- `code/src/preprocessing/common/gui/time_series_panel.py` — Add optional touch-boundary support to `TimeSeriesPanel`

No changes needed for `BeforeAfterStepViewer` or `PostprocessedSceneViewer` — they import `NeuralDataPanel` and will pick up the new feature automatically.

---

## Implementation Plan

### Phase 1: NeuralDataPanel touch bands
**Goal:** Add background shading and toggle to NeuralDataPanel
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 1.1 — Add `extract_touch_boundaries(touch_ids: np.ndarray) -> List[Tuple[int, int, int]]` module-level function after `_parse_contact_points_cell` (~line 139). Extracts `(start_idx, end_idx, touch_id)` tuples for each contiguous non-zero block using `np.diff` on non-zero mask.
- [x] Task 1.2 — In `NeuralDataPanel.__init__`: after `_setup_axes(merged_df)` (line 360), extract boundaries from `merged_df['single_touch_id']` if column exists. Store as `self._touch_boundaries`.
- [x] Task 1.3 — Add "Touch bands" `QCheckBox` in the toolbar row (between `btn_layout.addStretch()` at line 338 and the `±` label at line 340). Connected to `_on_touch_bands_toggled`.
- [x] Task 1.4 — Add `_draw_touch_bands()` method: iterates `self._touch_boundaries`, calls `ax.axvspan(start, end, color=color, alpha=0.12, zorder=0)` on all three axes. Alternates `'#ff4444'` (odd IDs) / `'#44ff44'` (even IDs). Stores span artists in `self._touch_spans`.
- [x] Task 1.5 — Add `_on_touch_bands_toggled(checked: bool)` method: toggles `set_visible()` on all `self._touch_spans`, calls `canvas.draw_idle()`.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — Add `extract_touch_boundaries()`, modify `NeuralDataPanel.__init__`, add `_draw_touch_bands()`, add `_on_touch_bands_toggled()`, add checkbox to toolbar

**Dependencies:** None

### Phase 2: TimeSeriesPanel generic API
**Goal:** Add optional touch-boundary support to the generic panel

- [x] Task 2.1 — In `TimeSeriesPanel.__init__` (~line 89): initialize `self._touch_boundaries = []`, `self._touch_bands_visible = True`, `self._touch_spans = []`
- [x] Task 2.2 — Add "Touch bands" `QCheckBox` in the toolbar row (after the per-subplot visibility checkboxes, before `addStretch`). Hidden by default; shown only when boundaries are set.
- [x] Task 2.3 — In `_rebuild_axes()` (line 145): after the cursor-line loop but before `self.canvas.draw()`, draw touch-band spans if `self._touch_boundaries` is non-empty and `self._touch_bands_visible` is True. Boundaries are in seconds (matching the x-axis).
- [x] Task 2.4 — Add public method `set_touch_boundaries(boundaries: List[Tuple[float, float, int]])` that stores boundaries, shows the checkbox, and calls `_rebuild_axes()`.
- [x] Task 2.5 — Add `_on_touch_bands_toggled(checked: bool)` that sets `self._touch_bands_visible` and calls `_rebuild_axes()`.

**Files Modified:**
- `code/src/preprocessing/common/gui/time_series_panel.py` — Add state, checkbox, `set_touch_boundaries()`, modify `_rebuild_axes()`, add toggle handler

**Dependencies:** None (can be done in parallel with Phase 1)

---

## Testing Plan

### Manual Verification
- [ ] Launch NeuralKinectViewer (`merging_pipeline_neuron_to_kinect_visualisation.py`) with a merged CSV that contains `single_touch_id` — verify bands appear
- [ ] Toggle "Touch bands" checkbox off/on — verify bands disappear/reappear
- [ ] Zoom in/out with mouse wheel — verify bands clip correctly at zoom boundaries
- [ ] Launch with a merged CSV that does NOT have `single_touch_id` — verify no error, no bands, checkbox hidden or disabled
- [ ] Launch BeforeAfterStepViewer and PostprocessedSceneViewer — verify bands appear there too (they use same NeuralDataPanel)
- [ ] Verify data lines and red cursor remain clearly visible over the bands (alpha is appropriate)

### Edge Cases
- [ ] Merged CSV with all `single_touch_id = 0` (no touches) — no bands drawn
- [ ] Recording with only 1 touch (single contiguous block) — one band drawn
- [ ] Very long recording with 100+ touches — verify no performance degradation

---

## Documentation Plan

- [ ] No CLAUDE.md changes needed (no architectural change)
- [ ] No README changes needed (no new commands)

---

## Rollback Plan

Revert commits on the feature branch. No data changes, no config changes. Pure additive UI code.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Alpha too low/high against dark background | Med | Low | Tune alpha after visual test; 0.08-0.15 range |
| Older merged CSVs lack `single_touch_id` | Med | Low | Guard with `if col in df.columns`; hide checkbox |
| Many spans slow down matplotlib draw | Low | Low | Typical: 60-180 patches; well within capability |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~40 lines of code | None |
| Phase 2 | ~30 lines of code | None |

---

## References

- Existing axvspan pattern: `code/src/preprocessing/trial_segmentation/adjust_chunks_viewer.py:185`
- Touch segmentation source: `code/scripts/_3_preprocessing/_6_metadata_matching/find_single_touches.py`
- NeuralDataPanel: `code/src/merging/gui/neural_kinect_scene_viewer.py:296-434`
- TimeSeriesPanel: `code/src/preprocessing/common/gui/time_series_panel.py:36-244`
