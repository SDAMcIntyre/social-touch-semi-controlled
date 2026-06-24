# Plan: Touch Population Vertex Threshold Slider

**Date:** 2026-05-06
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-11 07:31
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements` (continuation)

---

## Overview

Add a vertex-level threshold slider to the Touch Population Explorer that filters
the 3D heatmap by per-vertex unique-touch overlap count, and add axis padding to
the 2D scatter plot so edge points are visible.

## Problem Statement

1. When many touches are displayed, low-overlap vertices (contacted by only 1-2
   touches) add noise to the heatmap, obscuring the spatial core of the receptive
   field. There is no way to filter by overlap depth.
2. The scatter plot axis limits are set to the 1st/99th percentile of visible
   data with no padding, so dots at the extremes are clipped against the axes.

## Goals

### In Scope
1. Horizontal slider in the right panel controlling a per-vertex unique-touch
   threshold (range 1..N, where N = currently filtered touch count)
2. Vertices below the threshold rendered in light grey (`[0.75, 0.75, 0.75]`),
   distinct from uncontacted dark grey (`[0.3, 0.3, 0.3]`)
3. Slider range updates dynamically when filters change; hides in single-touch mode
4. Scatter plot axis limits padded by +/-10% of the data range

### Out of Scope
- Per-vertex touch count display or labelling on the 3D mesh
- Changing the heatmap colormaps or the uncontacted vertex colour
- Modifications to single-touch mode rendering

## Success Criteria

- [ ] Slider appears below gesture checkboxes, shows `value / max`
- [ ] Dragging the slider right greys out weakly-overlapping vertices in real time
- [ ] Filter changes (rect drag, checkbox toggle) update the slider range
- [ ] Slider hidden in single-touch mode, restored in population mode
- [ ] Session switch resets the slider to 1
- [ ] Scatter dots at data extremes are visible with padding
- [ ] Uncontacted vertices remain dark grey throughout

---

## Technical Design

### Approach

**Threshold rendering** uses PyVista's `below_color` parameter. Vertices below the
threshold get a sentinel scalar value of `-1.0` (below `clim[0]=0.0`), causing
PyVista to render them with `below_color=[0.75, 0.75, 0.75]`. This requires no
manual RGB composition or lookup-table manipulation.

**Unique touch count** per vertex is computed by deduplicating `(vertex_idx,
touch_idx)` pairs using `np.unique` on a 1D-encoded key
(`vertex * max_touch + touch`), then `np.bincount` on the deduplicated vertex
indices. This is O(C log C) where C = contact point count.

**Scatter padding** adds a 10% margin to the existing 1st/99th percentile axis
limits in `_draw_scatter`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `below_color` sentinel | Simple, no extra actors, works with existing scalar pipeline | Adds one parameter to `add_mesh` | **Chosen** |
| Manual RGB vertex colours | Full control over colour per vertex | Breaks scalar bar, must compose colours manually | Rejected |
| Separate point cloud actor for below-threshold | Clear separation | Two actors to manage, camera sync issues | Rejected |
| 2D `np.unique(axis=0)` for touch count | Readable | Slow on large arrays due to row-wise sorting | Rejected in favour of 1D encoding |

### Architecture Changes

No new modules or classes. All changes are within `TouchPopulationExplorer` in
`touch_population_explorer.py`.

New methods:
- `_compute_unique_touch_count(cp_mask, n_verts)` — per-vertex unique touch count
- `_apply_vertex_threshold(heatmap_val, unique_touch_count)` — sentinel assignment
- `_update_threshold_slider_range(n_filtered_touches)` — dynamic range update
- `_on_threshold_changed(value)` — slider callback

---

## Implementation Plan

### Phase 1: Slider UI and threshold logic
**Goal:** Add the slider widget and the core computation methods.

**Tasks:**
- [x] Task 1.1 — Add `QSlider` to imports
- [x] Task 1.2 — Add `self._vertex_threshold = 1` in `__init__`
- [x] Task 1.3 — Build slider widgets in `_build_ui` right panel (label + horizontal
      slider + value label), between gesture checkboxes and stretch
- [x] Task 1.4 — Implement `_compute_unique_touch_count(cp_mask, n_verts)` using
      1D-encoded `np.unique` + `np.bincount`
- [x] Task 1.5 — Implement `_apply_vertex_threshold(heatmap_val, unique_touch_count)`
      setting below-threshold contacted vertices to `-1.0`
- [x] Task 1.6 — Implement `_update_threshold_slider_range(n_filtered_touches)` with
      `blockSignals` guard and clamping
- [x] Task 1.7 — Implement `_on_threshold_changed(value)` connecting to
      `_apply_filter_update`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` —
  imports, `__init__`, `_build_ui`, four new methods

**Dependencies:** None

### Phase 2: Integrate threshold into rendering pipeline
**Goal:** Wire the threshold into `_render_3d` and `_apply_filter_update`.

**Tasks:**
- [x] Task 2.1 — Add `below_color=[0.75, 0.75, 0.75]` to `add_mesh` in `_render_3d`
- [x] Task 2.2 — In `_render_3d`, compute unique touch count and apply threshold
      after heatmap computation (skip when `self._single_touch_mode`)
- [x] Task 2.3 — In `_render_3d`, call `_update_threshold_slider_range(n_total)`
      after rendering
- [x] Task 2.4 — Restructure `_apply_filter_update`: extract `active_touch_indices`
      and `cp_mask` before the RF/standard if-else, then compute unique touch count,
      update slider range, and apply threshold after getting `heatmap_val`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` —
  `_render_3d`, `_apply_filter_update`

**Dependencies:** Phase 1

### Phase 3: Mode integration and session handling
**Goal:** Handle single-touch mode toggle and session switches.

**Tasks:**
- [x] Task 3.1 — In `_set_single_touch_mode`: hide threshold widgets when enabled,
      show when disabled
- [x] Task 3.2 — In `_load_session`: rebuild threshold slider in the right panel
      teardown/rebuild block, reset `self._vertex_threshold = 1`. Handle sub-layout
      cleanup for the threshold row `QHBoxLayout`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` —
  `_set_single_touch_mode`, `_load_session`

**Dependencies:** Phase 2

### Phase 4: Scatter plot axis padding
**Goal:** Add 10% margin to scatter axis limits.

**Tasks:**
- [x] Task 4.1 — In `_draw_scatter`, after computing `x_lo/x_hi/y_lo/y_hi` from
      percentiles (lines 330-337), compute `margin_x = (x_hi - x_lo) * 0.1` and
      `margin_y = (y_hi - y_lo) * 0.1`, then set limits to
      `(x_lo - margin_x, x_hi + margin_x)` and `(y_lo - margin_y, y_hi + margin_y)`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` —
  `_draw_scatter`

**Dependencies:** None (independent of phases 1-3)

---

## Testing Plan

### Manual Verification
- [ ] Launch Touch Population Explorer with a multi-touch session
- [ ] Slider appears in right panel below gesture checkboxes, shows "1 / N"
- [ ] Drag slider right: low-overlap vertices turn light grey, high-overlap stay coloured
- [ ] Drag slider to max: only the most-overlapped vertices remain coloured
- [ ] Drag slider back to 1: all contacted vertices show heatmap colours
- [ ] Drag the filter rectangle: slider max updates to match filtered count
- [ ] Uncheck a gesture checkbox: slider max updates
- [ ] Switch heatmap mode (spike density / IFF): threshold persists
- [ ] Switch to RF heatmap mode (if available): threshold works
- [ ] Enter single-touch mode: slider hides
- [ ] Exit single-touch mode: slider reappears with correct range
- [ ] Switch sessions: slider resets to 1
- [ ] Uncontacted vertices remain dark grey [0.3, 0.3, 0.3] at all times
- [ ] Scatter plot: dots at data extremes are visible, not clipped against axes

### Edge Cases
- [ ] Session with only 1 touch: slider range is 1/1, effectively no threshold
- [ ] All checkboxes unchecked: slider disabled, 3D view shows no heatmap data
- [ ] Filter rectangle selecting 0 touches: slider disabled
- [ ] Session with no Stage 3 features: slider still works (scatter is empty,
      but 3D heatmap with threshold renders correctly)

---

## Documentation Plan

- [ ] No external documentation needed (internal GUI enhancement)

---

## Rollback Plan

All changes are in a single file on an existing feature branch. Revert with:
```
git revert <commit-sha>
```

No data migrations or breaking API changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `below_color` not persisting after `scalar_range` update | Low | Med | VTK LUT preserves below-range colour on range update; verified in PyVista docs |
| Slider `valueChanged` recursion via `_update_threshold_slider_range` | Med | Low | `blockSignals(True/False)` guard (KB: note-qt-itemchanged-signal-recursion) |
| `np.unique` slow on very large contact arrays | Low | Low | 1D encoding avoids 2D row-sort; fallback to `pandas.factorize` if needed |
| Right-panel sub-layout cleanup in `_load_session` | Med | Low | Recursively remove child widgets from layout items before `takeAt` |
