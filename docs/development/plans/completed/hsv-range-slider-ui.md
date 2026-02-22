# Plan: HSV Range Slider UI for Skin-Colour Filter

**Date:** 2026-02-20
**Author:** Claude (AI-assisted)
**Status:** Completed
**Branch:** `feature/hsv-range-slider-ui`

---

## Overview

**What:** Replace the six independent HSV bound sliders in `ArmSegmentation`'s interactive GUI with three unified range controls — a circular arc selector for the cyclic Hue channel and paired range sliders for the linear Saturation and Value channels.
**Why:** The current design exposes `hsv_lower_bound` and `hsv_upper_bound` as separate flat values, making it impossible to express wrap-around hue ranges (e.g. reds spanning 330°–30°) and forcing the user to mentally track six interdependent values instead of three ranges.
**How:** Refactor the parameter model to `hsv_h_range / hsv_s_range / hsv_v_range`; fix the filter to handle cyclic hue; build a custom Open3D-Canvas-based `HueRangeCircle` widget for H and coupled-slider `RangeSliderRow` for S and V; wire everything into the existing interactive loop.

---

## Problem Statement

`_apply_skin_color_filter` (line 236) compares hue with a strict `lb[0] <= H <= ub[0]` inequality.
This silently fails for any skin population near the red/pink end of the wheel (H ≈ 0° / 360°): the user would need `lb_H = 330, ub_H = 30`, but `330 <= H <= 30` is always false.

The GUI exposes this as **six individual sliders** (lower H, lower S, lower V, upper H, upper S, upper V).
The user must keep the three lower values below the three upper values and has no visual feedback that the hue selection is cyclic.

Concretely:
- No wraparound hue range is expressible.
- The coupling between lower and upper sliders is invisible and easy to violate.
- The flat layout gives no spatial intuition of the colour wheel.

---

## Goals

### In Scope
1. Refactor `_DEFAULT_PARAMS['color_skin_filter']` from `hsv_lower_bound` / `hsv_upper_bound` to `hsv_h_range`, `hsv_s_range`, `hsv_v_range`.
2. Fix `_apply_skin_color_filter` to correctly handle wrap-around hue ranges.
3. Implement a `RangeSliderRow` helper (two coupled sliders, low ≤ high enforced) for S and V.
4. Implement a `HueRangeCircle` widget (custom Open3D Canvas drawing) that:
   - Renders a full hue colour wheel.
   - Exposes two draggable handles (start angle, end angle) defining a clockwise arc.
   - Visually highlights the selected arc over the wheel.
   - Allows any combination including wraparound (e.g., 330°–30°).
5. Replace the six-slider layout in `_display_pointcloud` with the three new controls.
6. Update the HSV hover readout to remain consistent with new parameter names.

### Out of Scope
- GPU-accelerated colour filtering.
- Changing any segmentation step other than `color_skin_filter`.
- Saving/loading HSV presets to disk.
- Changes to `extract_arm` or the DBSCAN step.
- Unit-test infrastructure setup (no test runner configured for this module yet).

---

## Success Criteria

- [ ] `ArmSegmentation(params={'color_skin_filter': {'hsv_h_range': [330, 30]}})` selects red hues (wrap-around) without raising an exception and returns a non-empty point cloud from skin-coloured input.
- [ ] `ArmSegmentation(params={'color_skin_filter': {'hsv_h_range': [0, 25]}})` behaves identically to the old `hsv_lower_bound=[0,0,0], hsv_upper_bound=[25,1,1]` for the same S/V ranges.
- [ ] The interactive GUI shows exactly three controls under `color_skin_filter`: one `HueRangeCircle` and two `RangeSliderRow` rows (S, V).
- [ ] Dragging the H start handle past the H end handle (or vice versa) produces a wrap-around selection, not an empty selection.
- [ ] S and V range sliders enforce `low ≤ high`; setting `low > high` clamps the other handle automatically.
- [ ] The HSV hover readout continues to display correctly after the widget refactor.
- [ ] The `__main__` demo block runs without errors with the new parameter keys.
- [ ] No regression in the box-filter, downsampling, or DBSCAN interactive steps.

---

## Technical Design

### Approach

**Parameter model** is refactored from two 3-element lists to three 2-element lists:

```python
# OLD
'hsv_lower_bound': [0,   0.0, 0.0]   # [H_lo, S_lo, V_lo]
'hsv_upper_bound': [25,  1.0, 1.0]   # [H_hi, S_hi, V_hi]

# NEW
'hsv_h_range': [0,   25 ]            # [H_start, H_end] degrees, cyclic
'hsv_s_range': [0.0, 1.0]            # [S_low,   S_high]
'hsv_v_range': [0.0, 1.0]            # [V_low,   V_high]
```

**Cyclic hue filter** replaces the scalar comparison:

```python
def _hue_in_range(H, h_start, h_end):
    if h_start <= h_end:
        return (H >= h_start) & (H <= h_end)
    else:                               # wrap-around
        return (H >= h_start) | (H <= h_end)
```

**`RangeSliderRow`** — a helper that creates a horizontal row with two `gui.Slider` widgets
labelled *lo* and *hi*, wired so that moving *lo* above *hi* clamps *hi* upward and vice versa.
Returns a `get_range()` closure → `(lo, hi)`.

**`HueRangeCircle`** — a widget drawn onto a `gui.ImageWidget` (updated each frame by
re-rendering a NumPy RGBA image):
- A full hue wheel rendered once as a polar bitmap.
- Two small circular handles at `h_start` and `h_end` angles.
- The shorter or longer arc between the handles is highlighted (user toggles which arc is
  selected via a checkbox or by drag direction).
- Mouse-down/drag events update the nearest handle angle.
- On change, triggers an immediate `on_process()` call (same pattern as other widgets).

Open3D's `gui.ImageWidget` accepts a `o3d.geometry.Image` and can be refreshed by
assigning a new image. Mouse events are captured via `set_on_mouse()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Custom Canvas via `gui.ImageWidget`** | Full control, no extra deps, stays inside Open3D | Must implement hit-testing and rendering manually | **Chosen** |
| Native `gui.Slider` pair (just clamped) | Simple, already proven | Cannot express cyclic H; still 6 values | Rejected for H; used for S/V |
| Dear ImGui / custom Open3D widget subclass | Native look | Requires C++ extension, not feasible in pure Python | Rejected |
| Matplotlib colour picker popup | Rich widgets | Requires separate window, breaks single-window UX | Rejected |

### Architecture Changes

**Files modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — all changes are contained here.

**New internal helpers added inside `ArmSegmentation`:**

```
ArmSegmentation
├── _hue_in_range(H, h_start, h_end)        # static, phase 1
├── _make_range_slider_row(label, cfg, lo, hi)  # phase 2
└── _make_hue_range_circle(h_start, h_end)  # phase 3
    ├── _render_hue_wheel_image(size)
    ├── _render_hue_arc_overlay(img, h_start, h_end)
    └── _handle_circle_mouse(event, ...)
```

No new files or external dependencies are introduced.
`_SLIDER_CONFIGS['color_skin_filter']` is updated to describe range pairs instead of per-bound lists.

---

## Implementation Plan

### Phase 1: Data model and filter logic refactor
**Branch:** `feature/hsv-range-slider-ui/phase-1-data-model`
**Goal:** Replace the old `hsv_lower_bound` / `hsv_upper_bound` parameters with `hsv_h_range`, `hsv_s_range`, `hsv_v_range` everywhere, and implement correct cyclic hue filtering. No GUI changes yet — the existing slider scaffolding is adapted to the new keys as a temporary fallback.

**Tasks:**
- [ ] 1.1 — Update `_DEFAULT_PARAMS['color_skin_filter']`: remove `hsv_lower_bound` / `hsv_upper_bound`; add `hsv_h_range: [0, 25]`, `hsv_s_range: [0.0, 1.0]`, `hsv_v_range: [0.0, 1.0]`.
- [ ] 1.2 — Update `_SLIDER_CONFIGS['color_skin_filter']` to reflect the new keys (each is a list of 2 values with lo/hi configs).
- [ ] 1.3 — Add static method `_hue_in_range(H, h_start, h_end)` with cyclic logic.
- [ ] 1.4 — Rewrite `_apply_skin_color_filter` to read `hsv_h_range`, `hsv_s_range`, `hsv_v_range` and call `_hue_in_range`.
- [ ] 1.5 — Update the `__main__` demo block to use the new parameter keys.
- [ ] 1.6 — Manual smoke test: run `__main__` in both interactive and non-interactive mode; confirm skin-coloured cluster is still isolated.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — `_DEFAULT_PARAMS`, `_SLIDER_CONFIGS`, `_apply_skin_color_filter`, `__main__`.

**Dependencies:** None

**Commit & merge procedure:**
```
# On phase branch
git add code/src/preprocessing/forearm_extraction/arm_segmentation.py
git commit -m "refactor(forearm-extraction): replace HSV bound pairs with h/s/v range params"
# Open PR phase-1-data-model → feature/hsv-range-slider-ui
```

---

### Phase 2: Linear range slider for S and V
**Branch:** `feature/hsv-range-slider-ui/phase-2-sv-range-slider`
**Goal:** Replace the two separate sliders per channel (lo, hi) with a single `RangeSliderRow` that exposes a unified range and enforces `lo ≤ hi` automatically.

**Tasks:**
- [ ] 2.1 — Implement `_make_range_slider_row(label, lo_cfg, hi_cfg, lo_init, hi_init)` returning `(widget_row, get_range_fn)`. The row contains: `[label] [lo_slider] [lo_textbox] — [hi_slider] [hi_textbox]`. Moving lo above hi clamps hi to lo; moving hi below lo clamps lo to hi.
- [ ] 2.2 — Update the widget-building loop in `_display_pointcloud` to detect 2-element range lists (`hsv_s_range`, `hsv_v_range`) and call `_make_range_slider_row` instead of building two separate `_make_paired_row` rows.
- [ ] 2.3 — Update `on_process` to read the new `get_range_fn` return values and write back into `self.params['color_skin_filter']`.
- [ ] 2.4 — Manual smoke test: S and V range sliders appear in GUI, clamping works, "Process" applies correct filter.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — `_make_range_slider_row`, `_display_pointcloud`, `on_process` inner closure.

**Dependencies:** Phase 1

**Commit & merge procedure:**
```
git add code/src/preprocessing/forearm_extraction/arm_segmentation.py
git commit -m "feat(forearm-extraction): add RangeSliderRow for S and V HSV channels"
# Open PR phase-2-sv-range-slider → feature/hsv-range-slider-ui
```

---

### Phase 3: Circular hue range selector
**Branch:** `feature/hsv-range-slider-ui/phase-3-hue-circle`
**Goal:** Replace the H range linear sliders with a circular arc widget rendered via a NumPy bitmap and displayed in a `gui.ImageWidget`. The user drags two handles around the wheel to define a clockwise arc; the arc may wrap around 0°/360°.

**Tasks:**
- [ ] 3.1 — Implement `_render_hue_wheel(size_px) -> np.ndarray` (RGBA uint8 image): draw concentric rings coloured by hue at S=1, V=1; inner grey disc as background.
- [ ] 3.2 — Implement `_render_hue_arc_overlay(base_img, h_start, h_end, handle_radius) -> np.ndarray`: draw a semi-transparent coloured arc on the ring, plus two circular drag handles at each angle. The highlighted arc is always the clockwise arc from `h_start` to `h_end`.
- [ ] 3.3 — Implement `_make_hue_range_circle(h_start_init, h_end_init)` returning `(image_widget, get_hue_range_fn)`:
  - Creates a `gui.ImageWidget`.
  - Initialises the bitmap with the above render helpers.
  - Registers a `set_on_mouse()` handler that: detects which handle is nearest the click, tracks drag, updates `h_start`/`h_end`, re-renders the bitmap, and refreshes the `ImageWidget`.
  - Returns a `get_hue_range_fn()` → `(h_start, h_end)`.
- [ ] 3.4 — Update the widget-building loop in `_display_pointcloud` to call `_make_hue_range_circle` for `hsv_h_range` instead of the range slider row.
- [ ] 3.5 — Add two companion `NumberEdit` boxes next to the wheel showing the current start/end degrees (read-only or editable as direct input fallback).
- [ ] 3.6 — Manual smoke test: drag both handles, verify highlighted arc updates, verify wrap-around range selects correct hue population in the point cloud.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — `_render_hue_wheel`, `_render_hue_arc_overlay`, `_make_hue_range_circle`, `_display_pointcloud`.

**Dependencies:** Phase 2

**Commit & merge procedure:**
```
git add code/src/preprocessing/forearm_extraction/arm_segmentation.py
git commit -m "feat(forearm-extraction): add HueRangeCircle widget for cyclic H selection"
# Open PR phase-3-hue-circle → feature/hsv-range-slider-ui
```

---

### Phase 4: Integration, hover readout, documentation
**Branch:** `feature/hsv-range-slider-ui/phase-4-integration`
**Goal:** Final wiring, edge-case hardening, and documentation update. Merge the feature branch into `dev`.

**Tasks:**
- [ ] 4.1 — Verify `on_process` correctly reads all three new widget types (`HueRangeCircle` + 2× `RangeSliderRow`) and writes back into `self.params['color_skin_filter']`.
- [ ] 4.2 — Update HSV hover readout label: display hue as `H: 210.3° (in range ✓)` using `_hue_in_range` for the live check.
- [ ] 4.3 — Add docstrings to all new helpers (`_hue_in_range`, `_make_range_slider_row`, `_render_hue_wheel`, `_render_hue_arc_overlay`, `_make_hue_range_circle`).
- [ ] 4.4 — Update the module-level class docstring to mention the cyclic hue range.
- [ ] 4.5 — Update `docs/development/plans/active/hsv-range-slider-ui.md` status → `Completed`; move to `docs/development/plans/completed/`.
- [ ] 4.6 — Full end-to-end manual test with a real Kinect recording.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — `on_process`, `on_hover`, docstrings.
- `docs/development/plans/active/hsv-range-slider-ui.md` → `docs/development/plans/completed/hsv-range-slider-ui.md`.

**Dependencies:** Phase 3

**Commit & merge procedure:**
```
git add code/src/preprocessing/forearm_extraction/arm_segmentation.py
git commit -m "feat(forearm-extraction): integrate HSV range widgets and update hover readout"

git add docs/development/plans/completed/hsv-range-slider-ui.md
git commit -m "docs(plans): mark hsv-range-slider-ui plan as completed"

# Open PR feature/hsv-range-slider-ui → dev
```

---

## Testing Plan

### Unit Tests
*(No automated test runner is configured for this module; tests are manual for now.)*

- [ ] `_hue_in_range(np.array([10, 350, 180]), 330, 30)` → `[True, True, False]` (wrap-around)
- [ ] `_hue_in_range(np.array([10, 350, 180]), 0, 25)` → `[True, False, False]` (normal range)
- [ ] `_hue_in_range(np.array([0, 360]), 0, 360)` → `[True, True]` (full wheel)
- [ ] `_apply_skin_color_filter` with `hsv_h_range=[330, 30]` on a point cloud containing only H=355 points returns a non-empty result.
- [ ] `_apply_skin_color_filter` with `hsv_h_range=[330, 30]` on a point cloud containing only H=180 points returns an empty result.

### Integration Tests
- [ ] Running `preprocess()` → `extract_arm()` pipeline with default params on the `__main__` dummy cloud isolates the skin-coloured cluster (same outcome as before the refactor).
- [ ] Setting `hsv_h_range=[330, 30]` in `__main__` with appropriately coloured dummy points produces a non-empty arm extraction.

### Manual Verification
- [ ] Launch `__main__` in interactive mode; confirm three controls appear under *Skin Color Filter*: one circular H widget and two range slider rows (S, V).
- [ ] Drag H start handle past H end handle; confirm arc wraps and point cloud updates correctly on "Process".
- [ ] Set S low > S high via direct NumberEdit; confirm the other handle is clamped automatically.
- [ ] Hover over a skin-toned point in the cloud; confirm HSV readout shows plausible values with in-range indicator.
- [ ] Press Space bar to trigger processing; confirm shortcut still works with the new widgets in focus.
- [ ] Resize the panel; confirm the hue circle scales correctly within the 1/5 panel width.

### Edge Cases
- [ ] `hsv_h_range = [0, 0]` (degenerate single-value range) — should select only H=0 points.
- [ ] `hsv_h_range = [180, 180]` (same) — no crash.
- [ ] `hsv_h_range = [0, 360]` — full wheel, all hues pass.
- [ ] Point cloud with zero points entering `_apply_skin_color_filter` — returns empty PointCloud without crash (already guarded, verify still works).
- [ ] Interactive window resized very small — hue circle degrades gracefully (minimum size clamp).

---

## Documentation Plan

- [ ] Update inline class docstring in `arm_segmentation.py` to describe the cyclic hue range.
- [ ] Add docstrings to all new private methods.
- [ ] Move plan from `docs/development/plans/active/` to `docs/development/plans/completed/` on completion.
- [ ] No `CLAUDE.md` update required (no new CuPy usage introduced).

---

## Rollback Plan

All changes are confined to a single file: `arm_segmentation.py`.

1. **Before any phase merge:** the feature branch is independent of `dev`; simply abandon the branch.
2. **After a phase is merged into the feature branch but before merging to `dev`:** revert the feature branch to the previous phase tag:
   ```bash
   git revert <phase-commit-sha>
   ```
3. **After merging to `dev`:** revert the merge commit:
   ```bash
   git revert -m 1 <merge-commit-sha>
   ```
   No database migrations, no file-format changes (params are runtime-only), no downstream breakage — this is safe to revert at any time.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gui.ImageWidget` mouse events not firing reliably in Open3D on Windows/WSL | Med | High | Test Phase 3 drag interaction on target platform early; fallback to two `NumberEdit` boxes for manual angle entry if drag fails |
| Rendering the hue wheel bitmap each frame causes perceptible lag | Low | Med | Cache the base wheel bitmap; only re-render the arc overlay on handle move |
| Existing callers passing `hsv_lower_bound`/`hsv_upper_bound` in their `params` dict at construction time silently get ignored after Phase 1 | Med | Med | Add a `__init__` migration warning: detect old keys and raise `DeprecationWarning` with instructions |
| `deep_update` with the new 2-element list keys merges incorrectly (list vs. dict) | Low | Med | Lists are already overwritten (not deep-merged) by `deep_update`; verify and add a test case in Phase 1 |
| Panel too narrow to display hue circle meaningfully (1/5 of small window) | Low | Low | Add a minimum canvas size and a fallback text label if width < threshold |

---

## Timeline

| Phase | Estimated Effort | Branch | Dependencies |
|-------|-----------------|--------|--------------|
| Phase 1: Data model & filter logic | ~1 h | `phase-1-data-model` | None |
| Phase 2: S/V range sliders | ~1.5 h | `phase-2-sv-range-slider` | Phase 1 |
| Phase 3: Hue range circle | ~3–4 h | `phase-3-hue-circle` | Phase 2 |
| Phase 4: Integration & docs | ~1 h | `phase-4-integration` | Phase 3 |

---

## References

- Source file: `code/src/preprocessing/forearm_extraction/arm_segmentation.py`
- Related plan: `docs/development/plans/completed/forearm-frame-averaging.md`
- Open3D GUI docs: `gui.ImageWidget`, `gui.Slider`, `set_on_mouse`
- Bug context: `docs/development/knowledge-base/bug-cupy-bool8-import-order.md` (CuPy import order constraint — not affected by this change)
