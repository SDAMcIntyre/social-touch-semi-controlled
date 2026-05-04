# Plan: Linear Interpolation and Viewer Debug Signals

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 21:38
**Base Branch:** `dev`
**Branch:** `feature/linear-interpolation-and-viewer-signals`

---

## Overview

Switch all Kinect-sampled column interpolation from adaptive cubic to always-linear in the preparation stage, and add `trial_id`/`single_touch_id` debug time-series subplots to the TouchPreparationViewer. Cubic splines overshoot near contact boundaries, producing artificially inflated peaks between 30 Hz samples; linear eliminates this while remaining accurate at Kinect sample rate.

## Problem Statement

The preparation pipeline upsamples 30 Hz Kinect columns to 1 kHz by interpolating NaN gaps. The current cubic spline method overshoots near zero-crossings for `contact_depth` and `contact_area`, and introduces unnecessary oscillation for sticker positions. The post-interpolation clamp catches negative values but cannot correct positive overshoot. Additionally, the TouchPreparationViewer lacks visibility into grouping metadata (`trial_id`, `single_touch_id`), making it harder to verify that forward-fill grouping is correct.

## Goals

### In Scope
1. Switch all `SCALAR_COLUMNS` and `SPATIAL_COLUMNS` to always-linear interpolation
2. Remove the now-dead cubic adaptive degradation logic and `method` parameter
3. Remove the unused `interpolation.method` DAG config key
4. Add `trial_id` and `single_touch_id` as time-series subplots in TouchPreparationViewer

### Out of Scope
- Changing interpolation for binary/categorical columns (already forward-filled)
- Changing the post-interpolation clamp logic (still needed with linear)
- Adding new signal columns beyond trial_id/single_touch_id

## Success Criteria

- [x] All Kinect-sampled columns use linear interpolation (no cubic path remains)
- [x] `method` parameter removed from `interpolate_touch_columns` and `_interpolate_group`
- [x] DAG config `interpolation.method` key removed
- [x] All 3 callers updated (no broken calls)
- [x] TouchPreparationViewer shows trial_id and single_touch_id subplots
- [ ] Existing prepared CSVs regenerate cleanly with `force_processing=True`

---

## Technical Design

### Approach

Merge the scalar and spatial interpolation loops into a single always-linear loop. Since all interpolated columns now use identical logic, the two category constants (`SCALAR_COLUMNS`, `SPATIAL_COLUMNS`) collapse into one `INTERPOLATED_COLUMNS` list. The `method` parameter becomes dead code and is removed from both internal and public API. For the viewer, `trial_id` and `single_touch_id` are added to the signal column lists — they're already present in the DataFrame (required columns) so no data-loading changes are needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Merge into single `INTERPOLATED_COLUMNS` | Simplest; one loop, no dead code | Loses the semantic distinction between scalar and spatial | Chosen — distinction is no longer meaningful when all use linear |
| Keep separate lists, apply same linear logic | Preserves category naming | Two loops doing identical work; `method` param still dangling | Rejected — unnecessary complexity |
| New `LINEAR_SCALAR_COLUMNS` for depth/area only | Smallest diff | Sticker positions still use cubic (user wants all linear) | Rejected — incomplete |

### Column Treatment Summary

| Column(s) | Current | Proposed | Rationale |
|---|---|---|---|
| `contact_depth` | Cubic (≥4) / linear (2-3) / ffill (1) | **Always linear** | Overshoot near zero-crossings |
| `contact_area` | Cubic (≥4) / linear (2-3) / ffill (1) | **Always linear** | Bounded physical quantity |
| `sticker_*_position_x/y/z` (9 cols) | Cubic (≥4) / linear (2-3) / ffill (1) | **Always linear** | Consistent; cubic unnecessary at 30 Hz |
| `contact_location_x/y/z` | Always linear | Unchanged (linear) | Already linear |
| `Nerve_spike`, `Nerve_freq`, `Nerve_TTL`, `time` | No interpolation (already 1 kHz) | Unchanged | Native 1 kHz |
| `led_on`, `contact_detected`, `contact_points` | Forward-fill | Unchanged | Binary/categorical |
| `trial_id`, `single_touch_id`, `*_metadata` (6 cols) | Forward-fill (pre-grouping) | Unchanged | Group labels |

---

## Implementation Plan

### Phase 1: Simplify interpolation to always-linear
**Goal:** Remove cubic path, unify interpolation loop, clean up dead code
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 1.1 — Replace `SCALAR_COLUMNS` and `SPATIAL_COLUMNS` with single `INTERPOLATED_COLUMNS` list (all 14 columns)
- [x] Task 1.2 — Simplify `_interpolate_group()`: remove `method` parameter, replace two loops with one always-linear loop
- [x] Task 1.3 — Remove `method` parameter from `interpolate_touch_columns()` public API
- [x] Task 1.4 — Update callers to remove `method` argument
- [x] Task 1.5 — Remove `interpolation.method` key from DAG config
- [x] Task 1.6 — Keep post-interpolation clamp unchanged (still guards linear undershoot)

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/interpolation.py` — merge column lists, simplify loops, remove `method` param
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — remove `method=interp_method` from call
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — remove `method` argument from call
- `code/src/analysis/touch_analytics/series_pipeline.py` — remove `method` argument from call
- `configs/analyse_workflow_dag.yaml` — remove `interpolation.method` config key

**Dependencies:** None

### Phase 2: Add trial_id and single_touch_id to viewer
**Goal:** Add debug time-series subplots for grouping verification
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Task 2.1 — Add `'trial_id'` and `'single_touch_id'` to `SIGNAL_COLUMNS` in `preparation_viewer_data.py`
- [x] Task 2.2 — Add `'trial_id'` and `'single_touch_id'` to `_DEFAULT_SIGNALS` in `touch_preparation_viewer.py`
- [x] Task 2.3 — Add labels to `_SIGNAL_LABELS`: `'trial_id': 'Trial'`, `'single_touch_id': 'Touch'`

**Files Modified:**
- `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py` — add to `SIGNAL_COLUMNS`
- `code/src/analysis/touch_analytics/gui/touch_preparation_viewer.py` — add to `_DEFAULT_SIGNALS` and `_SIGNAL_LABELS`

**Dependencies:** None (independent of Phase 1)

---

## Testing Plan

### Manual Verification
- [ ] Run preparation on a session with `force_processing=True` — confirm no errors
- [ ] Open TouchPreparationViewer — verify depth/area curves show clean linear interpolation (no overshoot spikes)
- [ ] Verify sticker position curves are linear (no cubic oscillation)
- [ ] Verify trial_id and single_touch_id subplots show constant values within each touch group
- [ ] Verify no negative values in depth/area columns (clamp still active)

### Edge Cases
- [ ] Touch group with < 4 valid Kinect samples — should still interpolate linearly (previously fell back from cubic to linear; now always linear)
- [ ] Touch group with exactly 1 valid sample — should ffill/bfill (unchanged behavior)
- [ ] Touch group with 0 valid samples — should remain NaN (unchanged behavior)

---

## Documentation Plan

- [ ] Update module docstring in `interpolation.py` to reflect linear-only approach
- [ ] Update `code/src/analysis/CLAUDE.md` stage 1 description (remove "cubic by default")

---

## Rollback Plan

1. `git revert` the feature merge commit — restores cubic interpolation and old viewer
2. No data migration needed — re-run preparation with `force_processing=True` to regenerate CSVs with either method
3. No breaking changes to downstream stages — they consume the same column names regardless of interpolation method

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Slightly less smooth kinematics from linear vs cubic | Low | Low | 30 Hz sample rate is coarse; cubic smoothness was artificial anyway. Downstream feature extraction aggregates (mean, std, etc.) are robust to this |
| Callers outside the 3 known files pass `method` | Low | Low | Grep for `interpolate_touch_columns` to confirm all call sites before removing parameter |
| Existing cached prepared CSVs not regenerated | Med | Med | Document that `force_processing=True` is needed; old CSVs still work structurally |
