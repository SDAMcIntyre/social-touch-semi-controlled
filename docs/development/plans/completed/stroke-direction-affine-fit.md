# Plan: Stroke Direction Classification via Affine Fit

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 21:25
**Base Branch:** `dev`
**Branch:** `feature/stroke-direction-affine-fit`
**Started:** 2026-05-04
**Phase 1 Completed:** 2026-05-04

---

## Overview

Replace the endpoint-delta stroke direction classifier with a linear regression
(affine fit) on `contact_location_x` over frame index. The slope sign determines
proximal vs distal. This uses all valid data points instead of just two endpoints,
making the classification more robust to noise at touch boundaries.

## Problem Statement

`classify_gesture_type()` in `preparation/gesture_type.py` currently compares the
first and last non-NaN `contact_location_x` values to decide proximal vs distal.
This two-point delta is sensitive to noise: a single noisy boundary sample can
flip the classification. The diagnostic script (`diagnose_stroke_direction.py`)
already flags sessions where the median Δx is near zero, indicating noise-driven
classification.

A linear fit across all valid samples averages out per-frame noise and yields a
slope whose sign is a more reliable direction indicator.

## Goals

### In Scope
1. Replace the endpoint-delta logic with `np.polyfit(index, x, 1)` inside
   `classify_gesture_type()`
2. Use the slope sign: positive → `stroke_proximal`, negative/zero → `stroke_distal`
3. Preserve the `stroke_unknown` sentinel when all `contact_location_x` values are NaN
4. Handle edge case: single valid sample (slope undefined) → `stroke_unknown`
5. Update unit tests to cover the new regression-based logic

### Out of Scope
- Changing the `assign_gesture_type()` wrapper or its call-sites
- Adding new return values or changing the function signature
- Modifying the interpolation stage
- Addressing PCA sign-flip issues (orthogonal to this change)

## Success Criteria

- [x] `classify_gesture_type()` uses linear regression slope instead of endpoint delta
- [x] All existing test scenarios still pass (after adapting expected values where needed)
- [x] New tests cover: multi-point noisy stroke where endpoints disagree with trend,
      single-point stroke, two-point stroke (degenerate fit)
- [ ] Diagnostic script (`diagnose_stroke_direction.py`) shows no regression in
      classification balance on real data

---

## Technical Design

### Approach

Inside `classify_gesture_type()`, for stroke groups:

```python
valid_x = group['contact_location_x'].dropna()
if valid_x.empty:
    return 'stroke_unknown'
if len(valid_x) < 2:
    return 'stroke_unknown'
slope, _ = np.polyfit(np.arange(len(valid_x)), valid_x.values, 1)
return 'stroke_proximal' if slope > 0 else 'stroke_distal'
```

Key decisions:
- **Regress on sequential index** (`np.arange`), not on timestamps — the data is
  already uniformly sampled at 1 kHz after interpolation, so frame index *is* time.
- **Threshold at zero** — consistent with the current delta approach. A near-zero
  slope is ambiguous by nature; no benefit to adding an epsilon dead-zone.
- **`stroke_unknown` for < 2 valid points** — a single point cannot define a slope.
  The current code would return `stroke_distal` (Δx = 0) for a single point; the
  new behaviour is more honest.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Endpoint delta (current) | Simple, fast | Noise-sensitive, uses only 2 points | Replace |
| `np.polyfit` degree 1 | Uses all points, robust to boundary noise, minimal code change | Adds numpy dependency (already present) | **Chosen** |
| Robust regression (Theil-Sen) | Outlier-resistant | Overkill for monotonic strokes; adds scipy dependency | Rejected |
| Weighted linear fit (weight interior higher) | Extra noise suppression | Complexity not justified by data; boundary NaNs already dropped | Rejected |

### Architecture Changes

None. The change is confined to the body of one function. No new modules, no new
interfaces, no change to function signatures or return values.

---

## Implementation Plan

### Phase 1: Replace classification logic and update tests
**Goal:** Swap endpoint-delta for affine fit; ensure test coverage.

**Tasks:**
- [x] Task 1.1 — Add `import numpy as np` to `gesture_type.py`
- [x] Task 1.2 — Replace the endpoint-delta block (lines 58-72) with the
      `np.polyfit` regression block described above
- [x] Task 1.3 — Update `stroke_unknown` condition: return `stroke_unknown`
      when `len(valid_x) < 2` (currently only when `valid_x.empty`)
- [x] Task 1.4 — Update existing tests in `test_gesture_type.py` to reflect
      single-point strokes now returning `stroke_unknown`
- [x] Task 1.5 — Add test: multi-point stroke where first/last endpoints
      disagree with overall trend (`[3.0, 1.0, 2.0, 3.0]` — start=end so
      endpoint delta=0 → old code: distal; OLS slope≈+0.1 → new code: proximal)
- [x] Task 1.6 — Add test: two-point stroke (minimal valid fit)
- [x] Task 1.7 — Run `pytest code/src/analysis/touch_analytics/preparation/test_gesture_type.py`
      — **18/18 passed**

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/gesture_type.py` — replace
  classification logic (lines 58-72), add numpy import
- `code/src/analysis/touch_analytics/preparation/test_gesture_type.py` — update
  single-point expectation, add regression-specific cases

**Dependencies:** None

---

## Testing Plan

### Unit Tests
- [ ] Tap → `'tap'` (unchanged)
- [ ] Stroke with clear positive slope → `'stroke_proximal'`
- [ ] Stroke with clear negative slope → `'stroke_distal'`
- [ ] Stroke with zero slope (flat) → `'stroke_distal'`
- [ ] Stroke with all NaN → `'stroke_unknown'`
- [ ] Stroke with single valid point → `'stroke_unknown'`
- [ ] Stroke with two valid points → matches sign of their delta
- [ ] Stroke where endpoints disagree with overall trend → slope wins
- [ ] NaN at start/end with valid interior → uses interior points only

### Manual Verification
- [ ] Run `diagnose_stroke_direction.py` on a real database before and after
      the change; compare Phase 1 (summary triage) counts and Phase 2
      (delta-x distribution) to confirm classification balance improves or
      remains stable

### Edge Cases
- [ ] Single-frame stroke (only 1 valid `contact_location_x`)
- [ ] Two-frame stroke (minimal regression)
- [ ] Very short stroke with noise (Δx ≈ 0 but slight trend)

---

## Documentation Plan

- [x] Update docstring of `classify_gesture_type()` to describe regression approach
- [ ] No CLAUDE.md or README changes needed (internal implementation detail)

---

## Rollback Plan

1. Revert the single commit on `feature/stroke-direction-affine-fit`
2. No data migrations, no config changes, no downstream schema changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Single-point strokes now classified as `stroke_unknown` instead of `stroke_distal` — could affect downstream counts | Low | Low | Diagnostic script will detect count changes; `stroke_unknown` is already handled downstream |
| `np.polyfit` on very short series (2-3 points) is numerically equivalent to endpoint delta | Low | None | Acceptable — the improvement is for longer strokes where boundary noise matters |
| Slope near zero gives unstable classification | Med | Low | Same issue exists with current delta approach; no regression in behaviour |

---

## References

- Current implementation: `code/src/analysis/touch_analytics/preparation/gesture_type.py`
- Diagnostic script: `code/scripts/diagnose_stroke_direction.py`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Knowledge base: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`
