# Plan: Fix Velocity Units (mm/frame to mm/s)

**Date:** 2026-03-20
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/fix-velocity-units`

---

## Overview

The kinematic feature extraction pipeline computes sticker velocity as raw
frame-to-frame displacement (mm/frame) instead of a proper physical velocity
(mm/s). This makes all velocity-derived features appear ~30x too low when
compared against real motion observed in the video recordings. The fix
multiplies displacement by the capture frame rate (30 fps) at the source
function so all downstream consumers automatically receive mm/s values.

## Problem Statement

`compute_velocity_magnitudes()` in `kinematics.py` returns the Euclidean
distance between consecutive blue-sticker positions — a displacement in
**mm/frame**. At 30 fps a gentle stroke at 6 cm/s appears as ~2 mm/f instead
of the expected 60 mm/s, making velocity features look unrealistically low
when compared to videos.

Additionally, `mos_extractor.py` has a misleading comment claiming the input
is mm/s and a strain-rate formula that divides by fps unnecessarily. The two
errors happen to cancel numerically (strain_rate output is accidentally
correct), but the intermediate values and comments are wrong.

## Goals

### In Scope

1. Convert `compute_velocity_magnitudes()` output from mm/frame to mm/s
2. Convert `compute_acceleration_magnitudes()` output from mm/frame² to mm/s²
3. Fix the misleading conversion and formula in `mos_extractor.py`
4. Update the knowledge-base documentation to reflect the new units

### Out of Scope

- Changing the velocity compass GUI (it receives raw `pos - prev_pos`
  independently, not from `kinematics.py`, and is intentionally mm/frame for
  real-time display)
- Changing the motion-correction outlier detector (already uses `np.gradient`
  with proper dt scaling — independent of `kinematics.py`)
- Recalibrating downstream thresholds or plot axes — these are research
  outputs regenerated on each run

## Success Criteria

- [ ] `compute_velocity_magnitudes()` returns mm/s (displacement * fps)
- [ ] `compute_acceleration_magnitudes()` returns mm/s² (velocity diff * fps)
- [ ] `mos_extractor.py` strain_rate uses `velocity / h` (no `/fps`)
- [ ] All extractors pass fps from config to kinematics functions
- [ ] Knowledge-base note updated from mm/frame to mm/s
- [ ] A gentle stroke (~5 cm/s) yields max_velocity ~50 mm/s (not ~1.7 mm/f)

---

## Technical Design

### Approach

Add an `fps` parameter (default 30.0) to both kinematics functions. The
displacement (or velocity diff) is multiplied by fps inside the function,
producing proper physical units. All callers read fps from their config dict
and forward it. This is a minimal, localized change — no new modules, no API
restructuring.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Multiply by fps inside `kinematics.py` (chosen) | Single source of truth, all consumers get correct units automatically | Slightly changes the function signature | **Chosen** |
| Multiply by fps at each call site | No signature change | Scattered conversions, easy to forget, violates DRY | Rejected |
| Create separate `compute_velocity_mm_per_s()` | Backward compatible | Two nearly identical functions, confusing | Rejected |

### Architecture Changes

No new modules or classes. Only parameter additions and formula corrections
in existing files.

### Knowledge-Base Relevance

- **`note-somatosensory-units-and-calculations.md`** — directly relevant.
  Documents the current mm/frame unit, the 30 fps capture rate, and the
  coordinate system. Must be updated to reflect mm/s after this change.
- All other knowledge-base notes: not applicable.

---

## Implementation Plan

### Phase 1: Core fix
**Goal:** Convert kinematics functions to return physical units (mm/s, mm/s²)

**Tasks:**

- [x] Task 1.1 — Add `fps: float = 30.0` parameter to
  `compute_velocity_magnitudes()`, multiply displacement by fps, update
  docstring to state mm/s
- [x] Task 1.2 — Add `fps: float = 30.0` parameter to
  `compute_acceleration_magnitudes()`, multiply velocity diff by fps, update
  docstring to state mm/s²

**Files Modified:**

- `code/src/analysis/touch_analytics/feature_extraction/kinematics.py` — add
  fps parameter, multiply by fps, update docstrings

**Dependencies:** None

### Phase 2: Fix consumers
**Goal:** All extractors pass fps from config and use correct formulas

**Tasks:**

- [x] Task 2.1 — `mos_extractor.py`: pass `fps` to
  `compute_velocity_magnitudes()`, fix strain_rate formula
  (`vel / h` instead of `vel / (h * fps)`), fix comment on line 52
- [x] Task 2.2 — `max_extractor.py`: read fps from config, pass to
  kinematics functions
- [x] Task 2.3 — `statistical_extractor.py`: read fps from config, pass to
  kinematics functions
- [x] Task 2.4 — `touch_analysis.py`: replace inline velocity computation
  (lines 69-72) with call to `compute_velocity_magnitudes()` from kinematics
- [x] Task 2.5 — `temporal_extractor.py`: remove unused import of
  `compute_velocity_magnitudes` (line 5)

**Files Modified:**

- `code/src/analysis/touch_analytics/feature_extraction/mos_extractor.py` —
  fix conversion comment, fix strain_rate formula, pass fps
- `code/src/analysis/touch_analytics/feature_extraction/max_extractor.py` —
  pass fps from config
- `code/src/analysis/touch_analytics/feature_extraction/statistical_extractor.py` —
  pass fps from config
- `code/src/analysis/touch_analytics/touch_analysis.py` —
  replace inline computation with kinematics import
- `code/src/analysis/touch_analytics/feature_extraction/temporal_extractor.py` —
  remove dead import

**Dependencies:** Phase 1

### Phase 3: Documentation
**Goal:** Update knowledge-base to reflect new units

**Tasks:**

- [x] Task 3.1 — Update `note-somatosensory-units-and-calculations.md`
  Section 2 (Velocity): unit → mm/s, remove "multiply by 30" conversion note,
  update code snippet
- [x] Task 3.2 — Update Section 6 (Reusable pattern): note that
  `compute_velocity_magnitudes()` handles fps internally

**Files Modified:**

- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Run the analysis pipeline on a known session; confirm max_velocity
  values are ~30x larger than before (e.g., ~50 mm/s for a gentle stroke
  instead of ~1.7 mm/f)
- [ ] Confirm strain_rate values from MoS extractor are numerically unchanged
  (the two canceling errors mean the output stays the same)
- [ ] Confirm the velocity compass GUI still displays mm/f correctly
  (unaffected by this change)

### Edge Cases

- [ ] Touch with zero displacement (stationary frame) — velocity should be 0
- [ ] Single-frame touch — diff produces NaN filled to 0, velocity = 0

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`

---

## Rollback Plan

1. Revert the single commit on `feature/fix-velocity-units`
2. No data migration needed — feature CSVs are regenerated on each pipeline run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream code assumes mm/frame values | Low | Med | Grep for all consumers of `compute_velocity_magnitudes` — all identified and accounted for |
| MoS strain_rate changes unexpectedly | Low | Med | Two canceling errors mean strain_rate output is numerically unchanged after fix |
| Velocity compass breaks | None | — | Compass uses raw `pos - prev_pos`, not kinematics.py |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Core function: `code/src/analysis/touch_analytics/feature_extraction/kinematics.py`
