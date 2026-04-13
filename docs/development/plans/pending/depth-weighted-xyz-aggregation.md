# Plan: Depth-Weighted XYZ Aggregation

**Created:** 2026-04-13 14:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/depth-weighted-xyz-aggregation`

---

## Overview

**What:** Replace the homogeneous/spread-out branching logic in `EllipseDepthExtractor._aggregate_depth` with a depth-accuracy weighted aggregation that always favours pixels closer to the camera (lower z).

**Why:** When stickers are tilted on the curved forearm surface, the Kinect depth values on the far side of the ellipse are biased toward the background (floor). The current median-based aggregation pulls the extracted position downward. Pixels closer to the camera are inherently more accurate (lower Kinect noise), so weighting by relative depth corrects both the inclination bias and the noise gradient.

**How:** Compute per-pixel weights based on relative z-position within the candidate set (exponential decay from z_min), then aggregate using weighted median instead of plain median. The sticker-size guard is retained as a safety net.

## Problem Statement

- The Kinect camera faces downward; small z = close to camera, large z = close to floor.
- When a sticker sits on a tilted or curved forearm surface, the ellipse mask captures a depth gradient across the sticker area.
- The current algorithm has two paths based on `z_std`: homogeneous (all points, plain median) and spread-out (25th percentile shallow cluster, then median). Neither accounts for the systematic depth bias caused by surface inclination.
- In the homogeneous path (`z_std < 10mm`), no correction is applied at all — yet the bias can still be significant within that threshold.
- In the spread-out path, the hard 25th-percentile cutoff is arbitrary and discards potentially useful data.
- Result: extracted sticker positions are systematically biased toward the floor on inclined/curved surfaces.

## Goals

### In Scope
1. Replace the two-branch aggregation (`z_std < threshold` → all points vs. shallow cluster) with a single depth-weighted aggregation path
2. Implement a `_weighted_median` helper for 1-D weighted median computation
3. Keep the sticker-size guard as a safety net (applied before weighting)
4. Update unit tests to cover the new weighting behaviour
5. Maintain backward-compatible output signature: `(x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped)`

### Out of Scope
- Surface-normal estimation or plane-fitting approaches
- Changes to the ellipse mask construction or depth sampling steps
- Changes to the extractor interface or factory
- Motion correction pipeline changes
- Re-processing existing data (that is a separate step after the code change)

## Success Criteria

- [ ] `_aggregate_depth` uses depth-weighted median instead of plain median
- [ ] Homogeneous/spread-out branching logic is removed (single code path)
- [ ] On a synthetic tilted-surface point cloud, the weighted result is closer to the "near-camera center" than the old median result
- [ ] Sticker-size guard still fires when z-range exceeds `sticker_diameter_mm`
- [ ] All existing tests pass (updated as needed for new signature/behaviour)
- [ ] New tests cover: weighting on tilted surface, weighting on uniform surface (degenerate case), weighted median correctness

---

## Technical Design

### Approach

**Depth-accuracy weighting** assigns each depth sample a weight based on its relative z-position within the candidate set. Pixels with lower z (closer to camera, more accurate on Kinect) receive higher weight. This replaces the binary homogeneous/spread-out branching with a single continuous weighting scheme.

**Weighting function** — exponential decay from the minimum z:

```python
z_norm = (z_i - z_min) / (z_max - z_min)    # 0 = nearest, 1 = farthest
w_i = exp(-z_norm / sigma)                    # sigma controls decay rate
```

With `sigma = 0.3`:
| z_norm | Weight |
|--------|--------|
| 0.0    | 1.00   |
| 0.1    | 0.72   |
| 0.3    | 0.37   |
| 0.5    | 0.19   |
| 1.0    | 0.04   |

**Degenerate case:** When all z values are equal (`z_max == z_min`), `z_norm` is undefined. In this case, all weights are set to 1.0 (uniform), which reproduces the current median behaviour exactly.

**Aggregation** — weighted median for all three axes:

```python
x_mm = weighted_median(candidate_xs, weights)
y_mm = weighted_median(candidate_ys, weights)
z_mm = weighted_median(candidate_zs, weights)
```

The weighted median is the value `v` that minimizes `sum(w_i * |x_i - v|)`. It is computed by sorting the values, computing cumulative weights, and finding the value where cumulative weight crosses 50% of total weight.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Depth-accuracy weighting | Simple, no surface model needed, handles curves, exploits known Kinect accuracy gradient | Reduces but may not fully eliminate bias for extreme tilts | **Chosen** |
| Plane-fit centroid projection | Directly models tilt, evaluates z at exact center | Assumes planar surface (sticker is on curved forearm), sensitive to outliers, needs ≥6 pixels | Rejected |
| Gaussian center-weighted | Weights by distance from ellipse center | Doesn't exploit the depth-accuracy gradient; center pixel isn't necessarily the most accurate | Rejected |
| Hybrid (plane + weighted fallback) | Best accuracy when plane fits | Over-complex for the problem; plane model is wrong for curved surfaces | Rejected |

### Architecture Changes

No new modules or classes. Changes are confined to:

```
code/src/preprocessing/stickers_analysis/xyz/core/
└── xyz_extractor_ellipse_depth.py    — modify _aggregate_depth, add _weighted_median
```

The extractor interface, factory, and orchestrator are unchanged.

---

## Implementation Plan

### Phase 1: Core Algorithm Change
**Goal:** Replace branching with depth-weighted aggregation in `_aggregate_depth`
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 1.1 — Add static method `_weighted_median(values, weights)` to `EllipseDepthExtractor`
- [ ] Task 1.2 — Add module-level constant `_DEPTH_WEIGHT_SIGMA = 0.3` (exponential decay rate)
- [ ] Task 1.3 — Add `depth_weight_sigma` parameter to `__init__` (default `_DEPTH_WEIGHT_SIGMA`)
- [ ] Task 1.4 — Rewrite `_aggregate_depth`: remove `spread_threshold_mm` parameter, remove homogeneous/spread-out branching, add depth-weighted median aggregation
- [ ] Task 1.5 — Update `extract()` to pass new parameters to `_aggregate_depth`
- [ ] Task 1.6 — Remove `spread_threshold_mm` from `__init__` (no longer used)
- [ ] Task 1.7 — Clean up module-level constants: remove `_DEFAULT_SPREAD_THRESHOLD_MM` and `_SHALLOW_CLUSTER_PERCENTILE`

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py` — all changes in this file

**Dependencies:** None

### Phase 2: Test Updates
**Goal:** Update existing tests and add new coverage for depth-weighted behaviour
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 2.1 — Update `TestAggregateDepth` tests: remove spread-threshold-based assertions, add weighted-aggregation assertions
- [ ] Task 2.2 — Add test: uniform z values → weighted median equals plain median (degenerate case)
- [ ] Task 2.3 — Add test: tilted surface (linear z gradient) → weighted result is biased toward low-z (near-camera) side
- [ ] Task 2.4 — Add test: bimodal z distribution (hand + background) → weighted result favours near-camera cluster
- [ ] Task 2.5 — Add test: `_weighted_median` correctness on known values
- [ ] Task 2.6 — Add test: sticker-size guard still fires when z-range > sticker_diameter_mm
- [ ] Task 2.7 — Verify `test_output_monitor_keys` reflects current output keys (fix existing staleness if needed)

**Files Modified:**
- `code/tests/test_ellipse_depth_extractor.py` — update and extend

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `_weighted_median` returns correct value for simple known inputs (e.g., values=[1,2,3], weights=[1,1,100] → 3)
- [ ] `_weighted_median` with uniform weights equals plain median
- [ ] `_aggregate_depth` on uniform z → same result as plain median (weights degenerate to uniform)
- [ ] `_aggregate_depth` on tilted surface (z = 700 to 710 linear gradient) → z_mm closer to 700 than 705
- [ ] `_aggregate_depth` on bimodal distribution (30 pts at 700, 30 pts at 800) → z_mm close to 700
- [ ] Sticker-size guard fires when z-range > sticker_diameter_mm after weighting

### Integration Tests
- [ ] `extract()` produces valid output with depth-weighted path on synthetic tilted point cloud
- [ ] `extract()` fallback path (no ellipse data) still works unchanged

### Manual Verification
- [ ] Run extraction on a known session with tilted-sticker frames, compare old vs new z_mm values
- [ ] Visually inspect extracted trajectories in the scene viewer for reduced floor-ward bias

---

## Documentation Plan

- [ ] Update inline docstrings in `_aggregate_depth` to describe depth-weighting
- [ ] Add knowledge-base note: `docs/development/knowledge-base/note-depth-inclination-bias-correction.md`

---

## Rollback Plan

1. The change is confined to `_aggregate_depth` and `_weighted_median` in a single file
2. Revert the single commit to restore the old branching logic
3. No data migration needed — output CSV format is unchanged

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sigma value (0.3) too aggressive for some sessions | Med | Med | Expose as constructor parameter; can be tuned per-session if needed |
| Weighted median is slower than plain median | Low | Low | N is typically 50-300 pixels; sorting cost is negligible |
| Degenerate case (all z equal) causes division by zero | Low | High | Explicit check: if z_max == z_min, use uniform weights |

---

## References

- Existing plan: `docs/development/plans/completed/sticker-depth-edge-gradient-bias-correction.md` (introduced the current branching logic)
- Test file: `code/tests/test_ellipse_depth_extractor.py`
- Misc test scripts: `code/scripts/__misc/test_ellipse_depth_spread_threshold.py`
