# Plan: Sticker Depth Edge-Gradient Bias Correction

**Date:** 2026-02-23
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/sticker-depth-edge-gradient-bias`

---

## Overview

The Kinect depth sensor produces a smooth depth gradient at object boundaries, biasing
sticker 3D positions away from the hand surface. The current `CentroidPointCloudExtractor`
samples a single centre pixel from the point cloud, making it blind to this edge effect.
This plan introduces a new extractor that samples all depth points within the sticker's
fitted ellipse and uses a homogeneity-adaptive aggregation to return an accurate depth
value plus a per-frame quality metric.

## Problem Statement

`get_xyz_from_point_cloud()` in `xyz_extractor_centroid.py:344` samples one pixel. When
the sticker sits on a hand edge (fingertips, knuckles), that pixel may land on the depth
gradient ramp — a blend of hand-surface depth and background depth. The 2D quality gate
(`score` threshold, ellipse fit) does not catch this because the 2D detection is fine; the
problem only appears at the 3D lift step. The result is a systematic z-bias away from the
camera on edge-adjacent frames.

ST13-03 block 07 notes "position of blue stickers which can lack of accuracy" — likely
this effect.

## Goals

### In Scope
1. New extractor class that samples all depth points within the tracked ellipse
2. Homogeneity-adaptive aggregation: median for clean surfaces, shallow-cluster selection
   for edge-straddling ellipses
3. Per-frame depth spread metric (`z_std`) emitted alongside coordinates
4. Graceful fallback to single-pixel sampling when ellipse data is unavailable
5. Registration in the extractor factory so it can be selected via CLI

### Out of Scope
- Background-snap correction (temporal, consecutive-frame problem — separate idea)
- Downstream smoothing (Savitzky-Golay — applied after both fixes)
- Green-sticker forearm separation (geometric ambiguity, different root cause)
- Changes to the 2D tracker, ellipse fitter, or consolidation step

## Success Criteria

- [ ] New extractor registered as `ExtractorChoice.ELLIPSE_DEPTH` (`"ellipse_depth"`)
- [ ] Extractor correctly falls back to single-pixel when ellipse columns are absent
- [ ] Output CSV contains `z_std` and `n_depth_pixels` monitor columns
- [ ] On a known edge-heavy recording, z-bias is visibly reduced vs. `centroid` extractor
- [ ] Existing `centroid` and `roi_centroid` extractors are unchanged

---

## Technical Design

### Approach

Create a **new extractor class** `EllipseDepthExtractor` following the existing strategy
pattern. This preserves the original extractors for A/B comparison and fits cleanly into
the factory/registry architecture.

The extraction logic:
1. Build an ellipse pixel mask from the tracked ellipse parameters in `tracked_obj_row`
2. Sample all non-zero depth values within the mask from the point cloud
3. Assess depth homogeneity via z standard deviation
4. **Homogeneous** (z_std < threshold): return median of all valid z values, with x/y
   from the median of valid points within the mask
5. **Spread-out** (z_std >= threshold): the ellipse straddles an edge; select the
   shallow (nearest-to-camera) cluster by taking the lower percentile (10th–25th) of z
   values and medianing those
6. Return (x_mm, y_mm, z_mm) + (px, py, z_std, n_depth_pixels)

### Data Availability

The consolidated tracks CSV (produced by `consolidate_2d_tracking_data.py`) merges ROI
and ellipse data. The `tracked_obj_row` Series passed to the extractor already contains:

| Column | Source | Description |
|--------|--------|-------------|
| `center_x`, `center_y` | Consolidation | Final pixel centre (ellipse or ROI fallback) |
| `ellipse_center_x`, `ellipse_center_y` | Ellipse fit | Original ellipse centre in global frame |
| `axes_major`, `axes_minor` | Ellipse fit | Ellipse semi-axes in pixels |
| `angle` | Ellipse fit | Ellipse rotation angle |
| `score` | Ellipse fit | Confidence score (NaN → no ellipse) |
| `roi_x`, `roi_y`, `roi_width`, `roi_height` | ROI tracker | Bounding box |
| `status` | ROI tracker | Tracking status |

When `score` is below the consolidation threshold, the ellipse columns are NaN (from the
left merge). The new extractor uses `pd.notna(axes_major)` to decide which sampling path
to take.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New extractor class | Clean separation, A/B comparison, follows factory pattern | Slight code duplication for fallback path | **Chosen** |
| Modify existing `CentroidPointCloudExtractor` | No new files | Loses comparison ability, risk of regression | Rejected |
| ROI bounding-box mask instead of ellipse | Simpler mask | Includes background pixels by definition, less precise | Rejected |

### Architecture Changes

```
xyz/core/
├── xyz_extractor_interface.py          # unchanged
├── xyz_extractor_centroid.py           # unchanged
├── xyz_extractor_roi_centroid.py       # unchanged
├── xyz_extractor_ellipse_depth.py      # NEW — EllipseDepthExtractor
├── xyz_extractor_factory.py            # add ELLIPSE_DEPTH enum + registry entry
└── xyz_extracting_orchestrator.py      # unchanged (already generic)
```

No changes to the orchestrator, consolidation step, or data models. The new extractor
receives the same `tracked_obj_row` Series and `point_cloud` array as existing extractors.

### Key Design Decisions

**Ellipse mask construction**: Use `cv2.ellipse()` to draw a filled ellipse on a
zero-initialised mask of point-cloud dimensions. The ellipse parameters come from
`tracked_obj_row['axes_major']`, `axes_minor`, `angle`, and the centre pixel at
`(center_x, center_y)`.

**z aggregation for the spread-out case**: Use a simple percentile approach (lower 25th
percentile of z values) rather than a full bimodal GMM fit. This is cheaper, deterministic,
and sufficient given the Kinect depth gradient is roughly monotonic near edges.

**x/y coordinates**: For the homogeneous case, use the median x and median y of valid
points. For the spread-out case, use the median x/y of the selected shallow-cluster
points. This avoids the single-centre-pixel dependency for x/y as well.

**Fallback**: When ellipse columns (`axes_major`) are missing or the ellipse mask yields
fewer than a minimum number of valid depth pixels (e.g., < 3), fall back to single-pixel
extraction identical to `CentroidPointCloudExtractor.get_xyz_from_point_cloud()`.

**Spread threshold**: Default 10 mm. This is ~2x the Kinect v2 depth noise at typical
hand-distance ranges (0.5–1.0 m). Configurable via constructor parameter.

---

## Implementation Plan

### Phase 1: New Extractor Class
**Goal:** Implement `EllipseDepthExtractor` with ellipse-mask depth sampling

**Tasks:**
- [ ] Task 1.1 — Create `xyz_extractor_ellipse_depth.py` with `EllipseDepthExtractor`
      class implementing `XYZExtractorInterface`
- [ ] Task 1.2 — Implement `_build_ellipse_mask()`: static method that returns a boolean
      2D array from ellipse centre, axes, and angle
- [ ] Task 1.3 — Implement `_sample_depth_within_mask()`: extract z values from point
      cloud using the mask; return arrays of (x, y, z) for valid (non-zero) pixels
- [ ] Task 1.4 — Implement `_aggregate_depth()`: homogeneity check + adaptive aggregation
      (median vs. shallow-cluster percentile); return (x_mm, y_mm, z_mm, z_std, n_pixels)
- [ ] Task 1.5 — Implement `extract()`: orchestrate mask → sample → aggregate pipeline,
      with fallback to single-pixel when ellipse unavailable
- [ ] Task 1.6 — Implement `can_process()`, `should_process_row()`, `get_empty_result()`
      matching the interface contract

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py` — **NEW**

**Dependencies:** None

### Phase 2: Factory Registration
**Goal:** Make the new extractor selectable via the factory

**Tasks:**
- [ ] Task 2.1 — Add `ELLIPSE_DEPTH = "ellipse_depth"` to `ExtractorChoice` enum
- [ ] Task 2.2 — Add registry entry mapping `ELLIPSE_DEPTH` → `EllipseDepthExtractor`
- [ ] Task 2.3 — Update `__init__.py` exports to include `EllipseDepthExtractor`

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_factory.py` — add enum
  + registry entry
- `code/src/preprocessing/stickers_analysis/xyz/__init__.py` — export new class
- `code/src/preprocessing/stickers_analysis/__init__.py` — re-export if needed

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `_build_ellipse_mask()` returns correct mask shape and approximate pixel count for
      known ellipse parameters
- [ ] `_aggregate_depth()` returns median z when all z values are within threshold
- [ ] `_aggregate_depth()` returns shallow-cluster z when z values span a wide range
- [ ] `extract()` falls back to single-pixel when ellipse columns are missing
- [ ] `extract()` falls back to single-pixel when mask yields < 3 valid depth pixels
- [ ] `get_empty_result()` returns NaN for all fields including z_std, n_depth_pixels

### Manual Verification
- [ ] Run XYZ extraction on a known recording using `method="ellipse_depth"` and compare
      output CSV against `method="centroid"` for the same recording
- [ ] Inspect z trajectories for edge-adjacent stickers (blue, on knuckles/fingertips) —
      verify reduced z oscillation and fewer outlier spikes
- [ ] Check that `z_std` column is populated and correlates with visually ambiguous frames

### Edge Cases
- [ ] Ellipse entirely outside point cloud bounds → fallback to single-pixel
- [ ] Very small ellipse (< 3 valid pixels) → fallback to single-pixel
- [ ] All depth values within ellipse are zero (sensor dropout) → NaN result
- [ ] `score` below threshold in consolidated tracks (ROI centre used, no ellipse) →
      fallback works correctly

---

## Documentation Plan

- [ ] Archive idea file: move `ideas/sticker-tracking-failures.md` to indicate promoted
- [ ] Add inline comments in the new extractor explaining the aggregation strategy

---

## Rollback Plan

1. The new extractor is additive — no existing code is modified beyond factory registration
2. To revert: remove `ELLIPSE_DEPTH` from factory enum/registry, delete the new file
3. Existing `centroid` and `roi_centroid` extractors remain fully functional throughout
4. No data migrations; output CSV schema adds columns but is backward-compatible

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ellipse mask too small on distant hands (few pixels) | Med | Low | Fallback to single-pixel when < 3 valid depth pixels |
| Percentile approach selects noise rather than surface | Low | Med | Use 25th percentile (not 10th) as default; make configurable |
| Performance overhead of per-frame mask construction | Low | Low | Mask is tiny (~100-500 pixels); cv2.ellipse is fast |
| Ellipse columns sometimes missing in old CSVs | Med | Low | Explicit column check with graceful fallback |

---

## References

- Idea: `docs/development/plans/ideas/sticker-tracking-failures.md`
- Related ideas: sticker-background-snap, green-sticker-forearm-separation
- Key source: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_centroid.py`
- Factory: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_factory.py`
- Orchestrator: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extracting_orchestrator.py`
- Consolidation: `code/scripts/_3_preprocessing/_1_sticker_tracking/consolidate_2d_tracking_data.py`
