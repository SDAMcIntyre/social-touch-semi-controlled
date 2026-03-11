# Plan: Sticker Depth Edge-Gradient Bias Correction

**Date:** 2026-02-23
**Author:** Basil Duvernoy
**Status:** In Progress
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
- [ ] Output CSV contains `z_range_clipped` monitor column indicating when the
      sticker-size guard fired
- [ ] No frame produces a z-range > 10 mm in the output (guard fires and corrects it)
- [ ] On a known edge-heavy recording, z-bias is visibly reduced vs. `centroid` extractor
- [ ] `sticker_diameter_mm` can be overridden at instantiation without touching defaults
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

**Sticker physical diameter**: The stickers used in all recordings have a known diameter
of approximately **10 mm**. The ellipse fit will typically produce an ellipse *smaller*
than the physical sticker due to detection noise and partial occlusion at the boundary.
This physical size acts as a hard upper bound on legitimate depth variation within the
sticker area: if the observed z-range within the ellipse mask exceeds this diameter,
some sampled pixels must be capturing the background depth gradient rather than the
sticker surface. In that case the aggregation must be refined to retain only the
upper-Z (closest-to-camera, minimum-Z) values.

A module-level constant `STICKER_DIAMETER_MM = 10.0` is introduced in the new extractor
file and passed as a default argument to `EllipseDepthExtractor.__init__()`, keeping
it easy to override per-instantiation without touching call sites that rely on the
default.

---

## Implementation Plan

### Phase 1: New Extractor Class
**Goal:** Implement `EllipseDepthExtractor` with ellipse-mask depth sampling

**Tasks:**
- [x] Task 1.1 — Create `xyz_extractor_ellipse_depth.py` with `EllipseDepthExtractor`
      class implementing `XYZExtractorInterface`
- [x] Task 1.2 — Implement `_build_ellipse_mask()`: static method that returns a boolean
      2D array from ellipse centre, axes, and angle
- [x] Task 1.3 — Implement `_sample_depth_within_mask()`: extract z values from point
      cloud using the mask; return arrays of (x, y, z) for valid (non-zero) pixels
- [x] Task 1.4 — Implement `_aggregate_depth()`: homogeneity check + adaptive aggregation
      (median vs. shallow-cluster percentile); return (x_mm, y_mm, z_mm, z_std, n_pixels)
- [x] Task 1.5 — Implement `extract()`: orchestrate mask → sample → aggregate pipeline,
      with fallback to single-pixel when ellipse unavailable
- [x] Task 1.6 — Implement `can_process()`, `should_process_row()`, `get_empty_result()`
      matching the interface contract

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py` — **NEW**

**Dependencies:** None

### Phase 3: Sticker-Size-Constrained Depth Validation
**Goal:** Use the known 10 mm sticker diameter as a physical plausibility guard and
refine depth estimation when the observed z-spread exceeds it

**Context:**
The sticker diameter (~10 mm) is a physical constant known before any processing begins.
When the z-range across the ellipse mask exceeds this value, at least part of the sampled
area is contaminated by the background depth ramp.  The fix is to discard those far
pixels and keep only the upper-Z (nearest-to-camera) values — i.e. the lowest-Z subset,
since Kinect Z increases with distance from the sensor.

**Tasks:**
- [x] Task 3.1 — Define `STICKER_DIAMETER_MM: float = 10.0` as a module-level constant
      in `xyz_extractor_ellipse_depth.py`
- [x] Task 3.2 — Add `sticker_diameter_mm: float = STICKER_DIAMETER_MM` parameter to
      `EllipseDepthExtractor.__init__()`; store as `self._sticker_diameter_mm`
- [x] Task 3.3 — In `_aggregate_depth()`, add a post-homogeneity-check step: compute
      `z_range = z_max - z_min` for the current candidate z array; if
      `z_range > self._sticker_diameter_mm`, override the chosen cluster with the upper-Z
      (minimum-Z) subset — e.g., all z values ≤ `z_min + self._sticker_diameter_mm`
- [x] Task 3.4 — Emit a boolean or float monitor column `z_range_clipped` (or include it
      in existing monitor columns) so the caller can tell when the size-guard fired
- [x] Task 3.5 — Update `get_empty_result()` and `extract()` to include the new monitor
      column with the correct NaN / False default

**Where the constant lives (design decision):**
The first step is a module-level default (`STICKER_DIAMETER_MM = 10.0`) in the new
extractor file, which is then wired in as the default constructor argument.  This avoids
a global config dependency while still being easy to override at the call site (e.g.,
when different sticker sizes are used in a future study).  A central config file can be
considered once multiple classes need the same value.

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py`
  — new constant, new constructor param, refined aggregation logic, new monitor field

**Dependencies:** Phase 1

---

### Phase 2: Factory Registration
**Goal:** Make the new extractor selectable via the factory

**Tasks:**
- [x] Task 2.1 — Add `ELLIPSE_DEPTH = "ellipse_depth"` to `ExtractorChoice` enum
- [x] Task 2.2 — Add registry entry mapping `ELLIPSE_DEPTH` → `EllipseDepthExtractor`
- [x] Task 2.3 — Update `__init__.py` exports to include `EllipseDepthExtractor`

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_factory.py` — add enum
  + registry entry
- `code/src/preprocessing/stickers_analysis/xyz/__init__.py` — export new class
- `code/src/preprocessing/stickers_analysis/__init__.py` — re-export if needed

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [x] `_build_ellipse_mask()` returns correct mask shape and approximate pixel count for
      known ellipse parameters
- [x] `_aggregate_depth()` returns median z when all z values are within threshold
- [x] `_aggregate_depth()` returns shallow-cluster z when z values span a wide range
- [x] `extract()` falls back to single-pixel when ellipse columns are missing
- [x] `extract()` falls back to single-pixel when mask yields < 3 valid depth pixels
- [x] `get_empty_result()` returns NaN for all fields including z_std, n_depth_pixels

### Manual Verification
- [ ] Run XYZ extraction on a known recording using `method="ellipse_depth"` and compare
      output CSV against `method="centroid"` for the same recording
- [ ] Inspect z trajectories for edge-adjacent stickers (blue, on knuckles/fingertips) —
      verify reduced z oscillation and fewer outlier spikes
- [ ] Check that `z_std` column is populated and correlates with visually ambiguous frames

### Edge Cases
- [x] Ellipse entirely outside point cloud bounds → fallback to single-pixel
- [x] Very small ellipse (< 3 valid pixels) → fallback to single-pixel
- [x] All depth values within ellipse are zero (sensor dropout) → NaN result
- [x] `score` below threshold in consolidated tracks (ROI centre used, no ellipse) →
      fallback works correctly
- [x] z-range within ellipse exceeds `sticker_diameter_mm` → size guard fires, upper-Z
      subset selected, `z_range_clipped` is True in output
- [x] z-range within ellipse is within `sticker_diameter_mm` → size guard does not fire,
      `z_range_clipped` is False
- [x] Custom `sticker_diameter_mm` passed at construction → guard uses the overridden
      value instead of the module-level default

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
| z-range guard too aggressive on curved hand surfaces | Low | Med | Guard uses known physical diameter (10 mm) as threshold; curved surfaces won't span more than the sticker itself |
| Sticker sizes differ across studies | Low | Low | `sticker_diameter_mm` is a constructor parameter; different values can be passed per-study without changing defaults |

---

## References

- Idea: `docs/development/plans/ideas/sticker-tracking-failures.md`
- Related ideas: sticker-background-snap, green-sticker-forearm-separation
- Key source: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_centroid.py`
- Factory: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_factory.py`
- Orchestrator: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extracting_orchestrator.py`
- Consolidation: `code/scripts/_3_preprocessing/_1_sticker_tracking/consolidate_2d_tracking_data.py`
