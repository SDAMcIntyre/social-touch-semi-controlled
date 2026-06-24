# Idea: Centroid XYZ extractor — parallax tolerance

**Date:** 2026-04-14
**Status:** Superseded by `kinect-frame-parallax-correction` plan — correction applied at the `KinectFrame` data-access layer instead.

## Summary

`CentroidPointCloudExtractor.get_xyz_from_point_cloud(point_cloud, px, py)`
performs a single `point_cloud[iy, ix]` lookup at an RGB-derived sticker
centroid. The audit in
`docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
flagged this site as **unaccounted**: at the median ±10 px (worst ±25 px)
RGB↔depth parallax in the 500–800 mm operating range, the lookup
typically lands outside the sticker's depth blob and returns a z value
that may be metres off (background or sticker edge).

## Rough Approach

Replace the bare single-pixel lookup with a small-area sample plus a
depth-plausibility filter, mirroring what
`xyz_extractor_ellipse_depth.py` already does but without requiring
ellipse parameters:

1. Sample `point_cloud[iy ± k, ix ± k]` for some `k` chosen to cover
   the worst-case offset (≥ 25 px in u, slightly less in v).
2. Drop NaN / zero entries.
3. Keep only pixels within one sticker-diameter (~10 mm) of the nearest
   z value in the sample (same z-guard as the ellipse extractor).
4. Return the (weighted) median XYZ of the surviving subset.

## Notes

- The +u sign-consistency (97 %) means an asymmetric sampling window
  biased toward `ix − 10 ± k` would beat a symmetric one for the same
  pixel budget. Worth measuring.
- This is functionally a degenerate case of the ellipse extractor; if
  every caller can supply ellipse parameters, the centroid path could
  simply be deprecated rather than patched.
- Source: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_centroid.py`
  (`get_xyz_from_point_cloud`, ~line 344).
- Reference band: KB note above; CSVs under
  `F:/_tmp/kinect-measurement-offsets/`.
