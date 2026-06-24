# Idea: KinectPointCloudView — surface the parallax envelope

**Date:** 2026-04-14
**Status:** Superseded by `kinect-frame-parallax-correction` plan — correction applied at the `KinectFrame` data-access layer instead.

## Summary

`KinectPointCloudView` (in
`code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py`)
flattens `(H, W, 3)` color and `(H, W, 3)` `transformed_depth_point_cloud`
as parallel `(N, 3)` vectors, presenting per-pixel-paired data as the
public API. The audit in
`docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
flagged this wrapper as **unaccounted**: the implicit pairing carries the
~10 px median / 25 px worst-case parallax into every consumer that
combines `.color` with `.points`.

The colored point cloud as rendered by `neural_kinect_scene_viewer.py` is
visually fine (the offset is sub-perceptual at viewer scale), but any
**measurement** built on the wrapper — picking a 3D point by its color,
seeding a forearm segmentation from RGB-derived hue, etc. — inherits the
offset.

## Rough Approach

Two non-mutually-exclusive options:

1. **Document-only:** Add a docstring warning to
   `_create_frame_view` referencing the KB note, plus an in-line
   comment at the `flat_colors_vector = color_rgb.reshape(-1, 3)` line
   stating the implicit parallax. Cheap, no behaviour change.
2. **API hardening:** Add an optional `tolerance_band_px: tuple[int, int]`
   constructor argument that, when set, refuses to expose the
   per-pixel-paired flattened view and instead returns separate color
   and depth grids the caller must explicitly reconcile. Stronger
   guarantee but pushes work onto consumers that may not need it.

Pick option 1 first; revisit option 2 only if a consumer is found that
silently relies on the pairing for measurements (not visualisation).

## Notes

- Source: `code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py`
  (`_create_frame_view`, ~line 42).
- Direct upstream consumer: `code/src/merging/gui/neural_kinect_scene_viewer.py`
  (renders only — current use is fine).
- Reference band: `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`.
- Related: `docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md`.
