# Note: Circular Crop Image Scale Consistency Across Sessions

**Date:** 2026-06-11 (updated 2026-06-23)
**Context:** `spatial_extract_boundaries` task — `*_rf_population_all_circular_*.png` outputs;
comparison pipelines — `spatial_compare_proximal_distal`, `spatial_compare_tap_stroke`,
`spatial_compare_boundaries`

## Summary

The circular crop PNGs produced by `render_population_rf_circular_crop()` have
a **consistent pixel-to-mm scale across all sessions**. N pixels in session A
represents the same physical distance as N pixels in session B.

## Two rendering modes

### Per-image viewport (no shared limits)

Used when `xlim`/`ylim` are not provided. Each image is centered on its own
`center_uv ± radius_uv`, and `bbox_inches='tight'` crops to the circle content.

```
radius_mm = 50.0                          # constant (hardcoded in pipeline)
scale     = compute_uv_to_mm_scale(...)   # varies per session mesh
radius_uv = radius_mm / scale             # varies inversely with scale

extent_uv = 2 * radius_uv                # varies
extent_mm = extent_uv * scale             # cancels out → always 100 mm
```

The figure dimensions are fixed (matplotlib default 6.4 x 4.8 in, `dpi=300`,
`set_aspect('equal')`, `bbox_inches='tight'`), so the output pixel count is
constant across sessions.

### Fixed viewport (shared limits)

Used when `xlim`/`ylim` are provided (all comparison pipelines). The figure
is sized to match the viewport aspect ratio, the axes fill the entire figure
(`subplots_adjust(left=0, right=1, top=1, bottom=0)`), and the image is
saved with `pad_inches=0` **without** `bbox_inches='tight'`.

This guarantees all images within a session share the exact same viewport:
forearm landmarks stay at the same pixel position across gesture types.
`bbox_inches='tight'` must not be used here — it would crop each image to
its circle's bounding box, defeating the shared viewport.

Within each session, the **circle center is fixed** from the `'all'` gesture
type (centroid, peak, or contour_center depending on the crop category). The
heatmap data varies per gesture type, but the spatial window is identical.

## Margin parameter

`circular_crop_margin` (default `0.0`) adds padding around the circle as a
fraction of `radius_uv`. It is set in the DAG config options and passed
through the workflow scripts to the pipeline functions. A value of `0.0`
means the circle edge coincides with the viewport edge (for per-image mode)
or contributes no extra padding to the shared limits (for fixed-viewport mode).

## Caveat: Local UV Distortion

`compute_uv_to_mm_scale` returns a **global median** of the 3D-to-UV edge
length ratio across all mesh edges. SLIM parameterizations are
area-minimizing but not perfectly isometric, so some local
stretching/compression exists. This means:

- The circle boundary is at exactly 50 mm from the center (correct globally).
- Local distances *within* the circle may be slightly distorted depending on
  where on the forearm the crop is centered.
- This is inherent to the UV mapping, not a rendering defect.
