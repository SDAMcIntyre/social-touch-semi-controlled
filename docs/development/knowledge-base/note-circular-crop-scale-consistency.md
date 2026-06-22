# Note: Circular Crop Image Scale Consistency Across Sessions

**Date:** 2026-06-11
**Context:** `spatial_extract_boundaries` task — `*_rf_population_all_circular_*.png` outputs

## Summary

The circular crop PNGs produced by `render_population_rf_circular_crop()` have
a **consistent pixel-to-mm scale across all sessions**. N pixels in session A
represents the same physical distance as N pixels in session B.

## Mechanism

The conversion chain from mm to UV space introduces a session-dependent scale
factor, but the figure-sizing pipeline cancels it out exactly:

```
radius_mm = 50.0                          # constant (hardcoded in pipeline)
scale     = compute_uv_to_mm_scale(...)   # varies per session mesh
radius_uv = radius_mm / scale             # varies inversely with scale

extent_uv = 2 * radius_uv * 1.05         # varies (includes 5% margin)
extent_mm = extent_uv * scale             # cancels out → always 105 mm
```

The figure dimensions are fixed (matplotlib default 6.4 x 4.8 in, `dpi=300`,
`set_aspect('equal')`, `bbox_inches='tight'`), so the output pixel count is
constant across sessions.

## Code references

| What | File | Line(s) |
|------|------|---------|
| `radius_mm=50.0` call sites | `pipelines/rf_population_response_field_pipeline.py` | 628, 654 |
| `compute_uv_to_mm_scale()` | `rendering/rf_population_map_renderer.py` | 515-559 |
| `render_population_rf_circular_crop()` | `rendering/rf_population_map_renderer.py` | 562-668 |
| `radius_uv = radius_mm / scale` | `rendering/rf_population_map_renderer.py` | 627 |
| `fig, ax = plt.subplots(1, 1)` (no figsize) | `rendering/rf_population_map_renderer.py` | 635 |
| xlim/ylim set from `radius_uv` | `rendering/rf_population_map_renderer.py` | 662-663 |
| `savefig(dpi=dpi, bbox_inches='tight')` | `rendering/rf_population_map_renderer.py` | 666 |

All paths relative to `code/src/analysis/receptive_field_mapping/`.

## Caveat: Local UV Distortion

`compute_uv_to_mm_scale` returns a **global median** of the 3D-to-UV edge
length ratio across all mesh edges. SLIM parameterizations are
area-minimizing but not perfectly isometric, so some local
stretching/compression exists. This means:

- The circle boundary is at exactly 50 mm from the center (correct globally).
- Local distances *within* the circle may be slightly distorted depending on
  where on the forearm the crop is centered.
- This is inherent to the UV mapping, not a rendering defect.
