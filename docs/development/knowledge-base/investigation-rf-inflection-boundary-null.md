# Investigation: RF Inflection Boundary Returns Null

**Date:** 2026-05-19
**Branch:** `feature/population-rf-inflection-boundary`
**Status:** Open — 3/4 gestures fixed, `tap` still fails

---

## Summary

The inflection boundary feature had two bugs, discovered sequentially:

1. **Argument order** (fixed, committed) — `grid_z` was passed where `grid_u`
   was expected at both call sites.
2. **Contour enclosure topology** (fixed, under test) — the original algorithm
   used `_select_enclosing_contour()` which required a Laplacian zero-crossing
   contour to geometrically enclose the peak via `Path.contains_point()`. On
   real forearm data this never succeeded because the zero-crossing contours
   wrapped around low-IFF valleys, not around the peak.

## Root Cause: Contour Enclosure Topology

**Observed via debug PNG** (Laplacian RdBu + contours + peak overlay):

- The peak (maximum IFF) sits in a **negative-Laplacian region** (concave
  down) — correct for a maximum.
- The Laplacian zero-crossing contours form **closed loops around the positive
  (red) regions** — the flanks/valleys.
- The peak is **geometrically outside** all these contours, so
  `Path.contains_point(peak)` returned False for every contour.
- This happened for all gesture types with `sigma=2.0`.

**Why synthetic tests passed:** Test data uses a centered Gaussian on a fully
populated grid. The zero-crossing forms a clean ring around the peak. Real
data has 70–81% NaN (forearm-shaped valid region) and complex interpolated
surfaces where the zero-crossing topology differs.

## Fix Applied: Flood-Fill Basin Approach

Replaced `_select_enclosing_contour()` with `_select_peak_basin_contour()`:

1. Create binary mask: `laplacian < 0` (concave region)
2. Label connected components via `scipy.ndimage.label`
3. Find the component containing the peak pixel
4. Extract the boundary of that component via `find_contours(mask, 0.5)`
5. Select the largest contour (outer boundary of the basin)

This works because the peak is guaranteed to be in a negative-Laplacian cell,
and the boundary of that connected region IS the inflection contour —
regardless of which "side" of the contour the peak sits on geometrically.

**Result after fix:**

| Gesture | Before | After |
|---------|--------|-------|
| `all` | null | **boundary found** |
| `tap` | null | **null** (peak basin contour failed) |
| `stroke_proximal` | null | **boundary found** |
| `stroke_distal` | null | **boundary found** |

## Remaining Issue: `tap` Gesture

`tap` at `peak_rc=(92, 64)` still returns None from
`_select_peak_basin_contour()`. Diagnostic logging was added to identify which
check fails (peak not in negative Laplacian? component too small? contour too
short?). Debug PNG will also be saved to the output directory on failure.

**Next step:** Run the pipeline, read the `[DIAG] peak_basin:` log line for
`tap`, and inspect the debug PNG.

## Temporary Diagnostics In Place

All diagnostic code is tagged `[DIAG]` for easy removal. Files modified:

| File | Diagnostics |
|------|-------------|
| `rf_inflection_boundary.py` | `import logging` + `logger` (keep permanently), `[DIAG]` warnings at each return-None path, `_diag_save_laplacian_png()` debug PNG function, detailed logging in `_select_peak_basin_contour()` |
| `rf_population_map_pipeline.py` | `[DIAG]` pre-call grid_z statistics at both call sites (Pass 1 line ~262, Pass 2 line ~349) |
| `rf_population_map_renderer.py` | Contour color changed from green `#00ff88` to violet `#9b59b6` |

**Cleanup:** grep for `[DIAG]` across both files and remove all tagged lines.
Keep `import logging` / `logger` in `rf_inflection_boundary.py` and the
`_select_peak_basin_contour()` function. Remove `_diag_save_laplacian_png()`
and `_select_enclosing_contour()` (now unused). Remove `_diag_output_dir` /
`_diag_label` parameters from `compute_inflection_boundary()`.

## Files Involved

| File | Role |
|------|------|
| `code/src/analysis/receptive_field_mapping/rf_inflection_boundary.py` | Core computation — `compute_inflection_boundary()`, new `_select_peak_basin_contour()` |
| `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py` | Call sites (Pass 1 line ~270, Pass 2 line ~355) |
| `code/src/analysis/receptive_field_mapping/rf_population_map_renderer.py` | `_draw_inflection_boundary()` — contour color |
| `code/tests/test_rf_inflection_boundary.py` | Synthetic unit tests (23/23 pass with new approach) |
| `configs/analyse_workflow_processing_dag.yaml` | `inflection_sigma: 2.0` |
