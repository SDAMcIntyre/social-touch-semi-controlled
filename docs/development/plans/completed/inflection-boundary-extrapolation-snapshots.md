# Plan: Inflection Boundary — Border Extrapolation + Step Snapshots

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/reorganize-receptive-field-mapping`
**Branch:** `feature/inflection-boundary-extrapolation-snapshots`

---

## Overview

The Laplacian computation in `rf_inflection_boundary.py` uses a 3×3 finite-difference kernel (`scipy.ndimage.laplace`) that needs valid neighbors in all directions. The current approach handles NaN boundaries by discarding a 2-pixel border (5×5 dilation), shrinking the valid Laplacian domain — hotspots near the edge of the forearm-shaped valid region get their Laplacian NaN'd out, causing the basin flood-fill to fail or produce truncated contours. This plan replaces the dilation with nearest-neighbor extrapolation and adds per-step snapshot output for visual inspection.

## Problem Statement

On real forearm data (70–81% NaN), the 2-pixel NaN dilation in `_compute_masked_laplacian` erodes the valid domain inward from every NaN edge. A peak within ~4 pixels of the data boundary either has its Laplacian masked as NaN or its negative-Laplacian basin truncated so severely that `_select_peak_basin_contour` returns `None`. This is a systematic blind spot for border-adjacent receptive fields.

Additionally, the algorithm operates as a black box — there is no way to inspect intermediate results (normalized Gaussian, extrapolated field, Laplacian, basin mask) to diagnose why a particular gesture fails or to validate that the pipeline is behaving correctly.

## Goals

### In Scope
1. Replace NaN dilation with nearest-neighbor extrapolation so the Laplacian domain extends to the original data boundary
2. Generate a per-step 2×3 snapshot PNG at every `compute_inflection_boundary` call (when output dir is provided)
3. Clean up dead code and diagnostic artifacts from the previous investigation
4. Fix two bugs found during code review (perimeter closing segment, docstring)

### Out of Scope
- Diagnosing the `tap` gesture failure (separate investigation; extrapolation may help but is not targeted at it)
- Changing the basin flood-fill algorithm itself (`_select_peak_basin_contour`)
- Modifying the population RF map renderer (`rf_population_map_renderer.py`) beyond removing `[DIAG]` lines
- Arc-length-weighted PCA or other metric refinements

## Success Criteria

- [ ] `_compute_masked_laplacian` uses `distance_transform_edt` extrapolation instead of `binary_dilation`
- [ ] Valid Laplacian domain extends to the original NaN boundary (no 2-pixel erosion)
- [ ] Snapshot PNG (2×3 panels) is generated for every call when `snapshot_dir` is provided
- [ ] Snapshot shows: input, normalized Gaussian, extrapolated field, Laplacian, basin mask, final contour
- [ ] All dead code removed (`_select_enclosing_contour`, `_contour_area_pixels`, `_is_contour_closed`, `_diag_save_laplacian_png`)
- [ ] All `[DIAG]` tags replaced with permanent log messages at appropriate levels
- [ ] `_compute_polygon_perimeter` includes the closing segment
- [ ] `_sample_grid_along_contour` docstring no longer claims NaN exclusion
- [ ] New tests pass for partial-NaN grids and border peaks
- [ ] Existing 23 tests still pass

---

## Technical Design

### Approach

Replace the conservative NaN-dilation strategy with nearest-neighbor extrapolation:

1. **Normalized Gaussian convolution** (keep as-is) — replaces NaN with 0, smooths both values and weight mask, divides to normalize
2. **Nearest-neighbor extrapolation** (new) — fill remaining NaN cells in the normalized result with the nearest valid value via `scipy.ndimage.distance_transform_edt` with `return_indices=True`
3. **Laplacian** on the fully populated grid — no dilation needed since all cells have valid neighbors
4. **Re-mask** only the original NaN cells (not a dilated region)

The extrapolated values are flat extensions (constant from the nearest valid cell), so the Laplacian in the transition zone naturally tends toward zero — no finite-difference artifacts.

Expose intermediate arrays via a `_LaplacianResult` dataclass so the orchestrator can pass them to the snapshot renderer without recomputing anything.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Nearest-neighbor extrapolation | Zero artifacts, extends domain fully, simple | Boundary contour shape reflects flat extension, not true curvature | **Chosen** — flat extension is physically reasonable (no data = no curvature) |
| Scale dilation with sigma | Minimal code change | Still erodes domain; just less so for small sigma | Rejected — doesn't solve the fundamental problem |
| Mirror/reflect padding | Better curvature continuity at boundary | Complex to implement for irregular NaN shapes; assumes symmetric data | Rejected — forearm NaN shapes are irregular |
| Inpainting (e.g. Navier-Stokes) | Smooth interpolation into NaN region | Heavy dependency, slow, over-engineered for this use case | Rejected |

### Architecture Changes

**New dataclass** in `rf_inflection_boundary.py`:
```python
@dataclass
class _LaplacianResult:
    nan_mask: np.ndarray
    normalized: np.ndarray
    extrapolated: np.ndarray
    extrapolation_mask: np.ndarray
    laplacian: np.ndarray
```

**Modified function signatures:**
- `_compute_masked_laplacian(grid_z, gaussian_sigma) → _LaplacianResult` (was `→ np.ndarray`)
- `_select_peak_basin_contour(laplacian, peak_rc) → tuple[np.ndarray | None, np.ndarray | None]` (returns contour + component mask)
- `compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma, snapshot_dir, snapshot_label)` — replaces `_diag_output_dir`/`_diag_label`

**New function:**
- `_save_inflection_snapshot(...)` — 2×3 panel matplotlib renderer

**Deleted functions:**
- `_select_enclosing_contour`, `_contour_area_pixels`, `_is_contour_closed`, `_diag_save_laplacian_png`

---

## Implementation Plan

### Phase 1: Core Fix + Cleanup
**Goal:** Replace dilation with extrapolation, expose intermediates, clean up dead code and bugs
**Started:** 2026-05-20
**Completed:** 2026-05-20

**Tasks:**
- [x] 1.1 — Add `_LaplacianResult` dataclass
- [x] 1.2 — Rewrite `_compute_masked_laplacian` to use `distance_transform_edt` extrapolation instead of `binary_dilation`, return `_LaplacianResult`
- [x] 1.3 — Modify `_select_peak_basin_contour` to return `(contour, component_mask)` tuple
- [x] 1.4 — Update `compute_inflection_boundary` orchestrator to use new return types and rename params (`snapshot_dir`/`snapshot_label`)
- [x] 1.5 — Delete dead functions: `_select_enclosing_contour`, `_contour_area_pixels`, `_is_contour_closed`
- [x] 1.6 — Delete `_diag_save_laplacian_png`
- [x] 1.7 — Replace all `[DIAG]` log messages with permanent ones (`.debug` for verbose, `.warning` for failures, `.info` for success)
- [x] 1.8 — Fix imports: move `logging` to stdlib group, remove `binary_dilation` and `Path` from matplotlib, add `distance_transform_edt`
- [x] 1.9 — Change default `gaussian_sigma` from `1.0` to `2.0`
- [x] 1.10 — Fix `_compute_polygon_perimeter`: add closing segment `+ np.linalg.norm(contour_uv[-1] - contour_uv[0])`
- [x] 1.11 — Fix `_sample_grid_along_contour` docstring: remove false NaN-exclusion claim

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_inflection_boundary.py` — all changes above

**Dependencies:** None

### Phase 2: Snapshot Renderer + Pipeline Wiring
**Goal:** Add per-step snapshot output and wire into pipeline call sites
**Started:** 2026-05-20
**Completed:** 2026-05-20

**Tasks:**
- [x] 2.1 — Create `_save_inflection_snapshot()` function — 2×3 panel figure:
  - (0,0) Input `grid_z` — jet colormap, NaN regions grey
  - (0,1) Normalized Gaussian — jet colormap
  - (0,2) Extrapolated field — jet colormap, extrapolation region outlined (dashed contour at boundary)
  - (1,0) Laplacian — RdBu symmetric colormap, zero-crossing contour overlaid
  - (1,1) Basin mask — binary, peak marked red ×, selected component highlighted
  - (1,2) Final — `grid_z` with inflection contour overlay (violet) + peak marker
- [x] 2.2 — Call `_save_inflection_snapshot` from `compute_inflection_boundary` on **both** success and failure paths when `snapshot_dir is not None`
- [x] 2.3 — Update `rf_population_map_pipeline.py` Pass 1 call site: replace `_diag_output_dir`/`_diag_label` with `snapshot_dir`/`snapshot_label`
- [x] 2.4 — Update `rf_population_map_pipeline.py` Pass 2 call site: same replacement
- [x] 2.5 — Remove any remaining `[DIAG]` lines from `rf_population_map_pipeline.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_inflection_boundary.py` — snapshot renderer, orchestrator wiring
- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py` — call-site parameter updates

**Dependencies:** Phase 1

### Phase 3: Tests
**Goal:** Add NaN-path tests and fix existing perimeter test
**Started:** 2026-05-20
**Completed:** 2026-05-20

**Tasks:**
- [x] 3.1 — New `TestPartialNaNGrid` class: circular NaN mask around a centered Gaussian peak (simulating forearm-shaped valid region at ~70% NaN). Assert boundary is found.
- [x] 3.2 — New `TestBorderPeak` class: Gaussian peak positioned close to (but not at) the NaN boundary edge. Assert boundary is found with the extrapolation fix.
- [x] 3.3 — Fix `test_square_perimeter`: change test polygon to open form `[[0,0],[1,0],[1,1],[0,1]]` (remove trailing `[0,0]`) so the test exercises the closing-segment fix
- [x] 3.4 — Run full test suite, confirm all pass

**Files Modified:**
- `code/tests/test_rf_inflection_boundary.py` — new test classes, fix perimeter test

**Dependencies:** Phase 1 (Phase 2 is not required for tests)

---

## Testing Plan

### Unit Tests
- [x] `TestPartialNaNGrid` — Gaussian on 100×100 grid with circular mask (~70% NaN), boundary is not None, contour has valid shape
- [x] `TestBorderPeak` — Peak at ~5px from NaN boundary, boundary is found (would have been None with old 2px dilation)
- [x] `test_square_perimeter` (fixed) — Open polygon `[0,0]→[1,0]→[1,1]→[0,1]` has perimeter 4.0 (closing segment included)
- [x] All 23 existing tests pass unchanged (now 28 total)

### Manual Verification
- [ ] Run pipeline on a session: `python code/scripts/analysis_workflow.py` with `visualize_population_rf_maps` enabled
- [ ] Inspect `inflection_steps_*.png` files in `4_analysed/population_rf_maps/{session_id}/`
- [ ] Verify row 0 col 2 (extrapolated field) extends smoothly beyond NaN boundary
- [ ] Verify row 1 col 0 (Laplacian) has no artifacts at former NaN boundary
- [ ] Verify row 1 col 2 (final contour) extends closer to data edge than before
- [ ] Compare boundary detection results for all 4 gesture types against previous run

### Edge Cases
- [ ] All-NaN grid → returns None, no snapshot generated (no `snapshot_dir`)
- [ ] Flat surface → returns None, snapshot shows flat Laplacian
- [ ] Peak at grid edge (within 2px) → returns None early, snapshot still generated if dir provided
- [ ] Fully populated grid (no NaN) → extrapolation is a no-op, behavior unchanged

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/investigation-rf-inflection-boundary-null.md` — mark cleanup as completed, document extrapolation fix
- [ ] Update `code/src/analysis/CLAUDE.md` if snapshot output pattern needs documenting

---

## Rollback Plan

1. All changes are in a single feature branch — `git revert` the merge commit
2. No data format changes — sentinel JSON and NPZ structure unchanged
3. No config changes — `inflection_sigma` parameter meaning is unchanged
4. Snapshot PNGs are additive output — removing the branch simply stops generating them

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Extrapolated values affect basin shape at boundary | Medium | Low | Flat extension produces near-zero Laplacian in extrapolated zone — basin stops at real data boundary naturally |
| Snapshot rendering slows pipeline | Low | Low | Matplotlib Agg backend, 150 dpi, only when `snapshot_dir` is provided; ~0.5s per call |
| `tap` gesture still fails after fix | Medium | Low | Out of scope — extrapolation may help but the `tap` failure is a separate investigation |
| Perimeter fix changes circularity values | High | Low | Expected — previous values were slightly wrong; no downstream decisions depend on exact circularity |

---

## References

- Knowledge base: `docs/development/knowledge-base/investigation-rf-inflection-boundary-null.md`
- Parent plan: `docs/development/plans/active/population-rf-inflection-boundary.md`
- Analysis CLAUDE.md: `code/src/analysis/CLAUDE.md` (coordinate spaces, pipeline architecture)
