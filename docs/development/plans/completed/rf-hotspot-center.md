# Plan: RF Hotspot Center (Peak UV) for Proximal-Distal Comparison

**Date:** 2026-05-25
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-26 19:08
**Base Branch:** `feature/rf-center-proximal-distal-comparison`
**Branch:** `feature/rf-hotspot-center`

---

## Overview

Add the **hotspot center** (peak/summit of the interpolated response field) as
a new metric alongside the existing **boundary centroid** (geometric center of
the inflection contour).  The hotspot is where the neuron responds most
intensely and can be spatially shifted from the boundary centroid when the RF
is asymmetric.  This metric is added upstream (to `InflectionBoundary` and the
population NPZ) and consumed downstream in the proximal-distal center
comparison pipeline.

## Problem Statement

The current proximal-distal comparison only uses the boundary centroid — the
geometric center of the Laplacian zero-crossing contour.  While this
captures the RF's spatial extent, it does not capture the **location of
maximum response intensity** (the hotspot).  For asymmetric or skewed
response fields, these two centers diverge, and the hotspot location may be
the more biologically relevant measure of spatial tuning.

The peak is already computed internally (`_find_peak_location` in
`rf_inflection_boundary.py`, line 118) as a flood-fill seed, but its UV
coordinates are never stored or exported.

## Goals

### In Scope

1. Compute hotspot UV coordinates from the existing grid-z peak and store
   them in the `InflectionBoundary` dataclass.
2. Persist hotspot UV and back-projected XYZ in the per-session population
   response field NPZ.
3. Produce per-session hotspot-marked heatmap PNGs (separate from centroid
   PNGs).
4. Produce a cross-session aggregate scatter plot for hotspot offsets
   (separate from the centroid aggregate plot).
5. Add hotspot columns to the summary CSV alongside centroid columns.

### Out of Scope

- Sub-pixel quadratic refinement of the peak location — grid-cell resolution
  on the 150x150 grid (~0.002 UV units per pixel) is sufficient.
- Combined figures showing both centroid and hotspot on the same heatmap —
  separate figures were chosen for clarity.
- Statistical testing on hotspot vs centroid shifts — future work.
- Hotspot computation for cluster-level RF metrics (`rf_metrics.py`) — this
  plan targets population-level response fields only.

## Success Criteria

- [ ] `InflectionBoundary.peak_uv` field exists and is populated for all
      successful boundary extractions.
- [ ] Per-session NPZ files contain `boundary_peak_uv_{gtype}` and
      `boundary_peak_xyz_{gtype}` keys after re-running upstream with
      `force_processing: true`.
- [ ] Per-session directories contain `_rf_hotspot_{gtype}_{cmap}.png` PNGs
      with a red `*` marker at the visible peak of the heatmap.
- [ ] `rf_hotspot_proximal_distal_aggregate.png` is produced showing hotspot
      offset vectors per session.
- [ ] Summary CSV includes hotspot coordinate, offset, and distance columns.
- [ ] Sessions with pre-existing NPZ files (missing hotspot keys) are handled
      gracefully — centroid outputs are unaffected, hotspot columns show NaN.
- [ ] All existing tests pass; new `peak_uv` tests pass.

---

## Technical Design

### Approach

Convert the existing `peak_rc` grid indices (already computed inside
`compute_inflection_boundary`) to UV coordinates using `_contour_pixels_to_uv`,
add the result as `peak_uv` to the `InflectionBoundary` dataclass, and store it
in the NPZ via the same serialization path used for `centroid_uv`.  Downstream,
the comparison pipeline loads hotspot UV from NPZ and produces separate render
outputs mirroring the existing centroid outputs.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Upstream: add `peak_uv` to `InflectionBoundary` + NPZ | Computed once, available to all consumers, consistent with centroid pattern | Requires re-running upstream to regenerate NPZ | **Chosen** |
| Downstream: compute peak from `grid_z` in comparison pipeline | No upstream changes | Duplicates peak-finding logic, not available to other consumers | Rejected |
| Sub-pixel quadratic refinement around peak | Slightly more precise | Grid resolution already ~0.002 UV/pixel; complexity not justified | Rejected |

### Architecture Changes

No new modules.  Changes are additive within existing files:

```
metrics/rf_inflection_boundary.py           ← add peak_uv field + serialization
pipelines/rf_population_response_field_pipeline.py  ← store peak in NPZ
rendering/rf_center_comparison_renderer.py  ← add peak_uv param + hotspot aggregate fn
pipelines/rf_proximal_distal_center_pipeline.py     ← load/render/CSV hotspot data
```

Existing function signatures gain optional parameters only — no breaking changes.

---

## Implementation Plan

### Phase 1: Upstream Metric
**Goal:** Add `peak_uv` to `InflectionBoundary` and its serialization.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 1.1 — Add `peak_uv: tuple[float, float]` field to the
      `InflectionBoundary` dataclass after `centroid_uv`.
- [x] Task 1.2 — In `compute_inflection_boundary()`, convert `peak_rc` to
      UV via `_contour_pixels_to_uv` and pass to the constructor.
- [x] Task 1.3 — Add `"peak_uv"` entry to `inflection_boundary_to_dict()`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
  — add field, UV conversion, serialization

**Dependencies:** None

### Phase 2: Upstream Storage
**Goal:** Persist hotspot UV and XYZ in the population response field NPZ.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 2.1 — In `_save_response_fields_npz()`, store
      `boundary_peak_uv_{gtype}` from `boundary.peak_uv`.
- [x] Task 2.2 — Back-project peak UV to 3D via `uv_points_to_xyz()` and
      store as `boundary_peak_xyz_{gtype}`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
  — add two NPZ keys per gesture type

**Dependencies:** Phase 1

### Phase 3: Renderer Updates
**Goal:** Support hotspot marker rendering and a hotspot aggregate plot.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 3.1 — Add optional `peak_uv: np.ndarray | None` parameter to
      `render_center_marked_heatmap()`.  When provided, plot a red `*`
      marker (markersize=10, markeredgewidth=1.5, zorder=8).
- [x] Task 3.2 — Add `render_proximal_distal_hotspot_aggregate()` function
      mirroring `render_proximal_distal_aggregate()` but with title
      "Proximal vs Distal RF Hotspot Offsets" and axes
      "ΔU/ΔV from stroke hotspot".

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py`
  — add optional param + new aggregate function

**Dependencies:** None (can proceed in parallel with Phase 2)

### Phase 4: Pipeline Consumer
**Goal:** Load hotspot from NPZ, render hotspot PNGs, write CSV + aggregate.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 4.1 — In Pass 1 (loading), attempt to load
      `boundary_peak_uv_{stroke,stroke_proximal,stroke_distal}`.  If any
      key is missing, set `hotspot_available = False` for that session (log
      warning), continue with centroid data.
- [x] Task 4.2 — Compute hotspot offsets relative to `hotspot_stroke`:
      `hotspot_offset_proximal`, `hotspot_offset_distal`,
      `hotspot_dist_uv`, `centroid_hotspot_distance_uv_stroke`.
- [x] Task 4.3 — In Pass 2 (per-session rendering), for each gesture type
      with hotspot data, render
      `{session}_rf_hotspot_{gtype}_{cmap}.png` using `peak_uv` parameter.
- [x] Task 4.4 — In Pass 3 (cross-session), add hotspot columns to summary
      CSV: `hotspot_u_*`, `hotspot_v_*`, `hotspot_offset_u_*`,
      `hotspot_offset_v_*`, `hotspot_proximal_distal_distance_uv`,
      `centroid_hotspot_distance_uv_stroke`.  Sessions without hotspot data
      get NaN.
- [x] Task 4.5 — Render `rf_hotspot_proximal_distal_aggregate.png` using
      only sessions with hotspot data.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py`
  — load hotspot, compute offsets, render, write CSV + aggregate

**Dependencies:** Phases 2 and 3

### Phase 5: Tests
**Goal:** Verify `peak_uv` is correct and serializable.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 5.1 — `TestCircularGaussian::test_peak_uv_near_grid_center` —
      centered Gaussian peak should be near (0.5, 0.5).
- [x] Task 5.2 — `TestCircularGaussian::test_peak_uv_type` — verify
      `peak_uv` is a tuple of two floats.
- [x] Task 5.3 — `TestSerialization::test_expected_keys_present` — add
      `"peak_uv"` to the expected keys set.
- [x] Task 5.4 — `TestEllipticalGaussian::test_peak_uv_near_centroid` —
      for a symmetric Gaussian, peak and centroid should be close.

**Files Modified:**
- `code/tests/test_rf_inflection_boundary.py` — add 4 test methods, update 1

**Dependencies:** Phase 1

### Phase 6: Documentation
**Goal:** Update CLAUDE.md and the parent plan to reflect the hotspot addition.
**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Task 6.1 — Update `code/src/analysis/CLAUDE.md`: add
      `boundary_peak_uv_{gtype}` to the NPZ field documentation; update
      the RF center comparison entry to mention hotspot center.
- [x] Task 6.2 — Update
      `docs/development/plans/active/rf-center-proximal-distal-comparison.md`:
      note that upstream modifications are now in scope for the hotspot
      center requirement.

**Files Modified:**
- `code/src/analysis/CLAUDE.md` — documentation updates
- `docs/development/plans/active/rf-center-proximal-distal-comparison.md`
  — scope update

**Dependencies:** All prior phases

---

## Testing Plan

### Unit Tests

- [ ] Centered Gaussian: `peak_uv` is near grid center (0.5, 0.5) within
      tolerance.
- [ ] `peak_uv` is a tuple of two Python floats.
- [ ] Serialization round-trip includes `"peak_uv"` key with two-element
      list.
- [ ] Symmetric elliptical Gaussian: `peak_uv` and `centroid_uv` are close
      (within 0.05).

### Integration Tests

- [ ] Re-run `extract_population_rf_response_field_boundaries` with
      `force_processing: true` on one session.  Verify NPZ contains
      `boundary_peak_uv_*` keys.
- [ ] Run `compare_rf_center_proximal_distal` with `force_processing: true`.
      Verify hotspot PNGs, hotspot aggregate plot, and CSV hotspot columns
      are all produced.

### Manual Verification

- [ ] Open a hotspot heatmap PNG — red `*` marker should sit at the visible
      peak (highest-intensity region) of the heatmap.
- [ ] Compare with the centroid heatmap — violet `+` marker may be at a
      different location if the RF is asymmetric.
- [ ] Open hotspot aggregate plot — verify per-session markers with
      connecting lines, same style as centroid aggregate.

### Edge Cases

- [ ] Pre-existing NPZ without hotspot keys — downstream pipeline logs
      warning, centroid outputs unaffected, hotspot CSV columns are NaN.
- [ ] Session where boundary detection fails (returns None) — no hotspot
      stored, session skipped in downstream, no crash.
- [ ] Single-session run — both aggregate plots render with one pair.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `peak_uv` field docs,
      update RF center comparison entry.
- [ ] Update parent plan doc — note upstream scope expansion.
- [ ] No README.md or user guide changes — task is auto-discovered by DAG
      launcher GUI.

---

## Rollback Plan

The feature is additive: one new dataclass field, two new NPZ keys per
gesture, optional renderer parameters, additional CSV columns, and one new
aggregate PNG.

1. **Rollback procedure:**
   - Revert the `peak_uv` field from `InflectionBoundary`.
   - Remove NPZ storage lines from the population pipeline.
   - Remove hotspot rendering and CSV columns from the comparison pipeline.
   - Remove the hotspot aggregate function from the renderer.
   - Revert test additions.
2. **Data considerations:**
   - Re-run upstream to regenerate NPZ files without hotspot keys (or leave
     them — extra keys are harmless).
   - Delete hotspot PNG outputs from `4_analysed/rf_center_proximal_distal/`.
   - Centroid-only CSV can be regenerated by re-running the comparison task.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing NPZ files lack hotspot keys | Certain (until re-run) | Low | Downstream treats missing keys as unavailable; centroid outputs unaffected |
| Peak at grid edge (within 2px of border) | Low | None | `compute_inflection_boundary` already returns `None` in this case; no hotspot stored |
| Hotspot and centroid nearly identical for symmetric RFs | Medium | Low | Expected behavior; the interesting signal is in asymmetric cases |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (upstream metric) | ~20 min | None |
| Phase 2 (upstream storage) | ~15 min | Phase 1 |
| Phase 3 (renderer) | ~30 min | None |
| Phase 4 (pipeline consumer) | ~45 min | Phases 2, 3 |
| Phase 5 (tests) | ~20 min | Phase 1 |
| Phase 6 (documentation) | ~10 min | All |

Total: ~2.5 hours.

---

## References

- Parent plan: `docs/development/plans/active/rf-center-proximal-distal-comparison.md`
- Peak finding: `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py::_find_peak_location`
- Grid-to-UV conversion: `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py::_contour_pixels_to_uv`
- NPZ serialization: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py::_save_response_fields_npz`
- Downstream pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py`
- Renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py`
- Knowledge base: `note-analysis-pipeline-coordinate-spaces.md`, `note-3d-to-2d-surface-projection-algorithms.md`

---
