# Plan: UV-to-mm Conversion for RF Shift Distances and PCA Axes

**Created:** 2026-06-08 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/uv-to-mm-conversion`

---

## Overview

**What:** Convert all SLIM UV-parameter-space distances, offsets, and shape metrics
to physical millimetres in the two downstream comparison pipelines.
**Why:** The SLIM parameterization constrains the mesh boundary to a unit circle,
placing UV coordinates in an arbitrary scale — not mm. Cross-session distances and
PCA axes reported in UV units are not physically interpretable.
**How:** Apply the existing `compute_uv_to_mm_scale()` function (median 3D/UV
edge-length ratio) per session, multiply all UV-space outputs by the resulting
scalar, and rename columns/labels to `_mm`.

## Problem Statement

- The open-issues doc (issue #2) flags that centroid- and hotspot-shift distances are
  "in SLIM UV parameter space, not physical mm — physical-mm conversion pending."
- PCA semi-axis lengths (`pca_major_uv`, `pca_minor_uv`) in the boundary comparison
  summary CSV are also in UV space, making them uninterpretable for the paper.
- A conversion function already exists (`compute_uv_to_mm_scale()` in
  `rf_population_map_renderer.py:509-553`) but is only used in one renderer — it has
  never been applied to the comparison pipelines.

## Goals

### In Scope
1. Convert centroid/hotspot offset vectors and scalar distances to mm in the
   proximal-distal center pipeline
2. Convert PCA semi-axis lengths, UV area, and UV perimeter to mm in the session
   boundary comparison pipeline
3. Rename CSV columns from `_uv` to `_mm` suffixes; add `uv_to_mm_scale` column for
   traceability in both CSVs
4. Update aggregate plot axis labels to show `(mm)` units

### Out of Scope
- Moving `compute_uv_to_mm_scale()` to a utility module — no new dependency created,
  unnecessary refactor
- Correcting `analysis/CLAUDE.md` coordinate spaces section and
  `note-analysis-pipeline-coordinate-spaces.md` (both claim SLIM UV is in mm) —
  documentation-only follow-up

## Success Criteria

- [ ] Proximal-distal summary CSV reports distances/offsets in mm with `_mm` suffixes
- [ ] Boundary comparison summary CSV reports PCA axes, area, perimeter in mm
- [ ] Both CSVs include per-session `uv_to_mm_scale` factor
- [ ] Aggregate scatter plot axes read `(mm)`
- [ ] Boundary metric bar-chart panels show `_mm` labels
- [ ] Both pipelines run end-to-end without error on existing NPZ data
- [ ] Raw UV centroid/hotspot absolute coordinates remain unchanged in CSV

---

## Technical Design

### Approach

Compute `uv_to_mm_scale` per session from the mesh data already stored in the NPZ
(`forearm_uv`, `forearm_V`, `forearm_faces`). Multiply all UV-space offset vectors
and scalar distances by this factor before storing in the CSV and passing to
renderers. The single-scalar approximation is consistent with how
`compute_uv_to_mm_scale()` is already used in `render_population_rf_circular_crop()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-session scalar (median edge ratio) | Simple, consistent with existing usage, one import | Isotropic approximation | Chosen |
| Per-edge anisotropic tensor | More accurate for distorted parameterizations | Complex, SLIM minimizes distortion so benefit is marginal | Rejected |
| Pre-store scale in NPZ during `spatial_extract_boundaries` | Avoids recomputing at read time | Requires NPZ format change + re-run of upstream pipeline | Rejected |

### Architecture Changes

No new modules or classes. Two existing pipeline files gain an import of
`compute_uv_to_mm_scale` from its current location in `rf_population_map_renderer`.
The proximal-distal pipeline already imports from that module
(`compute_standalone_figwidth`), so no new cross-package dependency is created.

### Key function

`compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces) -> float`
(`rf_population_map_renderer.py:509-553`): computes the median ratio of 3D edge
length (mm) to UV edge length across all unique mesh edges. Returns mm-per-UV-unit.

---

## Implementation Plan

### Phase 1: Proximal-Distal Center Pipeline
**Goal:** Apply mm conversion to all distances/offsets in the proximal-distal center
comparison pipeline.
**Started:** 2026-06-08
**Completed:** 2026-06-08

- [x] Add `compute_uv_to_mm_scale` to the existing import from `rf_population_map_renderer`
- [x] Load `forearm_V` and `forearm_faces` from NPZ; move `forearm_uv` loading up to
      join them (remove duplicate at current line 151)
- [x] Compute `uv_to_mm = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)`
- [x] Multiply offset vectors by `uv_to_mm`: `offset_proximal`, `offset_distal`,
      `hotspot_offset_proximal`, `hotspot_offset_distal`
- [x] Multiply scalar distances by `uv_to_mm` and rename variables:
      `dist_uv` -> `dist_mm`, `hotspot_dist_uv` -> `hotspot_dist_mm`,
      `centroid_hotspot_distance_uv_stroke` -> `centroid_hotspot_distance_mm_stroke`
- [x] Update `valid_data` dict keys to match new variable names
- [x] Rename CSV column keys: offsets get `_mm` suffix, distances get `_mm` replacing
      `_uv`, add `uv_to_mm_scale`
- [x] Update `df.astype()` block to match new column names
- [x] Rename helpers: `_compute_aggregate_uv_limits` -> `_compute_aggregate_mm_limits`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — all changes above

**Dependencies:** None

### Phase 2: Boundary Comparison Pipeline
**Goal:** Convert PCA semi-axis lengths and UV area/perimeter to mm.
**Started:** 2026-06-08
**Completed:** 2026-06-08

- [x] Add import for `compute_uv_to_mm_scale` from `rf_population_map_renderer`
- [x] In `_load_boundary_metrics_from_npz()`: load `forearm_uv`, `forearm_V`,
      `forearm_faces` from NPZ once before the gesture loop
- [x] Compute `uv_to_mm = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)`
- [x] Multiply `pca_major_uv` and `pca_minor_uv` by `uv_to_mm` -> `pca_major_mm`,
      `pca_minor_mm`; `pca_aspect_ratio` unchanged (scale-invariant)
- [x] Multiply `area_uv` by `uv_to_mm**2` -> `area_uv_mm2`; `perimeter_uv` by
      `uv_to_mm` -> `perimeter_uv_mm`
- [x] Add `uv_to_mm_scale` to row dicts (both `has_boundary` and NaN branches)
- [x] Update `PANEL_METRICS` list: `pca_major_uv` -> `pca_major_mm`,
      `pca_minor_uv` -> `pca_minor_mm`
- [x] Update `df.astype()` block in `_build_summary_dataframe()` to match new names

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py` — all changes above

**Dependencies:** None (independent of Phase 1)

### Phase 3: Renderer Labels
**Goal:** Update axis labels to reflect mm units.
**Started:** 2026-06-08
**Completed:** 2026-06-08

- [x] `render_proximal_distal_aggregate`: axis labels -> `'... (mm)'`
- [x] `render_proximal_distal_hotspot_aggregate`: axis labels -> `'... (mm)'`
- [x] Rename `uv_limits` parameter to `mm_limits` in both aggregate functions
      (2 signatures + 2 call sites in pipeline)
- [x] Per-session heatmap labels (`'U'`/`'V'`) stay unchanged — raw UV overlay

Note: `render_boundary_metric_panels` auto-derives bar-chart titles from the column
name, so renaming in `PANEL_METRICS` updates chart labels automatically.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py` — label and parameter renames

**Dependencies:** Phase 1 (call sites for `mm_limits` parameter)

---

## Testing Plan

### Manual Verification
- [ ] Run `spatial_compare_rf_centers` with `force_processing=True`
- [ ] Inspect `rf_center_proximal_distal_summary.csv`: `uv_to_mm_scale` present,
      offset/distance columns have `_mm` suffix, values in plausible mm range
- [ ] Open aggregate PNGs: axis labels show `(mm)`
- [ ] Run `spatial_compare_boundaries` with `force_processing=True`
- [ ] Inspect `session_rf_boundary_summary.csv`: `pca_major_mm`, `pca_minor_mm`,
      `area_uv_mm2`, `perimeter_uv_mm`, `uv_to_mm_scale` present
- [ ] Open metric panel PNGs: PCA bar-chart titles show `_mm`
- [ ] Spot-check one session: `old_uv_value * uv_to_mm_scale == new_mm_value`

### Edge Cases
- [ ] Session with degenerate UV: `compute_uv_to_mm_scale` raises `ValueError` —
      pipeline crashes loudly (fail-fast)
- [ ] Sessions missing hotspot keys: existing NaN degradation path unchanged

---

## Documentation Plan

- [ ] Update `docs/article-scoping/06-open-issues.md` issue #2 to note conversion is
      now applied
- [ ] No CLAUDE.md changes needed (coordinate-space doc correction is out of scope)

---

## Rollback Plan

1. `git revert` the merge commit — only 4 files modified, no data migrations
2. Delete sentinel files (`rf_center_proximal_distal_done.json`,
   `session_rf_boundary_comparison_done.json`) from each `iff_*` output directory

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| UV parameterization highly anisotropic for some sessions | Low | Low | SLIM minimizes distortion; single scalar is consistent with existing `circular_crop` usage |
| Downstream notebooks read old CSV column names | Med | Low | No known in-repo consumers; external notebooks must update column refs |
| Scale factor magnitude unexpectedly large/small | Low | Med | Log scale factor per session for sanity checking |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~30 min | None |
| Phase 2 | ~20 min | None |
| Phase 3 | ~10 min | Phase 1 |

---

## References

- Open issue: `docs/article-scoping/06-open-issues.md` (issue #2, ST15-01 outlier)
- Scale function: `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py:509-553`
- SLIM boundary constraint: `code/src/analysis/receptive_field_mapping/surface/slim_helpers.py:1041-1043`
- Knowledge base: `note-analysis-pipeline-coordinate-spaces.md`, `note-somatosensory-units-and-calculations.md`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py
- docs/development/plans/active/uv-to-mm-conversion.md
