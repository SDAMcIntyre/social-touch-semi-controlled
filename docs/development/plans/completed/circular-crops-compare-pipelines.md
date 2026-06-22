# Plan: Circular Crop PNG Outputs for Comparison Pipelines

**Date:** 2026-06-11
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-22 17:57
**Base Branch:** `feature/forearm-skin-neuron-type-coloring`
**Branch:** `feature/circular-crops-compare-pipelines`

---

## Overview

Add per-session circular crop PNG outputs to `spatial_compare_rf_centers` and
`spatial_compare_boundaries`, matching the transparent circular crops already produced
by `spatial_extract_boundaries`. Each crop is a zoomed-in `radius_mm=50.0` view centred
on the RF boundary centroid, rendered at `dpi=300` with consistent mm-per-pixel scale
across sessions.

## Problem Statement

`spatial_extract_boundaries` generates circular crop PNGs for each session (centred on
the centroid and hotspot peak), but the two downstream comparison pipelines
(`spatial_compare_rf_centers`, `spatial_compare_boundaries`) only produce full-forearm
heatmaps. This makes it impossible to visually compare zoomed RF regions side-by-side
across sessions within the comparison output directories.

## Goals

### In Scope
1. Add centroid-centred and peak-centred circular crops to `spatial_compare_rf_centers` (one per session × gesture type)
2. Add centroid-centred circular crops to `spatial_compare_boundaries` (one per session × gesture type)
3. Add `heatmap_space` / `cmap` options to `spatial_compare_boundaries` to drive the new heatmap output
4. Keep all rendering consistent with `spatial_extract_boundaries` (`radius_mm=50.0`, `dpi=300`, transparent PNG)

### Out of Scope
- Peak-centred crops for `spatial_compare_boundaries` (boundary comparison is about contour geometry, not hotspot tracking)
- Changing any existing output files or rendering in either pipeline
- Adding circular crops to any other pipeline stage

## Success Criteria

- [ ] `spatial_compare_rf_centers` session subdirs contain `{session_id}_rf_circular_centroid_{gtype}_{cmap}.png` for every gesture type with a centroid
- [ ] `spatial_compare_rf_centers` session subdirs contain `{session_id}_rf_circular_peak_{gtype}_{cmap}.png` for every gesture type with a detected peak
- [ ] `spatial_compare_boundaries` produces `circular_crops/{session_id}/{session_id}_rf_circular_centroid_{gtype}.png` for every gesture type with a boundary centroid
- [ ] All circular crops are transparent PNG, circular alpha mask, consistent scale across sessions
- [ ] All existing outputs from both pipelines are unchanged

---

## Technical Design

### Approach

Reuse `render_population_rf_circular_crop()` from
`analysis.receptive_field_mapping.rendering.rf_population_map_renderer` directly in both
pipeline modules. This function already implements the full circular clipping, UV-to-mm
scale conversion, transparent PNG export, and `bbox_inches='tight'` sizing that give
scale consistency across sessions.

All rendering parameters are fixed to match `spatial_extract_boundaries`:
`radius_mm=50.0`, `dpi=300`, `transparent=True`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse `render_population_rf_circular_crop` directly | Zero duplication; scale consistency guaranteed by same code path | None | **Chosen** |
| New dedicated renderer per pipeline | Isolated | Code duplication; scale drift risk if one is updated | Rejected |
| PIL post-processing of existing PNG outputs | No matplotlib changes | Requires loading saved PNGs; lossy; no transparency | Rejected |

### Architecture Changes

- No new modules or renderer functions.
- `_load_heatmap_rendering_data_from_npz()` private helper added to
  `rf_session_boundary_comparison_pipeline.py` — loads NPZ rendering arrays
  (`forearm_uv/V/faces`, `slim_vertex_colors`, per-gtype grids and centroids) separately
  from the existing `_load_boundary_metrics_from_npz()` which handles scalar metrics.
- `spatial_compare_boundaries_flow` gains `heatmap_space` and `cmap` parameters,
  propagated to the pipeline function and read from the DAG options lambda.

### Knowledge Base Notes Applied

- `note-circular-crop-scale-consistency.md` — confirmed that using the same `radius_mm`
  constant and the same `render_population_rf_circular_crop` path (no `figsize` override,
  `dpi=300`, `bbox_inches='tight'`) yields pixel-consistent output across sessions.

---

## Implementation Plan

### Phase 1: `spatial_compare_rf_centers` circular crops

**Goal:** Emit centroid and peak circular crops in the per-session subdirectories
alongside existing full-forearm heatmaps.

**Started:** 2026-06-11
**Completed:** 2026-06-11

- [x] Task 1.1 — In `rf_proximal_distal_center_pipeline.py`, add `'forearm_V': npz['forearm_V'].astype(np.float64)` to the `valid_data` dict (it is loaded at line 77 but currently not stored)
- [x] Task 1.2 — Extend the import from `rf_population_map_renderer` to also import `render_population_rf_circular_crop`
- [x] Task 1.3 — In Pass 2, after the two existing `render_center_marked_heatmap` calls per `gtype`, add a centroid circular crop call and a peak circular crop call (guarded by `peak_uv_gtype is not None`)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — store `forearm_V`, extend import, add two crop calls per gtype in Pass 2

**Dependencies:** None

### Phase 2: `spatial_compare_boundaries` circular crops

**Goal:** Emit centroid-centred circular crops in a new `circular_crops/` subdirectory
of the boundary comparison output.

**Started:** 2026-06-11
**Completed:** 2026-06-11

- [x] Task 2.1 — Add `heatmap_space: str = "linear"` and `cmap: str = "inferno"` parameters to `run_session_rf_boundary_comparison`
- [x] Task 2.2 — Import `render_population_rf_circular_crop` alongside the existing `compute_uv_to_mm_scale` import
- [x] Task 2.3 — Write `_load_heatmap_rendering_data_from_npz(npz_path: Path) -> dict` helper; returns `forearm_uv`, `forearm_V`, `forearm_faces`, `slim_vertex_colors`, and `per_gtype` dict with `grid_u/v/z` and `centroid_uv` per gesture type
- [x] Task 2.4 — Add circular-crops rendering block at the end of `run_session_rf_boundary_comparison` (before `_write_sentinel`): load data per session, compute global `vmin`/`vmax`, render crops into `output_dir / 'circular_crops' / session_id /`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py` — new `_load_heatmap_rendering_data_from_npz`, updated signature, new rendering block

**Dependencies:** None (independent of Phase 1)

### Phase 3: DAG config and workflow wiring

**Goal:** Expose `heatmap_space` and `cmap` for `spatial_compare_boundaries` in the
DAG config and workflow flow function.

**Started:** 2026-06-11
**Completed:** 2026-06-11

- [x] Task 3.1 — Add `heatmap_space: linear` and `cmap: inferno` under `spatial_compare_boundaries.options` in `configs/analyse_workflow_processing_dag.yaml`
- [x] Task 3.2 — Add `heatmap_space: str = "linear"` and `cmap: str = "inferno"` parameters to `spatial_compare_boundaries_flow` in `code/scripts/analysis_workflow_processing.py`; pass them to `run_session_rf_boundary_comparison`
- [x] Task 3.3 — In the DAG params lambda for `"spatial_compare_boundaries"`, read `heatmap_space` and `cmap` from `dag_handler.get_task_options("spatial_compare_boundaries")`

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — two new options under `spatial_compare_boundaries`
- `code/scripts/analysis_workflow_processing.py` — updated flow signature and params lambda

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] No new unit tests required — `render_population_rf_circular_crop` is already tested via integration; new code is straight call-site wiring

### Manual Verification
- [ ] Force-reprocess one session through `spatial_compare_rf_centers`; confirm `{session_id}_rf_circular_centroid_all_inferno.png` and sibling files appear in the session subdir
- [ ] Confirm all circular PNGs have transparent background and circular alpha mask (open in image viewer)
- [ ] Force-reprocess one session through `spatial_compare_boundaries`; confirm `circular_crops/{session_id}/` directory contains `{session_id}_rf_circular_centroid_all.png` and per-gtype siblings
- [ ] Confirm pixel dimensions of circular crops are consistent across two different sessions (same `radius_mm`)
- [ ] Confirm all pre-existing outputs from both pipelines are unchanged (full-forearm heatmaps, aggregate scatter plots, contour overlays, metric panels)

### Edge Cases
- [ ] Session NPZ missing `slim_vertex_colors` key — circular crop renders with grey background (handled by `render_population_rf_circular_crop` default)
- [ ] Session NPZ missing `boundary_centroid_uv_{gtype}` for a gesture type in `spatial_compare_boundaries` — that gtype is silently skipped via `if gdata['centroid_uv'] is None: continue`
- [ ] `peak_uv_gtype is None` for a gesture type in `spatial_compare_rf_centers` — peak crop skipped, centroid crop still rendered

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` entry for `spatial_compare_rf_centers` to mention the new circular crop outputs
- [ ] Update `code/src/analysis/CLAUDE.md` entry for `spatial_compare_boundaries` to mention `heatmap_space`/`cmap` options and the new `circular_crops/` subdir
- [ ] Update DAG YAML comment block above `spatial_compare_boundaries` to mention the new outputs

---

## Rollback Plan

All changes are additive — no existing outputs are removed or modified.

1. To revert: `git revert` the feature branch merge commit, or simply delete the new circular crop files from the output directory.
2. No data migrations; NPZ files produced by `spatial_extract_boundaries` are read-only inputs.
3. Sentinel files are untouched in both pipelines; force-reprocessing is controlled by the existing `force_processing` flag.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NPZ for older sessions missing `grid_u_{gtype}` keys | Low | Medium | `_load_heatmap_rendering_data_from_npz` skips missing keys; circular crop silently omitted for that gtype |
| `compute_uv_to_mm_scale` raises on degenerate UV mesh | Very low | High | Already raises `ValueError` with descriptive message; pipeline fails fast per convention |
| Double NPZ read per session in `spatial_compare_boundaries` adds noticeable runtime | Low | Low | NPZ files are small (≤50 MB); second read is acceptable; separation of concerns outweighs the cost |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-circular-crop-scale-consistency.md`
- Source of truth renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — `render_population_rf_circular_crop` (line 603)
- Reference implementation: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` (lines 609–668)

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/circular-crops-compare-pipelines.md
