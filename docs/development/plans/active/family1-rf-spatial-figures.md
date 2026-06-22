# Plan: Family 1 — RF Spatial Figures Infrastructure Gaps

**Date:** 2026-06-15
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-contour-color-and-tuning-enhancements`
**Branch:** `feature/family1-rf-spatial-figures`

---

## Overview

The article figure specification (`docs/article-scoping/07-figure-specification.md`) defines Family 1 as RF spatial feature figures across a 4-level hierarchy (L0-L3). A codebase readiness assessment (`.claude/plans/analyse-docs-article-scoping-07-figure-s-optimized-grove.md`) identified that L0, L2, and L3 2D grids are fully ready. Three gaps remain to complete Family 1: (1) centroid shift pairwise distance for tap vs stroke, (2) multi-fit polynomial degree support in tuning curves, and (3) binned RF spatial tuning curves for RF metrics as Y-axis + area dismissal figure.

## Problem Statement

The current pipeline computes all 6 core RF spatial metrics (area, circularity, PCA ratio, PCA angle, centroid, perimeter) and supports gesture-type subsetting (all/tap/stroke/stroke_proximal/stroke_distal). However, three capabilities are missing:

1. **Centroid shift as a metric column:** Centroids are stored as separate XYZ coordinates per gesture type in NPZ files, but no pairwise distance column exists in the summary DataFrame. The cross-neuron bar charts cannot display centroid shift.
2. **Single polynomial degree:** The `stimulus_response_tuning` pipeline accepts `fit_degree` as a single int. The spec requires overlaying fits of degree 1-4 simultaneously to compare R² and select the best fit per neuron.
3. **No RF spatial metrics in tuning curves:** The tuning curve Y-axis only supports IFF metrics (mean IFF, max IFF, spike count). RF boundary metrics (area, circularity) cannot be plotted against stimulus parameters (velocity, depth, contact_area). This also blocks the area dismissal figure.

## Goals

### In Scope
1. Add centroid shift pairwise distance columns to the boundary comparison summary DataFrame and bar charts
2. Extend `fit_degree` to accept a list of integers, overlay all fits on the same axes with R² annotations
3. Build a binned RF spatial tuning pipeline: bin touches by stimulus parameter, compute population RF heatmap per bin, extract inflection boundary metrics, render scatter + fit
4. Produce the area dismissal figure using the binned approach

### Out of Scope
- Family 2 (stimulus sensitivity / temporal features) — separate plan
- Cross-neuron comparison renderer with Okabe-Ito grouped strip charts — existing `render_boundary_metric_panels()` already supports neuron-type coloring; refinements for publication layout are styling, not infrastructure
- Response dynamics (IFF trace alignment) — needs new infrastructure, separate plan
- Centroid unit conversion (UV → mm) — already handled by `compute_uv_to_mm_scale()`; the open question is editorial, not code

## Success Criteria

- [x] Summary CSV from `spatial_compare_boundaries` includes `centroid_shift_tap_vs_stroke_mm` and `centroid_shift_proximal_vs_distal_mm` columns
- [x] Bar chart panels include centroid shift metric
- [x] `stimulus_response_tuning` with `fit_degree: [1, 2, 3, 4]` renders all 4 polynomial fits on the same scatter plot with distinct line styles and an R² legend
- [x] New `rf_spatial_tuning_pipeline.py` produces scatter plots of velocity → RF area, velocity → circularity (and depth, contact_area variants) using binned population heatmaps
- [x] Area dismissal figure: contact_area → RF area scatter across all neurons shows flat/non-significant relationship with Spearman p-value annotation
- [ ] All existing tests pass; no regressions in `spatial_extract_boundaries`, `spatial_compare_boundaries`, or `stimulus_response_tuning` outputs

---

## Technical Design

### Approach

Three independent modifications, each building on existing infrastructure:

1. **Centroid shift:** Post-process the existing summary DataFrame in `rf_session_boundary_comparison_pipeline.py` to compute pairwise Euclidean distances between gesture-type centroids. Add the columns to `PANEL_METRICS` for rendering.

2. **Multi-fit:** Parse `fit_degree` as either `int` or `list[int]`. For each degree, call the existing `_fit_polynomial()` and collect results. Modify renderer to loop over fits, drawing each with a distinct color/linestyle from a predefined palette, and add an R² annotation box.

3. **Binned RF spatial tuning:** New pipeline that reuses `compute_rf_heatmap()` (from `rf_population_heatmap.py`) and `compute_inflection_boundary()` (from `rf_inflection_boundary.py`). Binning reuses the touch-level feature CSVs from `stimulus_extract_features`. Per bin: filter touches → compute population heatmap → extract inflection boundary → collect (bin_center, metric_value). Render as scatter + polynomial fit(s).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Binned RF heatmaps (Option C)** | Biologically meaningful ("RF shape at velocity X"), reuses population heatmap + boundary infrastructure | Fewer data points (~10-25 per session), depends on bin count | Chosen |
| Per-touch RF metrics (Option A) | One dot per touch, consistent with IFF tuning curves | A single touch doesn't have a meaningful "RF area"; computationally expensive | Rejected — not biologically valid |
| Grid-cell marginals (Option B) | Already computed in velocity_depth_2d | Loses actual RF contour shape; averages pre-computed cell metrics, confounds velocity and depth | Rejected — less interpretable |

### Architecture Changes

No new modules or classes beyond:
- `rf_spatial_tuning_pipeline.py` (new) — orchestrator for binned RF spatial tuning
- `rf_spatial_tuning_renderer.py` (new) — scatter + fit renderer for RF metrics vs stimulus parameter

Integration points:
- Imports from `rf_population_heatmap.py`: `compute_rf_heatmap`, `build_gesture_touch_indices`, `apply_vertex_threshold`
- Imports from `rf_inflection_boundary.py`: `compute_inflection_boundary`
- Imports from `rf_population_map_renderer.py`: `compute_uv_to_mm_scale`
- Imports from `rf_response_tuning_renderer.py`: `_fit_polynomial` (make public as `fit_polynomial`)
- Imports from `neuron_type_colors.py`: `build_session_color_scheme`

### Constraint: Neuron-centroid invariant

Per knowledge base (`note-3d-to-2d-surface-projection-algorithms.md`): population RF maps must project using the **neuron-wide contact centroid** as origin, not per-bin centroid. The binned approach inherits this from `compute_rf_heatmap()` which uses the session's pre-computed SLIM UV mapping.

---

## Implementation Plan

### Phase 1: Centroid Shift Pairwise Distance
**Goal:** Add centroid shift columns to the boundary comparison summary and bar charts.

**Started:** 2026-06-15
**Completed:** 2026-06-15

- [x] Task 1.1 — In `_load_boundary_metrics_from_npz()`, after loading per-gesture centroids, compute Euclidean distance between tap and stroke centroids (3D mm space), and between proximal and distal centroids
- [x] Task 1.2 — Add `centroid_shift_tap_vs_stroke_mm` and `centroid_shift_proximal_vs_distal_mm` columns to the summary DataFrame
- [x] Task 1.3 — Add both columns to `PANEL_METRICS` list so they appear in bar chart panels
- [x] Task 1.4 — Handle missing centroids gracefully (NaN when a gesture type has no inflection boundary)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py` — add pairwise distance computation and columns

**Dependencies:** None

### Phase 2: Multi-Fit Polynomial Degree
**Goal:** Support overlaying multiple polynomial fit lines on tuning curve scatter plots.

- [x] Task 2.1 — In `rf_response_tuning_pipeline.py`, parse `fit_degree` from options as `int | list[int]` (backward-compatible: single int still works)
- [x] Task 2.2 — Make `_fit_polynomial` public (rename to `fit_polynomial`) in `rf_response_tuning_renderer.py`
- [x] Task 2.3 — Modify `render_session_raw_dots()` to accept `fit_degrees: list[int]`, loop over degrees, draw each fit line with a distinct color from a palette (e.g., tab10), add an R² annotation box listing all degrees and their R² values
- [x] Task 2.4 — Apply the same multi-fit logic to `render_overlay_raw_dots()` for the cross-session overlay plots
- [x] Task 2.5 — Update DAG config documentation comment for `fit_degree` to indicate list support

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py` — parse fit_degree as int or list
- `code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py` — multi-fit rendering in `render_session_raw_dots()` and `render_overlay_raw_dots()`
- `configs/analyse_workflow_processing_dag.yaml` — update `fit_degree` comment

**Dependencies:** None (independent of Phase 1)

### Phase 3: Binned RF Spatial Tuning Curves
**Goal:** Create a pipeline that plots RF boundary metrics (area, circularity, PCA ratio, PCA angle) as a function of stimulus parameters (velocity, depth, contact_area).

**Started:** 2026-06-15
**Completed:** 2026-06-15

- [x] Task 3.1 — Create `rf_spatial_tuning_pipeline.py` with `run_spatial_tuning()` entry point
- [x] Task 3.2 — Implement touch binning: load feature CSV from `stimulus_extract_features/mean/`, bin touches by selected stimulus feature using equal-width bins (reuse bin edge logic from `_compute_bin_windows`)
- [x] Task 3.3 — Per bin: filter touch indices, call `compute_rf_heatmap()` with filtered indices, apply vertex threshold, interpolate grid, call `compute_inflection_boundary()`, extract boundary metrics, apply `compute_uv_to_mm_scale()` for mm conversion
- [x] Task 3.4 — Collect (bin_center, area_mm2, circularity, pca_aspect_ratio, pca_orientation_deg) tuples across bins; skip bins with too few touches or no inflection boundary
- [x] Task 3.5 — Create `rf_spatial_tuning_renderer.py` with scatter + fit rendering: X = bin center, Y = RF metric, one subplot per RF metric, polynomial fit overlay using the multi-fit infrastructure from Phase 2
- [x] Task 3.6 — Add Spearman correlation + p-value annotation to each subplot (for the area dismissal figure)
- [x] Task 3.7 — Iterate over gesture subsets (all, tap, stroke, stroke_proximal, stroke_distal) and tuning features (velocity, depth, contact_area)
- [x] Task 3.8 — Add per-session and cross-session (overlay by neuron type) rendering modes
- [x] Task 3.9 — Register as DAG task `spatial_tuning_rf_metrics` in `analysis_workflow_processing.py` and add config section to `analyse_workflow_processing_dag.yaml`
- [x] Task 3.10 — Wire into the analysis workflow DAG with `depends_on: [spatial_extract_boundaries, stimulus_extract_features]`

**Files Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_spatial_tuning_pipeline.py` — pipeline orchestrator
- `code/src/analysis/receptive_field_mapping/rendering/rf_spatial_tuning_renderer.py` — scatter + fit renderer

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — register new flow
- `configs/analyse_workflow_processing_dag.yaml` — add `spatial_tuning_rf_metrics` task config
- `code/src/analysis/CLAUDE.md` — document new pipeline

**Dependencies:** Phase 2 (reuses multi-fit rendering)

---

## Testing Plan

### Unit Tests
- [ ] Test centroid shift computation: given two known 3D centroids, verify Euclidean distance
- [ ] Test multi-fit parsing: verify `fit_degree=4` and `fit_degree=[1,2,3,4]` both work
- [ ] Test `_fit_polynomial` returns correct R² for known linear data

### Integration Tests
- [ ] Run `spatial_compare_boundaries` on full dataset; verify centroid shift columns in output CSV are non-NaN for sessions with both tap and stroke boundaries
- [ ] Run `stimulus_response_tuning` with `fit_degree: [1, 2, 4]`; verify PNGs show 3 fit lines with R² legend

### Manual Verification
- [ ] Inspect centroid shift bar charts — values should be in reasonable range (0-20 mm)
- [ ] Inspect multi-fit plots — higher degrees should not wildly overfit; R² should generally increase with degree
- [ ] Run binned RF spatial tuning on one neuron (e.g., session 14-01); verify scatter makes biological sense (area should vary with velocity for mechanoreceptors)
- [ ] Run area dismissal: contact_area → RF area scatter should show no significant correlation (p > 0.05 or flat trend)

### Edge Cases
- [ ] Session with no inflection boundary for a gesture type → centroid shift = NaN, not crash
- [ ] Bin with only 1-2 touches → skip bin (insufficient for population heatmap), not crash
- [ ] `fit_degree: 1` (single int, backward compat) still works after refactor

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with new pipeline description
- [ ] Update `configs/analyse_workflow_processing_dag.yaml` inline comments for new task
- [ ] No README or user guide changes needed (internal pipeline)

---

## Rollback Plan

1. **Phase 1 (centroid shift):** Revert changes to `rf_session_boundary_comparison_pipeline.py`. Existing outputs unaffected — new columns are additive.
2. **Phase 2 (multi-fit):** Revert renderer changes. Set `fit_degree: 4` (single int) in DAG config to restore previous behavior.
3. **Phase 3 (binned RF tuning):** Delete new pipeline and renderer files. Remove DAG task config entry. No existing outputs affected.

All phases are additive — no breaking changes to existing outputs or APIs.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bins with too few touches produce unreliable RF heatmaps | Medium | Medium | Set minimum touch count per bin (e.g., 5); skip bins below threshold; log skipped bins |
| Inflection boundary fails for sparse bins (no zero-crossing found) | Medium | Low | `compute_inflection_boundary()` already returns `None` for failed cases; collect NaN and skip in scatter |
| Multi-fit overlay clutters small plots | Low | Low | Use distinct linestyles (solid, dashed, dotted, dashdot) + muted colors; add toggle in config to select which degrees |
| Bin count too high → many empty bins for rare stimulus combinations | Low | Medium | Use `clip_percentile` to trim extreme stimulus values; auto-select bin count based on touch count |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Centroid shift | ~30 min | None |
| Phase 2: Multi-fit degree | ~2-3 hours | None |
| Phase 3: Binned RF spatial tuning | ~4-6 hours | Phase 2 |

---

## References

- Figure specification: `docs/article-scoping/07-figure-specification.md`
- Readiness assessment: `.claude/plans/analyse-docs-article-scoping-07-figure-s-optimized-grove.md`
- Inflection boundary algorithm: `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
- Population heatmap: `code/src/analysis/receptive_field_mapping/data/rf_population_heatmap.py`
- Boundary comparison pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
- Tuning pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py`
- Tuning renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py`
- Knowledge base (projection): `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`

---

## Modified Files (all phases)

### Phase 1 — Centroid Shift
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`

### Phase 2 — Multi-Fit Polynomial Degree
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py`
- `configs/analyse_workflow_processing_dag.yaml`

### Phase 3 — Binned RF Spatial Tuning Curves
**Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_spatial_tuning_pipeline.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_spatial_tuning_renderer.py`

**Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py`
- `code/src/analysis/pipeline/output_dirs.py`
- `code/scripts/analysis_workflow_processing.py`
- `configs/analyse_workflow_processing_dag.yaml`
- `code/src/analysis/CLAUDE.md`

---
