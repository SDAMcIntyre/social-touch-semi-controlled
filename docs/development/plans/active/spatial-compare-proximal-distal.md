# Plan: Rename and Extend spatial_compare_rf_centers → spatial_compare_proximal_distal

**Date:** 2026-06-22
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/spatial-extract-rf-profiles`
**Branch:** `feature/spatial-proximal-distal-comparison`

---

## Overview

Rename the `spatial_compare_rf_centers` DAG task to `spatial_compare_proximal_distal` and extend it from a center-only comparison into a comprehensive proximal-vs-distal RF comparison. The extended task extracts all available boundary metrics from existing NPZ data (area, perimeter, circularity, PCA shape, orientation, response magnitude), computes paired deltas and overlap metrics, and produces publication-quality per-session and cross-session figures.

## Problem Statement

The current task compares only centroid and hotspot UV positions between proximal and distal strokes, despite the upstream NPZ files (`spatial_extract_boundaries`) already storing a rich set of per-gesture-type boundary metrics. The sibling task `spatial_compare_boundaries` shows absolute metrics across all gesture types but does not compute proximal-vs-distal paired contrasts (deltas, ratios, statistical summaries). There is no single task that answers the central question: "how does stroke direction modulate the spatial and response properties of mechanoreceptive afferent receptive fields?"

## Goals

### In Scope

1. Rename the DAG task, pipeline file, renderer file, output dir constant, flow function, and all internal references
2. Extract all scalar boundary metrics for `stroke_proximal` and `stroke_distal` from existing NPZ data
3. Compute paired deltas (proximal − distal), ratios, and derived geometric metrics (peak-to-centroid distance, contour overlap IoU/Dice, heatmap correlation, centroid shift decomposition along/across arm)
4. Add per-session contour overlay figures (proximal vs distal on same axes) and heatmap triptych (proximal | distal | difference)
5. Add cross-session aggregate figures: metric delta bar charts, population strip chart with Wilcoxon p-values, centroid shift decomposition arrows
6. Extend the summary CSV with all new columns
7. Accept `neuron_summary_xlsx` for neuron-type coloring in cross-session figures

### Out of Scope

- Modifications to the upstream `spatial_extract_boundaries` pipeline or NPZ format
- Modifications to the sibling `spatial_compare_boundaries` task
- Instruction-level stratification (speed, force, contact area within proximal/distal)
- 3D visualization of contour overlays on the forearm mesh
- Tap vs stroke comparison (handled by `spatial_compare_boundaries`)

## Success Criteria

- [ ] All references to `spatial_compare_rf_centers` are renamed; `pytest` passes after rename
- [ ] Summary CSV contains per-direction scalar metrics, deltas, ratios, overlap, and correlation columns with valid values
- [ ] Per-session contour overlay PNGs show proximal (cyan) and distal (magenta) contours with centroids and hotspots
- [ ] Per-session heatmap triptych PNGs show proximal, distal, and difference panels with appropriate colormaps
- [ ] Cross-session metric delta bar chart renders with neuron-type coloring
- [ ] Cross-session population strip chart shows Wilcoxon signed-rank p-values per delta metric
- [ ] Cross-session centroid shift decomposition shows along-arm and across-arm components
- [ ] Sessions missing a boundary for either direction produce NaN metrics and skip overlay/triptych with a warning
- [ ] Task runs via pipeline GUI with existing sessions

---

## Technical Design

### Approach

Extend the existing pipeline in-place after renaming. The NPZ files already contain all needed data per gesture type — the work is purely extraction, arithmetic, and rendering. Follow the metric extraction pattern from `rf_session_boundary_comparison_pipeline.py::_load_boundary_metrics_from_npz`. Use `matplotlib.path.Path.contains_points` for contour overlap (no new dependencies). Reuse `build_session_color_scheme` for neuron-type coloring.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend existing pipeline + renderer in-place (this plan) | Minimal new files; reuses all existing per-session heatmap rendering; clean rename | Larger files (~700-800 lines pipeline) | **Chosen** |
| Merge into `spatial_compare_boundaries` | Fewer tasks; boundary metrics already extracted there | Bloats an already-complex pipeline; mixes all-gesture overview with paired proximal-distal contrast | Rejected |
| New standalone pipeline reading from `spatial_compare_boundaries` CSV | Avoids re-reading NPZ | Adds cross-task dependency; boundary comparison CSV lacks contour geometry and grid data | Rejected |

### Architecture Changes

No new modules — existing files are renamed and extended:

```
code/src/analysis/receptive_field_mapping/
├── pipelines/
│   └── rf_proximal_distal_comparison_pipeline.py   ← RENAMED from rf_proximal_distal_center_pipeline.py
└── rendering/
    └── rf_proximal_distal_comparison_renderer.py    ← RENAMED from rf_center_comparison_renderer.py
```

Integration points (modifications to existing files):
- `code/src/analysis/pipeline/output_dirs.py` — rename constant + add to `RENAME_MAPPING`
- `configs/analyse_workflow_processing_dag.yaml` — rename task key, update comments, add `neuron_summary_xlsx` option
- `code/scripts/analysis_workflow_processing.py` — rename flow function + import + DAG dispatch entry
- `code/src/analysis/receptive_field_mapping/__init__.py` — update re-export
- `code/src/analysis/CLAUDE.md` — update documentation entry

### Rename Mapping

| What | Old | New |
|------|-----|-----|
| DAG key | `spatial_compare_rf_centers` | `spatial_compare_proximal_distal` |
| Output dir constant | `SPATIAL_COMPARE_RF_CENTERS` | `SPATIAL_COMPARE_PROXIMAL_DISTAL` |
| Pipeline file | `rf_proximal_distal_center_pipeline.py` | `rf_proximal_distal_comparison_pipeline.py` |
| Renderer file | `rf_center_comparison_renderer.py` | `rf_proximal_distal_comparison_renderer.py` |
| Public function | `run_proximal_distal_center_comparison` | `run_proximal_distal_comparison` |
| Flow function | `spatial_compare_rf_centers_flow` | `spatial_compare_proximal_distal_flow` |
| Sentinel | `rf_center_proximal_distal_done.json` | `rf_proximal_distal_comparison_done.json` |
| Summary CSV | `rf_center_proximal_distal_summary.csv` | `rf_proximal_distal_comparison_summary.csv` |

### New Metrics

**Scalar metrics per direction** (read from NPZ, for both `stroke_proximal` and `stroke_distal`):
`area_mm2`, `perimeter_mm`, `circularity`, `pca_major_mm`, `pca_minor_mm`, `pca_aspect_ratio`, `pca_orientation_deg`, `mean_iff_on_contour`, `n_touches`

**Deltas** (proximal − distal, signed):
`delta_area_mm2`, `delta_area_pct` (normalized by stroke area), `area_ratio`, `delta_perimeter_mm`, `delta_circularity`, `delta_pca_aspect_ratio`, `delta_pca_orientation_deg` (wrapped to [-90,90]), `delta_mean_iff_on_contour`

**Derived geometric metrics:**
`peak_to_centroid_mm_{proximal|distal}`, `delta_peak_to_centroid_mm`, `contour_overlap_iou`, `contour_overlap_dice`, `heatmap_pearson_r`, `centroid_shift_along_arm_mm`, `centroid_shift_across_arm_mm`

### New Figures

**Per-session:**
1. Contour overlay — proximal (cyan) + distal (magenta) + stroke reference (dashed grey) on forearm mesh, annotated with IoU and area ratio
2. Heatmap triptych — [proximal | distal | difference] with inferno + RdBu_r colormaps

**Cross-session aggregate:**
3. Metric delta bar charts — multi-panel, sessions colored by neuron type
4. Population strip chart — one column per delta metric, Wilcoxon p-value annotations
5. Centroid shift decomposition — arrow plot showing along-arm / across-arm components

### Key Implementation Notes

- **Angular delta wrapping:** `delta = ((orient_p - orient_d + 90) % 180) - 90` for PCA 180° ambiguity
- **Contour overlap:** rasterize both contour polygons onto the 150×150 grid using `matplotlib.path.Path.contains_points`, then compute pixel IoU and Dice — no Shapely dependency
- **Grid alignment:** proximal and distal `grid_z` share the same UV meshgrid per session, so element-wise subtraction is valid
- **Backward compat:** check for both old and new sentinel names during idempotency check
- **Missing boundaries:** when either direction lacks a boundary, all delta metrics become NaN and overlay/triptych are skipped

---

## Implementation Plan

### Phase 0: Rename
**Goal:** Rename all files, symbols, and references before adding new logic so the diff for new functionality is clean.
**Started:** 2026-06-22

- [x] Task 0.1 — `git mv` pipeline file to `rf_proximal_distal_comparison_pipeline.py`
- [x] Task 0.2 — `git mv` renderer file to `rf_proximal_distal_comparison_renderer.py`
- [x] Task 0.3 — Update `output_dirs.py`: rename constant, add old→new to `RENAME_MAPPING`
- [x] Task 0.4 — Update DAG YAML: key, comments, output path reference
- [x] Task 0.5 — Update `analysis_workflow_processing.py`: import, flow name, function call, params lambda
- [x] Task 0.6 — Update `receptive_field_mapping/__init__.py`: import path, re-export
- [x] Task 0.7 — Update internal function/variable names: `run_proximal_distal_center_comparison` → `run_proximal_distal_comparison`, sentinel name, CSV name, log messages
- [x] Task 0.8 — Grep for remaining references to old names; update docs/plans that reference the old DAG key
- [x] Task 0.9 — Run `pytest` to verify no regressions

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — rename constant + RENAME_MAPPING
- `configs/analyse_workflow_processing_dag.yaml` — rename task key + comments
- `code/scripts/analysis_workflow_processing.py` — import, flow function, DAG dispatch
- `code/src/analysis/receptive_field_mapping/__init__.py` — re-export
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py` — internal names
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py` — docstrings
- `code/src/analysis/CLAUDE.md` — documentation entry

**Dependencies:** None

### Phase 1: Metric Extraction
**Goal:** Load all boundary scalar metrics for both directions from NPZ, compute deltas and ratios, extend the summary CSV.

- [x] Task 1.1 — Define `_BOUNDARY_SCALAR_KEYS` constant listing NPZ key stems
- [x] Task 1.2 — In the per-session loading loop, extract all boundary scalars for `stroke_proximal` and `stroke_distal`, converting UV-space values to mm via `uv_to_mm`
- [x] Task 1.3 — Compute `pca_aspect_ratio` for each direction
- [x] Task 1.4 — Compute all delta metrics (proximal − distal) and `area_ratio`
- [x] Task 1.5 — Compute `delta_area_pct` (normalized by `boundary_area_xyz_mm2_stroke`)
- [x] Task 1.6 — Compute `peak_to_centroid_mm` for each direction
- [x] Task 1.7 — Add all new metrics to `valid_data` dict and `summary_rows`
- [x] Task 1.8 — Extend the DataFrame dtype spec and CSV export

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** Phase 0

### Phase 2: Contour Overlap and Heatmap Correlation
**Goal:** Compute contour overlap (IoU, Dice) and heatmap correlation, plus centroid shift decomposition.

- [x] Task 2.1 — Implement `_compute_contour_overlap(contour_a_uv, contour_b_uv, grid_u, grid_v)` using `matplotlib.path.Path.contains_points` on the 150×150 grid
- [x] Task 2.2 — Implement `_compute_heatmap_similarity(grid_z_a, grid_z_b)` for Pearson r (finite cells only)
- [x] Task 2.3 — Compute centroid shift decomposition: project proximal-distal centroid vector onto U axis (along-arm) and V axis (across-arm), convert to mm
- [x] Task 2.4 — Add all new metrics to `valid_data` and `summary_rows`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** Phase 1

### Phase 3: Per-Session Rendering
**Goal:** Add contour overlay and heatmap triptych figures per session.

- [x] Task 3.1 — Add `render_proximal_distal_contour_overlay()` to renderer: forearm mesh background, proximal contour (cyan), distal contour (magenta), stroke contour (dashed grey), centroids (+), hotspots (*), IoU and area ratio annotation
- [x] Task 3.2 — Add `render_proximal_distal_heatmap_triptych()` to renderer: 3-panel [proximal | distal | difference], shared inferno for first two, RdBu_r diverging for difference
- [x] Task 3.3 — Wire both renderers into pipeline Pass 2, loading contour and grid data from `valid_data`
- [x] Task 3.4 — Guard rendering on both directions having valid boundaries; skip with warning if either is missing

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** Phase 2

### Phase 4: Cross-Session Rendering
**Goal:** Add all cross-session aggregate figures with neuron-type coloring.

- [x] Task 4.1 — Add `neuron_summary_xlsx` parameter to `run_proximal_distal_comparison()` and build `SessionColorScheme` when provided
- [x] Task 4.2 — Add `render_proximal_distal_metric_deltas()`: 3×3 subplot grid of delta bar charts, sessions colored by neuron type
- [x] Task 4.3 — Add `render_proximal_distal_population_strips()`: strip plot with one column per delta metric, Wilcoxon signed-rank p-value annotation per metric, horizontal zero line
- [x] Task 4.4 — Add `render_centroid_shift_decomposition()`: arrow/quiver plot showing per-session centroid shift vectors decomposed into U (along-arm) and V (across-arm) components
- [x] Task 4.5 — Wire all aggregate renderers into pipeline Pass 3

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** Phase 3

### Phase 5: Integration and Documentation
**Goal:** Wire `neuron_summary_xlsx` through DAG config and update documentation.

- [x] Task 5.1 — Add `neuron_summary_xlsx` to the DAG params lambda in `analysis_workflow_processing.py`
- [x] Task 5.2 — Update DAG YAML comments to reflect new scope
- [x] Task 5.3 — Update `CLAUDE.md` entry with new outputs and metrics
- [x] Task 5.4 — Update `docs/architecture/analysis-pipeline-outcomes.md` if it references the old task

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py`
- `configs/analyse_workflow_processing_dag.yaml`
- `code/src/analysis/CLAUDE.md`

**Dependencies:** Phase 4

---

## Testing Plan

### Unit Tests

- [ ] Synthetic NPZ with known boundary metrics for proximal/distal → verify CSV delta columns match hand-computed values
- [ ] `_compute_contour_overlap` with two identical contours → IoU = 1.0, Dice = 1.0
- [ ] `_compute_contour_overlap` with non-overlapping contours → IoU = 0.0, Dice = 0.0
- [ ] `_compute_heatmap_similarity` with identical grids → Pearson r = 1.0
- [ ] Angular delta wrapping: verify `delta_pca_orientation_deg` stays in [-90, 90]

### Integration Tests

- [ ] Run the full DAG task on existing sessions → verify per-session directories contain all PNGs (existing + new)
- [ ] Verify summary CSV has all expected columns with valid values
- [ ] Verify `force_processing: false` respects sentinel (skip) and `force_processing: true` re-renders
- [ ] Verify backward compat: old sentinel name is also recognized

### Manual Verification

- [ ] Open a contour overlay PNG and verify proximal/distal contours align with their respective heatmaps
- [ ] Open a heatmap triptych and verify the difference panel shows sensible signed differences
- [ ] Open the population strip chart and verify Wilcoxon p-values are plausible
- [ ] Launch the pipeline GUI and confirm the renamed task appears and can be toggled

### Edge Cases

- [ ] Session where `stroke_proximal` has boundary but `stroke_distal` does not → NaN deltas, overlay skipped with warning
- [ ] Session where both directions lack boundaries → all metrics NaN, session excluded from aggregate figures
- [ ] Single-session run → aggregate plots render with one data point, Wilcoxon not computed (n < 5)
- [ ] Contours with degenerate shapes (very few points) → overlap computation handles gracefully

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — rename task entry, document new outputs and metrics
- [ ] Update `docs/architecture/analysis-pipeline-outcomes.md` if it references the old task name
- [ ] No README.md or user guide changes needed — task is auto-discovered by DAG launcher GUI

---

## Rollback Plan

The feature is a rename + extension of existing files. No new files are created (only renamed).

1. **Rollback procedure:**
   - `git mv` files back to original names
   - Restore original constant name in `output_dirs.py` (remove from `RENAME_MAPPING`)
   - Restore original DAG key and flow function names
   - Revert the pipeline/renderer content to the pre-extension state
2. **Data considerations:**
   - Output directory `4_analysed/spatial_compare_proximal_distal/` can be deleted
   - If old output directory `4_analysed/spatial_compare_rf_centers/` still exists, it remains valid
   - No upstream data is modified

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Some sessions lack boundary for one direction (upstream detection failed) | Medium | Low | Skip session with warning; NaN metrics; exclude from aggregate figures |
| `matplotlib.path.Path.contains_points` rasterization at 150×150 gives imprecise IoU for small contours | Low | Low | Resolution matches the heatmap grid; sufficient for comparative analysis |
| Wilcoxon signed-rank test unreliable with small N (≤ 5 sessions) | Medium | Low | Skip statistical annotation when N < 5; note in figure |
| Rename misses a reference in an obscure file | Low | Medium | Comprehensive grep after rename; run `pytest` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 0 (rename) | ~1 hour | None |
| Phase 1 (metric extraction) | ~1.5 hours | Phase 0 |
| Phase 2 (overlap + correlation) | ~1.5 hours | Phase 1 |
| Phase 3 (per-session rendering) | ~2 hours | Phase 2 |
| Phase 4 (cross-session rendering) | ~2.5 hours | Phase 3 |
| Phase 5 (integration + docs) | ~30 minutes | Phase 4 |

Total: ~9 hours.

---

## References

- Completed predecessor plan: `docs/development/plans/completed/rf-center-proximal-distal-comparison.md`
- Sibling pipeline (pattern-match for metric extraction): `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
- Existing pipeline being extended: `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py`
- Existing renderer being extended: `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py`
- Neuron-type coloring: `code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py`
- DAG config: `configs/analyse_workflow_processing_dag.yaml`
- Workflow script: `code/scripts/analysis_workflow_processing.py`
- NPZ data format: documented in `code/src/analysis/CLAUDE.md` under "Population response fields"
- Knowledge base notes consulted: `note-circular-crop-scale-consistency.md`, `note-analysis-pipeline-coordinate-spaces.md`, `note-neural-kinect-viewer-blitting.md`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/CLAUDE.md
- code/src/analysis/pipeline/output_dirs.py
- code/src/analysis/receptive_field_mapping/__init__.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/architecture/analysis-pipeline-outcomes.md
- docs/development/plans/active/spatial-compare-proximal-distal.md
