# Plan: Tap-vs-Stroke RF Comparison Pipeline

**Date:** 2026-06-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/rf-tap-stroke-comparison`

---

## Overview

Build a dedicated tap-vs-stroke RF comparison pipeline that performs paired
analysis of receptive field properties between static taps and dynamic strokes.
This fills the gap for Figure 2 of the paper: the proximal-distal comparison
pipeline exists and is mature, but no equivalent paired analysis exists for
tap vs stroke. A small set of new metrics is also added to both pipelines for
a consistent, richer property catalogue.

## Problem Statement

The session boundary comparison pipeline (`rf_session_boundary_comparison_pipeline.py`)
displays tap and stroke RF properties side-by-side but does not compute paired
deltas, shift decomposition, contour overlap, heatmap correlation, or statistical
tests. Figure 2 of the paper requires the same depth of analysis for tap vs stroke
as already exists for proximal vs distal.

## Goals

### In Scope
1. Dedicated `rf_tap_stroke_comparison_pipeline.py` with full paired analysis
2. Corresponding renderer (`rf_tap_stroke_comparison_renderer.py`)
3. New metrics added to both the tap-vs-stroke and proximal-vs-distal pipelines:
   `peak_iff`, `equivalent_diameter_mm`, `rf_sharpness`, `iff_at_centroid`,
   `containment_a_in_b` / `containment_b_in_a`
4. DAG wiring: output directory constant, YAML task entry, Prefect flow function
5. Individual output PNGs suitable for figure composition

### Out of Scope
- Composing the final Figure 2 layout (user handles this)
- Interactive GUI exploration of tap-vs-stroke differences
- Changes to the upstream `spatial_extract_boundaries` pipeline or NPZ format
- Stimulus-parameter binned comparisons (covered by `rf_spatial_tuning_pipeline.py`)

## Success Criteria

- [ ] `spatial_compare_tap_stroke` DAG task runs end-to-end without error
- [ ] Summary CSV (`rf_tap_stroke_comparison_summary.csv`) contains all metrics
      with correct sign convention (tap − stroke) and no unexpected NaN pattern
- [ ] Per-session PNGs render: contour overlay, heatmap triptych, circular crops
- [ ] Cross-session aggregate PNGs render: population strips with Wilcoxon/sign
      test annotations, metric delta bar charts, peak shift decomposition quiver,
      aggregate scatter
- [ ] New metrics (`peak_iff`, `equivalent_diameter_mm`, `rf_sharpness`,
      `iff_at_centroid`, `containment`) appear in both tap-vs-stroke and
      proximal-vs-distal summary CSVs
- [ ] Existing proximal-distal outputs remain unchanged except for the additional
      metric columns
- [ ] All tests pass (`pytest code/tests/`)

---

## Technical Design

### Approach

Create a dedicated pipeline module mirroring the structure of
`rf_proximal_distal_comparison_pipeline.py`. The new metrics are derived from
existing NPZ data (no upstream changes needed): `grid_z_{gtype}` provides
`peak_iff`, `boundary_area_xyz_mm2_{gtype}` provides `equivalent_diameter_mm`,
and `boundary_contour_uv_{gtype}` enables containment via
`matplotlib.path.Path.contains_points`.

The rendering layer reuses `render_shift_decomposition()` and
`render_center_marked_heatmap()` directly, and provides tap-vs-stroke variants
of the contour overlay, heatmap triptych, aggregate scatter, population strips,
and metric delta charts.

**Sign convention:** delta = tap − stroke (static minus dynamic).

**Primary center type:** peak (hotspot) — the grid cell with maximum IFF intensity.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Dedicated pipeline (mirror proximal-distal) | Self-contained, comparison-specific logic stays clean, matches existing pattern | Code duplication between the two pipelines | **Chosen** |
| Generalize proximal-distal pipeline to accept arbitrary gesture pairs | No duplication | Fragile parameterization, proximal-distal has specific logic (stroke as hotspot reference, contour-center offsets vs "all") that doesn't translate 1:1 | Rejected |
| Extend session boundary comparison pipeline | Minimal new files | That pipeline has a different structure (one row per session × gesture, not paired deltas); bolting on paired analysis would be awkward | Rejected |

### Architecture Changes

New modules:
```
code/src/analysis/receptive_field_mapping/
├── pipelines/
│   └── rf_tap_stroke_comparison_pipeline.py    # NEW
└── rendering/
    └── rf_tap_stroke_comparison_renderer.py     # NEW
```

Modified modules:
```
code/src/analysis/receptive_field_mapping/
├── pipelines/
│   └── rf_proximal_distal_comparison_pipeline.py  # Add new metrics
└── rendering/
    └── rf_proximal_distal_comparison_renderer.py   # Add new metrics to strip/delta lists

code/src/analysis/pipeline/
└── output_dirs.py                                  # Add SPATIAL_COMPARE_TAP_STROKE

code/scripts/
└── analysis_workflow_processing.py                 # Add flow function + task registration

configs/
└── analyse_workflow_processing_dag.yaml            # Add task entry
```

### Knowledge Base Constraints

- **Coordinate spaces** (`note-analysis-pipeline-coordinate-spaces.md`): new
  metrics computed from existing NPZ data already in UV/mm space — no new
  projections needed.
- **Units** (`note-somatosensory-units-and-calculations.md`): distances in mm,
  areas in mm², `peak_iff` and `rf_sharpness` are dimensionless IFF ratios,
  `containment` is a dimensionless fraction.
- **Idempotency**: sentinel JSON (`rf_tap_stroke_comparison_done.json`), NaN
  fallback for missing NPZ keys with logged warnings.

---

## Implementation Plan

### Phase 1: New Metrics in Proximal-Distal Pipeline
**Goal:** Add the new metric columns to the existing proximal-distal pipeline
so both comparisons share a consistent metric set.
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 1.1 — Extract `peak_iff` per direction: `np.nanmax(grid_z_{gtype})`
      where `grid_z > 0`
- [x] Task 1.2 — Compute `equivalent_diameter_mm`: `np.sqrt(4 * area_mm2 / np.pi)`
- [x] Task 1.3 — Compute `rf_sharpness`: `peak_iff / mean_iff_on_contour`
      (NaN when denominator is 0 or NaN)
- [x] Task 1.4 — Extract `iff_at_centroid` from NPZ key
      `boundary_iff_at_centroid_{gtype}`
- [x] Task 1.5 — Compute asymmetric `containment_proximal_in_distal` and
      `containment_distal_in_proximal` via `Path.contains_points` on the
      existing 150x150 grid
- [x] Task 1.6 — Add delta columns: `delta_peak_iff`, `delta_equivalent_diameter_mm`,
      `delta_rf_sharpness`, `delta_iff_at_centroid`
- [x] Task 1.7 — Update `_STRIP_METRICS` and `_DELTA_METRICS` lists in renderer
- [x] Task 1.8 — Update CSV dtype casting in DataFrame `.astype()` call

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py` — per-direction extraction, delta computation, summary row dict, dtype cast
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py` — `_STRIP_METRICS`, `_DELTA_METRICS`

**Dependencies:** None

### Phase 2: Tap-vs-Stroke Comparison Pipeline
**Goal:** Create the core pipeline that loads NPZ data, computes all metrics and
deltas for tap vs stroke, and writes a summary CSV.
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 2.1 — Create `rf_tap_stroke_comparison_pipeline.py` with entry point
      `run_tap_stroke_comparison(session_configs, output_dir, ...)`
- [x] Task 2.2 — Define `_REQUIRED_GTYPES = ('all', 'tap', 'stroke')` and
      `_COMPARED_GTYPES = ('tap', 'stroke')`
- [x] Task 2.3 — Pass 1: per-session NPZ loading, boundary scalar extraction
      for tap and stroke (area, perimeter, circularity, PCA, mean_iff,
      iff_at_centroid, n_touches, peak_to_centroid, peak_iff,
      equivalent_diameter, rf_sharpness)
- [x] Task 2.4 — Compute deltas (tap − stroke) for all scalar metrics
- [x] Task 2.5 — Compute peak shift decomposition (along-arm U, across-arm V)
      using hotspot locations
- [x] Task 2.6 — Compute centroid shift decomposition (secondary)
- [x] Task 2.7 — Compute contour overlap (IoU, Dice) and asymmetric containment
- [x] Task 2.8 — Compute heatmap Pearson correlation
- [x] Task 2.9 — Compute centroid/hotspot offsets from `all` gesture center
      (for aggregate scatter plots)
- [x] Task 2.10 — Write summary CSV with all metrics and proper dtype casting
- [x] Task 2.11 — Copy `_compute_contour_overlap` and `_compute_heatmap_similarity`
      helpers (or import from proximal-distal module)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py` — **new file**

**Dependencies:** Phase 1 (for consistent metric definitions)

### Phase 3: Tap-vs-Stroke Renderer
**Goal:** Create rendering functions for per-session and cross-session
tap-vs-stroke comparison figures.
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 3.1 — `render_tap_stroke_contour_overlay()`: tap contour (one color)
      vs stroke contour (another color), `all` gesture as dashed reference;
      annotated with IoU and area ratio
- [x] Task 3.2 — `render_tap_stroke_heatmap_triptych()`: Tap | Stroke |
      Difference (T − S) panels
- [x] Task 3.3 — `render_tap_stroke_aggregate()`: scatter of tap and stroke
      centroid offsets relative to `all` gesture center
- [x] Task 3.4 — `render_tap_stroke_hotspot_aggregate()`: scatter of peak
      offsets relative to `all` gesture peak
- [x] Task 3.5 — `render_tap_stroke_metric_deltas()`: multi-panel bar chart
      of delta metrics per session, colored by neuron type
- [x] Task 3.6 — `render_tap_stroke_population_strips()`: strip chart with
      Wilcoxon signed-rank + sign test + Clopper-Pearson CI per metric

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_tap_stroke_comparison_renderer.py` — **new file**

**Dependencies:** Phase 2

### Phase 4: Per-Session and Cross-Session Rendering Passes
**Goal:** Wire the renderer into the pipeline's Pass 2 (per-session figures) and
Pass 3 (cross-session aggregates).
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 4.1 — Pass 2: render per-session contour overlay, heatmap triptych,
      center-marked heatmaps, circular crops (per gesture type × center type)
- [x] Task 4.2 — Pass 3: render cross-session figures: population strips,
      metric delta bar chart, peak shift decomposition (reuse
      `render_shift_decomposition`), aggregate scatters
- [x] Task 4.3 — Build session color scheme from `neuron_summary_xlsx`
- [x] Task 4.4 — Write sentinel JSON (`rf_tap_stroke_comparison_done.json`)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py` — add Pass 2 and Pass 3

**Dependencies:** Phase 3

### Phase 5: DAG Wiring
**Goal:** Make the pipeline executable from the DAG launcher.
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 5.1 — Add `SPATIAL_COMPARE_TAP_STROKE = "spatial_compare_tap_stroke"`
      to `output_dirs.py`
- [x] Task 5.2 — Add task entry to `analyse_workflow_processing_dag.yaml`:
      ```yaml
      spatial_compare_tap_stroke:
        category: spatial_sensitivity
        enabled: true
        options:
          force_processing: true
          iff_metric: both
        depends_on: [spatial_extract_boundaries]
      ```
- [x] Task 5.3 — Add `spatial_compare_tap_stroke_flow` Prefect flow function
      to `analysis_workflow_processing.py` (mirrors proximal-distal flow)
- [x] Task 5.4 — Register flow in the task list with correct parameter
      resolution lambda

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — add constant
- `configs/analyse_workflow_processing_dag.yaml` — add task entry
- `code/scripts/analysis_workflow_processing.py` — add flow function + task list entry

**Dependencies:** Phase 4

### Phase 6: Testing
**Goal:** Verify correctness of the new pipeline and the new metrics in
the existing proximal-distal pipeline.
**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 6.1 — Add `test_rf_tap_stroke_comparison.py` with synthetic NPZ
      data (mirrors existing test patterns in `test_rf_inflection_boundary.py`)
- [x] Task 6.2 — Test new metric computation: peak_iff, equivalent_diameter,
      rf_sharpness, containment, with known inputs and expected outputs
- [x] Task 6.3 — Test delta sign convention: verify tap − stroke ordering
- [x] Task 6.4 — Test edge cases: session with missing tap or stroke boundary
      (should produce NaN + warning, not crash)
- [x] Task 6.5 — Run `pytest` — verify all existing tests still pass

**Files Modified:**
- `code/tests/test_rf_tap_stroke_comparison.py` — **new file**

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] New metric computation with known inputs (peak_iff, equivalent_diameter,
      rf_sharpness, containment)
- [ ] Delta computation with known tap/stroke values
- [ ] Contour overlap (IoU/Dice) with overlapping and non-overlapping synthetic
      polygons
- [ ] Containment with fully contained, partially contained, and disjoint contours
- [ ] Shift decomposition with known UV offsets

### Integration Tests
- [ ] Full pipeline run with synthetic NPZ data → verify CSV columns and PNG
      file existence
- [ ] Proximal-distal pipeline with new metrics → verify new columns in CSV

### Manual Verification
- [ ] Execute `spatial_compare_tap_stroke` via the DAG launcher on real data
- [ ] Inspect summary CSV: all sessions present, metrics have reasonable ranges
- [ ] Inspect per-session PNGs: contour overlay shows both boundaries, heatmap
      triptych shows difference pattern
- [ ] Inspect cross-session PNGs: population strips show p-values, shift arrows
      are visible
- [ ] Re-run `spatial_compare_proximal_distal` and verify new metric columns
      appear alongside originals

### Edge Cases
- [ ] Session with no tap boundary (too few touches → no inflection ring):
      metrics NaN, figures skipped with warning
- [ ] Session with no stroke boundary: same behavior
- [ ] Session with identical tap and stroke boundaries: deltas ≈ 0, IoU ≈ 1,
      containment ≈ 1
- [ ] All sessions missing one condition: pipeline exits with warning, no crash

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add tap-vs-stroke comparison
      section mirroring the proximal-distal section
- [ ] Update DAG config comments in `analyse_workflow_processing_dag.yaml`

---

## Rollback Plan

1. **Before merge:** all changes are on a feature branch; discard with
   `git branch -D feature/rf-tap-stroke-comparison`
2. **New metrics in proximal-distal pipeline:** revert the metric additions
   in a single commit (they are additive columns — no existing columns change)
3. **DAG entry:** remove the task from the YAML; downstream tasks have no
   dependency on it
4. **No data migration:** the pipeline creates new output directories; deleting
   `4_analysed/spatial_compare_tap_stroke/` removes all artifacts

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Some sessions lack a `stroke` boundary (too few combined proximal+distal touches) | Medium | Low — those sessions produce NaN rows | Log warning, skip rendering for that session, document in CSV |
| `rf_sharpness` denominator is 0 (mean_iff_on_contour = 0) | Low | Low | Guard with `if denominator > 0 else NaN` |
| Containment computation is slow on 150×150 grid | Low | Low | Same grid size as existing IoU/Dice — ~22k points, runs in <1s |
| Code duplication with proximal-distal pipeline | Medium | Low (maintenance cost) | Accept for now; if a third comparison pipeline emerges, extract shared utilities |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: New metrics in proximal-distal | Small | None |
| Phase 2: Tap-vs-stroke pipeline core | Medium | Phase 1 |
| Phase 3: Renderer | Medium | Phase 2 |
| Phase 4: Rendering passes | Medium | Phase 3 |
| Phase 5: DAG wiring | Small | Phase 4 |
| Phase 6: Testing | Small | Phase 5 |

---

## References

- Existing pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`
- Existing renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py`
- Containment pattern: `code/src/analysis/receptive_field_mapping/metrics/rf_baseline_deviation.py`
- Knowledge base: `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md`, `note-somatosensory-units-and-calculations.md`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/pipeline/output_dirs.py
- code/src/analysis/receptive_field_mapping/__init__.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_tap_stroke_comparison_renderer.py
- code/tests/test_rf_tap_stroke_comparison.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rf-tap-stroke-comparison.md
