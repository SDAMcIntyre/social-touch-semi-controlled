# Plan: RF Paired Metric Violin Plots

**Date:** 2026-06-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-boundary-method-config-switch`
**Branch:** `feature/rf-paired-metric-violins`

---

## Overview

Add a violin + paired-dot distribution figure to both the proximal-distal and
tap-stroke RF comparison pipelines. Each figure shows 4 subplots (Area,
Perimeter, Mean IFF, Max IFF) with per-condition distributions, per-neuron
paired observations, and Wilcoxon signed-rank statistical annotations. These
are standalone figures that will be manually assembled into Figure 02 of the
article.

## Problem Statement

The existing RF comparison pipelines produce center-shift scatter plots and
population strip charts (one dot per session at x=0), but no distribution
visualization that shows both the shape of the population spread and the
paired within-subject trajectories for the four key boundary metrics. The
article's Figure 02 requires this visualization style (violin + paired dots
with statistical significance).

## Goals

### In Scope
1. New generic renderer function `render_paired_metric_violins()` usable by
   both comparison pipelines
2. Wire the function into the proximal-distal pipeline (pass 3, cross-session)
3. Wire the function into the tap-stroke pipeline (pass 3, cross-session)
4. Statistical annotations: Wilcoxon signed-rank p-value, sign test p-value,
   consistency count with Clopper-Pearson CI (matching existing strip chart
   annotation style)

### Out of Scope
- Compositing sub-figures into the full Figure 02 (manual assembly)
- New statistical tests beyond the already-implemented Wilcoxon/sign-test
- Modifying existing strip chart or metric delta figures
- Effect size measures or power analysis
- Population RF map panels and center-shift plots (already exist)

## Success Criteria

- [ ] `render_paired_metric_violins()` produces a 1x4 subplot figure with
      violin distributions for each condition, paired dots connected by lines,
      and statistical annotations
- [ ] Proximal-distal pipeline outputs
      `rf_proximal_distal_metric_violins.png` + `.svg`
- [ ] Tap-stroke pipeline outputs `rf_tap_stroke_metric_violins.png` + `.svg`
- [ ] Existing pipeline outputs are unchanged (no regression)
- [x] `pytest code/tests/test_rf_inflection_boundary.py` passes

---

## Technical Design

### Approach

Add a single generic renderer function to the proximal-distal renderer module
(which already hosts the statistical helper functions) and call it from both
pipelines. The function accepts column-name pairs so it works for any
paired-condition comparison without duplicating code.

The per-condition raw metric columns already exist in both DataFrames (e.g.
`area_mm2_proximal`, `area_mm2_distal` / `area_mm2_tap`, `area_mm2_stroke`),
so no new metric computation is needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add function to proximal-distal renderer, import from tap-stroke | Reuses existing stat helpers; no new files | Slightly cross-module coupling | **Chosen** |
| New shared renderer module | Clean separation | Extra file for one function; stat helpers would need moving or re-importing | Rejected |
| Extend existing strip charts with violins | No new function | Strips show 18 delta metrics, not raw paired values; different visual purpose | Rejected |

### Architecture Changes

No new modules. One new public function in an existing renderer. Two new
import + call sites in existing pipelines.

### Knowledge Base

No applicable notes. This feature is a pure rendering addition with no
coordinate-space, import-order, or GUI interaction concerns.

---

## Implementation Plan

### Phase 1: Renderer function
**Goal:** Implement `render_paired_metric_violins()` in the proximal-distal
renderer.

**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 1.1 — Add `render_paired_metric_violins()` with signature:
      `(df, metric_pairs, condition_labels, output_path, session_colors,
      neuron_type_legend)`
- [x] Task 1.2 — Each subplot: two `ax.violinplot()` calls (x=0, x=1) with
      alpha-filled bodies, per-session paired dots connected by lines, colored
      by `session_colors`
- [x] Task 1.3 — Statistical annotations above each subplot using existing
      `_compute_axis_significance()` on `(col_a - col_b)` values; format with
      `_fmt_p()`; add consistency count with Clopper-Pearson CI
- [x] Task 1.4 — Neuron-type legend at figure bottom (same pattern as
      `render_proximal_distal_population_strips`)
- [x] Task 1.5 — Save PNG + SVG (same dual-save pattern as existing renderers)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py`
  — add `render_paired_metric_violins()`

**Dependencies:** None

### Phase 2: Pipeline wiring
**Goal:** Call the new renderer from both comparison pipelines.

**Started:** 2026-06-23
**Completed:** 2026-06-23

- [x] Task 2.1 — Import `render_paired_metric_violins` in
      `rf_proximal_distal_comparison_pipeline.py`; call in pass 3 with
      metric pairs: `(area_mm2_proximal, area_mm2_distal, 'Area (mm2)')`,
      `(perimeter_mm_proximal, perimeter_mm_distal, 'Perimeter (mm)')`,
      `(mean_iff_on_contour_proximal, mean_iff_on_contour_distal, 'Mean IFF')`,
      `(peak_iff_proximal, peak_iff_distal, 'Max IFF')`; condition labels
      `('Proximal', 'Distal')`
- [x] Task 2.2 — Import `render_paired_metric_violins` in
      `rf_tap_stroke_comparison_pipeline.py`; call in pass 3 with
      corresponding tap/stroke column pairs; condition labels
      `('Tap', 'Stroke')`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`
  — import + call after existing population strips
- `code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py`
  — import + call after existing population strips

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [x] Existing `test_rf_inflection_boundary.py` passes (no regression)

### Manual Verification
- [ ] Run `spatial_compare_proximal_distal` pipeline and confirm
      `rf_proximal_distal_metric_violins.png` exists with 4 subplots
- [ ] Run `spatial_compare_tap_stroke` pipeline and confirm
      `rf_tap_stroke_metric_violins.png` exists with 4 subplots
- [ ] Visual check: violins visible, paired dots connected, p-values annotated
- [ ] Visual check: neuron-type legend present when neuron summary xlsx provided

### Edge Cases
- [ ] Sessions with NaN metrics for one condition are gracefully skipped
      (no crash, violin drawn from available data only)
- [ ] Fewer than 5 sessions: statistical annotations show "n/a" (matching
      existing strip chart gating)

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (renderer addition follows existing
      patterns; output is discoverable in the pipeline output directory)

---

## Rollback Plan

1. Revert the three modified files to their prior state
2. No data migrations or config changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Violin KDE looks poor with N~11 | Medium | Low | Matplotlib's default KDE bandwidth is reasonable for N>=8; adjust `bw_method` if needed |
| Paired lines clutter with many sessions | Low | Low | Alpha transparency on lines; N~11 is manageable |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer | ~60 lines of code | None |
| Phase 2: Pipeline wiring | ~20 lines of code | Phase 1 |

---

## References

- Prior conversation: rough_drawing.png analysis (session `e70f6a2f`)
- Existing renderer: `rf_proximal_distal_comparison_renderer.py` (strip charts,
  shift decomposition)
- Existing pipelines: `rf_proximal_distal_comparison_pipeline.py`,
  `rf_tap_stroke_comparison_pipeline.py`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- docs/development/plans/active/rf-paired-metric-violins.md
