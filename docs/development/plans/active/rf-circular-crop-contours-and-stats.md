# Plan: RF Circular Crop Contour Lines, Centroid Marker & Proximal-Distal Statistical Tests

**Date:** 2026-06-22
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/rf-circular-crop-contours-and-stats`

---

## Overview

The circular RF crop images currently show only a flat IFF heatmap clipped to a circle. This plan adds topographic-style contour lines and a red centroid cross to these images, making the spatial structure of the receptive field immediately visible. It also strengthens the proximal-distal statistical analysis by adding an exact sign test and a consistency count annotation alongside the existing Wilcoxon signed-rank test.

## Problem Statement

1. **Missing topographic cues** — the circular RF crops show color-mapped IFF but lack contour lines, making it hard to read the spatial gradient structure at a glance. Topographic contour lines are standard in published somatosensory RF mapping figures.
2. **No centroid marker** — the centroid is computed and stored but not rendered on the circular crops, so the reader cannot see the RF center relative to the heatmap structure.
3. **Limited statistical evidence** — the existing Wilcoxon test reports a p-value but does not indicate how many neurons agree in direction or provide a nonparametric consistency measure. With N=5-11, readers need to see both the test and the raw consistency count.

## Goals

### In Scope
1. Add IFF contour lines (6 evenly-spaced levels) to all circular crop PNGs
2. Add red cross centroid marker to all circular crop PNGs
3. Add exact sign test (binomial) to the proximal-distal population strip chart
4. Add consistency count annotation with Clopper-Pearson CI to the strip chart

### Out of Scope
- Changing the heatmap colormap or color normalization
- Adding contour lines to the full-map (non-circular) renderings (they already have the inflection boundary)
- Effect size metrics (rank-biserial r, Cohen's d, bootstrap CI) — may be added later
- Contour labels (numeric annotations on the lines) — keep it clean for now

## Success Criteria

- [ ] Circular crop PNGs show 6 white contour lines evenly spaced between min and max IFF
- [ ] Circular crop PNGs show a red `+` marker at the centroid position
- [ ] Contour lines and centroid marker are clipped to the circular boundary (no leaking)
- [ ] Population strip chart panels show Wilcoxon p, sign-test p, and consistency count
- [ ] Existing tests pass without modification
- [ ] All changes are backward-compatible (old callers without new params produce unchanged output)

---

## Technical Design

### Approach

**Contour lines:** Use `ax.contour()` overlaid on the existing `pcolormesh`, with levels computed from the actual finite data range in `interp_grid`. The `contour()` LineCollection objects are automatically covered by the existing `ax.collections` clip loop, so clipping to the circular boundary requires no special handling.

**Centroid marker:** Use `ax.plot()` with `marker='+'` — identical to the existing `_draw_inflection_boundary()` pattern at line 223 of `rf_population_map_renderer.py`. The Line2D goes into `ax.lines`, so the clip loop must be extended to also iterate `ax.lines`.

**Sign test:** `scipy.stats.binomtest(n_positive, n_nonzero, 0.5)` provides an exact two-sided p-value. Consistency count is the raw fraction `k/N` with Clopper-Pearson CI via `scipy.stats.binom.ppf`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `contourf()` (filled contours) | Smoother appearance | Replaces the existing pcolormesh, loses mesh background visibility | Rejected |
| `contour()` (line contours) | Overlays cleanly on existing heatmap, standard topographic look | Need to pick line color for contrast | **Chosen** |
| Levels from vmin/vmax (global) | Consistent levels across sessions | May produce no contours if local data range is narrow | Rejected |
| Levels from local data range | Contours always show local RF structure | Levels differ across sessions | **Chosen** |
| Percentile-based levels | Better for skewed distributions | Less intuitive than even spacing; user requested even | Rejected |

### Architecture Changes

No new modules. Changes are confined to:
- One rendering function gains optional parameters
- Three pipeline files pass those parameters at existing call sites
- One renderer function gains additional statistical annotations

Reusable pattern: the `_draw_inflection_boundary()` function at `rf_population_map_renderer.py:215` already implements the exact centroid marker style (`color='red'`, `marker='+'`, `markersize=8`, `zorder=7`).

---

## Implementation Plan

### Phase 1: Renderer — contour lines and centroid marker
**Goal:** Add optional contour and centroid drawing to the circular crop function

**Tasks:**
- [x] Task 1.1 — Add 5 optional parameters to `render_population_rf_circular_crop()` signature
- [x] Task 1.2 — Add contour rendering block (after pcolormesh, before clip circle): compute levels from finite data, call `ax.contour()`, guard flat/empty data
- [x] Task 1.3 — Add centroid marker block: `ax.plot()` with red `+`
- [x] Task 1.4 — Extend clip loop to also iterate `ax.lines`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — modify `render_population_rf_circular_crop()` (lines 606-706)

**Dependencies:** None

### Phase 2: Callers — pass contour and centroid parameters
**Goal:** Wire up the new parameters at all 7 call sites across 3 pipelines

**Tasks:**
- [x] Task 2.1 — Proximal-distal pipeline: add `centroid_uv=centroid_uv_gtype, contour_levels=6` to centroid-centered crop (line 526) and peak-centered crop (line 545)
- [x] Task 2.2 — Session boundary comparison pipeline: add `centroid_uv=gdata['centroid_uv'], contour_levels=6` (line 160)
- [x] Task 2.3 — Population response field pipeline: add `centroid_uv=np.array(all_boundary.centroid_uv), contour_levels=6` to all 4 calls in the loop (lines 632, 658)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py` — 2 call sites
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py` — 1 call site
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — 4 call sites

**Dependencies:** Phase 1

### Phase 3: Statistical tests — sign test and consistency count
**Goal:** Add sign test p-value and consistency count annotation to the proximal-distal strip chart

**Tasks:**
- [x] Task 3.1 — After the existing Wilcoxon block (lines 473-486), add exact sign test via `scipy.stats.binomtest`
- [x] Task 3.2 — Add consistency count `k/N` with Clopper-Pearson CI annotation
- [x] Task 3.3 — Stack the three annotation lines (Wilcoxon p, Sign p, consistency count) at the top of each strip panel

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py` — modify `render_proximal_distal_population_strips()` (lines 442-512)

**Dependencies:** None (independent of Phases 1-2)

---

## Testing Plan

### Unit Tests
- [ ] Existing tests pass (`pytest code/tests/`)
- [ ] `test_rf_inflection_boundary.py` passes unchanged (boundary computation not affected)

### Manual Verification
- [ ] Run the proximal-distal pipeline on one session and inspect circular crop PNGs: 6 white contour lines visible, red centroid cross visible, both clipped within circle
- [ ] Run the cross-session aggregate and inspect population strip chart: all three annotations (Wilcoxon p, Sign p, consistency count) visible per metric panel
- [ ] Compare centroid-centered vs peak-centered crops: centroid cross should be at circle center vs offset

### Edge Cases
- [ ] Session with flat IFF data (all identical values) — no contour lines drawn, no error
- [ ] Session with very sparse data (mostly NaN grid) — contour lines only appear where data exists
- [ ] Metric where all deltas are zero — sign test skipped gracefully (n_nonzero = 0)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention contour/centroid params in the population response fields section
- [ ] No user-facing guide needed (pipeline output only, no new CLI flags)

---

## Rollback Plan

All changes are backward-compatible via default parameter values. To revert:
1. Remove the new parameter-passing from the 3 pipeline files (callers revert to defaults)
2. Remove the contour/centroid/stats blocks from the two renderer functions
3. No data migration needed — output PNGs are regenerated on each pipeline run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Contour lines not visible on inferno colormap | Low | Low | White at alpha=0.7 tested for contrast; tunable via `contour_color` param |
| Clip loop misses contour artists | Low | Medium | Verified: `ax.contour()` adds to `ax.collections`, covered by existing loop |
| Sign test p-value always non-significant at small N | Medium | Low | Expected — the sign test is conservative; the consistency count provides the complementary interpretable measure |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer | ~30 lines of code | None |
| Phase 2: Callers | ~14 lines across 3 files | Phase 1 |
| Phase 3: Stats | ~30 lines of code | None |

---

## References

- Approved plan sketch: `.claude/plans/i-want-to-add-polymorphic-salamander.md`
- Existing centroid marker pattern: `rf_population_map_renderer.py:215-226` (`_draw_inflection_boundary`)
- Existing Wilcoxon test: `rf_proximal_distal_comparison_renderer.py:473-486`
- Web research: evenly-spaced `contour()` levels are standard for topographic RF maps; 50% and 90% contours are common in microneurography literature but the user specifically requested even spacing

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- docs/development/plans/active/rf-circular-crop-contours-and-stats.md
