# Plan: RF Proximal-Distal Aggregate Figure Enhancements

**Date:** 2026-06-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-tap-stroke-comparison`
**Branch:** `feature/rf-proximal-distal-aggregate-enhancements`

---

## Overview

Enhance the proximal-distal RF comparison aggregate figures with SVG vector export, neuron-type-colored variants of the scatter plots, inline session ID labels, and marker shape legends. The current aggregate scatter plots use arbitrary tab20 colors with a per-session legend list that doesn't convey neuron type — a neuron-type-colored variant with inline labels will be more informative for publication and analysis.

## Problem Statement

The three aggregate scatter plots (`rf_center_proximal_distal_aggregate.png`, `rf_hotspot_proximal_distal_aggregate.png`, `rf_contour_center_proximal_distal_aggregate.png`) have three limitations:

1. **No vector export** — only PNG at 120 dpi, unusable for publication figures that require SVG/PDF
2. **No neuron-type coloring** — lines use tab20 colormap indexed by session order, hiding afferent type grouping. Other aggregate figures in the same pipeline (metric deltas, strip charts, shift decomposition) already use neuron-type colors.
3. **Legend clutter** — each session gets its own legend entry (color swatch + session ID). With 11 sessions this dominates the figure. The shift decomposition already places session IDs as inline text at arrow endpoints — a better pattern.
4. **No marker explanation** — circle = proximal and diamond = distal, but this is not documented in any legend.

## Goals

### In Scope
1. SVG vector export alongside PNG for all 6 cross-session aggregate render functions
2. A new neuron-type-colored variant (`*_by_type.png` + `*_by_type.svg`) for each of the 3 scatter aggregate plots
3. Inline session ID text labels near the proximal endpoint in the by-type variant
4. Neuron-type patch legend in the by-type variant (replaces per-session legend)
5. Marker shape legend (circle = proximal, diamond = distal) on both original and by-type scatter aggregates

### Out of Scope
- Changing per-session heatmap/contour overlay figures (pixel-heavy, SVG not useful)
- Changing the existing tab20-colored aggregate figures beyond adding marker legend + SVG
- Adding neuron-type coloring to metric deltas / strip charts / shift decomposition (already have it)

## Success Criteria

- [ ] Each of the 6 cross-session aggregate PNGs has a matching `.svg` alongside
- [ ] Three new `*_by_type.png` + `*_by_type.svg` files appear for scatter aggregates
- [ ] By-type figures color lines by neuron type (Okabe-Ito palette from `NEURON_TYPE_COLORS`)
- [ ] By-type figures show session ID text near each proximal marker
- [ ] By-type figures have neuron-type patch legend (not per-session legend)
- [ ] All scatter aggregates (original + by-type) show marker shape legend (circle/diamond)
- [ ] Original tab20-colored aggregates remain unchanged (except marker legend + SVG addition)
- [ ] `pytest` passes with no regressions

---

## Technical Design

### Approach

Add optional parameters (`session_colors`, `session_neuron_types`, `neuron_type_legend`) to the three scatter aggregate render functions. When provided, they generate a second figure with neuron-type coloring. SVG export is added unconditionally to all 6 aggregate functions. This follows the existing pattern where `render_shift_decomposition` and `render_proximal_distal_metric_deltas` already accept these same parameters.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add params to existing functions, generate second figure | Minimal code, consistent with other renderers | Slight complexity in single function | **Chosen** |
| Separate `render_*_aggregate_by_type()` functions | Clean separation | Heavy duplication of the 3 nearly-identical scatter functions | Rejected |
| Replace tab20 plots with neuron-type plots | Simpler, fewer output files | Loses per-session unique-color view | Rejected |

### Architecture Changes

No new modules. Two existing files modified:

- **Renderer** — 6 functions gain SVG export; 3 scatter functions gain neuron-type variant rendering
- **Pipeline** — passes color scheme data to the 3 scatter aggregate calls (already computed, just not forwarded)

### Key code to reuse

- `neuron_type_colors.py::SessionColorScheme` — provides `session_color`, `session_neuron_type`, `type_color`
- `neuron_type_colors.py::build_session_color_scheme()` — already called in pipeline (line 885)
- SVG save pattern from `rf_boundary_comparison_renderer.py:76`
- Neuron-type legend patch pattern from `render_shift_decomposition` (lines 649–659)
- Inline text label pattern from `render_shift_decomposition:642`

---

## Implementation Plan

### Phase 1: SVG Export + Marker Legend
**Goal:** Add vector export to all 6 cross-session aggregate functions and marker shape legend to the 3 scatter aggregates.

**Tasks:**
- [x] Task 1.1 — Add SVG savefig call after each PNG savefig in `render_proximal_distal_aggregate`, `render_proximal_distal_hotspot_aggregate`, `render_proximal_distal_contour_center_aggregate`, `render_proximal_distal_metric_deltas`, `render_proximal_distal_population_strips`, `render_shift_decomposition`
- [x] Task 1.2 — Add marker shape legend (circle = Proximal, diamond = Distal) to the 3 scatter aggregate functions using `matplotlib.lines.Line2D` marker handles

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py` — 6 SVG export additions + 3 marker legend additions

**Dependencies:** None

### Phase 2: Neuron-Type Variant + Pipeline Wiring
**Goal:** Generate neuron-type-colored by-type variant of each scatter aggregate when color scheme is available.

**Tasks:**
- [x] Task 2.1 — Add `session_colors`, `session_neuron_types`, `neuron_type_legend` parameters to `render_proximal_distal_aggregate`, `render_proximal_distal_hotspot_aggregate`, `render_proximal_distal_contour_center_aggregate`
- [x] Task 2.2 — In each function, when `session_colors` is provided, generate a second figure (`*_by_type.png` + `*_by_type.svg`) with: neuron-type line colors, inline session ID text near proximal endpoint, neuron-type patch legend, marker shape legend
- [x] Task 2.3 — In the pipeline, extract `session_neuron_types` from the `SessionColorScheme` alongside `session_colors` and `neuron_type_legend`, and pass all three to the 3 scatter aggregate render calls

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py` — neuron-type variant logic in 3 scatter functions
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py` — pass color scheme to scatter aggregate calls

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] Existing `pytest` suite passes without regressions

### Manual Verification
- [ ] Run the `spatial_compare_proximal_distal` pipeline task
- [ ] Confirm 6 new `.svg` files alongside existing aggregate PNGs
- [ ] Confirm 3 new `*_by_type.png` + `*_by_type.svg` files for scatter aggregates
- [ ] Open by-type PNGs — verify lines colored by neuron type, session ID text near proximal markers, neuron-type + marker shape legends
- [ ] Open original tab20 PNGs — verify unchanged except marker legend addition
- [ ] Open SVG files in a vector editor — verify they are valid vector graphics

### Edge Cases
- [ ] Pipeline run without `neuron_summary_xlsx` — by-type figures should be skipped, original figures unaffected
- [ ] Session with overlapping proximal/distal points — text labels may overlap but should still render

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (no new public API or architecture shift)

---

## Rollback Plan

1. Revert the two modified files — changes are additive (new params with defaults, new output files)
2. No data migration — only new output files are created alongside existing ones
3. Existing figures and pipeline behavior unchanged when `neuron_summary_xlsx` is not provided

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SVG file size large for complex figures | Low | Low | SVG is only for aggregate plots (few elements), not pixel-heavy heatmaps |
| Text label overlap when sessions cluster | Medium | Low | Small fontsize (6–7pt) + offset from marker; acceptable for analysis figures |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: SVG + marker legend | Small (~30 min) | None |
| Phase 2: By-type variant + wiring | Medium (~45 min) | Phase 1 |

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- docs/development/plans/active/rf-proximal-distal-aggregate-enhancements.md
