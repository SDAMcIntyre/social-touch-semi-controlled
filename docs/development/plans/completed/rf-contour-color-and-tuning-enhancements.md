# Plan: RF Contour Color Dropdown + Stimulus Response Tuning Enhancements

**Date:** 2026-06-12
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-22 17:57
**Base Branch:** `feature/circular-crops-compare-pipelines`
**Branch:** `feature/rf-contour-color-and-tuning-enhancements`

---

## Overview

Three enhancements from meeting notes:
(1) Replace the hardcoded violet RF boundary contour with a configurable color selectable via DAG GUI dropdown (default red).
(2) In the stimulus response tuning raw_dots mode, color individual dots by a secondary variable (depth when velocity is on X-axis, and vice versa) instead of a uniform session color.
(3) Add extra output images with per-neuron min/max normalized response values to the stimulus response tuning pipeline.

## Problem Statement

1. The RF inflection boundary contour is hardcoded as violet in multiple renderer functions. There is no way to change it without editing source code.
2. Raw-dot scatter plots in stimulus response tuning use uniform session colors, making it impossible to see how a secondary variable (e.g. depth) varies across touches.
3. Neurons have different absolute firing ranges, making cross-neuron comparison difficult on raw-value plots. No normalized view is available.

## Goals

### In Scope
1. Add a `contour_color` dropdown option to the DAG GUI for `spatial_extract_boundaries` and `spatial_compare_rf_centers`, threaded through the full pipeline to all renderer call sites
2. Add `secondary_color_by` dict option to `stimulus_response_tuning` that maps each primary feature to a secondary feature for dot coloring in raw_dots mode
3. Add `normalize_per_neuron` boolean option to `stimulus_response_tuning` that produces extra output images with per-neuron [0, 1] normalized response values

### Out of Scope
- Changing contour colors in `spatial_compare_boundaries` (uses per-session tab20 colors, not a single contour color)
- Secondary coloring in sliding_window mode (only applies to raw_dots)
- Z-score or other normalization methods beyond min/max
- Interactive GUI exploration of normalized data

## Success Criteria

- [ ] DAG GUI shows a dropdown for `contour_color` under `spatial_extract_boundaries` and `spatial_compare_rf_centers` with Red as default
- [ ] Output PNGs from `spatial_extract_boundaries` show red contours (not violet)
- [ ] Changing the dropdown to another color and re-running produces contours in that color
- [ ] Raw-dots PNGs for features with `secondary_color_by` mappings show a viridis colorbar for the secondary variable
- [ ] Overlay raw-dots PNGs pool secondary values across sessions for a shared colorbar range
- [ ] With `normalize_per_neuron: true`, a `normalized/` sub-folder appears with plots where Y-axis ranges from 0 to 1
- [ ] Normalized overlay plots show all sessions on a common [0, 1] scale

---

## Technical Design

### Approach

All three features follow the established option-threading pattern: YAML config option -> workflow script extraction -> Prefect @flow parameter -> pipeline function parameter -> renderer parameter. The `_OPTION_ENUMS` dict in `task_detail_panel.py` auto-renders QComboBox dropdowns for any matching option key.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Hardcode new color (red) | Zero effort | Loses flexibility | Rejected |
| Dropdown from `_OPTION_ENUMS` | Follows established pattern, zero GUI code | Limited to pre-defined choices | Chosen |
| Free-text color input | Any matplotlib color | Error-prone, no validation | Rejected |
| Normalize X and Y axes | Full normalization | Loses physical units on X-axis | Rejected (user chose Y-only) |

### Architecture Changes

No new modules. Purely parameter threading through existing functions. The rendering functions gain optional parameters with backward-compatible defaults.

---

## Implementation Plan

### Phase 1: RF Contour Color Dropdown
**Started:** 2026-06-12
**Completed:** 2026-06-12
**Goal:** Replace all hardcoded violet contour colors with a configurable option, defaulting to red.

- [x] Task 1.1 — Add `"contour_color"` to `_OPTION_ENUMS` in `task_detail_panel.py` with choices: Red (default), Violet, White, Cyan, Lime, Yellow, Orange, Magenta
- [x] Task 1.2 — Add `contour_color: red` to `spatial_extract_boundaries.options` and `spatial_compare_rf_centers.options` in `analyse_workflow_processing_dag.yaml`
- [x] Task 1.3 — Add `contour_color` param to `spatial_extract_boundaries_flow` and `spatial_compare_rf_centers_flow` in `analysis_workflow_processing.py`; add DAG params lambda extraction
- [x] Task 1.4 — Add `contour_color` param to `run_population_response_field_extraction()` in `rf_population_response_field_pipeline.py`; forward to all renderer calls
- [x] Task 1.5 — Add `contour_color` param to `run_proximal_distal_center_comparison()` in `rf_proximal_distal_center_pipeline.py`; forward to `render_center_marked_heatmap()` calls
- [x] Task 1.6 — Add `contour_color` param to `_draw_inflection_boundary()`, `render_population_rf_map()`, `render_population_rf_standalone_interpolated()`, `render_population_rf_composite()` in `rf_population_map_renderer.py`; replace `color='violet'`
- [x] Task 1.7 — Add `contour_color` param to `render_center_marked_heatmap()` in `rf_center_comparison_renderer.py`; replace `color='violet'`
- [x] Task 1.8 — Thread `contour_color` through `compute_inflection_boundary()` diagnostic renders in `rf_inflection_boundary.py`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add `contour_color` to `_OPTION_ENUMS`
- `configs/analyse_workflow_processing_dag.yaml` — add option to two tasks
- `code/scripts/analysis_workflow_processing.py` — add param to two flows + two DAG lambda extractions
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — thread param
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — thread param
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — add param to 4 functions, replace violet
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py` — add param, replace violet
- `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py` — thread param to diagnostic renders

**Dependencies:** None

### Phase 2: Raw Dots Secondary Variable Coloring
**Started:** 2026-06-12
**Completed:** 2026-06-12
**Goal:** In raw_dots mode, color each dot by a secondary stimulus feature using a continuous colormap and colorbar.

- [x] Task 2.1 — Add `secondary_color_by` dict option to `stimulus_response_tuning` in `analyse_workflow_processing_dag.yaml`
- [x] Task 2.2 — Add `secondary_color_by` param to `stimulus_response_tuning_flow` in `analysis_workflow_processing.py`; thread to options dict and DAG params lambda
- [x] Task 2.3 — Extract `secondary_color_by` in `run_response_tuning()`; in the raw_dots branch, load secondary feature column alongside primary, pass arrays to renderers, change overlay tuple from 2-element to 3-element
- [x] Task 2.4 — Add `secondary_vals`, `secondary_label`, `secondary_cmap` params to `render_session_raw_dots()` in `rf_response_tuning_renderer.py`; use `c=secondary_vals, cmap=...` with colorbar when present
- [x] Task 2.5 — Add same params to `render_overlay_raw_dots()`; pool secondary values for shared vmin/vmax; add colorbar

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add `secondary_color_by` mapping
- `code/scripts/analysis_workflow_processing.py` — add param to flow + DAG lambda
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py` — extract secondary column, change overlay data shape
- `code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py` — add secondary color support to two render functions

**Dependencies:** Phase 1 (shared YAML/workflow file edits)

### Phase 3: Per-Neuron Min/Max Normalized Output Images
**Started:** 2026-06-12
**Completed:** 2026-06-12
**Goal:** Produce extra output images where response values are normalized to [0, 1] per session/neuron.

- [x] Task 3.1 — Add `normalize_per_neuron: true` option to `stimulus_response_tuning` in `analyse_workflow_processing_dag.yaml`
- [x] Task 3.2 — Add `normalize_per_neuron` param to `stimulus_response_tuning_flow` in `analysis_workflow_processing.py`; thread to options dict and DAG params lambda
- [x] Task 3.3 — In `run_response_tuning()`, pre-compute per-session normalization ranges (min/max of response column across all gesture subsets) during Pass 1
- [x] Task 3.4 — After Pass 2, add a conditional Pass 3 gated by `normalize_per_neuron`: re-iterate feature x gesture x session loops, apply linear normalization to response values, render with `iff_ylim=(0.0, 1.0)` and `iff_ylabel="Normalized response"`, save to `normalized/` sub-folder
- [x] Task 3.5 — Render normalized overlay plots: each session normalized independently, then overlaid with shared [0, 1] ylim

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add boolean option
- `code/scripts/analysis_workflow_processing.py` — add param to flow + DAG lambda
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py` — add normalization pass

**Dependencies:** Phase 2 (must account for 3-tuple overlay data)

---

## Testing Plan

### Manual Verification
- [ ] Run `spatial_extract_boundaries` with default config -> confirm output PNGs show red contours
- [ ] Change dropdown to Cyan, re-run -> confirm cyan contours
- [ ] Run `spatial_compare_rf_centers` -> confirm red centroid markers
- [ ] Set `binning_strategy: raw_dots` with `secondary_color_by` configured -> confirm per-session PNGs show viridis-colored dots with colorbar
- [ ] Check overlay PNGs show shared colorbar range across sessions
- [ ] Features NOT in `secondary_color_by` -> confirm they still render with uniform session color (no regression)
- [ ] Enable `normalize_per_neuron: true` -> confirm `normalized/` sub-folder appears
- [ ] Check normalized per-session PNGs have Y-axis [0, 1]
- [ ] Check normalized overlay PNGs show all sessions on common [0, 1] scale
- [ ] Disable `normalize_per_neuron` -> confirm no `normalized/` output (no regression)

### Edge Cases
- [ ] Session with constant response (max == min) -> normalized values should be 0.0, not NaN/crash
- [ ] Feature without `secondary_color_by` mapping -> falls back to uniform session color
- [ ] Secondary feature column missing from a session's DataFrame -> falls back to uniform color with no crash

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention `contour_color` option in population response fields section
- [ ] No user-facing docs needed — options are self-documenting in the DAG GUI

---

## Rollback Plan

All changes are additive parameters with backward-compatible defaults. To revert:
1. Remove `contour_color`, `secondary_color_by`, `normalize_per_neuron` from `analyse_workflow_processing_dag.yaml`
2. Revert the param additions in the workflow script, pipeline, and renderer files
3. No data migrations or breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| contour_color "red" clashes with peak marker (also red) in center comparison | Low | Low | Peak uses `*` marker, centroid uses `+` — visually distinguishable |
| secondary_color_by with invalid column name | Low | Med | Validate column exists in DataFrame; fall back to uniform color if missing |
| Normalization pass doubles render time | Med | Low | Gated by boolean option; only runs when explicitly enabled |
| 3-tuple overlay data breaks existing code | Low | High | Always use 3-tuple with None sentinel; unpack conditionally |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Contour color | ~1 hour | None |
| Phase 2: Secondary color | ~1.5 hours | Phase 1 |
| Phase 3: Normalization | ~1 hour | Phase 2 |

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py
- code/src/utils/gui/dag_launcher/task_detail_panel.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rf-contour-color-and-tuning-enhancements.md
