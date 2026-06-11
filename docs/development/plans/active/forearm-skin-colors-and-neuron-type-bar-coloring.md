# Plan: Forearm Skin Colors + Neuron-Type Bar Coloring

**Date:** 2026-06-11
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/standalone-mkv-to-forearm-mesh`
**Branch:** `feature/forearm-skin-neuron-type-coloring`

---

## Overview

**What:** Propagate forearm PLY vertex colors (skin texture) as mesh
backgrounds to all per-session UV-space RF output images, and add
Okabe-Ito neuron-type color coding to the metric bar panels in
`spatial_compare_boundaries`.

**Why:** Currently only circular-crop images show real forearm skin;
all other renderers use grey `#404040` scatter dots. The metric bar
panels use uniform `steelblue`, making it impossible to distinguish
neuron types visually.

**How:** Extract a shared PolyCollection mesh-background helper,
propagate vertex colors through the rendering pipeline, persist them
in the per-session NPZ, and wire `neuron_summary_xlsx` into the
boundary comparison pipeline for Okabe-Ito bar coloring.

## Problem Statement

The `spatial_extract_boundaries` pipeline already loads forearm PLY
vertex colors and maps them to SLIM mesh vertices via KDTree, but only
the circular-crop renderer uses them. All other per-session heatmap
images (scatter+interpolated, composite, standalone interpolated, and
the center-marked heatmaps in `spatial_compare_rf_centers`) show a
grey scatter-dot background, which hides the anatomical context of the
forearm surface.

Separately, the `spatial_compare_boundaries` metric bar panels use a
uniform `steelblue` color for all sessions, making it impossible to
visually group sessions by neuron type (SAI, SAII, CT, Field, HFA).

## Goals

### In Scope

1. Replace grey scatter-dot backgrounds with forearm-skin-colored
   mesh triangles in all per-session UV-space renderers across
   `spatial_extract_boundaries` and `spatial_compare_rf_centers`.
2. Persist SLIM vertex colors in the per-session NPZ so downstream
   comparison pipelines can load them without re-deriving the SLIM
   mapping.
3. Update `NEURON_TYPE_COLORS` to the Okabe-Ito colorblind-safe
   palette (SAI=#E69F00, SAII=#56B4E9, CT=#009E73, Field=#0072B2,
   HFA=#D55E00).
4. Color-code each session's bars in `spatial_compare_boundaries`
   metric panels by neuron type, with a type-level legend.

### Out of Scope

- Changing contour overlay line colors (keep tab20 per-session).
- Changing aggregate scatter plot colors in `spatial_compare_rf_centers`
  (keep tab20).
- Changing session-gesture heatmap colors (keep viridis data matrix).
- Adding vertex-colored mesh background to cross-session plots
  (contour overlays, aggregate scatters) where multiple sessions'
  meshes would overlap.
- GPU-accelerated rendering.

## Success Criteria

- [ ] All per-gesture PNGs from `spatial_extract_boundaries` show
      forearm skin mesh instead of grey dots (scatter+interpolated,
      composite, standalone interpolated).
- [ ] Per-session heatmaps in `spatial_compare_rf_centers` show
      forearm skin mesh instead of grey dots.
- [ ] Circular-crop images continue to render correctly (no regression).
- [ ] `slim_vertex_colors` is stored in each session's
      `_population_response_fields.npz`.
- [ ] Older NPZs (without `slim_vertex_colors`) still load without
      error in comparison pipelines (backward compat).
- [ ] `NEURON_TYPE_COLORS` uses the Okabe-Ito palette.
- [ ] Metric bar panels in `spatial_compare_boundaries` show
      per-session bars colored by neuron type with a legend.
- [ ] `pytest` passes with no regressions.

---

## Technical Design

### Approach

**Mesh background:** Extract the PolyCollection mesh-drawing logic
already proven in `render_population_rf_circular_crop()` into a
reusable `_draw_forearm_mesh_background()` helper. Each renderer that
currently draws grey scatter dots calls this helper instead. When
`vertex_colors` is None, the helper falls back to grey faces.

**NPZ storage:** Add `slim_vertex_colors` (float64 RGBA, shape
(N, 4)) to the existing per-session NPZ. Comparison pipelines check
for key existence and fall back to None (same pattern as hotspot keys).

**Neuron-type coloring:** Wire `neuron_summary_xlsx` (already a DAG
parameter) into `spatial_compare_boundaries_flow()` via
`_build_pipeline_stages`. The pipeline builds a `SessionColorScheme`
via the existing `build_session_color_scheme()` and passes per-session
hex colors + a type-level legend to the bar chart renderer.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Store vertex colors in NPZ | Self-contained; comparison pipelines already load NPZ | Adds ~60KB/session | **Chosen** |
| Re-load from PLY at comparison time | No NPZ schema change | Duplicates SLIM mapping logic; needs PLY path resolution | Rejected |
| Scatter dots with per-vertex colors (no mesh) | Simpler rendering | No triangle fill; gaps between dots; less accurate surface | Rejected |

### Architecture Changes

No new modules. Changes are limited to adding parameters and a helper
function to existing renderer and pipeline modules.

The `_draw_forearm_mesh_background()` helper is private to
`rf_population_map_renderer.py` and imported by
`rf_center_comparison_renderer.py`.

---

## Implementation Plan

### Phase 1: Color Palette + Shared Helper
**Goal:** Update neuron-type colors and create the reusable mesh
background helper.

- [x] 1.1 Update `NEURON_TYPE_COLORS` in `neuron_type_colors.py` to
      Okabe-Ito hex values.
- [x] 1.2 Add `_draw_forearm_mesh_background(ax, forearm_uv,
      forearm_faces, vertex_colors=None)` to
      `rf_population_map_renderer.py`.
- [x] 1.3 Refactor `render_population_rf_circular_crop()` to call
      the new helper (DRY, no behavior change).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py`
  Update 5 hex color values.
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py`
  Add `_draw_forearm_mesh_background()`; refactor circular crop
  renderer to use it.

**Dependencies:** None

### Phase 2: Propagate Vertex Colors in spatial_extract_boundaries
**Goal:** All per-session renderers in spatial_extract_boundaries use
forearm skin mesh instead of grey scatter dots.

- [x] 2.1 Add `vertex_colors` param to `render_population_rf_map()`;
      replace grey scatter with `_draw_forearm_mesh_background()`.
- [x] 2.2 Add `vertex_colors` param to
      `render_population_rf_composite()`; replace grey scatter.
- [x] 2.3 Add `vertex_colors` and `forearm_faces` params to
      `render_population_rf_standalone_interpolated()`; replace grey
      scatter.
- [x] 2.4 Update all renderer call sites in
      `rf_population_response_field_pipeline.py` to pass
      `vertex_colors=sd.slim_vertex_colors` (and
      `forearm_faces=sd.forearm_faces` for standalone).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py`
  Add `vertex_colors` param to 3 renderer functions; replace scatter
  blocks.
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
  Pass `vertex_colors` to all renderer calls in Pass 1 and Pass 2.

**Dependencies:** Phase 1

### Phase 3: Persist Vertex Colors in NPZ + Load in Comparison Pipelines
**Goal:** Vertex colors are stored in NPZ and loaded by
`spatial_compare_rf_centers` for per-session heatmap backgrounds.

- [x] 3.1 Add `slim_vertex_colors` param to
      `_save_response_fields_npz()`; store in NPZ if not None.
- [x] 3.2 Update call site to pass `slim_vertex_colors`.
- [x] 3.3 Load `slim_vertex_colors` from NPZ in
      `rf_proximal_distal_center_pipeline.py` (with key-existence
      fallback).
- [x] 3.4 Store in `valid_data` dict; also load `forearm_faces` for
      each session (already loaded but currently only for
      `uv_to_mm_scale`).
- [x] 3.5 Add `vertex_colors` and `forearm_faces` params to
      `render_center_marked_heatmap()`; replace grey scatter with
      `_draw_forearm_mesh_background()`.
- [x] 3.6 Update call sites in
      `rf_proximal_distal_center_pipeline.py` to pass the new params.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
  `_save_response_fields_npz()` gains `slim_vertex_colors` param.
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py`
  Load vertex colors from NPZ; pass to renderer.
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py`
  Add params to `render_center_marked_heatmap()`; import and call
  mesh helper.

**Dependencies:** Phase 1, Phase 2

### Phase 4: Neuron-Type Bar Coloring in spatial_compare_boundaries
**Goal:** Metric bar panels show per-session bars colored by neuron
type with an Okabe-Ito legend.

- [x] 4.1 Add `session_colors` and `neuron_type_legend` params to
      `render_boundary_metric_panels()`; color bars per session;
      add legend.
- [x] 4.2 Add `neuron_summary_xlsx` param to
      `run_session_rf_boundary_comparison()`; build
      `SessionColorScheme`; pass to renderer.
- [x] 4.3 Add `neuron_summary_xlsx` param to
      `spatial_compare_boundaries_flow()`.
- [x] 4.4 Wire `neuron_summary_xlsx` into the `_build_pipeline_stages`
      params lambda for `spatial_compare_boundaries`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_boundary_comparison_renderer.py`
  `render_boundary_metric_panels()` gains `session_colors` +
  `neuron_type_legend` params.
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
  Accepts `neuron_summary_xlsx`; builds `SessionColorScheme`.
- `code/scripts/analysis_workflow_processing.py`
  `spatial_compare_boundaries_flow()` gains `neuron_summary_xlsx`;
  `_build_pipeline_stages` lambda wires it.

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `_draw_forearm_mesh_background` creates a PolyCollection with
      correct face count and RGBA colors from vertex_colors; falls
      back to grey when None.
- [ ] `render_boundary_metric_panels` with `session_colors` produces
      bars in the specified hex colors (spot-check via
      `ax.patches[i].get_facecolor()`).

### Integration Tests
- [ ] Existing `pytest` suite passes unchanged (renderers have
      default `vertex_colors=None`).

### Manual Verification
- [ ] Run `spatial_extract_boundaries` for one session; visually
      confirm all PNGs show forearm skin mesh instead of grey dots.
- [ ] Run `spatial_compare_boundaries`; confirm metric bar panels
      show colored bars with an Okabe-Ito neuron-type legend.
- [ ] Run `spatial_compare_rf_centers`; confirm per-session heatmaps
      show forearm skin mesh.
- [ ] Verify backward compat: load an older NPZ (without
      `slim_vertex_colors`); confirm comparison pipelines handle it
      gracefully.

### Edge Cases
- [ ] Session whose forearm PLY has no vertex colors (returns None)
      renders with grey mesh fallback, not a crash.
- [ ] NPZ missing `slim_vertex_colors` key (older file) loads
      without error and falls back to grey mesh.
- [ ] `neuron_summary_xlsx` is None (not configured) — metric panels
      fall back to uniform `steelblue` (no crash).

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention vertex-color
      propagation in the `render_population_rf_*` renderer notes and
      the `neuron_summary_xlsx` parameter for
      `spatial_compare_boundaries`.
- [ ] No README or user guide changes needed (internal pipeline tool).

---

## Rollback Plan

1. All changes are additive parameters with backward-compatible
   defaults (`vertex_colors=None`, `session_colors=None`).
2. `NEURON_TYPE_COLORS` change is the only non-backward-compatible
   edit; reverting the 5 hex values restores the old palette.
3. No migrations or database changes.
4. Rollback: revert the feature branch commits.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PolyCollection slow for large meshes | Low | Low | SLIM meshes have ~2-8K faces; PolyCollection handles this fine. Already proven in circular crop. |
| Existing NPZ consumers break on new key | Low | Med | New key is purely additive; no existing code reads `slim_vertex_colors`. Comparison pipelines use key-existence check. |
| Okabe-Ito palette change affects other plots | Med | Low | The palette is used by `stimulus_response_tuning` and `stimulus_response_instruction_tuning` — intentional: user wants a global palette update. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Palette + Helper | Small (1 file, ~30 lines) | None |
| Phase 2: Extract Boundaries Renderers | Medium (2 files, ~40 lines changed) | Phase 1 |
| Phase 3: NPZ + Compare RF Centers | Medium (3 files, ~30 lines changed) | Phase 1 |
| Phase 4: Neuron-Type Bar Coloring | Medium (3 files, ~40 lines changed) | Phase 1 |

---

## References

- Existing vertex color handling: `rf_population_map_renderer.py:562-668`
  (`render_population_rf_circular_crop`)
- Neuron type color module: `rendering/neuron_type_colors.py`
- DAG parameter wiring pattern: `analysis_workflow_processing.py:1569`
  (`neuron_summary_xlsx` for `stimulus_response_tuning`)
- Completed plan for neuron-type coloring in IFF tuning:
  `docs/development/plans/completed/iff-tuning-neuron-type-coloring.md`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py
- code/src/analysis/receptive_field_mapping/rendering/rf_boundary_comparison_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py
- docs/development/plans/active/forearm-skin-colors-and-neuron-type-bar-coloring.md
