# Plan: Replace Jet Colormap with Perceptually Uniform Default

**Date:** 2026-06-08
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/uv-to-mm-conversion`
**Branch:** `feature/replace-jet-colormap`

---

## Overview

Replace every hardcoded `'jet'` colormap with `'inferno'` (perceptually uniform,
built into matplotlib) across the RF population rendering pipeline, interactive
GUI viewers, and DAG config. The colormap remains user-selectable from a
dropdown in the DAG launcher so researchers can switch at will.

## Problem Statement

The `jet` (rainbow) colormap is used throughout the RF population heatmap
pipeline and interactive viewers. Peer-reviewed research shows jet creates
false contour boundaries in smooth data, hides real gradients in cyan/yellow
bands, distorts perceived values by >7% (CIEDE2000), and is inaccessible to
~8% of male viewers due to red-green CVD. See
`docs/development/knowledge-base/investigation-jet-colormap-perceptual-problems.md`
for the full literature review.

## Goals

### In Scope

1. Replace `'jet'` default with `'inferno'` in all renderer functions, pipeline
   functions, workflow scripts, and the DAG config YAML
2. Add a `cmap` parameter to the 4 renderer functions that currently hardcode
   `'jet'` and thread it through from the pipeline caller
3. Update the DAG launcher dropdown to offer perceptually uniform options and
   default to `'inferno'`
4. Update interactive GUI viewers that hardcode `'jet'` in PyVista `add_mesh()`
   calls
5. Update the `rf_cluster_gallery_viewer` default from `'jet'` to `'inferno'`

### Out of Scope

- **`RdYlBu_r` in `rf_cluster_visualizer.py`** — diverging colormap used for
  relative spike metrics; semantically different from sequential density maps
- **Diagnostic plots in `rf_inflection_boundary.py`** — debug visualisations
  not intended for publication
- **Depth video rendering** (`extract_color_to_mp4.py`) — preprocessing stage,
  not analysis output
- **Trajectory visualiser** (`trajectory_3d_visualiser.py`) — postprocessing,
  not RF analysis
- **Adding `cmcrameri`/`batlow` dependency** — `inferno` is built-in; `batlow`
  can be added to the dropdown list later if desired

## Success Criteria

- [ ] No remaining `cmap='jet'` or `cmap: "jet"` in any analysis renderer or
      pipeline function
- [ ] All 4 previously-hardcoded renderer functions accept a `cmap` parameter
- [ ] The DAG config defaults to `inferno` and the dropdown offers perceptually
      uniform alternatives
- [ ] Interactive GUI viewers default to `inferno`
- [ ] Running `spatial_extract_boundaries` and `spatial_compare_rf_centers`
      produces heatmaps rendered with `inferno`
- [ ] Selecting a different colormap from the DAG launcher dropdown correctly
      propagates through the full pipeline

---

## Technical Design

### Approach

Thread a `cmap` parameter through the rendering call chain (renderer ←
pipeline ← workflow ← DAG config) and change all defaults from `'jet'` to
`'inferno'`. The infrastructure for configurability already exists in the
upper layers (DAG config, workflow functions, pipeline entry points); the gap
is in 4 renderer functions that hardcode `'jet'` and the pipeline call sites
that don't forward the `cmap` they already receive.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `inferno` (matplotlib built-in) | Zero dependencies, high contrast, CVD-safe | Very different aesthetic from jet | **Chosen** |
| `batlow` (cmcrameri package) | Closest to jet's wide-hue aesthetic | Adds external dependency | Rejected for now |
| `viridis` (matplotlib default) | Most widely recognised | Narrower hue range | Available in dropdown |
| Centralised colormap constant | Single place to change | Over-engineering for ~15 call sites | Rejected |

### Architecture Changes

No new modules or classes. Changes are parameter additions and default value
swaps in existing functions.

**Key pattern:** the existing `render_population_rf_map()` already accepts
`cmap: str = "jet"` — the 4 sibling functions adopt the same signature
pattern.

---

## Implementation Plan

### Phase 1: Renderer + Pipeline (core fix)

**Started:** 2026-06-08
**Completed:** 2026-06-08

**Goal:** Add `cmap` parameter to hardcoded renderers and thread it from the
pipeline.

**Tasks:**

- [x] Add `cmap: str = "inferno"` parameter to `render_population_rf_standalone_interpolated()`
      and replace the hardcoded `cmap='jet'` at the `pcolormesh()` call
- [x] Add `cmap: str = "inferno"` parameter to `render_population_rf_colorbar()`
      and replace the hardcoded `cmap='jet'` at the `ScalarMappable()` call
- [x] Add `cmap: str = "inferno"` parameter to `render_population_rf_circular_crop()`
      and replace the hardcoded `cmap='jet'` at the `pcolormesh()` call
- [x] Add `cmap: str = "inferno"` parameter to `render_population_rf_composite()`
      and replace all 3 hardcoded `cmap='jet'` (scatter, pcolormesh, ScalarMappable)
- [x] Change the existing default in `render_population_rf_map()` from `"jet"`
      to `"inferno"`
- [x] In `rf_population_response_field_pipeline.py`, pass `cmap=cmap` to the 4
      renderer calls in PASS 2 that currently omit it (composite, standalone
      interpolated, colorbar, circular crop)
- [x] Change the existing default in `render_rf_center_with_heatmap()` from
      `"jet"` to `"inferno"` (`rf_center_comparison_renderer.py`)

**Files Modified:**

- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` —
  add `cmap` parameter to 4 functions, change default in 1 function
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` —
  forward `cmap=cmap` at 4 call sites in PASS 2
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py` —
  change default from `"jet"` to `"inferno"`

**Dependencies:** None

### Phase 2: Workflow scripts + DAG config

**Started:** 2026-06-08
**Completed:** 2026-06-08

**Goal:** Update all upstream defaults so the entire pipeline defaults to
`inferno`.

**Tasks:**

- [x] Change default `cmap` in `spatial_extract_boundaries_flow()` from `"jet"`
      to `"inferno"`
- [x] Change default `cmap` in `spatial_compare_rf_centers_flow()` from `"jet"`
      to `"inferno"`
- [x] Change `.get("cmap", "jet")` fallbacks to `.get("cmap", "inferno")` in
      the DAG handler section
- [x] Change the pipeline default in `run_population_response_field_extraction()`
      from `"jet"` to `"inferno"`
- [x] Change the pipeline default in `run_proximal_distal_center_comparison()`
      from `"jet"` to `"inferno"`
- [x] Update `configs/analyse_workflow_processing_dag.yaml`: change `cmap: jet`
      to `cmap: inferno` in both `spatial_extract_boundaries` and
      `spatial_compare_rf_centers` tasks

**Files Modified:**

- `code/scripts/analysis_workflow_processing.py` — change 4 default values
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` —
  change default
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` —
  change default
- `configs/analyse_workflow_processing_dag.yaml` — change 2 `cmap` values

**Dependencies:** Phase 1

### Phase 3: GUI dropdowns + interactive viewers

**Started:** 2026-06-08
**Completed:** 2026-06-08

**Goal:** Update GUI defaults and dropdown options.

**Tasks:**

- [x] In `task_detail_panel.py`, reorder the `cmap` enum list: put `inferno`
      first (as default), add `viridis`/`plasma`/`magma`/`cividis`, keep `jet`
      available at the end
- [x] In `rf_cluster_gallery_viewer.py`, change `_GallerySettings.cmap` default
      from `"jet"` to `"inferno"` and reorder `_COLORMAPS` list to put `inferno`
      first
- [x] Update hardcoded `cmap="jet"` in PyVista `add_mesh()` calls in 5
      interactive explorers:
      - `touch_population_explorer.py`
      - `touch_playback_explorer.py` (3 occurrences)
      - `rf_surface_viewer.py`
      - `rf_feature_space_explorer.py`
      - `single_touch_rf_explorer.py`

**Files Modified:**

- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — update `_OPTION_ENUMS["cmap"]`
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` —
  change default and reorder list
- 5 GUI explorer files — replace `cmap="jet"` with `cmap="inferno"`

**Dependencies:** None (can run in parallel with Phase 1-2)

---

## Testing Plan

### Manual Verification

- [ ] Run `spatial_extract_boundaries` on one session — confirm output PNGs
      use inferno (no rainbow colours)
- [ ] Run `spatial_compare_rf_centers` on one session — confirm centre-marked
      heatmaps use inferno
- [ ] Open the DAG launcher GUI — confirm the `cmap` dropdown defaults to
      `inferno` and lists perceptually uniform options
- [ ] Select `jet` from the dropdown — confirm pipeline still produces jet
      output (backward compatibility)
- [ ] Select `viridis` from the dropdown — confirm it propagates correctly
- [ ] Open `rf_cluster_gallery_viewer` — confirm default colormap is inferno
- [ ] Open one of the interactive explorers (e.g. `touch_population_explorer`) —
      confirm mesh colouring uses inferno

### Edge Cases

- [ ] DAG config has no `cmap` key at all — confirm the `.get()` fallback
      produces `"inferno"`, not `"jet"`
- [ ] Invalid colormap name in YAML — confirm matplotlib raises a clear error
      (fail-fast, no silent fallback)

---

## Documentation Plan

- [ ] Update `investigation-jet-colormap-perceptual-problems.md` status from
      "replacement not yet implemented" to "implemented" with date
- [ ] No CLAUDE.md changes needed (no architectural change)

---

## Rollback Plan

All changes are default-value swaps and parameter additions. To revert:

1. `git revert` the merge commit — restores all `'jet'` defaults
2. No data migration needed — output PNGs are regenerated on each run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing figures become inconsistent with new output | Med | Low | Re-run pipeline on all sessions after merge |
| Researchers prefer jet aesthetically | Low | Low | Jet remains selectable from the dropdown |
| A caller not found in this audit still hardcodes jet | Low | Low | Grep for `'jet'` during review |

---

## References

- Investigation: `docs/development/knowledge-base/investigation-jet-colormap-perceptual-problems.md`
- Crameri et al. (2020), Nature Communications 11, 5444
- Smart & Szafir (2020), UW Interactive Data Lab

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py
- code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py
- code/src/analysis/receptive_field_mapping/gui/rf_surface_viewer.py
- code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py
- code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py
- code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py
- code/src/utils/gui/dag_launcher/task_detail_panel.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/replace-jet-colormap.md
