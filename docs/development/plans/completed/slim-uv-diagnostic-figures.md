# Plan: SLIM UV Step-by-Step Diagnostic Figures

**Date:** 2026-05-21
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/radar-groups-configurable-features`
**Branch:** `feature/slim-uv-diagnostic-figures`

---

## Overview

Add a step-by-step diagnostic figure system to the SLIM UV precomputation
pipeline. When enabled, it produces 6 numbered PNG files showing every
transformation from raw mesh to final UV, with a skin-normal camera angle
computed from the mesh geometry. This replaces guesswork with a visual
pipeline trace when sessions produce unexpected projections.

## Problem Statement

2 out of 12 sessions produce unexpected SLIM UV projections. The current QC
figures (`slim_qc_figures.py`) only show the *final* 3D mesh and UV layout.
The 3D view uses matplotlib's default angle, which for some sessions looks
edge-on rather than normal to the skin surface. There is no way to tell which
pipeline step diverged: raw mesh quality, cleaning artefacts, centroid
placement near the boundary, bad harmonic/Tutte init, or SLIM optimisation
itself.

## Goals

### In Scope
1. Expose intermediate data from `flatten_slim()` via an optional diagnostics dict
2. Collect all intermediates in `precompute_forearm_slim_uv()` when a flag is set
3. Generate 6 step-numbered diagnostic PNGs per session in a `diagnostics/` subfolder
4. Compute a skin-normal view angle from area-weighted mean face normals
5. Show dual 3D views: auto skin-normal + saved RF camera settings (when available)
6. Wire the feature through the DAG config as a `save_diagnostics` option

### Out of Scope
- Interactive/GUI-based diagnostic viewer
- Fixing the 2 failing sessions (this feature diagnoses; the fix comes after)
- Changing the existing QC figures (`save_slim_qc_figures` remains untouched)
- Adding diagnostics to other pipelines (RF simple, cluster visualisation, etc.)

## Success Criteria

- [ ] 6 step PNGs generated per session when `save_diagnostics: true`
- [ ] Skin-normal view shows the forearm from above the skin in all 12 sessions
- [ ] Camera-settings view matches the orientation from `rf_camera_settings.json`
- [ ] No diagnostic files generated when `save_diagnostics: false` (default)
- [ ] Existing QC figures unchanged (backward compatible)
- [ ] All 12 sessions run to completion without error

---

## Technical Design

### Approach

Thread a mutable `diagnostics: dict | None` through `flatten_slim()` so it
records the init UV, init method, and trimming state without changing the
return type. `precompute_forearm_slim_uv()` captures additional intermediates
(raw mesh, cleaned mesh, centroid, boundary) and delegates to a new
`save_slim_diagnostic_figures()` orchestrator.

For the 3D camera, compute the area-weighted mean face normal of the mesh and
convert to matplotlib `(elev, azim)`. Load the per-session RF camera settings
(if available) for a second view via `camera_settings_to_rotation()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Mutable diagnostics dict | Efficient, exact consistency with pipeline | Slightly widens function signatures | **Chosen** |
| Recompute intermediates in diagnostics module | No changes to core functions | Duplicates pipeline logic, risk of divergence | Rejected |
| Return intermediates as extra tuple elements | Explicit | Changes return type, breaks all callers | Rejected |

### Architecture Changes

No new modules. Changes are localised to 3 files in
`code/src/analysis/receptive_field_mapping/surface/` plus the workflow
orchestrator and DAG config.

```
surface/
  slim_helpers.py       -- add diagnostics param to flatten_slim()
  forearm_slim_uv.py    -- collect intermediates, call diagnostic orchestrator
  slim_qc_figures.py    -- new save_slim_diagnostic_figures() + 6 step builders
```

### Knowledge Base Constraints

From `bug-slim-uv-non-manifold-flip.md`:
- Use `_find_boundary_loops()` to report boundary loop counts in step 2 annotations
- Use `_has_flipped_triangles()` to annotate flipped-face counts in step 4

From `note-3d-to-2d-surface-projection-algorithms.md`:
- Always project from neuron-wide contact centroid, not per-cluster

From `note-rf-camera-settings-connections.md`:
- Reuse `camera_settings_to_rotation()` for the camera-settings view; do not
  recompute a custom rotation

---

## Implementation Plan

### Phase 1: Expose intermediates from flatten_slim
**Goal:** Make init UV and method accessible to callers without changing the return type.
**Started:** 2026-05-21  **Completed:** 2026-05-21

- [x] Add `diagnostics: dict | None = None` parameter to `flatten_slim()`
- [x] Add `_trimmed = False` flag; set `True` inside the trim loop
- [x] Record `init_method` as `"harmonic"` or `"tutte"` at the appropriate branch
- [x] Record `init_uv` (copy of `uv_init`) and `trimmed` flag before the SLIM solve loop

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_helpers.py` — add parameter + 3 insertion points

**Dependencies:** None

### Phase 2: Diagnostic figure generation
**Goal:** Build the 6 step-numbered PNG generators and the skin-normal view computation.
**Started:** 2026-05-21  **Completed:** 2026-05-21

- [x] Add `_compute_skin_normal_view(V, F) -> (elev, azim)` private helper
- [x] Add `_plot_mesh_3d(ax, V, F, elev, azim, title, overlay_fn)` shared 3D rendering helper
- [x] Add step builder functions (each returns a `Figure`):
  - `_plot_step1_raw_mesh` — 1x2: skin-normal + camera view of raw BPA mesh
  - `_plot_step2_cleaned_mesh` — 1x2: same dual view, with V/F count annotations
  - `_plot_step3_centroid_boundary` — 1x2: centroid (red) + boundary loop (yellow)
  - `_plot_step4_uv_init` — 1x2: 3D mesh coloured by init-u + init UV 2D plot
  - `_plot_step5_slim_final` — 1x2: init UV vs final SLIM UV side-by-side
  - `_plot_step6_distortion` — delegates to existing `plot_slim_distortion_panel()`
- [x] Add `save_slim_diagnostic_figures()` orchestrator: mkdir, build+save each step, return paths

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_qc_figures.py` — ~300 new lines

**Dependencies:** Phase 1

### Phase 3: Pipeline integration
**Goal:** Wire diagnostics through `precompute_forearm_slim_uv` and the DAG config.
**Started:** 2026-05-21  **Completed:** 2026-05-21

- [x] Add `save_diagnostics` and `camera_settings_dir` params to `precompute_forearm_slim_uv()`
- [x] Capture raw mesh (V, F) before `clean_mesh()`
- [x] Capture cleaned mesh (V, F), center_vid, boundary before `flatten_slim()`
- [x] Pass `slim_diag = {}` to `flatten_slim()` when diagnostics enabled
- [x] Load camera settings (optional, try/except) from `camera_settings_dir`
- [x] Call `save_slim_diagnostic_figures()` with all intermediates
- [x] Add `save_diagnostics` param to `precompute_forearm_slim_uv_flow()`
- [x] Wire from DAG config options lambda
- [x] Add `save_diagnostics: true` to DAG YAML

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py` — capture intermediates, call diagnostics
- `code/scripts/analysis_workflow_processing.py` — flow param + DAG wiring
- `configs/analyse_workflow_processing_dag.yaml` — add option

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `flatten_slim` with `diagnostics={}` populates `init_uv`, `init_method`, `trimmed`
- [ ] `flatten_slim` with `diagnostics=None` behaves identically to before (no side effects)
- [ ] `_compute_skin_normal_view` returns valid `(elev, azim)` for a flat disk mesh

### Integration Tests
- [ ] `precompute_forearm_slim_uv` with `save_diagnostics=True` produces 6 PNGs in `diagnostics/`
- [ ] `precompute_forearm_slim_uv` with `save_diagnostics=False` produces no `diagnostics/` folder

### Manual Verification
- [ ] Run on all 12 sessions with `force_processing: true` and `save_diagnostics: true`
- [ ] Inspect skin-normal view: forearm visible from above in all sessions (no edge-on views)
- [ ] Compare 2 failing sessions with 10 working sessions to identify divergence step
- [ ] Confirm existing QC figures (`_qc_*.png`, `_distortion_*.png`) are unchanged

### Edge Cases
- [ ] Session where camera settings JSON is missing — camera view panel shows fallback annotation
- [ ] Session where `flatten_slim` takes the Tutte fallback — step 4 annotation correct
- [ ] Session where mesh trimming occurs — step 4/5 use trimmed mesh with init_uv correctly

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention diagnostic figures under SLIM UV section
- [ ] Docstrings on all new public functions

---

## Rollback Plan

All changes are additive (new parameter defaults, new functions). Rollback:

1. Revert the 3 source files to their pre-change state
2. Revert DAG config option
3. Delete any generated `diagnostics/` folders

No data migrations. No breaking changes to existing callers (all new
parameters have defaults).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| matplotlib 3D rendering slow for large meshes (>10k faces) | Low | Low | Use 200 DPI (not 300), `edgecolor="none"` |
| Skin-normal view orientation ambiguous (inside vs outside) | Med | Low | Use area-weighted mean normal; forearm meshes have consistent winding after `clean_mesh()` |
| Camera settings missing when diagnostics run | Med | Low | Graceful fallback: show default view with annotation |
| `init_uv` shape mismatch after mesh trimming | Low | High | Pair `init_uv` with `final_V`/`final_F` (the trimmed mesh), not `cleaned_V`/`cleaned_F` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: flatten_slim diagnostics | ~15 min | None |
| Phase 2: Figure generation | ~60 min | Phase 1 |
| Phase 3: Pipeline integration | ~20 min | Phase 2 |

---

## Output Structure

```
4_analysed/forearm_slim_uv/<session_id>/
  <session_id>_slim_uv.npz                        # existing cache
  <session_id>_slim_uv_qc_<timestamp>.png          # existing QC figure
  <session_id>_slim_uv_distortion_<timestamp>.png   # existing distortion figure
  diagnostics/                                      # NEW
    step1_raw_mesh.png
    step2_cleaned_mesh.png
    step3_centroid_boundary.png
    step4_uv_initialization.png
    step5_slim_uv_final.png
    step6_distortion.png
```
