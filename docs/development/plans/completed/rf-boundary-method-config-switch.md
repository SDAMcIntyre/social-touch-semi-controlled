# Plan: Configurable RF Boundary Method Switch

**Date:** 2026-06-23
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-24 12:13
**Base Branch:** `feature/rf-proximal-distal-aggregate-enhancements`
**Branch:** `feature/rf-boundary-method-config-switch`

---

## Overview

**What:** Add a `boundary_method` config option to select which RF boundary definition (Laplacian inflection or gradient ridge) drives all downstream comparison, rendering, and profiling pipelines.
**Why:** The gradient ridge boundary (peak of first derivative) was implemented alongside the existing Laplacian inflection boundary, but downstream consumers still hardcode the inflection method. Switching the default requires a config-driven dispatch.
**How:** The NPZ `boundary_*` keys become method-agnostic (canonical), always containing the active method's data. Both methods are stored under explicit `inflection_*` and `gradient_*` prefixes for comparison. Downstream consumers read `boundary_*` and need zero changes.

## Problem Statement

The gradient ridge boundary module (`rf_gradient_boundary.py`) computes a boundary and stores it under `gradient_*` NPZ keys, but all downstream pipelines (session comparison, proximal-distal, tap-stroke, profile extraction, spatial tuning) read from `boundary_*` keys which only contain inflection data. There is no way to switch which method is canonical without editing code in 10+ files.

## Goals

### In Scope
1. Add `boundary_method` config option to DAG YAML (`"gradient"` or `"inflection"`, default `"gradient"`)
2. Wire the option through the orchestrator to all pipeline functions that compute or consume boundaries
3. Make NPZ `boundary_*` keys contain the active method's data (including 3D-projected metrics)
4. Store both methods under explicit prefixes (`inflection_*`, `gradient_*`) in every NPZ for comparison
5. Rename renderer parameters from `inflection_boundary` to method-agnostic names (`boundary`)
6. Dispatch spatial tuning pipeline to use the configured method

### Out of Scope
- Adding new boundary methods beyond inflection and gradient
- Changing downstream comparison/rendering pipelines (they read `boundary_*` — automatic)
- Modifying the gradient ridge algorithm itself
- Adding boundary method to the profile CSV output columns

## Success Criteria

- [ ] `boundary_method: gradient` in DAG config → NPZ `boundary_contour_uv_{gtype}` matches `gradient_contour_uv_{gtype}`
- [ ] `boundary_method: inflection` in DAG config → NPZ `boundary_contour_uv_{gtype}` matches `inflection_contour_uv_{gtype}`
- [ ] NPZ contains `boundary_method` metadata string
- [ ] NPZ contains both `inflection_*` and `gradient_*` keys regardless of active method
- [ ] Downstream comparison pipelines produce correct output without code changes
- [ ] Spatial tuning pipeline dispatches to the configured method
- [ ] All existing tests pass

---

## Technical Design

### Approach

**Canonical key approach:** The `boundary_*` NPZ keys are the single interface between the boundary producer (`spatial_extract_boundaries`) and all downstream consumers. By changing what data fills these keys based on a config flag, we switch the entire pipeline without touching consumer code.

Both methods are always computed and stored under their explicit prefixes, so switching config and re-running the producer is all that's needed. The profile pipeline uses the explicit keys directly (not `boundary_*`) to always show both methods side by side.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Canonical key switch (config selects what fills `boundary_*`) | Zero changes to 7 downstream files; single config flag | Requires explicit `inflection_*` keys (new) | **Chosen** |
| Config-aware consumers (each pipeline reads config and picks keys) | Explicit dispatch in every consumer | 10+ files changed; boundary_method must propagate everywhere | Rejected — high blast radius |
| Separate NPZ files per method | Clean separation | Breaks all downstream file paths; massive refactor | Rejected — disproportionate effort |
| Registry dispatcher pattern (like `rf_projection.py`) | Extensible for future methods | Over-engineered for 2 methods; adds indirection | Rejected — YAGNI |

### Architecture Changes

No new modules. Changes are scoped to parameter threading and NPZ key management.

**NPZ key layout (after change):**

| Prefix | Content | Status |
|--------|---------|--------|
| `boundary_*` | Active method (selected by config) | Repurposed |
| `inflection_*` | Always inflection boundary | New |
| `gradient_*` | Always gradient boundary | Unchanged |
| `boundary_method` | `"gradient"` or `"inflection"` string | New |
| `laplacian_*`, `smoothed_*` | Always present (needed by both methods) | Unchanged |

**Knowledge base constraints applied:**
- Fail-fast: validate `boundary_method in ("gradient", "inflection")` at pipeline entry
- NPZ metadata: store `boundary_method` string for diagnostic verification
- Cache invalidation: existing sentinel/force_processing mechanism handles recomputation when config changes

---

## Implementation Plan

### Phase 1: Config + orchestrator wiring
**Goal:** Thread `boundary_method` from DAG YAML through to pipeline functions

- [x] Add `boundary_method: gradient` to `spatial_extract_boundaries` options in DAG config
- [x] Add `boundary_method: gradient` to `spatial_tuning_rf_metrics` options in DAG config
- [x] Add `boundary_method` parameter to `spatial_extract_boundaries_flow()` in orchestrator
- [x] Wire `boundary_method` from DAG config to flow function via lambda params
- [x] Add `boundary_method` parameter to `spatial_tuning_rf_metrics_flow()` in orchestrator
- [x] Wire `boundary_method` from DAG config to spatial tuning flow function

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add `boundary_method: gradient` to two task option blocks
- `code/scripts/analysis_workflow_processing.py` — add parameter to two flow functions + wire from config

**Dependencies:** None

### Phase 2: NPZ producer — canonical key switch
**Goal:** Make `boundary_*` keys contain the active method's data; store both methods under explicit prefixes

- [x] Add `boundary_method: str = "gradient"` parameter to `run_population_response_field_extraction()`
- [x] Validate `boundary_method in ("gradient", "inflection")` at entry
- [x] Pass 1: after computing both boundaries, select active based on config → store in `gesture_boundaries`
- [x] Maintain separate `gesture_inflection_boundaries` dict for explicit NPZ keys
- [x] Pass 2 (composite): dispatch boundary computation based on `boundary_method`
- [x] Update `_save_response_fields_npz()` to accept both boundary dicts + `boundary_method`
- [x] Save `boundary_method` string in NPZ
- [x] Save inflection boundary under `inflection_*` keys (new explicit prefix)
- [x] Copy active method to `boundary_*` keys (canonical)
- [x] Add 3D projection (`uv_points_to_xyz`) for gradient boundary contour/centroid/peak
- [x] Save gradient 3D metrics under `gradient_contour_xyz_{gtype}`, `gradient_area_xyz_mm2_{gtype}`, etc.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — boundary selection logic, NPZ save restructuring

**Dependencies:** Phase 1

### Phase 3: Renderer parameter renames
**Goal:** Make population map renderer method-agnostic

- [x] Rename `_draw_inflection_boundary()` → `_draw_boundary()`
- [x] Rename `inflection_boundary` param → `boundary` in `render_population_rf_map()`
- [x] Rename `inflection_boundaries` param → `boundaries` in `render_population_rf_composite()`
- [x] Rename `inflection_contour_uv` → `boundary_contour_uv` in `render_population_rf_circular_crop()`
- [x] Rename `inflection_color` → `boundary_color` in `render_population_rf_circular_crop()`
- [x] Update all call sites in the population pipeline to use new parameter names

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — rename parameters and internal function
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — update call sites

**Dependencies:** Phase 2

### Phase 4: Spatial tuning dispatch
**Goal:** Spatial tuning pipeline uses the configured boundary method

- [x] Add `boundary_method: str = "gradient"` parameter to `run_spatial_tuning()`
- [x] At per-bin boundary computation (line ~406), dispatch based on method
- [x] At per-touch boundary computation (line ~440), dispatch based on method
- [x] For gradient: `compute_laplacian_arrays()` then `compute_gradient_ridge()`
- [x] For inflection: `compute_inflection_boundary()` (unchanged)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_spatial_tuning_pipeline.py` — add method dispatch at two call sites

**Dependencies:** Phase 1

### Phase 5: Profile pipeline — explicit method keys
**Goal:** Profile pipeline loads both methods' contours for side-by-side comparison

- [x] Load inflection contour from `inflection_contour_uv_{gtype}` (new key) instead of `boundary_contour_uv_{gtype}`
- [x] Load gradient contour from `gradient_contour_uv_{gtype}` (unchanged)
- [x] Always render both methods' markers on IFF/smoothed profiles (inflection = red, gradient = green)
- [x] Centers (centroid, peak) still come from `boundary_*` keys (active method)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_profile_extraction_pipeline.py` — switch contour loading from `boundary_*` to explicit method keys

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] Existing `test_rf_inflection_boundary.py` passes (35 tests) — renamed helpers still work
- [ ] Existing `test_rf_tap_stroke_comparison.py` passes — reads `boundary_*` keys unchanged

### Integration Tests
- [ ] Run `spatial_extract_boundaries` with `boundary_method: gradient` → verify NPZ key contents
- [ ] Run `spatial_extract_boundaries` with `boundary_method: inflection` → verify `boundary_*` matches `inflection_*`
- [ ] Run `spatial_compare_boundaries` → verify it reads the active method's metrics without code changes
- [ ] Run `spatial_extract_rf_profiles` → verify both marker sets visible on smoothed profiles

### Manual Verification
- [ ] Inspect NPZ: `boundary_method` string present, `inflection_*` and `gradient_*` keys coexist
- [ ] Compare `boundary_contour_uv_all` with `gradient_contour_uv_all` when gradient is active — should match
- [ ] Toggle to `inflection` and re-run — verify `boundary_contour_uv_all` now matches `inflection_contour_uv_all`
- [ ] Run downstream comparison pipelines — verify output reflects the configured method

### Edge Cases
- [ ] Invalid `boundary_method` value → `ValueError` raised at pipeline entry
- [ ] Gradient boundary returns `None` for a gesture → `boundary_*` keys for that gesture should be absent (same as inflection None)
- [ ] Missing `boundary_method` in old NPZ → profile pipeline handles gracefully (falls back to loading `boundary_*`)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note `boundary_method` config option and canonical key semantics
- [ ] Update DAG config comments — document the `boundary_method` option

---

## Rollback Plan

1. Set `boundary_method: inflection` in DAG config to restore original behaviour
2. Re-run `spatial_extract_boundaries` to regenerate NPZ files with inflection as canonical
3. All downstream pipelines automatically use inflection data — no code rollback needed
4. If code rollback needed: revert the commits on the feature branch; NPZ key structure returns to pre-change

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missing 3D metrics for gradient boundary breaks downstream | Med | High | Add `uv_points_to_xyz` projection for gradient contour in Phase 2 |
| Profile pipeline shows duplicate markers when active method = gradient | Low | Low | Phase 5 loads from explicit keys, not `boundary_*` |
| Old NPZ files lack `inflection_*` keys | Med | Low | Profile pipeline falls back to `boundary_*` if `inflection_*` absent |
| Renderer duck-typing fails for one dataclass | Low | Med | Both `InflectionBoundary` and `GradientBoundary` share identical field API |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Config + orchestrator | Small (~30 lines) | None |
| Phase 2: NPZ producer switch | Medium (~100 lines) | Phase 1 |
| Phase 3: Renderer renames | Small (~20 lines, mostly renames) | Phase 2 |
| Phase 4: Spatial tuning dispatch | Small (~20 lines) | Phase 1 |
| Phase 5: Profile pipeline keys | Small (~15 lines) | Phase 2 |

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_profile_extraction_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_spatial_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rf-boundary-method-config-switch.md

---

## References

- Related Plan: `docs/development/plans/active/` (gradient ridge boundary — parent feature)
- Knowledge Base: `note-3d-to-2d-surface-projection-algorithms.md` — registry pattern reference
- Knowledge Base: `investigation-rf-inflection-boundary-null.md` — DAG parameter flow pattern

---
