# Plan: Population RF Map Visualisation Pipeline

**Date:** 2026-05-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/slim-uv-iff-weighted-centroid`
**Branch:** `feature/visualize-population-rf-maps`

---

## Overview

Add a batch pipeline task that generates per-session 2D projected RF population
heatmap PNGs.  For each session, the task produces one image combining all
single touches across all gesture types, plus one image per gesture type
(tap, stroke_proximal, stroke_distal).  Projection is always SLIM UV.
Vertices with fewer unique touches than a configurable min-overlap threshold
(25 % by default) are masked out.

## Problem Statement

The Touch Population Explorer GUI (`touch_population_explorer.py`) can display
per-vertex mean RF heatmaps filtered by gesture type, but only interactively.
There is no headless pipeline step to batch-render these maps as images for
inclusion in publications or automated comparison across sessions.

The SLIM UV projection (Phase 1–4 of the slim-forearm-projection plan) now
provides high-quality 2D parameterisations of the forearm surface.  Combining
SLIM projection with the explorer's RF heatmap computation yields publication-
ready 2D RF maps — but only if the rendering can run unattended.

## Goals

### In Scope

1. New shared computation module (`rf_population_heatmap.py`) extracting the
   RF heatmap aggregation, unique-touch count, and vertex-threshold logic from
   the Touch Population Explorer GUI into pure functions.
2. Refactor the explorer GUI to delegate to the shared module (no behaviour
   change).
3. New 2D renderer (`rf_population_map_renderer.py`) that renders per-vertex
   RF values onto a 2D SLIM-projected forearm surface as a two-panel PNG
   (scatter + interpolated heatmap).
4. New pipeline orchestrator (`rf_population_map_pipeline.py`) that, for each
   session, loads population data + RF data, computes heatmaps for all
   gestures and per gesture type, projects via SLIM, applies the min-overlap
   threshold, and saves PNGs.
5. New DAG task `visualize_population_rf_maps` wired into
   `analysis_workflow.py` and the processing DAG config.
6. Min-overlap vertex threshold at 25 % by default, configurable via DAG
   option `min_overlap_pct`.

### Out of Scope

- Configurable projection method — always SLIM.
- Interactive GUI or viewer for these maps (the explorer already exists).
- Per-cluster RF maps (handled by `visualize_receptive_fields_clustered`).
- PDF or SVG output (PNG only).
- Unit tests for the renderer (manual verification on real sessions is
  sufficient for visual correctness).

## Success Criteria

- [ ] `visualize_population_rf_maps` runs end-to-end on at least 2 real
      sessions and produces 4 PNGs each (all, tap, stroke_proximal,
      stroke_distal).
- [ ] All-gestures map visually matches the Touch Population Explorer's
      RF heatmap mode (same data, same colour scale, SLIM projection).
- [ ] Per-gesture-type maps show spatially distinct patterns.
- [ ] Colour scale is consistent across all 4 PNGs for a session
      (uses `session_max_value`).
- [ ] Vertices below the 25 % min-overlap threshold appear as grey
      forearm outline, not coloured.
- [ ] Re-running without changes to inputs is a no-op (idempotency).
- [ ] The Touch Population Explorer GUI still works identically after
      the extraction refactor.

---

## Technical Design

### Approach

Two-layer design: a shared computation module and a rendering pipeline.

The computation layer extracts four pure functions from the Touch Population
Explorer GUI into `rf_population_heatmap.py`.  Both the GUI and the new
pipeline import from this module, eliminating duplication.  The gesture-type
constant `GESTURE_TYPES` is imported from `clustering_pipeline` (canonical
source) — no new constant is defined.

**Extracted function signatures:**

```python
def compute_rf_heatmap(
    touch_indices: list[int],
    rf_vertex_indices: list[np.ndarray],
    rf_values: list[np.ndarray],
    n_verts: int,
) -> np.ndarray:
    """Mean RF value per vertex across selected touches.  NaN for uncontacted."""

def compute_unique_touch_count(
    cp_vertex_idx: np.ndarray,
    cp_touch_idx: np.ndarray,
    cp_mask: np.ndarray,
    n_verts: int,
) -> np.ndarray:
    """Per-vertex count of distinct touches among masked contact points."""

def compute_threshold_from_ratio(ratio_pct: float, n_filtered: int) -> int:
    """Convert a percentage threshold to an absolute integer count."""

def apply_vertex_threshold(
    heatmap_val: np.ndarray,
    unique_touch_count: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Set below-threshold contacted vertices to -1.0 (grey marker)."""
```

The GUI delegation wrappers are thin: each unpacks `self._data` /
`self._rf_data` attributes into the positional arguments above.

**`cp_mask` construction for the pipeline:**  The GUI builds `cp_mask` from
rect + checkbox state.  The pipeline builds it from touch indices:
`cp_mask = np.isin(pop_data.cp_touch_idx, gesture_touch_indices)`.  This
translation lives in the pipeline orchestrator, not in the shared module.

The rendering layer creates a new matplotlib-based renderer
(`rf_population_map_renderer.py`) that takes per-vertex RF values projected
to 2D and produces a two-panel figure (scatter + interpolated heatmap).  The
visual style mirrors `rf_2d_renderer.py` but uses linear normalisation and
the `jet` colormap to match the explorer.  If `griddata` interpolation fails
(too few non-NaN vertices), the renderer raises `ValueError` — no silent
fallback to scatter-only.

The pipeline orchestrator (`rf_population_map_pipeline.py`) ties it together:
load data, compute heatmaps for all subsets first (two-pass approach),
determine the session-wide `vmax`, then render all subsets with a consistent
colour scale.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend `render_2d_heatmap` with a new `display_metric` | Reuses existing renderer | Data shape differs (per-vertex vs. per-contact); would need significant branching inside the function | Rejected |
| New renderer function (chosen) | Clean interface tailored to per-vertex data; no risk of breaking existing cluster rendering | Small amount of visual-style duplication | **Chosen** |
| PyVista offscreen rendering (3D screenshots) | Matches explorer exactly | Headless PyVista on Linux CI can be fragile; matplotlib is more portable and produces vector-quality output | Rejected |
| Configurable projection method | Flexibility | User requirement: always SLIM. Unnecessary complexity. | Rejected |

### Architecture Changes

```
code/src/analysis/receptive_field_mapping/
├── rf_population_heatmap.py        — NEW: shared computation (extracted from GUI)
├── rf_population_map_renderer.py   — NEW: 2D matplotlib renderer
├── rf_population_map_pipeline.py   — NEW: per-session pipeline orchestrator
├── __init__.py                     — MODIFY: export run_population_rf_maps
├── gui/
│   └── touch_population_explorer.py — MODIFY: delegate to shared module

code/scripts/
└── analysis_workflow.py            — MODIFY: add @flow + wiring

configs/
└── analyse_workflow_processing_dag.yaml — MODIFY: add task entry
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Upstream (already exist)                                     │
│                                                              │
│  map_single_touch_rf  → single_touch_rf_maps.npz            │
│  set_rf_camera_settings → rf_camera_settings.json            │
│  precompute_forearm_slim_uv → {session}_slim_uv.npz          │
│  touch_series_transforms → {session}_series_augmented.csv    │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────┐
│ visualize_population_rf_maps (NEW)                            │
│                                                               │
│  Per session:                                                 │
│  1. load_population_data(series_csv, forearm_ply)             │
│  2. load_population_rf_data(npz, triple_keys, V)             │
│  3. load_rf_camera_rotation(camera_dir, session_id)           │
│  4. Resolve SLIM cache:                                       │
│     4_analysed/forearm_slim_uv/{sid}/{sid}_slim_uv.npz        │
│  5. forearm_uv = project_to_2d(                               │
│       forearm_verts, forearm_verts, contact_centroid,          │
│       method="slim", slim_cache_path=cache_path)              │
│     (forearm_vertices and contact_centroid are required        │
│      positional args but unused by SLIM;                      │
│      mirrors rf_cluster_visualizer.py:195)                    │
│                                                               │
│  Pass 1 — compute all heatmaps:                               │
│  6. For each subset (all, tap, stroke_prox, stroke_dist):     │
│     a. touch_indices = np.where(gesture_types == gtype)[0]    │
│        (all → range(n_touches))                               │
│     b. cp_mask = np.isin(cp_touch_idx, touch_indices)         │
│     c. compute_rf_heatmap(indices, rf_verts, rf_vals, V)      │
│     d. compute_unique_touch_count(cp_v, cp_t, cp_mask, V)     │
│     e. threshold = compute_threshold_from_ratio(pct, n)       │
│     f. apply_vertex_threshold(rf_vals, touch_count, thresh)   │
│  7. session_vmax = max(nanmax(h) for h in all heatmaps)       │
│                                                               │
│  Pass 2 — render with consistent colour scale:                │
│  8. For each subset:                                          │
│     render_population_rf_map(forearm_uv, rf_vals,             │
│                              vmax=session_vmax, ...)          │
│                                                               │
│  Output: 4_analysed/population_rf_maps/{session_id}/          │
│    {sid}_rf_population_all.png                                │
│    {sid}_rf_population_tap.png                                │
│    {sid}_rf_population_stroke_proximal.png                    │
│    {sid}_rf_population_stroke_distal.png                      │
└───────────────────────────────────────────────────────────────┘
```

### Cross-references to Knowledge Base

- `note-igl-slim-api-version-mismatch.md` — SLIM API handled by existing
  `_slim_helpers.py`; this plan does not call SLIM directly.
- `note-mesh-parameterization-interior-pin-foldovers.md` — interior-pin
  avoidance handled by the precompute task; this plan consumes the cached UV.
- `bug-slim-uv-non-manifold-flip.md` — Tutte+1-ring fallback for
  non-manifold forearm meshes; all 12 sessions now pass SLIM UV
  precompute.  This plan consumes the cached UV and is not affected
  by the fallback, but the KB note documents why some sessions
  previously failed.

### Reuse from Existing Code

| Component | Source | Reuse |
|-----------|--------|-------|
| RF heatmap aggregation | `touch_population_explorer._compute_rf_heatmap` (L654-684) | **Extract** to `rf_population_heatmap.py` |
| Unique touch count | `touch_population_explorer._compute_unique_touch_count` (L829-860) | **Extract** to `rf_population_heatmap.py` |
| Vertex threshold | `touch_population_explorer._apply_vertex_threshold` (L868-897) | **Extract** to `rf_population_heatmap.py` |
| Threshold ratio→absolute | `touch_population_explorer._effective_threshold` (L862-866) | **Extract** to `rf_population_heatmap.py` |
| Population data loading | `touch_population_data.load_population_data` | Reuse as-is |
| RF data loading | `touch_population_data.load_population_rf_data` | Reuse as-is |
| 2D projection | `rf_projection.project_to_2d` (L236) | Reuse as-is (`method="slim"`, `slim_cache_path=…`); `forearm_vertices` and `contact_centroid` positional args required but unused by SLIM — see `rf_cluster_visualizer.py:195` for canonical call pattern |
| Camera rotation | `rf_extraction_io.load_rf_camera_rotation` | Reuse as-is |
| Forearm PLY resolution | `rf_data_loader.resolve_forearm_ply` | Reuse as-is |
| Gesture type constant | `clustering_pipeline.GESTURE_TYPES` | **Import** (canonical source — no new constant) |
| Visual conventions | `rf_2d_renderer.render_2d_heatmap` | Replicate style in new renderer |
| Session ID extraction | `pipeline_shared.session_id_from_path` | Reuse as-is |

---

## Implementation Plan

### Phase 1: Extract shared computation from GUI
**Started:** 2026-05-18
**Completed:** 2026-05-18

**Goal:** Move RF heatmap aggregation, unique-touch count, and vertex-threshold
logic from the Touch Population Explorer into a shared module.  Refactor the
GUI to delegate.

- [x] Task 1.1 — Create `rf_population_heatmap.py` with the four extracted
      functions (`compute_rf_heatmap`, `compute_unique_touch_count`,
      `apply_vertex_threshold`, `compute_threshold_from_ratio`) using the
      explicit signatures defined in the Approach section.  Import
      `GESTURE_TYPES` from `clustering_pipeline` (not a new constant).
      Add a new utility `build_gesture_touch_indices(gesture_types, gtype)`
      that returns `np.where(gesture_types == gtype)[0]` — this is a new
      helper for the pipeline, not an extraction from the GUI.
- [x] Task 1.2 — Refactor `touch_population_explorer.py`:
      `_compute_rf_heatmap` delegates to `compute_rf_heatmap`;
      `_compute_unique_touch_count` delegates to `compute_unique_touch_count`;
      `_apply_vertex_threshold` delegates to `apply_vertex_threshold`.
- [x] Task 1.3 — Add unit tests for the extracted functions in
      `code/tests/test_rf_population_heatmap.py` (see Testing Plan for
      details).
- [ ] Task 1.4 — Verify the explorer GUI still works identically (manual
      launch, check RF heatmap mode, gesture filtering, threshold slider).

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_population_heatmap.py` — NEW
- `code/tests/test_rf_population_heatmap.py` — NEW
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — refactor 3 methods

**Dependencies:** None.

### Phase 2: Renderer and pipeline orchestrator
**Started:** 2026-05-18
**Completed:** 2026-05-18

**Goal:** Create the 2D renderer and per-session pipeline that produces the
PNG images.

- [x] Task 2.1 — Create `rf_population_map_renderer.py` with
      `render_population_rf_map()`: two-panel figure (scatter + interpolated
      heatmap), jet colormap, linear [0, vmax], black background, SLIM axis
      labels, distance-based NaN masking, title with session/gesture/touches/
      threshold info.  Raise `ValueError` if `griddata` interpolation fails
      (too few non-NaN vertices) — no silent fallback to scatter-only.
- [x] Task 2.2 — Create `rf_population_map_pipeline.py` with
      `run_population_rf_maps()`: per-session loop loading PopulationData +
      PopulationRFData, projecting via SLIM, then a two-pass approach:
      (1) compute heatmaps for all subsets, constructing `cp_mask` per
      gesture type via `np.isin(cp_touch_idx, gesture_touch_indices)`,
      applying threshold, and collecting results;
      (2) determine `session_vmax` from all computed heatmaps, then render
      all subsets with that consistent colour scale.
      Idempotency via sentinel JSON.  Skip gesture types with zero touches
      (logged warning, no PNG produced).
      Note: `project_slim()` loads the cache via `load_slim_uv_cache(path)`
      with path-only (no staleness check against PLY or rf_maps_npz).
      The DAG dependency `precompute_forearm_slim_uv` ensures cache
      freshness — the pipeline does not need to resolve rf_maps_npz
      or forearm_ply paths for cache validation.

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_population_map_renderer.py` — NEW
- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py` — NEW

**Dependencies:** Phase 1.

### Phase 3: Integration and verification
**Started:** 2026-05-18
**Completed:** 2026-05-18

**Goal:** Wire into the pipeline infrastructure and verify on real sessions.

- [x] Task 3.1 — Add `run_population_rf_maps` to `__init__.py` imports and
      `__all__`.
- [x] Task 3.2 — Add `visualize_population_rf_maps_flow` to
      `analysis_workflow.py`: @flow function, `available_tasks` registration,
      dispatcher kwargs (`neuron_mode`, `min_overlap_pct`,
      `disjoint_mask_distance_mm`).
- [x] Task 3.3 — Add `visualize_population_rf_maps` task entry to
      `analyse_workflow_processing_dag.yaml` with
      `depends_on: [map_single_touch_rf, set_rf_camera_settings, precompute_forearm_slim_uv]`,
      `min_overlap_pct: 25`, `neuron_mode: iff`.
- [ ] Task 3.4 — Manual verification: run on at least 2 sessions, confirm
      4 PNGs each, compare against explorer GUI.

**Files:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — add export
- `code/scripts/analysis_workflow.py` — add flow + wiring
- `configs/analyse_workflow_processing_dag.yaml` — add task entry

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests

- [x] `test_compute_rf_heatmap` — synthetic RF data (3 touches, 10 vertices):
      verify mean aggregation, NaN for uncontacted vertices, empty-touch-list
      returns all-NaN.
- [x] `test_compute_unique_touch_count` — synthetic contact points with
      overlapping vertex/touch pairs: verify deduplication and correct counts.
- [x] `test_apply_vertex_threshold` — verify below-threshold contacted
      vertices are set to -1.0, uncontacted (NaN) vertices are unchanged.
- [x] `test_compute_threshold_from_ratio` — verify ratio→absolute conversion,
      minimum of 1.

The renderer is a matplotlib figure — visual correctness is verified manually.

### Integration Tests

- [ ] DAG dry-run: `visualize_population_rf_maps` appears after
      `map_single_touch_rf`, `set_rf_camera_settings`, and
      `precompute_forearm_slim_uv` in the resolved task order.

### Manual Verification

- [ ] Run pipeline on at least 2 sessions with `visualize_population_rf_maps`
      enabled.
- [ ] Confirm 4 PNGs per session in `4_analysed/population_rf_maps/{sid}/`.
- [ ] Compare all-gestures map against the Touch Population Explorer's RF
      heatmap mode (should match visually).
- [ ] Confirm per-gesture-type maps show spatially distinct patterns.
- [ ] Confirm colour scale consistency across all 4 PNGs.
- [ ] Confirm below-threshold vertices appear grey (forearm outline), not
      coloured.
- [ ] Re-run without changes — should be a no-op (idempotency).

### Edge Cases

- [ ] Session with zero touches of one gesture type — should skip that PNG
      with a logged warning, not raise.
- [ ] Session where SLIM cache is missing — should raise `FileNotFoundError`
      with a clear message naming the missing file and suggesting to enable
      `precompute_forearm_slim_uv`.
- [ ] Gesture type with too few non-NaN vertices for `griddata` — should
      raise `ValueError` with a message naming the session, gesture type, and
      vertex count (fail-fast, no scatter-only fallback).
- [ ] Session with old-format SLIM cache (pre-IFF-weighted centroid) —
      `load_slim_uv_cache` raises `RuntimeError` with a message to
      delete the cache and re-run `precompute_forearm_slim_uv`.  The
      DAG dependency chain prevents this in normal operation, but a
      manually-triggered run could hit it.

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` — mention `rf_population_heatmap.py`,
      `rf_population_map_renderer.py`, `rf_population_map_pipeline.py`, and the
      `visualize_population_rf_maps` task.

---

## Rollback Plan

1. **Before merging:** Work lives on `feature/visualize-population-rf-maps`.
   Revert is a branch deletion or selective file revert.
2. **After merging:**
   - Disable `visualize_population_rf_maps` in the DAG config (`enabled: false`).
   - The shared module (`rf_population_heatmap.py`) is harmless — the GUI
     imports from it but behaviour is unchanged.
   - Output PNGs in `4_analysed/population_rf_maps/` are safe to delete.
3. **No database, no migrations.** Only new PNG files are created.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SLIM cache missing for a session (precompute not run) | Med | Med | Fail-fast with clear message naming the file; DAG dependency enforces ordering |
| griddata interpolation fails on sparse RF data (few non-NaN vertices) | Low | Med | Fail-fast: raise `ValueError` naming the session, gesture type, and vertex count so the researcher can investigate upstream data quality or lower the threshold |
| Large vertex arrays (30k vertices) slow down rendering | Low | Low | Use `rasterized=True` on scatter; stride subsampling on background dots |
| GUI refactor breaks explorer behaviour | Low | High | Thin delegation wrappers; unit tests (Task 1.3) + manual verification (Task 1.4) |
| Gesture type not present in session data | Med | Low | Skip with logged warning; don't raise |
| Old-format SLIM cache (pre-IFF-weighted centroid) | Low | Low | `load_slim_uv_cache` raises `RuntimeError` with actionable message; DAG dependency chain prevents this in normal operation |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Extract shared computation | 1 hour | None |
| Phase 2: Renderer + pipeline | 3 hours | Phase 1 |
| Phase 3: Integration + verification | 2 hours | Phase 2 |
| **Total** | **~6 hours** | |

---

## References

- Prerequisite plan: `docs/development/plans/active/slim-forearm-projection.md`
  (Phases 1–4 provide the SLIM UV precompute this plan depends on)
- Knowledge base:
  - `docs/development/knowledge-base/note-igl-slim-api-version-mismatch.md`
  - `docs/development/knowledge-base/note-mesh-parameterization-interior-pin-foldovers.md`
