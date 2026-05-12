# Plan: Session Comparison Heatmap

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-simple-step-diagnostics`
**Branch:** `feature/session-comparison-heatmap`

---

## Overview

**What:** A new pipeline task that produces cross-session comparison heatmaps — one per gesture type (tap, stroke_proximal, stroke_distal) — where the X-axis shows bins of a single feature (velocity by default), the Y-axis lists sessions, and the color encodes a per-cell RF metric (mean IFF by default).

**Why:** The current population RF grid pipeline produces per-session 2D heatmaps (feature1 × feature2). There is no way to compare how RF metrics vary across sessions for the same feature range. This comparison is needed to identify between-session differences in neural response patterns.

**How:** Reuse the existing `run_population_rf_grid` and `run_population_rf_grid_metrics` functions with a 1-feature configuration (1D grid), writing to separate output directories. A new renderer reads those 1D metrics CSVs across all sessions and stacks them into a cross-session heatmap.

## Problem Statement

- The population RF grid pipeline produces heatmaps per session individually, with no cross-session view
- Researchers need to compare how neural response (mean IFF) distributes across velocity bins between sessions
- The existing `visualize_population_rf_grid_metrics` renderer enforces exactly 2 `_center` columns, so it cannot render 1D grids

## Goals

### In Scope
1. New pipeline task `visualize_session_comparison` that builds a 1D feature grid, computes metrics, and renders cross-session heatmaps
2. Reuse existing `run_population_rf_grid` and `run_population_rf_grid_metrics` for grid-building and metrics computation
3. Produce 3 PNG heatmaps (one per gesture type: tap, stroke_proximal, stroke_distal) with sessions on Y-axis, feature bins on X-axis, color = metric value
4. Configurable feature (default: `hand_velocity_amplitude_mean_during_iff`) and metric (default: `mean_iff`) via DAG config

### Out of Scope
- Interactive PyQt5 viewer for session comparison (future enhancement)
- Multi-feature (2D+) cross-session comparison
- Deviation metrics / baseline comparison across sessions
- Modifications to the existing 2D population RF grid pipeline

## Success Criteria

- [ ] Running the DAG produces 3 PNGs at `4_analysed/session_comparison_heatmaps/<metric>/{tap,stroke_proximal,stroke_distal}.png`
- [ ] X-axis shows velocity bins, Y-axis shows session IDs, color encodes mean IFF
- [ ] All 3 gesture-type heatmaps share the same global color scale
- [ ] Intermediate 1D grid and metrics artifacts are cached and idempotent (skip re-computation if up-to-date)
- [ ] Existing 2D population RF grid outputs are unaffected

---

## Technical Design

### Approach

Chain 3 processing steps in a single DAG task, reusing existing functions for steps 1-2:

```
Step 1: run_population_rf_grid()            [REUSE — 1-feature config]
  → 4_analysed/session_comparison_rf_grid/<session_id>/*.npz

Step 2: run_population_rf_grid_metrics()    [REUSE — reads 1D NPZ files]
  → 4_analysed/session_comparison_rf_grid_metrics/<session_id>/*.csv

Step 3: run_session_comparison_visualization()  [NEW renderer]
  → 4_analysed/session_comparison_heatmaps/<metric>/<gesture_type>.png
```

The existing grid and metrics functions handle 1D grids natively:
- `build_feature_grid()` produces `(G, 1)` grid centers with 1 feature
- `_sweep_grid()` filters on 1 dimension
- `_build_metrics_dataframe()` writes 1 `_center` column in the CSV
- Per-gesture-type splitting works unchanged

Separate output directories (`session_comparison_rf_grid/`, `session_comparison_rf_grid_metrics/`) avoid overwriting the existing 2D grid outputs.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse grid + metrics, new renderer | No code duplication; proven grid logic; intermediate caching | Single DAG task chains 3 conceptual steps | **Chosen** |
| Collapse existing 2D metrics CSVs to 1D | No re-computation; pure visualization | Aggregation (averaging across pressure axis) loses RF map fidelity — averaged metrics ≠ metrics of averaged RF | Rejected |
| Duplicate grid/metrics logic in new module | Full control over 1D-specific behavior | Significant code duplication; maintenance burden | Rejected |

### Architecture Changes

New module:
```
code/src/analysis/receptive_field_mapping/
├── rf_session_comparison_renderer.py     ← NEW: cross-session heatmap renderer
├── rf_population_grid_pipeline.py        ← unchanged (reused)
├── rf_population_grid_metrics_pipeline.py ← unchanged (reused)
└── __init__.py                           ← add export
```

Output directory structure:
```
4_analysed/
├── session_comparison_rf_grid/           ← NEW: 1D grid NPZ files per session
│   └── <session_id>/
│       ├── population_rf_grid_tap.npz
│       ├── population_rf_grid_stroke_proximal.npz
│       ├── population_rf_grid_stroke_distal.npz
│       └── population_rf_grid_all_gestures.npz
├── session_comparison_rf_grid_metrics/   ← NEW: 1D metrics CSVs per session
│   └── <session_id>/
│       ├── population_rf_grid_metrics_tap.csv
│       └── ...
└── session_comparison_heatmaps/          ← NEW: cross-session PNGs
    └── <metric>/
        ├── tap.png
        ├── stroke_proximal.png
        └── stroke_distal.png
```

---

## Implementation Plan

### Phase 1: Cross-session renderer module
**Goal:** Create the new renderer module that chains grid building, metrics computation, and cross-session heatmap rendering.
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 1.1 — Create `rf_session_comparison_renderer.py` with `_build_session_feature_matrix()`: takes `{session_id: df}` for one gesture type, pivots each session's 1-`_center`-column CSV into a row, returns a DataFrame (index=session_ids, columns=feature_bin_centers)
- [x] Task 1.2 — Add `render_session_comparison_heatmap()`: renders one heatmap via `ax.pcolormesh()` with `viridis` colormap, `set_bad(color="black")` for NaN, session IDs as Y-axis tick labels, dpi=150
- [x] Task 1.3 — Add `run_session_comparison_visualization()` entry point that chains the 3 steps: (1) call `run_population_rf_grid()` with 1-feature `PopulationRFGridConfig` (per_gesture_type=True, compute_baseline=False) writing to `session_comparison_rf_grid/`; (2) call `run_population_rf_grid_metrics()` writing to `session_comparison_rf_grid_metrics/`; (3) load metrics CSVs, validate 1 `_center` column, compute global (vmin, vmax), render per-gesture-type cross-session heatmaps to `session_comparison_heatmaps/<metric>/`; (4) write sentinel JSON

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_session_comparison_renderer.py` — **NEW**: ~150 lines, 3 functions

**Dependencies:** None

### Phase 2: Pipeline integration
**Goal:** Wire the new renderer into the analysis pipeline DAG.
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 2.1 — Add `run_session_comparison_visualization` import and `__all__` entry to `__init__.py`
- [x] Task 2.2 — Add `visualize_session_comparison_flow()` Prefect flow function to `analysis_workflow.py`: resolve input items (series CSVs, single-touch RF NPZs, forearm PLYs, touch_features_dir per session), pass DAG options (features, neuron_mode, projection_method, vertex_threshold_ratio, metric_name) to entry point
- [x] Task 2.3 — Register `("visualize_session_comparison", visualize_session_comparison_flow)` in `available_tasks` list after `visualize_population_rf_grid_metrics`
- [x] Task 2.4 — Add option forwarding in `run_batch_analysis` kwargs construction for `metric_name`, `projection_method`, `vertex_threshold_ratio`
- [x] Task 2.5 — Add `visualize_session_comparison` task block to `analyse_workflow_processing_dag.yaml` with default feature config and `depends_on: [map_single_touch_rf, touch_feature_extraction, set_rf_camera_settings]`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — Add 1 import + 1 `__all__` entry
- `code/scripts/analysis_workflow.py` — Add flow function (~40 lines), task registration, option forwarding
- `configs/analyse_workflow_processing_dag.yaml` — Add ~15 lines task block

**Dependencies:** Phase 1

### Phase 3: Heatmap quality improvements
**Goal:** Ensure x-axis shows only meaningful (positive) bins and sessions are arranged by similarity.

- [ ] Task 3.1 — Filter x-axis to strictly positive bins: in `_build_session_feature_matrix()`, after sorting columns, keep only `c > 0` and raise `ValueError` if none remain
- [ ] Task 3.2 — Add scipy imports at top of `rf_session_comparison_renderer.py`: `from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering` and `from scipy.spatial.distance import squareform`
- [ ] Task 3.3 — Add `_compute_session_order(matrix: pd.DataFrame) -> list[str]`: compute pairwise Pearson correlation across sessions, convert to distance (`1 - corr`, clipped to 0), run average-linkage hierarchical clustering + `optimal_leaf_ordering`, return session IDs in leaf order; fall back to `sorted()` if fewer than 2 sessions
- [ ] Task 3.4 — In `run_session_comparison_visualization()`, derive `session_order` once from the `all_gestures` matrix (if present and ≥ 2 sessions), fall back to alphabetical with a `logger.warning`; apply ordering to each gesture-type matrix via `reindex` before rendering

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_session_comparison_renderer.py` — ~30 new lines across 3 locations

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Enable `visualize_session_comparison` in DAG config (ensure upstream tasks already run or are enabled)
- [ ] Run `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_processing_dag.yaml`
- [ ] Verify intermediate 1D grid NPZ files in `4_analysed/session_comparison_rf_grid/<session>/`
- [ ] Verify intermediate metrics CSVs in `4_analysed/session_comparison_rf_grid_metrics/<session>/` — confirm 1 `_center` column
- [ ] Verify output PNGs at `4_analysed/session_comparison_heatmaps/mean_iff/{tap,stroke_proximal,stroke_distal}.png`
- [ ] Open PNGs: confirm X-axis = velocity bins, Y-axis = session IDs, color = mean IFF
- [ ] Confirm consistent color scale across the 3 gesture-type figures
- [ ] Re-run with `force_processing: false` — verify idempotency (skips re-computation)
- [ ] Verify existing 2D grid outputs (`population_rf_grid/`, `population_rf_grid_metrics/`) are untouched

### Edge Cases
- [ ] Single session — renders a valid single-row heatmap; seriation falls back to alphabetical order
- [ ] Session missing a gesture type — absent from that heatmap, other sessions render normally
- [ ] All sessions missing a gesture type — that gesture's heatmap is skipped with a log warning
- [ ] Feature grid configured with min ≤ 0 — non-positive bins silently filtered; `ValueError` if all bins are non-positive
- [ ] `all_gestures` absent from loaded CSVs — alphabetical session order used with a warning
- [ ] Session row is all-NaN (constant/zero variance) — `corr.fillna(0.0)` treats it as neutral distance, session still appears in heatmap

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add session comparison task to orchestration section
- [ ] Add DAG config comments describing the new task (inline in YAML)

---

## Rollback Plan

1. Revert the 4 file changes (1 new file + 3 modified files)
2. Remove the DAG task entry
3. No data migration: intermediate and output artifacts are write-only PNGs/CSVs/NPZs that can be safely deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1D grid produces different metrics CSV schema than expected by renderer | Low | Med | Validate `_center` column count at load time; existing metrics pipeline is schema-stable |
| Large number of sessions makes Y-axis unreadable | Low | Low | Auto-scale figure height based on session count (0.6 in per session) |
| Velocity feature name changes in future DAG configs | Low | Low | Feature name is configurable via `features` dict in DAG options, not hardcoded |
| Feature grid `min` set to 0 removes all zero-value bins leaving a usable subset | Low | Low | Filter is `> 0`, consistent with physiological interpretation; `ValueError` only fires if all bins are removed |
| `scipy.cluster.hierarchy.optimal_leaf_ordering` unavailable in installed scipy | Low | High | scipy >= 1.0.0 required; function present in all recent conda/pip scipy builds — no workaround needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer module | ~1 hour | None |
| Phase 2: Pipeline integration | ~30 min | Phase 1 |

---

## References

- Related Plans: N/A
- Existing pipeline entry points: `rf_population_grid_pipeline.py::run_population_rf_grid`, `rf_population_grid_metrics_pipeline.py::run_population_rf_grid_metrics`
- Existing renderer reference: `rf_population_grid_metrics_renderer.py::render_grid_metric_heatmap`

---
