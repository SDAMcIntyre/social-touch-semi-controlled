# Plan: Baseline RF + Deviation Framework

**Date:** 2026-05-07
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/reduce-population-rf-grid-metrics`
**Branch:** `feature/reduce-population-rf-grid-metrics`

---

## Overview

**What:** Compute a baseline RF (all touches pooled) and per-gesture-type aggregate RFs per session (neuron), then quantify deviation of each condition from its baseline using 8 normalised metrics.

**Why:** The pipeline currently produces only absolute RF metrics (e.g. `convex_hull_area_mm2 = 12`). These cannot be compared across neurons with different baseline RF sizes. Normalised deviation metrics (e.g. `area_ratio = 1.3`) enable cross-neuron statistical comparison of how gesture type and stimulus parameters modulate receptive field properties.

**How:** Extend the existing population RF grid pipeline and metrics pipeline (no new DAG tasks). Add baseline computation to the grid sweep stage, a pure-function deviation module, deviation columns to the metrics CSV, and heatmap rendering with diverging colormaps.

## Problem Statement

The scientific discussion identified three missing capabilities:

1. **No baseline RF per neuron** — there is no reference response field to compare conditions against.
2. **No deviation metrics** — no way to quantify how a condition (gesture type, velocity-pressure cell) changes the RF relative to baseline.
3. **No cross-neuron comparability** — absolute metrics like area or IFF are neuron-specific and cannot be compared across neurons without normalisation.

Without these, the pipeline can describe each neuron's RF under each condition, but cannot answer "did this neuron's RF expand under taps vs strokes?" or compare such modulations across neurons.

## Goals

### In Scope

1. Global baseline RF map per session (all touches pooled, no gesture or feature filtering)
2. Per-gesture-type aggregate RF maps (all touches of each type, no feature filtering)
3. Scalar metrics for baseline and aggregate RFs via existing `compute_grid_cell_metrics()`
4. 8 deviation metrics comparing condition to baseline (area ratio, hotspot area ratio, centroid shift, centroid shift normalised, mean IFF ratio, peak IFF ratio, vertex overlap Jaccard, vertex containment)
5. Two comparison levels: gesture-vs-global CSV + deviation columns in grid metrics CSVs
6. Heatmap rendering of deviation metrics on the velocity-pressure grid

### Out of Scope

- Cross-neuron statistical tests (downstream consumer of this output)
- Adaptive or rolling-window baselines
- Per-neuron velocity x pressure heatmaps (separate feature)
- IFF-based configurable percentile hotspots (separate feature)
- New DAG tasks (all work extends existing pipeline stages)

## Success Criteria

- [ ] Baseline NPZ files (`population_rf_grid_baseline_global.npz`, `population_rf_grid_baseline_{gtype}.npz`) saved per session alongside grid NPZs
- [ ] `rf_baseline_comparison.csv` per session with 1 baseline + N gesture-type rows, all scalar metrics + 8 deviation columns
- [ ] Grid metrics CSVs (`population_rf_grid_metrics_{gtype}.csv`) include 8 new `deviation_*` columns per non-empty cell
- [ ] Deviation heatmaps rendered with diverging colormaps (ratios centred at 1.0)
- [ ] Existing pipeline outputs unchanged when `compute_baseline: false`
- [ ] Backward compatible: missing baseline NPZs produce CSVs without deviation columns and a logged warning

---

## Technical Design

### Approach

Extend the three existing pipeline stages rather than adding new DAG tasks. The baseline RF uses the same `compute_cell_rf()` machinery already in the grid pipeline — only the touch mask differs (all-True for global, gesture-filtered for per-type). Deviation metrics are pure functions operating on pairs of metric dicts and RF map arrays.

### Knowledge Base Constraints

- All coordinates are in mm throughout the pipeline (note-somatosensory-units-and-calculations) — deviation distance metrics inherit this directly
- Data is already in Space 5 (RF-centered, note-spatial-alignment-pipeline) — no new spatial transforms needed
- Tangent-plane projection is the recommended default for 20-60mm patches (note-3d-to-2d-surface-projection-algorithms)
- Follow the existing orchestration pattern: load data, compute metrics, save CSV (note-rf-cluster-visualization-overview)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New DAG tasks for baseline + deviation | Clean separation of concerns | 3 more tasks, more config, more orchestration overhead | Rejected |
| Extend existing pipelines | Minimal config changes, natural data flow, baseline always available with grid data | Slightly larger functions | **Chosen** |
| Store baseline as row 0 in existing grid NPZs | Simple file structure | Breaks existing NPZ consumers; baseline is not a grid cell | Rejected |
| Separate deviation CSV files | Clean separation | Deviates from existing pattern (note-rf-cluster-visualization-overview recommends single-file source of truth) | Rejected |

### Architecture Changes

**New file:**
- `code/src/analysis/receptive_field_mapping/rf_baseline_deviation.py` — `BaselineDeviationMetrics` dataclass + `compute_baseline_deviation()` pure function

**Modified files:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` — add `compute_baseline` config field, baseline computation after data loading
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` — load baselines, compute deviation per cell, produce comparison CSV
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_renderer.py` — add deviation metrics to render list, diverging colormaps
- `code/src/analysis/receptive_field_mapping/__init__.py` — re-export new symbols
- `code/scripts/analysis_workflow.py` — forward `compute_baseline` option
- `configs/analyse_workflow_processing_dag.yaml` — add `compute_baseline: true`

### Deviation Metrics Specification (8 total)

| Metric | Formula | Colormap | Domain |
|--------|---------|----------|--------|
| `deviation_area_ratio` | condition_hull_area / baseline_hull_area | RdBu_r, centre=1.0 | (0, +inf), 1.0 = no change |
| `deviation_hotspot_area_ratio` | condition_hotspot_area / baseline_hotspot_area | RdBu_r, centre=1.0 | (0, +inf) |
| `deviation_centroid_shift_mm` | 3D Euclidean distance between weighted centroids | Reds, sequential | [0, +inf) mm |
| `deviation_centroid_shift_normalized` | centroid_shift_mm / baseline_equivalent_diameter | Reds, sequential | [0, +inf), dimensionless |
| `deviation_mean_iff_ratio` | condition_mean_iff / baseline_mean_iff | RdBu_r, centre=1.0 | (0, +inf) |
| `deviation_peak_iff_ratio` | condition_max_iff / baseline_max_iff | RdBu_r, centre=1.0 | (0, +inf) |
| `deviation_vertex_overlap_jaccard` | \|C intersection B\| / \|C union B\| on active vertex sets | RdYlGn, 0-1 | [0, 1] |
| `deviation_vertex_containment` | \|C intersection B\| / \|B\| on active vertex sets | RdYlGn, 0-1 | [0, 1] |

All ratios return NaN when denominator is zero or NaN.

### Two Comparison Levels

**Level 1 — Gesture vs global baseline** (`rf_baseline_comparison.csv`):
- Row 0: global baseline (all touches) — deviation ratios = 1.0, centroid shift = 0.0, overlap = 1.0
- Rows 1-N: per gesture type — absolute metrics + deviation from global baseline
- Purpose: "does tap expand/shrink/shift the RF vs the neuron's overall field?"

**Level 2 — Grid cell vs gesture-type baseline** (deviation columns in `population_rf_grid_metrics_{gtype}.csv`):
- Each tap grid cell compared to the all-taps baseline (not the global baseline)
- Purpose: "within taps, how does velocity x pressure modulate the RF?"
- Rationale: grid CSVs are already split by gesture type, so gesture-type baseline isolates the velocity-pressure effect from the gesture-type effect

**Level 3 — All-gestures grid vs global baseline** (deviation columns in `population_rf_grid_metrics_all_gestures.csv`):
- Grid sweep with ALL touches pooled (no gesture filter), deviation from global baseline
- Purpose: "across all stimulus types, how does velocity x pressure modulate the RF?"
- Produced alongside per-gesture grids when `per_gesture_type: true`

### Baseline NPZ Format

```
population_rf_grid/<session_id>/
    population_rf_grid_baseline_global.npz
    population_rf_grid_baseline_tap.npz
    population_rf_grid_baseline_stroke_proximal.npz
    population_rf_grid_baseline_stroke_distal.npz
```

Keys per NPZ: `rf_map` (V,) float64, `touch_count` (scalar) int64, `neuron_mode` str, `vertex_threshold_ratio` float, `baseline_type` str, `session_id` str

### Output File Layout

```
population_rf_grid_metrics/<session_id>/
    rf_baseline_comparison.csv                         # NEW
    population_rf_grid_metrics_tap.csv                 # EXTENDED: +8 deviation_* columns
    population_rf_grid_metrics_stroke_proximal.csv
    population_rf_grid_metrics_stroke_distal.csv
    population_rf_grid_metrics_all_gestures.csv        # NEW: all touches pooled, deviation vs global baseline

population_rf_grid_metrics_heatmaps/
    deviation_area_ratio/<session_id>_<gtype>.png       # NEW
    deviation_centroid_shift_mm/<session_id>_<gtype>.png
    ... (8 metric folders total)
```

### Config Extension

Add `compute_baseline: bool = True` to `PopulationRFGridConfig`. DAG YAML:

```yaml
map_population_rf_grid:
  options:
    compute_baseline: true    # NEW — all other options unchanged
```

---

## Implementation Plan

### Phase 1: Baseline RF Computation
**Goal:** Compute and save baseline RF maps alongside existing grid outputs
**Started:** 2026-05-07
**Completed:** 2026-05-07

- [x] Add `compute_baseline: bool = True` field to `PopulationRFGridConfig` dataclass
- [x] Add `_compute_and_save_baseline()` helper function that calls `compute_cell_rf()` with the given touch mask and saves to a single-map NPZ
- [x] In `run_population_rf_grid()`, after loading `pop_data`/`rf_data` and before the grid sweep, compute baseline RF maps when `config.compute_baseline`:
  - Global baseline: `touch_mask = np.ones(T, dtype=bool)` → `population_rf_grid_baseline_global.npz`
  - Per gesture type: `touch_mask = (gesture_types == gtype)` → `population_rf_grid_baseline_{safe_gtype}.npz`
- [x] Forward `compute_baseline` through the workflow: add parameter to `map_population_rf_grid_flow()`, add kwargs forwarding block, add to `PopulationRFGridConfig` construction
- [x] Add `compute_baseline: true` to DAG config

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` — config field + `_compute_and_save_baseline()` + baseline computation block (~40 new lines)
- `code/scripts/analysis_workflow.py` — `map_population_rf_grid_flow()` parameter + kwargs forwarding (~line 1018, ~10 new lines)
- `configs/analyse_workflow_processing_dag.yaml` — add `compute_baseline: true` (1 line)

**Dependencies:** None

### Phase 2: Deviation Metrics Module
**Goal:** Pure-function module for computing deviation between condition and baseline
**Started:** 2026-05-07
**Completed:** 2026-05-07

- [x] Create `rf_baseline_deviation.py` with:
  - `BaselineDeviationMetrics` dataclass (8 float fields, all default NaN)
  - `compute_baseline_deviation(condition_metrics: dict, baseline_metrics: dict, condition_rf_map: np.ndarray, baseline_rf_map: np.ndarray) -> dict` — returns flat dict with `deviation_` prefixed keys
  - `_safe_ratio(numerator: float, denominator: float) -> float` — returns NaN on zero/NaN denominator
  - `_centroid_3d_distance(metrics_a: dict, metrics_b: dict) -> float` — Euclidean distance between `weighted_centroid_3d_x/y/z` keys
  - `_vertex_set_overlap(rf_map_a: np.ndarray, rf_map_b: np.ndarray) -> tuple[float, float]` — returns (Jaccard, containment) from non-NaN vertex index sets
- [x] Export `compute_baseline_deviation` and `BaselineDeviationMetrics` from `receptive_field_mapping/__init__.py`

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_baseline_deviation.py` (~120 lines)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — re-export new symbols

**Dependencies:** None (pure functions, no pipeline imports)

### Phase 3: Metrics Pipeline Integration
**Goal:** Load baselines, compute deviation per grid cell, produce gesture-level comparison CSV
**Started:** 2026-05-07
**Completed:** 2026-05-07

- [x] Add `_load_baseline_npz(npz_path: Path) -> dict` — load and validate a baseline NPZ, return `{"rf_map": (V,), "touch_count": int, "baseline_type": str}`
- [x] Add `_compute_baseline_metrics(rf_map: np.ndarray, forearm_vertices: np.ndarray, projection_method: str) -> dict` — thin wrapper calling existing `compute_grid_cell_metrics()` on a single baseline RF map
- [x] Extend `_build_metrics_dataframe()` signature with optional `baseline_metrics: dict | None` and `baseline_rf_map: np.ndarray | None` — when provided, call `compute_baseline_deviation()` per non-empty cell and merge 8 `deviation_*` columns into each row
- [x] Add `_build_baseline_comparison_dataframe()` — produces DataFrame with 1 global baseline row + N gesture-type rows, each with absolute metrics + deviation from global baseline
- [x] Modify `run_population_rf_grid_metrics()` main loop:
  1. After loading `forearm_vertices`, glob for baseline NPZs in `grid_dir` (pattern: `population_rf_grid_baseline_*.npz`)
  2. If found: load global + per-gesture baselines, compute metrics for each, pass gesture-type baseline to `_build_metrics_dataframe()`, build and save `rf_baseline_comparison.csv`
  3. If missing: log warning, proceed without deviation columns (backward compatible)
- [x] Update sentinel JSON with `has_baseline: true/false`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` (~100 new lines)

**Dependencies:** Phase 1, Phase 2

### Phase 4: Deviation Heatmap Rendering
**Goal:** Render deviation metrics as heatmaps on the velocity-pressure grid
**Started:** 2026-05-07
**Completed:** 2026-05-07

- [x] Define `DEVIATION_METRICS` list in `rf_population_grid_metrics_renderer.py` (8 metric names)
- [x] Add `_DEVIATION_COLORMAP_CONFIG` dict mapping metric names to `(colormap_name, center_value | None)`:
  - Ratio metrics → `("RdBu_r", 1.0)`
  - Distance metrics → `("Reds", None)`
  - Overlap metrics → `("RdYlGn", None)` with vmin=0, vmax=1
- [x] Extend `render_grid_metric_heatmap()` with optional `center_value: float | None` parameter — when provided, use `matplotlib.colors.TwoSlopeNorm(vcenter=center_value, vmin=vmin, vmax=vmax)` for diverging colour scale
- [x] Extend `run_population_rf_grid_metrics_visualization()` to:
  1. Detect whether deviation columns exist in loaded CSVs
  2. If present, append `DEVIATION_METRICS` to the metrics-to-render list
  3. Apply deviation-specific colormaps and shared ranges

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_renderer.py` (~60 new lines)

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `_safe_ratio(10, 5)` -> 2.0; `_safe_ratio(10, 0)` -> NaN; `_safe_ratio(NaN, 5)` -> NaN; `_safe_ratio(10, NaN)` -> NaN
- [ ] `_centroid_3d_distance()` with known 3-4-5 triangle -> 5.0; NaN centroid -> NaN; identical centroids -> 0.0
- [ ] `_vertex_set_overlap()`: disjoint sets -> (0, 0); identical sets -> (1, 1); strict subset -> (containment=1, jaccard<1); empty baseline -> (NaN, NaN); empty condition -> (0, NaN)
- [ ] `compute_baseline_deviation()` end-to-end with synthetic metric dicts — verify all 8 output keys present and correct

### Integration Tests
- [ ] Synthetic session with 3 gesture types, 10 touches each → verify 4 baseline NPZ files created with correct structure
- [ ] Grid metrics CSV has 8 deviation columns when baselines present, 0 when absent
- [ ] `rf_baseline_comparison.csv` has exactly 4 rows (1 global + 3 gesture types) with correct deviation values
- [ ] Global baseline row: `deviation_area_ratio` = 1.0, `deviation_centroid_shift_mm` = 0.0, `deviation_vertex_overlap_jaccard` = 1.0

### Manual Verification
- [ ] Run full pipeline on a real session (e.g. ST13-03)
- [ ] Inspect `rf_baseline_comparison.csv` — global baseline row should have ratios = 1.0, shifts = 0.0
- [ ] Inspect grid metrics CSV — cells with high touch counts should have overlap/containment near 1.0
- [ ] Inspect deviation heatmaps — ratio maps should show white/neutral at 1.0

### Edge Cases
- [ ] Baseline with 0 active vertices (all NaN RF map) — all metrics NaN, all deviation NaN
- [ ] Session with only 1 gesture type — global = gesture baseline; deviation ratios all 1.0
- [ ] `compute_baseline: false` — no baseline NPZs written; metrics pipeline skips deviation
- [ ] Old grid data (pre-baseline, no baseline NPZs) — warning logged, CSVs produced without deviation columns

**Test file:** `code/tests/test_rf_baseline_deviation.py`

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add baseline deviation to RF mapping section
- [ ] Update `CLAUDE.md` — add `rf_baseline_deviation` to analysis stage description if needed
- [ ] Document deviation metric definitions in `rf_baseline_deviation.py` docstrings

---

## Rollback Plan

1. Set `compute_baseline: false` in DAG config — disables all new computation immediately
2. If code revert needed:
   - Remove baseline computation block from `rf_population_grid_pipeline.py`
   - Remove deviation logic from `rf_population_grid_metrics_pipeline.py`
   - Remove deviation rendering from `rf_population_grid_metrics_renderer.py`
   - Delete `rf_baseline_deviation.py` and `test_rf_baseline_deviation.py`
3. No data migration — baseline NPZs and comparison CSVs are additive outputs; existing grid CSVs regenerate without deviation columns on next forced run
4. No breaking changes to existing outputs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Baseline uninformative for neurons with few total touches | Medium | Low | `touch_count` recorded in baseline NPZ and comparison CSV; downstream analysis can filter sessions below a threshold |
| `vertex_threshold_ratio` masks different vertex sets in baseline vs condition | Medium | Medium | Expected behaviour — threshold reflects reliable response at each data level; Jaccard/containment metrics explicitly quantify the overlap |
| Diverging colormaps unfamiliar to collaborators | Low | Low | Only ratio metrics use diverging cmap; distance and overlap metrics use familiar sequential colormaps |
| Performance: computing metrics on baselines adds time | Low | Low | Only 1 global + 3 gesture baselines per session (4 extra metric computations per session, not per grid cell) |

---

## Timeline

| Phase | Estimated Effort | Lines | Dependencies |
|-------|-----------------|-------|--------------|
| Phase 1: Baseline Computation | ~50 new lines | 3 files | None |
| Phase 2: Deviation Module | ~120 new lines | 1 new file + 1 modified | None |
| Phase 3: Metrics Integration | ~100 new lines | 1 file | Phases 1, 2 |
| Phase 4: Deviation Heatmaps | ~60 new lines | 1 file | Phase 3 |
| Tests | ~200 new lines | 1 new file | Phases 1-3 |
| **Total** | **~530 lines** | **7 modified + 2 new** | |

---

## References

- Existing pipeline: `rf_population_grid_pipeline.py` -> `rf_population_grid_metrics_pipeline.py` -> `rf_population_grid_metrics_renderer.py`
- Metrics engine: `rf_grid_cell_metrics.py::compute_grid_cell_metrics()`, `rf_metrics.py::compute_rf_metrics()`
- Knowledge base: `note-somatosensory-units-and-calculations.md`, `note-3d-to-2d-surface-projection-algorithms.md`, `note-rf-cluster-visualization-overview.md`, `note-spatial-alignment-pipeline.md`
- Active plans: `reduce-population-rf-grid-metrics.md`, `visualize-population-rf-grid-metrics.md`
