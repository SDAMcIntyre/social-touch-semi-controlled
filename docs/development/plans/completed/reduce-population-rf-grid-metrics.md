# Plan: Reduce Population RF Grid to Scalar Metrics

**Date:** 2026-05-06
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/reduce-population-rf-grid-metrics`
**Branch:** `feature/reduce-population-rf-grid-metrics`

---

## Overview

Add a new pipeline task (`reduce_population_rf_grid`) that reads the
per-gesture-type population RF grid NPZ files produced by
`map_population_rf_grid` and reduces each grid cell's vertex-level IFF
heatmap `(V,)` into a row of scalar summary metrics. The output is one CSV
per session per gesture type, enabling cross-session/cross-neuron comparison
of RF characteristics across feature space -- like comparing topographic
maps where elevation = IFF intensity.

## Problem Statement

`map_population_rf_grid` produces a `(G, V)` array of RF heatmaps -- one
per grid cell -- but each heatmap is a raw vertex-level array with thousands
of values. There is no way to systematically compare RF properties (size,
shape, intensity, concentration) across grid cells, sessions, or neurons
without first reducing each heatmap to a fixed set of scalar descriptors.

A reduction step would:
- Enable tabular cross-session comparison (one CSV row per grid cell)
- Quantify how RF shape/size/intensity varies as stimulus features change
- Produce data suitable for statistical analysis and plotting

## Goals

### In Scope
1. New metrics module computing IFF intensity, topographic, distributional,
   and shape metrics from a vertex-based RF heatmap
2. Reuse of `compute_rf_metrics()` for spatial metrics (convex hull, ellipse,
   Gaussian fit, hotspot, centroid) by bridging vertex data to its expected
   DataFrame format
3. New pipeline module that loads per-gesture-type grid NPZ files, computes
   per-cell metrics, and outputs one CSV per session per gesture type
4. DAG task registration with `depends_on: [map_population_rf_grid]`

### Out of Scope
- Visualization / plotting of the metrics table (future enhancement)
- Cross-session statistical tests (future task downstream of this one)
- GPU acceleration
- Mesh-based gradient/curvature metrics (require triangle connectivity;
  deferred to a later phase if needed)
- Pooled (all-gesture-type) mode -- data is always split per gesture type

## Success Criteria

- [ ] Task runs end-to-end for a session, producing one CSV per gesture type
      with one row per grid cell
- [ ] Empty cells (0 touches) produce a row with NaN for all metrics and
      `touch_count=0`
- [ ] Cells with fewer than 3 active vertices have NaN for spatial metrics
      but valid IFF intensity metrics
- [ ] Each gesture type's NPZ produces its own CSV (e.g.,
      `population_rf_grid_metrics_tap.csv`)
- [ ] All existing `RFMetrics` spatial fields are present in the output
      (via `compute_rf_metrics()` reuse)
- [ ] New IFF-specific metrics (max/mean/median/std, hypsometric integral,
      Gini, entropy, etc.) are correct for known synthetic inputs
- [ ] Task is skippable via `enabled: false` in DAG config
- [ ] Existing pipeline tasks are unaffected
- [ ] All units are mm / mm^2 consistent with upstream data

---

## Technical Design

### Approach

**Two-layer metric computation:**

1. **Reuse `compute_rf_metrics()`** for all spatial/shape/Gaussian metrics.
   Bridge vertex-based data to the expected format by constructing a DataFrame:
   ```python
   active_mask = ~np.isnan(rf_map)
   active_positions = forearm_vertices[active_mask]  # (N, 3) mm
   active_values = rf_map[active_mask]               # (N,) IFF
   df = pd.DataFrame({
       'x': active_positions[:, 0],
       'y': active_positions[:, 1],
       'z': active_positions[:, 2],
       'spike_count': active_values,  # IFF values used as weights
   })
   rf_metrics = compute_rf_metrics(df, forearm_vertices, projection_method)
   ```
   Then drop the response-group fields (`peak_spike_count`, `mean_spike_count`,
   `total_spikes`) and metadata fields from the output since they have
   misleading names for IFF data -- the dedicated IFF metrics below replace them.

2. **New metric functions** for IFF-specific, topographic, and distributional
   metrics that operate directly on the `(N,)` active IFF values array.

### Per-Gesture-Type Data Flow

The upstream `map_population_rf_grid` with `per_gesture_type: true` produces
separate NPZ files per gesture type:
```
population_rf_grid/<session_id>/
    population_rf_grid_tap.npz
    population_rf_grid_stroke_proximal.npz
    population_rf_grid_stroke_distal.npz
    population_rf_grid_tap_summary.json
    ...
```

This task discovers all `population_rf_grid_*.npz` files (excluding summary
JSONs) in each session's directory and processes each independently, producing
matching CSVs:
```
population_rf_grid_metrics/<session_id>/
    population_rf_grid_metrics_tap.csv
    population_rf_grid_metrics_stroke_proximal.csv
    population_rf_grid_metrics_stroke_distal.csv
    population_rf_grid_metrics_summary.json   (sentinel)
```

Each CSV contains a `gesture_type` column for downstream concatenation.

### Metrics Inventory

| Category | Metric | Source | Description |
|----------|--------|--------|-------------|
| **IFF Intensity** | `max_iff` | New | Peak IFF value (Hz) |
| | `mean_iff` | New | Mean IFF across active vertices |
| | `median_iff` | New | Median IFF |
| | `std_iff` | New | Std dev of IFF |
| | `iff_range` | New | max - min IFF |
| | `n_active_vertices` | New | Count of non-NaN vertices |
| **Topographic** | `hypsometric_integral` | New | (mean-min)/(max-min); 0-1 |
| | `coefficient_of_variation` | New | std/mean; relative variability |
| **Distribution** | `iff_skewness` | New | Third moment (asymmetry) |
| | `iff_kurtosis` | New | Fourth moment (peakedness) |
| | `gini_coefficient` | New | Inequality; 0=uniform, 1=concentrated |
| | `shannon_entropy` | New | Information spread; low=focal, high=diffuse |
| **Shape (new)** | `perimeter_mm` | New | Convex hull perimeter in mm |
| | `circularity` | New | 4*pi*area/perimeter^2; 1.0=circle |
| | `eccentricity` | New | sqrt(1-(minor/major)^2); 0=circle |
| **Size** | `convex_hull_area_mm2` | RFMetrics | Convex hull area |
| | `threshold_area_mm2` | RFMetrics | Half-peak convex hull area |
| | `equivalent_diameter_mm` | RFMetrics | Diameter of equal-area circle |
| | `gaussian_sigma_major/minor_mm` | RFMetrics | Gaussian fit sigmas |
| **Shape (existing)** | `aspect_ratio` | RFMetrics | Major/minor axis ratio |
| | `ellipse_major/minor_mm` | RFMetrics | PCA ellipse axes |
| | `ellipse_orientation_deg` | RFMetrics | Major axis angle |
| **Center** | `weighted_centroid_3d_x/y/z` | RFMetrics | IFF-weighted 3D centroid |
| | `weighted_centroid_2d_u/v` | RFMetrics | IFF-weighted 2D centroid |
| **Concentration** | `hotspot_area_mm2` | RFMetrics | Top 20% convex hull area |
| | `hotspot_fraction` | RFMetrics | Hotspot / total area ratio |
| | `sparsity_index` | RFMetrics | 1 - mean/peak |
| **Gaussian fit** | `gaussian_converged` | RFMetrics | Fit convergence flag |
| | `gaussian_r_squared` | RFMetrics | Fit quality |
| | `gaussian_explained_variance` | RFMetrics | Explained variance score |
| | `gaussian_amplitude` | RFMetrics | Fitted peak amplitude |
| | `gaussian_x0/y0_mm` | RFMetrics | Fitted center |
| | `gaussian_theta_deg` | RFMetrics | Fitted orientation |

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend `RFMetrics` with IFF fields | Single class, no duplication | Mixes spike-count and IFF semantics; changes existing API | Rejected |
| New metrics + reuse `compute_rf_metrics()` via bridge | Clean separation; full reuse of proven spatial code | Two-step computation per cell | **Chosen** |
| Reimplement spatial metrics from scratch | No bridge needed | Duplicates ~200 lines of tested code; divergence risk | Rejected |

### Architecture Changes

**New files:**
```
code/src/analysis/receptive_field_mapping/
    rf_grid_cell_metrics.py                  -- metric functions + GridCellRFMetrics dataclass
    rf_population_grid_metrics_pipeline.py   -- pipeline orchestration, NPZ loading, CSV output
```

**Modified files:**
- `code/src/analysis/receptive_field_mapping/__init__.py` -- re-export new public API
- `code/scripts/analysis_workflow.py` -- add flow function + register in available_tasks
- `configs/analyse_workflow_processing_dag.yaml` -- add task entry

### Key Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `compute_rf_metrics()` | `rf_metrics.py:389` | All spatial/shape/Gaussian metrics |
| `metrics_to_dict()` | `rf_metrics.py:546` | RFMetrics to JSON-safe dict |
| `load_forearm_vertices()` | `rf_data_loader.py` | Load forearm mesh vertices |
| `resolve_forearm_ply()` | `rf_data_loader.py` | Resolve forearm PLY path |
| `session_id_from_path()` | `pipeline_shared.py` | Extract session ID from CSV path |
| `should_process_task()` | `utils/should_process_task.py` | Idempotency check |
| `project_to_2d()` | `rf_projection.py` | 2D projection (used internally by compute_rf_metrics) |

---

## Implementation Plan

### Phase 1: Grid Cell Metrics Module
**Goal:** Implement all new metric functions and the orchestrator
**Started:** 2026-05-06
**Completed:** 2026-05-06

**Tasks:**
- [x] Task 1.1 -- Create `rf_grid_cell_metrics.py` with `GridCellRFMetrics`
      dataclass holding all new IFF-specific fields
- [x] Task 1.2 -- Implement `compute_iff_intensity_metrics(iff_values)` ->
      dict with max/mean/median/std/range/n_active
- [x] Task 1.3 -- Implement `compute_topographic_metrics(iff_values)` ->
      dict with hypsometric_integral, coefficient_of_variation
- [x] Task 1.4 -- Implement `compute_distribution_metrics(iff_values)` ->
      dict with skewness, kurtosis, gini_coefficient, shannon_entropy.
      Use `scipy.stats.skew`, `scipy.stats.kurtosis`, `scipy.stats.entropy`
- [x] Task 1.5 -- Implement `compute_boundary_shape_metrics(uv_2d, hull_area,
      ellipse_major, ellipse_minor)` -> dict with perimeter_mm, circularity,
      eccentricity. Note: scipy ConvexHull 2D convention: `hull.volume` = area,
      `hull.area` = perimeter
- [x] Task 1.6 -- Implement `compute_grid_cell_metrics(rf_map, forearm_vertices,
      projection_method)` -> flat dict. Orchestrator that:
      1. Extracts active vertices from `rf_map` (non-NaN mask)
      2. If no active vertices, return all-NaN dict
      3. Bridges to DataFrame format and calls `compute_rf_metrics()`
      4. Calls new metric functions on the active IFF values
      5. Drops misleading response/metadata fields from RFMetrics output
         (`peak_spike_count`, `mean_spike_count`, `total_spikes`,
          `n_points`, `projection_method`, `projection_fallback`)
      6. Merges all dicts into one flat row

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_grid_cell_metrics.py` -- new

**Dependencies:** None

### Phase 2: Pipeline Module
**Goal:** Wire NPZ loading, per-cell iteration, and CSV output
**Started:** 2026-05-06
**Completed:** 2026-05-06

**Tasks:**
- [x] Task 2.1 -- Create `rf_population_grid_metrics_pipeline.py` with
      `PopulationRFGridMetricsConfig` dataclass (projection_method field)
- [x] Task 2.2 -- Implement `_load_grid_npz(npz_path)` -> dict with rf_maps,
      grid_centers, feature_names, touch_counts, neuron_mode, gesture_type,
      vertex_threshold_ratio
- [x] Task 2.3 -- Implement `_build_metrics_dataframe(rf_maps, grid_centers,
      feature_names, touch_counts, forearm_vertices, ...)` -> DataFrame.
      Iterates grid cells; for each cell calls `compute_grid_cell_metrics()`;
      prepends grid metadata columns (grid_cell_index, feature center values,
      touch_count, session_id, gesture_type, neuron_mode).
      Progress logging every 100 cells
- [x] Task 2.4 -- Implement `run_population_rf_grid_metrics(input_items,
      output_dir, config, force)` -> list[Path]. Main entry:
      - Discovers all `population_rf_grid_*.npz` files per session (each is
        one gesture type)
      - For each NPZ: checks idempotency via `should_process_task()`, loads
        NPZ + forearm vertices, calls `_build_metrics_dataframe()`, saves CSV
      - Writes JSON sentinel after all gesture types processed

**Output structure:**
```
4_analysed/population_rf_grid_metrics/<session_id>/
    population_rf_grid_metrics_tap.csv
    population_rf_grid_metrics_stroke_proximal.csv
    population_rf_grid_metrics_stroke_distal.csv
    population_rf_grid_metrics_summary.json   (sentinel)
```

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` -- new

**Dependencies:** Phase 1

### Phase 3: DAG Integration
**Goal:** Register the task in the pipeline
**Started:** 2026-05-06
**Completed:** 2026-05-06

**Tasks:**
- [x] Task 3.1 -- Add imports to `__init__.py`:
      `run_population_rf_grid_metrics`, `PopulationRFGridMetricsConfig`
- [x] Task 3.2 -- Add `reduce_population_rf_grid_flow` to
      `analysis_workflow.py` following the `map_population_rf_grid_flow`
      pattern. The flow resolves input paths per session:
      - `forearm_ply_path` via `resolve_forearm_ply()`
      - `grid_dir` = `db_path / '4_analysed' / 'population_rf_grid' / session_id`
      - Validates `grid_dir` exists and contains at least one NPZ
- [x] Task 3.3 -- Register `("reduce_population_rf_grid",
      reduce_population_rf_grid_flow)` in `available_tasks` list (after
      `map_population_rf_grid`)
- [x] Task 3.4 -- Add kwargs forwarding block for
      `task_name == "reduce_population_rf_grid"` in `run_batch_analysis`:
      forward `projection_method` (covered by the existing generic
      `if options.get("projection_method"):` handler at line ~990 — no
      separate block needed)
- [x] Task 3.5 -- Add DAG config entry to
      `configs/analyse_workflow_processing_dag.yaml`:
      ```yaml
      reduce_population_rf_grid:
        category: processing
        enabled: true
        options:
          force_processing: false
          projection_method: tangent_plane
        depends_on: [map_population_rf_grid]
      ```

**Files:**
- `code/src/analysis/receptive_field_mapping/__init__.py` -- modify
- `code/scripts/analysis_workflow.py` -- modify
- `configs/analyse_workflow_processing_dag.yaml` -- modify

**Dependencies:** Phase 2

### Phase 4: Testing
**Goal:** Validate metric correctness and pipeline integration
**Started:** 2026-05-06
**Completed:** 2026-05-06

**Tasks:**
- [x] Task 4.1 -- Unit tests for `compute_iff_intensity_metrics()`: known
      values, single element, all-equal
- [x] Task 4.2 -- Unit tests for `compute_topographic_metrics()`: known HI
      (e.g. [0,1,2,3,4] -> HI=0.5), uniform input (max==min -> NaN)
- [x] Task 4.3 -- Unit tests for `compute_distribution_metrics()`: uniform
      Gini~=0, concentrated Gini~=1, uniform entropy=log2(n), symmetric
      skewness~=0
- [x] Task 4.4 -- Unit tests for `compute_boundary_shape_metrics()`:
      circle-like points -> circularity~=1, elongated -> eccentricity near 1
- [x] Task 4.5 -- Unit test for `compute_grid_cell_metrics()`: all-NaN RF map
      returns all-NaN; synthetic Gaussian blob returns plausible values;
      output contains both RFMetrics-derived and new metric fields
- [x] Task 4.6 -- Integration test: synthetic `(G, V)` rf_maps with G=4
      cells, verify output DataFrame has 4 rows, correct metadata columns,
      correct metric values for a known cell

**Files:**
- `code/tests/test_rf_grid_cell_metrics.py` -- new

**Dependencies:** Phase 1

---

## Output CSV Column Specification

**Metadata columns** (prepended per row):
`grid_cell_index`, `session_id`, `gesture_type`, `neuron_mode`, `touch_count`,
`<feature_1>_center`, `<feature_2>_center`, ...

**Metric columns** (per row): all metrics from the inventory table above, flat.
Centroid tuples expanded to `weighted_centroid_3d_x/y/z` and
`weighted_centroid_2d_u/v` (via `metrics_to_row()` pattern).

All spatial units are mm / mm^2 (inherited from Azure Kinect SDK native
coordinates through forearm mesh vertices).

---

## Testing Plan

### Unit Tests
- [ ] IFF intensity metrics with known input array
- [ ] Hypsometric integral: [0,1,2,3,4] -> 0.5; uniform -> NaN
- [ ] Gini: all-equal -> ~0; one-hot -> ~1
- [ ] Shannon entropy: all-equal -> log2(n); one-hot -> 0
- [ ] Circularity: square ~0.785; circle ~1.0
- [ ] Eccentricity: circle -> 0; 10:1 ellipse -> ~0.995
- [ ] All-NaN rf_map -> all-NaN metrics row

### Integration Tests
- [ ] Synthetic 4-cell grid -> correct DataFrame shape and values
- [ ] Empty cells (0 touches) produce NaN metric rows

### Manual Verification
- [ ] Run on a real session, load output CSV, spot-check a non-empty cell's
      metrics against manual computation
- [ ] Verify that each gesture type produces its own CSV

### Edge Cases
- [ ] Cell with 1 active vertex: intensity metrics valid, spatial metrics NaN
- [ ] Cell with 2 active vertices: convex hull degenerate
- [ ] All cells empty: CSV with all-NaN metric rows
- [ ] Session with only one gesture type: single CSV produced

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` receptive field mapping section
      with new task description

---

## Rollback Plan

1. Remove task from `available_tasks` in `analysis_workflow.py`
2. Remove DAG config entry from `analyse_workflow_processing_dag.yaml`
3. Remove imports from `__init__.py`
4. Delete `rf_grid_cell_metrics.py`, `rf_population_grid_metrics_pipeline.py`,
   and `test_rf_grid_cell_metrics.py`
5. No data migrations -- output is additive (new CSV files in new directory)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Performance: Gaussian fit per cell (5000 iter) x 4000 cells | Medium | Medium -- slow | Progress logging; cells with <7 vertices skip fit automatically |
| Memory: loading full `(G, V)` rf_maps array | Low | Low | Already loaded by upstream task; ~300MB worst case |
| `compute_rf_metrics()` field naming (spike_count vs IFF) | Certain | Low | Drop response-group fields, use dedicated IFF metrics |
| scipy ConvexHull 2D convention (volume=area, area=perimeter) | Medium | Medium | Document in code; unit test explicitly |
| Unit confusion in output columns | Low | Medium | All mm/mm^2; verify via knowledge base constraint |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Grid Cell Metrics | ~200 lines | None |
| Phase 2: Pipeline Module | ~150 lines | Phase 1 |
| Phase 3: DAG Integration | ~50 lines | Phase 2 |
| Phase 4: Testing | ~200 lines | Phase 1 |

---

## References

- Upstream task: `map_population_rf_grid` in `rf_population_grid_pipeline.py`
- Existing metrics: `RFMetrics` in `rf_metrics.py`
- Active plan: `docs/development/plans/active/map-population-rf-grid.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Literature: DiCarlo et al. 1998 (RF size); Fitzgerald et al. 2006 (RF shape);
  Dumoulin & Wandell 2008 (pRF sigma); Amatulli et al. 2018 (topographic vars)
