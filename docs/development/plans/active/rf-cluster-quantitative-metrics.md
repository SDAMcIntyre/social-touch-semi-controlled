# Plan: Quantitative RF Metrics for Cluster Pipeline

**Date:** 2026-04-17
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-quantitative-metrics`

---

## Overview

**What:** Add a quantitative receptive field metrics module to the cluster-based RF mapping pipeline, computing standard neuroscience RF characterization metrics (size, shape, center, response strength, spatial concentration, boundary, Gaussian fit) per cluster.

**Why:** The RF cluster pipeline currently produces only visual outputs (heatmap PNGs) and minimal metadata counts. Standard neuroscience publications require formal quantitative RF characterization for reporting and cross-condition comparison.

**How:** A new `rf_metrics.py` module computes all metrics on 2D-projected spike count data (via existing `project_to_2d()`), integrated into the per-cluster loop of `rf_cluster_pipeline.py` after spike aggregation. Outputs structured JSON per cluster and a pooled summary CSV.

## Problem Statement

- The RF cluster pipeline (`rf_cluster_pipeline.py`) outputs only heatmap PNGs and basic counts (`n_sessions`, `n_touches`, `total_spike_points`, `total_spikes`) in `rf_cluster_summary.json`
- No formal RF size, shape, response profile, or spatial structure metrics are computed
- Without quantitative metrics, RF maps cannot be compared across clusters, conditions, or sessions in a statistically rigorous way
- Scientific publications on somatosensory RFs require standardized metrics (Fitzgerald et al. 2006, DiCarlo et al. 1998)

## Goals

### In Scope
1. Compute all 7 metric categories (size, shape, center, response strength, concentration, boundary, Gaussian fit) per cluster label
2. Output per-cluster `rf_metrics.json` with structured metric groups
3. Output pooled `rf_metrics_summary.csv` (one row per cluster) at the combo/clusterer level
4. Graceful degradation: degenerate cases (few points, no PLY) produce flagged partial metrics, not failures

### Out of Scope
- Metrics for the simple RF pipeline (per-session, not per-cluster)
- Selectivity-based metrics (would require extending extraction to track `total_counts` — separate future work)
- Visualization of metric overlays on heatmaps (e.g., ellipse outlines, boundary contours)
- Statistical comparison tests between clusters or conditions
- Changes to the DAG config or Prefect flow signatures

## Success Criteria

- [ ] `rf_metrics.json` is produced in every `cluster_{label}/` directory alongside `spike_counts.csv`
- [ ] `rf_metrics_summary.csv` is produced at the `{combo}/{clusterer}/` level with one row per cluster
- [ ] All 7 metric categories are populated for clusters with >= 3 non-collinear points
- [ ] Gaussian fit reports `converged: false` (not crash) when fitting fails
- [ ] Clusters with < 3 points or no PLY produce valid JSON with zeroed/NaN metrics and appropriate flags
- [ ] Existing outputs (`spike_counts.csv`, heatmap PNGs, `rf_cluster_summary.json`) are unchanged
- [ ] Metric values are physically plausible on real data (RF area 10-500 mm² for forearm)

---

## Technical Design

### Approach

Create a dedicated `rf_metrics.py` module following the existing single-responsibility pattern in the RF mapping package (cf. `rf_projection.py`, `rf_mapping_engine.py`, `rf_2d_renderer.py`). The module contains:
- An `RFMetrics` dataclass holding all computed values
- Pure computation functions for each metric category
- A `compute_rf_metrics()` orchestrator that projects points to 2D via existing `project_to_2d()`, runs all computations, and returns a populated `RFMetrics`

Response strength metrics use `spike_count` as the response signal (the direct observable), not selectivity (which requires `total_counts` not currently tracked in the cluster pipeline).

3D-to-2D projection uses tangent-plane as default. Per knowledge base note `note-3d-to-2d-surface-projection-algorithms.md`, this has zero distortion at RF center and ~5-10% compression at 60mm patch edges — acceptable for typical 20-60mm RF patches. Per `note-somatosensory-units-and-calculations.md`, all coordinates are in mm (Azure Kinect SDK native), so computed areas are natively in mm².

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New `rf_metrics.py` module | Single responsibility, clean separation, testable in isolation | One more file in package | **Chosen** |
| Extend `rf_mapping_engine.py` | No new file | Would bloat a focused 142-line module with 250+ lines of unrelated computation | Rejected |
| Compute selectivity-based metrics (requires `total_counts`) | Richer metrics | Requires modifying `_extract_spike_contact_points` data flow, larger scope | Deferred (future work) |
| Cylindrical unwrap as default projection | Better for full-forearm patches | Seam artifacts, worse for local RF patches | Rejected as default (available as option) |

### Architecture Changes

**New module:**
```
code/src/analysis/receptive_field_mapping/
├── rf_metrics.py              # NEW — RFMetrics dataclass + compute_rf_metrics()
├── rf_cluster_pipeline.py     # MODIFIED — ~30-40 lines added in per-cluster loop
├── __init__.py                # MODIFIED — new exports
├── rf_mapping_engine.py       # unchanged
├── rf_mapping_config.py       # unchanged
├── rf_projection.py           # unchanged (reused)
├── rf_data_loader.py          # unchanged (reused)
└── ...
```

**Integration point:** `rf_cluster_pipeline.py` line 339 (after `pooled_df.to_csv(spike_counts_csv)`), before heatmap rendering (line 356).

**Output structure:**
```
receptive_field_maps_clustered/
  {feature_combination}/
    {clusterer}/
      cluster_{label}/
        spike_counts.csv           # existing, unchanged
        rf_metrics.json            # NEW
        {session_id}_rf_heatmap.png # existing, unchanged
      rf_cluster_summary.json      # existing, gains "rf_metrics_computed" flag
      rf_metrics_summary.csv       # NEW — one row per cluster
```

### Metrics Specification

| Category | Metric | Unit / Range | Method |
|----------|--------|-------------|--------|
| **Size** | `convex_hull_area_mm2` | mm² | `scipy.spatial.ConvexHull` on 2D-projected points |
| | `threshold_area_mm2` | mm² | Convex hull of points above 50% of peak spike count |
| | `gaussian_sigma_major_mm` | mm | Larger sigma from 2D Gaussian fit |
| | `gaussian_sigma_minor_mm` | mm | Smaller sigma from 2D Gaussian fit |
| | `equivalent_diameter_mm` | mm | `sqrt(4 * convex_hull_area / pi)` |
| **Shape** | `aspect_ratio` | >= 1.0 | major / minor axis from weighted PCA ellipse |
| | `ellipse_major_mm` | mm | 2-sigma major axis length |
| | `ellipse_minor_mm` | mm | 2-sigma minor axis length |
| | `ellipse_orientation_deg` | degrees | Angle of major axis in projected plane |
| **Center** | `weighted_centroid_3d` | (mm, mm, mm) | Spike-count-weighted mean of (x, y, z) |
| | `weighted_centroid_2d` | (mm, mm) | Same, after 2D projection |
| **Response** | `peak_spike_count` | int | Max spike_count across all points |
| | `mean_spike_count` | float | Mean spike_count across all points |
| | `total_spikes` | int | Sum of all spike counts |
| **Concentration** | `hotspot_area_mm2` | mm² | Convex hull of top-20% points by spike count |
| | `hotspot_fraction` | 0-1 | hotspot_area / convex_hull_area |
| | `sparsity_index` | 0-1 | `1 - (mean_spike_count / peak_spike_count)` |
| **Boundary** | `half_peak_n_points` | int | Points above 50% of peak |
| | `convex_hull_n_vertices` | int | Hull vertex count |
| **Gaussian fit** | `converged` | bool | Whether `curve_fit` succeeded |
| | `r_squared` | 0-1 | Coefficient of determination |
| | `explained_variance` | 0-1 | Variance explained by Gaussian model |
| | `amplitude` | float | Peak amplitude of fitted Gaussian |
| | `x0_mm`, `y0_mm` | mm | Fitted center |
| | `theta_deg` | degrees | Fitted orientation |

### Key Functions in `rf_metrics.py`

- `compute_rf_metrics(spike_counts_df, forearm_vertices, projection_method) -> RFMetrics` — main entry point
- `_compute_weighted_centroid(points_3d, weights) -> np.ndarray` — spike-count-weighted mean
- `_compute_convex_hull_area(uv_2d) -> float` — `scipy.spatial.ConvexHull`, catches `QhullError` for degenerate cases
- `_compute_threshold_boundary(uv_2d, weights, threshold_frac=0.5) -> (area, n_points)` — half-peak boundary
- `_compute_hotspot(uv_2d, weights, top_frac=0.2) -> (area, fraction)` — top-response concentration
- `_fit_ellipse(uv_2d, weights) -> EllipseFitResult` — weighted PCA covariance eigendecomposition
- `_fit_2d_gaussian(uv_2d, weights) -> GaussianFitResult` — `scipy.optimize.curve_fit` with try/except
- `metrics_to_dict(m) -> dict` — JSON serialization (tuples to lists, NaN to null)
- `metrics_to_row(m, cluster_label, combo, clusterer) -> dict` — flat dict for CSV row

### Dependencies

All already in the project — no new packages:
- `numpy` — weighted centroid, PCA covariance
- `scipy.optimize.curve_fit` — 2D Gaussian fit
- `scipy.spatial.ConvexHull` — convex hull area
- `rf_projection.project_to_2d` — 3D-to-2D transformation
- `rf_data_loader.resolve_forearm_ply` — PLY file resolution

---

## Implementation Plan

### Phase 1: Core Metric Functions
**Goal:** Create `rf_metrics.py` with all computation functions and the `RFMetrics` dataclass.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 1.1 — Create `RFMetrics` dataclass with all fields from the metrics table, plus `n_points`, `projection_method`, `projection_fallback` metadata
- [x] 1.2 — Implement `_compute_weighted_centroid(points_3d, weights)` returning (3,) array
- [x] 1.3 — Implement `_compute_convex_hull_area(uv_2d)` using `scipy.spatial.ConvexHull`; return 0.0 for < 3 points or collinear points (catch `QhullError`)
- [x] 1.4 — Implement `_compute_threshold_boundary(uv_2d, weights, threshold_frac=0.5)` — filter points above `threshold_frac * max(weights)`, compute convex hull area, return `(area_mm2, n_points)`
- [x] 1.5 — Implement `_compute_hotspot(uv_2d, weights, top_frac=0.2)` — top 20% of points by weight, convex hull area, return `(area_mm2, fraction_of_total_area)`
- [x] 1.6 — Implement `_fit_ellipse(uv_2d, weights)` — weighted PCA: `cov = (w * (uv - centroid)).T @ (uv - centroid) / sum(w)`, eigendecompose, extract major/minor axes (2-sigma), orientation, aspect ratio
- [x] 1.7 — Implement `_fit_2d_gaussian(uv_2d, weights)` — model `f(xy, x0, y0, sigma_x, sigma_y, theta, amp, offset)`, fit with `curve_fit`, initial guess from data moments; on `RuntimeError`/`OptimizeWarning` set `converged=False`, return NaN for fit params
- [x] 1.8 — Implement `compute_rf_metrics(spike_counts_df, forearm_vertices, projection_method)` — orchestrate: project to 2D, call each sub-function, assemble `RFMetrics`
- [x] 1.9 — Implement `metrics_to_dict(m)` and `metrics_to_row(m, cluster_label, combo, clusterer)` for serialization

**Files created:**
- `code/src/analysis/receptive_field_mapping/rf_metrics.py` (~250-350 lines)

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Hook metrics into `run_cluster_rf_mapping()` and produce output files.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 2.1 — Import `compute_rf_metrics`, `metrics_to_dict`, `metrics_to_row` in `rf_cluster_pipeline.py`
- [x] 2.2 — After `pooled_df.to_csv(spike_counts_csv)` (line 339): resolve forearm PLY from first session with data, load vertices via Open3D, compute contact centroid from `pooled_df`, call `compute_rf_metrics`, save `rf_metrics.json` to cluster output directory
- [x] 2.3 — Accumulate metric rows in a list; after the per-cluster loop, write `rf_metrics_summary.csv` to `base_output`
- [x] 2.4 — Add `"rf_metrics_computed": true` to `summary_data[cluster_label]` in existing `rf_cluster_summary.json`
- [x] 2.5 — Pass `projection_method` through to `compute_rf_metrics` (default `"tangent_plane"` when `None`)

**Files modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` (~30-40 lines added)

**Dependencies:** Phase 1

### Phase 3: Exports
**Goal:** Wire up package exports.
**Started:** 2026-04-17
**Completed:** 2026-04-17

- [x] 3.1 — Add `compute_rf_metrics` and `RFMetrics` to `__init__.py` exports

**Files modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py`

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `_compute_convex_hull_area` on a unit square (4 corners) returns 1.0
- [ ] `_compute_convex_hull_area` with < 3 points returns 0.0
- [ ] `_compute_convex_hull_area` with collinear points returns 0.0
- [ ] `_fit_ellipse` on points from a known 2:1 ellipse recovers aspect_ratio ~2.0 (within 10%)
- [ ] `_fit_2d_gaussian` on a synthetic 2D Gaussian (known sigma_x=5, sigma_y=3, theta=30deg) recovers parameters within 10% and R² > 0.95
- [ ] `_compute_threshold_boundary` at 50% peak returns area smaller than full convex hull
- [ ] `_compute_hotspot` top-20% area is smaller than threshold-50% area
- [ ] Sparsity index: uniform weights returns near 0; peaked weights returns near 1
- [ ] `_compute_weighted_centroid` with uniform weights returns geometric mean

### Integration Tests
- [ ] Run `run_cluster_rf_mapping` on test data — `rf_metrics.json` appears in each cluster directory
- [ ] `rf_metrics_summary.csv` has one row per cluster with all expected columns
- [ ] Existing outputs (`spike_counts.csv`, heatmap PNGs, `rf_cluster_summary.json`) are unchanged

### Manual Verification
- [ ] Run full pipeline on one real dataset; inspect `rf_metrics.json` for physically plausible values (RF area 10-500 mm² for forearm)
- [ ] Compare Gaussian fit sigma to visual RF extent in heatmap PNG images

### Edge Cases
- [ ] Empty DataFrame (0 points) — produces valid JSON with zeroed/NaN metrics
- [ ] Single point — produces valid JSON with zero area metrics
- [ ] Collinear points (all on a line) — convex hull returns 0.0 area, ellipse fit degenerates gracefully
- [ ] No forearm PLY available — falls back to raw XY, flags `projection_fallback: true`
- [ ] Gaussian fit diverges — reports `converged: false`, NaN for fit parameters

---

## Documentation Plan

- [ ] Add literature references as docstrings in `rf_metrics.py` public functions
- [ ] Document `rf_metrics.json` schema in module docstring of `rf_metrics.py`
- [ ] Update module docstring in `rf_cluster_pipeline.py` to mention metrics output

---

## Rollback Plan

1. **Before deployment:** All changes are additive (new file + new lines in existing files). No existing behavior is modified.
2. **Rollback procedure:** Revert the commits that added `rf_metrics.py` and the integration lines in `rf_cluster_pipeline.py`. No data migration needed — `rf_metrics.json` and `rf_metrics_summary.csv` are new output files that can simply be deleted.
3. **Data considerations:** No breaking changes to existing output files. The `rf_cluster_summary.json` gains an additive `rf_metrics_computed` flag only.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gaussian fit fails on sparse/irregular clusters | High | Low | `converged: false`, skip Gaussian-derived metrics — no crash |
| ConvexHull fails with degenerate point sets | Medium | Low | Catch `QhullError`, return 0.0 area |
| No forearm PLY available for some clusters | Medium | Medium | Fall back to raw XY projection, flag `projection_fallback: true` |
| Performance regression for many clusters | Low | Low | Metric computation is O(N log N) per cluster; N typically 50-500 points |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core Metric Functions | ~250-350 lines, 1 new file | None |
| Phase 2: Pipeline Integration | ~30-40 lines added to 1 file | Phase 1 |
| Phase 3: Exports | ~3 lines | Phase 1 |

---

## References

- Fitzgerald, Lane, Thakur & Bhatt (2006) — *RF Size, Shape, and Somatotopic Organization* — J Neurosci 26(24):6485
- DiCarlo, Johnson & Hsiao (1998) — *Structure of Receptive Fields in Area 3b* — J Neurosci 18(7):2626
- Schellekens et al. (2021) — *A touch of hierarchy: pRF fingertip integration* — Brain Struct Funct
- Li et al. (2021) — *pRF Characteristics in between- and within-digit dimensions* — Cereb Cortex 31(10):4427
- Niell & Stryker (2008) — *Highly Selective Receptive Fields in Mouse Visual Cortex* — J Neurosci 28(30):7520
- Knowledge base: `note-somatosensory-units-and-calculations.md`, `note-3d-to-2d-surface-projection-algorithms.md`
- Related plans: `docs/development/plans/completed/tangent-plane-rf-alignment.md` (2D projection infrastructure)
