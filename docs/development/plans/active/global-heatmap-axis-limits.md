# Plan: Global Heatmap Axis Limits and Bin Edges Across Sessions

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Per-session heatmaps currently compute axis ranges and bin edges independently from each session's data subset, making cross-session visual comparison unreliable. This plan introduces globally consistent bin edges, axis limits, and color-scale normalization so that every per-session heatmap is rendered on identical axes.

## Problem Statement

In `_generate_session_heatmaps()`, each session's DataFrame subset is passed to `generate_touch_density_heatmap()`, which calls `_get_global_edges()` on that subset alone. The result: a session with a narrow feature range gets stretched to fill the same plot area as one with a wide range. Two sessions that should look visually different (one dense, one sparse) can appear identical because the color scale is also session-local.

Additionally, `BinningClusterer` derives bin edges from the pooled data via `pd.cut()` but discards the edge arrays — they are not stored in metadata and cannot be reused downstream.

## Goals

### In Scope
1. Compute heatmap bin edges once from the full pooled DataFrame (all sessions) and pass them into every per-session heatmap call
2. Compute a single global maximum bin count for LogNorm color scaling, shared across all session heatmaps
3. Store explicit bin edge arrays in `BinningClusterer` metadata for reproducibility
4. Default the heatmap x-axis to the velocity feature column when available (consistent with `matrix_generation.py` which uses `max_velocity` as x-axis)

### Out of Scope
- Changing the per-session heatmap concept to a single pooled heatmap
- Adding configurable fixed axis limits from YAML (data-driven limits are sufficient)
- Modifying `generate_population_heatmaps()` (already handles global edges correctly)
- Unit conversions — all features remain in Kinect-native mm-based units (see `note-somatosensory-units-and-calculations.md`)

## Success Criteria

- [ ] All per-session heatmap PNGs within a clustering run share identical x-axis and y-axis tick labels and ranges
- [ ] LogNorm color scale is consistent across session heatmaps — sparse sessions appear lighter, dense sessions darker
- [ ] `cluster_metadata.json` for binning runs contains a `bin_edges` key with per-feature edge arrays
- [ ] Backward compatibility: heatmaps still generate correctly for kmeans and DBSCAN clustering (where no `bin_edges` metadata exists)
- [ ] Heatmap x-axis defaults to velocity column (e.g. `max_velocity`) when present among feature columns; falls back to highest-variance if no velocity column exists

---

## Technical Design

### Approach

Compute global bin edges and global max count in `_generate_session_heatmaps()` from the full `result_df` before iterating sessions. Pass these as optional parameters to `generate_touch_density_heatmap()`. When the parameters are `None`, the reporter falls back to current per-subset behavior — keeping full backward compatibility.

For `BinningClusterer`, use `pd.cut(..., retbins=True)` and `pd.qcut(..., retbins=True)` to capture the actual bin edge arrays and store them in metadata.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pre-compute edges in pipeline, pass to reporter | Minimal changes, backward compatible, follows existing `generate_population_heatmaps` pattern | Reporter gains two new optional params | **Chosen** |
| Reuse `BinningClusterer` bin edges for heatmaps | Exact match between cluster bins and visual bins | Only works for binning clusterer, not kmeans/DBSCAN; bin features may differ from heatmap axes | Rejected |
| Add a global-edges computation method to reporter | Encapsulates edge logic in one place | Requires passing full pooled df to reporter before session loop; more invasive | Rejected |

### Architecture Changes

No new modules. Three existing files modified:

- `BinningClusterer.fit_predict()` — additional `bin_edges` key in returned metadata
- `VisualReportingStrategy.generate_touch_density_heatmap()` — two new optional parameters
- `_generate_session_heatmaps()` — pre-compute edges and max before session loop

---

## Implementation Plan

### Phase 1: Store bin edges in BinningClusterer metadata
**Goal:** Make bin edge arrays available in `cluster_metadata.json` for reproducibility.

- [ ] Initialize `bin_edges: dict[str, list[float]] = {}` before the column loop
- [ ] Use `retbins=True` in `pd.cut()` and `pd.qcut()` calls to capture edge arrays
- [ ] Store `bin_edges[col] = edges.tolist()` for each binned feature
- [ ] Add `'bin_edges': bin_edges` to the metadata dict

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/binning_clusterer.py` — `fit_predict()`: use `retbins=True`, store edges (~8 lines)

**Dependencies:** None

### Phase 2: Accept pre-computed edges in the reporter
**Goal:** Allow `generate_touch_density_heatmap()` to use externally computed edges and color-scale max.

- [ ] Add `global_edges: Optional[Dict[str, np.ndarray]] = None` parameter
- [ ] Add `global_max_count: Optional[int] = None` parameter
- [ ] In the edge computation block (lines 332–341): use `global_edges[col]` when available, fall back to `_get_global_edges()` otherwise
- [ ] After the existing `global_max` computation: override with `global_max_count` when provided

**Files Modified:**
- `code/src/analysis/touch_analytics/reporting.py` — `generate_touch_density_heatmap()`: two new optional params, conditional edge/max logic (~15 lines)

**Dependencies:** None

### Phase 3: Pre-compute global edges and max in the pipeline
**Goal:** Compute edges and max from full pooled data, default x-axis to velocity, and pass to each session's heatmap call.

- [ ] Change `x_col` selection: prefer the first feature column whose name contains `velocity` (e.g. `max_velocity`); fall back to `_select_highest_variance_feature()` if none matches. This aligns with `matrix_generation.py` which hardcodes `heat_x = 'max_velocity'`.
- [ ] After selecting `x_col` and `y_cols`, compute `global_edges` dict using `np.linspace(series.min(), series.max(), num_bins + 1)` for each feature column from the full `result_df`
- [ ] Pre-compute `global_max_count` by iterating all sessions × interaction types × features with the global edges
- [ ] Pass `global_edges` and `global_max_count` to each `reporter.generate_touch_density_heatmap()` call

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — `_generate_session_heatmaps()`: pre-compute edges/max, pass to reporter (~35 lines)

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run unified pipeline with binning clustering on ≥ 2 sessions
- [ ] Open two session heatmap PNGs side by side — confirm identical axis tick labels and ranges
- [ ] Verify that a session with fewer touches appears visually sparser (lighter colors) than a denser session
- [ ] Inspect `cluster_metadata.json` — confirm `bin_edges` key exists with per-feature float arrays
- [ ] Run unified pipeline with kmeans clustering — confirm heatmaps still generate correctly (backward compat)

### Edge Cases
- [ ] Extraction profile without a velocity column (e.g. `mos` extractor) — verify fallback to highest-variance feature
- [ ] Single-session run — global edges are that session's edges, behavior unchanged
- [ ] Session with all-constant feature — `np.linspace(v, v, n+1)` degenerates; verify no crash (handled by existing skip logic in BinningClusterer)
- [ ] Session where all touches are one type (all tap, no stroke) — verify empty subplot renders "No Data" text

---

## Documentation Plan

- [ ] No external docs needed — this is an internal pipeline improvement
- [ ] Code comments on the global-edges computation explaining why it exists

---

## Rollback Plan

1. Revert the three modified files to their pre-change state
2. No data migration needed — output CSVs and PNGs are regenerated on each run
3. `cluster_metadata.json` gains a new key (`bin_edges`) but existing consumers ignore unknown keys

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `pd.cut` with global edges assigns NaN to values outside range | Low (edges come from pooled data, no session can exceed) | Med | Verify with `include_lowest=True` and check for NaN in output |
| Pre-computing global_max doubles iteration over sessions | Low | Low | Negligible cost — crosstab on small DataFrames (~hundreds of rows per session) |
| Heatmap tick labels become crowded with wide global range | Med | Low | Existing `fmt_cats` with `:.2f` formatting handles this; can adjust later |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — confirms mm-native units throughout, no hidden conversions
- Existing pattern: `VisualReportingStrategy.generate_population_heatmaps()` already pre-computes global edges before iterating populations (reporting.py lines 98–106)
- Related plan: `docs/development/plans/active/per-session-touch-density-heatmaps.md`
