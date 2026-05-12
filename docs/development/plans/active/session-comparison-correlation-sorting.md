# Plan: Session Comparison Correlation Sorting

**Date:** 2026-05-12
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/neural-kinect-viewer-responsive-navigation`
**Branch:** `feature/neural-kinect-viewer-responsive-navigation`
**Started:** 2026-05-12

---

## Overview

Add hierarchical-clustering-based row reordering and a dendrogram panel to the session comparison heatmap, so that sessions with similar RF metric profiles are grouped visually rather than listed alphabetically.

## Problem Statement

The session comparison heatmap (`visualize_session_comparison`) currently sorts sessions alphabetically on the Y-axis. This makes it difficult to spot which sessions have similar RF response profiles across velocity bins. Correlation-based reordering with a dendrogram provides immediate visual grouping of similar sessions — a standard technique in heatmap visualisation (analogous to gene-expression clustermaps).

## Goals

### In Scope
1. Reorder heatmap rows by hierarchical clustering (correlation distance, average linkage)
2. Render a dendrogram panel to the left of the heatmap
3. Handle edge cases: <3 sessions (skip clustering), all-NaN rows, constant rows

### Out of Scope
- Column (velocity bin) clustering — bins remain in ascending numerical order
- Interactive dendrogram or GUI integration
- New DAG config options — dendrogram+reorder becomes the default behaviour

## Success Criteria

- [ ] Sessions with similar metric profiles appear adjacent in the heatmap
- [ ] Dendrogram panel renders to the left, aligned with heatmap rows
- [ ] Fewer than 3 sessions: falls back to alphabetical order, no dendrogram
- [ ] All-NaN rows handled (appended at bottom, excluded from clustering)
- [ ] Existing heatmap layout (colorbar, labels, title) unchanged
- [ ] Unit tests pass for the distance computation and clustering logic

---

## Technical Design

### Approach

**Distance metric: Correlation distance** (`1 - Pearson r`). Captures profile shape similarity — two sessions that rise and fall in the same bins cluster together, regardless of absolute magnitude. This is the standard choice for heatmap clustermaps.

**Linkage: Average (UPGMA) with `optimal_ordering=True`**. Ward's requires Euclidean distance. Average linkage is the standard pairing with correlation distance. `optimal_ordering=True` (on `linkage()`, scipy 1.0+) minimises distance between adjacent leaves for a more coherent heatmap.

**NaN handling: Pairwise-complete correlation**. For each session pair, correlate only the bins where both have finite values. <2 shared bins or constant rows → distance 1.0.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Correlation distance + average linkage | Shape-based similarity, standard for heatmaps | Requires NaN handling | **Chosen** |
| Euclidean distance + Ward's linkage | Already used in `hierarchical_clusterer.py` | Dominated by magnitude, not shape | Rejected |
| seaborn `clustermap` | Built-in dendrogram + heatmap | Different rendering style, less control over layout, adds dependency coupling | Rejected |

### Architecture Changes

All changes are confined to a single file: `rf_session_comparison_renderer.py`. Two new private functions are added; two existing functions are modified. No new modules, no API changes.

---

## Implementation Plan

### Phase 1: Add clustering functions
**Goal:** Implement correlation distance and hierarchical clustering logic

**Tasks:**
- [x] Task 1.1 — Add imports: `from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage`
- [x] Task 1.2 — Add `_nan_safe_correlation_distance(matrix)` → condensed distance vector. O(n^2) pair loop (n typically 5-20). For each pair: mask to shared finite bins, compute `1 - corrcoef`, clamp to [0, 2]. <2 shared bins or constant row → distance 1.0.
- [x] Task 1.3 — Add `_cluster_session_rows(matrix)` → `(reordered_matrix, linkage_matrix | None)`. Separate all-NaN rows → if <3 clusterable rows return `(matrix, None)` → compute distance → `linkage(dist, method='average', optimal_ordering=True)` → `leaves_list(Z)` → reorder → append all-NaN rows at bottom.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_session_comparison_renderer.py` — add imports (line 8 area), add two functions between `_build_session_feature_matrix` and `render_session_comparison_heatmap`

**Dependencies:** None

### Phase 2: Wire clustering into orchestrator and renderer
**Goal:** Connect clustering output to heatmap rendering with dendrogram

**Tasks:**
- [x] Task 2.1 — In `run_session_comparison_visualization`, after `_build_session_feature_matrix()` call (line 227), add `matrix, linkage_matrix = _cluster_session_rows(matrix)`
- [x] Task 2.2 — Pass `linkage_matrix` to `render_session_comparison_heatmap()`
- [x] Task 2.3 — Add `linkage_matrix: np.ndarray | None = None` parameter to `render_session_comparison_heatmap`
- [x] Task 2.4 — When `linkage_matrix is not None`: create 2-column GridSpec (`width_ratios=[1, 5]`), render dendrogram on left axis with `orientation='left'`, heatmap on right axis. Increase `fig_w` by 1.5 for the dendrogram panel. Dendrogram style: uniform dark gray (`color_threshold=0, above_threshold_color='#444444'`), `no_labels=True`, `ax_dendro.set_axis_off()`.
- [x] Task 2.5 — When `linkage_matrix is None`: keep existing single-axis layout (backward compatible)
- [x] Task 2.6 — Align dendrogram Y-limits with heatmap Y-limits so leaves match rows. Dendrogram leaves at `10*i + 5`, heatmap rows at `i` to `i+1` — set `ax_dendro.set_ylim(0, n_sessions * 10)` and `ax.set_ylim(0, n_sessions)`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_session_comparison_renderer.py` — modify `render_session_comparison_heatmap()` (lines 46-87), modify `run_session_comparison_visualization()` (lines 227-238)

**Dependencies:** Phase 1

### Phase 3: Tests
**Goal:** Verify clustering and distance computation

**Tasks:**
- [x] Task 3.1 — `test_nan_safe_correlation_distance_basic`: 3 rows with known correlation values (perfectly correlated → 0, anticorrelated → 2, uncorrelated → ~1)
- [x] Task 3.2 — `test_nan_safe_correlation_distance_with_nans`: partial NaN overlap between rows
- [x] Task 3.3 — `test_nan_safe_correlation_distance_constant_row`: zero-variance row → distance 1.0
- [x] Task 3.4 — `test_cluster_session_rows_reorders`: verify reordering happens and linkage returned
- [x] Task 3.5 — `test_cluster_session_rows_few_sessions`: 1-2 sessions → `(matrix, None)`
- [x] Task 3.6 — `test_cluster_session_rows_all_nan_row`: all-NaN row appended at bottom

**Files Modified:**
- `code/tests/test_rf_session_comparison_renderer.py` — new test file

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] Distance computation with known correlations (perfectly correlated → 0, anticorrelated → 2, uncorrelated → 1)
- [ ] NaN handling: partial overlap, no overlap (<2 shared bins), all-NaN rows
- [ ] Clustering fallback: <3 sessions returns `None` linkage
- [ ] Reordering: verify similar rows become adjacent

### Manual Verification
- [ ] Run the pipeline on existing data, compare old (alphabetical) vs new (clustered) PNGs
- [ ] Verify dendrogram aligns visually with heatmap rows
- [ ] Verify <3 sessions still renders without dendrogram

### Edge Cases
- [ ] Single session — no clustering, no dendrogram, renders normally
- [ ] Two sessions — no clustering, no dendrogram
- [ ] All sessions have identical profiles — dendrogram is flat, order is arbitrary but stable
- [ ] One session entirely NaN — excluded from clustering, appended at bottom

---

## Documentation Plan

- [ ] No README or CLAUDE.md changes needed (internal rendering change, no API change)

---

## Rollback Plan

1. Revert the single modified file (`rf_session_comparison_renderer.py`) and delete the test file
2. No data format changes, no config changes — fully reversible

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dendrogram Y-axis misalignment with heatmap rows | Med | Med | Explicit Y-limit alignment after both axes rendered; visual verification |
| `optimal_ordering=True` slow for many sessions | Low | Low | Typical session count is 5-20; O(n^3) negligible at that scale |
| NaN-heavy matrix produces degenerate clustering | Low | Med | Fallback to alphabetical when <3 clusterable sessions; distance=1.0 for uncorrelatable pairs |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Clustering functions | ~50 lines | None |
| Phase 2 — Renderer integration | ~30 lines changed | Phase 1 |
| Phase 3 — Tests | ~80 lines | Phase 1 |
