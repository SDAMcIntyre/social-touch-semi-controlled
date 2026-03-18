# Plan: Per-Session Touch Density Heatmaps

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-18 11:29
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

**What:** Add per-session touch density heatmaps to the unified touch analysis pipeline, generated for each (extraction profile × clustering profile) combination.

**Why:** After clustering, there is no visualization of how touches distribute across feature space for each session. These heatmaps reveal the density landscape of individual sessions — showing where touches concentrate in pairwise feature projections, split by interaction type.

**How:** Add a new `generate_touch_density_heatmap()` method to `VisualReportingStrategy` that renders an N×2 subplot grid (N remaining features × tap/stroke), and wire it into `_cluster_profile()` so heatmaps are generated automatically after each clustering run.

## Problem Statement

- The unified pipeline produces per-session CSVs and pooled clustered CSVs but no visualizations.
- There is no way to inspect how touches spread across feature dimensions per session without external tooling.
- The existing `generate_population_heatmaps()` in `reporting.py` uses a fixed 2×2 layout with hardcoded axis choices, which does not generalize to arbitrary extractor outputs.

## Goals

### In Scope

1. Per-session density heatmaps: one figure per session, per (extraction × clustering) combo
2. N×2 layout: N = num_features − 1 rows (one per remaining feature), 2 columns (tap | stroke)
3. Shared X-axis across all rows: the feature with highest variance
4. Y-axis per row: each remaining feature, binned
5. Color: touch count with LogNorm scaling
6. Add `session_id` column to the pooled clustered CSV so sessions can be split
7. Output grouped by (extraction × clustering) under `heatmaps/` subdirectory

### Out of Scope

- Cluster label visualization in the heatmap (density only, no cluster coloring)
- AP efficacy / spike_elicited visualization (separate existing feature)
- Interactive or animated figures
- Changes to existing `generate_population_heatmaps()` method

## Success Criteria

- [ ] `pooled_touch_summary_clustered.csv` contains a `session_id` column
- [ ] `heatmaps/` directory exists under each `clustering/<clusterer>/` output
- [ ] One PNG per session in each `heatmaps/` directory
- [ ] Figure shows N×2 grid (N = num_features − 1), columns are tap and stroke
- [ ] X-axis label matches the highest-variance feature name
- [ ] Each row Y-axis label matches a different remaining feature
- [ ] Color uses LogNorm (log-scaled count), bins with 0 touches are masked
- [ ] Axis labels use correct units (mm, mm²) per knowledge base note
- [ ] Pipeline runs end-to-end without errors; heatmap generation does not break idempotency

---

## Technical Design

### Approach

Add a new visualization method to the existing `VisualReportingStrategy` class that handles the N×2 layout with dynamic feature selection. Wire it into the existing `_cluster_profile()` function, which already has `feature_cols` and the clustered DataFrame available. Add `session_id` to the pooled CSV during the concat step so per-session filtering is possible.

The highest-variance feature is selected at runtime from the clustered data, ensuring the X-axis always shows the most discriminative dimension for that particular extraction method.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New method on `VisualReportingStrategy` | Reuses existing infra (_get_global_edges, _save_plot), consistent style | Adds to an existing class | **Chosen** |
| Standalone function in `unified_pipeline.py` | Self-contained | Duplicates bin-edge logic, breaks separation of concerns | Rejected |
| Extend `generate_population_heatmaps()` | Single method | Already complex (295 lines), different layout (2×2 vs N×2) | Rejected |

### Architecture Changes

No new modules. Two existing files modified:

```
code/src/analysis/touch_analytics/
├── reporting.py           # MODIFIED — add generate_touch_density_heatmap()
└── unified_pipeline.py    # MODIFIED — add session_id, variance helper, heatmap wiring
```

Output structure added:

```
unified_touches/<extraction>/clustering/<clusterer>/
  pooled_touch_summary_clustered.csv   # existing (now includes session_id)
  cluster_metadata.json                # existing
  heatmaps/                            # NEW
    <session_id>_touch_density.png     # one per session
```

#### Heatmap Layout Example (max extractor, 4 features)

Highest-variance feature = `max_velocity` → X-axis (shared). Remaining 3 → rows.

```
            tap                    stroke
         x = max_velocity         x = max_velocity
    ┌──────────────────┐     ┌──────────────────┐
    │ Y: max_depth      │     │ Y: max_depth      │
    │ (count heatmap)   │     │ (count heatmap)   │
    ├──────────────────┤     ├──────────────────┤
    │ Y: max_contact_   │     │ Y: max_contact_   │
    │    area            │     │    area            │
    ├──────────────────┤     ├──────────────────┤
    │ Y: max_           │     │ Y: max_           │
    │    acceleration    │     │    acceleration    │
    └──────────────────┘     └──────────────────┘

Color = touch count (LogNorm, cmap='magma')
```

---

## Implementation Plan

### Phase 1: Add `session_id` to pooled CSV
**Goal:** Enable per-session filtering of the clustered data.

- [ ] Task 1.1 — In `_cluster_profile()`, derive `session_id` from each session CSV filename during pooling (same pattern as `session_summary.py:91–95`: split on `_semicontrolled_`)
- [ ] Task 1.2 — Add `'session_id'` to `_SHARED_COLUMNS` so it is excluded from feature column detection

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — pooling loop in `_cluster_profile()`, `_SHARED_COLUMNS` list

**Dependencies:** None

### Phase 2: Heatmap rendering method
**Goal:** Create the N×2 density heatmap renderer.

- [ ] Task 2.1 — Add `generate_touch_density_heatmap(df, x_col, y_cols, type_col, num_bins, log_axis, title_suffix, filename)` to `VisualReportingStrategy`
- [ ] Task 2.2 — Reuse `_get_global_edges()` for bin computation (both X and each Y)
- [ ] Task 2.3 — Pre-compute `global_max` count across all subplots for shared LogNorm
- [ ] Task 2.4 — Render N×2 subplots: each row = one y_col, each column = tap/stroke. Use `pd.crosstab(y_binned, x_binned)` for count matrix, `sns.heatmap` with `LogNorm(vmin=1, vmax=global_max)`, mask zero bins
- [ ] Task 2.5 — Handle empty subplots ("No Data" text, same pattern as existing `generate_population_heatmaps` line ~280)
- [ ] Task 2.6 — Set suptitle, axis labels (include feature names), tight_layout, save via `_save_plot()`

**Files Modified:**
- `code/src/analysis/touch_analytics/reporting.py` — new method on `VisualReportingStrategy`

**Dependencies:** None (can be done in parallel with Phase 1)

### Phase 3: Orchestration wiring
**Goal:** Wire heatmap generation into the clustering step.

- [ ] Task 3.1 — Add `_select_highest_variance_feature(df, feature_cols)` helper in `unified_pipeline.py`
- [ ] Task 3.2 — Add `_generate_session_heatmaps(result_df, out_dir, feature_cols)` helper: selects X-axis, computes y_cols, iterates sessions, calls `generate_touch_density_heatmap()`
- [ ] Task 3.3 — Wire `_generate_session_heatmaps()` into `_cluster_profile()` after the CSV/JSON save block (~line 368). `feature_cols` and `result_df` are already available at that point.

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — two new helpers, one call added to `_cluster_profile()`

**Dependencies:** Phases 1 and 2

---

## Testing Plan

### Manual Verification

- [ ] Run `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml` end-to-end with `unified_touch_analysis` enabled
- [ ] Verify `pooled_touch_summary_clustered.csv` contains `session_id` column
- [ ] Verify `heatmaps/` directory exists under each `clustering/<clusterer>/` output
- [ ] Verify one PNG per session in each `heatmaps/` directory
- [ ] Open a PNG: confirm N×2 layout, X-axis label = highest-variance feature, each row Y = different feature
- [ ] Confirm color scale is log-based (LogNorm), bins with 0 touches are blank/masked
- [ ] Confirm figure title includes session ID

### Edge Cases

- [ ] Session with only tap touches (no strokes) — stroke panel should show "No Data"
- [ ] Session with only stroke touches (no taps) — tap panel should show "No Data"
- [ ] Extractor with only 1 feature (e.g. hypothetical) — should log warning and skip (need ≥ 2 features)
- [ ] Session with very few touches — heatmap still renders, most bins empty/masked
- [ ] `session_id` column is excluded from `feature_cols` — verify it doesn't appear as a heatmap axis

---

## Documentation Plan

- [ ] No external documentation changes needed (internal pipeline enhancement)
- [ ] Inline docstrings on new methods (`generate_touch_density_heatmap`, `_select_highest_variance_feature`, `_generate_session_heatmaps`)

---

## Rollback Plan

1. Revert the merge commit — all changes are in 2 files
2. No data migration needed — `session_id` is a new column in the CSV, downstream consumers ignore unknown columns
3. `heatmaps/` directories can be deleted without affecting any other pipeline output

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large N (many features) makes figure too tall | Medium | Low | Scale figsize height dynamically: `figsize=(14, 5*N)` |
| `_get_global_edges()` returns degenerate bins for near-constant features | Low | Low | Already handled — falls back to `np.linspace(0, 1, num_bins+1)` for empty series |
| Heatmap generation slows pipeline for many sessions | Low | Medium | Heatmaps are generated once per clustering run, tied to same idempotency check |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: session_id | ~10 LOC | None |
| Phase 2: heatmap method | ~80 LOC | None |
| Phase 3: wiring | ~30 LOC | Phases 1, 2 |

---

## References

- Existing heatmap renderer: `code/src/analysis/touch_analytics/reporting.py` — `VisualReportingStrategy`
- Clustering orchestrator: `code/src/analysis/touch_analytics/unified_pipeline.py` — `_cluster_profile()`
- Session ID extraction pattern: `code/src/analysis/touch_analytics/session_summary.py:91–95`
- Units reference: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (all spatial in mm/mm²)
- Parent plan: `docs/development/plans/pending/unified-touch-analysis-pipeline.md`
