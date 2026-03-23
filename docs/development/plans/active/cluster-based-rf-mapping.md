# Plan: Cluster-Based Receptive Field Mapping

**Created:** 2026-03-17 00:30
**Revised:** 2026-03-23
**Approved:** ---
**Completed:** ---
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/cluster-based-rf-mapping`

---

## Overview

**What:** A new analysis task that reads clustered touch summaries from `touch_clustering`, traces each touch back to its session's aggregated CSV to extract spike-filtered contact points (with proper 30Hz-to-1kHz forward-fill), counts spikes per contact point, and renders 3D forearm PLY heatmaps per session per cluster.

**Why:** The existing `map_receptive_fields` flow groups by experimenter-defined metadata (`type_metadata`, `direction`) and uses selectivity+DBSCAN. The clustering pipeline now assigns data-driven cluster labels; RF mapping should leverage these labels. The existing loader also has a data-loss bug: it skips rows where `contact_points` is NaN instead of forward-filling, dropping ~97% of 1kHz spike rows.

**How:** Create a new pipeline module (`rf_cluster_pipeline.py`) and visualizer (`rf_cluster_visualizer.py`) that implement the cluster-first iteration, forward-fill, spike counting, and 3D heatmap rendering. Wire as a new Prefect flow depending on `touch_clustering`.

## Problem Statement

- The clustering pipeline assigns `cluster_label` to touches, but the existing RF mapping flow ignores these labels entirely.
- The existing RF data loader does not forward-fill `contact_points` (30Hz) to match the 1kHz nerve data rate, silently losing most spike-associated contact points.
- The existing loader uses `(trial_id, single_touch_id)` as the touch key; the correct unique key is `(session_id, block_order_id, trial_id, single_touch_id)`.
- Visualization is 2D matplotlib scatter; the user needs 3D forearm PLY with spike-count heatmap overlay and camera normal to the contact surface.

## Goals

### In Scope

1. Read `pooled_touch_summary_clustered.csv` and group by `cluster_label`
2. Trace touches back to per-session aggregated CSVs using the 4-column key
3. Forward-fill `contact_points` (30Hz -> 1kHz), filter `Nerve_spike == 1`, count spikes per `(x, y, z)` point
4. Save per-cluster `spike_counts.csv` (pooled across all sessions)
5. Render per-session 3D forearm heatmap PNGs with camera normal to contact surface
6. DAG integration as `map_receptive_fields_clustered` depending on `touch_clustering`

### Out of Scope

- Modifying the existing `map_receptive_fields` flow (it continues to work as-is)
- Modifying existing `rf_data_loader.py`, `rf_mapping_engine.py`, or `rf_visualizer.py`
- Adding new extraction or clustering methods
- Interactive Open3D visualization
- Cross-cluster or cross-session comparison analysis (future work)

## Success Criteria

- [ ] New task `map_receptive_fields_clustered` runs after `touch_clustering` via DAG
- [ ] For each (feature_combination, clusterer, cluster_label): a `spike_counts.csv` is produced with columns `(x, y, z, spike_count)`
- [ ] For each (feature_combination, clusterer, cluster_label, session_id): a `<session_id>_rf_heatmap.png` showing 3D forearm with heatmap overlay
- [ ] Contact points are forward-filled within each touch before spike filtering
- [ ] The 4-column touch key `(session_id, block_order_id, trial_id, single_touch_id)` is used throughout
- [ ] Idempotency: re-running with unchanged inputs skips processing
- [ ] Existing `map_receptive_fields` flow produces identical output (no regression)

---

## Technical Design

### Algorithm

For each enabled (feature_combination, clusterer) pair:

1. Read `pooled_touch_summary_clustered.csv` from `4_analysed/touch_clusters/<combination>/<clusterer>/`
2. Group by `cluster_label`
3. For each `cluster_label`:
   a. Extract unique `(session_id, block_order_id, trial_id, single_touch_id)` combinations
   b. Group these combinations by `session_id`
   c. For each `session_id`:
      - Load the session's `*_semicontrolled_aggregated_session.csv`
      - Filter to rows matching `(block_order_id, trial_id, single_touch_id)`
      - Keep only: `contact_points`, `Nerve_freq`, `Nerve_spike` (plus the 4 key columns)
      - Forward-fill `contact_points` within each touch group (30Hz -> 1kHz alignment)
      - Filter rows where `Nerve_spike == 1`
      - Parse `contact_points`, count spikes per `(x, y, z)` point
   d. Aggregate spike counts across all sessions -> save `spike_counts.csv`
   e. For each session: load forearm PLY, render 3D heatmap -> save PNG

### Forward-Fill Detail

The aggregated CSV has one row per 1kHz neural sample. `contact_points` is populated only every ~33 rows (30Hz kinect). Between populated rows it is NaN. Forward-fill within each touch group assigns each neural sample the most recent contact-point observation:

```python
touch_df['contact_points'] = touch_df.groupby(
    ['block_order_id', 'trial_id', 'single_touch_id']
)['contact_points'].ffill()
```

This must be done **within** each touch (not across touches) to prevent leaking contact points from one touch into the next.

### Approach

Create **two new modules** in `code/src/analysis/receptive_field_mapping/`, keeping the new code cleanly separated from the existing selectivity+DBSCAN pipeline:

- `rf_cluster_pipeline.py` — data loading, spike counting, orchestration
- `rf_cluster_visualizer.py` — 3D forearm heatmap rendering via matplotlib 3D

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone modules | Clean separation; no risk of breaking existing RF code | Two new files to maintain | **Chosen** |
| Extend existing `rf_data_loader.py` + `rf_mapping_engine.py` | Reuses existing infrastructure | Fundamentally different computation (spike counts vs selectivity); would pollute API | Rejected |
| Open3D offscreen rendering for 3D heatmaps | Native 3D, consistent with existing viewer | Known driver issues on some machines; harder camera control for static export | Rejected |
| Matplotlib 3D for heatmaps | Portable; no GPU driver issues; easy camera control; native PNG export | Slightly less polished 3D rendering | **Chosen** |

### Architecture Changes

New files:

```
code/src/analysis/receptive_field_mapping/
  rf_cluster_pipeline.py      -- Data loading, spike counting, orchestration
  rf_cluster_visualizer.py    -- 3D forearm heatmap rendering (matplotlib 3D)
```

Modified files:

```
code/src/analysis/receptive_field_mapping/__init__.py  -- Export new public API
code/scripts/analysis_workflow.py                       -- New flow + registration
configs/analyse_workflow_dag.yaml                       -- New task entry
```

Output directory structure:

```
4_analysed/receptive_field_maps_clustered/
  <combination>/                      # e.g. only_mean, only_max
    <clusterer>/                      # e.g. kmeans, hierarchical
      cluster_<N>/
        spike_counts.csv              # columns: x, y, z, spike_count (pooled)
        <session_id>_rf_heatmap.png   # 3D forearm + contact heatmap
      cluster_<M>/
        spike_counts.csv
        <session_id>_rf_heatmap.png
      rf_cluster_summary.json         # metadata for all clusters
```

### Reused Code

| Function | Location | Usage |
|----------|----------|-------|
| `parse_contact_points()` | `rf_data_loader.py:28` | Parse contact_points string to list of (x, y, z) |
| `session_id_from_path()` | `pipeline_shared.py:87` | Extract session_id from CSV filename |
| `filter_enabled_profiles()` | `pipeline_shared.py:82` | Filter enabled feature_combinations / clustering_profiles |
| `should_process_task()` | `utils/should_process_task.py` | Idempotency check |

---

## Implementation Plan

### Phase 1: Spike-Count Data Pipeline
**Goal:** Build the core data pipeline that reads clustered CSV, traces touches to aggregated session CSVs, forward-fills contact_points, filters by Nerve_spike==1, and counts spikes per contact point.
**Started:** 2026-03-23
**Completed:** 2026-03-23

- [x] Create `rf_cluster_pipeline.py` with:
  - `_resolve_session_paths(input_items) -> dict[str, Path]` — maps session_id to session_merged_output_dir using `session_id_from_path()`; aggregated CSV found via `session_dir.glob("*_semicontrolled_aggregated_session.csv")`; forearm PLY at `session_dir / "forearm_pca_calibrated" / f"{session_id}_forearm.ply"`
  - `_group_touches_by_cluster(clustered_df) -> dict[str, pd.DataFrame]` — groups by `cluster_label`, returns unique 4-col touch keys per cluster
  - `_build_session_touch_map(cluster_df) -> dict[str, list[tuple]]` — groups touch keys by session_id -> list of `(block_order_id, trial_id, single_touch_id)`
  - `_extract_spike_contact_points(aggregated_csv_path, touch_keys) -> Counter` — reads with `usecols` for memory efficiency, filters to matching touches, forward-fills `contact_points` within each touch group, filters `Nerve_spike == 1`, parses contact_points, returns `Counter[(x,y,z)] -> spike_count`
  - `_aggregate_spike_counts(session_counters: list[Counter]) -> pd.DataFrame` — sums all session Counters into DataFrame with columns `(x, y, z, spike_count)`
- [x] Reuse `parse_contact_points()` from `rf_data_loader.py`
- [x] Handle edge cases: missing aggregated CSV, empty touches, NaN-only contact_points

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`

**Dependencies:** None (reads existing outputs from touch_clustering)

### Phase 2: 3D Forearm Heatmap Visualization
**Goal:** Create the visualization module that renders the forearm PLY with a spike-count heatmap overlay, camera oriented normal to the contact surface.
**Started:** 2026-03-23
**Completed:** 2026-03-23

- [x] Create `rf_cluster_visualizer.py` with:
  - `render_forearm_heatmap(forearm_ply_path, spike_counts_df, output_path, session_id, cluster_label)`:
    - Load forearm PLY via `o3d.io.read_point_cloud()`, extract vertices as numpy array
    - Plot forearm vertices as subtle grey scatter in matplotlib 3D (`Axes3D`, small point size, low alpha)
    - Overlay contact points colored by spike_count using a colormap (e.g. `hot` or `YlOrRd`)
    - Compute camera angle normal to contact surface:
      - Centroid of contact points
      - KD-tree from forearm vertices, find K nearest neighbors of centroid
      - PCA on neighbors -> smallest eigenvector = surface normal
      - Convert to `ax.view_init(elev, azim)` angles
      - Fall back to sensible default (45 deg elevation) if normal is degenerate
    - Add colorbar for spike count, title with session_id and cluster_label
    - Axis labels in mm (per knowledge base: all coordinates are in mm from Kinect SDK)
    - Save PNG at 200 DPI
- [x] Handle edge cases: missing forearm PLY (skip with warning), empty spike_counts (skip)

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py`

**Dependencies:** Phase 1

### Phase 3: Orchestration and DAG Integration
**Goal:** Wire the spike-count computation and visualization into a Prefect flow, register in the DAG, implement idempotency.
**Started:** 2026-03-23
**Completed:** 2026-03-23

- [x] Add `run_cluster_rf_mapping()` to `rf_cluster_pipeline.py`:
  - Accept: `clustering_dir`, `input_items`, `output_dir`, `feature_combinations`, `clustering_profiles`, `force`
  - Iterate enabled (combination, clusterer) pairs via `filter_enabled_profiles()`
  - Locate `pooled_touch_summary_clustered.csv` at `clustering_dir/<combination>/<clusterer>/`
  - Idempotency via `should_process_task()` with clustered CSV as input, `rf_cluster_summary.json` as output sentinel
  - Group by cluster_label, process each cluster, save `spike_counts.csv`
  - For each (cluster, session): call `render_forearm_heatmap()`
  - Save `rf_cluster_summary.json` with metadata (cluster counts, session coverage, spike statistics)
- [x] Add `map_receptive_fields_clustered_flow()` to `analysis_workflow.py`:
  - Signature: `(input_items, force_processing, feature_combinations, clustering_profiles)`
  - Compute `clustering_dir = database_path / '4_analysed' / 'touch_clusters'`
  - Compute `output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'`
  - Build session_id -> session_merged_output_dir mapping from input_items
  - Call `run_cluster_rf_mapping()`
- [x] Register `("map_receptive_fields_clustered", map_receptive_fields_clustered_flow)` in `available_tasks`
- [x] Add kwargs dispatch for `feature_combinations` and `clustering_profiles` options (reuses existing dispatch)
- [x] Add task to `configs/analyse_workflow_dag.yaml`:
  ```yaml
  map_receptive_fields_clustered:
    enabled: false
    options:
      force_processing: false
      feature_combinations:
        only_mean: {enabled: true, features: [mean]}
        only_max: {enabled: true, features: [max]}
        only_median: {enabled: true, features: [median]}
      clustering_profiles:
        binning: {method: binning}
        kmeans: {method: kmeans}
        dbscan: {method: dbscan}
        hierarchical: {method: hierarchical}
    depends_on: [touch_clustering]
  ```
- [x] Export `run_cluster_rf_mapping` from `__init__.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add orchestration function
- `code/scripts/analysis_workflow.py` — new flow, registration, kwargs dispatch
- `configs/analyse_workflow_dag.yaml` — new task entry
- `code/src/analysis/receptive_field_mapping/__init__.py` — export additions

**Dependencies:** Phases 1 and 2

---

## Testing Plan

### Integration Tests
- [ ] Run `touch_clustering` then `map_receptive_fields_clustered` end-to-end; verify output directory structure matches spec
- [ ] Verify `spike_counts.csv` has columns `(x, y, z, spike_count)` with plausible positive integer values
- [ ] Verify PNGs are generated for each (cluster, session) with non-zero contact points
- [ ] Run existing `map_receptive_fields` flow; verify identical output (no regression)

### Manual Verification
- [ ] Inspect a `spike_counts.csv`: coordinates are in mm, spike_count values are positive integers
- [ ] Inspect a heatmap PNG: forearm shape visible as grey background, contact points as colored overlay, colorbar present
- [ ] Verify forward-fill: for a known touch, the spike_counts reflect the expected contact location

### Edge Cases
- [ ] Cluster with touches from only one session — spike_counts.csv still produced
- [ ] Session with no touches in a cluster — no PNG for that session, no crash
- [ ] Missing aggregated CSV for a session_id — log warning, skip gracefully
- [ ] Missing forearm PLY — skip figure, log warning, still produce spike_counts.csv
- [ ] All contact_points NaN for a touch (no kinect data) — skip that touch
- [ ] Cluster with 0 spikes after filtering — produce empty spike_counts.csv or skip with warning
- [ ] Re-run with unchanged inputs — idempotency skips processing

---

## Documentation Plan

- [ ] Docstrings on all public functions in `rf_cluster_pipeline.py` and `rf_cluster_visualizer.py`
- [ ] Update `__init__.py` module docstring to mention cluster-based pipeline
- [ ] Inline comments for forward-fill logic and camera normal computation

---

## Rollback Plan

1. Revert the commits on `feature/cluster-based-rf-mapping`
2. All new code is in new files (`rf_cluster_pipeline.py`, `rf_cluster_visualizer.py`) — existing code untouched except `analysis_workflow.py` (new flow function) and `__init__.py` (new exports)
3. Remove `map_receptive_fields_clustered` task from `analyse_workflow_dag.yaml`
4. No data migrations; output is in `receptive_field_maps_clustered/`, separate from existing `receptive_field_maps/`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large aggregated CSVs (1kHz data) cause memory issues | Med | Med | Read only needed columns via `usecols`; process one session at a time |
| Forward-fill across touch boundaries produces incorrect data | Med | High | Forward-fill within `groupby([block_order_id, trial_id, single_touch_id])`, never across touches |
| Cluster labels change on re-run (non-deterministic clustering) | High | Low | Inherent to clustering; `rf_cluster_summary.json` provides traceability; idempotency prevents unnecessary re-runs |
| Matplotlib 3D camera normal computation produces bad angle | Med | Low | Fall back to sensible default (45 deg elevation) if PCA normal is degenerate |
| contact_points parsing slow for millions of rows | Med | Med | Use `.itertuples()` instead of `.iterrows()` for hot loops |

---

## References

- Existing RF mapping: `code/src/analysis/receptive_field_mapping/`
- Clustering pipeline: `code/src/analysis/touch_analytics/clustering_pipeline.py`
- Pipeline shared utilities: `code/src/analysis/touch_analytics/pipeline_shared.py`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (coordinates in mm)
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md` (forearm PLY registration)

---
