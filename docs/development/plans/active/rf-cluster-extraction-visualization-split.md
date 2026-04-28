# Plan: Split RF Cluster Pipeline into Extraction and Visualization Flows

**Date:** 2026-04-27
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-extraction-visualization-split`

---

## Overview

**What:** Split the monolithic `run_cluster_rf_mapping()` function into two independent DAG tasks — an extraction flow (data loading, spike counting) and a visualization flow (metrics, heatmap rendering).

**Why:** Currently, changing any visualization parameter (`projection_method`, `disjoint_mask_distance_mm`) forces a full re-extraction (~30-120s per session of CSV parsing) that produces identical results. The single idempotency sentinel doesn't distinguish between data-dependent and vis-param-dependent work.

**How:** Factor the function into `run_cluster_rf_extraction()` and `run_cluster_rf_visualization()`, connected by well-defined intermediate artifacts on disk (`.npy` arrays, CSVs, JSON metadata). Register as separate Prefect flows with independent sentinels. Keep a backward-compatible wrapper.

## Problem Statement

`rf_cluster_pipeline.py::run_cluster_rf_mapping()` (750 lines) mixes two concerns with different parameter dependencies:

- **Extraction** — CSV I/O, forward-fill, regex contact-point parsing, spike counting. Cost: ~30-120s per session. Depends only on `clustered_csv` + session aggregated CSVs.
- **Visualization** — RF metrics computation, 2D projection, heatmap rendering. Cost: ~1-3s per cluster per session. Depends on extraction output + `projection_method`, `disjoint_mask_distance_mm`.

The sole idempotency mechanism (`rf_cluster_summary.json` mtime check) treats both as one unit. When a user changes `projection_method` from `tangent_plane` to `cylindrical_unwrap`, the entire extraction re-runs — loading hundreds of MB of CSVs, regex-parsing every `contact_points` cell, counting every spike — only to produce byte-identical `spike_counts.csv` files.

This blocks future interactive parameter exploration, where a user would toggle visualization parameters and expect near-instant re-rendering.

## Goals

### In Scope

1. Factor `run_cluster_rf_mapping()` into two functions: `run_cluster_rf_extraction()` and `run_cluster_rf_visualization()`, each independently callable.
2. Define and persist intermediate artifacts (see Architecture Changes) so the visualization flow can run without access to aggregated CSVs or `clustering_dir`.
3. Register both as separate Prefect flows and DAG tasks with independent idempotency sentinels.
4. Cache forearm PLY vertices as `.npy` to eliminate `open3d` dependency in the visualization flow.
5. Keep `run_cluster_rf_mapping()` as a thin backward-compatible wrapper that calls both flows sequentially.

### Out of Scope

- Interactive GUI for parameter exploration (future work enabled by this split).
- Changes to the upstream clustering pipeline or feature extraction.
- Changes to `rf_metrics.py`, `rf_cluster_visualizer.py`, `rf_2d_renderer.py`, or `rf_projection.py` internals (they are consumers, not targets).
- Parallelizing extraction across sessions or clusters (orthogonal optimization).
- Changing the simple RF mapping pipeline (`rf_simple_pipeline.py`).

## Success Criteria

- [ ] `run_cluster_rf_extraction()` produces all intermediate artifacts and writes `extraction_summary.json` sentinel.
- [ ] `run_cluster_rf_visualization()` reads only from intermediate artifacts (never from aggregated CSVs or `clustering_dir`) and produces metrics + PNGs identical to the current pipeline.
- [ ] Changing `projection_method` in the DAG config and rerunning triggers only the visualization flow — extraction is skipped (verified via log output).
- [ ] Changing the clustered CSV (newer mtime) triggers both flows.
- [ ] `run_cluster_rf_mapping()` wrapper produces identical output to the current implementation.
- [ ] Both flows respect `force_processing=True` independently.
- [ ] Forearm vertices are loaded from `.npy` cache when available, falling back to PLY + open3d on cache miss.
- [ ] Fail-fast convention respected: missing intermediate artifacts → `ValueError` with diagnostic message; corrupt `.npy` → `ValueError`.
- [ ] Existing `rf_cluster_summary.json` format preserved for backward compatibility.

---

## Technical Design

### Approach

Split at the natural boundary between data extraction and visualization. The extraction flow writes a self-contained artifact directory; the visualization flow reads it. Each flow has its own sentinel for idempotency. No caching framework — just files on disk with mtime-based staleness checks, following the existing `should_process_task()` pattern.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Two independent flows** (chosen) | Clean separation, independent sentinels, enables future GUI, follows DAG pattern | More files on disk, two DAG tasks to configure | **Chosen** — aligns with pipeline architecture |
| **Cache layer inside monolithic function** | Minimal API change, single DAG task | Still one function mixing concerns, cache invalidation complex, harder to call visualization independently | Rejected — doesn't enable independent reruns |
| **In-memory cache (lru_cache / dict)** | Zero disk overhead | Lost between runs, no persistence, doesn't help with pipeline restarts | Rejected — must survive process boundaries |

### Architecture Changes

**New module:**
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — save/load helpers for intermediate artifacts

**Modified modules:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — split into extraction + visualization functions
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — add `load_forearm_vertices()`
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — accept pre-loaded forearm vertices
- `code/scripts/analysis_workflow.py` — register two new Prefect flows
- `configs/analyse_workflow_dag.yaml` — split task configuration

**Intermediate artifact directory structure:**

```
4_analysed/receptive_field_maps_clustered/{combo}/{clusterer}/
    extraction_summary.json                  # Extraction sentinel
    neuron_touches.json                      # {session_id: count} across all clusters
    sessions/
        {session_id}/
            neuron_contacts_xyz.npy          # (N, 3) float64 — all neuron contacts
            forearm_vertices.npy             # (M, 3) float64 — cached from PLY
    cluster_00/
        spike_counts.csv                     # Pooled — already exists
        cluster_description.json             # Already exists
        neuron_cluster_touches.json          # {session_id: count} within this cluster
        sessions/
            {session_id}/
                session_spike_counts.csv     # Single-session spike DataFrame
                cluster_contacts_xyz.npy     # (M, 3) float64 — cluster contacts
    cluster_01/
        ...
    rf_metrics_summary.csv                   # Written by visualization flow
    rf_visualization_summary.json            # Visualization sentinel
```

**New artifacts** (not currently persisted):

| Artifact | Per | Shape | Purpose |
|----------|-----|-------|---------|
| `neuron_contacts_xyz.npy` | session | (N, 3) | Hull boundaries (neuron-all-clusters) |
| `forearm_vertices.npy` | session | (M, 3) | Avoid open3d import in visualization |
| `session_spike_counts.csv` | cluster × session | DataFrame | Per-session heatmap rendering |
| `cluster_contacts_xyz.npy` | cluster × session | (M, 3) | Hull boundaries (neuron ∩ cluster) |
| `neuron_touches.json` | combo/clusterer | dict | Touch counts per session |
| `neuron_cluster_touches.json` | cluster | dict | Touch counts per session in cluster |
| `extraction_summary.json` | combo/clusterer | dict | Extraction sentinel |
| `rf_visualization_summary.json` | combo/clusterer | dict | Visualization sentinel |

**Function signatures:**

```python
def run_cluster_rf_extraction(
    clustering_dir: Path,
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    force: bool = False,
) -> List[Path]:
    """Extract spike data from aggregated CSVs grouped by cluster labels.
    No visualization parameters. Writes intermediate artifacts to output_dir.
    """

def run_cluster_rf_visualization(
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    projection_method: Optional[str] = None,
    disjoint_mask_distance_mm: float = 8.0,
    force: bool = False,
) -> List[Path]:
    """Compute RF metrics and render heatmaps from extraction artifacts.
    No input_items or clustering_dir needed. Reads everything from output_dir.
    """

def run_cluster_rf_mapping(...) -> List[Path]:
    """Backward-compatible wrapper. Calls extraction then visualization."""
```

**DAG configuration:**

```yaml
extract_receptive_fields_clustered:
    enabled: true
    options:
      force_processing: false
      cluster_groups: [kinematics_simple_mean, pressure_velocity_mean]
    depends_on: [touch_clustering]

visualize_receptive_fields_clustered:
    enabled: true
    options:
      force_processing: false
      cluster_groups: [kinematics_simple_mean, pressure_velocity_mean]
      projection_method: cylindrical_unwrap
      disjoint_mask_distance_mm: 8.0
      camera_angle_mode:
        auto:
          enabled: true
    depends_on: [extract_receptive_fields_clustered]
```

**Idempotency design:**

- *Extraction:* `extraction_summary.json` mtime compared against `clustered_csv` mtime via `should_process_task()`. Written last after all artifacts are complete.
- *Visualization:* `rf_visualization_summary.json` records `{projection_method, disjoint_mask_distance_mm, extraction_summary_mtime}`. Re-renders if any field differs from current params or `force=True`.

**Knowledge base constraints applied:**
- All coordinates remain in mm (per somatosensory metric units note).
- Tangent-plane projection reused as-is from `tangent_plane_alignment.py` (per 3D-to-2D projection note).
- Extraction flow reads from aggregated CSVs (not MKVs), so kinect depth single-path constraint is not directly applicable but is respected upstream.

---

## Implementation Plan

### Phase 1: Forearm Vertices Cache
**Goal:** Eliminate open3d PLY parsing on repeated runs. Standalone improvement, independent of the flow split.
**Started:** 2026-04-27
**Completed:** 2026-04-27

**Tasks:**
- [x] 1.1 — Add `load_forearm_vertices(ply_path: Path) -> Optional[np.ndarray]` to `rf_data_loader.py`. Checks for `{stem}_vertices.npy` alongside the PLY; loads from PLY via open3d + saves `.npy` on cache miss; returns cached `.npy` on hit (mtime-gated).
- [x] 1.2 — Replace `open3d.io.read_point_cloud()` call in `rf_cluster_pipeline.py` (line 606-608) with `load_forearm_vertices()`.
- [x] 1.3 — Replace `open3d.io.read_point_cloud()` calls in `rf_cluster_visualizer.py` (2D branch line 249-254, 3D branch line 356-359) with `load_forearm_vertices()`. Also restructured 3D branch to `if/elif/else` pattern; scatter fallback uses `lightgrey` (PLY colors unavailable via .npy cache).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — new function
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — replace PLY load
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — replace PLY loads

**Dependencies:** None

### Phase 2: Intermediate Artifact I/O Module
**Goal:** Build save/load helpers for all intermediate artifacts, with fail-fast validation.
**Started:** 2026-04-27
**Completed:** 2026-04-27

**Tasks:**
- [x] 2.1 — Create `rf_extraction_io.py` with functions: `save_neuron_contacts()`, `load_neuron_contacts()`, `save_cluster_session_data()`, `load_cluster_session_data()`, `save_forearm_vertices()`, `save_neuron_touches()`, `load_neuron_touches()`, `save_neuron_cluster_touches()`, `load_neuron_cluster_touches()`, `save_extraction_summary()`, `load_extraction_summary()`, `save_visualization_summary()`, `load_visualization_summary()`. Also added `save_sessions_metadata()` / `load_sessions_metadata()` for PLY path lookup.
- [x] 2.2 — Each `load_*` function validates file existence and data shape; raises `ValueError` on corrupt/missing data (fail-fast).
- [x] 2.3 — Add unit tests in `code/tests/test_rf_extraction_io.py`: round-trip save/load for each artifact type, corrupt `.npy` raises ValueError, missing file raises ValueError. 26 tests, all passing.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — **NEW**
- `code/tests/test_rf_extraction_io.py` — **NEW**

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Extract `run_cluster_rf_extraction()`
**Goal:** Factor out extraction from `run_cluster_rf_mapping()` into a standalone function.
**Started:** 2026-04-27
**Completed:** 2026-04-27

**Tasks:**
- [x] 3.1 — Create `run_cluster_rf_extraction()` in `rf_cluster_pipeline.py` containing: pair iteration, idempotency check against `extraction_summary.json`, clustered CSV loading, session CSV loading + neuron contact parsing, per-cluster spike extraction loop, cluster description building. Saves intermediate artifacts via `rf_extraction_io`. Also extracted shared `_build_pairs()` helper.
- [x] 3.2 — Extraction function accepts no vis-params.
- [x] 3.3 — Write `extraction_summary.json` as the last operation (sentinel pattern: written last, checked first).
- [x] 3.4 — Idempotency: `should_process_task(input_paths=[clustered_csv], output_paths=[extraction_summary_json], force=force)`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — new function, refactored from existing code

**Dependencies:** Phase 2

### Phase 4: Extract `run_cluster_rf_visualization()`
**Goal:** Factor out visualization into a standalone function that reads only from intermediate artifacts.
**Started:** 2026-04-27
**Completed:** 2026-04-27

**Tasks:**
- [x] 4.1 — Create `run_cluster_rf_visualization()` in `rf_cluster_pipeline.py`. Discovers cluster folders on disk, loads all artifacts via `rf_extraction_io`, computes metrics, renders heatmaps. PLY paths recovered from `sessions_metadata.json`.
- [x] 4.2 — Visualization function accepts no data-path params: `(output_dir, cluster_groups, cluster_group_defs, projection_method, disjoint_mask_distance_mm, force)`.
- [x] 4.3 — Idempotency via `rf_visualization_summary.json`: records `{projection_method, disjoint_mask_distance_mm, extraction_summary_mtime}`. Reruns if any field differs or `force=True`.
- [x] 4.4 — Validates `extraction_summary.json` exists before starting; raises `ValueError` if missing.
- [x] 4.5 — `run_cluster_rf_mapping()` refactored to thin wrapper calling extraction then visualization.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — new function, wrapper refactor

**Dependencies:** Phase 3

### Phase 5: DAG + Workflow Integration
**Goal:** Register both flows as independent DAG tasks in the pipeline infrastructure.
**Started:** 2026-04-27
**Completed:** 2026-04-27

**Tasks:**
- [x] 5.1 — Added `extract_receptive_fields_clustered_flow()` Prefect flow to `analysis_workflow.py`.
- [x] 5.2 — Added `visualize_receptive_fields_clustered_flow()` Prefect flow to `analysis_workflow.py`. Includes camera angle assignment.
- [x] 5.3 — `map_receptive_fields_clustered_flow()` kept intact (disabled in DAG); backward-compatible wrapper remains.
- [x] 5.4 — Both new flows registered in `available_tasks` and `cluster_group_defs` forwarding logic.
- [x] 5.5 — `analyse_workflow_dag.yaml`: `map_receptive_fields_clustered` disabled; new `extract_receptive_fields_clustered` (depends on `touch_clustering`) and `visualize_receptive_fields_clustered` (depends on `extract_receptive_fields_clustered`) added.
- [x] 5.6 — Verify DAG resolution: `DagConfigHandler` correctly resolves the dependency chain `touch_clustering → extract → visualize`. Verified via `pipeline_config_manager.DagConfigHandler` — all three `can_run()` calls return `True` in sequence after each predecessor completes.

**Files Modified:**
- `code/scripts/analysis_workflow.py` — two new flows, updated registration
- `configs/analyse_workflow_dag.yaml` — split task config

**Dependencies:** Phase 4

### Phase 6: Validation
**Goal:** End-to-end correctness and performance verification.
**Started:** —
**Completed:** —

**Tasks:**
- [ ] 6.1 — Run extraction → verify intermediate artifacts match current pipeline's in-memory data
- [ ] 6.2 — Run visualization → verify metrics + PNGs identical to current pipeline output
- [ ] 6.3 — Change `projection_method` in DAG config → verify only visualization reruns (check logs for "skipping extraction")
- [ ] 6.4 — Touch clustered CSV (newer mtime) → verify extraction reruns, visualization follows
- [ ] 6.5 — Run with `force_processing: true` on extraction only → verify extraction reruns, visualization skips if params unchanged
- [ ] 6.6 — Corrupt a `.npy` file → verify fail-fast `ValueError` with diagnostic message
- [ ] 6.7 — Run legacy `run_cluster_rf_mapping()` wrapper → verify identical output to split flows
- [ ] 6.8 — Performance measurement: time a vis-param-only change run (target: extraction wall time → 0)

**Files Modified:** None (test-only phase)

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `test_rf_extraction_io.py` — Round-trip save/load for each artifact type (`.npy`, `.csv`, `.json`)
- [ ] `test_rf_extraction_io.py` — Corrupt `.npy` (truncated file) → `ValueError`
- [ ] `test_rf_extraction_io.py` — Missing file → `ValueError` with path in message
- [ ] `test_rf_extraction_io.py` — `.npy` with wrong shape → `ValueError`
- [ ] `test_rf_extraction_io.py` — Visualization summary param comparison (same params → up-to-date, different → stale)

### Integration Tests
- [ ] Extraction flow produces all expected files in the correct directory structure
- [ ] Visualization flow reads extraction artifacts and produces metrics + PNGs

### Manual Verification
- [ ] Run full pipeline on a multi-session dataset → compare output directory trees before/after split (diff)
- [ ] Change `projection_method`, rerun → confirm extraction logs show "up-to-date, skipping" while visualization logs show active rendering
- [ ] Inspect a rendered PNG → visually identical to pre-split output
- [ ] Run via the GUI launcher → both tasks appear and execute in correct order

### Edge Cases
- [ ] Cluster with zero spikes → `spike_counts.csv` is empty, visualization skips rendering, no crash
- [ ] Session missing from `input_items` but present in clustered CSV → extraction logs warning, skips session
- [ ] Single-session dataset → all artifacts created, visualization renders one session per cluster
- [ ] `force_processing=True` on visualization only → extraction untouched, visualization re-renders

---

## Documentation Plan

- [ ] Update `CLAUDE.md` Architecture Overview table to mention the extraction/visualization split
- [ ] Update inline docstrings for the three public functions (extraction, visualization, wrapper)
- [ ] No user guide needed — pipeline is launched via existing GUI/DAG infrastructure

---

## Rollback Plan

1. **Before deployment:**
   - The wrapper `run_cluster_rf_mapping()` preserves the old API — callers are unaffected.
   - Old DAG config (`map_receptive_fields_clustered`) still works via the wrapper.

2. **Rollback procedure:**
   - Revert the commits on the feature branch.
   - Restore original `map_receptive_fields_clustered` in `analyse_workflow_dag.yaml`.
   - Delete any `.npy` / intermediate artifacts in the output directory (they are not consumed by anything else).

3. **Data considerations:**
   - No migrations. Existing `spike_counts.csv`, `rf_metrics.json`, and PNG outputs remain in the same locations with the same format.
   - New intermediate artifacts (`.npy`, session CSVs, sentinels) are additive — they can be deleted without affecting existing outputs.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Extraction output contract breaks silently (visualization reads stale/mismatched artifacts) | Med | High | Visualization validates `extraction_summary.json` exists and checks artifact file existence at startup. Missing → `ValueError`. |
| `unique_touch_spike_count` pooling correctness when loaded from per-session CSVs | Med | High | Pooled `spike_counts.csv` is saved by extraction (already correctly aggregated). Per-session CSVs are used only for rendering — no re-pooling in visualization flow. |
| Disk space for new intermediate artifacts | Low | Low | Per-session `.npy` is ~100KB. Session spike CSVs are small. Total overhead negligible vs existing 200-DPI PNGs. |
| Interrupted extraction leaves partial artifacts | Med | Med | Sentinel written last. Visualization checks sentinel before reading. Missing sentinel → "extraction incomplete" error. |
| DAG resolver doesn't handle the new task dependency | Low | Med | Test with `DagConfigHandler` in Phase 5.6. Pattern is identical to existing `touch_clustering → map_receptive_fields_clustered`. |

---

## References

- Current implementation: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
- Active related plans:
  - `docs/development/plans/active/rf-cluster-visualization-improvements.md`
  - `docs/development/plans/active/rf-cluster-hull-from-contact-points.md`
  - `docs/development/plans/active/rf-cluster-per-neuron-projection-frame.md`
- Knowledge base notes:
  - `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
  - `docs/development/knowledge-base/note-somatosensory-metric-units.md`
