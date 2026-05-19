# Plan: Separate RF Metrics from Visualization

**Date:** 2026-05-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Started:** 2026-05-19
**Base Branch:** `dev`
**Branch:** `feature/separate-rf-metrics-from-visualization`

---

## Overview

Extract RF metrics computation (centroid, hull area, Gaussian fit) from `run_cluster_rf_visualization()` into a standalone DAG task `compute_receptive_field_metrics`, and document the 3D-to-2D coordinate-space boundary in the analysis pipeline. The split decouples metrics re-computation (e.g. after changing projection method) from heatmap re-rendering, and the documentation makes the coordinate flow explicit for future development.

## Problem Statement

`run_cluster_rf_visualization()` currently performs two logically independent operations in a single function: (A) computing quantitative RF metrics and writing `rf_metrics.json` per cluster, and (B) rendering per-session heatmap PNGs. Rendering never reads from `rf_metrics.json` — they share only input artifacts (spike counts, forearm vertices, camera settings).

This entanglement means:
- Changing projection method forces re-rendering all heatmaps even when only metrics need updating
- Changing `disjoint_mask_distance_mm` (rendering-only) forces re-computing all metrics
- The function is 340+ lines and mixes two distinct concerns

Additionally, the 3D-to-2D coordinate boundary in the analysis pipeline is undocumented, making it hard to reason about which tasks operate in which coordinate space.

## Goals

### In Scope
1. Extract metrics computation into a new function `run_cluster_rf_metrics_computation()` with its own idempotency sentinel
2. Add a new `compute_receptive_field_metrics` DAG task and Prefect flow
3. Slim down `run_cluster_rf_visualization()` to pure rendering
4. Update `run_cluster_rf_mapping()` backward-compat wrapper to call all three steps
5. Document the 3D-to-2D coordinate-space boundary in `code/src/analysis/CLAUDE.md`
6. Create a knowledge-base note on coordinate spaces in the analysis pipeline

### Out of Scope
- Making projection methods pluggable (future work — the current dispatch via `project_to_2d()` is sufficient)
- Changing the `compute_rf_metrics()` algorithm itself
- Adding new projection methods
- Refactoring the rendering pipeline

## Success Criteria

- [ ] `run_cluster_rf_metrics_computation()` produces identical `rf_metrics.json` and `rf_metrics_summary.csv` as the current combined function
- [ ] `run_cluster_rf_visualization()` produces identical heatmap PNGs without computing metrics
- [ ] Both tasks skip when up-to-date (independent sentinels)
- [ ] Changing `projection_method` triggers only metrics re-computation, not re-rendering
- [ ] Changing `disjoint_mask_distance_mm` triggers only re-rendering, not metrics re-computation
- [ ] `run_cluster_rf_mapping()` wrapper still produces all artifacts (metrics + PNGs)
- [ ] DAG config parses and dependency graph is acyclic
- [ ] Coordinate-space documentation is present in `code/src/analysis/CLAUDE.md` and knowledge base

---

## Technical Design

### Approach

Split the cluster loop inside `run_cluster_rf_visualization()` at the natural seam: metrics computation (lines 1231-1263, 1392-1394) becomes a new function; the rendering loop (lines 1278-1390) stays. Both functions share the same iteration structure (`_build_pairs()`, gesture-type runs) and read from the same extraction artifacts, but write to different output files and use different idempotency sentinels.

The new function mirrors the existing `run_cluster_rf_extraction()` / `run_cluster_rf_visualization()` pattern: same parameter conventions, same sentinel discipline, same SLIM cache path resolution.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extract into separate function in same file | Matches existing pattern, shared helpers, minimal import changes | File grows longer | **Chosen** — consistent with extraction/visualization split |
| Extract into new file `rf_metrics_pipeline.py` | Cleaner separation | Redundant imports, splits closely-related orchestration code, `_build_pairs()` would need to be shared or duplicated | Rejected |
| Make visualization depend on metrics (sequential) | Simpler dependency graph | Creates false coupling — rendering doesn't need metrics | Rejected |

### Architecture Changes

**New dependency graph** (both tasks independent after extraction):
```
extract_receptive_fields_clustered ──┬── compute_receptive_field_metrics
                                     │      ↑ set_rf_camera_settings
                                     └── visualize_receptive_fields_clustered
                                            ↑ set_rf_camera_settings
```

**New function**: `run_cluster_rf_metrics_computation()` in `rf_cluster_pipeline.py`
**New sentinel**: `rf_metrics_computation_summary.json` (tracks projection_method, extraction mtime, camera settings mtime)
**New Prefect flow**: `compute_receptive_field_metrics_flow` in `analysis_workflow.py`
**New DAG task**: `compute_receptive_field_metrics` in `analyse_workflow_processing_dag.yaml`

### Knowledge-Base Constraints

From `note-3d-to-2d-surface-projection-algorithms.md`:
- Use **neuron-wide contact centroid** (not per-cluster) as projection origin to preserve hull enclosure
- All projections receive 3D points and output 2D `(u, v)` in mm

From `note-rf-cluster-visualization-overview.md`:
- Hull invariant must hold: `heatmap_points ⊆ cluster_contacts ⊆ neuron_contacts`
- Sentinel discipline: separate sentinels for extraction, metrics, visualization

From `note-somatosensory-units-and-calculations.md`:
- All spatial metrics inherit mm units from Kinect SDK — no conversion needed

---

## Implementation Plan

### Phase 1: Idempotency Infrastructure
**Goal:** Add sentinel save/load/check functions for the new metrics task.

- [x] Task 1.1 — Add `save_metrics_computation_summary()` to `rf_extraction_io.py`
- [x] Task 1.2 — Add `load_metrics_computation_summary()` to `rf_extraction_io.py`
- [x] Task 1.3 — Add `metrics_computation_is_up_to_date()` to `rf_extraction_io.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — Three new functions after the visualization summary section (line 306), mirroring `save_visualization_summary` / `visualization_is_up_to_date` but without `disjoint_mask_distance_mm`

**Dependencies:** None

### Phase 2: Extract Metrics Computation
**Goal:** Create `run_cluster_rf_metrics_computation()` and slim down `run_cluster_rf_visualization()`.

- [x] Task 2.1 — Create `run_cluster_rf_metrics_computation()` with signature matching plan
- [x] Task 2.2 — Remove metrics code from `run_cluster_rf_visualization()` (lines 1197, 1231-1263, 1392-1394)
- [x] Task 2.3 — Update `run_cluster_rf_mapping()` wrapper to call all three steps

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — New function (~100 lines), removal of ~40 lines from visualization, wrapper update

**Dependencies:** Phase 1

### Phase 3: Wire into Orchestration
**Goal:** Register the new task in the Prefect flow, DAG config, and package exports.

- [x] Task 3.1 — Add `run_cluster_rf_metrics_computation` to `__init__.py` imports and `__all__`
- [x] Task 3.2 — Add `compute_receptive_field_metrics_flow` Prefect flow to `analysis_workflow.py`
- [x] Task 3.3 — Register in `available_tasks` dict and kwargs-forwarding block
- [x] Task 3.4 — Add `compute_receptive_field_metrics` DAG task entry to YAML config
- [x] Task 3.5 — Update `visualize_receptive_fields_clustered` comment to "Step 2b"

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — Add import and `__all__` entry
- `code/scripts/analysis_workflow.py` — New flow, registration, kwargs forwarding
- `configs/analyse_workflow_processing_dag.yaml` — New task entry between extraction and visualization

**Dependencies:** Phase 2

### Phase 4: Documentation
**Goal:** Document the 3D-to-2D coordinate-space boundary.

- [x] Task 4.1 — Add "Coordinate spaces" section to `code/src/analysis/CLAUDE.md`
- [x] Task 4.2 — Create `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md`

**Files Modified:**
- `code/src/analysis/CLAUDE.md` — New section after "Receptive field mapping"
- `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md` — New file

**Dependencies:** None (can be done in parallel with other phases)

---

## Testing Plan

### Unit Tests
- [ ] Existing `test_rf_metrics.py` tests still pass (`compute_rf_metrics()` is unchanged)
- [ ] `test_dag_config_model.py` passes with new DAG task entry (acyclic graph, valid schema)

### Integration Tests
- [ ] Import check: `from analysis.receptive_field_mapping import run_cluster_rf_metrics_computation`
- [ ] Sentinel round-trip: `save_metrics_computation_summary()` → `load_metrics_computation_summary()` → `metrics_computation_is_up_to_date()` returns True

### Manual Verification
- [ ] Enable all three cluster RF tasks, run pipeline, verify `rf_metrics.json` + `rf_metrics_summary.csv` appear (from metrics task)
- [ ] Verify heatmap PNGs appear (from visualization task) without `rf_metrics.json` write
- [ ] Re-run with no changes — both tasks skip (idempotency)
- [ ] Change `projection_method` — only metrics task re-runs
- [ ] Change `disjoint_mask_distance_mm` — only visualization task re-runs

### Edge Cases
- [ ] Empty `spike_counts.csv` — metrics task writes `rf_metrics.json` with zero/NaN fields (existing behavior of `compute_rf_metrics()`)
- [ ] Missing camera settings — metrics task raises `ValueError` (fail-fast, existing behavior)
- [ ] `slim_uv_cache_dir` is None (non-SLIM projection) — no cache path passed, works as before

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` with coordinate-spaces section and updated task list
- [x] Create knowledge-base note `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md`
- [x] Update knowledge-base `README.md` index with new note

---

## Rollback Plan

1. **Before deployment:** All changes are on a feature branch. Revert = don't merge.
2. **Data considerations:** No data migration. Output files (`rf_metrics.json`, PNGs) are unchanged in format. The new sentinel file (`rf_metrics_computation_summary.json`) is additive — removing it just means the metrics task re-runs next time.
3. **Rollback procedure:** `git revert` the merge commit into dev. The old combined `run_cluster_rf_visualization()` still works. The backward-compat wrapper `run_cluster_rf_mapping()` is the only caller that chains all three steps — reverting restores the two-step chain.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Visualization sentinel still tracks `extraction_summary_mtime` but metrics are now separate | Low | Medium | Both tasks independently check `extraction_summary_mtime`. No coupling — metrics writes `rf_metrics.json`, viz writes PNGs |
| Gallery viewer or explorer reads `rf_metrics.json` directly | Low | Low | Verified: `rf_gallery_data.py` and `rf_explorer_data.py` do not read `rf_metrics.json` |
| DAG config YAML editing breaks comments/formatting | Low | Medium | Use `ruamel.yaml` round-trip mode per project convention |
| SLIM cache path resolution diverges between metrics and viz flows | Low | High | Both use identical pattern: `slim_uv_cache_dir / sid / f"{sid}_slim_uv.npz"`. Extract to shared helper if it diverges in future |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Idempotency | ~30 min | None |
| Phase 2: Extract metrics | ~1 hour | Phase 1 |
| Phase 3: Orchestration | ~30 min | Phase 2 |
| Phase 4: Documentation | ~30 min | None |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-cluster-visualization-overview.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Claude plan (working notes): `.claude/plans/investigate-the-analysis-pipeline-floofy-lollipop.md`
