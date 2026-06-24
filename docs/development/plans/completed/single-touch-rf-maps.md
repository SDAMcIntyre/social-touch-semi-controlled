# Plan: Single-Touch Receptive Field Maps

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-11 07:31
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements`

---

## Overview

**What:** A new processing task (`map_single_touch_rf`) that computes a receptive field map for each individual single touch in a session and saves all maps in a dedicated `.npz` file.

**Why:** The pipeline currently has RF mapping at the session level (simple pipeline) and cluster level (clustered pipeline), but nothing at the single-touch granularity. Per-touch RF maps are needed to study how the receptive field varies across individual touches within a session.

**How:** Read the prepared CSV (output of `touch_preparation`, already interpolated) via `load_playback_data()` from `touch_playback_data.py`, compute per-vertex mean neuron values (IFF or spike) using the same accumulation pattern as the Touch Playback Explorer, and save results as sparse vertex-value pairs in an `.npz` file.

## Problem Statement

- No pipeline task produces per-single-touch RF maps. The simple pipeline aggregates all spikes across an entire session; the clustered pipeline groups by cluster assignment.
- Researchers need to examine the RF map of each individual touch (identified by `block_order_id, trial_id, single_touch_id`) to study touch-by-touch variability.
- The Touch Playback Explorer GUI already computes this on-the-fly during scrubbing, but the values are ephemeral (not saved to disk) and only available interactively.

## Goals

### In Scope
1. New processing task that computes per-touch RF maps for all single touches in a session
2. Configurable neuron mode: IFF (default) or spike
3. Output saved as `.npz` with dictionary-based structure (touch key mapping + sparse vertex-value pairs)
4. Full DAG integration (config entry, Prefect flow, idempotency)

### Out of Scope
- Visualization or rendering of per-touch RF maps (downstream consumer responsibility)
- Tangent-plane rotation or distance filtering (this task uses raw coordinates, matching the playback explorer)
- Cross-session aggregation or comparison of per-touch RF maps

## Success Criteria

- [ ] Running `map_single_touch_rf` produces one `.npz` file per session under `4_analysed/single_touch_rf_maps/<session_id>/`
- [ ] The `.npz` contains `touch_id_map` (dict mapping touch key tuples to incremental IDs), `rf_data` (dict mapping IDs to vertex-value pair lists), and `neuron_mode`
- [ ] Values match the Touch Playback Explorer's heatmap when scrubbed to the end of a touch in the corresponding mode
- [ ] Task is idempotent (skips on re-run when outputs are up-to-date)
- [ ] Both `neuron_mode: iff` and `neuron_mode: spike` produce correct output

---

## Technical Design

### Approach

Read from the **prepared CSV** produced by `touch_preparation` (`4_analysed/preparation/<session>_prepared.csv`), which already has NaN gaps interpolated (cubic) and `block_order_id` / `gesture_type` assigned. This avoids working on raw data with NaN gaps.

The prepared CSV is passed to `load_playback_data()` from `touch_playback_data.py`, which handles:
- CSV reading with required columns (`contact_points`, `block_order_id`, `trial_id`, `single_touch_id`, `Nerve_spike`, `Nerve_freq`, `gesture_type`)
- Grouping by `(block_order_id, trial_id, single_touch_id)`
- Forward-filling `contact_points` within each touch group (30Hz to 1kHz alignment)
- Parsing contact point strings and snapping to forearm vertices via KDTree (raw coordinates, no distance threshold)
- `.npz` sidecar caching for fast reload

For each `TouchEvent`, accumulate per-vertex neuron values using the same `np.add.at` pattern as `touch_playback_explorer.py:511-528`:
```python
for fi in range(n_frames):
    verts = touch.frame_vertex_indices[fi]
    np.add.at(val_sum, verts, neuron_value[fi])
    np.add.at(contact_count, verts, 1.0)
mean_val = val_sum / contact_count  # only for contacted vertices
```

Then emit only vertices where `contact_count > 0` as `(vertex_idx, mean_value)` pairs.

**Note — vertex_idx vs (x, y, z):** The output uses integer vertex indices (from the forearm PLY mesh) rather than raw `(x, y, z)` coordinates. This is compact and avoids floating-point deduplication issues. However, it ties the output to a specific PLY mesh. If `(x, y, z)` proves more practical downstream (e.g. for cross-session comparison), switching is a low-cost change: `forearm_vertices[vertex_idx]` at save time.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse `load_playback_data()` | No duplication, consistent with playback explorer, has caching | Loads `frame_contact_pts` we don't need | **Chosen** — overhead is negligible, consistency benefit is high |
| Reuse `load_population_data()` | Vectorised `bincount`, already has `cp_touch_idx` | Applies tangent-plane rotation + 15mm filter, changes coordinate space | Rejected — user wants raw coordinates matching playback |
| Custom minimal loader | Only loads what's needed | ~400 lines of duplicated parsing logic | Rejected — maintenance cost outweighs minor perf gain |

### Architecture Changes

New file in existing module — no architectural changes:

```
code/src/analysis/receptive_field_mapping/
├── rf_single_touch_pipeline.py    # NEW — run_single_touch_rf_mapping()
├── touch_playback_data.py         # REUSED — load_playback_data()
├── __init__.py                    # MODIFIED — add export
└── ...
```

Integration points:
- `code/scripts/analysis_workflow.py` — new `@flow` + task registration
- `configs/analyse_workflow_processing_dag.yaml` — new task entry

### Output Format

Per session: `<output_dir>/<session_id>/single_touch_rf_maps.npz` (saved with `allow_pickle=True`).

| Key | Type | Description |
|-----|------|-------------|
| `touch_id_map` | `dict` | `{(block_order_id, trial_id, single_touch_id): int}` — maps each unique touch key to an incremental ID (0, 1, 2, ...) |
| `rf_data` | `dict` | `{int: list[(vertex_idx, value)]}` — maps each incremental ID to the list of (forearm vertex index, mean neuron value) pairs for that touch. Only vertices with at least one contact are included. |
| `neuron_mode` | `str` | `"iff"` or `"spike"` |

Usage:
```python
data = np.load(path, allow_pickle=True)
touch_id_map = data['touch_id_map'].item()
rf_data = data['rf_data'].item()
neuron_mode = str(data['neuron_mode'])

touch_idx = touch_id_map[(1, 3, 2)]
vertex_value_pairs = rf_data[touch_idx]  # [(v1, val1), (v2, val2), ...]
```

### Knowledge Base Relevance

- `bug-rf-explorer-nearest-vertex-distance.md` — 15mm distance filter issues. Not applicable: this task uses raw vertex snapping with no distance filter (matching playback).
- `note-3d-to-2d-surface-projection-algorithms.md` — Not applicable: no 2D projection, raw 3D vertex data only.
- No other applicable notes.

---

## Implementation Plan

### Phase 1: Pipeline function + DAG integration
**Goal:** New processing task that computes and saves per-touch RF maps.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 1.1 — Create `rf_single_touch_pipeline.py` with `run_single_touch_rf_mapping(input_items, output_dir, force, neuron_mode, preparation_dir)` function
- [x] 1.2 — Resolve prepared CSVs from `preparation_dir` (mapping session IDs to `<session>_prepared.csv`), pass to `load_playback_data()`
- [x] 1.3 — Implement per-touch RF computation: iterate `PlaybackData` touches, accumulate per-vertex with `np.add.at`, emit sparse `(vertex_idx, value)` pairs
- [x] 1.4 — Implement `.npz` output with `touch_id_map`, `rf_data`, `neuron_mode`
- [x] 1.5 — Use `should_process_task()` for idempotency (sentinel: `single_touch_rf_summary.json`)
- [x] 1.6 — Export `run_single_touch_rf_mapping` from `receptive_field_mapping/__init__.py`
- [x] 1.7 — Add Prefect `@flow` wrapper `map_single_touch_rf_flow` in `analysis_workflow.py`, accepting `preparation_dir`
- [x] 1.8 — Register in `available_tasks` list and forward `neuron_mode` + `preparation_dir` kwargs
- [x] 1.9 — Add `map_single_touch_rf` entry to `analyse_workflow_processing_dag.yaml` with `depends_on: [touch_preparation]`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_single_touch_pipeline.py` — **new file**, ~80 lines
- `code/src/analysis/receptive_field_mapping/__init__.py` — add import + `__all__` entry
- `code/scripts/analysis_workflow.py` — add flow function, register task, forward `neuron_mode` + `preparation_dir` kwargs
- `configs/analyse_workflow_processing_dag.yaml` — add task entry with `neuron_mode: iff`, `depends_on: [touch_preparation]`

**Dependencies:** `touch_preparation` (reads prepared CSV with interpolated data)

---

## Testing Plan

### Manual Verification
- [ ] Run the pipeline with `map_single_touch_rf` enabled — verify `.npz` output exists per session
- [ ] Load the `.npz` and verify structure: `touch_id_map` has all touches, `rf_data` has matching keys, each value is a list of `(vertex, value)` pairs
- [ ] Cross-check a touch's RF values against the Touch Playback Explorer (scrub to end of same touch in IFF mode, compare vertex values)
- [ ] Run with `neuron_mode: spike` — verify output uses spike values
- [ ] Run twice without `force_processing` — verify second run skips ("up-to-date")
- [ ] Run with `force_processing: true` — verify reprocessing occurs

### Edge Cases
- [ ] Session with no spikes — all touches should still have RF maps (IFF values at contacted vertices, just no spike-associated signal)
- [ ] Touch with empty contact points in all frames — should produce empty vertex-value list in `rf_data`

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `map_single_touch_rf` to the task list in the Orchestration section

---

## Rollback Plan

1. Delete `rf_single_touch_pipeline.py`
2. Revert changes to `__init__.py`, `analysis_workflow.py`, and the DAG config
3. No data migrations — output `.npz` files are standalone and can be deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Vertex indices tied to specific PLY mesh | Low | Med | Can switch to (x,y,z) tuples with a one-line change if needed downstream |
| `allow_pickle=True` portability concerns | Low | Low | Internal pipeline file, not shared externally; consistent with existing `.npz` usage in codebase |
| Large `.npz` for sessions with many touches | Low | Low | Sparse format (only contacted vertices) keeps size manageable |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~1 hour | None |

---

## References

- Touch Playback Explorer heatmap accumulation: `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py:511-528`
- Playback data model: `code/src/analysis/receptive_field_mapping/touch_playback_data.py`
- Simple RF pipeline (similar task structure): `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py`
