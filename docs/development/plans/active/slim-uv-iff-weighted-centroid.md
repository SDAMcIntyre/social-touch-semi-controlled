# Plan: Switch SLIM UV Centroid to IFF-Weighted via `map_single_touch_rf`

**Date:** 2026-05-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Started:** 2026-05-18
**Completed:** —
**Base Branch:** `feature/dag-graph-view`
**Branch:** `feature/slim-uv-iff-weighted-centroid`

---

## Overview

Switch `precompute_forearm_slim_uv` to depend on `map_single_touch_rf` (per-touch
IFF-weighted vertex maps in NPZ format) instead of `map_receptive_fields_simple`
(spike_positions.csv). This shortens the DAG chain and produces a semantically
better centroid — weighted by neural response magnitude rather than spike count.

## Problem Statement

`precompute_forearm_slim_uv` depends on `map_receptive_fields_simple` solely for
`spike_positions.csv`, creating a deep DAG chain:

```
touch_preparation → touch_series_transforms → set_rf_camera_settings
    → map_receptive_fields_simple → precompute_forearm_slim_uv
```

This forces SLIM UV precomputation to wait for camera settings and series
transforms, even though the centroid only needs spatial contact data + neural
response values — both available from `map_single_touch_rf`, which depends
only on `touch_preparation`.

Additionally, the current centroid is an unweighted mean of spike-contact
positions (every spike frame contributes equally). An IFF-weighted centroid
better reflects the neuroscientifically relevant hotspot.

## Goals

### In Scope
1. Replace `spike_positions_csv` parameter with `rf_maps_npz` in `precompute_forearm_slim_uv()`
2. Compute IFF-weighted 3D centroid from per-touch RF maps
3. Update DAG dependency from `map_receptive_fields_simple` to `map_single_touch_rf`
4. Update cache schema (`spike_csv_mtime` → `rf_npz_mtime`) with old-cache guard
5. Update all tests to use NPZ fixtures

### Out of Scope
- Changing `map_receptive_fields_simple` or its `spike_positions.csv` output
- Modifying downstream SLIM UV consumers (`rf_projection.py::project_slim`)
- Altering the SLIM parameterization algorithm itself
- Supporting dual centroid modes (spike-count vs IFF)

## Success Criteria

- [ ] `pytest code/tests/test_forearm_slim_uv.py` passes (all 5 tests)
- [ ] DAG resolves `precompute_forearm_slim_uv` after `map_single_touch_rf` (not `map_receptive_fields_simple`)
- [ ] Full pipeline run produces valid SLIM UV caches with IFF-weighted centroids
- [ ] QC figures show UV origin on neural hotspot
- [ ] Old-format caches raise explicit `RuntimeError` (not silent `KeyError`)

---

## Technical Design

### Approach

Load the single-touch RF maps NPZ, aggregate per-vertex mean IFF across all
touches, then compute `np.average(positions, weights=iff)` as the 3D centroid.
This mirrors the aggregation pattern already used in
`touch_population_explorer.py:654–684`.

The NPZ vertex indices reference the raw PLY point cloud. The cleaned mesh `V`
(after `clean_mesh`) has different vertex count/ordering. The current code works
in 3D coordinate space (CSV has x,y,z), so the new code does the same: resolve
NPZ indices → 3D positions via raw PLY vertices, compute centroid in 3D, then
KDTree-snap to cleaned mesh `V` for `center_vid`.

```
NPZ rf_data: { touch_id → [(vertex_idx, mean_iff), ...] }
                    ↓
    load_forearm_vertices(ply) → raw_verts (N, 3)
                    ↓
    aggregate per-vertex mean IFF across all touches
                    ↓
    np.average(raw_verts[contacted], weights=per_vertex_mean_iff)
                    ↓
    KDTree(V_cleaned).query(centroid_3d) → center_vid
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| IFF-weighted centroid from single-touch RF NPZ | Shorter DAG; weights by response magnitude; reuses existing aggregation pattern | Centroid shifts slightly vs current | **Chosen** |
| Keep spike_positions.csv, just change depends_on | No centroid shift | Still needs `map_receptive_fields_simple` to produce the CSV; can't decouple | Rejected |
| Spike-count centroid from single-touch RF NPZ | Closer to current behavior | NPZ pre-aggregates per-touch; loses spike-level granularity; IFF is more meaningful | Rejected |

### Architecture Changes

No new modules. Changes are confined to:
- `forearm_slim_uv.py` — signature, centroid computation, cache schema
- `analysis_workflow.py` — flow path resolution
- DAG YAML — dependency edge
- Tests — fixtures

### Knowledge-Base Constraints

- **Interior-pin foldover** (`note-mesh-parameterization-interior-pin-foldovers.md`):
  `center_vid` is applied post-hoc via `canonicalise_uv()`, not as a SLIM interior
  pin. This remains unchanged.
- **Non-manifold mesh repair** (`bug-slim-uv-non-manifold-flip.md`):
  `clean_mesh()` and `flatten_slim()` fallback chain is untouched — only the
  centroid input changes.
- **Coordinates in mm** (`note-somatosensory-units-and-calculations.md`):
  NPZ vertex positions inherit mm units from Kinect pipeline. No conversion.

---

## Implementation Plan

### Phase 1: Core logic + cache schema
**Goal:** Replace CSV-based centroid with IFF-weighted centroid from NPZ

- [x] Task 1.1 — Rename `SlimUvCache.spike_csv_mtime` → `rf_npz_mtime`
- [x] Task 1.2 — Change `precompute_forearm_slim_uv` parameter from `spike_positions_csv` to `rf_maps_npz`
- [x] Task 1.3 — Replace CSV-reading block with NPZ loading + per-vertex IFF aggregation + weighted centroid
- [x] Task 1.4 — Add `from .rf_data_loader import load_forearm_vertices` to resolve NPZ vertex indices to 3D
- [x] Task 1.5 — Remove `import pandas as pd` (no longer needed)
- [x] Task 1.6 — Update `load_slim_uv_cache`: rename kwarg, read `rf_npz_mtime`, update staleness check
- [x] Task 1.7 — Add old-cache guard: if NPZ has `spike_csv_mtime` but not `rf_npz_mtime`, raise `RuntimeError`
- [x] Task 1.8 — Update provenance write and NPZ save keys
- [x] Task 1.9 — Update docstrings and error messages

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/forearm_slim_uv.py` — signature, centroid logic, cache schema, staleness check

**Dependencies:** None

### Phase 2: DAG wiring
**Goal:** Route the pipeline through `map_single_touch_rf` instead of `map_receptive_fields_simple`

- [x] Task 2.1 — Update `precompute_forearm_slim_uv_flow` path resolution: `single_touch_rf_maps/<session>/single_touch_rf_maps.npz`
- [x] Task 2.2 — Update `should_process_task` input_paths and `_precompute()` call
- [x] Task 2.3 — Update flow docstring
- [x] Task 2.4 — Change DAG config `depends_on: [map_receptive_fields_simple]` → `depends_on: [map_single_touch_rf]`
- [x] Task 2.5 — Update DAG config comment block

**Files Modified:**
- `code/scripts/analysis_workflow.py` — flow path resolution, parameter passing, docstring
- `configs/analyse_workflow_processing_dag.yaml` — dependency + comment

**Dependencies:** Phase 1

### Phase 3: Tests
**Goal:** All existing tests pass with NPZ fixtures

- [x] Task 3.1 — Replace `_write_spike_csv` helper with `_write_rf_npz` that creates synthetic NPZ in `map_single_touch_rf` format
- [x] Task 3.2 — Update `test_precompute_writes_cache`: NPZ with one touch targeting `center_vid` with nonzero IFF
- [x] Task 3.3 — Rename + update `test_missing_spike_csv_raises` → `test_missing_rf_npz_raises`
- [x] Task 3.4 — Rename + update `test_empty_spike_csv_raises` → `test_empty_rf_data_raises` (empty `rf_data` dict)
- [x] Task 3.5 — Update `test_centre_on_boundary_raises`: NPZ targeting boundary vertices
- [x] Task 3.6 — Update `TestBarycentricUvLookup.slim_cache` fixture: `spike_csv_mtime=0.0` → `rf_npz_mtime=0.0`

**Files Modified:**
- `code/tests/test_forearm_slim_uv.py` — all test fixtures and assertions

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `test_precompute_writes_cache` — NPZ with known IFF at center vertex produces valid cache with correct `center_vid`
- [ ] `test_missing_rf_npz_raises` — nonexistent NPZ path raises `FileNotFoundError`
- [ ] `test_empty_rf_data_raises` — NPZ with no touches raises `ValueError`
- [ ] `test_centre_on_boundary_raises` — IFF-weighted centroid on boundary vertex raises `ValueError`
- [ ] `test_vertex_round_trip` — barycentric UV lookup still works with updated `SlimUvCache`

### Manual Verification
- [ ] Run full analysis DAG with `precompute_forearm_slim_uv` enabled
- [ ] Inspect QC figures — UV center sits on neural hotspot
- [ ] Verify old caches produce explicit `RuntimeError` (not silent failure)

### Edge Cases
- [ ] All-zero IFF (neuron not responding) — should raise `ValueError` about zero weights
- [ ] Single touch with single vertex — should still produce valid centroid
- [ ] NPZ vertex index exceeds raw PLY vertex count — out-of-bounds indices silently skipped (safety guard)

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/slim-forearm-projection.md` to reflect new data source

---

## Rollback Plan

1. Revert the commit (all changes are in a single atomic commit)
2. Old SLIM UV caches are invalidated anyway (`force_processing: true` in DAG config)
3. No data migrations — only cache files change, and they regenerate on next run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| IFF-weighted centroid shifts UV origin significantly | Low | Med | Visual QC figures allow immediate comparison; centroid only moves within the neural hotspot region |
| NPZ vertex indices don't match raw PLY indices | Low | High | Use `load_forearm_vertices()` (same loader as `map_single_touch_rf`); bounds-check indices |
| Old caches cause silent failures | Low | Med | Explicit `RuntimeError` guard for old-format caches |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core logic | ~30 min | None |
| Phase 2: DAG wiring | ~10 min | Phase 1 |
| Phase 3: Tests | ~20 min | Phase 1 |

---

## References

- Related Plan: `docs/development/plans/active/slim-forearm-projection.md`
- Knowledge Base: `note-mesh-parameterization-interior-pin-foldovers.md`, `bug-slim-uv-non-manifold-flip.md`
