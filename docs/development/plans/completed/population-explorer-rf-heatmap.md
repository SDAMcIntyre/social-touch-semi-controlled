# Plan: RF Heatmap Mode in Touch Population Explorer

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements`

---

## Overview

**What:** A new heatmap mode in the Touch Population Explorer that uses pre-computed per-touch RF maps (from `map_single_touch_rf`) instead of the current raw contact-point approach.

**Why:** The existing heatmap modes (spike density, mean IFF, cumulative IFF) aggregate raw frame-level signal values at contact points — they are not true receptive fields. The `map_single_touch_rf` task already computes proper per-touch RF maps (mean neuron value per vertex across all frames of a touch), but the population explorer cannot use them yet.

**How:** Load the RF `.npz` alongside `PopulationData`, align touches via explicit key matching, add a conditional "RF Mean" heatmap mode that aggregates RF maps across selected touches using per-vertex mean.

## Problem Statement

- The Touch Population Explorer's heatmap modes compute from raw contact-point arrays (`cp_vertex_idx`, `cp_iff`, `cp_spike`). These show where contact happened and what the instantaneous signal was, but don't represent the per-touch RF map that accumulates neuron response across the full duration of each touch.
- The `map_single_touch_rf` pipeline task (Phase 1 complete) produces exactly these per-touch RF maps as `.npz` files, but only the `SingleTouchRFExplorer` can consume them — one touch at a time.
- Researchers need to see aggregated RF maps for filtered subsets of touches (e.g., "all taps with high velocity") directly in the population explorer's scatter + heatmap interface.

## Goals

### In Scope
1. New conditional heatmap mode ("RF Mean IFF" or "RF Mean Spike") in the Touch Population Explorer, visible only when RF `.npz` data exists for the session
2. Mean aggregation strategy: for each vertex, average the per-touch mean values across all selected touches that contacted it
3. Works in both rectangle filter mode (aggregate across filtered touches) and single-touch mode (show one touch's RF map)
4. Graceful handling when RF data is unavailable (mode hidden, no error)

### Out of Scope
- Additional aggregation strategies (sum, max) — mean is sufficient for now
- Cross-session RF comparison or aggregation
- Changes to the `map_single_touch_rf` pipeline task itself
- Changes to the `SingleTouchRFExplorer` viewer

## Success Criteria

- [ ] When RF `.npz` exists for a session, an extra heatmap mode appears in the dropdown (label includes neuron mode, e.g., "RF Mean IFF")
- [ ] Rectangle filter: selecting touches via scatter rectangle shows the mean RF map across selected touches — vertices contacted by more touches show the average, not the sum
- [ ] Single-touch mode: clicking a touch shows its pre-computed RF map (should match what `SingleTouchRFExplorer` shows for the same touch)
- [ ] When RF `.npz` does not exist, no RF mode appears and no error is raised
- [ ] Switching sessions correctly shows/hides the RF mode and falls back to spike_density if the new session lacks RF data
- [ ] Color scale uses a stable per-session maximum (not recomputed per filter update)

---

## Technical Design

### Approach

Load RF data as a separate companion object (`PopulationRFData`) alongside `PopulationData`. Align touches using explicit `(block_order_id, trial_id, single_touch_id)` key matching — this requires adding a `touch_triple_keys` field to `PopulationData`.

The RF heatmap computation uses `np.add.at` for vectorized per-vertex accumulation across selected touches, with sparse pairs pre-converted to numpy arrays at load time for performance during rectangle drag.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Separate `PopulationRFData` + key-based alignment | Decoupled from cache lifecycle, explicit alignment | Requires `touch_triple_keys` field + cache version bump | **Chosen** — correctness guarantee worth the one-time cache invalidation |
| Index-based alignment (assume same touch order) | No cache change needed | Fragile if row filtering diverges between series_augmented and prepared CSVs | Rejected — silent misalignment risk in research pipeline is unacceptable |
| Embed RF data inside `PopulationData` | Single object to pass around | Couples optional RF to required cache, bloats dataclass with nullable fields | Rejected — different lifecycle and optionality |

### Architecture Changes

No new files. Changes to existing modules:

```
code/src/analysis/receptive_field_mapping/
├── touch_population_data.py          # MODIFIED — add touch_triple_keys, PopulationRFData, loader
├── gui/
│   └── touch_population_explorer.py  # MODIFIED — RF heatmap mode, _compute_rf_heatmap()
├── rf_cluster_pipeline.py            # MODIFIED — launcher loads RF .npz per session
code/scripts/
├── analysis_workflow.py              # MODIFIED — forward neuron_mode for explore_touch_population
configs/
├── analyse_workflow_viewers_dag.yaml # MODIFIED — add neuron_mode option
```

Integration points:
- `PopulationRFData` reads the same `.npz` format produced by `rf_single_touch_pipeline.py`
- Touch alignment via `touch_triple_keys` matches against `touch_id_map` keys in the `.npz`
- Vertex indices from the RF `.npz` index directly into `PopulationData.forearm_vertices` (rotation changes coordinates, not indices)

### Knowledge Base Relevance

- `bug-rf-explorer-nearest-vertex-distance.md` — Not applicable: RF data uses raw vertex indices from `load_playback_data()` (no distance filtering). The population explorer applies its own 15mm filter to contact points, but RF data bypasses that entirely.
- `note-rf-feature-space-explorer-gui-components.md` — Reference for heatmap mode dropdown pattern. The Touch Population Explorer already follows this pattern.
- No other applicable notes.

---

## Implementation Plan

### Phase 1: Data model + RF loader
**Goal:** Add touch key alignment field to `PopulationData` and create `PopulationRFData` with its loader.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 1.1 — Add `touch_triple_keys: np.ndarray` field (T, 3) to `PopulationData` dataclass
- [x] 1.2 — Build `touch_triple_keys` from `touch_records` in `load_population_data()` (data already available at line ~451)
- [x] 1.3 — Add `touch_triple_keys` to `_save_population_cache` and `_load_population_cache` with shape validation
- [x] 1.4 — Bump `_CACHE_SCHEMA_VERSION` from 3 to 4
- [x] 1.5 — Add `PopulationRFData` dataclass with `rf_vertex_indices`, `rf_values`, `neuron_mode`, `session_max_value`
- [x] 1.6 — Implement `load_population_rf_data(npz_path, touch_triple_keys, n_vertices)`: read `.npz`, align touches by key, pre-convert sparse pairs to numpy arrays, compute `session_max_value`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_population_data.py` — add field, update cache, add dataclass + loader (~80 new lines)

**Dependencies:** None

### Phase 2: Viewer RF heatmap mode
**Goal:** Add conditional RF heatmap mode to the Touch Population Explorer with mean aggregation.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 2.1 — Update `__init__` to accept optional `rf_sessions: list[Optional[PopulationRFData]]` parameter
- [x] 2.2 — Add `_sync_heatmap_modes()` helper to conditionally add/remove the RF mode from the heatmap dropdown
- [x] 2.3 — Implement `_compute_rf_heatmap(touch_indices, n_verts)` using `np.add.at` for mean aggregation
- [x] 2.4 — Update `_heatmap_clim()` to return `(0.0, session_max_value)` for RF mode
- [x] 2.5 — Update `_render_3d()` to use `_compute_rf_heatmap(all indices)` when RF mode active
- [x] 2.6 — Update `_apply_filter_update()` to use `_compute_rf_heatmap(selected indices)` when RF mode active
- [x] 2.7 — Update `_apply_single_touch_display()` to use `_compute_rf_heatmap([idx])` when RF mode active
- [x] 2.8 — Update `_load_session()` to switch `self._rf_data` from `self._rf_sessions`, call `_sync_heatmap_modes()`, fall back to spike_density if RF unavailable

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — RF mode integration (~60 new lines)

**Dependencies:** Phase 1

### Phase 3: Pipeline integration
**Goal:** Wire RF data loading through the launcher and DAG config.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 3.1 — Update `launch_touch_population_explorer()` to accept `neuron_mode` parameter, resolve RF `.npz` per session, load `PopulationRFData` (or `None` if absent), pass `rf_sessions` to viewer
- [x] 3.2 — Update `explore_touch_population_flow()` to accept and forward `neuron_mode`
- [x] 3.3 — Add kwargs dispatch for `explore_touch_population` in `analysis_workflow.py` (forward `neuron_mode` from options)
- [x] 3.4 — Add `neuron_mode: iff` under `explore_touch_population.options` in `analyse_workflow_viewers_dag.yaml`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — update launcher (~20 new lines)
- `code/scripts/analysis_workflow.py` — forward neuron_mode (~5 lines)
- `configs/analyse_workflow_viewers_dag.yaml` — add option (~1 line)

**Dependencies:** Phase 1, Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `map_single_touch_rf` for a test session, then launch `explore_touch_population` — verify RF mode appears in dropdown
- [ ] Rectangle filter in RF mode: drag rectangle, observe heatmap updates with mean aggregation
- [ ] Single-touch mode in RF mode: click a touch, compare visually with `explore_single_touch_rf` for the same touch
- [ ] Switch between RF and non-RF heatmap modes — verify correct heatmap recomputation
- [ ] Delete RF `.npz` and relaunch — verify RF mode does not appear, no error

### Edge Cases
- [ ] Session with RF data followed by session without RF data — verify mode fallback when RF mode was active
- [ ] Session where all touches have empty RF maps (no contacted vertices) — verify `session_max_value` fallback to 1.0
- [ ] Mismatched touch count between PopulationData and RF `.npz` — verify fail-fast `ValueError`

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention RF heatmap mode in the Touch Population Explorer description

---

## Rollback Plan

1. Revert changes to all 5 files
2. Delete cached `.npz` sidecars (schema version 4 won't be recognized by old code — they'll auto-recompute with the reverted version 3 schema)
3. No data migrations — RF `.npz` files are unaffected (read-only consumer)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cache version bump invalidates all existing population caches | Certain | Low | One-time cost; recomputation is automatic and takes ~10s per session |
| Touch key alignment mismatch between PopulationData and RF `.npz` | Low | High | Explicit key-based alignment + fail-fast `ValueError` on mismatch |
| Performance of `_compute_rf_heatmap` during rectangle drag with many touches | Low | Med | Pre-converted numpy arrays + `np.add.at` vectorization; 30ms debounce timer already in place |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data model + RF loader | ~30 min | None |
| Phase 2: Viewer RF mode | ~30 min | Phase 1 |
| Phase 3: Pipeline integration | ~15 min | Phase 1, Phase 2 |

---

## References

- Single-touch RF maps plan: `docs/development/plans/active/single-touch-rf-maps.md`
- Single-touch RF explorer (inspiration): `code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py`
- RF pipeline task: `code/src/analysis/receptive_field_mapping/rf_single_touch_pipeline.py`
- Touch Population Explorer: `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py`
- Population data model: `code/src/analysis/receptive_field_mapping/touch_population_data.py`
