# Plan: Touch Population Explorer

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-11 07:31
**Base Branch:** `feature/touch-playback-raw-coordinates`
**Branch:** `feature/touch-population-explorer`

---

## Overview

**What:** A new "Single Touch Population Explorer" GUI viewer that plots single touches in a 2D feature space with a linked 3D forearm heatmap, enabling interactive exploration of how touch characteristics map to receptive field activation.
**Why:** The existing RF Feature Space Explorer operates per-frame (1kHz), conflating autocorrelated observations within a touch and limiting scatter axes to raw frame-level signals. Single touches are the natural independent unit and already carry richer extracted features.
**How:** A new data model groups frame-level contact points and IFF by single touch, and a new PyQt5+PyVista viewer (following the existing explorer's architecture) plots touches in configurable feature space with live heatmap filtering.

## Problem Statement

- The RF Feature Space Explorer's per-frame unit of analysis is statistically misleading: ~33 consecutive 1kHz frames within a single 30Hz contact event are nearly identical, inflating apparent sample size.
- Scatter axes are limited to the two continuous signals available per frame (velocity, pressure). Touch-level features (impulse, strain rate, stress, contact depth, mechanics-of-solids metrics) are inaccessible.
- Filtering by velocity/pressure bands is a proxy for touch characteristics. The user wants to filter by actual touch features and see the resulting receptive field.

## Goals

### In Scope
1. New `PopulationData` data model grouping frame-level contact points + IFF by single touch, with `.npz` sidecar caching
2. New `TouchPopulationExplorer` GUI with configurable scatter axes (dropdown from all touch-level features), draggable filter rectangle, gesture type checkboxes, and 3D forearm heatmap
3. Three heatmap modes: spike density, mean IFF per vertex, cumulative IFF per vertex
4. Optional enrichment from Stage 3 touch feature CSVs (additional scatter axes)
5. Pipeline integration: launch function, Prefect flow, DAG config entry in the viewers DAG

### Out of Scope
- Replacing the existing per-frame RF Feature Space Explorer (it remains as-is)
- Per-touch temporal waveform display (that's the Touch Playback Explorer's role)
- Clustering or statistical analysis within this viewer (the touch_analytics pipeline handles that)
- Multi-session overlay (each session viewed independently, switched via combo box)

## Success Criteria

- [ ] Scatter plot shows one dot per single touch (not per frame)
- [ ] X and Y axes switchable via dropdown from all available touch-level features
- [ ] Draggable rectangle + gesture checkboxes filter touch subpopulations
- [ ] 3D forearm heatmap updates in real time reflecting filtered touches' spatial RF
- [ ] Three heatmap modes (spike density, mean IFF, cumulative IFF) selectable via combo
- [ ] `.npz` cache warms after first load; subsequent loads are near-instant
- [ ] Launchable from the pipeline GUI via `explore_touch_population` DAG task

---

## Technical Design

### Approach

Follow the established viewer pattern: a data module (`touch_population_data.py`) loads and caches a pure-numpy data model, and a GUI module (`touch_population_explorer.py`) renders it. The pipeline entry point (`launch_touch_population_explorer`) lives in `rf_cluster_pipeline.py` alongside the existing explorer launchers.

The core data model insight: store IFF and spike values **per contact point** (since each contact point belongs to exactly one frame), not per frame. This collapses the frame-to-vertex indirection into a single flat array and makes heatmap computation fully vectorized (5 numpy ops, no Python loops).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Modify existing RF Feature Space Explorer to support touch-level mode | Less code duplication | Complicates existing viewer; two fundamentally different data granularities in one class | Rejected |
| New standalone viewer with shared `DraggableFilterRect` | Clean separation; each viewer has one unit of analysis; can evolve independently | Some code duplication in heatmap rendering | Chosen |
| Per-touch Python objects (list of TouchRecord dataclasses) | Conceptually clear | Slow filtering (Python loops); cache serialization overhead | Rejected in favour of flat numpy arrays |

### Architecture Changes

New modules:
```
code/src/analysis/receptive_field_mapping/
    touch_population_data.py        # NEW: PopulationData dataclass + loader + .npz cache
    gui/
        touch_population_explorer.py  # NEW: PyQt5+PyVista GUI
```

Key data model (all numpy, no per-touch Python objects):
```python
@dataclass
class PopulationData:
    # Session geometry
    forearm_vertices: np.ndarray       # (V, 3) rotated mesh
    tangent_rotation: np.ndarray       # (3, 3)

    # Per-touch arrays (length T = n_touches)
    gesture_types: np.ndarray          # (T,) object — gesture type strings
    spike_elicited: np.ndarray         # (T,) bool
    pressure_mean: np.ndarray          # (T,) float64
    velocity_amplitude_mean: np.ndarray# (T,) float64

    # Extra features from Stage 3 CSVs (optional)
    extra_feature_names: list[str]
    extra_feature_matrix: np.ndarray   # (T, F) float64

    # Per-contact-point flat arrays (all touches concatenated, length C)
    cp_vertex_idx: np.ndarray          # (C,) int64 — forearm vertex per contact pt
    cp_touch_idx: np.ndarray           # (C,) int64 — owning touch index
    cp_iff: np.ndarray                 # (C,) float64 — IFF value at that frame
    cp_spike: np.ndarray               # (C,) bool — spike at that frame
```

Heatmap aggregation at filter time (vectorized):
```python
cp_mask = np.isin(cp_touch_idx, np.where(touch_mask)[0])
active_verts = cp_vertex_idx[cp_mask]
contact_count = np.bincount(active_verts, minlength=n_verts)

if mode == "mean_iff":
    val = bincount(active_verts, weights=cp_iff[cp_mask]) / contact_count
elif mode == "cumulative_iff":
    val = bincount(active_verts, weights=cp_iff[cp_mask])
elif mode == "spike_density":
    val = bincount(active_verts, weights=cp_spike[cp_mask]) / contact_count
```

### Architecture Constraints (from knowledge base)

- **Debounce 3D heatmap updates** with 30ms `QTimer.singleShot()` during rectangle drag (note: `note-rf-feature-space-explorer-gui-components.md`)
- **Deferred PyVista init** in `showEvent()` before any render calls; release VTK resources in `closeEvent()`
- **15mm distance threshold** for KDTree contact-point-to-vertex snapping (established in `rf_explorer_data.py`)
- **All geometry in mm**, pressure in mm^-1, velocity in mm/s (note: `note-somatosensory-units-and-calculations.md`)
- **Contact points are pre-registered** (ICP transforms already applied upstream); no re-registration needed
- **Use `limit_area='inside'`** if any interpolation of contact columns is added (note: `bug-rf-explorer-nearest-vertex-distance.md`)

---

## Implementation Plan

### Phase 1: Data Model + Loader
**Goal:** Implement `PopulationData` and `load_population_data()` with `.npz` caching.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Define `PopulationData` dataclass with all-numpy arrays
- [x] Implement `load_population_data(series_csv_path, forearm_ply_path, touch_features_dir=None)`:
  - Load series CSV, group by `(block_order_id, trial_id, single_touch_id)`
  - Parse `contact_points` strings with 30Hz-to-1kHz deduplication (reuse pattern from `rf_explorer_data.py:228-301`)
  - Tangent plane alignment + KDTree vertex snapping with 15mm threshold (reuse `tangent_plane_alignment.py`, `rf_data_loader.py`)
  - Per-touch: compute `pressure_mean`, `velocity_amplitude_mean`, `spike_elicited` from frame data
  - Build flat `cp_*` arrays: for each touch's frames, record vertex indices + IFF + spike per contact point
- [x] Implement `.npz` sidecar cache (save/load) with mtime validation against CSV + PLY
- [x] Implement optional Stage 3 CSV merge: scan `touch_features_dir` for matching session CSVs, merge on touch key, populate `extra_feature_names` + `extra_feature_matrix`
- [x] Add `get_feature_array(name: str) -> np.ndarray` helper method for scatter axis data

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_population_data.py` — New file (~300 lines)

**Dependencies:** None

### Phase 2: GUI
**Goal:** Implement `TouchPopulationExplorer` with scatter + heatmap + interactive filtering.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Create `TouchPopulationExplorer(QMainWindow)` following `RFFeatureSpaceExplorer` layout:
  - Toolbar: session combo, heatmap mode combo, X-axis combo, Y-axis combo
  - Top: PyVista `QtInteractor` with forearm mesh + heatmap
  - Bottom: matplotlib scatter + `DraggableFilterRect` + gesture checkboxes + touch count label
- [x] Import and reuse `DraggableFilterRect` from `rf_feature_space_explorer.py`
- [x] Implement `_draw_scatter()`: one dot per touch, color by gesture type, alpha encodes `spike_elicited`, larger point size than frame-level explorer
- [x] Implement axis selector combos: populate from `["pressure_mean", "velocity_amplitude_mean"] + extra_feature_names`; on change → redraw scatter, reinitialize rectangle to 25th-75th percentile, update slider ranges
- [x] Implement `_apply_filter_update()`: touch-level boolean mask from rectangle bounds + gesture checkboxes → `np.isin` propagation to contact points → `np.bincount` heatmap aggregation → update PyVista scalars
- [x] Implement 3-mode heatmap toggle: "Spike density" / "Mean IFF (Hz)" / "Cumulative IFF (Hz)" with appropriate clim ranges
- [x] Deferred PyVista init in `showEvent()`, resource cleanup in `closeEvent()`
- [x] 30ms debounce via `QTimer` on rectangle drag/resize
- [x] Session switching via combo box with camera persistence

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — New file (~700 lines, based on `rf_feature_space_explorer.py` template)

**Dependencies:** Phase 1

### Phase 3: Pipeline Integration
**Goal:** Wire the viewer into the pipeline so it's launchable from the GUI.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Add `launch_touch_population_explorer(input_items)` to `rf_cluster_pipeline.py` — same pattern as `launch_feature_space_explorer` and `launch_touch_playback_explorer`: resolve session paths via `_resolve_explorer_session_paths()`, thread-pooled loading, QApplication launch
- [x] Export `TouchPopulationExplorer` from `gui/__init__.py`
- [x] Add `explore_touch_population_flow` to `analysis_workflow.py`
- [x] Add `explore_touch_population` task entry to `analyse_workflow_viewers_dag.yaml` with `depends_on: [touch_series_transforms]`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — Add launch function (~40 lines)
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — Add export (1 line)
- `code/scripts/analysis_workflow.py` — Add flow function + task tuple entry (~20 lines)
- `configs/analyse_workflow_viewers_dag.yaml` — Add task definition (~8 lines)

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `load_population_data()` with synthetic series CSV and forearm PLY: verify touch count, feature values, contact point counts, and flat array consistency
- [ ] `.npz` cache round-trip: save then load, verify all arrays match
- [ ] Heatmap aggregation: create PopulationData with known vertex assignments and IFF values, apply known touch mask, verify mean IFF / cumulative IFF / spike density produce expected vertex values
- [ ] `get_feature_array()`: verify built-in features and extra features return correct arrays; verify KeyError on unknown feature

### Integration Tests
- [ ] Load a real session's series-augmented CSV: verify PopulationData has plausible touch counts and feature ranges
- [ ] Load with Stage 3 enrichment: verify extra features populate correctly

### Manual Verification
- [ ] Launch viewer with real data; verify scatter shows touch-count dots (hundreds, not 100k frames)
- [ ] Drag rectangle — heatmap updates, touch count label reflects filtered count
- [ ] Switch X/Y axis combos — scatter redraws, rectangle reinitializes with appropriate bounds
- [ ] Toggle heatmap modes — visible difference between spike density, mean IFF, cumulative IFF
- [ ] Switch sessions — data reloads, scatter + heatmap update, camera persists per session
- [ ] Gesture checkboxes filter both scatter visibility and heatmap data
- [ ] Compare mean IFF heatmap with existing per-frame explorer on same data — spatial pattern should be qualitatively consistent

### Edge Cases
- [ ] Session with no spikes (all `spike_elicited=False`): heatmap should be all-grey, no crash
- [ ] Session with only one gesture type: checkboxes for absent types should be disabled or hidden
- [ ] Stage 3 CSVs not available: viewer launches with only built-in features (pressure, velocity), no crash
- [ ] Single touch in session: scatter shows one dot, rectangle collapses gracefully

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add Touch Population Explorer to the GUI bullet in the RF mapping section
- [ ] Add inline module docstring to `touch_population_data.py` and `touch_population_explorer.py`

---

## Rollback Plan

1. **Before deployment:** all changes are additive (new files + new exports). No existing files are structurally modified.
2. **Rollback procedure:** delete the two new files, revert the 4 modified files (single-line additions in each). The existing viewers and pipeline are unaffected.
3. **Data considerations:** `.npz` sidecar caches are inert; they can be left in place or deleted without impact.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stage 3 feature CSVs may not exist for all sessions | Medium | Low | Make Stage 3 enrichment optional; built-in features (pressure, velocity) always derived from frame data |
| `np.isin` slow for large cp arrays (>1M contact points) | Low | Medium | Profile and switch to `np.searchsorted` on sorted arrays if needed |
| Axis combo overwhelmed with too many features | Medium | Low | Start with curated subset of meaningful features; consider grouping or "All features" toggle later |
| DraggableFilterRect import coupling with existing explorer | Low | Low | Import the class directly; it has no dependencies on ExplorerData |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data model + loader | ~300 lines, 1 session | None |
| Phase 2: GUI | ~700 lines, 1-2 sessions | Phase 1 |
| Phase 3: Pipeline integration | ~70 lines, < 1 session | Phase 2 |

---

## References

- Existing viewer template: `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
- Data loader template: `code/src/analysis/receptive_field_mapping/rf_explorer_data.py`
- Knowledge base: `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
