# Plan: Single-Touch RF Explorer

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements`

---

## Overview

**What:** A PyQt5 viewer that loads pre-computed per-touch RF maps (`.npz` files from `map_single_touch_rf`) and displays them as heatmaps on a 3D forearm mesh, with cascading dropdowns for touch navigation.

**Why:** The `map_single_touch_rf` pipeline task produces per-touch RF data but there is no way to inspect it. Researchers need to examine individual touch RF maps to study touch-by-touch variability and validate pipeline output.

**How:** A single-3D-view `QMainWindow` (following the established explorer pattern) with Session/Block/Trial/Touch toolbar dropdowns. Selecting a touch populates the forearm mesh with that touch's sparse vertex-value heatmap.

## Problem Statement

- The `map_single_touch_rf` task (implemented in `rf_single_touch_pipeline.py`) computes per-touch RF maps and saves them as sparse `.npz` files, but no downstream consumer exists to visualize them.
- The Touch Playback Explorer shows ephemeral frame-by-frame heatmaps during animation, but cannot display the pre-computed per-touch aggregates.
- Without a dedicated viewer, researchers must write ad-hoc scripts to load and render the `.npz` data.

## Goals

### In Scope
1. PyQt5 viewer with a single PyVista 3D forearm view and toolbar controls
2. Cascading Session / Block / Trial / Touch dropdowns for navigation
3. Heatmap rendering of pre-computed per-touch RF data (sparse vertex-value pairs)
4. Full DAG + workflow integration (launcher, flow, config entry)

### Out of Scope
- Playback/animation of frame-by-frame data (that is what the Touch Playback Explorer does)
- Scatter plot or 2D views (no feature-space axes here)
- Dual 3D view (single view suffices for static pre-computed data)
- Modifications to `rf_single_touch_pipeline.py` (the data producer is complete)
- Gesture-type filtering or filter rectangles

## Success Criteria

- [ ] Viewer launches from the pipeline GUI via `explore_single_touch_rf` DAG task
- [ ] Session dropdown populates; Block/Trial/Touch cascade correctly
- [ ] Selecting a touch renders its RF heatmap on the forearm mesh (jet colormap, NaN=grey)
- [ ] Color limits are fixed per session for stable cross-touch comparison
- [ ] Empty touches (no contacted vertices) display all-grey mesh with "0 vertices" in info label
- [ ] Camera persists when switching touches within a session

---

## Technical Design

### Approach

Follow the established explorer GUI pattern: a `QMainWindow` with a `QToolBar` (cascading combos) and a `pyvistaqt.QtInteractor` for 3D rendering. Data is loaded from the `.npz` files via a lightweight dataclass + loader function defined in the new viewer file.

The viewer is deliberately simple — a single 3D view with no animation, no scatter, no filter rectangle. This matches the user's request: "like explore touch playback, just with the 3D view and the options above."

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single 3D view + toolbar | Simple, matches request, fast to navigate | No comparative view | **Chosen** |
| Dual 3D (current touch vs. session aggregate) | Side-by-side comparison | Heavier, not requested, session aggregate already in other viewers | Rejected |
| Extend Touch Playback Explorer with a "pre-computed" mode | Reuses existing code | Overloads an already complex viewer; different data source (npz vs. live computation) | Rejected |

### Architecture Changes

New file:
```
code/src/analysis/receptive_field_mapping/gui/
    single_touch_rf_explorer.py     # NEW — dataclass, loader, viewer class
```

Modified files:
```
code/src/analysis/receptive_field_mapping/gui/__init__.py        # export SingleTouchRFExplorer
code/src/analysis/receptive_field_mapping/__init__.py            # export launch_single_touch_rf_explorer
code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py # add launcher function
code/scripts/analysis_workflow.py                                 # add flow + register task
configs/analyse_workflow_viewers_dag.yaml                          # add DAG entry
```

### Data Model

```python
@dataclass
class SingleTouchRFViewerData:
    session_id: str
    forearm_vertices: np.ndarray            # (V, 3)
    forearm_vertex_colors: np.ndarray | None # (V, 3) uint8
    touch_id_map: dict                       # (block, trial, touch) -> int
    rf_data: dict                            # int -> list[(vertex_idx, mean_value)]
    neuron_mode: str                         # "iff" or "spike"
    block_order_ids: list[str]               # sorted unique
    trial_ids_by_block: dict[str, list[int]]
    touches_by_block_trial: dict[tuple, list[int]]  # -> sorted single_touch_ids
    session_max_value: float                 # max mean_value across all touches
```

### Data Source

Pipeline output: `<database_path>/4_analysed/single_touch_rf_maps/<session_id>/single_touch_rf_maps.npz`

NPZ contents (loaded with `allow_pickle=True`, dicts via `.item()`):
- `touch_id_map`: `dict[(block_order_id, trial_id, single_touch_id) -> int]`
- `rf_data`: `dict[int -> list[(vertex_idx, mean_value)]]`
- `neuron_mode`: `str`

Forearm PLY: resolved via existing `resolve_forearm_ply()`, vertices/colors loaded via `load_forearm_vertices()` / `load_forearm_vertex_colors()` from `rf_data_loader.py`.

### Reused Functions

- `rf_data_loader.py::resolve_forearm_ply()` — find forearm PLY for a session
- `rf_data_loader.py::load_forearm_vertices()` — load PLY with `.npy` sidecar cache
- `rf_data_loader.py::load_forearm_vertex_colors()` — load vertex colors from PLY
- `touch_analytics.pipeline_shared::session_id_from_path()` — extract session ID from CSV path
- `_resolve_explorer_session_paths()` — not reused (resolves series CSVs, not needed here)

---

## Implementation Plan

### Phase 1: Data model and loader
**Goal:** Define the data model and loading function for single-touch RF viewer data.

**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 1.1 — Define `SingleTouchRFViewerData` dataclass
- [x] 1.2 — Implement `load_single_touch_rf_data(npz_path, forearm_ply_path, session_id)`: load `.npz` with `allow_pickle=True`, normalize tuple keys to `(str, int, int)`, load forearm vertices/colors, reconstruct block/trial/touch hierarchy, compute `session_max_value`
- [x] 1.3 — Handle edge cases: empty RF data (all touches have no contacts), `session_max_value == 0` (use fallback `clim (0, 1)`)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py` — NEW: dataclass + loader

**Dependencies:** None

### Phase 2: Viewer GUI
**Goal:** Build the PyQt5 viewer class with toolbar and 3D forearm heatmap display.

**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 2.1 — Implement `SingleTouchRFExplorer(QMainWindow)` with `_build_ui()` / `_build_toolbar()`
- [x] 2.2 — Toolbar: Session, Block, Trial, Touch cascading `QComboBox` dropdowns with `blockSignals()` guards
- [x] 2.3 — Single `QtInteractor` 3D view: `pv.PolyData(vertices)` with `"heatmap"` scalar, `cmap="jet"`, `clim=(0, session_max_value)`, `nan_color=[0.3, 0.3, 0.3]`, black background
- [x] 2.4 — `_update_heatmap(single_touch_id)`: lookup sparse pairs from `rf_data`, build full scalar array (NaN for untouched), assign to `cloud["heatmap"]`, `cloud.Modified()`, render
- [x] 2.5 — Info `QLabel` in toolbar: "Touch {id} | {n} vertices | {neuron_mode}"
- [x] 2.6 — Deferred VTK init: `showEvent()` -> `QTimer.singleShot(0, _deferred_start)`
- [x] 2.7 — Per-session camera state persistence in `_camera_states` dict
- [x] 2.8 — `closeEvent()`: close plotter

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py` — viewer class

**Dependencies:** Phase 1

### Phase 3: Pipeline integration
**Goal:** Wire the viewer into the DAG pipeline so it can be launched from the GUI.

**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] 3.1 — Add `launch_single_touch_rf_explorer(input_items, neuron_mode)` to `rf_cluster_pipeline.py`: resolve npz + PLY paths per session, thread-pooled loading, QApplication + viewer launch
- [x] 3.2 — Export `SingleTouchRFExplorer` from `gui/__init__.py`
- [x] 3.3 — Export `launch_single_touch_rf_explorer` from `receptive_field_mapping/__init__.py`
- [x] 3.4 — Add `explore_single_touch_rf` entry to `analyse_workflow_viewers_dag.yaml` (category: viewer, enabled: false, options: neuron_mode: iff)
- [x] 3.5 — Add `explore_single_touch_rf_flow` Prefect flow to `analysis_workflow.py`
- [x] 3.6 — Register `("explore_single_touch_rf", explore_single_touch_rf_flow)` in `available_tasks` (viewer section)
- [x] 3.7 — Add `neuron_mode` option forwarding for the new task in the kwargs block

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — launcher function
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — export
- `code/src/analysis/receptive_field_mapping/__init__.py` — export
- `configs/analyse_workflow_viewers_dag.yaml` — DAG entry
- `code/scripts/analysis_workflow.py` — flow + task registration

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `load_single_touch_rf_data` returns correct hierarchy from mock `.npz`
- [ ] Tuple key normalization handles numpy string types
- [ ] `session_max_value` fallback to 1.0 when all RF data is empty

### Manual Verification
- [ ] Enable `explore_single_touch_rf` in viewers DAG, launch via pipeline GUI
- [ ] Session dropdown populates with all sessions that have `.npz` files
- [ ] Block -> Trial -> Touch cascade works correctly
- [ ] 3D forearm renders with heatmap for the selected touch
- [ ] Switching touches updates the heatmap instantly
- [ ] Camera position persists when switching touches
- [ ] Camera resets when switching sessions
- [ ] Touch with no contacted vertices shows all-grey mesh + "0 vertices" in info

### Edge Cases
- [ ] Session where all touches have empty RF data (no contacts)
- [ ] Session with only one block / one trial / one touch
- [ ] `.npz` file missing for a session (fail-fast with clear error)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add viewer to RF mapping GUI list
- [ ] Comment in `analyse_workflow_viewers_dag.yaml` describing the new task

---

## Rollback Plan

1. Revert the feature branch — no database migrations, no breaking changes
2. The `.npz` files produced by `map_single_touch_rf` are unaffected (read-only consumer)
3. No existing viewers or pipeline tasks are modified in behavior

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `np.load(allow_pickle=True)` fails on dict round-trip | Low | High | Tested in pipeline; `.item()` unwrap is standard numpy pattern |
| Tuple keys lose type after numpy pickle | Medium | Medium | Explicit normalization to `(str, int, int)` in loader |
| Large session (thousands of touches) slows combo population | Low | Low | Combo items are lightweight strings; heatmap update is O(V) |

---

## References

- Depends on: `docs/development/plans/active/single-touch-rf-maps.md` (data producer)
- Pattern reference: `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
- Data source: `rf_single_touch_pipeline.py::run_single_touch_rf_mapping()`
