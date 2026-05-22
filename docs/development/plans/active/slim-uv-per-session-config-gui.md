# Plan: Per-Session Configurable SLIM UV Pipeline with Interactive GUI

**Date:** 2026-05-22
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/slim-uv-configurable-mesh-method`
**Branch:** `feature/slim-uv-per-session-config-gui`

---

## Overview

Add a per-session interactive GUI for configuring the SLIM UV pipeline. Currently every session shares the same mesh method and cleaning settings from the global DAG config. Different forearm scans have different artifacts and need different cleaning strategies. The researcher will open a GUI per session, toggle settings, press "Process" to see immediate 3D/UV results, iterate, then "Accept" to save a per-session YAML config consumed by the batch pipeline.

## Problem Statement

The SLIM UV pipeline applies identical `mesh_method`, `clean_steps`, `n_iter`, and `max_edge_mm` to all sessions via the DAG config. In practice, sessions vary: some forearm scans have severe non-manifold artifacts requiring aggressive cleaning, others are clean and need minimal processing. The current workflow forces the researcher to either pick a single compromise config for all sessions, or manually edit the DAG YAML and re-run per session — slow, error-prone, and without visual feedback.

## Goals

### In Scope
1. Per-session SLIM UV config (mesh_method, max_edge_mm, clean_steps, n_iter, save_diagnostics) persisted as YAML in each session's output folder
2. Interactive PyQt5/PyVista GUI where the researcher configures, processes, and inspects mesh cleaning + SLIM UV results per session before accepting
3. New DAG task `configure_forearm_slim_uv` (category: `viewer_required`) as a prerequisite to `precompute_forearm_slim_uv`
4. `precompute_forearm_slim_uv` reads per-session configs when present, falls back to DAG defaults when absent
5. Config-hash-based staleness detection in the NPZ cache (replaces current `mesh_method`-only check)

### Out of Scope
- Exposing SLIM energy type (fixed to `SYMMETRIC_DIRICHLET` per knowledge base)
- Exposing `soft_p` or interior pin constraints (fixed to `soft_p=0`, boundary-only)
- Reordering mesh cleaning steps (fixed sequential pipeline)
- Manual mesh editing tools in the GUI
- GPU-accelerated SLIM solver

## Success Criteria

- [ ] Per-session `slim_uv_config.yaml` written to `4_analysed/forearm_slim_uv/<session_id>/` on "Accept"
- [ ] GUI shows 3D mesh + UV result after "Process", with step-by-step navigator
- [ ] `precompute_forearm_slim_uv` uses per-session config when present, DAG defaults when absent
- [ ] Changing any config parameter triggers automatic recompute (config hash staleness)
- [ ] Existing pipeline runs without per-session configs continue to work unchanged (backward compatible)
- [ ] Session combo shows green background for configured sessions (matching camera settings pattern)

---

## Technical Design

### Approach

Follow the established `set_rf_camera_settings` pattern: a separate interactive DAG task saves per-session config files, then the batch processing task reads them. The GUI embeds a PyVista 3D viewer with a config panel (dropdowns, checkboxes, spinboxes) and a step navigator. Processing runs in a `QThread` to keep the GUI responsive.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Interactive GUI prerequisite task (camera-settings pattern) | Proven pattern, clean separation, per-session persistence | Requires new DAG task | **Chosen** |
| Auto-generate YAML defaults, user edits manually | No GUI code needed | No visual feedback, error-prone, bad UX | Rejected |
| Pre-run config editor (single GUI for all sessions) | One window | Can't show per-session mesh results side-by-side | Rejected |
| Extend existing `SlimUvStepsViewer` with config controls | Reuse existing widget | Violates SRP, read-only viewer becomes editor, confusing hybrid | Rejected |

### Architecture Changes

```
New files:
  surface/slim_uv_config_io.py      Config dataclass, YAML I/O, hash
  gui/slim_uv_config_viewer.py      Interactive config GUI (QMainWindow)
  gui/slim_uv_process_worker.py     QThread worker for background processing

Modified files:
  surface/forearm_slim_uv.py        Extract run_slim_pipeline_core + build_slim_steps
  gui/__init__.py                   Export SlimUvConfigViewer
  surface/__init__.py               Export config I/O functions
  pipelines/rf_cluster_gui_launchers.py   Add launch_slim_uv_config_viewer
  __init__.py (rf mapping)          Export launcher
  analysis_workflow_processing.py   Add configure_forearm_slim_uv_flow, modify precompute flow
  analyse_workflow_processing_dag.yaml   Add configure_forearm_slim_uv task
```

### Knowledge Base Constraints

- **Mesh cleaning order is fixed** (`bug-slim-uv-non-manifold-flip.md`): steps 1-3 mandatory, steps 4-8 toggleable but always in order. GUI exposes on/off toggles only, no reordering.
- **No interior pins** (`note-mesh-parameterization-interior-pin-foldovers.md`): boundary-only SLIM, `soft_p=0`. Not exposed in GUI.
- **libigl nanobind API** (`note-igl-slim-api-version-mismatch.md`): use `igl.slim_precompute` / `igl.slim_solve` signatures.
- **GUI debounce** (`note-rf-feature-space-explorer-gui-components.md`): use `QTimer.singleShot` for expensive updates.
- **Deferred init** (`note-rf-feature-space-explorer-gui-components.md`): defer PyVista plotter creation to `showEvent`.

---

## Implementation Plan

### Phase 1: Config I/O Module
**Goal:** Read/write per-session SLIM UV config as round-trip YAML.

- [x] Create `SlimUvConfig` dataclass (session_id, mesh_method, max_edge_mm, clean_steps, n_iter, save_diagnostics, created_at, modified_at)
- [x] Implement `load_slim_uv_config(path)`, `save_slim_uv_config(path, config)` using ruamel.yaml round-trip mode
- [x] Implement `config_path_for_session(output_dir, session_id)` returning `output_dir / session_id / "slim_uv_config.yaml"`
- [x] Implement `config_hash(config)` — deterministic SHA-256 of parameter values (excludes timestamps)
- [x] Implement `make_default_config(session_id, **dag_overrides)` for building config from DAG defaults
- [x] Define `DEFAULT_CLEAN_STEPS` constant (all True)

**Files:**
- `code/src/analysis/receptive_field_mapping/surface/slim_uv_config_io.py` — new
- `code/src/analysis/receptive_field_mapping/surface/__init__.py` — export config I/O

**Dependencies:** None

### Phase 2: Extract Reusable Processing Core
**Goal:** Avoid duplicating mesh-build/clean/flatten logic between the batch pipeline and the GUI worker.

- [x] Extract steps 2-11 of `precompute_forearm_slim_uv` into `run_slim_pipeline_core()` returning a result dict (V_raw, F_raw, clean_diag, V_clean, F_clean, V_final, F_final, uv_final, centroid_3d, center_vid, bloop_pre, slim_diag, raw_mesh_colors, clean_mesh_colors)
- [x] Refactor `precompute_forearm_slim_uv` to call `run_slim_pipeline_core` then handle caching/QC/viewer (steps 12-19)
- [x] Extract step-list construction from `_launch_slim_steps_viewer` into `build_slim_steps()` returning `list[SlimStep]`
- [x] Make `_launch_slim_steps_viewer` a thin wrapper calling `build_slim_steps` + show viewer
- [x] Add `config_hash` field to `SlimUvCache` dataclass (default `""`)
- [x] Include `config_hash` in `np.savez` call (step 14)
- [x] Read `config_hash` in `load_slim_uv_cache` with backward-compat default

**Files:**
- `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py` — refactor

**Dependencies:** Phase 1

### Phase 3: Processing Worker Thread
**Goal:** Run mesh processing in a background thread so the GUI stays responsive.

- [x] Create `SlimUvProcessWorker(QThread)` with signals: `finished(object)`, `progress(str)`
- [x] Constructor takes: forearm_ply_path, rf_maps_npz, config (SlimUvConfig)
- [x] `run()` calls `run_slim_pipeline_core(collect_diagnostics=True)` then `build_slim_steps()`
- [x] Emits result dict on success, error string on failure
- [x] Only numpy arrays and SlimStep dataclasses cross the thread boundary (no VTK objects)

> Note: signal named `result_ready` instead of `finished` to avoid colliding with QThread's built-in `finished` signal.

**Files:**
- `code/src/analysis/receptive_field_mapping/gui/slim_uv_process_worker.py` — new

**Dependencies:** Phase 2

### Phase 4: Config Viewer GUI
**Goal:** Interactive per-session config + preview GUI.

- [x] Create `SlimUvConfigViewer(QMainWindow)` following `RFCameraSettingsViewer` lifecycle patterns
- [x] Layout: toolbar (session combo) + left config panel + right PyVista viewer + bottom step navigator
- [x] Session combo with green/default background per config presence
- [x] Config panel: mesh_method combo (bpa/delaunay), max_edge_mm spinbox (enabled for delaunay, 0.0=auto), 5 clean_steps checkboxes, n_iter spinbox (1-200)
- [x] "Process" button: disables controls, spawns `SlimUvProcessWorker`, shows progress in status label, populates step navigator on completion
- [x] Step navigator: list widget + info label, renders selected step in PyVista viewer (reuse `SlimUvStepsViewer._render_step` logic)
- [x] "Accept" button: saves `slim_uv_config.yaml` via `save_slim_uv_config()`, updates combo background
- [x] "Skip" button: advances to next unconfigured session or closes
- [x] Deferred init via `showEvent` + `QTimer.singleShot(0, _deferred_start)`
- [x] On session change: load existing config if present, else populate with DAG defaults; show raw point cloud

**Files:**
- `code/src/analysis/receptive_field_mapping/gui/slim_uv_config_viewer.py` — new
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — export `SlimUvConfigViewer`

**Dependencies:** Phase 3

### Phase 5: Launcher and Package Wiring
**Goal:** Register the viewer in the package and provide a launcher function.

- [x] Add `launch_slim_uv_config_viewer(input_items, dag_defaults)` to `rf_cluster_gui_launchers.py` following `launch_rf_camera_settings_viewer` pattern (resolve session paths, create QApplication, instantiate viewer, show, exec_)
- [x] Export `launch_slim_uv_config_viewer` from `receptive_field_mapping/__init__.py`

**Files:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — add launcher
- `code/src/analysis/receptive_field_mapping/__init__.py` — export launcher

**Dependencies:** Phase 4

### Phase 6: Pipeline Integration
**Goal:** Wire the GUI into the DAG and make `precompute_forearm_slim_uv_flow` use per-session configs.

- [x] Add `configure_forearm_slim_uv` task to DAG YAML (category: `viewer_required`, depends_on: `[map_single_touch_rf]`, options: DAG-level defaults for mesh_method, clean_steps, n_iter, max_edge_mm)
- [x] Update `precompute_forearm_slim_uv.depends_on` to include `configure_forearm_slim_uv`
- [x] Add `configure_forearm_slim_uv_flow` Prefect flow: check which sessions lack config YAML, launch GUI if any missing
- [x] Modify `precompute_forearm_slim_uv_flow` per-session loop: load `slim_uv_config.yaml` if present (override DAG params), else use DAG defaults; log which source is used
- [x] Replace `mesh_method`-only staleness check with full `config_hash` comparison against NPZ
- [x] Add `configure_forearm_slim_uv` to `_build_pipeline_stages` list before existing `precompute_forearm_slim_uv` entry

**Files:**
- `configs/analyse_workflow_processing_dag.yaml` — add task, update depends_on
- `code/scripts/analysis_workflow_processing.py` — add flow, modify precompute flow, add to stage list

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `slim_uv_config_io.py`: round-trip YAML read/write preserves all fields
- [ ] `slim_uv_config_io.py`: `config_hash` is deterministic (same config = same hash)
- [ ] `slim_uv_config_io.py`: `config_hash` changes when any parameter changes
- [ ] `slim_uv_config_io.py`: `make_default_config` applies DAG overrides correctly
- [ ] `slim_uv_config_io.py`: validation raises `ValueError` for invalid `mesh_method`

### Integration Tests
- [ ] `run_slim_pipeline_core` produces identical results to the original inline code in `precompute_forearm_slim_uv`
- [ ] `build_slim_steps` produces the same step list as the original `_launch_slim_steps_viewer` body

### Manual Verification
- [ ] Run `configure_forearm_slim_uv` on one session: verify controls, Process runs and shows steps, Accept writes YAML
- [ ] Run `precompute_forearm_slim_uv` on same session without force: verify it uses saved config (check log message)
- [ ] Change a clean_step in saved YAML, run precompute without force: verify recompute triggers
- [ ] Delete config YAML, run precompute with `interactive: false`: verify DAG defaults used, no error
- [ ] Configure two sessions with different settings, run full pipeline: verify each NPZ reflects its own config

### Edge Cases
- [ ] Session with no RF maps NPZ (dependency not met): `configure_forearm_slim_uv_flow` should still resolve paths; error surfaces at Process time with clear message
- [ ] User closes GUI without accepting any session: pipeline continues, precompute uses DAG defaults
- [ ] Rapid "Process" clicks before previous run completes: second click ignored while worker is running

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `configure_forearm_slim_uv` task to orchestration section, document per-session config pattern
- [ ] Update DAG config comments in `analyse_workflow_processing_dag.yaml`
- [ ] Inline docstrings for all new public functions and classes

---

## Rollback Plan

1. **Before deployment:**
   - All new code is additive (new files + new DAG task). Reverting = delete 3 new files, revert modifications to 7 existing files.
   - Per-session YAML configs are output artifacts, not source — deleting them restores DAG-default behavior.

2. **Data considerations:**
   - NPZ caches with `config_hash` field are backward-compatible: `load_slim_uv_cache` uses default `""` for old caches.
   - No migrations. No breaking changes to existing NPZ schema (additive field only).

3. **Rollback procedure:**
   - `git revert` the merge commit. Delete any `slim_uv_config.yaml` files from output folders. Re-run `precompute_forearm_slim_uv` with global DAG settings.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| QThread + PyVista rendering conflict | Med | High | Worker thread only produces numpy arrays; all VTK rendering on main thread. No VTK objects cross thread boundary. |
| Duplicated processing logic between worker and precompute | Med | Med | Extract `run_slim_pipeline_core` shared by both. Single source of truth. |
| Large meshes cause slow "Process" (>30s) | Med | Low | Status label shows progress. Worker emits step-by-step messages. User can close and re-open. Future: cache raw mesh across settings changes. |
| Config YAML schema changes in future | Low | Med | Hash excludes timestamps. Add version field if schema evolves. |
| Knowledge base: mesh cleaning must not be fully skipped | Low | High | Steps 1-3 always mandatory (hardcoded). GUI only toggles steps 4-8. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Config I/O | Small | None |
| Phase 2: Extract core | Medium | Phase 1 |
| Phase 3: Worker thread | Small | Phase 2 |
| Phase 4: Config viewer GUI | Large | Phase 3 |
| Phase 5: Package wiring | Small | Phase 4 |
| Phase 6: Pipeline integration | Medium | Phase 5 |

---

## References

- Precedent task: `set_rf_camera_settings` in `analysis_workflow_processing.py`
- Precedent GUI: `RFCameraSettingsViewer` in `gui/rf_camera_settings_viewer.py`
- Knowledge base: `bug-slim-uv-non-manifold-flip.md`, `note-mesh-parameterization-interior-pin-foldovers.md`, `note-igl-slim-api-version-mismatch.md`
- Existing step viewer: `gui/slim_uv_steps_viewer.py`
- Core pipeline: `surface/forearm_slim_uv.py`
