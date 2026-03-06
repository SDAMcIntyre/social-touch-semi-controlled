# Plan: Per-Step Force-Processing Flags in Forearm Extraction Pipeline

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completion Date:** 2026-03-06
**Branch:** `feature/forearm-per-step-force-processing`

---

## Overview

The forearm extraction pipeline (`preprocess_pipeline_extract_forearm_manual.py`) has a single
`FORCE_PROCESSING` flag that only controls session-level skipping via `.SUCCESS` flags. The four
processing steps (extract, clean, normals, mesh) and the registration step always run
unconditionally. This plan adds per-step force flags so individual steps can be skipped when
their output already exists on disk, then moves those flags to a YAML DAG config (matching the
pattern used in `preprocess_workflow_kinect_auto.py`).

## Problem Statement

Re-running the forearm pipeline on a session that was already partially or fully processed forces
every step to re-execute — including the interactive segmentation GUI (step 1). This wastes time
and requires unnecessary user interaction. The other pipelines already support `force_processing`
per sub-flow via YAML-driven DAG configs; the forearm pipeline should follow the same convention.

## Goals

### In Scope
1. ~~Add per-step force flags (`FORCE_EXTRACT`, `FORCE_CLEAN`, `FORCE_NORMALS`, `FORCE_MESH`, `FORCE_REGISTRATION`)~~ **Done**
2. ~~Skip each step when its primary output file exists and the flag is `False`~~ **Done**
3. ~~Keep `FORCE_PROCESSING` for session-level `.SUCCESS` flag skipping (unchanged)~~ **Done**
4. ~~Move all `FORCE_*` constants and the forearm configs directory path into a YAML DAG config file~~ **Done**
5. ~~Load the config via `DagConfigHandler` (matching the `workflow_kinect_auto` pattern exactly)~~ **Done**
6. ~~Use `dag_handler.can_run()` / `mark_completed()` so `enabled: false` on a step propagates
   correctly to its dependents (e.g. disabling `clean_forearm` also blocks `define_normals`)~~ **Done**

### Out of Scope
- `PipelineMonitor` integration (no live dashboard for forearm — sessions are interactive)
- Parallel session execution (forearm pipeline is inherently sequential due to interactive GUI)
- Modifying sub-function internals (`extract_forearm`, `clean_forearm_pointcloud`, etc.)

## Success Criteria

- [x] Re-running a fully-processed session with all flags `False` skips all 5 steps with log messages
- [x] Setting a single flag to `True` re-runs only that step
- [x] Fresh sessions (no output) run all steps normally regardless of flag values
- [x] Interactive GUI (step 1) does not open when extract output already exists
- [x] All `FORCE_*` module-level constants are removed; per-step flags are read from YAML
- [x] `forearm_configs_directory` is read from YAML `parameters` (no hardcoded path in `__main__`)
- [x] Disabling a step in YAML (`enabled: false`) also skips all dependent downstream steps

---

## Technical Design

### Phase 1 Approach (Done)

Module-level constants with skip guards in `execute_frame_batch()` and `run_session()`. Each step
checks its primary output file before calling the sub-function. Already implemented in the script.

### Phase 2 Approach

Replace module-level constants with a YAML DAG config read via `DagConfigHandler`. The forearm
pipeline will gain the same configuration surface as `workflow_kinect_auto`:

- A `parameters` section for global settings (configs directory, session-level force flag)
- A `tasks` section with one entry per step: `enabled`, `options.force_processing`, `depends_on`

`TaskExecutor` is reused with `monitor=None` (no live dashboard) to get `can_run()` /
`mark_completed()` dependency propagation for free. One `DagConfigHandler` instance per frame
batch (via `.copy()`) so each batch has independent completion state, matching the per-block copy
pattern in `run_batch_processing()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Module-level constants with skip guards | Simple, minimal diff | Not configurable from outside; flags require code edit to change | Chosen for Phase 1, superseded by Phase 2 |
| `force_*` kwargs threaded through all functions | More testable | Adds parameter plumbing across 4 functions; no external config surface | Rejected |
| DAG config integration | Matches kinect_auto exactly; flags changeable in YAML without code edit | Requires threading `dag_handler` through call stack | **Chosen for Phase 2** |

### Architecture Changes

**New file:** `configs/preprocess_forearm_manual_dag.yaml`

**Modified file:** `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
- Add `setup_environment()` (mirrors kinect_auto's `setup_environment()`)
- Add `dag_handler` parameter to `batch_process_all_sessions()`, `_should_skip_session()`,
  `run_session()`, `_execute_all_batches()`, `execute_frame_batch()`
- Replace module-level `FORCE_*` reads with `dag_handler.get_task_options(step_name)['force_processing']`
- Replace hardcoded `configs/forearm_configs` with `dag_handler.get_parameter('forearm_configs_directory')`
- Replace `FORCE_PROCESSING` with `dag_handler.get_parameter('force_session_processing', True)`
- Remove all 6 module-level `FORCE_*` constants

---

## Implementation Plan

### Phase 1: Add flags and skip guards — **COMPLETE**

All tasks implemented. The per-step force flags and skip guards are present in the script.

### Phase 2: DAG Config Integration

**Goal:** Move FORCE_* constants to a YAML DAG config; load via `DagConfigHandler`

#### Task 1 — Create `configs/preprocess_forearm_manual_dag.yaml`

```yaml
# =============================================================================
# FOREARM EXTRACTION PIPELINE CONFIGURATION & TASK GRAPH (DAG)
# =============================================================================

parameters:
  forearm_configs_directory: "forearm_configs"
  force_session_processing: true  # Controls .SUCCESS flag skipping

tasks:
  # Step 1: Interactive forearm segmentation (ROI annotation + point cloud extraction)
  extract_forearm:
    enabled: true
    options:
      force_processing: false
    depends_on: []

  # Step 2: Point cloud noise removal
  clean_forearm:
    enabled: true
    options:
      force_processing: false
    depends_on: [extract_forearm]

  # Step 3: Surface normal estimation
  define_normals:
    enabled: true
    options:
      force_processing: false
    depends_on: [clean_forearm]

  # Step 4: Mesh reconstruction
  build_mesh:
    enabled: true
    options:
      force_processing: false
    depends_on: [define_normals]

  # Step 5: Multi-snapshot ICP registration
  register_forearms:
    enabled: true
    options:
      force_processing: false
    depends_on: [build_mesh]
```

#### Task 2 — Add `setup_environment()` to the script

Mirror the `setup_environment()` function from `preprocess_workflow_kinect_auto.py`:

```python
def setup_environment():
    project_root      = Path(__file__).resolve().parents[2]
    dag_config_path   = project_root / "configs" / "preprocess_forearm_manual_dag.yaml"
    project_data_root = path_tools.get_project_data_root()
    return project_root, project_data_root, dag_config_path
```

#### Task 3 — Update `__main__` to load `DagConfigHandler`

Replace the hardcoded directory path with a DAG-driven one:

```python
if __name__ == "__main__":
    project_root, project_data_root, dag_config_path = setup_environment()
    dag_handler = DagConfigHandler(dag_config_path)
    forearm_configs_dir = project_root / "configs" / dag_handler.get_parameter('forearm_configs_directory')
    batch_process_all_sessions(forearm_configs_dir, project_data_root, dag_handler)
```

#### Task 4 — Thread `dag_handler` through the call stack

Signature changes (add `dag_handler: DagConfigHandler` parameter):
- `batch_process_all_sessions(configs_forearm_dir, project_data_root, dag_handler)`
- `_should_skip_session(session_config, dag_handler)` — replace `FORCE_PROCESSING` with
  `not dag_handler.get_parameter('force_session_processing', True)`
- `run_session(session_config, project_data_root, dag_handler)`
- `_execute_all_batches(batches, dag_handler, interactive=True)`
- `execute_frame_batch(batch, dag_handler, interactive=True)`

#### Task 5 — Replace `FORCE_*` reads in `execute_frame_batch()` and `run_session()`

Use `TaskExecutor` with `monitor=None` per frame batch. One `dag_handler.copy()` is created per
batch (so completion state is independent across frames), matching the per-block copy pattern in
`run_batch_processing()`.

Step name → DAG task name mapping:

| Code step | DAG task name |
|-----------|--------------|
| `extract_forearm()` | `extract_forearm` |
| `clean_forearm_pointcloud()` | `clean_forearm` |
| `define_normals()` | `define_normals` |
| `define_forearm_mesh()` | `build_mesh` |
| `register_session_forearms()` | `register_forearms` |

Pattern for each step (replacing the `if not FORCE_X and batch.output.exists():` guards):

```python
# In execute_frame_batch(batch, dag_handler, interactive):
batch_dag = dag_handler.copy()

executor = TaskExecutor('extract_forearm', batch.depth_video.stem, batch_dag, monitor=None)
with executor:
    if not executor.can_run:
        pass  # disabled or dependency not met — skip silently
    else:
        opts = batch_dag.get_task_options('extract_forearm')
        force = opts.get('force_processing', False)
        if not force and batch.raw_ply.exists():
            print(f"  ⏭️  [1/4] Output already exists: {batch.raw_ply.name}. Skipping extract.")
        else:
            extract_forearm(...)

# ... same pattern for clean_forearm, define_normals, build_mesh
```

For registration in `run_session()`:
```python
executor = TaskExecutor('register_forearms', session_config.session_id, session_dag, monitor=None)
with executor:
    if not executor.can_run:
        pass
    else:
        opts = session_dag.get_task_options('register_forearms')
        force = opts.get('force_processing', False)
        if not force and unified_ply.exists() and transforms_json.exists():
            print(f"  ⏭️  Registration already exists. Skipping.")
        else:
            register_session_forearms(...)
```

#### Task 6 — Remove module-level `FORCE_*` constants

Delete the 6 module-level constants:
- `FORCE_PROCESSING`
- `FORCE_EXTRACT`
- `FORCE_CLEAN`
- `FORCE_NORMALS`
- `FORCE_MESH`
- `FORCE_REGISTRATION`

#### Task 7 — Add `DagConfigHandler` and `TaskExecutor` imports

```python
from utils import DagConfigHandler, TaskExecutor
```

**Files Modified:**
- `configs/preprocess_forearm_manual_dag.yaml` — new file
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — Phase 2 changes

**Dependencies:** Phase 1 must be complete (it is).

---

## Testing Plan

### Phase 1 Manual Verification (rerun to confirm no regression after Phase 2)

- [x] **Fresh run:** All flags `False`, no prior output → all 5 steps run, no skip messages
- [x] **Full re-run:** Same session, all flags `False` → all steps skipped with `⏭️` messages, no GUI opens
- [x] **Selective re-run:** Set `FORCE_MESH = True` only → steps 1-3 skipped, step 4 re-runs
- [x] **Force all:** All flags `True` → pipeline behaves identically to before this change

### Phase 2 Manual Verification

- [x] **YAML-driven skip:** Set `force_processing: false` in all YAML tasks, session already processed → all steps skipped (same as Phase 1 full re-run)
- [x] **YAML-driven force:** Set `force_processing: true` for `build_mesh` only in YAML → steps 1-3 skipped, step 4 re-runs, step 5 skipped
- [x] **Disabled step propagation:** Set `enabled: false` on `clean_forearm` → step 1 runs, steps 2-5 skipped due to unmet `depends_on`
- [x] **Directory from YAML:** Change `forearm_configs_directory` in YAML → script discovers sessions from the new directory
- [x] **Session-level force from YAML:** Set `force_session_processing: false` → sessions with `.SUCCESS` flag are skipped

### Edge Cases

| Scenario | Expected behavior |
|----------|------------------|
| Primary output exists but metadata missing | Step skipped (only primary file checked) |
| `force_session_processing: false` + `.SUCCESS` exists | Entire session skipped before reaching per-step logic |
| Upstream step skipped + its output missing | Downstream step fails with `FileNotFoundError` — clear, expected error |
| `enabled: false` on `extract_forearm` | All 4 downstream steps also skipped (deps never completed) |

---

## Documentation Plan

- [x] Delete idea file `docs/development/plans/ideas/forearm-pipeline-per-step-force-processing.md`
- [x] Move this plan to `docs/development/plans/active/` when branch opens

---

## Rollback Plan

1. Revert the commits — Phase 1 and Phase 2 changes are separable by commit
2. No migrations, no breaking changes, no data format changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Partial output detected as complete (PLY exists, metadata missing) | Low | Low | Only primary file checked; metadata is supplementary. User can set force flag in YAML to regenerate. |
| User edits YAML and forgets `enabled: false` propagates to dependents | Low | Medium | `TaskExecutor` logs skipped tasks with reason; behaviour is clearly documented here |
| Per-frame `dag_handler.copy()` overhead | Very Low | None | Deep copy of a small dict; negligible at the session scale |

---

## References

- Idea: `docs/development/plans/ideas/forearm-pipeline-per-step-force-processing.md`
- Reference pattern: `code/scripts/preprocess_workflow_kinect_auto.py` — `setup_environment()`, `run_batch_processing()`, `run_single_session_pipeline()`
- Reference DAG config: `configs/preprocess_workflow_kinect_auto_dag.yaml`
- `DagConfigHandler` and `TaskExecutor`: `code/src/utils/pipeline/pipeline_config_manager.py`, `code/src/utils/pipeline/task_executor.py`
- Registration outputs: `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py` (lines 98-99)
- `define_normals` force param: `code/scripts/_3_preprocessing/_3_forearm_extraction/define_normals.py` (line 14)
