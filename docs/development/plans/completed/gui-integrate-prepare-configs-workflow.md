# Plan: Integrate Prepare-Configs Scripts into Pipeline GUI

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/gui-integrate-prepare-configs-workflow`

---

## Overview

The two scripts in `code/scripts/_2_primary_processing/_1_prepare_configs/` generate the
YAML config files that every downstream workflow consumes. They are currently CLI-only.
This plan integrates them into the pipeline GUI as a new "Setup" workflow so users can
run the config-generation step from the same interface they use for everything else.

## Problem Statement

`create_kinect_configs.py` and `create_forearm_configs.py` must be run manually from the
command line before the GUI can be used for any other workflow. Users must know these
scripts exist, understand their execution order, and navigate to the correct directory.
This is error-prone and undiscoverable — the GUI gives no indication that this
prerequisite step exists.

## Goals

### In Scope
1. Register a new "Prepare Configs" workflow in the GUI launcher
2. Create a DAG config with two tasks (`create_kinect_configs` -> `create_forearm_configs`)
3. Create a workflow script that wraps the existing core functions (no duplication)
4. Hide the session config selector panel for workflows that don't use session configs

### Out of Scope
- Modifying the original standalone scripts (they remain usable from CLI)
- Adding progress bars or per-video granularity to the GUI
- Auto-detection of whether configs are stale and need regeneration

## Success Criteria

- [ ] "Setup > Prepare Configs" button appears in the GUI workflow selector
- [ ] Clicking it loads the DAG with two tasks and hides the session config selector
- [ ] Switching to another workflow re-shows the session config selector
- [ ] Running the workflow generates kinect configs then forearm configs
- [ ] Disabling a task in the DAG correctly skips it
- [ ] Original standalone scripts still work unchanged

---

## Technical Design

### Approach

Create a thin workflow script that imports the existing core functions
(`generate_yaml_config` and `create_session_configs`) and wraps them as Prefect `@flow`
tasks, following the same pattern as `primary_workflow_kinect_auto.py`. The DAG config
omits `kinect_configs`/`forearm_configs` parameters since this workflow creates them
rather than consuming them. A new `has_session_configs()` method on `DagConfigModel`
lets the GUI detect this and hide the session selector.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Wrap existing functions in workflow script | Reuses code, consistent pattern | Slight coupling to internal functions | **Chosen** |
| Copy logic into workflow script | Self-contained | Code duplication, maintenance burden | Rejected |
| Omit `dag_config` in launcher.yaml (use existing "no-DAG" path) | No DAG changes needed | Loses task enable/disable and dependency gating | Rejected |

### Architecture Changes

- **New `DagConfigModel.has_session_configs()`** — returns `True` if the YAML parameters
  contain `kinect_configs` or `forearm_configs`. This is the detection mechanism for
  workflows that don't use session configs.
- **Conditional visibility in `LauncherWindow._load_workflow()`** — hides the session
  selector when the model reports no session configs, shows it otherwise.
- No changes to `TaskExecutor`, `DagConfigHandler`, `SessionConfigSelector`, or any
  other infrastructure class.

---

## Implementation Plan

### Phase 1: DAG Config and Workflow Script
**Goal:** Create the new workflow files

- [ ] Create `configs/prepare_configs_workflow_dag.yaml` with two tasks and no session config params
- [ ] Create `code/scripts/prepare_configs_workflow.py` wrapping existing functions via `TaskExecutor`

**Files Created:**
- `configs/prepare_configs_workflow_dag.yaml` — DAG config (two tasks, dependency chain)
- `code/scripts/prepare_configs_workflow.py` — Workflow entry point

**Dependencies:** None

### Phase 2: GUI Integration
**Goal:** Register the workflow and handle the no-session-configs UI case

- [ ] Add `has_session_configs()` method to `DagConfigModel`
- [ ] Update `LauncherWindow._load_workflow()` to conditionally hide session selector
- [ ] Add "Setup" category with "Prepare Configs" entry to `configs/launcher.yaml`

**Files Modified:**
- `code/src/utils/pipeline/dag_config_model.py` — add `has_session_configs()` (~4 lines)
- `code/src/utils/gui/dag_launcher/launcher_window.py` — conditional visibility (~4 lines in `_load_workflow`)
- `configs/launcher.yaml` — add new category and workflow entry

**Dependencies:** Phase 1

---

## Key Implementation Details

### `prepare_configs_workflow.py` structure

```
imports: argparse, Path, logging, prefect.flow
         utils.path_tools, DagConfigHandler, TaskExecutor
         _2_primary_processing._1_prepare_configs.create_kinect_configs.generate_yaml_config
         _2_primary_processing._1_prepare_configs.create_forearm_configs.create_session_configs
         primary_processing.KinectConfigFileHandler

@flow create_kinect_configs(project_data_root, project_code_root, *, force_processing):
    # Load project_config.yaml, find *_kinect.mkv files, call generate_yaml_config() per video

@flow create_forearm_configs(project_code_root, project_data_root, *, force_processing):
    # Call create_session_configs(source_dir, output_dir, database_path)

run_prepare_configs(dag_handler):
    # Resolve paths, iterate pipeline_stages with TaskExecutor (no per-session loop)

main():
    # argparse --dag-config, DagConfigHandler, run_prepare_configs
```

The workflow does NOT iterate over session configs — it runs once globally.

### `prepare_configs_workflow_dag.yaml` structure

```yaml
parameters:
  parallel_execution: false
  # No kinect_configs or forearm_configs — this workflow creates them

tasks:
  create_kinect_configs:
    enabled: true
    options:
      force_processing: false
    depends_on: []

  create_forearm_configs:
    enabled: true
    options:
      force_processing: false
    depends_on: [create_kinect_configs]
```

### `DagConfigModel.has_session_configs()`

```python
def has_session_configs(self) -> bool:
    params = self._data.get("parameters", {}) or {}
    return "kinect_configs" in params or "forearm_configs" in params
```

### `LauncherWindow._load_workflow()` change

After creating the model, before populating panels:
```python
has_sessions = self._model.has_session_configs()
self._kinect_selector.setVisible(has_sessions)
if has_sessions:
    self._kinect_selector.populate(self._model)
```

Qt's `QSplitter` automatically collapses space for hidden children.

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI — "Setup" category with "Prepare Configs" appears
- [ ] Select "Prepare Configs" — task panel shows 2 tasks, session selector hidden
- [ ] Select another workflow — session selector reappears with correct content
- [ ] Toggle back and forth between workflows — no UI glitches
- [ ] Run "Prepare Configs" with data root available — configs generated in `configs/kinect_configs/` and `configs/forearm_configs/`
- [ ] Disable `create_forearm_configs` task, run — only kinect configs generated
- [ ] Disable `create_kinect_configs` task — forearm task skipped due to unmet dependency

### Edge Cases
- [ ] Switch from Prepare Configs to another workflow while dirty — unsaved changes dialog works
- [ ] Save the DAG config — YAML written correctly without session config keys

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (no new user-facing concepts beyond the GUI button)
- [ ] The launcher.yaml comments already document how to add workflows

---

## Rollback Plan

All changes are additive:
1. Remove the "Setup" category from `configs/launcher.yaml`
2. Delete `configs/prepare_configs_workflow_dag.yaml`
3. Delete `code/scripts/prepare_configs_workflow.py`
4. Revert the 4-line additions to `dag_config_model.py` and `launcher_window.py`

No migrations, no breaking changes to existing workflows.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `generate_yaml_config` relies on `__file__` for project root resolution | Med | Med | Workflow script passes `project_code_root` explicitly (CWD from GUI) |
| `path_tools.get_project_data_root()` opens tkinter dialog in subprocess | Low | Low | Same behavior as all existing workflows — already tested |
| Session selector `populate()` called with model that has no config entries | Low | Med | Guard with `has_session_configs()` check — skip `populate()` entirely |

---

## References

- Existing scripts: `code/scripts/_2_primary_processing/_1_prepare_configs/`
- Pattern reference: `code/scripts/primary_workflow_kinect_auto.py`
- GUI entry point: `code/scripts/launch_pipeline_gui.py`
- Knowledge base: CuPy import order not applicable (no preprocessing imports)
