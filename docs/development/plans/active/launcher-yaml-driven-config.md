# Plan: Launcher YAML-Driven Script Configuration

**Date:** 2026-03-09
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/launcher-yaml-driven-config`

---

## Overview

Replace the naming-convention-based script discovery in the DAG Config Launcher GUI
with a single `configs/launcher.yaml` file that explicitly declares all available
scripts, their categories, and their locked variables (e.g., DAG config path). Scripts
receive their DAG config path via `--dag-config` CLI argument instead of hardcoding it.

## Problem Statement

The current GUI launcher uses a fragile naming convention to associate DAG YAML files
with Python scripts: strip `_dag` from the YAML filename, apply `_SCRIPT_OVERRIDES`
for exceptions, and look for a matching script in `code/scripts/`. Each script also
hardcodes its own DAG config path in a `setup_environment()` function.

This creates three problems:
1. **Adding a new script requires editing Python code** (`_SCRIPT_OVERRIDES`,
   `_ORDERED_STEMS`) instead of just editing a config file
2. **Scripts are tightly coupled** to a specific DAG config path — running a script
   with a different config requires code edits
3. **Category grouping is auto-detected** from filename prefixes, giving no control
   over display order or naming

## Goals

### In Scope
1. A `configs/launcher.yaml` that is the single source of truth for the GUI: defines
   categories, script paths, display names, and locked variables (dag_config)
2. GUI reads the launcher YAML at startup — no more filesystem scanning or naming
   conventions
3. All 9 scripts accept `--dag-config` as a required CLI argument — no fallback to
   hardcoded paths
4. GUI passes `--dag-config` when launching scripts via subprocess

### Out of Scope
- Changes to the DAG YAML files themselves (task definitions, dependencies, parameters)
- Changes to `DagConfigModel` or `DagConfigHandler`
- Changes to the task panel, kinect directory selector, or save/dirty-state logic
- Support for arbitrary extra CLI arguments (future enhancement)
- Pipeline execution changes (monitoring, parallel execution, Prefect flows)

## Success Criteria

- [ ] `configs/launcher.yaml` exists and declares all 9 workflows across 5 categories
- [ ] GUI startup reads `launcher.yaml` and displays workflows grouped by category
      with explicit display names and ordering
- [ ] `_SCRIPT_OVERRIDES`, `_ORDERED_STEMS`, `_resolve_script()`, `_category_prefix()`
      are all removed
- [ ] Clicking a workflow button loads the correct DAG config into the task panel and
      kinect selector (same behavior as before)
- [ ] Clicking Run passes `--dag-config <path>` to the subprocess
- [ ] All 9 scripts work with `--dag-config` and fail clearly without it
- [ ] Adding a new script to the launcher requires only editing `launcher.yaml`

---

## Technical Design

### Approach

Introduce a `configs/launcher.yaml` that declares the full script registry with
explicit categories and script-to-config mappings. The GUI parses this at startup
into `WorkflowEntry` dataclass instances. The `WorkflowSelector` is driven by these
entries (not by filesystem scanning), and `LauncherWindow` uses the entry's script
and dag_config paths directly (no naming-convention resolution).

On the script side, each script's hardcoded DAG config path is replaced by an
`argparse` `--dag-config` required argument. The GUI's `_on_run()` builds the
subprocess command with `--dag-config`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Launcher YAML (explicit registry) | Full control, easy to extend, no naming magic | One more config file to maintain | **Chosen** |
| Enhanced naming convention (smarter auto-detection) | No new files | Still fragile, still requires `_SCRIPT_OVERRIDES` for exceptions | Rejected |
| Script self-registration (each script declares metadata) | Decentralized | Requires importing/scanning all scripts at GUI startup, complex | Rejected |

### Architecture Changes

**New files:**
```
configs/launcher.yaml                                    # Script registry
code/src/utils/gui/dag_launcher/launcher_config.py       # WorkflowEntry + parser
```

**Modified files:**
```
code/scripts/launch_dag_config_gui.py                    # Parse launcher.yaml
code/src/utils/gui/dag_launcher/workflow_selector.py     # Driven by entries, not filesystem
code/src/utils/gui/dag_launcher/launcher_window.py       # Remove naming convention, pass --dag-config
code/scripts/preprocess_workflow_kinect_auto.py           # argparse --dag-config
code/scripts/primary_workflow_kinect_auto.py              # argparse --dag-config
code/scripts/merging_pipeline_neuron_to_kinect_auto.py    # argparse --dag-config
code/scripts/preprocess_pipeline_extract_forearm_manual.py# argparse --dag-config
code/scripts/analysis_workflow.py                         # argparse --dag-config
code/scripts/postprocess_workflow_kinect_auto.py          # argparse --dag-config
code/scripts/preprocess_workflow_kinect_manual.py         # argparse --dag-config
code/scripts/preprocess_workflow_kinect_visualisation.py  # argparse --dag-config
code/scripts/merging_view_neural_kinect.py                # argparse --dag-config
```

### Key Design Decisions

**`WorkflowEntry` dataclass** — Carries `name`, `script: Path`, `dag_config: Path | None`,
`category: str`. Emitted by `WorkflowSelector` signal (replaces the bare `Path` signal).
Lives in a new `launcher_config.py` module alongside the YAML parser function.

**`dag_config` is optional** — A script entry in `launcher.yaml` may omit `dag_config`
for scripts that don't use DAG configs. The GUI hides the task panel and kinect selector
in that case. Currently all 9 scripts use DAG configs, but this future-proofs the design.

**Startup validation** — `parse_launcher_config()` warns and skips entries whose `script`
or `dag_config` paths don't exist on disk. The GUI still launches with the valid entries.

**Knowledge base check** — No applicable notes found. The knowledge base covers Open3D
widget layout, CuPy import order, ICP registration, and somatosensory units — none
overlap with this refactoring.

---

## Implementation Plan

> **Parallelism notes for AI assistants:** Phases 1 and 3 are independent and can be
> implemented in parallel (e.g., via background subagents). Phase 2 depends on Phase 1.
> Within Phase 3, all 9 script modifications are independent and can be done in parallel.

### Phase 1: Foundation — Launcher Config
**Goal:** Create the launcher YAML and its parser, without changing existing behavior.

- [ ] 1.1 — Create `configs/launcher.yaml` with all 9 workflows in 5 categories
- [ ] 1.2 — Create `code/src/utils/gui/dag_launcher/launcher_config.py` with:
  - `WorkflowEntry` dataclass (`name: str`, `script: Path`, `dag_config: Path | None`,
    `category: str`)
  - `parse_launcher_config(yaml_path: Path, project_root: Path) -> list[WorkflowEntry]`
    — loads YAML, validates paths, returns entries (warns and skips invalid ones)

**Files:**
- `configs/launcher.yaml` — **CREATE**
- `code/src/utils/gui/dag_launcher/launcher_config.py` — **CREATE**

**Dependencies:** None

### Phase 2: GUI Refactor — Wire Launcher Config
**Goal:** The GUI reads `launcher.yaml` instead of scanning the filesystem. Run button
passes `--dag-config` as a CLI argument.

- [ ] 2.1 — Refactor `workflow_selector.py`:
  - Constructor accepts `list[WorkflowEntry]` instead of `configs_dir: Path`
  - Remove `_ORDERED_STEMS`, `_category_prefix()`, `_scan()`
  - New `_populate(entries)` groups by `entry.category`, creates buttons from
    `entry.name`
  - Signal changes to `pyqtSignal(object)` emitting `WorkflowEntry`
  - Rename `current_path()` → `current_entry()`,
    `select_path()` → `select_entry()`
- [ ] 2.2 — Refactor `launcher_window.py`:
  - Constructor accepts `entries: list[WorkflowEntry]` and `configs_dir: Path`
  - Remove `_SCRIPT_OVERRIDES` dict and `_resolve_script()` method
  - Store `self._current_entry: WorkflowEntry | None`
  - `_load_workflow(entry)`: if `entry.dag_config`, load `DagConfigModel`; else clear
    panels
  - `_update_run_bar()`: check `self._current_entry.script.exists()` directly
  - `_on_run()`: build `[sys.executable, str(entry.script), "--dag-config",
    str(entry.dag_config)]`
- [ ] 2.3 — Update `launch_dag_config_gui.py`:
  - Parse `configs/launcher.yaml` via `parse_launcher_config()`
  - Pass `entries` to `LauncherWindow`

**Files:**
- `code/src/utils/gui/dag_launcher/workflow_selector.py` — **MODIFY**
- `code/src/utils/gui/dag_launcher/launcher_window.py` — **MODIFY**
- `code/scripts/launch_dag_config_gui.py` — **MODIFY**

**Dependencies:** Phase 1

### Phase 3: Script CLI Arguments
**Goal:** All 9 scripts accept `--dag-config` as required arg. Hardcoded paths removed.

> **Parallelism note:** All 9 scripts are independent. Can be modified in parallel
> (e.g., 3 background subagents each handling 3 scripts).

For each script, the pattern is:
1. Add `import argparse` if not present
2. Add `parser = argparse.ArgumentParser()` + `parser.add_argument("--dag-config",
   type=Path, required=True)`
3. Remove the hardcoded `dag_config_path = ...` line
4. Wire `args.dag_config` into the existing flow

| Script | Hardcoded line | Entry point style |
|--------|---------------|-------------------|
| `preprocess_workflow_kinect_auto.py` | L639 `setup_environment()` | `def main()` |
| `primary_workflow_kinect_auto.py` | L162 `setup_environment()` | `def main()` |
| `merging_pipeline_neuron_to_kinect_auto.py` | L356 `setup_environment()` | `def main()` |
| `preprocess_pipeline_extract_forearm_manual.py` | L67 `setup_environment()` | `__main__` block |
| `analysis_workflow.py` | L297 `main()` | `def main()` |
| `postprocess_workflow_kinect_auto.py` | L294 `main()` | `def main()` |
| `preprocess_workflow_kinect_manual.py` | L401 `__main__` | `__main__` block |
| `preprocess_workflow_kinect_visualisation.py` | L238 `__main__` | `__main__` block |
| `merging_view_neural_kinect.py` | L198 `__main__` | `__main__` block |

**Files:** All 9 scripts listed above — **MODIFY**

**Dependencies:** None (independent of Phases 1–2, but tested together)

---

## Testing Plan

### Integration Tests
- [ ] GUI startup: `python code/scripts/launch_dag_config_gui.py` shows all 9
      workflows in 5 category groups with correct names and ordering
- [ ] Workflow selection: clicking each button loads the correct DAG config into the
      task panel and kinect selector
- [ ] Run: clicking Run for a workflow shows correct `--dag-config` argument in the
      subprocess command (verify via status bar or console output)

### Manual Verification
- [ ] Run at least 2 scripts via the GUI (one preprocessing, one analysis) and verify
      they execute correctly with the passed DAG config
- [ ] Run a script standalone from CLI: `python code/scripts/preprocess_workflow_kinect_auto.py --dag-config configs/preprocess_workflow_kinect_auto_dag.yaml`
- [ ] Run a script without `--dag-config` — verify it prints a clear argparse error
- [ ] Edit `launcher.yaml` to add a fake entry — verify the GUI warns about the
      missing script and still loads the valid entries

### Edge Cases
- [ ] `launcher.yaml` missing — GUI prints clear error and exits
- [ ] Script entry without `dag_config` — GUI shows Run button but hides task
      panel/kinect selector
- [ ] Invalid YAML syntax in `launcher.yaml` — GUI prints parse error and exits

---

## Documentation Plan

- [ ] Inline docstrings for `WorkflowEntry` and `parse_launcher_config()`
- [ ] Comment header in `configs/launcher.yaml` explaining the format and how to add
      new scripts

---

## Rollback Plan

1. Revert `launcher_window.py` and `workflow_selector.py` to restore `_SCRIPT_OVERRIDES`,
   `_resolve_script()`, `_ORDERED_STEMS`, and filesystem scanning
2. Revert `launch_dag_config_gui.py` to pass `configs_dir` instead of entries
3. Revert each script's `setup_environment()` / `__main__` to restore the hardcoded
   `dag_config_path`
4. Delete `configs/launcher.yaml` and `launcher_config.py`
5. No data changes — all DAG YAML files remain valid regardless

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scripts break when run standalone (no `--dag-config`) | Med | Med | Clear argparse error message; document in launcher.yaml header |
| Launcher YAML gets out of sync with actual scripts/configs | Low | Low | Startup validation warns about missing paths |
| Signal type change (`Path` → `object`) breaks other code | Low | Low | `workflow_changed` is only connected inside `LauncherWindow` |

---

## References

- Original GUI plan: `docs/development/plans/pending/dag-config-launcher-gui.md`
- DAG config model: `code/src/utils/pipeline/dag_config_model.py`
- Pipeline config handler: `code/src/utils/pipeline/pipeline_config_manager.py`
- Current launcher window: `code/src/utils/gui/dag_launcher/launcher_window.py`
- Current workflow selector: `code/src/utils/gui/dag_launcher/workflow_selector.py`
