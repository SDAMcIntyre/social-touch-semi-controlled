# Plan: DAG Config Launcher GUI

**Date:** 2026-02-23
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/dag-config-launcher-gui`

---

## Overview

A PyQt5 GUI that acts as a unified launcher for every DAG workflow config in the
project. Instead of manually editing YAML files to switch kinect config directories,
toggle tasks, or change processing options, the user opens the GUI, makes selections
via button lists, checkboxes, and toggles, and the changes are written back to the
DAG YAML on disk before launching the pipeline.

The window is organized as a **three-column layout**:

| Left column | Middle column | Right column |
|-------------|---------------|--------------|
| Workflow picker — exclusive button list (one active at a time) | Kinect config directory & file selector — checkable tree | Task table — one column per attribute (name, enabled, force_processing, monitor, depends_on, …) |

Selecting a workflow button loads its DAG YAML and repopulates both the middle and
right columns.

## Problem Statement

The project has 8 DAG workflow YAML files that control the entire processing pipeline.
To change which sessions to process, which tasks to enable, or which options to flip,
the user must open a YAML file in a text editor, locate the right keys, edit values,
save, and then run the corresponding script. This is error-prone (typos, forgetting to
re-enable a task, pointing at the wrong kinect config directory) and tedious when
iterating across multiple session groups or toggling tasks for debugging. There is no
visual overview of task dependencies or a way to quickly see which tasks are enabled
across a workflow.

## Goals

### In Scope
1. Workflow picker — scan `configs/*.yaml` for all root-level DAG files and present
   them as an exclusive button list in the left column; clicking one populates the
   middle and right columns
2. Kinect config directory selector — list every subdirectory under
   `configs/kinect_configs/` as selectable items; support both the singular
   `kinect_configs_directory` parameter and the plural `kinect_configs_directories`
   list parameter
3. Per-file granularity — once a directory is selected, show its individual YAML
   config files as a nested checklist so the user can exclude specific recording
   blocks
4. Task table — render all tasks from the loaded DAG as a table with dedicated
   columns: **Task Name**, **Enabled**, **Force Processing**, **Monitor**,
   **Depends On**, plus additional columns for any other option fields; boolean
   options are rendered as checkboxes, non-boolean options as read-only text
5. Save — write the modified config back to the original YAML file on disk
6. Visual dependency overview — the task table's "Depends On" column shows each
   task's dependencies as a comma-separated list with clickable navigation; tasks
   are listed in YAML-defined order so the user can see execution order

### Out of Scope
- Pipeline execution from within the GUI (the GUI edits config; the user runs the
  script separately)
- Editing kinect config file contents (session metadata, paths) — those are
  machine-generated
- Creating new DAG files or tasks from the GUI
- Real-time pipeline monitoring (already handled by `PipelineMonitor`)
- Undo/redo history or version control of config changes
- DAG validation (cycle detection, missing dependency warnings) — future enhancement

## Success Criteria

- [ ] GUI launches, scans `configs/`, and lists all 8 DAG YAML files as buttons in
      the left column (only one selected at a time)
- [ ] Clicking a workflow button populates the kinect directory selector (middle
      column) and task table (right column)
- [ ] Kinect directory selector correctly handles both `kinect_configs_directory`
      (string) and `kinect_configs_directories` (list) parameter variants
- [ ] Per-file checklist appears when a kinect config directory is expanded, showing
      all `.yaml` files within it
- [ ] Task table checkbox columns for `enabled` and `force_processing` update the
      in-memory config
- [ ] Save button writes back a valid YAML file that `DagConfigHandler` can load
- [ ] Round-trip test: load → toggle a task → save → reload → verify toggle persisted
- [ ] YAML comments and ordering are preserved as much as possible on save

---

## Technical Design

### Approach

Build a standalone PyQt5 application with a clear model/view separation:

- **ConfigModel** — in-memory representation of a loaded DAG YAML, with methods to
  get/set parameters, toggle task options, and serialize back to YAML. Wraps
  `ruamel.yaml` for comment-preserving round-trip I/O.
- **LauncherWindow** (QMainWindow) — the GUI shell with a **three-column layout**:
  workflow button list (left column), kinect directory/file selector (middle column),
  task table (right column). Save button in the toolbar.
- **WorkflowSelector** — a `QButtonGroup` of exclusive `QPushButton`s (styled to
  highlight the active workflow). Clicking a button loads the corresponding DAG YAML
  and repopulates the middle and right columns.
- **TaskTableWidget** — a `QTableWidget` where each row is a task and each column
  maps to a task attribute: **Task Name** (read-only text), **Enabled** (checkbox),
  **Force Processing** (checkbox), **Monitor** (checkbox), **Depends On** (read-only
  comma-separated list), plus dynamically added columns for any non-standard options
  (rendered as read-only text for non-boolean types, checkbox for boolean types).

The GUI does **not** reuse `DagConfigHandler` for writing, because that class uses
`yaml.safe_load` (which discards comments and ordering). Instead, the GUI uses
`ruamel.yaml` with `RoundTripLoader`/`RoundTripDumper` for lossless round-trip
editing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PyQt5 standalone app | Already a project dependency, rich widget set, form-based layout natural fit | Heavier than tkinter | **Chosen** |
| Tkinter app | Stdlib, no extra dependency | Less polished, no tree-widget built-in, cumbersome for nested layouts | Rejected |
| Web-based (Flask + browser) | Modern UI | Overkill, adds web dependency, networking complexity | Rejected |
| Extend DagConfigHandler for writing | Reuses existing class | Uses `yaml.safe_load` which destroys comments; would require rewriting internals | Rejected — new model class with `ruamel.yaml` instead |

### Architecture Changes

```
code/src/utils/pipeline/
├── pipeline_config_manager.py          # unchanged (DagConfigHandler, read-only)
└── dag_config_model.py                 # NEW — round-trip YAML model for GUI

code/src/utils/gui/
└── dag_launcher/
    ├── __init__.py                     # NEW
    ├── launcher_window.py              # NEW — QMainWindow, three-column layout
    ├── workflow_selector.py            # NEW — exclusive button list (left column)
    ├── kinect_directory_selector.py    # NEW — QTreeWidget for dirs + files (middle column)
    └── task_table_widget.py            # NEW — QTableWidget with per-attribute columns (right column)

code/scripts/
└── launch_dag_config_gui.py           # NEW — entry-point script
```

**Existing files modified:**
- `pyproject.toml` — add `ruamel.yaml` dependency (if not already present)

**No changes to:**
- `DagConfigHandler` — the existing read-only handler remains untouched
- Any workflow script — they continue to read YAML via `DagConfigHandler`
- Any existing GUI module

### Key Design Decisions

**Comment-preserving YAML round-trip**: The DAG YAML files contain section headers,
stage annotations, and inline comments (e.g., `# STAGE 1: LED Signal Extraction`).
Using `ruamel.yaml` with round-trip mode preserves these on save. This is critical
for usability — the user should not lose their manual annotations.

**Per-file exclusion mechanism**: The idea mentions letting the user exclude specific
kinect config files within a directory. This is implemented as a new optional
parameter `exclude_files` (list of filenames) in the DAG YAML. The GUI writes this
parameter; the batch-processing loop in the workflow scripts must be patched to read
and honour it. This is the only change to existing workflow scripts.

```yaml
parameters:
  kinect_configs_directory: "kinect_configs/valid_configs_ST13-02"
  exclude_files:                         # NEW — GUI-managed
    - "kinect_config_2022-06-14_ST13-02_semicontrolled_block-order03.yaml"
```

**Singular vs. plural directory parameter**: The `analyse_workflow_dag.yaml` uses
`kinect_configs_directories` (a list), while most others use
`kinect_configs_directory` (a string). The GUI detects which key is present and
renders accordingly: a single-select radio group for singular, a multi-select
checklist for plural.

**Task table column layout**: The task table uses a `QTableWidget` with fixed columns
for the known attributes and dynamically appended columns for workflow-specific
options. The fixed columns are:

| Column | Widget | Editable |
|--------|--------|----------|
| Task Name | `QTableWidgetItem` (text) | Read-only |
| Enabled | `QCheckBox` (centred via `setCellWidget`) | Yes |
| Force Processing | `QCheckBox` | Yes |
| Monitor | `QCheckBox` | Yes |
| Depends On | `QTableWidgetItem` (comma-separated text) | Read-only |

If a task does not define a particular option (e.g., some tasks have no `monitor`),
the corresponding cell is left empty and disabled. Dynamic columns for non-standard
options (e.g., `crop_half_size_mm`) are appended after the fixed columns and rendered
as read-only text showing the current value, since the GUI is not designed to edit
arbitrary typed parameters. Boolean dynamic options get a checkbox.

**Dependency visualization**: The "Depends On" column shows a comma-separated list of
dependency task names. Clicking a dependency name in the cell selects and scrolls to
the referenced task row. A future enhancement could render a graphical DAG, but for
the initial version the table column is sufficient.

### ConfigModel API Sketch

```python
class DagConfigModel:
    """Round-trip YAML model for GUI editing."""

    def __init__(self, config_path: Path):
        """Load YAML with ruamel.yaml RoundTripLoader."""

    # --- Parameters ---
    def get_kinect_dir_mode(self) -> Literal["single", "multi"]:
        """Detect singular vs plural kinect directory parameter."""

    def get_kinect_directories(self) -> list[str]:
        """Return list of configured kinect config directories."""

    def set_kinect_directories(self, dirs: list[str]) -> None:
        """Update kinect directory parameter(s)."""

    def get_exclude_files(self) -> list[str]:
        """Return list of excluded kinect config filenames."""

    def set_exclude_files(self, filenames: list[str]) -> None:
        """Update the exclude_files parameter."""

    # --- Tasks ---
    def get_task_names(self) -> list[str]:
        """Return task names in YAML-defined order."""

    def is_task_enabled(self, task_name: str) -> bool: ...
    def set_task_enabled(self, task_name: str, enabled: bool) -> None: ...

    def get_task_option(self, task_name: str, option: str) -> Any: ...
    def set_task_option(self, task_name: str, option: str, value: Any) -> None: ...

    def get_task_dependencies(self, task_name: str) -> list[str]: ...

    # --- Persistence ---
    def save(self) -> None:
        """Write back to original file, preserving comments."""

    def save_as(self, path: Path) -> None:
        """Write to a different path."""
```

---

## Implementation Plan

### Phase 1: ConfigModel with Round-Trip YAML
**Goal:** A testable model layer that can load, modify, and save DAG YAML files
without losing comments or ordering.

**Tasks:**
- [ ] Task 1.1 — Add `ruamel.yaml` to `pyproject.toml` dependencies
- [ ] Task 1.2 — Create `dag_config_model.py` with `DagConfigModel` class:
      round-trip load, parameter getters/setters, task toggle methods, save
- [ ] Task 1.3 — Implement `get_kinect_dir_mode()` to detect singular vs. plural
      parameter key
- [ ] Task 1.4 — Implement `get/set_exclude_files()` for per-file exclusion
- [ ] Task 1.5 — Write unit tests for round-trip fidelity: load → modify → save →
      reload → assert comments preserved and values correct

**Files Modified:**
- `pyproject.toml` — add `ruamel.yaml` dependency
- `code/src/utils/pipeline/dag_config_model.py` — **NEW**

**Dependencies:** None

### Phase 2: GUI Shell and Workflow Selector
**Goal:** A launchable PyQt5 window with a three-column layout and a working
workflow button list that loads a DAG and repopulates the other columns.

**Tasks:**
- [ ] Task 2.1 — Create `launch_dag_config_gui.py` entry-point script with CuPy
      import guard (even though no preprocessing imports are needed now, follow
      convention for future-proofing)
- [ ] Task 2.2 — Create `launcher_window.py` with `LauncherWindow(QMainWindow)`:
      menu bar, three-column layout using `QSplitter` or `QHBoxLayout` (left:
      workflow buttons, middle: kinect directory tree, right: task table), save
      button in toolbar
- [ ] Task 2.3 — Create `workflow_selector.py` with `WorkflowSelector(QWidget)`:
      scan `configs/*_dag.yaml`, create one `QPushButton` per DAG file in a
      vertical `QButtonGroup` (exclusive — only one active at a time), style the
      active button distinctly, emit signal on selection change
- [ ] Task 2.4 — Wire workflow button click to `DagConfigModel` instantiation;
      on selection change, clear and repopulate the kinect directory selector
      (middle column) and task table (right column); verify all 8 DAG files load
      without error

**Files Modified:**
- `code/scripts/launch_dag_config_gui.py` — **NEW**
- `code/src/utils/gui/dag_launcher/__init__.py` — **NEW**
- `code/src/utils/gui/dag_launcher/launcher_window.py` — **NEW**
- `code/src/utils/gui/dag_launcher/workflow_selector.py` — **NEW**

**Dependencies:** Phase 1

### Phase 3: Kinect Directory and File Selector
**Goal:** Middle column showing kinect config directories and their files as a
checkable tree, repopulated whenever the active workflow changes.

**Tasks:**
- [ ] Task 3.1 — Create `kinect_directory_selector.py` with
      `KinectDirectorySelector(QWidget)` containing a `QTreeWidget`
- [ ] Task 3.2 — Implement directory scanning: list subdirs of
      `configs/kinect_configs/`, mark currently-selected dir(s) as checked
- [ ] Task 3.3 — Implement per-file expansion: on directory expand, list `.yaml`
      files within; check all by default, uncheck files listed in `exclude_files`
- [ ] Task 3.4 — Wire checkbox state changes to `DagConfigModel.set_kinect_directories()`
      and `set_exclude_files()`
- [ ] Task 3.5 — Handle singular vs. plural mode: single-select (radio-like
      exclusive checkboxes) for `kinect_configs_directory`, multi-select for
      `kinect_configs_directories`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/kinect_directory_selector.py` — **NEW**

**Dependencies:** Phase 2

### Phase 4: Task Table
**Goal:** Right column showing all tasks as a table with one column per attribute,
repopulated whenever the active workflow changes.

**Tasks:**
- [ ] Task 4.1 — Create `task_table_widget.py` with `TaskTableWidget(QWidget)`
      containing a `QTableWidget`. Fixed columns: **Task Name** (read-only text),
      **Enabled** (centred checkbox), **Force Processing** (centred checkbox),
      **Monitor** (centred checkbox), **Depends On** (read-only comma-separated
      text). Each row corresponds to one task in YAML-defined order.
- [ ] Task 4.2 — Implement dynamic column detection: scan all tasks in the loaded
      model for option keys beyond the three standard ones (`force_processing`,
      `monitor`, `enabled`). For each extra key found in any task, append a column.
      Boolean extras get a checkbox; non-boolean extras get read-only text.
- [ ] Task 4.3 — Populate the table from `DagConfigModel`: one row per task,
      cells filled from the model. Tasks that lack a particular option column have
      a disabled/empty cell.
- [ ] Task 4.4 — Wire checkbox `stateChanged` signals to
      `DagConfigModel.set_task_enabled()` and `set_task_option()`
- [ ] Task 4.5 — Implement dependency navigation: clicking a task name in the
      "Depends On" cell selects and scrolls to that task's row

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_table_widget.py` — **NEW**

**Dependencies:** Phase 2

### Phase 5: Save and Integration
**Goal:** End-to-end workflow: load, edit, save, verify.

**Tasks:**
- [ ] Task 5.1 — Wire save button to `DagConfigModel.save()`; show confirmation
      dialog on success
- [ ] Task 5.2 — Add "Save As" option to write to a different path (avoids
      overwriting the original during experimentation)
- [ ] Task 5.3 — Add visual dirty-state indicator (e.g., asterisk in title bar)
      when unsaved changes exist
- [ ] Task 5.4 — Add unsaved-changes prompt on window close or workflow switch
- [ ] Task 5.5 — Patch batch-processing loops in workflow scripts to honour the
      `exclude_files` parameter when loading kinect configs from a directory

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py` — save wiring, dirty state
- `code/scripts/preprocess_workflow_kinect_auto.py` — honour `exclude_files`
- `code/scripts/primary_workflow_kinect_auto.py` — honour `exclude_files`
- `code/scripts/postprocess_workflow_kinect_auto.py` — honour `exclude_files`
- `code/scripts/analyse_workflow.py` — honour `exclude_files`
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — honour `exclude_files`

**Dependencies:** Phases 3, 4

---

## Testing Plan

### Unit Tests
- [ ] `DagConfigModel` round-trip: load every DAG YAML, modify a task toggle,
      save to temp file, reload, assert toggle persisted and comments preserved
- [ ] `DagConfigModel` singular vs. plural kinect directory detection on
      `preprocess_workflow_kinect_auto_dag.yaml` (singular) and
      `analyse_workflow_dag.yaml` (plural)
- [ ] `DagConfigModel.set_exclude_files()` adds/removes the `exclude_files` key
      correctly
- [ ] `get_task_names()` returns tasks in YAML-defined order for all 8 configs

### Integration Tests
- [ ] Full GUI launch, click each of the 8 workflow buttons — no crashes, task table
      and kinect selector repopulate correctly for each
- [ ] Select a kinect directory → expand → uncheck a file → save → reload → file
      appears in `exclude_files`
- [ ] Toggle `enabled` checkbox in task table → save → run `DagConfigHandler` on
      saved file → `can_run()` reflects the change

### Manual Verification
- [ ] Visual inspection: task table matches the YAML file's task list
- [ ] Visual inspection: kinect directory tree shows correct subdirectories
- [ ] YAML diff after save: only the toggled values changed, comments and structure
      intact
- [ ] Workflow scripts skip files listed in `exclude_files` during batch processing

### Edge Cases
- [ ] DAG file with no `parameters` section → GUI shows empty directory selector
- [ ] DAG file with custom task options (e.g., `crop_half_size_mm`) → extra column
      appended to task table, rendered as read-only text (not checkbox)
- [ ] Empty kinect config directory (no YAML files inside) → shown but not expandable
- [ ] `exclude_files` references a file that no longer exists → ignored silently

---

## Documentation Plan

- [ ] Add inline docstrings to all new classes and public methods
- [ ] Add a usage section to the project README or `docs/guides/` explaining how to
      launch and use the GUI
- [ ] Document the `exclude_files` parameter in the DAG YAML template files under
      `configs/_dag_templates/`

---

## Rollback Plan

1. The GUI is entirely additive — no existing code is modified except the
   `exclude_files` integration in workflow scripts (Phase 5)
2. To revert the GUI: delete `code/src/utils/gui/dag_launcher/`,
   `code/scripts/launch_dag_config_gui.py`, and `dag_config_model.py`
3. To revert `exclude_files`: remove the parameter-reading patches from the
   workflow scripts; the parameter is ignored if absent, so leftover YAML keys
   cause no harm
4. No data migrations; YAML files remain valid regardless of whether the GUI exists

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ruamel.yaml` round-trip loses edge-case formatting (multi-line strings, anchors) | Low | Med | Test against all 8 real DAG files; fall back to `save_as` for safety |
| PyQt5 version incompatibility on different machines | Low | Med | Pin minimum version in `pyproject.toml`; PyQt5 is already used elsewhere in the project |
| `exclude_files` parameter ignored by scripts user forgets to update | Med | Low | Phase 5 patches all 5 workflow scripts; parameter is documented in YAML templates |
| Users accidentally overwrite a carefully hand-tuned YAML | Med | Med | Dirty-state indicator + confirmation dialog; "Save As" option for experimentation |
| Scope creep into pipeline execution or DAG validation | Med | Low | Explicitly out of scope; the GUI is a config editor only |

---

## References

- Idea: `docs/development/plans/ideas/dag-config-launcher-gui.md`
- DAG handler: `code/src/utils/pipeline/pipeline_config_manager.py`
- Workflow scripts: `code/scripts/preprocess_workflow_kinect_auto.py` et al.
- DAG configs: `configs/*_dag.yaml` (8 files)
- Existing PyQt5 GUI: `code/src/preprocessing/motion_analysis/hand_tracking/gui/hand_mask_selector_gui.py`
- Knowledge base constraint: `docs/development/knowledge-base/note-cupy-import-order.md` — CuPy import guard in entry-point script
