# Building a GUI-Driven DAG Pipeline Launcher

## Architectural Reference for External Teams

This document describes the architecture of a GUI-driven pipeline launcher
that uses declarative YAML configuration to define, visualise, and execute
directed acyclic graphs (DAGs) of processing tasks. It is written as a
standalone reference — all domain-specific details have been replaced with
generic examples so that any team working with YAML-configured pipelines
can adapt the patterns to their own system.

### How to read this document

The architecture is composed of five layers that can be adopted
incrementally:

| Layer | What it provides | Adoption effort |
|-------|------------------|-----------------|
| 1. Launcher Config | Registry of available workflows | Hours |
| 2. DAG Config Format | Declarative task graphs in YAML | Hours |
| 3. Dual Model Classes | GUI editing + runtime execution models | 1-2 days |
| 4. GUI Shell | Interactive 3-column PyQt5 application | 1-2 weeks |
| 5. Execution Engine | Subprocess dispatch + context-manager runner | 1-2 days |

Layers 1-2 and 5 can be adopted independently of the GUI. Layer 3 bridges
GUI editing (Layer 4) with runtime execution (Layer 5).

### Prerequisites

- Python 3.10+
- PyQt5 (Layer 4 only)
- `ruamel.yaml` — round-trip YAML editing with comment preservation (Layer 3)
- `PyYAML` — fast YAML loading for runtime (Layer 5)
- `grandalf` — hierarchical graph layout (Layer 4 graph view, optional)
- A workflow orchestrator such as Prefect (optional; the core architecture works without one)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Layer 1 — The Launcher Config](#2-layer-1--the-launcher-config)
3. [Layer 2 — The DAG Config Format](#3-layer-2--the-dag-config-format)
4. [Layer 3 — The Dual Model Classes](#4-layer-3--the-dual-model-classes)
5. [Layer 4 — The GUI Shell](#5-layer-4--the-gui-shell)
6. [Layer 5 — The Execution Engine](#6-layer-5--the-execution-engine)
7. [Incremental Adoption Guide](#7-incremental-adoption-guide)
8. [Appendix A — Complete YAML Schema Reference](#appendix-a--complete-yaml-schema-reference)
9. [Appendix B — Class Diagram](#appendix-b--class-diagram)

---

## 1. System Overview

The system has two processes — a GUI application and a pipeline subprocess —
that communicate through a single shared artefact: a YAML file on disk.

```
                        GUI Process                          Pipeline Subprocess
                ┌─────────────────────────────┐     ┌────────────────────────────────┐
                │                             │     │                                │
launcher.yaml ──┤  LauncherWindow             │     │  DagConfigHandler               │
                │  ├── WorkflowSelector       │     │  ├── can_run(task)              │
                │  ├── TaskPanel (graph/table) │     │  ├── mark_completed(task)       │
                │  ├── SessionConfigSelector   │     │  └── completed_tasks: set       │
                │  └── ConsoleWidget           │     │                                │
                │        │                    │     │  TaskExecutor (context manager)  │
                │        │ DagConfigModel      │     │  ├── __enter__ → check deps    │
                │        │ (ruamel.yaml)       │     │  └── __exit__  → mark done     │
                │        │                    │     │                                │
                │        ▼                    │     │  Session loop:                  │
                │   *_dag.yaml ◄── save       │     │   for session in configs:       │
                │                             │     │     dag = template.copy()       │
                └──────────┬──────────────────┘     │     run_session(dag)            │
                           │                        │                                │
                           │  subprocess.Popen      └────────────────────────────────┘
                           │  [python, script,               ▲
                           │   --dag-config, path]           │
                           └─────────────────────────────────┘
                                  stdout piped to ConsoleWidget
```

**The fundamental insight:** the GUI and execution engine never share a
Python process. The GUI edits and saves YAML to disk; the subprocess reads
it fresh. YAML is the interface contract between the two sides.

This separation provides:

- **Crash isolation** — a failed pipeline cannot freeze the GUI.
- **Headless mode** — workflow scripts can run from the command line without
  the GUI.
- **No stale state** — the subprocess always reads the latest saved
  configuration.

---

## 2. Layer 1 — The Launcher Config

The launcher config is a master YAML registry that tells the GUI what
workflows exist, what script to run, and where the DAG config lives.

### 2.1 Schema

```yaml
# launcher.yaml — single source of truth for the GUI
#
# To add a new workflow:
#   1. Add an entry under the appropriate category.
#   2. Set `script` to the Python script path (relative to project root).
#   3. Set `dag_config` to the DAG YAML path (relative to project root).
#   4. Restart the GUI — no Python code changes required.

categories:
  - name: Data Ingestion
    workflows:
      - name: Import CSV Files
        script: scripts/import_csv.py
        dag_config: configs/import_csv_dag.yaml

      - name: Validate Schema
        script: scripts/validate_schema.py
        dag_config: configs/validate_schema_dag.yaml

  - name: Transformation
    workflows:
      - name: ETL Pipeline
        script: scripts/etl_pipeline.py
        dag_config: configs/etl_pipeline_dag.yaml

  - name: Reporting
    workflows:
      - name: Generate Dashboard
        script: scripts/generate_dashboard.py
        # dag_config omitted — this script doesn't use a DAG config
```

### 2.2 Parser

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WorkflowEntry:
    """A single workflow declared in launcher.yaml."""
    name: str           # Display name shown on the workflow button
    script: Path        # Absolute path to the Python script
    dag_config: Path | None  # Absolute path to the DAG YAML, or None
    category: str       # Category header for grouping


def parse_launcher_config(
    yaml_path: Path, project_root: Path
) -> list[WorkflowEntry]:
    """Load launcher.yaml and return validated WorkflowEntry instances.

    Entries whose script or dag_config paths don't exist on disk are
    skipped with a warning. The GUI still starts with the remaining
    valid entries.
    """
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    entries: list[WorkflowEntry] = []
    for category_block in data.get("categories", []):
        category = category_block["name"]
        for wf in category_block.get("workflows", []):
            name = wf["name"]
            script = project_root / wf["script"]
            dag_config_rel = wf.get("dag_config")
            dag_config = (project_root / dag_config_rel) if dag_config_rel else None

            if not script.exists():
                warnings.warn(f"Skipping '{name}' — script not found: {script}")
                continue
            if dag_config is not None and not dag_config.exists():
                warnings.warn(f"Skipping '{name}' — dag_config not found: {dag_config}")
                continue

            entries.append(WorkflowEntry(
                name=name, script=script, dag_config=dag_config, category=category
            ))
    return entries
```

### 2.3 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Paths are relative to project root** | Portable across machines and clones |
| **`dag_config` is optional** | Supports scripts that don't need a task graph (e.g. simple launchers) |
| **Invalid entries are skipped, not fatal** | The GUI still starts — one broken entry shouldn't block access to all others |
| **Adding a workflow = editing YAML** | Zero Python code changes required; just restart the GUI |

---

## 3. Layer 2 — The DAG Config Format

Each workflow has a DAG config file that declaratively defines its task
graph: what tasks exist, what options they accept, and how they depend on
each other.

### 3.1 Schema

```yaml
# etl_pipeline_dag.yaml

parameters:
  parallel_execution: false
  input_configs: [batch_2024_q1, batch_2024_q2]

tasks:
  # --- Stage 1: Validation ---
  validate_input:
    enabled: true
    category: processing
    options:
      force_processing: false
      strict_mode: true
    depends_on: []

  # --- Stage 2: Transformation ---
  normalize_data:
    enabled: true
    category: processing
    options:
      force_processing: false
      method: z_score
      filter_params:
        butterworth:
          order: 2
          cutoff_hz: 6.0
        savgol:
          window_length: 7
          polyorder: 5
    depends_on: [validate_input]

  aggregate_data:
    enabled: true
    category: processing
    options:
      force_processing: false
      groupby_columns: [region, date]
    depends_on: [normalize_data]

  # --- Stage 3: Output ---
  generate_report:
    enabled: true
    category: viewer
    options:
      force_processing: false
      format: html
    depends_on: [aggregate_data]

  visualize_results:
    enabled: false
    category: viewer
    options:
      show_interactive: true
    depends_on: [aggregate_data]
```

### 3.2 Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `parameters` | dict | Yes | Global workflow settings |
| `parameters.parallel_execution` | bool | Yes | Process sessions sequentially or in parallel |
| `parameters.<config_key>` | str or list | No | References to input data config directories or files |
| `tasks` | ordered dict | Yes | Task name → task spec mapping (insertion order defines iteration order) |
| `tasks.<name>.enabled` | bool | Yes | Toggle task on/off without removing its config |
| `tasks.<name>.category` | str | No | Classification for GUI color-coding (e.g. `processing`, `viewer`) |
| `tasks.<name>.options` | dict | No | Open dict of task-specific parameters passed to the task function |
| `tasks.<name>.options.force_processing` | bool | No | Convention: bypass cached results and recompute |
| `tasks.<name>.depends_on` | list[str] | Yes | Names of tasks that must complete before this one can run |

### 3.3 Example DAG Visualisation

The example above produces this task graph:

```
  validate_input
       │
       ▼
  normalize_data
       │
       ▼
  aggregate_data
      ╱ ╲
     ╱   ╲
    ▼     ▼
generate   visualize
_report    _results
           (disabled)
```

### 3.4 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`enabled` is a toggle, not delete** | Users can disable tasks without losing complex option blocks. Re-enabling is a single boolean flip. |
| **`depends_on` uses task names (strings)** | Order-independent and refactoring-safe. Renaming a task only requires updating the strings that reference it. |
| **`options` is an open dict** | Each task defines its own parameter schema. The framework imposes no constraints — it just passes the dict through. This keeps the config format extensible without framework changes. |
| **`force_processing` is per-task, not global** | Different tasks have different cache-invalidation needs. A global override would force recomputation of expensive upstream tasks unnecessarily. |
| **`category` has no execution semantics** | It only drives GUI rendering (node colours, grouping). The execution engine ignores it. |
| **YAML comments are preserved** | Users annotate their DAG configs with stage headers and explanatory notes. Round-trip editing (Layer 3) ensures these survive save cycles. |

### 3.5 Advanced: Nested Option Patterns

The `options` dict can contain arbitrarily nested structures. Common
patterns include:

**Filter parameters with multiple algorithms:**
```yaml
options:
  filter_method: butterworth
  filter_params:
    butterworth:
      order: 2
      cutoff_hz: 6.0
    savgol:
      window_length: 7
      polyorder: 5
```

**Profile-based methods (each with an `enabled` flag):**
```yaml
options:
  clustering_methods:
    kmeans:
      method: kmeans
      n_clusters: 8
      enabled: false
    dbscan:
      method: dbscan
      eps: 0.5
      min_samples: 5
      enabled: true
```

**Feature checklists (list of strings):**
```yaml
options:
  extracted_features:
    - mean_value
    - std_value
    - peak_count
```

**Grid groups (dict-of-dicts with numeric bounds):**
```yaml
options:
  grid_groups:
    velocity_pressure_2d:
      enabled: true
      features:
        velocity_mean: {min: 0, max: 500, step: 5, span: 10}
        pressure_mean: {min: 0.002, max: 0.22, step: 0.02, span: 0.04}
```

The GUI (Layer 4) auto-renders different widget types based on these
structures — see [Section 5.4](#54-options-type-dispatch).

---

## 4. Layer 3 — The Dual Model Classes

This is the most important architectural insight in the system. Two
separate Python classes interpret the same YAML file for different
purposes. Neither class knows about the other.

### 4.1 DagConfigModel (GUI Layer)

**Purpose:** Round-trip YAML editing with comment and formatting
preservation.

**YAML library:** `ruamel.yaml` (round-trip mode)

```python
from pathlib import Path
from typing import Any
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

class DagConfigModel:
    """In-memory representation of a DAG YAML config with round-trip fidelity.

    Uses ruamel.yaml so that save-reload cycles preserve section headers,
    inline comments, and key ordering.
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        with open(config_path, "r") as fh:
            self._data = self._yaml.load(fh)
        self._dirty = False

    # --- Properties ---

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    # --- Read ---

    def get_task_names(self) -> list[str]:
        """Return task names in YAML-defined order."""
        tasks = self._data.get("tasks", {}) or {}
        return list(tasks.keys())

    def is_task_enabled(self, task_name: str) -> bool:
        return bool(self._get_task(task_name).get("enabled", False))

    def get_task_options(self, task_name: str) -> dict[str, Any]:
        return dict(self._get_task(task_name).get("options", {}) or {})

    def get_task_option(self, task_name: str, option: str) -> Any:
        opts = self._get_task(task_name).get("options", {}) or {}
        return opts.get(option)

    def get_task_dependencies(self, task_name: str) -> list[str]:
        return list(self._get_task(task_name).get("depends_on", []))

    # --- Write (mutate in-memory, track dirty) ---

    def set_task_enabled(self, task_name: str, enabled: bool) -> None:
        self._get_task(task_name)["enabled"] = enabled
        self._dirty = True

    def set_task_option(self, task_name: str, option: str, value: Any) -> None:
        task = self._get_task(task_name)
        if "options" not in task or task["options"] is None:
            task["options"] = {}
        task["options"][option] = value
        self._dirty = True

    # --- Session config entries ---

    def has_session_configs(self) -> bool:
        params = self._data.get("parameters", {}) or {}
        return "input_configs" in params

    def get_config_entries(self) -> list[str]:
        params = self._data.get("parameters", {}) or {}
        val = params.get("input_configs")
        if val is None:
            return []
        if isinstance(val, str):
            return [val] if val else []
        return list(val)

    def set_config_entries(self, entries: list[str]) -> None:
        """Persist entries, using flow-style list for multiple items."""
        params = self._data.get("parameters")
        if params is None:
            return
        if not entries:
            params["input_configs"] = ""
        elif len(entries) == 1:
            params["input_configs"] = entries[0]
        else:
            seq = CommentedSeq(entries)
            seq.fa.set_flow_style()
            params["input_configs"] = seq
        self._dirty = True

    # --- Persistence ---

    def save(self) -> None:
        """Write back to the original file, preserving comments."""
        self.save_as(self._path)
        self._dirty = False

    def save_as(self, path: Path) -> None:
        """Atomic write: temp file then replace (crash-safe)."""
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", dir=str(path.parent))
        try:
            with open(tmp_fd, "w") as fh:
                self._yaml.dump(self._data, fh)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        if path == self._path:
            self._dirty = False

    def reload(self) -> None:
        """Re-read from disk, discarding in-memory changes."""
        with open(self._path, "r") as fh:
            self._data = self._yaml.load(fh)
        self._dirty = False

    # --- Internal ---

    def _get_task(self, task_name: str) -> dict:
        tasks = self._data.get("tasks", {}) or {}
        task = tasks.get(task_name)
        if task is None:
            raise KeyError(f"Task '{task_name}' not found in config")
        return task
```

**Key behaviours:**

- `ruamel.yaml` round-trip mode preserves comments, blank lines, key
  ordering, and flow-style sequences across load/save cycles.
- `dirty` flag tracks whether unsaved changes exist, enabling "Unsaved
  changes — discard?" prompts in the GUI.
- `save()` uses atomic writes (temp file + replace) to prevent file
  corruption if the process crashes mid-write.
- This class has **no execution state** — no `completed_tasks`, no
  `can_run()`.

### 4.2 DagConfigHandler (Execution Layer)

**Purpose:** Lightweight runtime DAG state machine for pipeline execution.

**YAML library:** `PyYAML` (`yaml.safe_load`)

```python
import copy
from pathlib import Path
from typing import Any
import yaml

class DagConfigHandler:
    """Runtime DAG execution handler.

    Loads a DAG config and tracks task completion state for a single
    pipeline run.
    """

    def __init__(self, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(
                f"DAG configuration file not found at: {config_path}"
            )
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.tasks: dict[str, Any] = self.config.get("tasks", {})
        self.parameters: dict[str, Any] = self.config.get("parameters", {})
        self.completed_tasks: set[str] = set()

    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        return self.parameters.get(param_name, default)

    def get_task_options(self, task_name: str) -> dict[str, Any]:
        task_config = self.tasks.get(task_name, {})
        return task_config.get("options", {})

    def can_run(self, task_name: str) -> bool:
        """True if the task is enabled AND all dependencies have been completed."""
        task_config = self.tasks.get(task_name)
        if not task_config:
            return False
        if not task_config.get("enabled", False):
            return False
        dependencies: list[str] = task_config.get("depends_on", [])
        if not set(dependencies).issubset(self.completed_tasks):
            return False
        return True

    def mark_completed(self, task_name: str) -> None:
        """Mark a task as done, allowing dependent tasks to proceed."""
        if task_name in self.tasks:
            self.completed_tasks.add(task_name)

    def copy(self):
        """Deep copy for per-session independent state."""
        return copy.deepcopy(self)
```

**Key behaviours:**

- Uses `yaml.safe_load()` — fast, no `ruamel.yaml` dependency at runtime.
- `completed_tasks: set[str]` is the DAG execution cursor — the only
  mutable state.
- `can_run()` encodes the core scheduling logic: `enabled AND all
  dependencies in completed_tasks`.
- `copy()` creates a deep copy for per-session isolation (explained in
  [Section 6.2](#62-per-session-dag-copies)).

### 4.3 Why Two Classes?

| Concern | DagConfigModel | DagConfigHandler |
|---------|----------------|------------------|
| YAML library | `ruamel.yaml` (round-trip) | `PyYAML` (`safe_load`) |
| Preserves comments | Yes | No |
| Tracks dirty state | Yes | No |
| Tracks completed tasks | No | Yes |
| Has `can_run()` | No | Yes |
| Used by | GUI process | Subprocess (execution) |
| Mutates file on disk | Yes (save) | Never |

**Rationale:** GUI editing requires preserving human-readable formatting
(comments, key order, blank lines). Runtime execution requires tracking
mutable state (which tasks have completed). Combining both into one class
would create coupling between the GUI dependency (`ruamel.yaml`) and the
execution scripts, which should be lightweight and runnable headless.

The YAML file on disk is the interface contract between the two. The GUI
writes it; the subprocess reads it. They never need to share an object.

---

## 5. Layer 4 — The GUI Shell

A three-column PyQt5 application structured as a composition of
independent widgets.

### 5.1 Layout

```
+------------------------------------------------------------------+
|  [Save] [Save As...]                                   Toolbar   |
+------------------------------------------------------------------+
|              |                        |                           |
|  Workflow    |    Task Panel          |  Session Config            |
|  Selector    |  +------------------+ |  Selector                  |
|              |  | [Graph] [Table]  | |                           |
|  INGESTION   |  |  +-----------+   | |  [x] batch_2024_q1/       |
|   *Import*   |  |  | DAG Graph |   | |    [x] config_001.yaml    |
|   Validate   |  |  | (Sugiyama)|   | |    [x] config_002.yaml    |
|              |  |  +-----------+   | |  [ ] batch_2024_q2/       |
|  TRANSFORM   |  |                   | |    [ ] config_003.yaml    |
|   ETL        |  |  Task Detail:     | |                           |
|              |  |  normalize_data   | |                           |
|  REPORTING   |  |   method: z_score | |                           |
|   Dashboard  |  |   [x] enabled     | |                           |
|              |  |   [ ] force       | |                           |
+------------------------------------------------------------------+
|  Console Output                                                  |
|  > [batch_001] Running task: validate_input                      |
|  > [batch_001] Task 'validate_input' marked as completed.        |
|  > [batch_001] Running task: normalize_data                      |
|  > [batch_001] Task 'normalize_data' marked as completed.        |
+------------------------------------------------------------------+
|  Script: scripts/etl_pipeline.py                  [Run] [Abort]  |
+------------------------------------------------------------------+
```

The window uses nested `QSplitter` widgets:

- **Horizontal splitter** (top): workflow selector (25%) | task panel
  (50%) | session config selector (25%)
- **Vertical splitter**: horizontal splitter (70%) | console (30%)
- **Run bar** (fixed): script path label + Run/Abort buttons

### 5.2 Component Responsibilities

#### WorkflowSelector (left column)

- **Input:** `list[WorkflowEntry]` from `parse_launcher_config()`
- **UI:** Category-grouped exclusive toggle buttons (`QButtonGroup`)
- **Signal:** `workflow_changed(WorkflowEntry)` — emitted when a button is
  clicked
- **Behaviour:** Clicking a workflow button triggers the main window to
  load that workflow's DAG config

```python
class WorkflowSelector(QWidget):
    workflow_changed = pyqtSignal(object)  # emits WorkflowEntry

    def __init__(self, entries: list[WorkflowEntry]) -> None:
        # For each entry: create QPushButton(entry.name), group by category
        # with QLabel headers and QFrame separator lines.
        # QButtonGroup with exclusive=True ensures only one is active.
        ...

    def _on_button_clicked(self, btn_id: int) -> None:
        self.workflow_changed.emit(self._entries[btn_id])
```

#### TaskPanel (centre column)

- **Input:** `DagConfigModel`
- **UI:** Stacked widget with two views toggled by buttons:
    1. **Graph View** — interactive DAG visualisation (grandalf Sugiyama
       layout)
    2. **Table View** — 3-column `QTableWidget` (task name | enabled
       checkbox | depends_on text)
- **Sub-panel:** `TaskDetailPanel` — scrollable options editor that
  auto-renders widgets based on option value types
- **Signal:** `task_changed()` — emitted on any toggle/edit; marks the
  model dirty

```python
class TaskPanel(QWidget):
    task_changed = pyqtSignal()

    def populate(self, model: DagConfigModel) -> None:
        # Rebuild both views from the model's task list.
        self._graph_view.populate(model)
        # Build table rows with checkboxes for enabled/force.
        ...
```

#### SessionConfigSelector (right column)

- **Input:** `DagConfigModel` (reads config type and current entries)
- **UI:** `QTreeWidget` with checkable directory/file hierarchy
- **Behaviour:** Fully-checked directory emits the directory name (compact
  representation); partially-checked directory emits individual file paths
- **Signal:** `selection_changed()` — emitted on check/uncheck

#### ConsoleWidget (bottom)

- **UI:** Read-only `QPlainTextEdit` with monospace font and dark theme
- **Features:**
    - Auto-scroll toggle checkbox
    - Maximum block count (e.g. 10,000 lines) to prevent memory bloat
    - Carriage-return handling for progress bars (e.g. tqdm): `replace_last_line()`
      overwrites the current line instead of appending

```python
class ConsoleWidget(QWidget):
    _MAX_BLOCKS = 10_000

    def append_line(self, text: str) -> None:
        """Append one line and auto-scroll if enabled."""
        self._text.appendPlainText(text)
        if self._auto_scroll_enabled:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def replace_last_line(self, text: str) -> None:
        """Overwrite the last line (carriage-return semantics)."""
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.insertText(text)
```

### 5.3 Signal Flow

```
WorkflowSelector.workflow_changed(entry)
  └── LauncherWindow._load_workflow(entry)
        ├── DagConfigModel(entry.dag_config)
        ├── TaskPanel.populate(model)
        └── SessionConfigSelector.populate(model)

TaskPanel.task_changed()
  └── LauncherWindow._mark_dirty()
        ├── model._dirty = True
        └── window title shows " *"

SessionConfigSelector.selection_changed()
  └── LauncherWindow._on_config_changed()
        ├── model.set_config_entries(selection)
        └── _mark_dirty()

RunButton.clicked()
  └── LauncherWindow._on_run()
        ├── model.save()                      # persist GUI edits to disk
        ├── subprocess.Popen(                 # launch workflow script
        │     [python, script, --dag-config, dag_yaml],
        │     stdout=PIPE, stderr=STDOUT
        │   )
        ├── ProcessOutputReader(process)
        │     .line_received → Console.append_line
        │     .cr_line_received → Console.replace_last_line
        └── QTimer(500ms) → _poll_process()   # detect completion
```

### 5.4 Options Type Dispatch

The `TaskDetailPanel` auto-renders different widgets based on the type and
structure of each option value:

| Value Type | Widget | Example YAML |
|------------|--------|--------------|
| `bool` | `QCheckBox` | `strict_mode: true` |
| `int`, `float`, `str` | `QLineEdit` | `cutoff_hz: 6.0` |
| Registered enum | `QComboBox` | `method: z_score` (dropdown: z_score, min_max, robust) |
| `list[str]` | Checklist of `QCheckBox`es | `extracted_features: [mean, std]` |
| `dict` with `method` keys in sub-dicts | Profile checkboxes + params dialog | `clustering_methods: {kmeans: {method: ...}}` |
| `dict` with `enabled` + `features` | Grid/cluster group dialog | `grid_groups: {combo_1: {enabled: true, features: {...}}}` |
| Complex/unrecognised | Clickable label → YAML editor dialog | Any deeply nested structure |

This dispatch is done by inspecting the option value at render time — no
schema registration is needed for basic types. Only enums and checklists
require explicit registration (a dict mapping option names to their allowed
values).

### 5.5 DAG Graph Visualisation

The graph view uses the `grandalf` library for Sugiyama-style hierarchical
layout:

1. Each task becomes a `Vertex` with a view object providing width/height
2. Dependencies become directed edges (`GEdge(parent, child)`)
3. `grandalf.SugiyamaLayout` computes layered positions
4. Multiple disconnected components are laid out side by side with padding
5. Nodes are `QGraphicsRectItem` subclasses with embedded checkboxes for
   `enabled` and `force_processing`
6. Edges are `QGraphicsPathItem` subclasses drawn as directed arrows
7. Nodes are colour-coded by `category`:

```python
CATEGORY_COLOURS = {
    "processing":      QColor("#d0e8ff"),  # light blue
    "viewer":          QColor("#e8d0ff"),  # light purple
    "viewer_required": QColor("#ffe0b0"),  # light orange
    "none":            QColor("#f0f0f0"),  # light grey
}
# Disabled tasks use a desaturated colour (#e8e8e8)
```

**Layout persistence:** Node positions are saved to a `.layout.json` file
(sibling to the DAG YAML) so manual rearrangements survive GUI restarts.
Saving is debounced (800ms timer) to avoid excessive disk writes during
drag operations.

**Interactions:**
- Scroll wheel: zoom in/out
- Middle-click drag: pan
- Left-click node: select and show options in the detail panel
- `Ctrl+0`: fit all nodes in view

### 5.6 Main Window Lifecycle

```python
class LauncherWindow(QMainWindow):

    def __init__(self, entries, configs_dir):
        # Build toolbar (Save, Save As)
        # Build UI (3-column splitter + console + run bar)
        # Connect signals
        # Optionally start orchestrator server (e.g. Prefect)
        ...

    def _load_workflow(self, entry: WorkflowEntry) -> None:
        """Called when user clicks a workflow button."""
        if self._model and self._model.dirty:
            if not self._confirm_discard():
                return
        if entry.dag_config is None:
            self._model = None  # No task panel for scripts without DAG
            return
        self._model = DagConfigModel(entry.dag_config)
        self._task_panel.populate(self._model)
        self._session_selector.populate(self._model)

    def _on_run(self) -> None:
        """Launch the workflow script as a subprocess."""
        if self._model:
            self._on_save()  # persist any GUI edits

        cmd = [sys.executable, str(self._current_entry.script)]
        if self._current_entry.dag_config is not None:
            cmd += ["--dag-config", str(self._current_entry.dag_config)]

        self._process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Async output reading via QThread
        self._reader = ProcessOutputReader(self._process)
        self._reader.line_received.connect(self._console.append_line)
        self._reader.cr_line_received.connect(self._console.replace_last_line)
        self._reader.start()

        # Poll for completion every 500ms
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_process)
        self._poll_timer.start()

    def _poll_process(self) -> None:
        retcode = self._process.poll()
        if retcode is None:
            return  # still running
        self._poll_timer.stop()
        label = "Finished" if retcode == 0 else "Failed"
        self.statusBar().showMessage(f"{label} (exit code {retcode})")
        self._run_button.setEnabled(True)

    def _on_abort(self) -> None:
        """Terminate the running subprocess."""
        if self._process:
            self._process.terminate()

    def closeEvent(self, event) -> None:
        # Terminate any running subprocess
        # Prompt if unsaved changes exist
        # Stop orchestrator server
        ...
```

### 5.7 ProcessOutputReader

A `QThread` that reads stdout from the subprocess line by line and emits
signals back to the GUI thread:

```python
class ProcessOutputReader(QThread):
    line_received = pyqtSignal(str)
    cr_line_received = pyqtSignal(str)

    def __init__(self, process: subprocess.Popen):
        super().__init__()
        self._process = process

    def run(self):
        for raw_line in iter(self._process.stdout.readline, b""):
            text = raw_line.decode("utf-8", errors="replace")
            # Strip ANSI escape codes
            text = _strip_ansi(text)
            if "\r" in text and not text.endswith("\n"):
                self.cr_line_received.emit(text.rstrip("\r"))
            else:
                self.line_received.emit(text.rstrip("\n"))
```

### 5.8 Error Handling in the GUI

| Error Source | How it's surfaced |
|--------------|-------------------|
| DAG config fails to parse | `QMessageBox.critical("Load Error", ...)` |
| Save fails (I/O error) | `QMessageBox.critical("Save Error", ...)` |
| Workflow script fails | Exit code in status bar + full traceback in console |
| Script not found | Run button disabled, status label shows "Script not found" |
| User closes with unsaved changes | "Discard?" confirmation dialog |

---

## 6. Layer 5 — The Execution Engine

### 6.1 Subprocess Isolation

```
GUI Process                                Pipeline Subprocess
-----------                                --------------------
model.save()                                                     
   │  writes to disk                                             
   ▼                                                             
*_dag.yaml ─────────────────────────── DagConfigHandler(path)    
                                              │                  
Popen([python, script,                        ▼                  
       --dag-config, path])            resolve_session_configs() 
                                              │                  
                                              ▼                  
                                       for session in configs:   
                                           dag = template.copy() 
                                           run_session(dag)      
```

The workflow script is a standalone Python program that:

1. Parses `--dag-config <path>` from the command line
2. Creates a `DagConfigHandler` from the YAML
3. Reads `input_configs` from parameters
4. Resolves session configs to a list of YAML files
5. Loops over sessions, running the task graph for each

This means the same script works from both the GUI and the command line:

```bash
# From GUI:
subprocess.Popen([python, "scripts/etl_pipeline.py",
                  "--dag-config", "configs/etl_pipeline_dag.yaml"])

# From command line (headless):
python scripts/etl_pipeline.py --dag-config configs/etl_pipeline_dag.yaml
```

### 6.2 Per-Session DAG Copies

```python
dag_template = DagConfigHandler(dag_config_path)

for session_config_file in resolved_session_configs:
    dag_instance = dag_template.copy()  # deep copy
    run_session(session_config_file, dag_instance)
```

**Why copy?** `DagConfigHandler` tracks `completed_tasks` as mutable
state. Without copying, session 2 would see session 1's completed tasks
and skip tasks that should run again. `copy.deepcopy()` ensures each
session starts with a fresh, empty `completed_tasks` set while preserving
the task definitions and parameters.

### 6.3 Session Config Resolution

The DAG `parameters` section references input data by directory name or
file path. The resolver expands these to a flat list of YAML files:

```python
def resolve_session_configs(
    entries: str | list[str],
    config_root: Path,
) -> list[Path]:
    """Resolve directory and file entries to a sorted flat list of YAML paths.

    - Directory entry → all *.yaml files inside, sorted
    - File entry → included directly
    - Result: deduplicated, order-preserving list

    Raises FileNotFoundError if an entry resolves to a non-existent path.
    """
    if not entries:
        return []
    if isinstance(entries, str):
        entries = [entries]

    result: list[Path] = []
    seen: set[Path] = set()

    for entry in entries:
        path = config_root / entry
        if path.is_dir():
            for yaml_file in sorted(path.glob("*.yaml")):
                if yaml_file not in seen:
                    result.append(yaml_file)
                    seen.add(yaml_file)
        elif path.exists():
            if path not in seen:
                result.append(path)
                seen.add(path)
        else:
            raise FileNotFoundError(
                f"Config entry not found: '{entry}' (resolved to '{path}')"
            )
    return result
```

**Example:**

```yaml
# In DAG config
parameters:
  input_configs: [batch_2024_q1, batch_2024_q2]
```

```
configs/input_configs/batch_2024_q1/session_001.yaml
configs/input_configs/batch_2024_q1/session_002.yaml
configs/input_configs/batch_2024_q2/session_003.yaml
```

`resolve_session_configs(["batch_2024_q1", "batch_2024_q2"], Path("configs/input_configs"))`
returns all three paths sorted and deduplicated.

### 6.4 The TaskExecutor Context Manager

The core execution primitive. Every task runs inside a `TaskExecutor`
context manager that handles the bookkeeping:

```python
import traceback

class TaskExecutor:
    """Context manager that handles dependency checking, execution, and completion."""

    def __init__(self, task_name, block_name, dag_handler, monitor,
                 session_name: str = None):
        self.task_name = task_name
        self.block_name = block_name
        self.dag_handler = dag_handler
        self.monitor = monitor
        self.session_name = session_name
        self.can_run: bool = False
        self.error_msg: str = None

    def __enter__(self):
        if self.dag_handler.can_run(self.task_name):
            self.can_run = True
            print(f"[{self.block_name}] Running task: {self.task_name}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "RUNNING")
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if not self.can_run:
            return

        if exc_type:
            self.error_msg = f"Task '{self.task_name}' failed: {exc_value}"
            print(f"{self.error_msg}\n{traceback.format_exc()}")
            if self.monitor is not None:
                self.monitor.update(
                    self.block_name, self.task_name, "FAILURE", self.error_msg
                )
            return True  # swallow exception
        else:
            self.dag_handler.mark_completed(self.task_name)
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "SUCCESS")
        return False
```

### 6.5 Usage Pattern

```python
# Define pipeline stages as a list of dicts
pipeline_stages = [
    {
        "name": "validate_input",
        "func": validate_input,
        "params": lambda: {
            "input_path": session_config.input_path,
            "output_dir": session_config.output_dir,
        },
        "outputs": ["validated_path"],
    },
    {
        "name": "normalize_data",
        "func": normalize_data,
        "params": lambda: {
            "input_path": context.get("validated_path"),
            "output_dir": session_config.output_dir,
        },
        "outputs": ["normalized_path"],
    },
    {
        "name": "generate_report",
        "func": generate_report,
        "params": lambda: {
            "input_path": context.get("normalized_path"),
            "output_dir": session_config.output_dir,
        },
        "outputs": ["report_path"],
    },
]

# Context dict accumulates outputs from completed tasks
context = {}

for stage in pipeline_stages:
    task_name = stage["name"]
    executor = TaskExecutor(task_name, block_name, dag_handler, monitor)

    with executor:
        if not executor.can_run:
            continue

        # Get task options from the DAG config
        options = dag_handler.get_task_options(task_name)
        params = stage["params"]()

        # Inject standard options
        force = options.get("force_processing")
        if force is not None:
            params["force_processing"] = force

        # Execute the task function
        result = stage["func"](**params)

    # Store outputs for downstream tasks
    if "outputs" in stage and result is not None:
        if not isinstance(result, tuple):
            result = (result,)
        for i, key in enumerate(stage["outputs"]):
            if key and i < len(result):
                context[key] = result[i]

    # Handle failure: skip remaining tasks for this session
    if executor.error_msg:
        return {"status": "failed", "error": executor.error_msg}

return {"status": "success"}
```

### 6.6 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Context manager pattern** | Ensures `mark_completed()` always runs on success and error logging always runs on failure, regardless of how the task function exits. Prevents accidental omission of bookkeeping. |
| **`return True` on exception** | The executor swallows task exceptions so the pipeline loop can inspect `executor.error_msg` and decide whether to abort or continue. The decision is made outside the `with` block. |
| **`can_run` as a property check, not a gate** | The task loop iterates all stages unconditionally. Inside the `with` block, `can_run` is `False` if the task is disabled or dependencies aren't met. This keeps the loop structure uniform regardless of which tasks are enabled. |
| **Lambda for params** | Some parameters depend on outputs of earlier tasks (stored in the `context` dict). Using `lambda` defers evaluation until the task is about to run, when prior outputs are available. |
| **`context` dict for inter-task outputs** | Tasks return paths or values that downstream tasks need. The `outputs` key in each stage definition names the context keys where results are stored. This creates an explicit data-flow graph alongside the dependency graph. |

### 6.7 Batch Processing

The top-level workflow script orchestrates batch processing across
sessions:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()

    dag_handler = DagConfigHandler(args.dag_config)
    is_parallel = dag_handler.get_parameter("parallel_execution", False)
    entries = dag_handler.get_parameter("input_configs")
    config_files = resolve_session_configs(entries, Path("configs/input_configs"))

    dag_template = DagConfigHandler(args.dag_config)

    for config_file in config_files:
        session_config = load_session_config(config_file)
        dag_instance = dag_template.copy()  # fresh DAG state per session

        if is_parallel:
            # Submit as async flow (e.g. Prefect .submit())
            run_session.submit(session_config, dag_instance)
        else:
            run_session(session_config, dag_instance)
```

### 6.8 Monitoring (Optional)

An optional `PipelineMonitor` can track task status across sessions:

```
Workers --> Queue --> Coordinator Thread --> DataManager (in-memory)
                            |
                            ├── Excel/CSV report (throttled writes)
                            └── Live dashboard (separate process)
```

The monitor uses a single API call from task code:

```python
monitor.update(dataset_name, stage_name, status, message="")
```

Status values: `RUNNING`, `SUCCESS`, `FAILURE`, `SKIPPED`.

---

## 7. Incremental Adoption Guide

### Phase 1: Config Format + CLI Execution (1-2 days)

**What you get:** Declarative pipeline configs runnable from the command
line.

**What to build:**

1. Define your `*_dag.yaml` files following the schema in [Section 3](#3-layer-2--the-dag-config-format)
2. Implement `DagConfigHandler` (~70 lines) — [Section 4.2](#42-dagconfighandler-execution-layer)
3. Implement `TaskExecutor` (~50 lines) — [Section 6.4](#64-the-taskexecutor-context-manager)
4. Write a workflow script that parses `--dag-config` and runs the task
   loop — [Section 6.5](#65-usage-pattern)

**Dependency diagram:**

```
*_dag.yaml
    │
    ▼
DagConfigHandler ──► TaskExecutor ──► your task functions
```

At this point you can run pipelines from the terminal:

```bash
python scripts/my_pipeline.py --dag-config configs/my_pipeline_dag.yaml
```

Users edit the YAML directly to toggle tasks and change options.

### Phase 2: Session Resolution + Batch Processing (1-2 days)

**What you get:** Batch processing across multiple input configs with
per-session isolation.

**What to build:**

1. Implement `resolve_session_configs()` (~30 lines) — [Section 6.3](#63-session-config-resolution)
2. Add the per-session `dag_handler.copy()` loop — [Section 6.2](#62-per-session-dag-copies)
3. Add `parallel_execution` support if needed — [Section 6.7](#67-batch-processing)

**Dependency diagram:**

```
Phase 1 components
    │
    ▼
resolve_session_configs() ──► batch execution loop
                                    │
                                    ▼
                              dag_template.copy() per session
```

### Phase 3: The GUI (1-2 weeks)

**What you get:** A full interactive application for selecting workflows,
editing DAG configs, choosing input data, and monitoring execution.

**What to build (in order):**

1. `launcher.yaml` and `parse_launcher_config()` — [Section 2](#2-layer-1--the-launcher-config)
2. `DagConfigModel` with `ruamel.yaml` — [Section 4.1](#41-dagconfigmodel-gui-layer)
3. `WorkflowSelector` widget — [Section 5.2](#52-component-responsibilities)
4. `TaskPanel` with table view — [Section 5.2](#52-component-responsibilities)
5. `SessionConfigSelector` widget — [Section 5.2](#52-component-responsibilities)
6. `ConsoleWidget` + `ProcessOutputReader` — [Section 5.2](#52-component-responsibilities) and [Section 5.7](#57-processoutputreader)
7. `LauncherWindow` that wires everything together — [Section 5.6](#56-main-window-lifecycle)
8. (Optional) `DagGraphView` with grandalf Sugiyama layout — [Section 5.5](#55-dag-graph-visualisation)

**Dependency diagram:**

```
Phase 1 + Phase 2 components
    │
    ▼
DagConfigModel (ruamel.yaml) ◄── all GUI widgets read/write through this
    │
    ▼
LauncherWindow
├── WorkflowSelector
├── TaskPanel
│   ├── Table View
│   ├── Graph View (optional)
│   └── TaskDetailPanel
├── SessionConfigSelector
├── ConsoleWidget
└── ProcessOutputReader
```

---

## Appendix A — Complete YAML Schema Reference

### A.1 launcher.yaml

```yaml
# Top-level key: categories (required, list)
categories:
  - name: <string>        # Category display name (required)
    workflows:             # List of workflows in this category (required)
      - name: <string>    # Workflow display name (required)
        script: <string>  # Path to Python script, relative to project root (required)
        dag_config: <string>  # Path to DAG YAML, relative to project root (optional)
```

**Constraints:**
- `script` must point to an existing file
- `dag_config`, if present, must point to an existing file
- Categories are rendered in declaration order
- Workflows within a category are rendered in declaration order

### A.2 *_dag.yaml

```yaml
# Global parameters (required)
parameters:
  parallel_execution: <bool>      # Sequential vs parallel session processing (required)
  <config_key>: <string | list>   # Input data references (optional)
  # Additional custom parameters as needed

# Task graph (required)
tasks:
  <task_name>:                    # Unique task identifier (required)
    enabled: <bool>               # Toggle task on/off (required)
    category: <string>            # GUI colour classification (optional)
    options:                      # Task-specific parameters (optional)
      force_processing: <bool>    # Bypass cached results (optional, convention)
      <key>: <any>                # Open schema — any valid YAML
    depends_on: <list[string]>    # Task names that must complete first (required)
    description: <string>         # Human-readable task description (optional)
```

**Constraints:**
- Task names must be unique within a file
- `depends_on` entries must reference task names that exist in the same file
- The task graph must be acyclic (no circular dependencies)
- Tasks are iterated in YAML insertion order

---

## Appendix B — Class Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          GUI Process                                    │
│                                                                         │
│  ┌─────────────────┐       ┌──────────────────────────────────┐        │
│  │ WorkflowSelector │──────▶│ LauncherWindow                   │        │
│  └─────────────────┘       │                                  │        │
│                            │  owns ─┬─▶ TaskPanel             │        │
│  ┌────────────────────┐    │        │    ├── DagGraphView     │        │
│  │ SessionConfig      │◀───│        │    │   ├── DagTaskNode  │        │
│  │ Selector           │    │        │    │   └── DagEdge      │        │
│  └────────────────────┘    │        │    └── TaskDetailPanel  │        │
│                            │        │                         │        │
│  ┌────────────────────┐    │        ├─▶ ConsoleWidget         │        │
│  │ ProcessOutputReader │◀───│        │                         │        │
│  │ (QThread)          │    │        └─▶ ProcessOutputReader   │        │
│  └────────────────────┘    │                                  │        │
│                            │  uses ──▶ DagConfigModel         │        │
│                            │           (ruamel.yaml)          │        │
│                            └──────────────────────────────────┘        │
│                                          │                              │
│                                          │ save()                       │
│                                          ▼                              │
│                                    *_dag.yaml on disk                   │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                              subprocess.Popen (stdout piped)
                                           │
┌──────────────────────────────────────────┼──────────────────────────────┐
│                          Pipeline Subprocess                            │
│                                          │                              │
│                                          ▼                              │
│                               ┌─────────────────────┐                   │
│                               │ DagConfigHandler     │                   │
│                               │ (PyYAML)             │                   │
│                               │                     │                   │
│                               │ can_run()           │                   │
│                               │ mark_completed()    │                   │
│                               │ copy()              │                   │
│                               └────────┬────────────┘                   │
│                                        │                                │
│                                        ▼                                │
│                               ┌─────────────────────┐                   │
│                               │ TaskExecutor         │                   │
│                               │ (context manager)    │                   │
│                               │                     │                   │
│                               │ __enter__ → can_run  │                   │
│                               │ __exit__  → complete │                   │
│                               └────────┬────────────┘                   │
│                                        │                                │
│                                        ▼                                │
│                               ┌─────────────────────┐                   │
│                               │ Your Task Functions  │                   │
│                               │ (Prefect @flow or    │                   │
│                               │  plain functions)    │                   │
│                               └─────────────────────┘                   │
│                                                                         │
│  resolve_session_configs() ──▶ session loop with dag.copy()             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ownership relationships:**
- `LauncherWindow` owns all GUI widgets and the `DagConfigModel`
- `TaskPanel` owns `DagGraphView`, `TaskDetailPanel`, and the table
- `DagGraphView` owns `DagTaskNode` and `DagEdge` items
- In the subprocess, `DagConfigHandler` is used by `TaskExecutor` and the
  session loop
- The two processes share no objects — only the YAML file on disk
