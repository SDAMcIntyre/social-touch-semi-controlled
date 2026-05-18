# Plan: DAG Graph View for Task Panel

**Date:** 2026-05-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/slim-forearm-projection`
**Branch:** `feature/dag-graph-view`

---

## Overview

**What:** Add an interactive DAG graph visualization to the Tasks tab of the DAG
Config Launcher, showing task nodes and dependency edges instead of a flat table.
**Why:** The flat `QTableWidget` makes it hard to trace dependency chains in large
workflows (up to 18 tasks, 6-level chains). Users must mentally reconstruct the DAG
from comma-separated text.
**How:** Build a custom `QGraphicsScene`/`QGraphicsView` widget with `grandalf`
(Sugiyama layout), embedded checkboxes via `QGraphicsProxyWidget`, and a toggle to
switch between graph (default) and table views.

## Problem Statement

- The Tasks tab renders workflow tasks as a 3-column table: Task Name,
  Enabled/Force checkboxes, Depends On (comma-separated text).
- For small workflows (5-7 tasks), this is adequate. For the analysis processing
  workflow (18 tasks, dependency chains 5-6 levels deep), the flat layout obscures
  the pipeline structure.
- Users cannot see at a glance which tasks are upstream/downstream of a given node,
  which parallel branches exist, or how deep the dependency chain goes.
- The only dependency navigation is clicking the Depends On cell, which scrolls to
  the first dependency — no overview of the full graph.

## Goals

### In Scope
1. Interactive DAG graph view with task nodes and dependency edges
2. Enabled and Force checkboxes embedded in each node
3. Automatic Sugiyama (layered) layout, left-to-right direction
4. Toggle between graph view (default) and table view via button pair
5. Clicking a node selects it and shows its details in the `TaskDetailPanel`
6. Zoom (scroll wheel) and pan (middle-click drag) in the graph view
7. Visual distinction between enabled/disabled tasks and task categories
8. "Fit all" action to reset the viewport to show all nodes

### Out of Scope
- Drag-to-reposition nodes (layout is automatic only)
- Editing dependency edges in the graph (dependencies are YAML-driven)
- Animated transitions when toggling enabled/disabled state
- Collapsible sub-graphs or node grouping
- Topological sort validation or cycle detection in the model layer

## Success Criteria

- [ ] Graph view renders all tasks from any `*_dag.yaml` as nodes with edges
- [ ] Enabled and Force checkboxes in graph nodes update the `DagConfigModel`
- [ ] Clicking a graph node selects it and populates the `TaskDetailPanel`
- [ ] Toggle between graph (default) and table preserves the selected task
- [ ] Zoom and pan work in the graph view
- [ ] Layout has no node overlaps for the largest config (18 tasks)
- [ ] All existing table-view functionality is preserved and reachable

---

## Technical Design

### Approach

Build a DAG graph widget from Qt's `QGraphicsScene`/`QGraphicsView` with the
`grandalf` library for Sugiyama layout. Embed `QCheckBox` widgets inside graph
nodes via `QGraphicsProxyWidget`. Integrate into the existing `TaskPanel` via a
`QStackedWidget` that toggles between the current `QTableWidget` (page 1) and the
new `DagGraphView` (page 0, default).

This approach was selected because:
- The problem is displaying a read-only DAG with interactive property toggles, not
  building a full graph editor
- `grandalf` is pure Python with zero transitive dependencies (~600 lines of
  Sugiyama implementation)
- `QGraphicsProxyWidget` gives native Qt checkbox embedding — no custom painting
  needed
- Full control over visual design without fighting a library's abstraction model

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Custom QGraphicsScene + grandalf | Minimal deps, full control, native Qt | More upfront code (~500 lines) | **Chosen** |
| QNodeEditor + grandalf | Less node-rendering boilerplate | Two deps, port/connection model overhead for read-only DAG | Rejected |
| NodeGraphQt | Built-in auto-layout, batteries-included | Heavy API for viewer use case, property system instead of embedded QWidgets | Rejected |
| React Flow + QWebEngineView | Beautiful web-based UI | ~100MB QtWebEngine dep, cross-tech debugging, overkill | Rejected |
| NetworkX + Matplotlib | Simple, good layout | No interactivity (static raster), no checkboxes | Rejected |
| Mermaid.js + QWebEngineView | Text-based, simple | Static SVG, no interactivity at all | Rejected |

### Architecture Constraints

- **`note-qt-itemchanged-signal-recursion.md`**: Any checkbox `stateChanged`
  handler that mutates widget state must guard with `blockSignals(True/False)` to
  prevent re-entry. Applies to enabled/force toggle handlers in graph nodes.
- **ruamel.yaml only**: The `DagConfigModel` already handles YAML round-trip
  fidelity. The graph view only calls model methods; it never touches YAML directly.

### Architecture Changes

New widget hierarchy inside `TaskPanel`:

```
TaskPanel (QWidget)
  +- QGroupBox("Tasks")
       +- QVBoxLayout
            +- QHBoxLayout (toggle bar)
            |    +- QPushButton("Graph View")   -- checked by default
            |    +- QPushButton("Table View")
            +- QSplitter(Vertical)
                 +- QStackedWidget
                 |    +- page 0: DagGraphView   -- default
                 |    +- page 1: QTableWidget   -- existing
                 +- TaskDetailPanel             -- shared
```

New files:

```
code/src/utils/gui/dag_launcher/
  +- dag_graph_view.py    -- QGraphicsView subclass + scene builder + layout
  +- dag_graph_items.py   -- DagTaskNode + DagEdge QGraphicsItem subclasses
```

---

## Implementation Plan

### Phase 1: Graph Items and Layout Engine
**Goal:** Create the node and edge graphics items, and the grandalf layout bridge.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 1.1 -- Add `grandalf` to `requirements.txt`
- [x] Task 1.2 -- Create `dag_graph_items.py` with `DagTaskNode(QGraphicsRectItem)`:
  embedded task name `QLabel`, enabled `QCheckBox`, conditional force `QCheckBox`,
  visual states (enabled/disabled/selected), `node_clicked` and `enabled_changed`
  and `force_changed` signals
- [x] Task 1.3 -- Create `dag_graph_items.py` `DagEdge(QGraphicsPathItem)`: cubic
  bezier path from source node right edge to target node left edge, arrowhead
  painting, subtle color styling
- [x] Task 1.4 -- Create `dag_graph_view.py` with `DagGraphView(QGraphicsView)`:
  `populate(model)` method that builds `grandalf` graph from `DagConfigModel`,
  runs Sugiyama layout, creates `DagTaskNode` and `DagEdge` items, places them in
  the scene. Include zoom (scroll wheel with `AnchorUnderMouse`), pan (middle-click
  drag), and `fit_all()` method.

**Files Modified:**
- `requirements.txt` -- Add `grandalf`
- `code/src/utils/gui/dag_launcher/dag_graph_items.py` -- New file (~200-250 lines)
- `code/src/utils/gui/dag_launcher/dag_graph_view.py` -- New file (~300-400 lines)

**Dependencies:** None

### Phase 2: TaskPanel Integration
**Goal:** Wire the graph view into TaskPanel with toggle buttons and shared signals.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 2.1 -- Refactor `TaskPanel.__init__` to add toggle button bar and
  `QStackedWidget` wrapping the existing `QTableWidget` (page 1) and a new
  `DagGraphView` instance (page 0, default)
- [x] Task 2.2 -- Wire toggle buttons to `QStackedWidget.setCurrentIndex()`
- [x] Task 2.3 -- Wire `DagGraphView` node signals: `node_clicked` to
  `TaskDetailPanel.show_task()`, `enabled_changed`/`force_changed` to model
  setters and `task_changed.emit()`
- [x] Task 2.4 -- Update `TaskPanel.populate()` to call both
  `_populate_table(model)` and `_graph_view.populate(model)`
- [x] Task 2.5 -- Synchronize selected task when toggling views: switching from
  graph to table selects the same row, and vice versa
- [x] Task 2.6 -- Delete deprecated `dag_task_widget.py` stub

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` -- Refactor layout, add toggle,
  wire graph view (~60 lines added/modified)
- `code/src/utils/gui/dag_launcher/dag_task_widget.py` -- Deleted

**Dependencies:** Phase 1

### Phase 3: Visual Polish and Edge Cases
**Goal:** Handle edge cases and add visual refinements.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 3.1 -- Color-code nodes by category (`processing` vs `viewer_required`)
  using distinct fill colors
- [x] Task 3.2 -- Dim disabled nodes (semi-transparent fill, muted border)
- [x] Task 3.3 -- Handle single-task configs and configs with no dependencies
  (isolated nodes positioned cleanly)
- [x] Task 3.4 -- Handle configs with multiple root nodes (no dependencies)
- [x] Task 3.5 -- Add a "Fit All" button or keyboard shortcut (Ctrl+0) to reset
  the viewport
- [x] Task 3.6 -- Synchronize checkbox state between graph and table views when
  the user toggles enabled/force in one view and switches to the other

**Files Modified:**
- `code/src/utils/gui/dag_launcher/dag_graph_items.py` -- Category colors, disabled styling
- `code/src/utils/gui/dag_launcher/dag_graph_view.py` -- Fit-all action, edge-case layout
- `code/src/utils/gui/dag_launcher/task_panel.py` -- Cross-view checkbox sync on toggle

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `test_dag_graph_node_count` -- Build graph from synthetic YAML with 5 tasks,
  verify 5 `DagTaskNode` items in scene
- [ ] `test_dag_graph_edge_count` -- Build graph from synthetic YAML with known
  dependencies, verify correct number of `DagEdge` items
- [ ] `test_dag_graph_no_overlap` -- Run layout on the 18-task processing DAG,
  verify no two nodes have overlapping bounding rects
- [ ] `test_enabled_toggle_updates_model` -- Toggle enabled checkbox on a graph
  node, verify `model.is_task_enabled()` reflects the change
- [ ] `test_force_toggle_updates_model` -- Toggle force checkbox on a graph node,
  verify `model.get_task_option("force_processing")` reflects the change
- [ ] `test_graph_single_task` -- Build graph from a 1-task config, verify it
  renders without errors
- [ ] `test_graph_no_dependencies` -- Build graph from a config where all tasks
  have empty `depends_on`, verify nodes render without edges

### Integration Tests
- [ ] `test_populate_both_views` -- Call `TaskPanel.populate()`, verify both table
  and graph have the correct number of tasks
- [ ] `test_toggle_preserves_selection` -- Select a task in graph view, toggle to
  table view, verify the same task row is selected

### Manual Verification
- [ ] Launch GUI with `analyse_workflow_processing_dag.yaml`, verify all 18 tasks
  visible as nodes with correct dependency edges
- [ ] Toggle enabled/force checkboxes in graph view, verify model updates and
  detail panel reflects changes
- [ ] Click a node, verify `TaskDetailPanel` shows the correct task options
- [ ] Zoom in/out with scroll wheel, pan with middle-click drag
- [ ] Toggle between graph and table, verify selected task is preserved
- [ ] Click "Fit All" to reset viewport
- [ ] Test with `postprocess_workflow_kinect_auto_dag.yaml` (7 tasks) for a
  smaller config

### Edge Cases
- [ ] Config with a single task and no dependencies
- [ ] Config with all tasks disabled
- [ ] Config where no tasks have `force_processing` option
- [ ] Rapidly toggling between views while checkboxes are being clicked

---

## Documentation Plan

- [ ] Update `CLAUDE.md` architecture table if `dag_graph_view` warrants a mention
- [ ] Add inline docstrings to `DagGraphView.populate()` and `DagTaskNode`
- [ ] No user guide needed (GUI is self-explanatory with toggle buttons)

---

## Rollback Plan

1. The table view is preserved as page 1 of `QStackedWidget` — reverting to
   table-only requires removing the toggle and `QStackedWidget` wrapper from
   `task_panel.py`
2. New files (`dag_graph_view.py`, `dag_graph_items.py`) can be deleted without
   affecting any other module
3. `grandalf` dependency can be removed from `requirements.txt`
4. No data migrations, no config changes, no breaking changes to `DagConfigModel`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| grandalf layout produces overlapping nodes for wide DAGs | Low | Med | Test with all 14 real DAG configs; adjust inter-node spacing if needed |
| QGraphicsProxyWidget checkboxes don't receive mouse events reliably | Low | High | Known Qt pattern; test early in Phase 1; fallback to custom-painted checkboxes |
| Checkbox stateChanged causes signal recursion (KB note) | Med | Med | Guard all handlers with `blockSignals(True/False)` per established pattern |
| grandalf API changes or is unmaintained | Low | Low | Pure Python, ~600 lines; could vendor if needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Graph Items + Layout | ~3 hours | None |
| Phase 2: TaskPanel Integration | ~2 hours | Phase 1 |
| Phase 3: Visual Polish + Edge Cases | ~2 hours | Phase 2 |

---

## References

- Analysis plan: `.claude/plans/analyse-the-dag-config-wise-jellyfish.md`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- grandalf library: `https://github.com/bdcht/grandalf` (EPL-1.0, pure Python Sugiyama layout)
- Current task panel: `code/src/utils/gui/dag_launcher/task_panel.py`
- Largest DAG config: `configs/analyse_workflow_processing_dag.yaml` (18 tasks)
