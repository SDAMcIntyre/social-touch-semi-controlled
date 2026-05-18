# Handoff Note — `feature/dag-graph-view`

**Date:** 2026-05-18
**Branch:** `feature/dag-graph-view`
**Base branch:** `feature/slim-forearm-projection`

---

## What was built

A full DAG graph view was added to the Tasks tab of the pipeline launcher GUI.
It replaces the flat `QTableWidget` as the default view (table still reachable
via a toggle button).

### Committed at `c3764ff`

| File | Change |
|---|---|
| `code/src/utils/gui/dag_launcher/dag_graph_items.py` | New — `DagTaskNode` (rounded rect node with embedded checkboxes) + `DagEdge` (Bézier arrow) |
| `code/src/utils/gui/dag_launcher/dag_graph_view.py` | New — `DagGraphView(QGraphicsView)` with grandalf Sugiyama layout, zoom (scroll wheel), pan (middle-click drag), Ctrl+0 fit |
| `code/src/utils/gui/dag_launcher/task_panel.py` | Modified — toggle bar (Graph / Table / Fit All buttons), `QStackedWidget` wrapper, cross-view checkbox sync |
| `code/src/utils/gui/dag_launcher/dag_task_widget.py` | Deleted (was a one-line stub) |
| `requirements.txt` | `grandalf` added |

### Post-commit polish (uncommitted, in working tree)

| File | Change |
|---|---|
| `dag_graph_items.py` | Node width text-fitted per node (QFontMetrics, 13pt bold); `ItemIsMovable` enabled; `position_changed` signal; `itemChange` emits on drag |
| `dag_graph_view.py` | Nodes created before grandalf so layout uses real widths; drag → all edges call `update_path()` via `_on_node_moved`; layout saved to `<dag>.layout.json` (800ms debounce) and loaded on next `populate()`; `self._edges` list stored for path refresh |
| `launcher_window.py` | `setStretchFactor(0, 0)` — workflow column no longer gets extra stretch |
| `workflow_selector.py` | `_set_minimum_width()` — enforces minimum width from longest button text via `QFontMetrics(QApplication.font())`; splitter can no longer squeeze text away |
| `.gitignore` | Added `*.layout.json` |

These post-commit changes need to be committed before `/plan-finish`.

Suggested commit message:
```
refactor(dag-graph-view): text-fit nodes, drag with live edges, layout persistence, workflow selector min-width
```

Files to stage:
```
.gitignore
code/src/utils/gui/dag_launcher/dag_graph_items.py
code/src/utils/gui/dag_launcher/dag_graph_view.py
code/src/utils/gui/dag_launcher/launcher_window.py
code/src/utils/gui/dag_launcher/workflow_selector.py
```

---

## Known issues to fix

### 1. `grandalf` not installed
The user hit `No module named 'grandalf'`. One-time setup:
```bash
conda activate social-touch-env
pip install grandalf
```
Not a code fix — just environment setup.

### 2. Workflow selector minimum width may be slightly off
`QFontMetrics(QApplication.font())` measures with the app default font, but
buttons render with a stylesheet (`padding: 4px 10px`). The `+40` constant
(`10+10` btn padding + `6+6` group box margins + `4+4` border) was estimated.
May need tuning on Windows 11 if buttons are visually clipped or have excess
whitespace. Fix is in `workflow_selector.py::_set_minimum_width()`.

### 3. Arrowhead may lag on drag
Edges call `update_path()` inside `_on_node_moved` (triggered via
`ItemPositionHasChanged`). The arrowhead is also repainted in
`DagEdge.paint()` using `sceneBoundingRect()` of the target — this
recomputes every frame so it should track correctly. If the arrowhead
still lags visually, add `self.update()` at the end of `DagEdge.update_path()`.

### 4. Layout save not firing (if reported)
`_save_timer.start()` with no argument reuses the interval set in `__init__`
(800ms, single-shot). If layout is not persisting between sessions, check
that `_save_layout()` is being reached and that the config path is not `None`.
The layout file is written to `<dag_yaml_stem>.layout.json` next to the YAML.

---

## How to continue

```bash
# Install grandalf if not yet done
conda activate social-touch-env
pip install grandalf

# Launch the GUI to test
python code/scripts/launch_pipeline_gui.py
```

When all issues are resolved, finish the feature:
```
/plan-finish dag-graph-view
```

The plan document is at:
`docs/development/plans/active/dag-graph-view.md`
