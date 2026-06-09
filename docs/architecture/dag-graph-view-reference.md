# DAG GUI Launcher — Graph View Reference Analysis

## Context

This document captures the full architecture and behaviour of the DAG Graph View
in the pipeline GUI launcher, so it can be replicated in another project. The
analysis covers the technology stack, interactive features, layout algorithm, node
rendering, edge drawing, and position persistence.

---

## 1. Technology Stack

| Component | Library | Version | Role |
|-----------|---------|---------|------|
| GUI framework | **PyQt5** | 5.15.11 | Widgets, graphics scene, event loop |
| Graph layout | **grandalf** | 0.8 (unpinned in requirements.txt) | Sugiyama hierarchical layout algorithm |
| YAML editing | **ruamel.yaml** | (round-trip mode) | Comment-preserving config read/write |
| Position persistence | **json** (stdlib) | — | Save/load node positions |

No external graph rendering library (e.g. NodeGraphQt, graphviz) is used.
The entire graph view is built from PyQt5's `QGraphicsView` / `QGraphicsScene`
primitives.

---

## 2. Core Files

| File | Class | Purpose |
|------|-------|---------|
| `code/src/utils/gui/dag_launcher/dag_graph_view.py` | `DagGraphView(QGraphicsView)` | Viewport: zoom, pan, layout, persistence |
| `code/src/utils/gui/dag_launcher/dag_graph_items.py` | `DagTaskNode(QGraphicsRectItem)` | Node: rounded rect + embedded checkboxes |
| `code/src/utils/gui/dag_launcher/dag_graph_items.py` | `DagEdge(QGraphicsPathItem)` | Edge: cubic Bezier curve + arrowhead |
| `code/src/utils/gui/dag_launcher/task_panel.py` | `TaskPanel(QWidget)` | Hosts graph/table toggle + detail panel |

---

## 3. Graph Layout (Sugiyama via grandalf)

### Algorithm flow (`DagGraphView.populate()`)

1. **Create nodes** — one `DagTaskNode` per task, sized to fit text.
2. **Build grandalf graph** — each node becomes a `Vertex` with a `_VertexView(w, h)`;
   dependencies become `GEdge(parent, child)`.
3. **Handle disconnected components** — `Graph.C` gives connected components;
   each component is laid out independently, placed side-by-side with
   `x_offset = comp_max_x + comp_max_w * 1.4 * 1.5`.
4. **Run layout** — `SugiyamaLayout(component).init_all(optimize=True); .draw()`
   computes layered (x, y) positions.
5. **Apply spacing** — positions are multiplied by `_SPACING_FACTOR = 1.4` to
   prevent overlap.
6. **Load saved positions** — if a `.layout.json` file exists, override the
   computed positions with saved ones.
7. **Fit view** — if no saved layout was loaded, `fitInView()` auto-scales
   to show all nodes.

### grandalf integration detail

```python
from grandalf.graphs import Edge as GEdge, Graph, Vertex
from grandalf.layouts import SugiyamaLayout

class _VertexView:
    """Minimal view object grandalf requires on each Vertex."""
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.xy = (0.0, 0.0)

# Attach to vertex:
v = Vertex(task_name)
v.view = _VertexView(node_width, node_height)
```

---

## 4. Node Rendering (`DagTaskNode`)

### Dimensions
- Height: **90 px** (fixed, `_NODE_H`)
- Width: `max(180, font_metrics.horizontalAdvance(task_name) + 40)` — auto-expands for long names

### Visual design
- **Shape**: Rounded rectangle, 6 px corner radius
- **Category colour coding** (5 categories):
  - `processing` → `#d0e8ff` (blue), border `#2255aa`
  - `viewer` → `#e8d0ff` (purple), border `#6622aa`
  - `viewer_required` → `#ffe0b0` (orange), border `#aa6600`
  - `viewer_support` → `#d0ffe8` (green), border `#006633`
  - `none` → `#f0f0f0` (grey), border `#666666`
- **Disabled state**: background `#e8e8e8`, border `#888888`, opacity 55%
- **Selection**: orange border (`#ff8800`, 2 px pen)
- **Category badge**: 10x10 px filled square at top-right corner

### Embedded widgets (via `QGraphicsProxyWidget`)
- Bold task name label (13 pt)
- "Enabled" checkbox
- "Force" checkbox (only if the task has `force_processing` option)

### Flags
```python
self.setFlag(QGraphicsItem.ItemIsSelectable, True)
self.setFlag(QGraphicsItem.ItemIsMovable, True)
self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
```

### Signals (via nested `_Signals(QObject)`)
- `node_clicked(str)` — on `mousePressEvent`
- `enabled_changed(str, bool)` — on checkbox toggle
- `force_changed(str, bool)` — on checkbox toggle
- `position_changed(str, float, float)` — on `itemChange(ItemPositionHasChanged)`

---

## 5. Edge Rendering (`DagEdge`)

- **Type**: `QGraphicsPathItem`
- **Path**: Cubic Bezier from source node right-centre to target node left-centre
  - Control points offset by **60 px** horizontally
- **Arrowhead**: 8 px triangular filled polygon at the target tip
- **Colour**: `#555555`, 1.5 px pen
- **Dynamic update**: `update_path()` is called whenever a connected node moves

```python
src_pt = QPointF(src_rect.right(), src_rect.center().y())
tgt_pt = QPointF(tgt_rect.left(), tgt_rect.center().y())
c1 = QPointF(src_pt.x() + 60, src_pt.y())
c2 = QPointF(tgt_pt.x() - 60, tgt_pt.y())
path = QPainterPath(src_pt)
path.cubicTo(c1, c2, tgt_pt)
```

---

## 6. Interaction Model

| Interaction | Implementation |
|-------------|----------------|
| **Zoom** | `wheelEvent` — scale factor 1.15x (or 1/1.15x), anchored under mouse |
| **Pan** | Middle-click drag — manual scrollbar adjustment in `mouseMoveEvent` |
| **Move node** | Left-click drag — `ItemIsMovable` flag; `itemChange` emits `position_changed` |
| **Select node** | Left-click — `mousePressEvent` emits `node_clicked`, triggers detail panel |
| **Fit all** | `Ctrl+0` shortcut — `fitInView(scene.itemsBoundingRect(), KeepAspectRatio)` |
| **Toggle enabled** | Checkbox embedded in node — emits `enabled_changed` |
| **Toggle force** | Checkbox embedded in node — emits `force_changed` |

### Viewport settings
```python
self.setRenderHint(QPainter.Antialiasing)
self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
self.setDragMode(QGraphicsView.NoDrag)
```

---

## 7. Position Persistence

### File format
Sibling to the DAG YAML: `{dag_config_stem}.layout.json`

```json
{
  "task_name_a": [123.45, 67.89],
  "task_name_b": [234.56, 78.90]
}
```

### Save flow
1. Node is dragged → `itemChange(ItemPositionHasChanged)` → `position_changed` signal
2. `DagGraphView._on_node_moved()` updates all edge paths + restarts 800 ms debounce timer
3. Timer fires → `_save_layout()` collects `{name: [x, y]}` from all nodes → writes JSON

### Load flow
1. After `populate()` computes Sugiyama positions and places nodes, `_load_layout()` runs
2. If `.layout.json` exists and is valid, override node positions with saved values
3. Return `True` (skip `fitInView`) if any positions were loaded; `False` otherwise

### Key design decisions
- **Debounce (800 ms)** prevents excessive disk writes during drag operations
- **Positions are absolute scene coordinates** (not offsets from layout)
- **Unknown task names** in saved layout are silently ignored (safe for renamed/removed tasks)
- **No explicit "save layout" button** — positions auto-persist on every move

---

## 8. Task Panel — Graph/Table Toggle

`TaskPanel` hosts both views in a `QStackedWidget`:
- **Index 0**: `DagGraphView` (default)
- **Index 1**: `QTableWidget` (3 columns: Name, Enabled/Force checkboxes, Depends On)

Toggle buttons ("Graph View" / "Table View") switch the stack index.
A "Fit All" button calls `graph_view.fit_all()`.
Both views share a `TaskDetailPanel` below (vertical splitter, stretch 2:5).

---

## 9. Minimal Reproduction Checklist

To replicate this graph view in another PyQt5 project:

1. **Install**: `pip install PyQt5 grandalf`
2. **Copy/adapt three classes**:
   - `DagGraphView(QGraphicsView)` — viewport with zoom, pan, layout, persistence
   - `DagTaskNode(QGraphicsRectItem)` — node with embedded widgets
   - `DagEdge(QGraphicsPathItem)` — Bezier edge with arrowhead
3. **Provide a data model** with methods: `get_task_names()`, `get_task_dependencies(name)`, `is_task_enabled(name)`, `get_task_option(name, key)`
4. **Call `graph_view.populate(model)`** to render
5. **Position persistence**: layout files are saved next to the config file using `Path.with_suffix(".layout.json")`

### Adaptation points
- Node dimensions (`_NODE_H`, `_MIN_NODE_W`)
- Category colours (`_CATEGORY_COLORS`)
- Embedded widgets (the checkboxes are domain-specific — replace with your own)
- Spacing factor (`_SPACING_FACTOR = 1.4`)
- Debounce interval (800 ms)
- Edge control point offset (60 px)
