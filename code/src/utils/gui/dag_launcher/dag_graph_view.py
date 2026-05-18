"""DAG graph view: QGraphicsView subclass with grandalf Sugiyama layout."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QKeySequence
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QShortcut

logger = logging.getLogger(__name__)

from grandalf.graphs import Edge as GEdge
from grandalf.graphs import Graph
from grandalf.graphs import Vertex
from grandalf.layouts import SugiyamaLayout

from utils.gui.dag_launcher.dag_graph_items import DagEdge, DagTaskNode
from utils.pipeline.dag_config_model import DagConfigModel

_SPACING_FACTOR = 1.4


class _VertexView:
    """Minimal view object required by grandalf layout engine."""

    def __init__(self, w: float, h: float) -> None:
        self.w = w
        self.h = h
        self.xy = (0.0, 0.0)


class DagGraphView(QGraphicsView):
    """Interactive DAG graph view with zoom, pan, and embedded checkboxes.

    Signals
    -------
    node_clicked(task_name)
    enabled_changed(task_name, new_value)
    force_changed(task_name, new_value)
    """

    node_clicked = pyqtSignal(str)
    enabled_changed = pyqtSignal(str, bool)
    force_changed = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)

        self._nodes: dict[str, DagTaskNode] = {}
        self._edges: list[DagEdge] = []
        self._panning = False
        self._pan_start = QPoint()
        self._config_path: Path | None = None

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_layout)

        fit_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        fit_shortcut.activated.connect(self.fit_all)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        """Clear and rebuild the graph from *model*."""
        self._save_timer.stop()
        self._scene.clear()
        self._nodes = {}
        self._edges = []
        self._config_path = model.path

        task_names = model.get_task_names()
        if not task_names:
            return

        # Create nodes first so each one measures its own text-fitted width.
        pre_nodes: dict[str, DagTaskNode] = {
            name: DagTaskNode(name, model, category=model._get_task(name).get("category", "none"))
            for name in task_names
        }

        vertices: dict[str, Vertex] = {}
        for name, node in pre_nodes.items():
            v = Vertex(name)
            r = node.rect()
            v.view = _VertexView(r.width(), r.height())
            vertices[name] = v

        gedges: list[GEdge] = []
        for name in task_names:
            for dep in model.get_task_dependencies(name):
                if dep in vertices:
                    gedges.append(GEdge(vertices[dep], vertices[name]))

        g = Graph(list(vertices.values()), gedges)

        x_offset = 0.0
        for component in g.C:
            layout = SugiyamaLayout(component)
            layout.init_all(optimize=True)
            layout.draw()

            comp_min_x = min(v.view.xy[0] for v in component.sV)
            for v in component.sV:
                raw_x, raw_y = v.view.xy
                shifted_x = (raw_x - comp_min_x + x_offset) * _SPACING_FACTOR
                shifted_y = raw_y * _SPACING_FACTOR
                v.view.xy = (shifted_x, shifted_y)

            comp_max_x = max(v.view.xy[0] for v in component.sV)
            comp_max_w = max(pre_nodes[v.data].rect().width() for v in component.sV)
            x_offset = comp_max_x + comp_max_w * _SPACING_FACTOR * 1.5

        for name in task_names:
            node = pre_nodes[name]
            x, y = vertices[name].view.xy
            node.setPos(x, y)
            self._scene.addItem(node)
            self._nodes[name] = node

            node.signals.node_clicked.connect(self.node_clicked)
            node.signals.enabled_changed.connect(self.enabled_changed)
            node.signals.force_changed.connect(self.force_changed)
            node.signals.position_changed.connect(self._on_node_moved)

        for name in task_names:
            for dep in model.get_task_dependencies(name):
                if dep in self._nodes:
                    edge = DagEdge(self._nodes[dep], self._nodes[name])
                    self._scene.addItem(edge)
                    self._edges.append(edge)

        if not self._load_layout():
            self.fit_all()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_all(self) -> None:
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def select_task(self, task_name: str) -> None:
        for node in self._nodes.values():
            node.setSelected(False)
        if task_name in self._nodes:
            self._nodes[task_name].setSelected(True)

    def update_from_model(self, model: DagConfigModel) -> None:
        for node in self._nodes.values():
            node.update_from_model(model)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_path(self) -> Path | None:
        if self._config_path is None:
            return None
        return self._config_path.with_suffix(".layout.json")

    def _load_layout(self) -> bool:
        """Apply saved positions. Returns True if layout was loaded."""
        path = self._layout_path()
        if path is None or not path.exists():
            return False
        try:
            saved: dict[str, list[float]] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read layout file %s", path)
            return False
        loaded_any = False
        for name, (x, y) in saved.items():
            if name in self._nodes:
                self._nodes[name].setPos(x, y)
                loaded_any = True
        return loaded_any

    def _save_layout(self) -> None:
        path = self._layout_path()
        if path is None:
            return
        data = {name: [node.pos().x(), node.pos().y()] for name, node in self._nodes.items()}
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save layout to %s", path)

    def _on_node_moved(self, _task_name: str, _x: float, _y: float) -> None:
        for edge in self._edges:
            edge.update_path()
        self._save_timer.start()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    # ------------------------------------------------------------------
    # Middle-click pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
