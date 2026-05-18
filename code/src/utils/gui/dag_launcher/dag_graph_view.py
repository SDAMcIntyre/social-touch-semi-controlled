"""DAG graph view: QGraphicsView subclass with grandalf Sugiyama layout."""

from __future__ import annotations

from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QKeySequence
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QShortcut

from grandalf.graphs import Edge as GEdge
from grandalf.graphs import Graph
from grandalf.graphs import Vertex
from grandalf.layouts import SugiyamaLayout

from utils.gui.dag_launcher.dag_graph_items import DagEdge, DagTaskNode
from utils.pipeline.dag_config_model import DagConfigModel

_NODE_W = 180
_NODE_H = 70
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
        self._panning = False
        self._pan_start = QPoint()

        fit_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        fit_shortcut.activated.connect(self.fit_all)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        """Clear and rebuild the graph from *model*."""
        self._scene.clear()
        self._nodes = {}

        task_names = model.get_task_names()
        if not task_names:
            return

        vertices: dict[str, Vertex] = {}
        for name in task_names:
            v = Vertex(name)
            v.view = _VertexView(_NODE_W, _NODE_H)
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
            x_offset = comp_max_x + _NODE_W * _SPACING_FACTOR * 1.5

        for name in task_names:
            category = model._get_task(name).get("category", "none")
            node = DagTaskNode(name, model, category=category)
            x, y = vertices[name].view.xy
            node.setPos(x, y)
            self._scene.addItem(node)
            self._nodes[name] = node

            node.signals.node_clicked.connect(self.node_clicked)
            node.signals.enabled_changed.connect(self.enabled_changed)
            node.signals.force_changed.connect(self.force_changed)

        for name in task_names:
            for dep in model.get_task_dependencies(name):
                if dep in self._nodes:
                    edge = DagEdge(self._nodes[dep], self._nodes[name])
                    self._scene.addItem(edge)

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
