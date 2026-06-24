"""Centre panel — compact 3-column task list with per-task detail panel."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.gui.dag_launcher.dag_graph_view import DagGraphView
from utils.gui.dag_launcher.task_detail_panel import TaskDetailPanel
from utils.pipeline.dag_config_model import DagConfigModel


class TaskPanel(QWidget):
    """Panel that renders DAG tasks as a compact 3-column list with a detail panel."""

    task_changed = pyqtSignal()

    _COL_NAME = 0
    _COL_ENABLED = 1
    _COL_DEPENDS = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: DagConfigModel | None = None
        self._row_task: list[str] = []
        self._selected_task: str | None = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("Tasks")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)

        toggle_bar = QHBoxLayout()
        self._btn_graph = QPushButton("Graph View")
        self._btn_table = QPushButton("Table View")
        self._btn_graph.setCheckable(True)
        self._btn_table.setCheckable(True)
        self._btn_graph.setChecked(True)
        toggle_bar.addWidget(self._btn_graph)
        toggle_bar.addWidget(self._btn_table)
        self._btn_fit = QPushButton("Fit All")
        toggle_bar.addWidget(self._btn_fit)
        toggle_bar.addStretch()
        group_layout.addLayout(toggle_bar)

        self._splitter = QSplitter(Qt.Vertical)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.currentCellChanged.connect(self._on_current_cell_changed)

        self._graph_view = DagGraphView()
        self._graph_view.node_clicked.connect(self._on_graph_node_clicked)
        self._graph_view.enabled_changed.connect(self._on_graph_enabled_changed)
        self._graph_view.force_changed.connect(self._on_graph_force_changed)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._graph_view)
        self._stack.addWidget(self._table)
        self._stack.setCurrentIndex(0)

        self._detail = TaskDetailPanel()
        self._detail.task_changed.connect(self.task_changed)

        self._splitter.addWidget(self._stack)
        self._splitter.addWidget(self._detail)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 5)

        self._btn_graph.clicked.connect(self._show_graph_view)
        self._btn_table.clicked.connect(self._show_table_view)
        self._btn_fit.clicked.connect(self._graph_view.fit_all)

        group_layout.addWidget(self._splitter)
        outer_layout.addWidget(group)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        """Clear and rebuild the table from *model*."""
        self._model = model
        self._row_task.clear()
        self._selected_task = None

        self._graph_view.populate(model)

        task_names = model.get_task_names()

        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(len(task_names))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Task Name", "Enabled", "Depends On"])

        for col in range(2):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )

        for row, name in enumerate(task_names):
            self._row_task.append(name)

            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._table.setItem(row, self._COL_NAME, item_name)

            container = QWidget()
            cell_layout = QHBoxLayout(container)
            cell_layout.setContentsMargins(4, 0, 4, 0)
            cell_layout.setAlignment(Qt.AlignVCenter)

            cb_enabled = QCheckBox()
            cb_enabled.setChecked(model.is_task_enabled(name))
            cb_enabled.stateChanged.connect(self._make_enabled_handler(name, cb_enabled))
            cell_layout.addWidget(cb_enabled)

            force_val = model.get_task_option(name, "force_processing")
            if force_val is not None:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setFrameShadow(QFrame.Sunken)
                sep.setFixedWidth(10)
                cell_layout.addWidget(sep)
                cb_force = QCheckBox("Force")
                cb_force.setChecked(bool(force_val))
                cb_force.setToolTip("Force processing (ignore cached results)")
                cb_force.stateChanged.connect(self._make_force_handler(name, cb_force))
                cell_layout.addWidget(cb_force)

            self._table.setCellWidget(row, self._COL_ENABLED, container)

            dep_text = ", ".join(model.get_task_dependencies(name))
            item_dep = QTableWidgetItem(dep_text)
            item_dep.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if dep_text:
                item_dep.setToolTip("Click to scroll to the first dependency")
            self._table.setItem(row, self._COL_DEPENDS, item_dep)

        self._table.blockSignals(False)

        if task_names:
            self._table.selectRow(0)
            # _on_current_cell_changed fires from selectRow and calls show_task

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _show_graph_view(self) -> None:
        self._btn_graph.setChecked(True)
        self._btn_table.setChecked(False)
        self._stack.setCurrentIndex(0)
        if self._model is not None:
            self._graph_view.update_from_model(self._model)
        if self._selected_task is not None:
            self._graph_view.select_task(self._selected_task)

    def _sync_table_checkboxes(self) -> None:
        if self._model is None:
            return
        for row, name in enumerate(self._row_task):
            container = self._table.cellWidget(row, self._COL_ENABLED)
            if container is None:
                continue
            for cb in container.findChildren(QCheckBox):
                cb.blockSignals(True)
                if cb.text() == "Force":
                    val = self._model.get_task_option(name, "force_processing")
                    if val is not None:
                        cb.setChecked(bool(val))
                else:
                    cb.setChecked(self._model.is_task_enabled(name))
                cb.blockSignals(False)

    def _show_table_view(self) -> None:
        self._btn_graph.setChecked(False)
        self._btn_table.setChecked(True)
        self._stack.setCurrentIndex(1)
        self._sync_table_checkboxes()
        if self._selected_task is not None and self._selected_task in self._row_task:
            row = self._row_task.index(self._selected_task)
            self._table.selectRow(row)

    # ------------------------------------------------------------------
    # Graph view signal handlers
    # ------------------------------------------------------------------

    def _on_graph_node_clicked(self, task_name: str) -> None:
        if self._model is None:
            return
        self._detail.show_task(self._model, task_name)
        self._selected_task = task_name

    def _on_graph_enabled_changed(self, task_name: str, enabled: bool) -> None:
        if self._model is None:
            return
        self._model.set_task_enabled(task_name, enabled)
        self.task_changed.emit()

    def _on_graph_force_changed(self, task_name: str, value: bool) -> None:
        if self._model is None:
            return
        self._model.set_task_option(task_name, "force_processing", value)
        self.task_changed.emit()

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    def _make_enabled_handler(self, task_name: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None:
                return
            self._model.set_task_enabled(task_name, cb.isChecked())
            self.task_changed.emit()
        return _handler

    def _make_force_handler(self, task_name: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None:
                return
            self._model.set_task_option(task_name, "force_processing", cb.isChecked())
            self.task_changed.emit()
        return _handler

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_current_cell_changed(
        self, current_row: int, _cc: int, _pr: int, _pc: int
    ) -> None:
        if self._model is None or current_row < 0 or current_row >= len(self._row_task):
            return
        self._selected_task = self._row_task[current_row]
        self._detail.show_task(self._model, self._selected_task)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Handle depends-on click to scroll to the first dependency."""
        if self._model is None or col != self._COL_DEPENDS:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        names = [n.strip() for n in item.text().split(",") if n.strip()]
        if names:
            self._scroll_to_task(names[0])

    # ------------------------------------------------------------------
    # Scroll helper
    # ------------------------------------------------------------------

    def _scroll_to_task(self, task_name: str) -> None:
        """Scroll the table so that *task_name*'s row is visible."""
        try:
            target_row = self._row_task.index(task_name)
        except ValueError:
            return
        target_item = self._table.item(target_row, self._COL_NAME)
        if target_item is not None:
            self._table.scrollToItem(target_item, QAbstractItemView.PositionAtCenter)
            self._table.selectRow(target_row)
