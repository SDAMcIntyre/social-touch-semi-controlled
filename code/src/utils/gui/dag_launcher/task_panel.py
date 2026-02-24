"""Centre panel — QTableWidget-based task list for DAG config editing."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.pipeline.dag_config_model import DagConfigModel

# Colour used for disabled / N-A cells and non-boolean option text.
_GREY_BG = QColor("#e8e8e8")
_GREY_FG = QColor("#888888")


def _centred_checkbox(checked: bool) -> tuple[QWidget, QCheckBox]:
    """Return a (container_widget, checkbox) pair centred inside the container."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(Qt.AlignCenter)
    cb = QCheckBox()
    cb.setChecked(checked)
    layout.addWidget(cb)
    return container, cb


def _option_header(key: str) -> str:
    """Convert an option key to a human-readable column header."""
    return key.replace("_", " ").title()


class TaskPanel(QWidget):
    """Panel that renders DAG tasks in a :class:`QTableWidget`.

    Public interface
    ----------------
    populate(model)  — rebuild the table from a :class:`DagConfigModel`.
    task_changed     — signal emitted whenever any checkbox is toggled.
    """

    task_changed = pyqtSignal()

    # Column indices for fixed columns (option columns are inserted between
    # _COL_ENABLED and _COL_DEPENDS dynamically).
    _COL_NAME = 0
    _COL_ENABLED = 1
    # option columns: 2 … 2+len(option_keys)-1
    # _COL_DEPENDS = 2 + len(option_keys)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: DagConfigModel | None = None

        # _option_keys: ordered list of all option keys across all tasks.
        self._option_keys: list[str] = []
        # Maps (row, task_name) for reverse lookup.
        self._row_task: list[str] = []
        # Checkbox references: task_name → {"enabled": QCheckBox, opt_key: QCheckBox, …}
        self._checkboxes: dict[str, dict[str, QCheckBox]] = {}

        # Layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("Tasks")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.cellClicked.connect(self._on_cell_clicked)

        group_layout.addWidget(self._table)
        outer_layout.addWidget(group)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        """Clear and rebuild the entire table from *model*."""
        self._model = model
        self._checkboxes.clear()
        self._row_task.clear()

        task_names = model.get_task_names()

        # --- Discover option keys in first-seen order across all tasks ---
        seen: dict[str, None] = {}  # ordered set via dict
        for name in task_names:
            for key in model.get_task_options(name):
                seen[key] = None
        self._option_keys = list(seen.keys())

        # --- Configure columns ---
        n_option_cols = len(self._option_keys)
        n_cols = 2 + n_option_cols + 1  # Name, Enabled, <options>, Depends On
        depends_col = 2 + n_option_cols

        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(len(task_names))
        self._table.setColumnCount(n_cols)

        headers = (
            ["Task Name", "Enabled"]
            + [_option_header(k) for k in self._option_keys]
            + ["Depends On"]
        )
        self._table.setHorizontalHeaderLabels(headers)

        # Resize mode: interactive for all but the last (stretch).
        for col in range(n_cols - 1):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )

        # --- Populate rows ---
        for row, name in enumerate(task_names):
            self._row_task.append(name)
            options = model.get_task_options(name)
            dependencies = model.get_task_dependencies(name)
            self._checkboxes[name] = {}

            # Column 0 — Task Name
            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._table.setItem(row, self._COL_NAME, item_name)

            # Column 1 — Enabled checkbox
            container, cb_enabled = _centred_checkbox(model.is_task_enabled(name))
            self._checkboxes[name]["enabled"] = cb_enabled
            cb_enabled.stateChanged.connect(
                self._make_enabled_handler(name, cb_enabled)
            )
            self._table.setCellWidget(row, self._COL_ENABLED, container)

            # Option columns
            for col_offset, opt_key in enumerate(self._option_keys):
                col = 2 + col_offset
                if opt_key in options:
                    val: Any = options[opt_key]
                    if isinstance(val, bool):
                        container_opt, cb_opt = _centred_checkbox(val)
                        self._checkboxes[name][opt_key] = cb_opt
                        cb_opt.stateChanged.connect(
                            self._make_option_handler(name, opt_key, cb_opt)
                        )
                        self._table.setCellWidget(row, col, container_opt)
                    else:
                        item_val = QTableWidgetItem(str(val))
                        item_val.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        item_val.setForeground(QBrush(_GREY_FG))
                        self._table.setItem(row, col, item_val)
                else:
                    # Task has no value for this option — disabled grey cell.
                    item_na = QTableWidgetItem()
                    item_na.setFlags(Qt.NoItemFlags)
                    item_na.setBackground(QBrush(_GREY_BG))
                    self._table.setItem(row, col, item_na)

            # Depends On column
            dep_text = ", ".join(dependencies) if dependencies else ""
            item_dep = QTableWidgetItem(dep_text)
            item_dep.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if dep_text:
                item_dep.setToolTip("Click to scroll to the first dependency")
            self._table.setItem(row, depends_col, item_dep)

        self._table.blockSignals(False)

    # ------------------------------------------------------------------
    # Checkbox handler factories
    # ------------------------------------------------------------------

    def _make_enabled_handler(self, task_name: str, cb: QCheckBox):
        """Return a slot that writes the enabled state to the model."""
        def _handler(_state: int) -> None:
            if self._model is None:
                return
            self._model.set_task_enabled(task_name, cb.isChecked())
            self.task_changed.emit()
        return _handler

    def _make_option_handler(self, task_name: str, opt_key: str, cb: QCheckBox):
        """Return a slot that writes a boolean option value to the model."""
        def _handler(_state: int) -> None:
            if self._model is None:
                return
            self._model.set_task_option(task_name, opt_key, cb.isChecked())
            self.task_changed.emit()
        return _handler

    # ------------------------------------------------------------------
    # Cell click — dependency navigation
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if self._model is None:
            return
        depends_col = 2 + len(self._option_keys)
        if col != depends_col:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        names = [n.strip() for n in item.text().split(",") if n.strip()]
        if not names:
            return
        self._scroll_to_task(names[0])

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
