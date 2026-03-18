"""Centre panel — QTableWidget-based task list for DAG config editing."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.gui.dag_launcher.yaml_edit_dialog import YamlEditDialog

from utils.pipeline.dag_config_model import DagConfigModel

# Colour used for disabled / N-A cells.
_GREY_BG = QColor("#e8e8e8")
_GREY_FG = QColor("#888888")
# Colour for clickable complex-value cells (list / dict).
_COMPLEX_FG = QColor("#336699")


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


def _is_profile_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of dicts each containing a 'method' key.

    This heuristic identifies extraction_profiles / clustering_profiles containers
    while excluding plain booleans, lists, and generic nested dicts that do not
    follow the profile convention.
    """
    if not isinstance(val, dict) or not val:
        return False
    return all(isinstance(v, dict) and "method" in v for v in val.values())


def _profile_header(option_key: str, profile_name: str) -> str:
    """Return a column header like 'Extract: Max' for a profile column."""
    prefix = option_key.replace("_profiles", "").replace("_", " ").title()
    return f"{prefix}: {profile_name.replace('_', ' ').title()}"


def _col_header(spec: tuple) -> str:
    """Return the column header for a column spec tuple."""
    if spec[0] == "simple":
        return _option_header(spec[1])
    return _profile_header(spec[1], spec[2])


def _preview_text(val: Any) -> str:
    """Return a short display string for a list or dict option value."""
    if isinstance(val, list):
        items = [str(v) for v in val[:3]]
        preview = "[" + ", ".join(items) + (", ..." if len(val) > 3 else "") + "]"
        return f"{preview} ({len(val)} items)"
    if isinstance(val, dict):
        keys = list(val.keys())[:3]
        preview = "{" + ", ".join(str(k) for k in keys) + (", ..." if len(val) > 3 else "") + "}"
        return f"{preview} ({len(val)} keys)"
    return str(val)


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

        # _column_spec: ordered list of column descriptors.
        # Each entry is ("simple", key) or ("profile", key, profile_name).
        self._column_spec: list[tuple] = []
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
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.itemChanged.connect(self._on_item_changed)

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

        # --- Discover column spec in first-seen order across all tasks ---
        # Profile-container options (dicts of dicts with 'method' keys) are
        # expanded into one ("profile", key, profile_name) entry per profile;
        # all other options yield a single ("simple", key) entry.
        seen_specs: dict[tuple, None] = {}
        for name in task_names:
            for key, val in model.get_task_options(name).items():
                if _is_profile_dict(val):
                    for profile_name in val:
                        seen_specs[("profile", key, profile_name)] = None
                else:
                    seen_specs[("simple", key)] = None
        self._column_spec = list(seen_specs.keys())

        # --- Configure columns ---
        n_option_cols = len(self._column_spec)
        n_cols = 2 + n_option_cols + 1  # Name, Enabled, <options>, Depends On
        depends_col = 2 + n_option_cols

        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(len(task_names))
        self._table.setColumnCount(n_cols)

        headers = (
            ["Task Name", "Enabled"]
            + [_col_header(spec) for spec in self._column_spec]
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
            for col_offset, spec in enumerate(self._column_spec):
                col = 2 + col_offset
                if spec[0] == "simple":
                    opt_key = spec[1]
                    if opt_key in options:
                        val: Any = options[opt_key]
                        if isinstance(val, bool):
                            container_opt, cb_opt = _centred_checkbox(val)
                            self._checkboxes[name][opt_key] = cb_opt
                            cb_opt.stateChanged.connect(
                                self._make_option_handler(name, opt_key, cb_opt)
                            )
                            self._table.setCellWidget(row, col, container_opt)
                        elif isinstance(val, (int, float, str)):
                            # Tier 1 — inline editable scalar
                            item_val = QTableWidgetItem(str(val))
                            item_val.setFlags(
                                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
                            )
                            item_val.setData(Qt.UserRole, val)
                            item_val.setToolTip("Double-click to edit")
                            self._table.setItem(row, col, item_val)
                        else:
                            # Tier 2 — click to open YAML editor
                            item_val = QTableWidgetItem(_preview_text(val))
                            item_val.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                            item_val.setForeground(QBrush(_COMPLEX_FG))
                            item_val.setData(Qt.UserRole, ("complex", opt_key))
                            item_val.setToolTip("Click to edit in YAML editor")
                            self._table.setItem(row, col, item_val)
                    else:
                        item_na = QTableWidgetItem()
                        item_na.setFlags(Qt.NoItemFlags)
                        item_na.setBackground(QBrush(_GREY_BG))
                        self._table.setItem(row, col, item_na)
                else:  # ("profile", key, profile_name)
                    _, opt_key, profile_name = spec
                    container = options.get(opt_key)
                    if isinstance(container, dict) and profile_name in container:
                        is_enabled = model.get_profile_enabled(name, opt_key, profile_name)
                        container_opt, cb_opt = _centred_checkbox(is_enabled)
                        self._checkboxes[name][(opt_key, profile_name)] = cb_opt
                        cb_opt.stateChanged.connect(
                            self._make_profile_handler(name, opt_key, profile_name, cb_opt)
                        )
                        self._table.setCellWidget(row, col, container_opt)
                    else:
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

    def _make_profile_handler(self, task_name: str, opt_key: str, profile_name: str, cb: QCheckBox):
        """Return a slot that writes a profile enabled state to the model."""
        def _handler(_state: int) -> None:
            if self._model is None:
                return
            self._model.set_profile_enabled(task_name, opt_key, profile_name, cb.isChecked())
            self.task_changed.emit()
        return _handler

    # ------------------------------------------------------------------
    # Cell click — dependency navigation
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if self._model is None:
            return
        depends_col = 2 + len(self._column_spec)

        if col == depends_col:
            item = self._table.item(row, col)
            if item is None:
                return
            names = [n.strip() for n in item.text().split(",") if n.strip()]
            if names:
                self._scroll_to_task(names[0])
            return

        # Complex value cell — open YAML editor
        if 2 <= col < depends_col:
            item = self._table.item(row, col)
            if item is None:
                return
            data = item.data(Qt.UserRole)
            if not isinstance(data, tuple) or data[0] != "complex":
                return
            opt_key = data[1]
            task_name = self._row_task[row]
            current_value = self._model.get_task_option(task_name, opt_key)
            dlg = YamlEditDialog(task_name, opt_key, current_value, self)
            if dlg.exec_() == QDialog.Accepted:
                new_value = dlg.get_value()
                self._model.set_task_option(task_name, opt_key, new_value)
                item.setText(_preview_text(new_value))
                self.task_changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Validate and commit an inline scalar edit to the model."""
        if self._model is None:
            return
        row = item.row()
        col = item.column()
        depends_col = 2 + len(self._column_spec)
        if not (2 <= col < depends_col):
            return
        original_val = item.data(Qt.UserRole)
        if original_val is None or isinstance(original_val, tuple):
            return  # not a scalar cell
        original_type = type(original_val)
        new_text = item.text()
        try:
            new_val = original_type(new_text)
        except (ValueError, TypeError):
            # Revert without triggering another itemChanged
            self._table.blockSignals(True)
            item.setText(str(original_val))
            self._table.blockSignals(False)
            return
        task_name = self._row_task[row]
        spec = self._column_spec[col - 2]
        opt_key = spec[1]
        self._model.set_task_option(task_name, opt_key, new_val)
        item.setData(Qt.UserRole, new_val)
        self.task_changed.emit()

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
