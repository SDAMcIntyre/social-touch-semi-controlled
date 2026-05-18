"""Workflow selector widget — toggle buttons driven by launcher.yaml entries."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils.gui.dag_launcher.launcher_config import WorkflowEntry


class WorkflowSelector(QWidget):
    """Left-column widget with exclusive toggle buttons for each workflow entry."""

    workflow_changed = pyqtSignal(object)  # emits WorkflowEntry

    def __init__(self, entries: list[WorkflowEntry], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[WorkflowEntry] = []
        self._buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Workflow")
        self._btn_layout = QVBoxLayout(group_box)
        self._btn_layout.setContentsMargins(6, 4, 6, 4)
        self._btn_layout.setSpacing(4)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.idClicked.connect(self._on_button_clicked)

        layout.addWidget(group_box)

        self._populate(entries)

    # ------------------------------------------------------------------

    def _populate(self, entries: list[WorkflowEntry]) -> None:
        """Build buttons grouped by category from *entries*."""
        self._entries = list(entries)
        current_category: str | None = None
        for i, entry in enumerate(self._entries):
            if entry.category != current_category:
                current_category = entry.category
                header = QLabel(current_category)
                header.setStyleSheet(
                    "QLabel { font-size: 10px; font-weight: bold; color: #888888;"
                    " margin-top: 6px; margin-bottom: 1px; }"
                )
                self._btn_layout.addWidget(header)
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFrameShadow(QFrame.Sunken)
                self._btn_layout.addWidget(sep)

            btn = QPushButton(entry.name)
            btn.setCheckable(True)
            btn.setToolTip(str(entry.dag_config) if entry.dag_config else entry.name)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 10px; }"
                "QPushButton:checked { background-color: #4a90d9; color: white; "
                "font-weight: bold; }"
            )
            self._btn_group.addButton(btn, i)
            self._btn_layout.addWidget(btn)
            self._buttons.append(btn)

        self._btn_layout.addStretch()
        self._set_minimum_width()

    def _set_minimum_width(self) -> None:
        if not self._buttons:
            return
        fm = QFontMetrics(QApplication.font())
        max_text_w = max(fm.horizontalAdvance(btn.text()) for btn in self._buttons)
        # button padding (10px each side) + group box content margins (6+6) + group box border (~4+4)
        self.setMinimumWidth(max_text_w + 40)

    def _on_button_clicked(self, btn_id: int) -> None:
        if 0 <= btn_id < len(self._entries):
            self.workflow_changed.emit(self._entries[btn_id])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def current_entry(self) -> WorkflowEntry | None:
        btn_id = self._btn_group.checkedId()
        if 0 <= btn_id < len(self._entries):
            return self._entries[btn_id]
        return None

    def select_entry(self, entry: WorkflowEntry) -> None:
        """Programmatically select a workflow by its entry."""
        for i, e in enumerate(self._entries):
            if e is entry or (e.script == entry.script and e.dag_config == entry.dag_config):
                self._buttons[i].setChecked(True)
                return
