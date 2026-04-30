"""Read-only console panel for displaying subprocess output."""

from __future__ import annotations

from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget


class ConsoleWidget(QWidget):
    """Scrollable, read-only text panel for pipeline stdout/stderr output."""

    _MAX_BLOCKS = 10_000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self._MAX_BLOCKS)

        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self._text.setFont(font)
        self._text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")

        self._auto_scroll_enabled = True
        self._auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self._auto_scroll_checkbox.setChecked(True)
        self._auto_scroll_checkbox.stateChanged.connect(self._on_auto_scroll_toggled)

        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(4, 2, 4, 2)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self._auto_scroll_checkbox)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._text)
        layout.addWidget(bottom_bar)

    def append_line(self, text: str) -> None:
        """Append one line and auto-scroll to the bottom if enabled."""
        self._text.appendPlainText(text)
        if self._auto_scroll_enabled:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def replace_last_line(self, text: str) -> None:
        """Overwrite the last line (carriage-return semantics for tqdm bars)."""
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.insertText(text)
        self._text.setTextCursor(cursor)
        if self._auto_scroll_enabled:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def clear(self) -> None:
        """Clear all content."""
        self._text.clear()

    def _on_auto_scroll_toggled(self) -> None:
        """Handle auto-scroll checkbox state change."""
        self._auto_scroll_enabled = self._auto_scroll_checkbox.isChecked()
