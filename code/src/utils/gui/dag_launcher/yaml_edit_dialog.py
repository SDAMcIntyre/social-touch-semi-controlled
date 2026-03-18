"""Dialog for editing complex (list/dict) task option values as raw YAML."""

from __future__ import annotations

from io import StringIO
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from ruamel.yaml import YAML


class YamlEditDialog(QDialog):
    """Modal dialog for editing a complex task option value as a YAML fragment.

    Shows the current value serialised to YAML in a plain-text editor.  On OK,
    parses the text and rejects invalid YAML with an inline error message.
    """

    def __init__(
        self,
        task_name: str,
        option_key: str,
        current_value: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._value: Any = current_value

        self.setWindowTitle(f"Edit {option_key} — {task_name}")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        self._editor = QPlainTextEdit()
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self._editor.setFont(font)
        self._editor.setPlainText(self._serialize(current_value))
        layout.addWidget(self._editor)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setVisible(False)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _serialize(self, value: Any) -> str:
        """Serialise *value* to a YAML string."""
        buf = StringIO()
        self._yaml.dump(value, buf)
        return buf.getvalue()

    def _on_ok(self) -> None:
        text = self._editor.toPlainText()
        try:
            parsed = self._yaml.load(text)
        except Exception as exc:
            self._error_label.setText(f"Invalid YAML: {exc}")
            self._error_label.setVisible(True)
            return
        self._value = parsed
        self._error_label.setVisible(False)
        self.accept()

    def get_value(self) -> Any:
        """Return the parsed ruamel round-trip value accepted by the user."""
        return self._value
