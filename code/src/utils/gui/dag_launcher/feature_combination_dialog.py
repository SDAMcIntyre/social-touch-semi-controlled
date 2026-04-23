"""Dialog for creating or editing a feature combination entry."""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.touch_analytics.feature_extraction import AGGREGATION_NAMES, EXTRACTOR_REGISTRY


# Ordered feature list: sorted aggregation names first, then named extractors.
_FEATURE_ORDER = sorted(AGGREGATION_NAMES) + sorted(EXTRACTOR_REGISTRY)


class FeatureCombinationDialog(QDialog):
    """Dialog for creating or editing a feature combination.

    Create mode (``combo_name=None``): shows an editable name input and a
    feature checklist.  Validates name format and uniqueness on OK.

    Edit mode (``combo_name`` provided): shows a read-only name label and
    the feature checklist pre-populated with *selected_features*.
    """

    def __init__(
        self,
        task_name: str,
        existing_names: list[str],
        combo_name: Optional[str] = None,
        selected_features: Optional[list[str]] = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Parameters
        ----------
        task_name:
            Used in the window title.
        existing_names:
            Names already in use — prevents duplicates in create mode.
        combo_name:
            If provided, opens in edit mode with this name read-only.
        selected_features:
            Features pre-checked when the dialog opens.
        """
        super().__init__(parent)
        self._existing_names = existing_names
        self._edit_mode = combo_name is not None
        self._original_combo_name: Optional[str] = combo_name

        self.setWindowTitle(f"Feature Combination — {task_name}")
        self.setMinimumWidth(320)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. pressure velocity (spaces converted to underscores)")
        if self._edit_mode:
            self._name_edit.setText(combo_name)
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Select features:"))
        self._checkboxes: dict[str, QCheckBox] = {}
        selected = set(selected_features or [])
        for feature in _FEATURE_ORDER:
            cb = QCheckBox(feature)
            cb.setChecked(feature in selected)
            layout.addWidget(cb)
            self._checkboxes[feature] = cb

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setVisible(False)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        if not self._edit_mode:
            name = self._name_edit.text().strip()
            if not name:
                self._show_error("Name cannot be empty.")
                return
            if not re.fullmatch(r'[a-z0-9_ ]+', name):
                self._show_error(
                    "Name must contain only lowercase letters, digits, spaces, and underscores."
                )
                return
            name = name.replace(' ', '_')
            if name in self._existing_names:
                self._show_error(f"'{name}' already exists.")
                return
            self._combo_name = name

        if not self.get_selected_features():
            self._show_error("Select at least one feature.")
            return

        self._error_label.setVisible(False)
        self.accept()

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    def get_combo_name(self) -> Optional[str]:
        """Return the combination name (None if dialog was not accepted)."""
        return self._combo_name

    def get_original_combo_name(self) -> Optional[str]:
        """Return the original combination name (only in edit mode)."""
        return self._original_combo_name

    def get_selected_features(self) -> list[str]:
        """Return the list of checked feature names."""
        return [f for f, cb in self._checkboxes.items() if cb.isChecked()]
