"""Dialog for creating or editing a radar group spec in render_touch_feature_radar."""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

__all__ = ["RadarGroupDialog"]

from ._feature_catalog import DATA_TYPES

_DATA_TYPES = DATA_TYPES

_AGGREGATIONS: list[str] = [
    "mean",
    "min",
    "max",
    "median",
    "std",
    "range",
    "skewness",
    "mean_during_iff",
    "mean_before_iff",
]


class RadarGroupDialog(QDialog):
    """Single-window dialog for creating or editing a radar group.

    Shows group name and features (data types x aggregations) only — no
    clustering methods section.

    Create mode: ``name=""`` and ``spec=None``.
    Edit mode: ``name`` is the existing group name, ``spec`` is pre-populated.

    After ``exec_()`` returns ``Accepted``:
    - :meth:`get_group_name` returns the validated name.
    - :meth:`get_group_spec` returns the full spec dict.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        name: str = "",
        spec: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self._initial_name = name
        self._spec = dict(spec or {})

        self.setWindowTitle("Radar Group" if not name else f"Radar Group — {name}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_name_section())
        layout.addWidget(self._build_features_section(), stretch=1)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: red;")
        self._error_label.setVisible(False)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_name_section(self) -> QWidget:
        box = QGroupBox("Group Name")
        name_layout = QHBoxLayout(box)
        self._name_edit = QLineEdit(self._correct_name(self._initial_name))
        self._name_edit.setPlaceholderText("e.g. canonical_iff")
        self._name_edit.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self._name_edit)
        return box

    def _build_features_section(self) -> QWidget:
        box = QGroupBox("Features")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 4, 6, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(4, 4, 4, 4)
        container_layout.setSpacing(6)

        existing_features: dict = self._spec.get("features", {}) or {}

        self._type_checks: dict[str, QCheckBox] = {}
        self._agg_checks: dict[str, dict[str, QCheckBox]] = {}
        self._agg_rows: dict[str, QWidget] = {}

        for dtype in _DATA_TYPES:
            dtype_row = QWidget()
            dtype_layout = QVBoxLayout(dtype_row)
            dtype_layout.setContentsMargins(0, 0, 0, 0)
            dtype_layout.setSpacing(2)

            is_checked = dtype in existing_features
            cb = QCheckBox(dtype.replace("_", " ").title())
            cb.setChecked(is_checked)
            self._type_checks[dtype] = cb
            dtype_layout.addWidget(cb)

            agg_widget = QWidget()
            agg_layout = QHBoxLayout(agg_widget)
            agg_layout.setContentsMargins(20, 0, 0, 0)
            agg_layout.setSpacing(4)

            self._agg_checks[dtype] = {}
            existing_aggs = set(existing_features.get(dtype) or [])
            for agg in _AGGREGATIONS:
                agg_cb = QCheckBox(agg)
                agg_cb.blockSignals(True)
                agg_cb.setChecked(agg in existing_aggs)
                agg_cb.blockSignals(False)
                agg_layout.addWidget(agg_cb)
                self._agg_checks[dtype][agg] = agg_cb
            agg_layout.addStretch()

            self._agg_rows[dtype] = agg_widget
            agg_widget.setVisible(is_checked)
            cb.toggled.connect(self._make_type_toggle(dtype))
            dtype_layout.addWidget(agg_widget)

            container_layout.addWidget(dtype_row)

        container_layout.addStretch()
        scroll.setWidget(container)
        return box

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    @staticmethod
    def _correct_name(text: str) -> str:
        return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))

    def _on_name_changed(self, text: str) -> None:
        corrected = self._correct_name(text)
        if corrected != text:
            self._name_edit.blockSignals(True)
            pos = self._name_edit.cursorPosition()
            self._name_edit.setText(corrected)
            self._name_edit.setCursorPosition(min(pos, len(corrected)))
            self._name_edit.blockSignals(False)

    def _make_type_toggle(self, dtype: str):
        def _handler(checked: bool) -> None:
            agg_row = self._agg_rows.get(dtype)
            if agg_row is not None:
                agg_row.setVisible(checked)
        return _handler

    # ------------------------------------------------------------------
    # Validation & acceptance
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        name = self._name_edit.text()
        if not name:
            self._show_error("Group name must not be empty.")
            return

        selected_types = [dt for dt in _DATA_TYPES if self._type_checks[dt].isChecked()]
        if not selected_types:
            self._show_error("Select at least one data type.")
            return

        for dtype in selected_types:
            selected_aggs = [
                agg for agg in _AGGREGATIONS
                if self._agg_checks[dtype][agg].isChecked()
            ]
            if not selected_aggs:
                self._show_error(
                    f"'{dtype}' is selected but has no aggregations chosen. "
                    "Select at least one aggregation or deselect the data type."
                )
                return

        self._validated_name = name
        self._error_label.setVisible(False)
        self.accept()

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_group_name(self) -> str:
        return self._validated_name

    def get_group_spec(self) -> dict:
        features: dict = {}
        for dtype in _DATA_TYPES:
            if not self._type_checks[dtype].isChecked():
                continue
            features[dtype] = [
                agg for agg in _AGGREGATIONS
                if self._agg_checks[dtype][agg].isChecked()
            ]

        enabled = self._spec.get("enabled", True)
        return {
            "enabled": enabled,
            "features": features,
        }
