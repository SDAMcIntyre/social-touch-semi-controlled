"""Dialog for creating or editing a grid group spec in cross_map_feature_grid.

The dialog composes fully-qualified feature names of the form
``<data_type>_<aggregation_or_extractor>`` and edits the four per-feature
bounds (``min``, ``max``, ``step``, ``span``) plus the group-level scalars.

The spec produced by :meth:`GridGroupDialog.get_group_spec` is written back into
the ``grid_groups`` option of the DAG config and consumed by the downstream
population-grid workflow.
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

__all__ = ["GridGroupDialog", "GridGroupReadOnlyDialog"]

from ._feature_catalog import DATA_TYPES

_DATA_TYPES = DATA_TYPES

_AGGREGATION_NAMES: list[str] = sorted(
    ["max", "min", "mean", "median", "std", "range", "skewness"]
)
_EXTRACTOR_NAMES: list[str] = sorted(["mean_during_iff", "mean_before_iff"])

_ALL_FEATURES: list[str] = _AGGREGATION_NAMES + _EXTRACTOR_NAMES

_VALID_NEURON_MODES: tuple[str, ...] = ("iff", "spike")

_DEFAULT_BOUNDS: dict[str, str] = {"min": "0", "max": "1", "step": "0.1", "span": "0.2"}


def _parse_feature_key(key: str, data_types: list[str]) -> tuple[str, str]:
    for dt in sorted(data_types, key=len, reverse=True):
        prefix = dt + "_"
        if key.startswith(prefix):
            return dt, key[len(prefix):]
    raise ValueError(
        f"Cannot parse feature key {key!r}: no data_type prefix matches."
    )


class GridGroupDialog(QDialog):
    """Single-window dialog for creating or editing a population RF grid group.

    Create mode: ``existing_spec=None``.
    Edit mode: ``existing_spec`` is the dict returned by
    :meth:`~utils.pipeline.dag_config_model.DagConfigModel.get_grid_group_spec`.

    After ``exec_()`` returns ``Accepted``:
    - :meth:`get_group_name` returns the validated name.
    - :meth:`get_group_spec` returns the full spec dict.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        name: str = "",
        existing_spec: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self._initial_name = name
        self._spec = dict(existing_spec or {})

        self.setWindowTitle("Grid Group" if not name else f"Grid Group — {name}")
        self.setMinimumWidth(680)
        self.setMinimumHeight(600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_name_section())
        layout.addWidget(self._build_scalars_section())
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
        self._name_edit = QLineEdit(self._sanitise_name(self._initial_name))
        self._name_edit.setPlaceholderText("e.g. velocity_pressure_2d")
        self._name_edit.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self._name_edit)
        return box

    def _build_scalars_section(self) -> QWidget:
        box = QGroupBox("Group Settings")
        form = QFormLayout(box)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        self._enabled_cb = QCheckBox()
        self._enabled_cb.setChecked(bool(self._spec.get("enabled", True)))
        form.addRow("Enabled:", self._enabled_cb)

        self._neuron_mode_combo = QComboBox()
        for mode in _VALID_NEURON_MODES:
            self._neuron_mode_combo.addItem(mode)
        current_mode = self._spec.get("neuron_mode", _VALID_NEURON_MODES[0])
        idx = list(_VALID_NEURON_MODES).index(current_mode) if current_mode in _VALID_NEURON_MODES else 0
        self._neuron_mode_combo.setCurrentIndex(idx)
        form.addRow("Neuron Mode:", self._neuron_mode_combo)

        self._per_gesture_cb = QCheckBox()
        self._per_gesture_cb.setChecked(bool(self._spec.get("per_gesture_type", False)))
        form.addRow("Per Gesture Type:", self._per_gesture_cb)

        self._vertex_threshold_edit = QLineEdit(
            str(self._spec.get("vertex_threshold_ratio", 0.25))
        )
        self._vertex_threshold_edit.setFixedWidth(80)
        form.addRow("Vertex Threshold Ratio:", self._vertex_threshold_edit)

        self._compute_baseline_cb = QCheckBox()
        self._compute_baseline_cb.setChecked(bool(self._spec.get("compute_baseline", True)))
        form.addRow("Compute Baseline:", self._compute_baseline_cb)

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
        container_layout.setSpacing(4)

        existing_features: dict = self._spec.get("features") or {}

        parsed_features: dict[tuple[str, str], dict[str, str]] = {}
        for key, bounds in existing_features.items():
            dt, feat = _parse_feature_key(key, _DATA_TYPES)
            parsed_features[(dt, feat)] = {
                "min": str(bounds.get("min", _DEFAULT_BOUNDS["min"])),
                "max": str(bounds.get("max", _DEFAULT_BOUNDS["max"])),
                "step": str(bounds.get("step", _DEFAULT_BOUNDS["step"])),
                "span": str(bounds.get("span", _DEFAULT_BOUNDS["span"])),
            }

        self._feature_checks: dict[tuple[str, str], QCheckBox] = {}
        self._feature_bounds: dict[tuple[str, str], dict[str, QLineEdit]] = {}
        self._feature_bounds_rows: dict[tuple[str, str], QWidget] = {}
        self._dtype_checks: dict[str, QCheckBox] = {}
        self._dtype_rows: dict[str, QWidget] = {}

        for dtype in _DATA_TYPES:
            dtype_section = QWidget()
            dtype_section_layout = QVBoxLayout(dtype_section)
            dtype_section_layout.setContentsMargins(0, 0, 0, 0)
            dtype_section_layout.setSpacing(1)

            has_any_checked = any(
                (dtype, feat) in parsed_features for feat in _ALL_FEATURES
            )

            dtype_cb = QCheckBox()
            dtype_cb.setStyleSheet("font-weight: bold;")
            dtype_label_text = dtype.replace("_", " ").title()
            dtype_cb.setText(dtype_label_text)
            dtype_cb.setChecked(has_any_checked)
            self._dtype_checks[dtype] = dtype_cb
            dtype_section_layout.addWidget(dtype_cb)

            sub_widget = QWidget()
            sub_layout = QVBoxLayout(sub_widget)
            sub_layout.setContentsMargins(20, 0, 0, 2)
            sub_layout.setSpacing(1)

            for feat in _ALL_FEATURES:
                key_pair = (dtype, feat)
                is_checked = key_pair in parsed_features

                feat_row = QWidget()
                feat_row_layout = QHBoxLayout(feat_row)
                feat_row_layout.setContentsMargins(0, 0, 0, 0)
                feat_row_layout.setSpacing(4)

                feat_cb = QCheckBox(feat)
                feat_cb.setChecked(is_checked)
                self._feature_checks[key_pair] = feat_cb
                feat_row_layout.addWidget(feat_cb)

                bounds_widget = QWidget()
                bounds_layout = QHBoxLayout(bounds_widget)
                bounds_layout.setContentsMargins(8, 0, 0, 0)
                bounds_layout.setSpacing(4)

                saved = parsed_features.get(key_pair, _DEFAULT_BOUNDS)
                edits: dict[str, QLineEdit] = {}
                for field in ("min", "max", "step", "span"):
                    lbl = QLabel(f"{field}:")
                    lbl.setStyleSheet("color: #555;")
                    edit = QLineEdit(saved[field])
                    edit.setFixedWidth(58)
                    bounds_layout.addWidget(lbl)
                    bounds_layout.addWidget(edit)
                    edits[field] = edit
                bounds_layout.addStretch()

                self._feature_bounds[key_pair] = edits
                self._feature_bounds_rows[key_pair] = bounds_widget

                bounds_widget.setVisible(is_checked)
                feat_cb.toggled.connect(self._make_feat_toggle(key_pair))

                feat_row_layout.addWidget(bounds_widget)
                feat_row_layout.addStretch()
                sub_layout.addWidget(feat_row)

            self._dtype_rows[dtype] = sub_widget
            sub_widget.setVisible(has_any_checked)
            dtype_cb.toggled.connect(self._make_dtype_toggle(dtype))
            dtype_section_layout.addWidget(sub_widget)

            container_layout.addWidget(dtype_section)

        container_layout.addStretch()
        scroll.setWidget(container)
        return box

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_name(text: str) -> str:
        return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))

    def _on_name_changed(self, text: str) -> None:
        corrected = self._sanitise_name(text)
        if corrected != text:
            self._name_edit.blockSignals(True)
            try:
                pos = self._name_edit.cursorPosition()
                self._name_edit.setText(corrected)
                self._name_edit.setCursorPosition(min(pos, len(corrected)))
            finally:
                self._name_edit.blockSignals(False)

    def _make_dtype_toggle(self, dtype: str):
        def _handler(checked: bool) -> None:
            sub = self._dtype_rows.get(dtype)
            if sub is not None:
                sub.setVisible(checked)
            if not checked:
                for feat in _ALL_FEATURES:
                    key_pair = (dtype, feat)
                    cb = self._feature_checks.get(key_pair)
                    if cb is not None and cb.isChecked():
                        cb.blockSignals(True)
                        try:
                            cb.setChecked(False)
                        finally:
                            cb.blockSignals(False)
                        bounds_row = self._feature_bounds_rows.get(key_pair)
                        if bounds_row is not None:
                            bounds_row.setVisible(False)
        return _handler

    def _make_feat_toggle(self, key_pair: tuple[str, str]):
        def _handler(checked: bool) -> None:
            bounds_row = self._feature_bounds_rows.get(key_pair)
            if bounds_row is not None:
                bounds_row.setVisible(checked)
        return _handler

    # ------------------------------------------------------------------
    # Validation & acceptance
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        name = self._name_edit.text()
        if not name:
            self._show_error("Group name must not be empty.")
            return

        vtr_text = self._vertex_threshold_edit.text().strip()
        try:
            float(vtr_text)
        except ValueError:
            self._show_error("Vertex Threshold Ratio must be a number.")
            return

        checked_pairs = [
            pair for pair, cb in self._feature_checks.items() if cb.isChecked()
        ]
        if not checked_pairs:
            self._show_error("Select at least one feature.")
            return

        for pair in checked_pairs:
            dtype, feat = pair
            edits = self._feature_bounds[pair]
            values: dict[str, float] = {}
            for field, edit in edits.items():
                text = edit.text().strip()
                try:
                    values[field] = float(text)
                except ValueError:
                    self._show_error(
                        f"'{dtype}_{feat}': '{field}' must be a number."
                    )
                    return
            if values["min"] >= values["max"]:
                self._show_error(
                    f"'{dtype}_{feat}': min must be less than max."
                )
                return
            if values["step"] <= 0:
                self._show_error(
                    f"'{dtype}_{feat}': step must be greater than 0."
                )
                return
            if values["span"] <= 0:
                self._show_error(
                    f"'{dtype}_{feat}': span must be greater than 0."
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
        features: dict[str, dict[str, float]] = {}
        for pair, cb in self._feature_checks.items():
            if not cb.isChecked():
                continue
            dtype, feat = pair
            edits = self._feature_bounds[pair]
            features[f"{dtype}_{feat}"] = {
                field: float(edit.text().strip())
                for field, edit in edits.items()
            }

        return {
            "enabled": self._enabled_cb.isChecked(),
            "neuron_mode": self._neuron_mode_combo.currentText(),
            "per_gesture_type": self._per_gesture_cb.isChecked(),
            "vertex_threshold_ratio": float(self._vertex_threshold_edit.text().strip()),
            "compute_baseline": self._compute_baseline_cb.isChecked(),
            "features": features,
        }


class GridGroupReadOnlyDialog(QDialog):
    """Read-only detail view of a population RF grid group spec.

    Opened to inspect an existing group without allowing edits.
    """

    def __init__(self, name: str, spec: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Grid Group — {name}")
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        scalars_box = QGroupBox("Group Settings")
        scalars_form = QFormLayout(scalars_box)
        scalars_form.setContentsMargins(8, 8, 8, 8)

        enabled = spec.get("enabled", True)
        scalars_form.addRow("Enabled:", QLabel("Yes" if enabled else "No"))
        scalars_form.addRow("Neuron Mode:", QLabel(str(spec.get("neuron_mode", ""))))
        scalars_form.addRow(
            "Per Gesture Type:",
            QLabel("Yes" if spec.get("per_gesture_type", False) else "No"),
        )
        scalars_form.addRow(
            "Vertex Threshold Ratio:",
            QLabel(str(spec.get("vertex_threshold_ratio", ""))),
        )
        scalars_form.addRow(
            "Compute Baseline:",
            QLabel("Yes" if spec.get("compute_baseline", True) else "No"),
        )
        layout.addWidget(scalars_box)

        features_box = QGroupBox("Features")
        features_layout = QVBoxLayout(features_box)
        features_layout.setContentsMargins(6, 4, 6, 4)
        features: dict = spec.get("features") or {}
        if features:
            for feat_key, bounds in features.items():
                bounds_str = ", ".join(
                    f"{k}={bounds.get(k, '?')}" for k in ("min", "max", "step", "span")
                )
                features_layout.addWidget(QLabel(f"{feat_key}:  {{{bounds_str}}}"))
        else:
            features_layout.addWidget(QLabel("(none)"))
        layout.addWidget(features_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
