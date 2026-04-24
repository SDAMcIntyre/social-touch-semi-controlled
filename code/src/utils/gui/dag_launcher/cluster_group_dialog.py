"""Dialog for creating or editing a cluster group spec in touch_clustering."""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = ["ClusterGroupDialog", "ClusterGroupReadOnlyDialog"]

_DATA_TYPES: list[str] = [
    "contact_area",
    "contact_depth",
    "velocity",
    "acceleration",
    "pressure",
    "location",
    "touch_category",
]

_AGGREGATIONS: list[str] = ["mean", "min", "max", "median", "std", "range", "skewness"]

_NON_AGGREGATABLE: set[str] = {"touch_category"}

_CLUSTERING_METHODS: list[str] = [
    "binning",
    "kmeans",
    "dbscan",
    "hierarchical",
    "type_stratified",
]

_DEFAULT_PARAMS: dict[str, dict] = {
    "binning": {"n_bins": 20, "bin_method": "equal_width"},
    "kmeans": {"n_clusters": 8},
    "dbscan": {"eps": 0.5, "min_samples": 5},
    "hierarchical": {"n_clusters": 8, "linkage": "ward"},
    "type_stratified": {},
}


def _params_summary(params: dict) -> str:
    if not params:
        return ""
    parts = [f"{k}={v}" for k, v in params.items()]
    return "  " + ", ".join(parts)


class _ParamEditDialog(QDialog):
    """Simple key-value param editing dialog for a clustering method."""

    def __init__(self, method_name: str, params: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params = dict(params)
        self.setWindowTitle(f"Configure — {method_name}")
        self.setMinimumWidth(320)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self._edits: dict[str, QLineEdit] = {}
        for key, val in self._params.items():
            edit = QLineEdit(str(val))
            form.addRow(f"{key}:", edit)
            self._edits[key] = edit

        if not self._params:
            layout.addWidget(QLabel("No parameters to configure."))
        else:
            layout.addLayout(form)

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
        new_params: dict = {}
        for key, edit in self._edits.items():
            original_val = self._params[key]
            text = edit.text().strip()
            try:
                if isinstance(original_val, bool):
                    new_params[key] = text.lower() in ("true", "1", "yes")
                elif isinstance(original_val, int):
                    new_params[key] = int(text)
                elif isinstance(original_val, float):
                    new_params[key] = float(text)
                else:
                    new_params[key] = text
            except (ValueError, TypeError):
                self._error_label.setText(
                    f"Invalid value for '{key}': expected {type(original_val).__name__}."
                )
                self._error_label.setVisible(True)
                return
        self._result_params = new_params
        self._error_label.setVisible(False)
        self.accept()

    def get_params(self) -> dict:
        return self._result_params


class ClusterGroupDialog(QDialog):
    """Two-tab dialog for creating or editing a cluster group.

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

        self.setWindowTitle("Cluster Group" if not name else f"Cluster Group — {name}")
        self.setMinimumWidth(480)
        self.setMinimumHeight(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.addTab(self._build_features_tab(), "Features")
        tabs.addTab(self._build_algorithms_tab(), "Algorithms")
        layout.addWidget(tabs)

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
    # Tab builders
    # ------------------------------------------------------------------

    def _build_features_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
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

            if dtype not in _NON_AGGREGATABLE:
                agg_widget = QWidget()
                agg_layout = QHBoxLayout(agg_widget)
                agg_layout.setContentsMargins(20, 0, 0, 0)
                agg_layout.setSpacing(4)

                self._agg_checks[dtype] = {}
                existing_aggs = set(existing_features.get(dtype) or [])
                for agg in _AGGREGATIONS:
                    agg_cb = QCheckBox(agg)
                    agg_cb.setChecked(agg in existing_aggs)
                    agg_layout.addWidget(agg_cb)
                    self._agg_checks[dtype][agg] = agg_cb
                agg_layout.addStretch()

                self._agg_rows[dtype] = agg_widget
                agg_widget.setVisible(is_checked)
                cb.toggled.connect(self._make_type_toggle(dtype))
                dtype_layout.addWidget(agg_widget)
            else:
                cb.toggled.connect(lambda _checked: None)

            container_layout.addWidget(dtype_row)

        container_layout.addStretch()
        scroll.setWidget(container)
        return tab

    def _build_algorithms_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        name_box = QGroupBox("Group Name")
        name_layout = QHBoxLayout(name_box)
        self._name_edit = QLineEdit(self._initial_name)
        self._name_edit.setPlaceholderText("e.g. depth_velocity_group")
        name_layout.addWidget(self._name_edit)
        layout.addWidget(name_box)

        methods_box = QGroupBox("Clustering Methods")
        methods_layout = QVBoxLayout(methods_box)
        methods_layout.setSpacing(4)

        existing_methods: dict = (self._spec.get("clustering_methods") or {})

        self._method_checks: dict[str, QCheckBox] = {}
        self._method_params: dict[str, dict] = {}
        self._method_summary_labels: dict[str, QLabel] = {}

        for method in _CLUSTERING_METHODS:
            method_cfg = existing_methods.get(method)
            if method_cfg is not None:
                is_enabled = method_cfg.get("enabled", True)
                params = {
                    k: v for k, v in method_cfg.items()
                    if k not in ("method", "enabled")
                }
            else:
                is_enabled = False
                params = dict(_DEFAULT_PARAMS.get(method, {}))

            self._method_params[method] = params

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(method.replace("_", " ").title())
            cb.setChecked(is_enabled)
            self._method_checks[method] = cb
            row_layout.addWidget(cb)

            summary = QLabel(_params_summary(params))
            summary.setStyleSheet("color: #555;")
            self._method_summary_labels[method] = summary
            row_layout.addWidget(summary)

            row_layout.addStretch()

            configure_btn = QPushButton("Configure…")
            configure_btn.setFixedWidth(90)
            configure_btn.clicked.connect(self._make_configure_handler(method))
            row_layout.addWidget(configure_btn)

            methods_layout.addWidget(row)

        layout.addWidget(methods_box)
        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    def _make_type_toggle(self, dtype: str):
        def _handler(checked: bool) -> None:
            agg_row = self._agg_rows.get(dtype)
            if agg_row is not None:
                agg_row.setVisible(checked)
        return _handler

    def _make_configure_handler(self, method: str):
        def _handler(_checked: bool = False) -> None:
            params = dict(self._method_params.get(method, {}))
            dlg = _ParamEditDialog(method, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_params()
                self._method_params[method] = new_params
                self._method_summary_labels[method].setText(_params_summary(new_params))
        return _handler

    # ------------------------------------------------------------------
    # Validation & acceptance
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        name = self._name_edit.text().strip()
        if not re.fullmatch(r"[a-z0-9_]+", name):
            self._show_error(
                "Group name must contain only lowercase letters, digits, and underscores."
            )
            return

        selected_types = [dt for dt in _DATA_TYPES if self._type_checks[dt].isChecked()]
        if not selected_types:
            self._show_error("Select at least one data type.")
            return

        for dtype in selected_types:
            if dtype in _NON_AGGREGATABLE:
                continue
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

        enabled_methods = [m for m in _CLUSTERING_METHODS if self._method_checks[m].isChecked()]
        if not enabled_methods:
            self._show_error("Enable at least one clustering method.")
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
            if dtype in _NON_AGGREGATABLE:
                features[dtype] = []
            else:
                features[dtype] = [
                    agg for agg in _AGGREGATIONS
                    if self._agg_checks[dtype][agg].isChecked()
                ]

        clustering_methods: dict = {}
        for method in _CLUSTERING_METHODS:
            is_enabled = self._method_checks[method].isChecked()
            params = dict(self._method_params.get(method, {}))
            entry: dict = {"method": method}
            entry.update(params)
            if not is_enabled:
                entry["enabled"] = False
            clustering_methods[method] = entry

        enabled = self._spec.get("enabled", True)
        return {
            "enabled": enabled,
            "features": features,
            "clustering_methods": clustering_methods,
        }


class ClusterGroupReadOnlyDialog(QDialog):
    """Read-only detail view of a cluster group spec.

    Opened via the 'Details…' button in downstream task panels (touch_comparing,
    map_receptive_fields_clustered) to inspect a group defined in touch_clustering.
    """

    def __init__(self, name: str, spec: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Cluster Group — {name}")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        enabled = spec.get("enabled", True)
        state_lbl = QLabel(f"Enabled: {'Yes' if enabled else 'No'}")
        layout.addWidget(state_lbl)

        features_box = QGroupBox("Features")
        features_layout = QVBoxLayout(features_box)
        features_layout.setContentsMargins(6, 4, 6, 4)
        features: dict = spec.get("features") or {}
        if features:
            for dtype, aggs in features.items():
                if dtype == "touch_category":
                    text = dtype
                elif aggs:
                    text = f"{dtype}:  [{', '.join(str(a) for a in aggs)}]"
                else:
                    text = dtype
                features_layout.addWidget(QLabel(text))
        else:
            features_layout.addWidget(QLabel("(none)"))
        layout.addWidget(features_box)

        methods_box = QGroupBox("Clustering Methods")
        methods_layout = QVBoxLayout(methods_box)
        methods_layout.setContentsMargins(6, 4, 6, 4)
        methods: dict = spec.get("clustering_methods") or {}
        if methods:
            for method_name, method_cfg in methods.items():
                is_enabled = method_cfg.get("enabled", True)
                params = {
                    k: v for k, v in method_cfg.items()
                    if k not in ("method", "enabled")
                }
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                text = method_name
                if params_str:
                    text += f"  ({params_str})"
                lbl = QLabel(text)
                if not is_enabled:
                    lbl.setStyleSheet("color: #888;")
                methods_layout.addWidget(lbl)
        else:
            methods_layout.addWidget(QLabel("(none)"))
        layout.addWidget(methods_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
