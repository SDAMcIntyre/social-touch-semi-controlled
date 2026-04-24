"""Per-task option detail panel for the DAG launcher GUI."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ruamel.yaml.comments import CommentedSeq

from utils.gui.dag_launcher.cluster_group_dialog import ClusterGroupDialog, ClusterGroupReadOnlyDialog
from utils.gui.dag_launcher.feature_combination_dialog import FeatureCombinationDialog
from utils.gui.dag_launcher.yaml_edit_dialog import YamlEditDialog
from utils.pipeline.dag_config_model import DagConfigModel

_COMPLEX_FG = QColor("#336699")

# Option keys that should render as a dropdown. Values are (display_label, saved_value) pairs.
# None as saved_value writes YAML null (~ ), meaning "use the default".
_OPTION_ENUMS: dict[str, list[tuple[str, object]]] = {
    "projection_method": [
        ("3D (default)", None),
        ("tangent_plane", "tangent_plane"),
        ("cylindrical_unwrap", "cylindrical_unwrap"),
    ],
}


def _is_profile_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have a 'method' key."""
    if not isinstance(val, dict) or not val:
        return False
    return all(isinstance(v, dict) and "method" in v for v in val.values())


def _is_feature_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts with 'enabled' but not 'method' or 'features'."""
    if not isinstance(val, dict) or not val:
        return False
    return all(
        isinstance(v, dict) and "enabled" in v and "method" not in v and "features" not in v
        for v in val.values()
    )


def _is_feature_combinations_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have a 'features' key."""
    if not isinstance(val, dict) or not val:
        return False
    return all(isinstance(v, dict) and "features" in v for v in val.values())


def _is_cluster_groups_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have both 'features' and 'clustering_methods' keys."""
    if not isinstance(val, dict) or not val:
        return False
    return all(
        isinstance(v, dict) and "features" in v and "clustering_methods" in v
        for v in val.values()
    )


def _cluster_group_summary(spec: dict) -> str:
    features: dict = spec.get("features") or {}
    parts: list[str] = []
    for dtype, aggs in features.items():
        if dtype == "touch_category":
            parts.append("category")
        elif aggs:
            parts.append(f"{dtype}[{','.join(aggs)}]")
        else:
            parts.append(dtype)
    return " · ".join(parts)


def _option_header(key: str) -> str:
    return key.replace("_", " ").title()


def _preview_text(val: Any) -> str:
    if isinstance(val, list):
        items = [str(v) for v in val[:3]]
        preview = "[" + ", ".join(items) + (", ..." if len(val) > 3 else "") + "]"
        return f"{preview} ({len(val)} items)"
    if isinstance(val, dict):
        keys = list(val.keys())[:3]
        preview = "{" + ", ".join(str(k) for k in keys) + (", ..." if len(val) > 3 else "") + "}"
        return f"{preview} ({len(val)} keys)"
    return str(val)


class TaskDetailPanel(QWidget):
    """Panel with a pinned task-name header and a scrollable options area.

    Call :meth:`show_task` to populate. Emits :attr:`task_changed` whenever
    any option is modified.
    """

    task_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: DagConfigModel | None = None
        self._task_name: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Pinned header — always visible above the scroll area
        self._header = QLabel()
        font = QFont()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self._header.setFont(font)
        self._header.setStyleSheet(
            "padding: 4px 8px;"
            "background: #e8e8e8;"
            "border-bottom: 1px solid #c0c0c0;"
        )
        outer.addWidget(self._header)

        # Scroll area for options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        scroll.setWidget(self._content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_task(self, model: DagConfigModel, task_name: str) -> None:
        """Clear the panel and rebuild for *task_name*."""
        self._model = model
        self._task_name = task_name
        self._header.setText(task_name)

        # Clear option sections — keep only the trailing stretch
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # force_processing is rendered in the task list row, not here
        options = {
            k: v for k, v in model.get_task_options(task_name).items()
            if k != "force_processing"
        }

        if not options:
            lbl = QLabel("No options")
            lbl.setStyleSheet("color: #888;")
            self._layout.insertWidget(0, lbl)
            return

        for i, (key, val) in enumerate(options.items()):
            if key in _OPTION_ENUMS:
                widget = self._make_enum_section(key, val)
            elif key == "camera_angle_mode" and isinstance(val, dict) and "auto" in val:
                widget = self._make_camera_angle_mode_section(key, val)
            elif key == "cluster_groups" and isinstance(val, (list, CommentedSeq)):
                widget = self._make_downstream_cluster_groups_section(key, val)
            elif key == "cluster_groups" and _is_cluster_groups_dict(val):
                widget = self._make_cluster_groups_section(key, val)
            elif _is_profile_dict(val):
                widget = self._make_profile_section(key, val)
            elif _is_feature_combinations_dict(val):
                widget = self._make_combination_section(key, val)
            elif _is_feature_dict(val):
                cols = 1 if len(val) <= 6 else 3
                widget = self._make_feature_section(key, val, cols=cols)
            else:
                widget = self._make_scalar_section(key, val)
            self._layout.insertWidget(i, widget)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _make_enum_section(self, key: str, val: Any) -> QWidget:
        """QComboBox for options with a fixed set of allowed values."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        entries = _OPTION_ENUMS[key]
        combo = QComboBox()
        current_idx = 0
        for i, (label, saved) in enumerate(entries):
            combo.addItem(label)
            if val == saved:
                current_idx = i
        combo.setCurrentIndex(current_idx)
        combo.currentIndexChanged.connect(self._make_enum_handler(key, entries, combo))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(combo)
        row_layout.addStretch()
        layout.addWidget(row)
        return box

    def _make_scalar_section(self, key: str, val: Any) -> QWidget:
        """Bool checkbox, string/number line edit, or complex clickable label."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        if isinstance(val, bool):
            cb = QCheckBox("Enabled")
            cb.setChecked(val)
            cb.stateChanged.connect(self._make_bool_handler(key, cb))
            layout.addWidget(cb)
        elif isinstance(val, str) and val in {"auto", "manual"}:
            cb = QCheckBox("Auto")
            cb.setChecked(val == "auto")
            cb.stateChanged.connect(self._make_mode_toggle_handler(key, cb))
            layout.addWidget(cb)
        elif isinstance(val, (int, float, str)):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(str(val))
            edit.setFixedWidth(200)
            edit.editingFinished.connect(self._make_scalar_edit_handler(key, val, edit))
            row_layout.addWidget(edit)
            row_layout.addStretch()
            layout.addWidget(row_widget)
        else:
            lbl = QLabel(_preview_text(val))
            lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setToolTip("Click to edit in YAML editor")
            lbl.mousePressEvent = self._make_complex_click_handler(key, lbl)
            layout.addWidget(lbl)

        return box

    def _make_camera_angle_mode_section(self, key: str, val: dict) -> QWidget:
        """Group box titled 'Camera Angle Mode Auto' with an Enabled checkbox."""
        box = QGroupBox("Camera Angle Mode Auto")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        auto_cfg = val.get("auto") or {}
        enabled = auto_cfg.get("enabled", False)
        cb = QCheckBox("Enabled")
        cb.setChecked(enabled)
        cb.stateChanged.connect(self._make_camera_angle_mode_handler(key, cb))
        layout.addWidget(cb)
        return box

    def _make_feature_section(self, key: str, val: dict, cols: int = 3) -> QWidget:
        """Checkbox grid with optional '...' param buttons. cols=1 gives expanded per-item layout."""
        box = QGroupBox(_option_header(key))
        flow = QVBoxLayout(box)
        flow.setContentsMargins(6, 4, 6, 4)
        flow.setSpacing(2)

        if cols == 1:
            for i, (feature_name, feature_cfg) in enumerate(val.items()):
                if i > 0:
                    flow.addSpacing(6)

                lbl_row = QWidget()
                lbl_row_layout = QHBoxLayout(lbl_row)
                lbl_row_layout.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(feature_name.replace("_", " ") + ":")
                font = QFont()
                font.setBold(True)
                lbl.setFont(font)
                lbl_row_layout.addWidget(lbl)
                lbl_row_layout.addStretch()
                flow.addWidget(lbl_row)

                for param_key, param_val in feature_cfg.items():
                    if not isinstance(param_val, bool):
                        continue
                    param_row = QWidget()
                    param_layout = QHBoxLayout(param_row)
                    param_layout.setContentsMargins(14, 0, 0, 0)
                    param_layout.setSpacing(4)
                    cb = QCheckBox(param_key.replace("_", " "))
                    if param_key == "enabled":
                        cb.setChecked(
                            self._model.get_profile_enabled(self._task_name, key, feature_name)
                        )
                        cb.stateChanged.connect(self._make_profile_handler(key, feature_name, cb))
                    else:
                        cb.setChecked(bool(param_val))
                        cb.stateChanged.connect(
                            self._make_transform_bool_param_handler(key, feature_name, param_key, cb)
                        )
                    param_layout.addWidget(cb)
                    param_layout.addStretch()
                    flow.addWidget(param_row)

                has_other_params = any(
                    k != "enabled" and not isinstance(v, bool)
                    for k, v in feature_cfg.items()
                )
                if has_other_params:
                    btn_row = QWidget()
                    btn_layout = QHBoxLayout(btn_row)
                    btn_layout.setContentsMargins(14, 0, 0, 0)
                    btn = QPushButton("Parameters…")
                    btn.setFixedWidth(90)
                    btn.clicked.connect(self._make_transform_params_handler(key, feature_name))
                    btn_layout.addWidget(btn)
                    btn_layout.addStretch()
                    flow.addWidget(btn_row)

            return box

        flow.setSpacing(4)
        row_widget: QWidget | None = None
        row_layout: QHBoxLayout | None = None

        for i, (feature_name, feature_cfg) in enumerate(val.items()):
            if i % cols == 0:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                flow.addWidget(row_widget)

            is_enabled = self._model.get_profile_enabled(self._task_name, key, feature_name)
            has_params = any(k != "enabled" for k in feature_cfg)

            cb = QCheckBox(feature_name.replace("_", " "))
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, feature_name, cb))

            if has_params:
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)
                cell_layout.addWidget(cb)
                btn = QPushButton("...")
                btn.setFixedSize(24, 20)
                btn.setToolTip("Edit feature parameters")
                btn.clicked.connect(self._make_feature_params_handler(key, feature_name))
                cell_layout.addWidget(btn)
                row_layout.addWidget(cell)
            else:
                row_layout.addWidget(cb)

        remainder = len(val) % cols
        if remainder != 0 and row_layout is not None:
            for _ in range(cols - remainder):
                row_layout.addStretch()

        return box

    def _make_profile_section(self, key: str, val: dict) -> QWidget:
        """Checkbox per profile with '...' button for method params."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for profile_name in val:
            is_enabled = self._model.get_profile_enabled(self._task_name, key, profile_name)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(profile_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, profile_name, cb))
            row_layout.addWidget(cb)

            btn = QPushButton("...")
            btn.setFixedSize(24, 20)
            btn.setToolTip("Edit profile parameters")
            btn.clicked.connect(self._make_profile_params_handler(key, profile_name))
            row_layout.addWidget(btn)

            row_layout.addStretch()
            layout.addWidget(row)

        return box

    def _make_combination_section(self, key: str, val: dict) -> QWidget:
        """Checkbox + feature preview per combination, plus '+' add button."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for combo_name in val:
            is_enabled = self._model.get_profile_enabled(self._task_name, key, combo_name)
            features = self._model.get_combination_features(self._task_name, key, combo_name)

            row = QWidget()
            row.setContextMenuPolicy(Qt.CustomContextMenu)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(combo_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, combo_name, cb))
            row_layout.addWidget(cb)

            preview = QLabel(f"[{', '.join(features)}]")
            preview.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            preview.setCursor(Qt.PointingHandCursor)
            preview.setToolTip("Click to edit features")
            preview.mousePressEvent = self._make_combination_edit_handler(key, combo_name, preview)
            row_layout.addWidget(preview)

            row_layout.addStretch()

            for _w in (row, cb, preview):
                _w.setContextMenuPolicy(Qt.CustomContextMenu)
                _w.customContextMenuRequested.connect(
                    self._make_combination_context_handler(key, combo_name)
                )

            layout.addWidget(row)

        # "+" add button
        add_row = QWidget()
        add_layout = QHBoxLayout(add_row)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(24, 20)
        add_btn.setToolTip("Add new feature combination")
        add_btn.clicked.connect(self._make_combination_add_handler(key))
        add_layout.addWidget(add_btn)
        add_layout.addStretch()
        layout.addWidget(add_row)

        return box

    def _make_cluster_groups_section(self, key: str, val: dict) -> QWidget:
        """Per-group rows with enabled checkbox, summary label, Edit and Delete buttons."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for group_name in val:
            spec = val[group_name]
            is_enabled = bool(spec.get("enabled", True))
            summary = _cluster_group_summary(spec)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(
                self._make_cluster_group_enabled_handler(key, group_name, cb)
            )
            row_layout.addWidget(cb)

            summary_lbl = QLabel(summary)
            summary_lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            summary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            summary_lbl.setMinimumWidth(0)
            summary_lbl.setToolTip(summary)
            row_layout.addWidget(summary_lbl, stretch=1)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedWidth(48)
            edit_btn.clicked.connect(
                self._make_cluster_group_edit_handler(key, group_name, cb, summary_lbl)
            )
            row_layout.addWidget(edit_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedWidth(56)
            del_btn.clicked.connect(self._make_cluster_group_delete_handler(key, group_name))
            row_layout.addWidget(del_btn)

            layout.addWidget(row)

        new_row = QWidget()
        new_layout = QHBoxLayout(new_row)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_btn = QPushButton("New Group…")
        new_btn.setFixedWidth(100)
        new_btn.clicked.connect(self._make_cluster_group_new_handler(key))
        new_layout.addWidget(new_btn)
        new_layout.addStretch()
        layout.addWidget(new_row)

        return box

    def _make_downstream_cluster_groups_section(self, key: str, val: list) -> QWidget:
        """Checkbox per group defined in touch_clustering; checked if this task references it."""
        box = QGroupBox("Cluster Groups")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        sub_lbl = QLabel("From touch_clustering:")
        sub_font = QFont()
        sub_font.setItalic(True)
        sub_lbl.setFont(sub_font)
        sub_lbl.setStyleSheet("color: #555;")
        layout.addWidget(sub_lbl)

        try:
            all_groups = self._model.get_profile_names("touch_clustering", "cluster_groups")
        except (KeyError, AttributeError):
            all_groups = []

        if not all_groups:
            info = QLabel("No cluster groups defined in touch_clustering.")
            info.setStyleSheet("color: #888;")
            layout.addWidget(info)
            return box

        selected: list[str] = list(val)

        for group_name in all_groups:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(group_name in selected)
            cb.stateChanged.connect(
                self._make_downstream_group_handler(key, group_name, cb, all_groups, selected)
            )
            row_layout.addWidget(cb)

            row_layout.addStretch()

            details_btn = QPushButton("Details…")
            details_btn.setFixedWidth(72)
            details_btn.clicked.connect(
                self._make_downstream_group_details_handler(group_name)
            )
            row_layout.addWidget(details_btn)

            layout.addWidget(row)

        return box

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    def _make_enum_handler(self, key: str, entries: list, combo: QComboBox):
        def _handler(index: int) -> None:
            if self._model is None or self._task_name is None:
                return
            _, saved_val = entries[index]
            self._model.set_task_option(self._task_name, key, saved_val)
            self.task_changed.emit()
        return _handler

    def _make_mode_toggle_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, "auto" if cb.isChecked() else "manual")
            self.task_changed.emit()
        return _handler

    def _make_camera_angle_mode_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, {"auto": {"enabled": cb.isChecked()}})
            self.task_changed.emit()
        return _handler

    def _make_bool_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, cb.isChecked())
            self.task_changed.emit()
        return _handler

    def _make_scalar_edit_handler(self, key: str, original_val: Any, edit: QLineEdit):
        original_type = type(original_val)

        def _handler() -> None:
            if self._model is None or self._task_name is None:
                return
            try:
                new_val = original_type(edit.text())
            except (ValueError, TypeError):
                edit.setText(str(original_val))
                return
            self._model.set_task_option(self._task_name, key, new_val)
            self.task_changed.emit()

        return _handler

    def _make_complex_click_handler(self, key: str, lbl: QLabel):
        def _handler(_event) -> None:
            if self._model is None or self._task_name is None:
                return
            current_value = self._model.get_task_option(self._task_name, key)
            dlg = YamlEditDialog(self._task_name, key, current_value, self)
            if dlg.exec_() == QDialog.Accepted:
                new_value = dlg.get_value()
                self._model.set_task_option(self._task_name, key, new_value)
                lbl.setText(_preview_text(new_value))
                self.task_changed.emit()
        return _handler

    def _make_profile_handler(self, opt_key: str, profile_name: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(self._task_name, opt_key, profile_name, cb.isChecked())
            self.task_changed.emit()
        return _handler

    def _make_transform_bool_param_handler(
        self, opt_key: str, feature_name: str, param_key: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
            feature[param_key] = cb.isChecked()
            self._model._dirty = True
            self.task_changed.emit()
        return _handler

    def _make_transform_params_handler(self, opt_key: str, feature_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            feature_cfg = dict(opts.get(feature_name) or {})
            params = {
                k: v for k, v in feature_cfg.items()
                if k != "enabled" and not isinstance(v, bool)
            }
            dlg = YamlEditDialog(feature_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
                for k, v in new_params.items():
                    feature[k] = v
                for k in list(params.keys()):
                    if k not in new_params:
                        feature.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_feature_params_handler(self, opt_key: str, feature_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            feature_cfg = dict(opts.get(feature_name) or {})
            params = {k: v for k, v in feature_cfg.items() if k != "enabled"}
            dlg = YamlEditDialog(feature_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
                for k, v in new_params.items():
                    if k != "enabled":
                        feature[k] = v
                for k in list(feature_cfg.keys()):
                    if k != "enabled" and k not in new_params:
                        feature.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_profile_params_handler(self, opt_key: str, profile_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            profile_cfg = dict(opts.get(profile_name) or {})
            params = {k: v for k, v in profile_cfg.items() if k != "enabled"}
            dlg = YamlEditDialog(profile_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                profile = self._model.get_task_option(self._task_name, opt_key)[profile_name]
                for k, v in new_params.items():
                    if k != "enabled":
                        profile[k] = v
                for k in list(profile_cfg.keys()):
                    if k != "enabled" and k not in new_params:
                        profile.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_combination_edit_handler(self, opt_key: str, combo_name: str, preview_label: QLabel):
        def _handler(_event) -> None:
            if self._model is None or self._task_name is None:
                return
            current_features = self._model.get_combination_features(
                self._task_name, opt_key, combo_name
            )
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            existing = list(opts.keys())
            dlg = FeatureCombinationDialog(
                self._task_name,
                existing_names=[n for n in existing if n != combo_name],
                combo_name=combo_name,
                selected_features=current_features,
                parent=self,
            )
            if dlg.exec_() == QDialog.Accepted:
                new_combo_name = dlg.get_combo_name()
                original_combo_name = dlg.get_original_combo_name()
                new_features = dlg.get_selected_features()

                # Handle rename: remove old, add new
                if new_combo_name != original_combo_name:
                    self._model.remove_combination(self._task_name, opt_key, original_combo_name)
                    seq = CommentedSeq(new_features)
                    seq.fa.set_flow_style()
                    self._model.add_combination(
                        self._task_name, opt_key, new_combo_name, {"enabled": True, "features": seq}
                    )
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_combination_features(
                        self._task_name, opt_key, combo_name, new_features
                    )
                    preview_label.setText(f"[{', '.join(new_features)}]")
                    self.task_changed.emit()
        return _handler

    def _make_combination_add_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            existing = list(opts.keys())
            dlg = FeatureCombinationDialog(
                self._task_name,
                existing_names=existing,
                parent=self,
            )
            if dlg.exec_() == QDialog.Accepted:
                combo_name = dlg.get_combo_name()
                features = dlg.get_selected_features()
                seq = CommentedSeq(features)
                seq.fa.set_flow_style()
                self._model.add_combination(
                    self._task_name, opt_key, combo_name, {"enabled": True, "features": seq}
                )
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_combination_context_handler(self, opt_key: str, combo_name: str):
        def _handler(_pos) -> None:
            if self._model is None or self._task_name is None:
                return
            menu = QMenu(self)
            delete_action = menu.addAction(f"Delete '{combo_name}'")
            action = menu.exec_(QCursor.pos())
            if action == delete_action:
                reply = QMessageBox.question(
                    self,
                    "Delete Combination",
                    f"Delete combination '{combo_name}' from '{self._task_name}'?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._model.remove_combination(self._task_name, opt_key, combo_name)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
        return _handler

    def _make_cluster_group_enabled_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(
                self._task_name, opt_key, group_name, cb.isChecked()
            )
            self.task_changed.emit()
        return _handler

    def _make_cluster_group_edit_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox, summary_lbl: QLabel
    ):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            spec = self._model.get_cluster_group_spec(self._task_name, group_name)
            dlg = ClusterGroupDialog(self, name=group_name, spec=spec)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                if new_name != group_name:
                    self._model.remove_combination(self._task_name, opt_key, group_name)
                    self._model.set_cluster_group_spec(self._task_name, new_name, new_spec)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_cluster_group_spec(self._task_name, group_name, new_spec)
                    cb.setChecked(new_spec.get("enabled", True))
                    summary_lbl.setText(_cluster_group_summary(new_spec))
                    self.task_changed.emit()
        return _handler

    def _make_cluster_group_delete_handler(self, opt_key: str, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            reply = QMessageBox.question(
                self,
                "Delete Cluster Group",
                f"Delete cluster group '{group_name}' from '{self._task_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._model.remove_combination(self._task_name, opt_key, group_name)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_cluster_group_new_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            dlg = ClusterGroupDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                self._model.set_cluster_group_spec(self._task_name, new_name, new_spec)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_downstream_group_handler(
        self,
        opt_key: str,
        group_name: str,
        cb: QCheckBox,
        all_groups: list,
        selected: list,
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            if cb.isChecked():
                if group_name not in selected:
                    selected.append(group_name)
            else:
                if group_name in selected:
                    selected.remove(group_name)
            ordered = [g for g in all_groups if g in selected]
            self._model.set_downstream_cluster_group_names(self._task_name, ordered)
            self.task_changed.emit()
        return _handler

    def _make_downstream_group_details_handler(self, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None:
                return
            try:
                spec = self._model.get_cluster_group_spec("touch_clustering", group_name)
            except KeyError:
                return
            dlg = ClusterGroupReadOnlyDialog(group_name, spec, self)
            dlg.exec_()
        return _handler
