"""Left-panel widget for selecting session config directories and files."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.pipeline.dag_config_model import DagConfigModel


class SessionConfigSelector(QWidget):
    """Checkable tree of session config directories and their YAML files.

    Replaces the former ``KinectDirectorySelector``.  The tree works uniformly
    for both ``kinect_configs`` and ``forearm_configs`` workflows.

    Selection model:
      - Fully-checked directory → emits the directory name as a single entry.
      - Partially-checked directory → emits individual ``subdir/file.yaml`` paths.
      - Unchecked directory / file → not included.
    """

    selection_changed = pyqtSignal()

    def __init__(self, configs_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs_dir = configs_dir
        self._config_root = configs_dir / "kinect_configs"
        self._model: DagConfigModel | None = None
        self._populating = False  # guard against signal storms

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._group = QGroupBox("Session Configs")
        group_layout = QVBoxLayout(self._group)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name"])
        self._tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self._tree.itemChanged.connect(self._on_item_changed)
        group_layout.addWidget(self._tree)

        layout.addWidget(self._group)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        self._model = model
        self._populating = True
        self._tree.clear()

        config_type = model.get_config_type()
        self._config_root = self._configs_dir / config_type
        _HEADERS = {
            "forearm_configs": "Forearm Session Configs",
            "kinect_configs":  "Kinect Session Configs",
        }
        self._group.setTitle(_HEADERS.get(config_type, f"{config_type}"))

        entries = model.get_config_entries()
        # Build a set of selected directory names and a set of selected file paths
        # (relative to config_root).
        selected_dirs: set[str] = set()
        selected_files: set[str] = set()
        for entry in entries:
            if "/" in entry or entry.endswith(".yaml"):
                # Looks like a file path (may be "subdir/file.yaml")
                selected_files.add(entry)
            else:
                selected_dirs.add(entry)

        if not self._config_root.is_dir():
            self._populating = False
            return

        # --- Subdirectory items ---
        for subdir in sorted(self._config_root.iterdir()):
            if not subdir.is_dir():
                continue
            dir_item = QTreeWidgetItem([subdir.name])
            dir_item.setData(0, Qt.UserRole, subdir.name)
            dir_item.setData(0, Qt.UserRole + 1, "dir")
            dir_item.setCheckState(0, Qt.Unchecked)  # derived from children below

            yaml_files = sorted(subdir.glob("*.yaml"))
            for yf in yaml_files:
                file_entry = f"{subdir.name}/{yf.name}"
                file_item = QTreeWidgetItem([yf.name])
                file_item.setData(0, Qt.UserRole, file_entry)
                file_item.setData(0, Qt.UserRole + 1, "file")
                checked = (
                    subdir.name in selected_dirs
                    or file_entry in selected_files
                )
                file_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                dir_item.addChild(file_item)

            self._update_parent_check_state(dir_item)
            self._tree.addTopLevelItem(dir_item)

            if dir_item.checkState(0) != Qt.Unchecked:
                dir_item.setExpanded(True)

        # --- Root-level YAML files (e.g. forearm_configs/*.yaml) ---
        root_yamls = sorted(self._config_root.glob("*.yaml"))
        if root_yamls:
            root_item = QTreeWidgetItem([config_type])
            root_item.setData(0, Qt.UserRole, ".")
            root_item.setData(0, Qt.UserRole + 1, "dir")
            root_item.setCheckState(0, Qt.Unchecked)  # derived from children

            for yf in root_yamls:
                checked = (
                    "." in selected_dirs
                    or yf.name in selected_files
                    or yf.name in {e.split("/")[-1] for e in selected_files}
                )
                file_item = QTreeWidgetItem([yf.name])
                file_item.setData(0, Qt.UserRole, yf.name)
                file_item.setData(0, Qt.UserRole + 1, "file")
                file_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                root_item.addChild(file_item)

            self._update_parent_check_state(root_item)
            self._tree.addTopLevelItem(root_item)
            root_item.setExpanded(True)

        self._populating = False

    # ------------------------------------------------------------------
    # Selection readout
    # ------------------------------------------------------------------

    def get_selection(self) -> list[str]:
        """Return the current selection as a list of config entries.

        A fully-checked directory emits its directory name as a single entry.
        A partially-checked directory emits individual ``subdir/file.yaml`` paths
        for each checked file child.
        """
        result: list[str] = []

        for i in range(self._tree.topLevelItemCount()):
            dir_item = self._tree.topLevelItem(i)
            state = dir_item.checkState(0)
            if state == Qt.Unchecked:
                continue

            dir_key = dir_item.data(0, Qt.UserRole)  # e.g. "valid_configs_ST13-01" or "."

            if state == Qt.Checked:
                # All children selected → emit just the directory name
                result.append(dir_key)
            else:
                # Partial → emit each checked child's file entry
                for j in range(dir_item.childCount()):
                    file_item = dir_item.child(j)
                    if file_item.checkState(0) == Qt.Checked:
                        result.append(file_item.data(0, Qt.UserRole))

        return result

    # ------------------------------------------------------------------
    # Internal signals
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._populating:
            return

        kind = item.data(0, Qt.UserRole + 1)
        self._populating = True

        if kind == "dir":
            # Propagate parent check state to all children
            new_state = item.checkState(0)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, new_state)

        elif kind == "file":
            parent = item.parent()
            if parent is not None:
                self._update_parent_check_state(parent)

        self._populating = False
        self.selection_changed.emit()

    @staticmethod
    def _update_parent_check_state(parent: QTreeWidgetItem) -> None:
        """Set parent to Checked/PartiallyChecked/Unchecked based on children."""
        total = parent.childCount()
        checked = sum(
            1 for i in range(total)
            if parent.child(i).checkState(0) == Qt.Checked
        )
        if checked == 0:
            parent.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            parent.setCheckState(0, Qt.Checked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)


# Backward-compatible alias
KinectDirectorySelector = SessionConfigSelector
