"""Left-panel widget for selecting kinect config directories and files."""

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


class KinectDirectorySelector(QWidget):
    """Checkable tree of kinect config directories and their YAML files."""

    selection_changed = pyqtSignal()

    def __init__(self, configs_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs_dir = configs_dir
        self._config_root = configs_dir / "kinect_configs"
        self._model: DagConfigModel | None = None
        self._populating = False  # guard against signal storms

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._group = QGroupBox("Kinect Config Directories")
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

        root_name = model.get_config_dir_root_name()
        self._config_root = self._configs_dir / root_name
        _HEADERS = {
            "forearm_configs": "Forearm Config Directories",
            "kinect_configs":  "Kinect Config Directories",
        }
        self._group.setTitle(_HEADERS.get(root_name, f"{root_name} Directories"))

        mode = model.get_kinect_dir_mode()
        selected_dirs = {d for d in model.get_kinect_directories()}
        excluded_files = set(model.get_exclude_files())

        if not self._config_root.is_dir():
            self._populating = False
            return

        for subdir in sorted(self._config_root.iterdir()):
            if not subdir.is_dir():
                continue
            rel = f"{root_name}/{subdir.name}"
            dir_item = QTreeWidgetItem([subdir.name])
            dir_item.setData(0, Qt.UserRole, rel)
            dir_item.setData(0, Qt.UserRole + 1, "dir")

            # Check state based on selection
            if rel in selected_dirs:
                dir_item.setCheckState(0, Qt.Checked)
            else:
                dir_item.setCheckState(0, Qt.Unchecked)

            # Add child YAML files
            yaml_files = sorted(subdir.glob("*.yaml"))
            for yf in yaml_files:
                file_item = QTreeWidgetItem([yf.name])
                file_item.setData(0, Qt.UserRole, yf.name)
                file_item.setData(0, Qt.UserRole + 1, "file")
                if yf.name in excluded_files:
                    file_item.setCheckState(0, Qt.Unchecked)
                else:
                    file_item.setCheckState(0, Qt.Checked)
                dir_item.addChild(file_item)

            self._tree.addTopLevelItem(dir_item)

            # Expand selected directories so files are visible
            if rel in selected_dirs:
                dir_item.setExpanded(True)

        # If there are YAML files directly at the root (e.g. forearm_configs/),
        # show them as children of a top-level item whose rel IS the root name.
        # Use an include-list model: only files in forearm_config_files are checked;
        # if the list is empty every file starts unchecked (nothing selected = run all).
        root_yamls = sorted(self._config_root.glob("*.yaml"))
        if root_yamls:
            included_files = set(model.get_forearm_config_files())
            rel = root_name  # matches what the model stores, e.g. "forearm_configs"
            root_item = QTreeWidgetItem([root_name])
            root_item.setData(0, Qt.UserRole, rel)
            root_item.setData(0, Qt.UserRole + 1, "dir")
            if rel in selected_dirs:
                root_item.setCheckState(0, Qt.Checked)
            else:
                root_item.setCheckState(0, Qt.Unchecked)
            for yf in root_yamls:
                file_item = QTreeWidgetItem([yf.name])
                file_item.setData(0, Qt.UserRole, yf.name)
                file_item.setData(0, Qt.UserRole + 1, "file")
                file_item.setCheckState(
                    0, Qt.Checked if yf.name in included_files else Qt.Unchecked
                )
                root_item.addChild(file_item)
            self._tree.addTopLevelItem(root_item)
            root_item.setExpanded(True)

        self._populating = False

    # ------------------------------------------------------------------
    # Selection readout
    # ------------------------------------------------------------------

    def get_checked_forearm_files(self) -> list[str]:
        """Return filenames of checked file-items that are children of a root dir item.

        Used for the forearm include-list model: only these files will be processed.
        An empty list means no explicit selection, so the script will run all files.
        """
        result: list[str] = []
        for i in range(self._tree.topLevelItemCount()):
            dir_item = self._tree.topLevelItem(i)
            if dir_item.data(0, Qt.UserRole + 1) != "dir":
                continue
            for j in range(dir_item.childCount()):
                file_item = dir_item.child(j)
                if file_item.checkState(0) == Qt.Checked:
                    result.append(file_item.data(0, Qt.UserRole))
        return result

    def get_selection(self) -> tuple[list[str], list[str]]:
        """Return (selected_dirs, excluded_filenames)."""
        dirs: list[str] = []
        excluded: list[str] = []

        for i in range(self._tree.topLevelItemCount()):
            dir_item = self._tree.topLevelItem(i)
            rel = dir_item.data(0, Qt.UserRole)
            if dir_item.checkState(0) == Qt.Checked:
                dirs.append(rel)
                # Gather unchecked files within this directory
                for j in range(dir_item.childCount()):
                    file_item = dir_item.child(j)
                    if file_item.checkState(0) != Qt.Checked:
                        excluded.append(file_item.data(0, Qt.UserRole))

        return dirs, excluded

    # ------------------------------------------------------------------
    # Internal signals
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._populating:
            return

        kind = item.data(0, Qt.UserRole + 1)

        if kind == "dir" and self._model:
            self._populating = True

            # Propagate parent check state to all children
            new_state = item.checkState(0)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, new_state)

            # In single-select mode, uncheck other directories (and their children)
            mode = self._model.get_kinect_dir_mode()
            if mode == "single" and new_state == Qt.Checked:
                for i in range(self._tree.topLevelItemCount()):
                    other = self._tree.topLevelItem(i)
                    if other is not item:
                        other.setCheckState(0, Qt.Unchecked)
                        for j in range(other.childCount()):
                            other.child(j).setCheckState(0, Qt.Unchecked)

            self._populating = False

        self.selection_changed.emit()
