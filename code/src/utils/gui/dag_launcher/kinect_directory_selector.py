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
        self._kinect_root = configs_dir / "kinect_configs"
        self._model: DagConfigModel | None = None
        self._populating = False  # guard against signal storms

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Kinect Config Directories")
        group_layout = QVBoxLayout(group)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name"])
        self._tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self._tree.itemChanged.connect(self._on_item_changed)
        group_layout.addWidget(self._tree)

        layout.addWidget(group)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        self._model = model
        self._populating = True
        self._tree.clear()

        mode = model.get_kinect_dir_mode()
        selected_dirs = {d for d in model.get_kinect_directories()}
        excluded_files = set(model.get_exclude_files())

        if not self._kinect_root.is_dir():
            self._populating = False
            return

        for subdir in sorted(self._kinect_root.iterdir()):
            if not subdir.is_dir():
                continue
            rel = f"kinect_configs/{subdir.name}"
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

        self._populating = False

    # ------------------------------------------------------------------
    # Selection readout
    # ------------------------------------------------------------------

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

        # In single-select mode, uncheck other directories
        if kind == "dir" and self._model:
            mode = self._model.get_kinect_dir_mode()
            if mode == "single" and item.checkState(0) == Qt.Checked:
                self._populating = True
                for i in range(self._tree.topLevelItemCount()):
                    other = self._tree.topLevelItem(i)
                    if other is not item:
                        other.setCheckState(0, Qt.Unchecked)
                self._populating = False

        self.selection_changed.emit()
