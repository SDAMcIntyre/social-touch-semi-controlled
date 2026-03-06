"""Workflow selector widget — toggle buttons to pick a DAG YAML config file."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Desired display order for workflow buttons (stem without the _dag suffix).
# Any file not listed here is appended alphabetically at the end.
_ORDERED_STEMS = [
    "primary_workflow_kinect_auto",
    "preprocess_pipeline_extract_forearm_manual",
    "preprocess_workflow_kinect_auto",
    "preprocess_workflow_kinect_manual",
    "preprocess_workflow_kinect_visualisation",
    "merging_pipeline_neuron_to_kinect_auto",
    "merging_view_neural_kinect",
    "postprocess_workflow_kinect_auto",
    "analysis_workflow",
]


def _category_prefix(stem: str) -> str:
    """Return the category prefix (text before the first ``_``) of a DAG stem."""
    return stem.split("_")[0]


class WorkflowSelector(QWidget):
    """Left-column widget with exclusive toggle buttons for each ``*_dag.yaml`` file."""

    workflow_changed = pyqtSignal(Path)  # emits the selected YAML path

    def __init__(self, configs_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs_dir = configs_dir
        self._paths: list[Path] = []
        self._buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Workflow")
        self._btn_layout = QVBoxLayout(group_box)
        self._btn_layout.setContentsMargins(6, 4, 6, 4)
        self._btn_layout.setSpacing(4)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.idClicked.connect(self._on_button_clicked)

        layout.addWidget(group_box)

        self._scan()

    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Populate the button column with all ``*_dag.yaml`` files in defined order."""
        all_paths = list(self._configs_dir.glob("*_dag.yaml"))

        def _order_key(p: Path) -> tuple[int, str]:
            stem = p.stem.replace("_dag", "")
            try:
                return (_ORDERED_STEMS.index(stem), "")
            except ValueError:
                return (len(_ORDERED_STEMS), stem)

        self._paths = sorted(all_paths, key=_order_key)
        current_category: str | None = None
        for i, p in enumerate(self._paths):
            stem = p.stem.replace("_dag", "")
            category = _category_prefix(stem)
            if category != current_category:
                current_category = category
                header = QLabel(category.title())
                header.setStyleSheet(
                    "QLabel { font-size: 10px; font-weight: bold; color: #888888;"
                    " margin-top: 6px; margin-bottom: 1px; }"
                )
                self._btn_layout.addWidget(header)
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFrameShadow(QFrame.Sunken)
                self._btn_layout.addWidget(sep)
            # Derive a short human-readable label from the stem
            label = stem.replace("_", " ").title()
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(p.name)
            btn.setStyleSheet(
                "QPushButton { padding: 4px 10px; }"
                "QPushButton:checked { background-color: #4a90d9; color: white; "
                "font-weight: bold; }"
            )
            self._btn_group.addButton(btn, i)
            self._btn_layout.addWidget(btn)
            self._buttons.append(btn)

        self._btn_layout.addStretch()

    def _on_button_clicked(self, btn_id: int) -> None:
        if 0 <= btn_id < len(self._paths):
            self.workflow_changed.emit(self._paths[btn_id])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def current_path(self) -> Path | None:
        btn_id = self._btn_group.checkedId()
        if 0 <= btn_id < len(self._paths):
            return self._paths[btn_id]
        return None

    def select_path(self, path: Path) -> None:
        """Programmatically select a workflow by its path."""
        for i, p in enumerate(self._paths):
            if p == path:
                self._buttons[i].setChecked(True)
                return
