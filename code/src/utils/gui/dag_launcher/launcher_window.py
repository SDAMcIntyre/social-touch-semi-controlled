"""Main window for the DAG Config Launcher GUI."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from utils.gui.dag_launcher.kinect_directory_selector import SessionConfigSelector
from utils.gui.dag_launcher.launcher_config import WorkflowEntry
from utils.gui.dag_launcher.task_panel import TaskPanel
from utils.gui.dag_launcher.workflow_selector import WorkflowSelector
from utils.pipeline.dag_config_model import DagConfigModel


class LauncherWindow(QMainWindow):
    """Three-column GUI: workflow selector (left), tasks (center), kinect dirs (right)."""

    def __init__(
        self,
        entries: list[WorkflowEntry],
        configs_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entries = entries
        self._configs_dir = configs_dir
        self._current_entry: WorkflowEntry | None = None
        self._model: DagConfigModel | None = None
        self._initial_sizes_applied = False
        self._process: subprocess.Popen | None = None
        self._poll_timer: QTimer | None = None
        self._aborting: bool = False

        self.setWindowTitle("DAG Config Launcher")
        self.resize(1100, 700)

        self._build_toolbar()
        self._build_ui()
        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._save_action = QAction("Save", self)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.setEnabled(False)
        self._save_action.triggered.connect(self._on_save)
        tb.addAction(self._save_action)

        self._save_as_action = QAction("Save As\u2026", self)
        self._save_as_action.setShortcut("Ctrl+Shift+S")
        self._save_as_action.setEnabled(False)
        self._save_as_action.triggered.connect(self._on_save_as)
        tb.addAction(self._save_as_action)

    def _build_ui(self) -> None:
        # Three-column splitter
        self._splitter = QSplitter(Qt.Horizontal)

        # --- Left column: workflow selector ---
        self._workflow_selector = WorkflowSelector(self._entries)
        self._splitter.addWidget(self._workflow_selector)

        # --- Middle column: task panel (50%) ---
        self._task_panel = TaskPanel()
        self._splitter.addWidget(self._task_panel)

        # --- Right column: session config selector (25%) ---
        self._kinect_selector = SessionConfigSelector(self._configs_dir)
        self._splitter.addWidget(self._kinect_selector)

        self._splitter.setStretchFactor(0, 1)  # workflow selector  (1/4)
        self._splitter.setStretchFactor(1, 2)  # task panel        (1/2)
        self._splitter.setStretchFactor(2, 1)  # kinect selector   (1/4)

        # --- Wrap splitter + run bar in a central QWidget ---
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._splitter, stretch=1)
        layout.addWidget(self._build_run_bar())
        self.setCentralWidget(central)

        # --- Signals ---
        self._workflow_selector.workflow_changed.connect(self._load_workflow)
        self._kinect_selector.selection_changed.connect(self._on_kinect_changed)
        self._task_panel.task_changed.connect(self._mark_dirty)

    def _build_run_bar(self) -> QWidget:
        bar = QWidget()
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(6, 4, 6, 4)

        self._run_label = QLabel("No workflow selected")
        self._run_label.setStyleSheet("color: gray;")
        hbox.addWidget(self._run_label, stretch=1)

        self._run_button = QPushButton("Run")
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run)
        self._run_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 18px;"
            " padding: 8px 24px; border-radius: 4px; }"
            " QPushButton:hover { background-color: #45A049; }"
            " QPushButton:disabled { background-color: #A5D6A7; color: #E8E8E8; }"
        )
        hbox.addWidget(self._run_button)

        self._abort_button = QPushButton("Abort")
        self._abort_button.setVisible(False)
        self._abort_button.clicked.connect(self._on_abort)
        self._abort_button.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; font-size: 18px;"
            " padding: 8px 24px; border-radius: 4px; }"
            " QPushButton:hover { background-color: #D32F2F; }"
        )
        hbox.addWidget(self._abort_button)

        return bar

    # ------------------------------------------------------------------
    # Workflow loading
    # ------------------------------------------------------------------

    def _load_workflow(self, entry: WorkflowEntry) -> None:
        if self._model and self._model.dirty:
            if not self._confirm_discard():
                return
        self._current_entry = entry
        if entry.dag_config is None:
            self._model = None
            self._save_action.setEnabled(False)
            self._save_as_action.setEnabled(False)
            self._update_title()
            self._update_run_bar()
            return
        try:
            self._model = DagConfigModel(entry.dag_config)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._kinect_selector.populate(self._model)
        self._task_panel.populate(self._model)
        self._save_action.setEnabled(True)
        self._save_as_action.setEnabled(True)
        self._update_title()
        self._update_run_bar()
        self.statusBar().showMessage(f"Loaded {entry.dag_config.name}", 3000)

    # ------------------------------------------------------------------
    # Dirty-state management
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        if self._model:
            self._model._dirty = True
        self._update_title()

    def _update_title(self) -> None:
        base = "DAG Config Launcher"
        if self._model:
            dirty = " *" if self._model.dirty else ""
            self.setWindowTitle(f"{base} — {self._model.path.name}{dirty}")
        else:
            self.setWindowTitle(base)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        if not self._model:
            return
        try:
            self._model.save()
            self._update_title()
            self.statusBar().showMessage(f"Saved {self._model.path.name}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_save_as(self) -> None:
        if not self._model:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save DAG Config As",
            str(self._model.path),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            self._model.save_as(Path(path))
            self._update_title()
            self.statusBar().showMessage(f"Saved as {Path(path).name}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------
    # Kinect directory changes
    # ------------------------------------------------------------------

    def _on_kinect_changed(self) -> None:
        """Called when the user checks/unchecks a directory or file."""
        if not self._model:
            return
        self._model.set_config_entries(self._kinect_selector.get_selection())
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._initial_sizes_applied:
            w = self._splitter.width()
            if w > 0:
                self._splitter.setSizes([w // 4, w // 2, w // 4])
                self._initial_sizes_applied = True

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._process is not None:
            self._aborting = True
            self._process.terminate()
            self._process.wait()
            self._process = None
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None
        if self._model and self._model.dirty:
            if not self._confirm_discard():
                event.ignore()
                return
        event.accept()

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------

    def _update_run_bar(self) -> None:
        if self._current_entry is None:
            self._run_label.setText("No workflow selected")
            self._run_label.setStyleSheet("color: gray;")
            self._run_button.setEnabled(False)
            return
        if not self._current_entry.script.exists():
            self._run_label.setText("Script not found")
            self._run_label.setStyleSheet("color: gray;")
            self._run_button.setEnabled(False)
        else:
            self._run_label.setText(str(self._current_entry.script))
            self._run_label.setStyleSheet("")
            self._run_button.setEnabled(True)

    @staticmethod
    def _clear_prefect_db() -> None:
        """Remove the Prefect SQLite database to avoid stale 'database is locked' errors."""
        prefect_dir = Path.home() / ".prefect"
        for suffix in ("prefect.db", "prefect.db-wal", "prefect.db-shm"):
            db_file = prefect_dir / suffix
            if db_file.exists():
                try:
                    db_file.unlink()
                    logger.info("Removed %s", db_file)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", db_file, exc)

    def _on_run(self) -> None:
        if self._current_entry is None:
            return
        if self._model:
            self._on_save()
        self._clear_prefect_db()
        project_root = self._configs_dir.parent
        cmd = [sys.executable, str(self._current_entry.script)]
        if self._current_entry.dag_config is not None:
            cmd += ["--dag-config", str(self._current_entry.dag_config)]
        self._aborting = False
        self._process = subprocess.Popen(cmd, cwd=str(project_root))
        self._run_button.setEnabled(False)
        self._abort_button.setVisible(True)
        self.statusBar().showMessage("Running …")
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_process)
        self._poll_timer.start()

    def _poll_process(self) -> None:
        if self._process is None:
            return
        retcode = self._process.poll()
        if retcode is None:
            return  # still running
        self._poll_timer.stop()
        self._poll_timer = None
        self._abort_button.setVisible(False)
        if self._aborting:
            self.statusBar().showMessage("Aborted")
        else:
            label = "Finished" if retcode == 0 else "Failed"
            self.statusBar().showMessage(f"{label} (exit code {retcode})")
        self._run_button.setEnabled(True)
        self._process = None

    def _on_abort(self) -> None:
        """Terminate the running subprocess; let the poll timer handle cleanup."""
        if self._process is None:
            return
        self._aborting = True
        self._abort_button.setVisible(False)
        self.statusBar().showMessage("Aborting …")
        self._process.terminate()

    def _confirm_discard(self) -> bool:
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard
