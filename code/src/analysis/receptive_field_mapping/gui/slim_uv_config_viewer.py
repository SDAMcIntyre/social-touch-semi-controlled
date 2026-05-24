"""Per-session SLIM UV config viewer.

Interactive PyQt5/PyVista GUI for choosing the mesh method, cleaning toggles,
and SLIM iteration count for a single forearm session. The researcher can
press ``Process`` to run the pipeline in a background thread, inspect each
intermediate step in the embedded PyVista viewer, then ``Accept`` to persist
the chosen settings as ``slim_uv_config.yaml`` next to the session outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.gui.slim_uv_process_worker import (
    SlimUvProcessWorker,
)
from analysis.receptive_field_mapping.gui.slim_uv_steps_viewer import (
    SlimStep,
    _faces_to_vtk,
)
from analysis.receptive_field_mapping.surface.slim_uv_config_io import (
    SlimUvCleanSteps,
    SlimUvConfig,
    config_path_for_session,
    load_slim_uv_config,
    make_default_config,
    save_slim_uv_config,
)

logger = logging.getLogger(__name__)

_GREEN_BG = QColor("#90EE90")

_CLEAN_STEP_LABELS: tuple[tuple[str, str], ...] = (
    ("remove_non_manifold", "Remove Non-Manifold"),
    ("repair_pinch_vertices", "Repair Pinch Vertices"),
    ("remove_slivers", "Remove Slivers"),
    ("stitch_boundary_gaps", "Stitch Boundary Gaps"),
    ("fill_interior_holes", "Fill Interior Holes"),
)


class SlimUvConfigViewer(QMainWindow):
    """Per-session interactive SLIM UV config GUI.

    Constructor argument shape (consumed by the Phase 5 launcher):

    ``sessions``
        List of dicts, one per session, with keys:
            - ``session_id`` (str)
            - ``forearm_ply_path`` (pathlib.Path)
            - ``rf_maps_npz`` (pathlib.Path)
            - ``output_dir`` (pathlib.Path) — root folder under which
              ``<session_id>/slim_uv_config.yaml`` is read/written

    ``dag_defaults``
        Dict of DAG-level defaults applied when a session has no saved config.
        Recognised keys (any subset, forwarded to ``make_default_config``):
        ``mesh_method``, ``max_edge_mm``, ``clean_steps``, ``n_iter``,
        ``save_diagnostics``.
    """

    def __init__(
        self,
        sessions: list[dict],
        dag_defaults: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not sessions:
            raise ValueError("SlimUvConfigViewer: sessions list is empty.")
        for i, sess in enumerate(sessions):
            for key in ("session_id", "forearm_ply_path", "rf_maps_npz", "output_dir"):
                if key not in sess:
                    raise ValueError(
                        f"SlimUvConfigViewer: sessions[{i}] missing required key {key!r}."
                    )

        self._sessions = sessions
        self._dag_defaults = dict(dag_defaults)
        self._initialized = False
        self._current_session: Optional[dict] = None
        self._worker: Optional[SlimUvProcessWorker] = None
        self._last_steps: list[SlimStep] = []

        self.setWindowTitle("SLIM UV Config")

        # Toolbar with session combo.
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("Session: "))
        self._session_combo = QComboBox()
        for sess in sessions:
            self._session_combo.addItem(sess["session_id"])
        toolbar.addWidget(self._session_combo)
        self.addToolBar(toolbar)

        # Left config panel.
        self._config_panel = self._build_config_panel()

        # Right viewer placeholder (real QtInteractor created in _deferred_start).
        self._viewer_placeholder = QFrame()
        self._viewer_placeholder.setFrameShape(QFrame.StyledPanel)
        ph_layout = QVBoxLayout(self._viewer_placeholder)
        ph_layout.addWidget(
            QLabel("Initialising 3D viewer..."),
            alignment=Qt.AlignCenter,
        )
        self._plotter: Optional[QtInteractor] = None

        # Center horizontal splitter (config | viewer).
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._config_panel)
        top_splitter.addWidget(self._viewer_placeholder)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([360, 1200])

        # Bottom step navigator panel (hidden until first successful Process).
        self._step_panel = self._build_step_panel()
        self._step_panel.hide()

        # Vertical splitter: top_splitter / step_panel.
        outer_splitter = QSplitter(Qt.Vertical)
        outer_splitter.addWidget(top_splitter)
        outer_splitter.addWidget(self._step_panel)
        outer_splitter.setStretchFactor(0, 1)
        outer_splitter.setStretchFactor(1, 0)

        self.setCentralWidget(outer_splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._mesh_method_combo.currentTextChanged.connect(self._on_mesh_method_changed)
        self._process_btn.clicked.connect(self._on_process_clicked)
        self._accept_btn.clicked.connect(self._on_accept_clicked)
        self._skip_btn.clicked.connect(self._on_skip_clicked)
        self._close_btn.clicked.connect(self.close)
        self._step_list.currentRowChanged.connect(self._on_step_selected)

        self._update_combo_backgrounds()

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Mesh method group.
        mesh_group = QGroupBox("Mesh Method")
        mesh_form = QFormLayout(mesh_group)
        self._mesh_method_combo = QComboBox()
        self._mesh_method_combo.addItems(["bpa", "delaunay"])
        mesh_form.addRow("Method:", self._mesh_method_combo)
        self._max_edge_spin = QDoubleSpinBox()
        self._max_edge_spin.setRange(0.0, 100.0)
        self._max_edge_spin.setSingleStep(0.1)
        self._max_edge_spin.setDecimals(3)
        self._max_edge_spin.setSuffix(" mm")
        self._max_edge_spin.setToolTip("0.0 = auto (3.0 * mean nearest-neighbour distance)")
        mesh_form.addRow("Max edge:", self._max_edge_spin)
        layout.addWidget(mesh_group)

        # Clean steps group.
        clean_group = QGroupBox("Clean Steps")
        clean_layout = QVBoxLayout(clean_group)
        self._clean_checkboxes: dict[str, QCheckBox] = {}
        for field_name, label in _CLEAN_STEP_LABELS:
            cb = QCheckBox(label)
            clean_layout.addWidget(cb)
            self._clean_checkboxes[field_name] = cb
        layout.addWidget(clean_group)

        # SLIM iterations + diagnostics group.
        slim_group = QGroupBox("SLIM")
        slim_form = QFormLayout(slim_group)
        self._n_iter_spin = QSpinBox()
        self._n_iter_spin.setRange(1, 200)
        slim_form.addRow("Iterations:", self._n_iter_spin)
        self._save_diag_cb = QCheckBox("Save diagnostics")
        slim_form.addRow(self._save_diag_cb)
        layout.addWidget(slim_group)

        # Action buttons.
        actions_row1 = QHBoxLayout()
        self._process_btn = QPushButton("Process")
        self._accept_btn = QPushButton("Accept")
        actions_row1.addWidget(self._process_btn)
        actions_row1.addWidget(self._accept_btn)
        layout.addLayout(actions_row1)

        actions_row2 = QHBoxLayout()
        self._skip_btn = QPushButton("Skip")
        self._close_btn = QPushButton("Close")
        actions_row2.addWidget(self._skip_btn)
        actions_row2.addWidget(self._close_btn)
        layout.addLayout(actions_row2)

        # Status label for worker progress.
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addStretch(1)
        return panel

    def _build_step_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        lay = QHBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        self._step_list = QListWidget()
        self._step_list.setMinimumWidth(240)
        self._step_list.setMaximumWidth(360)
        self._step_info = QLabel("")
        self._step_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._step_info.setWordWrap(True)
        lay.addWidget(self._step_list)
        lay.addWidget(self._step_info, stretch=1)
        return panel

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        # Create the PyVista interactor lazily — eager init in __init__ has
        # been known to crash on some Windows/Mesa configurations.
        self._plotter = QtInteractor(self)
        self._plotter.set_background("white")

        parent_layout = self._viewer_placeholder.layout()
        # Clear placeholder widgets.
        while parent_layout.count():
            item = parent_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self._plotter.interactor)

        try:
            self._plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self._plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self._plotter.render_window.SetSize(sz.width(), sz.height())

        self._load_session(self._sessions[0])

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        if self._plotter is not None:
            self._plotter.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _on_session_changed(self, index: int) -> None:
        if 0 <= index < len(self._sessions):
            self._load_session(self._sessions[index])

    def _load_session(self, session: dict) -> None:
        self._current_session = session

        cfg_path = config_path_for_session(
            session["output_dir"], session["session_id"]
        )
        if cfg_path.exists():
            config = load_slim_uv_config(cfg_path)
        else:
            config = make_default_config(
                session_id=session["session_id"], **self._dag_defaults
            )

        self._populate_widgets_from_config(config)
        self._clear_step_navigator()
        self._status_label.setText("")
        self._render_raw_point_cloud(session["forearm_ply_path"])
        self._status.showMessage(
            f"Session: {session['session_id']}"
            + ("  (config loaded)" if cfg_path.exists() else "  (defaults)")
        )

    def _populate_widgets_from_config(self, config: SlimUvConfig) -> None:
        widgets = (
            self._mesh_method_combo,
            self._max_edge_spin,
            self._n_iter_spin,
            self._save_diag_cb,
            *self._clean_checkboxes.values(),
        )
        for w in widgets:
            w.blockSignals(True)
        try:
            idx = self._mesh_method_combo.findText(config.mesh_method)
            if idx < 0:
                raise ValueError(
                    f"mesh_method {config.mesh_method!r} not in combo box options."
                )
            self._mesh_method_combo.setCurrentIndex(idx)
            self._max_edge_spin.setValue(float(config.max_edge_mm))
            self._n_iter_spin.setValue(int(config.n_iter))
            self._save_diag_cb.setChecked(bool(config.save_diagnostics))
            clean_dict = config.clean_steps.to_dict()
            for field_name, cb in self._clean_checkboxes.items():
                cb.setChecked(bool(clean_dict[field_name]))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._apply_mesh_method_enablement(config.mesh_method)

    def _apply_mesh_method_enablement(self, mesh_method: str) -> None:
        self._max_edge_spin.setEnabled(mesh_method == "delaunay")

    def _on_mesh_method_changed(self, text: str) -> None:
        self._apply_mesh_method_enablement(text)

    # ------------------------------------------------------------------
    # Build SlimUvConfig from widget state
    # ------------------------------------------------------------------

    def _build_config_from_widgets(self) -> SlimUvConfig:
        if self._current_session is None:
            raise ValueError("No session is currently loaded.")
        clean_steps = SlimUvCleanSteps(
            **{
                field_name: cb.isChecked()
                for field_name, cb in self._clean_checkboxes.items()
            }
        )
        return SlimUvConfig(
            session_id=self._current_session["session_id"],
            mesh_method=self._mesh_method_combo.currentText(),
            max_edge_mm=float(self._max_edge_spin.value()),
            n_iter=int(self._n_iter_spin.value()),
            save_diagnostics=self._save_diag_cb.isChecked(),
            clean_steps=clean_steps,
        )

    # ------------------------------------------------------------------
    # Process button + worker plumbing
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self._mesh_method_combo,
            self._max_edge_spin,
            self._n_iter_spin,
            self._save_diag_cb,
            self._process_btn,
            self._accept_btn,
            self._skip_btn,
            self._session_combo,
            *self._clean_checkboxes.values(),
        ):
            w.setEnabled(enabled)
        if enabled:
            self._apply_mesh_method_enablement(self._mesh_method_combo.currentText())

    def _on_process_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._current_session is None:
            raise ValueError("No session is currently loaded.")

        try:
            config = self._build_config_from_widgets()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return

        self._set_controls_enabled(False)
        self._status_label.setText("Starting...")

        self._worker = SlimUvProcessWorker(
            forearm_ply_path=self._current_session["forearm_ply_path"],
            rf_maps_npz=self._current_session["rf_maps_npz"],
            config=config,
            parent=self,
        )
        self._worker.progress.connect(self._status_label.setText)
        self._worker.result_ready.connect(self._on_process_done)
        self._worker.start()

    def _on_process_done(self, result_dict: dict) -> None:
        self._set_controls_enabled(True)
        if result_dict.get("ok"):
            steps = result_dict.get("steps") or []
            self._last_steps = list(steps)
            self._populate_step_list(self._last_steps)
            self._step_panel.show()
            if self._last_steps:
                self._step_list.setCurrentRow(len(self._last_steps) - 1)
        else:
            err = result_dict.get("error") or "Unknown error"
            self._clear_step_navigator()
            QMessageBox.critical(self, "SLIM UV Failed", err)

    # ------------------------------------------------------------------
    # Step navigator
    # ------------------------------------------------------------------

    def _populate_step_list(self, steps: list[SlimStep]) -> None:
        self._step_list.blockSignals(True)
        self._step_list.clear()
        for i, step in enumerate(steps):
            self._step_list.addItem(f"{i + 1:2d}. {step.label}")
        self._step_list.blockSignals(False)

    def _clear_step_navigator(self) -> None:
        self._last_steps = []
        self._step_list.blockSignals(True)
        self._step_list.clear()
        self._step_list.blockSignals(False)
        self._step_info.setText("")
        self._step_panel.hide()

    def _on_step_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._last_steps):
            return
        step = self._last_steps[row]
        self._render_step(step)
        self._step_info.setText(f"<b>{step.label}</b><br>{step.info}")

    # ------------------------------------------------------------------
    # PyVista rendering
    # ------------------------------------------------------------------

    def _render_raw_point_cloud(self, forearm_ply_path: Path) -> None:
        if self._plotter is None:
            return
        try:
            mesh = pv.read(str(forearm_ply_path))
        except Exception as exc:
            self._plotter.clear()
            logger.warning("Failed to read PLY %s: %s", forearm_ply_path, exc)
            self._plotter.render()
            return
        self._plotter.clear()
        self._plotter.add_mesh(
            mesh,
            color="#888888",
            point_size=3,
            render_points_as_spheres=False,
            name="forearm_raw",
        )
        self._plotter.reset_camera()
        self._plotter.render()

    def _render_step(self, step: SlimStep) -> None:
        # Rendering mirrors SlimUvStepsViewer._render_step — keep behaviour aligned.
        if self._plotter is None:
            return
        self._plotter.clear()

        V = step.V
        F = step.F
        if V.shape[1] == 2:
            V_3d = np.column_stack([V, np.zeros(len(V), dtype=V.dtype)])
        else:
            V_3d = V

        if len(F) == 0:
            if step.vertex_colors is not None:
                rgb = (np.asarray(step.vertex_colors)[:, :3] * 255).astype(np.uint8)
                pcd_data = pv.PolyData(V_3d)
                pcd_data.point_data["rgb"] = rgb
                self._plotter.add_mesh(
                    pcd_data, scalars="rgb", rgb=True,
                    point_size=3, render_points_as_spheres=False, name="forearm",
                )
            else:
                self._plotter.add_points(
                    V_3d, color="#888888", point_size=3,
                    render_points_as_spheres=False, name="forearm",
                )
        else:
            mesh = pv.PolyData(V_3d, _faces_to_vtk(F))
            if step.face_scalars is not None:
                mesh.cell_data["scalar"] = np.asarray(step.face_scalars, dtype=np.float64)
                self._plotter.add_mesh(
                    mesh,
                    scalars="scalar",
                    cmap=step.cmap or "viridis",
                    clim=step.clim,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": step.scalar_bar_title or ""},
                    show_edges=False,
                    name="forearm",
                )
            elif step.vertex_colors is not None:
                rgb = (np.asarray(step.vertex_colors)[:, :3] * 255).astype(np.uint8)
                mesh.point_data["rgb"] = rgb
                self._plotter.add_mesh(
                    mesh,
                    scalars="rgb",
                    rgb=True,
                    show_edges=True,
                    edge_color="#404040",
                    line_width=0.3,
                    name="forearm",
                )
            else:
                self._plotter.add_mesh(
                    mesh,
                    color="#d8c8b8",
                    show_edges=True,
                    edge_color="#404040",
                    line_width=0.3,
                    name="forearm",
                )

            for name, pts in step.overlay_points.items():
                if pts is None or len(pts) == 0:
                    continue
                pts_arr = np.asarray(pts, dtype=np.float64)
                if pts_arr.ndim == 1:
                    pts_arr = pts_arr[np.newaxis, :]
                if pts_arr.shape[1] == 2:
                    pts_arr = np.column_stack(
                        [pts_arr, np.zeros(len(pts_arr), dtype=pts_arr.dtype)]
                    )
                color = step.overlay_colors.get(name, "red")
                point_size = 18 if name == "centroid" else 6
                self._plotter.add_points(
                    pts_arr,
                    color=color,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    name=f"overlay_{name}",
                )

        if V.shape[1] == 2:
            self._plotter.view_xy()
        self._plotter.reset_camera()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Accept / Skip / combo background
    # ------------------------------------------------------------------

    def _on_accept_clicked(self) -> None:
        if self._current_session is None:
            raise ValueError("No session is currently loaded.")
        config = self._build_config_from_widgets()
        cfg_path = config_path_for_session(
            self._current_session["output_dir"],
            self._current_session["session_id"],
        )
        save_slim_uv_config(cfg_path, config)
        self._update_combo_backgrounds()
        QMessageBox.information(
            self,
            "Saved",
            f"Config saved to:\n{cfg_path}",
        )
        self._status.showMessage(f"Saved {cfg_path}")

    def _on_skip_clicked(self) -> None:
        next_idx = self._find_next_unconfigured_index()
        if next_idx is not None:
            self._session_combo.setCurrentIndex(next_idx)
            return
        reply = QMessageBox.question(
            self,
            "Done",
            "All sessions configured. Close?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.close()

    def _find_next_unconfigured_index(self) -> Optional[int]:
        current = self._session_combo.currentIndex()
        n = len(self._sessions)
        for offset in range(1, n + 1):
            idx = (current + offset) % n
            sess = self._sessions[idx]
            cfg_path = config_path_for_session(
                sess["output_dir"], sess["session_id"]
            )
            if not cfg_path.exists():
                return idx
        return None

    def _update_combo_backgrounds(self) -> None:
        combo = self._session_combo
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                sess = self._sessions[i]
                cfg_path = config_path_for_session(
                    sess["output_dir"], sess["session_id"]
                )
                if cfg_path.exists():
                    combo.setItemData(i, _GREEN_BG, Qt.BackgroundRole)
                else:
                    combo.setItemData(i, None, Qt.BackgroundRole)
        finally:
            combo.blockSignals(False)
