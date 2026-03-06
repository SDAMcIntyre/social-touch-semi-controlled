# forearm_curation_gui.py

# --- Standard Library Imports ---
import logging
from typing import List, Optional

# --- Third-party Imports ---
import numpy as np
import open3d as o3d
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QStatusBar, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_FALLBACK_COLOR = 0.78  # light grey, used when the point cloud carries no RGB data


class ForearmCurationGUI(QMainWindow):
    """
    Interactive point cloud curation GUI for manual removal of forearm extraction artifacts.

    The operator can box-select unwanted points (table edges, clothing fragments,
    mis-segmented regions) before the cloud enters automated cleaning.

    Key features:
    - Selection Mode is active by default; press 'R' to toggle it.
    - Default filter mode is REMOVE: box selections mark points for removal.
    - Removed points are hidden by default; toggle "Show Removed Points" to reveal them as red spheres.
    - "Keep All" resets the mask; "Remove All" masks every point.
    - "Validate & Export" emits `curation_validated` with the list of removed indices.
    """

    curation_validated = pyqtSignal(list)

    def __init__(
        self,
        point_cloud: o3d.geometry.PointCloud,
        existing_removed_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.setWindowTitle("Forearm Curation - Point Removal")
        self.resize(1280, 850)

        self._pcd = point_cloud
        self._initial_removed = existing_removed_indices or []

        self._pv_cloud: Optional[pv.PolyData] = None
        self._excluded_mask: Optional[np.ndarray] = None
        self._kept_actor = None
        self._removed_actor = None
        self._kept_mesh: Optional[pv.PolyData] = None  # current picking target

        self._setup_ui()
        self._load_cloud_data()

        self.setFocusPolicy(Qt.StrongFocus)

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Viewport (left)
        self.plotter = QtInteractor(central)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plotter, stretch=3)

        # Controls panel (right)
        panel = QWidget()
        panel.setFixedWidth(300)
        ctrl = QVBoxLayout(panel)
        main_layout.addWidget(panel, stretch=1)

        # Instructions
        instr_group = QGroupBox("Workflow")
        instr_layout = QVBoxLayout()
        instr_label = QLabel(
            "<b>SHORTCUT: Press 'R'</b> to toggle Selection Mode.\n\n"
            "<b>IMPORTANT:</b> Click in the left 3D window first to activate it, "
            "otherwise 'R' may not register.\n\n"
            "1. Selection Mode ON by default: Drag box to select points.\n"
            "2. Navigation Mode: Rotate/Zoom freely.\n"
            "3. Toggle 'Show Removed Points' to see marked points.\n"
            "4. Colored points = kept."
        )
        instr_label.setWordWrap(True)
        instr_layout.addWidget(instr_label)
        instr_group.setLayout(instr_layout)
        ctrl.addWidget(instr_group)

        # Selection mode toggle
        mode_group = QGroupBox("Interaction Control")
        mode_layout = QVBoxLayout()
        self.chk_selection_active = QCheckBox("Activate Selection Mode")
        self.chk_selection_active.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #2196F3;"
        )
        self.chk_selection_active.setChecked(True)  # default: selection mode on
        self.chk_selection_active.toggled.connect(self._toggle_picking_tool)
        mode_layout.addWidget(self.chk_selection_active)
        self.chk_show_removed = QCheckBox("Hide Removed Points")
        self.chk_show_removed.setChecked(False)  # default: overlay hidden
        self.chk_show_removed.toggled.connect(self._refresh_visuals)
        mode_layout.addWidget(self.chk_show_removed)
        mode_group.setLayout(mode_layout)
        ctrl.addWidget(mode_group)

        # Filter behaviour (REMOVE is the default for this GUI)
        action_group = QGroupBox("Filter Behavior")
        action_layout = QVBoxLayout()
        action_layout.addWidget(QLabel("Define how selected points are treated:"))
        self.chk_filter_mode = QCheckBox("Toggle ON: Select to REMOVE")
        self.chk_filter_mode.setToolTip(
            "Checked (Default): box selection REMOVES points.\n"
            "Unchecked: box selection KEEPS (restores) points."
        )
        self.chk_filter_mode.setChecked(True)  # default: REMOVE mode
        action_layout.addWidget(self.chk_filter_mode)
        action_group.setLayout(action_layout)
        ctrl.addWidget(action_group)

        # Stats
        self.lbl_stats = QLabel("Removed: 0 points")
        ctrl.addWidget(self.lbl_stats)

        # Batch operations
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QVBoxLayout()
        btn_remove_all = QPushButton("Remove All Points")
        btn_remove_all.setStyleSheet("color: #D32F2F; font-weight: bold;")
        btn_remove_all.clicked.connect(self._remove_all)
        batch_layout.addWidget(btn_remove_all)
        btn_keep_all = QPushButton("Keep All Points")
        btn_keep_all.setStyleSheet("color: #388E3C; font-weight: bold;")
        btn_keep_all.clicked.connect(self._keep_all)
        batch_layout.addWidget(btn_keep_all)
        batch_group.setLayout(batch_layout)
        ctrl.addWidget(batch_group)

        ctrl.addStretch()

        btn_validate = QPushButton("Validate & Export")
        btn_validate.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; height: 40px;"
        )
        btn_validate.clicked.connect(self._finalize)
        ctrl.addWidget(btn_validate)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ──────────────────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_cloud_data(self):
        try:
            points = np.asarray(self._pcd.points)
            n_points = len(points)

            if n_points == 0:
                self.status_bar.showMessage("Empty point cloud — validate immediately to proceed.")
                return

            self._pv_cloud = pv.PolyData(points)

            # Explicitly create vertex cells so enable_cell_picking(through=True) has
            # cells to select. pv.PolyData(points) leaves n_cells=0 by default.
            verts = np.empty(2 * n_points, dtype=np.intp)
            verts[0::2] = 1                       # each cell contains 1 point
            verts[1::2] = np.arange(n_points)     # point index for that cell
            self._pv_cloud.verts = verts

            # Store original indices for pick-callback mapping
            self._pv_cloud.point_data["orig_ids"] = np.arange(n_points)

            # RGB colors: real if available, otherwise light grey
            if self._pcd.has_colors():
                colors = np.asarray(self._pcd.colors).astype(np.float32)
            else:
                colors = np.full((n_points, 3), _FALLBACK_COLOR, dtype=np.float32)
            self._pv_cloud.point_data["colors"] = colors

            # Exclusion mask: True = removed
            self._excluded_mask = np.zeros(n_points, dtype=bool)
            if self._initial_removed:
                valid = [i for i in self._initial_removed if 0 <= i < n_points]
                self._excluded_mask[valid] = True

            self.plotter.clear()
            self._refresh_visuals()
            self.plotter.reset_camera()
            self._update_stats()

            # Activate picking tool now that the cloud is ready (setChecked(True) in
            # _setup_ui fired before _pv_cloud existed, so _toggle_picking_tool was a no-op).
            if self.chk_selection_active.isChecked():
                self._toggle_picking_tool(True)

        except Exception as e:
            logging.error(f"Failed to load point cloud data: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Interaction
    # ──────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.chk_selection_active.setChecked(not self.chk_selection_active.isChecked())
        else:
            super().keyPressEvent(event)

    def _toggle_picking_tool(self, active: bool):
        if self._pv_cloud is None:
            return

        self.plotter.disable_picking()

        if active:
            mode = "REMOVE" if self.chk_filter_mode.isChecked() else "KEEP"
            self.status_bar.showMessage(
                f"Selection Mode (ON): Drag LEFT MOUSE to select points to {mode}."
            )
            self.plotter.enable_cell_picking(
                mesh=self._kept_mesh,
                callback=self._on_pick,
                show=False,
                through=True,
                font_size=10,
            )
        else:
            self.status_bar.showMessage("Navigation Mode: Rotate and Zoom. (Press 'R' to Select)")
            self.plotter.enable_trackball_style()

    def _reregister_picking(self):
        """Re-register cell picking after actors are rebuilt, if selection mode is active."""
        if self.chk_selection_active.isChecked():
            self._toggle_picking_tool(True)

    def _on_pick(self, picked):
        if picked is None:
            return
        try:
            selected_ids = picked.point_data.get("orig_ids")
            if selected_ids is None or len(selected_ids) == 0:
                return

            should_remove = self.chk_filter_mode.isChecked()
            self._excluded_mask[selected_ids] = should_remove

            self._refresh_visuals()
            self._update_stats()
        except Exception as e:
            logging.error(f"Pick callback error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Visual Updates
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_visuals(self, *_):
        # *_ absorbs the bool emitted by QCheckBox.toggled so the slot never raises TypeError.
        if self._pv_cloud is None or self._excluded_mask is None:
            return

        if self._kept_actor is not None:
            self.plotter.remove_actor(self._kept_actor)
            self._kept_actor = None
        if self._removed_actor is not None:
            self.plotter.remove_actor(self._removed_actor)
            self._removed_actor = None

        kept_indices = np.where(~self._excluded_mask)[0]
        removed_indices = np.where(self._excluded_mask)[0]

        # Kept points: always rendered with original RGB colors
        self._kept_mesh = None
        if len(kept_indices) > 0:
            self._kept_mesh = self._build_submesh(kept_indices, color_key="colors")
            self._kept_actor = self.plotter.add_mesh(
                self._kept_mesh,
                scalars="colors",
                rgb=True,
                point_size=5,
                render_points_as_spheres=True,
                style="points",
                pickable=True,
                reset_camera=False,
            )

        # Removed points: rendered in red only when toggle is OFF (not hiding)
        if len(removed_indices) > 0 and not self.chk_show_removed.isChecked():
            removed_mesh = self._build_submesh(removed_indices)
            self._removed_actor = self.plotter.add_mesh(
                removed_mesh,
                color="red",
                point_size=10,
                render_points_as_spheres=True,
                style="points",
                pickable=False,
                reset_camera=False,
            )

        self.plotter.render()

        # Actors were rebuilt — re-register cell picking on the new kept mesh.
        # Deferred via QTimer so we don't reconfigure picking from within a pick callback.
        if hasattr(self, 'chk_selection_active') and self.chk_selection_active.isChecked():
            QTimer.singleShot(0, self._reregister_picking)

    def _build_submesh(self, indices: np.ndarray, color_key: Optional[str] = None) -> pv.PolyData:
        """Build a vertex-cell PolyData for a subset of self._pv_cloud."""
        points = self._pv_cloud.points[indices]
        mesh = pv.PolyData(points)
        n = len(points)
        verts = np.empty(2 * n, dtype=np.intp)
        verts[0::2] = 1
        verts[1::2] = np.arange(n)
        mesh.verts = verts
        mesh.point_data["orig_ids"] = self._pv_cloud.point_data["orig_ids"][indices]
        if color_key is not None:
            mesh.point_data[color_key] = self._pv_cloud.point_data[color_key][indices]
        return mesh

    def _update_stats(self):
        if self._excluded_mask is None:
            return
        removed = int(np.count_nonzero(self._excluded_mask))
        total = int(self._excluded_mask.size)
        self.lbl_stats.setText(f"Removed: {removed} / {total} points")

    # ──────────────────────────────────────────────────────────────────────────
    # Batch Operations
    # ──────────────────────────────────────────────────────────────────────────

    def _keep_all(self):
        if self._excluded_mask is not None:
            self._excluded_mask[:] = False
            self._refresh_visuals()
            self._update_stats()

    def _remove_all(self):
        if self._excluded_mask is not None:
            self._excluded_mask[:] = True
            self._refresh_visuals()
            self._update_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # Finalisation
    # ──────────────────────────────────────────────────────────────────────────

    def _finalize(self):
        removed_indices = (
            np.where(self._excluded_mask)[0].tolist()
            if self._excluded_mask is not None
            else []
        )
        logging.info(f"Curation validated: {len(removed_indices)} points removed.")
        self.curation_validated.emit(removed_indices)
        self.close()
