# hand_mask_selector_gui.py

# --- Standard Library Imports ---
import logging
from pathlib import Path
from typing import List, Optional

# --- Third-party Imports ---
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QRadioButton, QGroupBox,
    QLabel, QPushButton, QSizePolicy, QMessageBox, QButtonGroup, QCheckBox,
    QStatusBar
)
from PyQt5.QtCore import Qt, pyqtSignal
from pyvistaqt import QtInteractor

# --- Local Application Imports ---
from ..core.hand_mesh_processor import HandMeshProcessor

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandMaskSelectorGUI(QMainWindow):
    """
    Advanced GUI for interactive hand mesh vertex masking.
    
    Key Features:
    - Press 'R' to toggle Selection Mode.
    - Visualize vertices as points (Small Tan = Keep, Large Red = Remove).
    - Batch operations for Mask All / Reset All.
    """

    # Signal to notify parent/external systems
    selection_validated = pyqtSignal(list)

    def __init__(
        self,
        hand_models_dir: Path,
        model_name: str,
        is_left_hand: bool,
        existing_excluded_indices: Optional[List[int]] = None
    ):
        super().__init__()
        self.setWindowTitle("Hand Mask Architect - Vertex Filter")
        self.resize(1280, 850)

        # --- Data Model ---
        self.hand_models_dir = Path(hand_models_dir)
        self.model_name = model_name
        self.is_left_hand = is_left_hand
        self.model_path = self.hand_models_dir / self.model_name
        
        # Mask State: False (0) = Keep, True (1) = Remove
        self.excluded_mask: Optional[np.ndarray] = None 
        self.initial_indices = existing_excluded_indices or []
        self.result_indices: List[int] = []

        # PyVista Objects
        self.pv_mesh: Optional[pv.PolyData] = None
        self.selected_actor = None
        
        # --- UI Initialization ---
        self._setup_ui()
        self._load_mesh_data()

        # Ensure the window accepts key events for the 'R' shortcut
        self.setFocusPolicy(Qt.StrongFocus)

    def _setup_ui(self):
        """Constructs the PyQt layout and widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # 1. 3D Viewport (Left)
        self.plotter = QtInteractor(self.central_widget)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.plotter, stretch=3)

        # 2. Controls Panel (Right)
        self.controls_panel = QWidget()
        self.controls_panel.setFixedWidth(300)
        self.ctrl_layout = QVBoxLayout(self.controls_panel)
        self.main_layout.addWidget(self.controls_panel, stretch=1)

        # --- Instruction Group ---
        instr_group = QGroupBox("Workflow")
        instr_layout = QVBoxLayout()
        # Modified instruction text as per requirements
        instr_label = QLabel(
            "<b>SHORTCUT: Press 'R'</b> to toggle Selection Mode.\n\n"
            "<b>IMPORTANT:</b> Click in the left 3D window first to activate it, otherwise the 'R' key may not register.\n\n"
            "1. Navigation Mode: Rotate/Zoom freely.\n"
            "2. Selection Mode: Drag box to select dots.\n"
            "3. Red Dots = Selected (Removed).\n"
            "4. Tan Dots = Unselected (Kept)."
        )
        instr_label.setWordWrap(True)
        instr_layout.addWidget(instr_label)
        instr_group.setLayout(instr_layout)
        self.ctrl_layout.addWidget(instr_group)

        # --- Interaction Mode Control ---
        mode_group = QGroupBox("Interaction Control")
        mode_layout = QVBoxLayout()
        
        # Renamed Checkbox as per requirements
        self.chk_selection_active = QCheckBox("Activate Manual Selection")
        self.chk_selection_active.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        self.chk_selection_active.toggled.connect(self._toggle_picking_tool)
        
        mode_layout.addWidget(self.chk_selection_active)
        mode_group.setLayout(mode_layout)
        self.ctrl_layout.addWidget(mode_group)

        # --- Mask Logic Group ---
        # Replaced Radio Buttons with a Boolean Toggle State
        action_group = QGroupBox("Filter Behavior")
        action_layout = QVBoxLayout()
        
        self.lbl_behavior_desc = QLabel("Define how selected vertices are treated:")
        self.lbl_behavior_desc.setWordWrap(True)
        action_layout.addWidget(self.lbl_behavior_desc)

        self.chk_filter_mode = QCheckBox("Toggle ON: Select to REMOVE")
        self.chk_filter_mode.setToolTip(
            "Unchecked (Default): Selection KEEPS vertices.\n"
            "Checked: Selection FILTERS OUT (deletes/hides) vertices."
        )
        # Default State 1: Unchecked. Selected vertices are KEPT (not filtered).
        self.chk_filter_mode.setChecked(False) 
        
        action_layout.addWidget(self.chk_filter_mode)
        action_group.setLayout(action_layout)
        self.ctrl_layout.addWidget(action_group)

        # --- Statistics / Reset ---
        self.lbl_stats = QLabel("Selected: 0 vertices")
        self.ctrl_layout.addWidget(self.lbl_stats)

        # Batch Operations Group
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QVBoxLayout()

        # NEW BUTTON: Mask All
        self.btn_mask_all = QPushButton("Mask All Vertices (Remove All)")
        self.btn_mask_all.setStyleSheet("color: #D32F2F; font-weight: bold;")
        self.btn_mask_all.clicked.connect(self._mask_all)
        batch_layout.addWidget(self.btn_mask_all)

        # EXISTING BUTTON: Reset
        self.btn_reset = QPushButton("Keep All Vertices")
        self.btn_reset.setStyleSheet("color: #388E3C; font-weight: bold;")
        self.btn_reset.clicked.connect(self._reset_mask)
        batch_layout.addWidget(self.btn_reset)

        batch_group.setLayout(batch_layout)
        self.ctrl_layout.addWidget(batch_group)

        self.ctrl_layout.addStretch()

        # --- Finalize ---
        self.btn_save = QPushButton("Validate & Export")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.btn_save.clicked.connect(self._finalize_selection)
        self.ctrl_layout.addWidget(self.btn_save)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_R:
            # Toggle the checkbox state, which triggers _toggle_picking_tool
            new_state = not self.chk_selection_active.isChecked()
            self.chk_selection_active.setChecked(new_state)
        else:
            super().keyPressEvent(event)

    def _load_mesh_data(self):
        """Loads mesh, converts to PyVista, initializes visualization layers."""
        if not self.model_path.exists():
            QMessageBox.critical(self, "IO Error", f"Model not found:\n{self.model_path}")
            return

        try:
            # 1. Load via Domain Processor
            # Note: Ensure HandMeshProcessor is available in context or mock it.
            mesh_params = {'left': self.is_left_hand}
            
            # Using the mock class if running standalone, or real import otherwise
            if 'HandMeshProcessor' not in globals():
                 # Fallback if class not imported (should be handled by __main__ block or import)
                 # In a real scenario, this import should be active.
                 pass
            
            # Note: In a production run, ensure HandMeshProcessor is imported. 
            # For this context, we assume the class definition in __main__ or import works.
            # If standard import fails, we rely on the logic in __main__
            if 'HandMeshProcessor' in globals():
                o3d_mesh = HandMeshProcessor.create_mesh(self.model_path, mesh_params)
            else:
                # Fallback for runtime safety if running purely as GUI test without core
                raise ImportError("HandMeshProcessor dependency missing.")
            
            # 2. Convert Open3D -> PyVista
            verts = np.asarray(o3d_mesh.vertices)
            tris = np.asarray(o3d_mesh.triangles)
            faces = np.hstack([np.full((tris.shape[0], 1), 3), tris]).flatten()
            
            self.pv_mesh = pv.PolyData(verts, faces)
            
            # 3. Initialize Mask
            n_points = self.pv_mesh.n_points
            self.excluded_mask = np.zeros(n_points, dtype=bool)
            
            if self.initial_indices:
                valid_idxs = [i for i in self.initial_indices if 0 <= i < n_points]
                self.excluded_mask[valid_idxs] = True
            
            # 4. Store Original IDs for mapping back from sub-selections
            self.pv_mesh.point_data["orig_ids"] = np.arange(n_points)
            
            # 5. Setup Rendering Layers
            self.plotter.clear()
            
            # Layer A: Wireframe/Surface (Context)
            self.plotter.add_mesh(
                self.pv_mesh,
                color="white",
                opacity=0.1,
                style='surface',
                show_edges=True,
                edge_color="#333333",
                pickable=False  # Do not pick the surface
            )

            # Layer B: All Vertices (Base Appearance)
            # These are the "Non Selected" ones (Tan, Small)
            self.plotter.add_mesh(
                self.pv_mesh,
                color="#388E3C",  # Tan
                point_size=5,     # Smaller
                render_points_as_spheres=True,
                style='points',
                pickable=True     # This is what we pick
            )

            # Layer C: Selected Vertices (Overlay)
            # These will be Red and Larger. Initialized in _refresh_visuals
            self._refresh_visuals()
            
            self.plotter.reset_camera()
            self._update_stats()

        except Exception as e:
            logging.error(f"Initialization Failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Initialization Error", str(e))

    def _toggle_picking_tool(self, active: bool):
        """
        Manages the transition between Navigation and Selection.
        Triggered by Checkbox or 'R' key.
        """
        if not self.pv_mesh:
            return

        self.plotter.disable_picking()

        if active:
            mode_str = "REMOVE" if self.chk_filter_mode.isChecked() else "KEEP"
            self.status_bar.showMessage(f"Selection Mode (ON): Drag LEFT MOUSE to select points to {mode_str}.")
            
            # Enable picking. We pick points via frustum (box).
            self.plotter.enable_cell_picking(
                mesh=self.pv_mesh,
                callback=self._on_picking_callback,
                show=False,       # Disable built-in red wireframe highlight
                through=True,     # Frustum selection
                font_size=10
            )
        else:
            self.status_bar.showMessage("Navigation Mode: Rotate and Zoom. (Press 'R' to Select)")
            self.plotter.enable_trackball_style()

    def _on_picking_callback(self, picked_item):
        """
        Callback triggered when a selection box is drawn.
        Updates the mask based on points inside the box and current toggle state.
        """
        if picked_item is None:
            return

        try:
            # Extract 'orig_ids' from the picked sub-mesh point data
            selected_ids = picked_item.point_data.get("orig_ids")
            
            if selected_ids is None or len(selected_ids) == 0:
                return

            # Apply Mask Logic based on Toggle State
            # State 1 (Unchecked): Selected = Keep (mask = False)
            # State 2 (Checked): Selected = Remove (mask = True)
            should_remove = self.chk_filter_mode.isChecked()
            
            if should_remove:
                self.excluded_mask[selected_ids] = True
            else:
                self.excluded_mask[selected_ids] = False

            # Visual Update
            self._refresh_visuals()
            self._update_stats()

        except Exception as e:
            logging.error(f"Picking Error: {e}")

    def _refresh_visuals(self):
        """
        Updates the 'Selected' overlay mesh.
        Draws large red spheres only at masked indices.
        """
        if self.pv_mesh is None:
            return
            
        # Remove previous overlay if it exists
        if self.selected_actor:
            self.plotter.remove_actor(self.selected_actor)
            self.selected_actor = None
        
        # Identify currently masked (removed) points
        mask_indices = np.where(self.excluded_mask)[0]
        
        if len(mask_indices) > 0:
            # Extract these points to a new PolyData object
            points = self.pv_mesh.points[mask_indices]
            cloud = pv.PolyData(points)
            
            # Add as a new actor: Red, Large, Spheres
            self.selected_actor = self.plotter.add_mesh(
                cloud,
                color="red",
                point_size=10, # Larger than the base size (5)
                render_points_as_spheres=True,
                style='points',
                pickable=False, # Overlay should not block picking underneath
                reset_camera=False
            )
            
        self.plotter.render()

    def _update_stats(self):
        count = np.count_nonzero(self.excluded_mask)
        total = self.excluded_mask.size
        self.lbl_stats.setText(f"Masked: {count} / {total} vertices")

    def _reset_mask(self):
        """Resets the mask so NO vertices are excluded."""
        if self.excluded_mask is not None:
            self.excluded_mask[:] = False
            self._refresh_visuals()
            self._update_stats()
            logging.info("Mask reset.")

    def _mask_all(self):
        """Sets the mask so ALL vertices are excluded."""
        if self.excluded_mask is not None:
            self.excluded_mask[:] = True
            self._refresh_visuals()
            self._update_stats()
            logging.info("All vertices masked.")

    def _finalize_selection(self):
        if self.excluded_mask is None:
            self.close()
            return

        self.result_indices = np.where(self.excluded_mask)[0].tolist()
        logging.info(f"Finalized. {len(self.result_indices)} vertices excluded.")
        self.selection_validated.emit(self.result_indices)
        self.close()
