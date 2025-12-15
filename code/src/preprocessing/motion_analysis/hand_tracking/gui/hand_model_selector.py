# hand_model_selector.py (Modified)

# --- Standard Library Imports ---
import logging
from pathlib import Path
from typing import List, Dict, Optional

# --- Third-party Imports ---
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QRadioButton, QGroupBox,
    QLabel, QSlider, QComboBox, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from pyvistaqt import QtInteractor

# --- Local Application Imports ---
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager, ColorFormat
from ..models.hand_metadata import HandMetadataManager
from ..core.hand_mesh_processor import HandMeshProcessor
from .point_selector_window import PointSelectorWindow

# Set up a basic logger for feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandModelSelectorGUI(QMainWindow):
    """
    An interactive GUI for selecting a 3D hand model, orientation, video frame,
    and specific points on the mesh.

    This version is refactored to use VideoMP4Manager for cleaner and safer
    video frame access.
    """
    def __init__(
        self,
        rgb_video_path: Path,
        hand_models_dir: Path,
        point_labels: List[str],
        default_points: Optional[List[Dict[str, any]]] = None
    ):
        """
        Initializes the GUI.

        Args:
            rgb_video_path (Path): Path to the RGB video file.
            hand_models_dir (Path): Directory containing hand model .txt files.
            point_labels (List[str]): A list of labels for the points to be selected.
            default_points (Optional[List[Dict[str, any]]]): An optional list of
                dictionaries to pre-populate selected points. Each dictionary should
                have 'label' and 'vertex_id' keys. If None, a hardcoded default is used.
        """
        super().__init__()
        self.setWindowTitle("Hand Model Selector")
        self.setGeometry(100, 100, 1200, 600)

        # --- State Attributes ---
        self.point_labels = point_labels
        
        # Set up default points
        if default_points is None:
            default_points = [
                {'label': 'sticker_yellow', 'vertex_id': 19},
                {'label': 'sticker_blue', 'vertex_id': 311},
                {'label': 'sticker_green', 'vertex_id': 21}
            ]
            
        # Initialize selected_points with None for all labels
        self.selected_points = {label: None for label in self.point_labels}
        # Populate with default values, ensuring the label is valid
        for point in default_points:
            if 'label' in point and point['label'] in self.selected_points:
                self.selected_points[point['label']] = point.get('vertex_id')

        self.is_left = False
        self.hand_models_dir = hand_models_dir
        
        # This attribute will hold the final output after validation
        self.result_metadata: dict | None = None

        # --- Internal Components ---
        self.pv_mesh = None
        self.selector_window = None
        self.mesh_cache = {}
        
        # --- Video setup using VideoMP4Manager ---
        self.video_manager = VideoMP4Manager(rgb_video_path, color_format=ColorFormat.RGB)
        
        self.total_frames = self.video_manager.total_frames
        self.current_frame_number = 0

        self.hand_model_paths = sorted([p for p in self.hand_models_dir.glob("*.txt")])
        if not self.hand_model_paths:
            raise IOError(f"No .txt files found in: {self.hand_models_dir}")
        self.selected_model_path = self.hand_model_paths[0]

        # --- UI Initialization ---
        self._setup_ui()

        self.update_display()
        self.raise_()
        self.activateWindow()

    def _setup_ui(self):
        """Helper method to initialize all UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel (Video)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.video_label = QLabel("Loading video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, self.total_frames - 1)
        self.frame_slider.valueChanged.connect(self.update_display)
        self.left_layout.addWidget(self.video_label)
        self.left_layout.addWidget(self.frame_slider)

        # Right Panel (3D Model Controls)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.orientation_groupbox = QGroupBox("Hand Orientation")
        self.orientation_layout = QHBoxLayout()
        self.left_radio = QRadioButton("Left Hand")
        self.right_radio = QRadioButton("Right Hand")
        self.right_radio.setChecked(not self.is_left)
        self.left_radio.setChecked(self.is_left)
        self.left_radio.toggled.connect(self._on_hand_orientation_changed)
        self.orientation_layout.addWidget(self.left_radio)
        self.orientation_layout.addWidget(self.right_radio)
        self.orientation_groupbox.setLayout(self.orientation_layout)
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([p.name for p in self.hand_model_paths])
        self.model_dropdown.currentIndexChanged.connect(self.update_display)
        self.plotter = QtInteractor(self.right_panel)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.select_points_button = QPushButton("Select 3D Points")
        self.select_points_button.clicked.connect(self.open_point_selector_window)
        self.validate_button = QPushButton("Validate and Finalize")
        self.validate_button.clicked.connect(self.validate_and_close)
        self.right_layout.addWidget(self.orientation_groupbox)
        self.right_layout.addWidget(self.model_dropdown)
        self.right_layout.addWidget(self.plotter)
        self.right_layout.addWidget(self.select_points_button)
        self.right_layout.addWidget(self.validate_button)
        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.right_panel, 1)

    # --- Methods for interaction and updates ---
    def _on_selections_validated(self, validated_points: dict):
        self.selected_points.update(validated_points)

    def open_point_selector_window(self):
        """
        Opens a non-modal window for selecting points. If a window is already
        open, it brings it to the front.
        """
        if self.pv_mesh is None:
            print("A 3D model must be loaded first.")
            return

        if self.selector_window and self.selector_window.isVisible():
            self.selector_window.raise_()
            self.selector_window.activateWindow()
        else:
            self.selector_window = PointSelectorWindow(
                mesh=self.pv_mesh,
                point_labels=self.point_labels,
                existing_points=self.selected_points
            )
            self.selector_window.pointSelected.connect(self._update_selected_point)
            self.selector_window.destroyed.connect(lambda: setattr(self, 'selector_window', None))
            self.selector_window.show()

    def _update_selected_point(self, label: str, vertex_id: int):
        """This method is a slot that connects to the pointSelected signal."""
        print(f"SLOT: Received point '{label}' -> vertex {vertex_id}")
        self.selected_points[label] = vertex_id

    def _on_hand_orientation_changed(self):
        self.is_left = self.left_radio.isChecked()
        self.update_display()

    def update_display(self):
        self.current_frame_number = self.frame_slider.value()
        selected_index = self.model_dropdown.currentIndex()
        self.selected_model_path = self.hand_model_paths[selected_index]
        self._update_video_player()
        self._update_3d_model()

    def _update_video_player(self):
        """
        Fetches and displays the current video frame using the VideoMP4Manager.
        """
        try:
            frame = self.video_manager[self.current_frame_number]
            
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        except IndexError:
            logging.error(f"Frame index {self.current_frame_number} is out of bounds.")
            self.video_label.setText(f"Error: Frame {self.current_frame_number} not found.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while updating the video player: {e}")
            self.video_label.setText("Error displaying video frame.")

    def _update_3d_model(self):
        cache_key = (str(self.selected_model_path), self.is_left)
        mesh_params = {'left': self.is_left}

        if cache_key in self.mesh_cache:
            o3d_mesh = self.mesh_cache[cache_key]
        else:
            print(f"Creating mesh for {self.selected_model_path.name} (Left: {self.is_left})")
            o3d_mesh = HandMeshProcessor.create_mesh(
                self.selected_model_path, mesh_params
            )
            self.mesh_cache[cache_key] = o3d_mesh

        vertices = np.asarray(o3d_mesh.vertices)
        faces_o3d = np.asarray(o3d_mesh.triangles)
        padding = np.full((faces_o3d.shape[0], 1), 3, dtype=np.int_)
        faces_pv = np.hstack((padding, faces_o3d))
        self.pv_mesh = pv.PolyData(vertices, faces_pv)

        self.plotter.clear()
        self.plotter.add_mesh(self.pv_mesh, color="tan", show_edges=True)
        self.plotter.reset_camera()
        
    def validate_and_close(self):
        """
        Validates the selections, generates the metadata dictionary, stores it,
        and closes the application.
        """
        manager = HandMetadataManager(
            source_video_path=self.video_manager.video_path,
            selected_hand_model_path=self.selected_model_path,
            selected_frame_number=self.current_frame_number,
            is_left_hand=self.is_left,
            selected_points=self.selected_points
        )

        try:
            self.result_metadata = manager.generate_output()
            logging.info("Metadata successfully generated. Closing window.")
        except ValueError as e:
            logging.error(f"Could not generate metadata: {e}")
            return

        self.close()

    def closeEvent(self, event):
        """
        Ensures resources are released when the window is closed.
        """
        if self.selector_window:
            self.selector_window.close()
        logging.info("Closing Hand Model Selector GUI.")
        super().closeEvent(event)