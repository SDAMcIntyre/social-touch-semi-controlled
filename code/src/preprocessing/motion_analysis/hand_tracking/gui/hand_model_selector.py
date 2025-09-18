# --- Imports from your original code ---
import json
from pathlib import Path
import cv2
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QRadioButton, QGroupBox,
    QLabel, QSlider, QComboBox, QPushButton, QSizePolicy, QDialog, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from pyvistaqt import QtInteractor
import logging

# Set up a basic logger for feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from ..models.hand_metadata import HandMetadataManager

class HandModelSelectorGUI(QMainWindow):
    """
    An interactive GUI for selecting a 3D hand model, orientation, video frame,
    and specific points on the mesh.

    This class is responsible for the user interface and interaction logic.
    It generates metadata but delegates the handling and saving of that data.
    """
    def __init__(self, rgb_video_path: Path, hand_models_dir: Path, point_labels: list[str]):
        super().__init__()
        self.setWindowTitle("Hand Model Selector")
        self.setGeometry(100, 100, 1200, 600)

        # --- State Attributes ---
        self.point_labels = point_labels
        self.selected_points = {label: None for label in self.point_labels}
        self.is_left = False
        self.rgb_video_path = rgb_video_path
        self.hand_models_dir = hand_models_dir
        
        # This attribute will hold the final output after validation
        self.result_metadata: dict | None = None

        # --- Internal Components ---
        self.pv_mesh = None
        self.selector_window = None
        self.mesh_cache = {}
        
        # --- Video setup ---
        self.video_capture = cv2.VideoCapture(str(self.rgb_video_path))
        if not self.video_capture.isOpened():
            raise IOError(f"Could not open video file: {self.rgb_video_path}")
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = 0

        self.hand_model_paths = sorted([p for p in self.hand_models_dir.glob("*.txt")])
        if not self.hand_model_paths:
            raise IOError(f"No .txt files found in: {self.hand_models_dir}")
        self.selected_model_path = self.hand_model_paths[0]

        # --- UI Initialization (identical to your original code) ---
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
        self.validate_button.clicked.connect(self.validate_and_close) # Changed method name slightly for clarity
        self.right_layout.addWidget(self.orientation_groupbox)
        self.right_layout.addWidget(self.model_dropdown)
        self.right_layout.addWidget(self.plotter)
        self.right_layout.addWidget(self.select_points_button)
        self.right_layout.addWidget(self.validate_button)
        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.right_panel, 1)

    # --- Methods for interaction and updates (identical to your original code) ---
    def _on_selections_validated(self, validated_points: dict):
        self.selected_points.update(validated_points)

    def open_point_selector_window(self):
        # This method's logic remains the same
        ...

    def _on_selector_window_destroyed(self):
        # This method's logic remains the same
        ...

    def _on_hand_orientation_changed(self):
        self.is_left = self.left_radio.isChecked()
        self.update_display()

    def update_display(self):
        # This method's logic remains the same, but it's now just updating the GUI state
        self.current_frame_number = self.frame_slider.value()
        selected_index = self.model_dropdown.currentIndex()
        self.selected_model_path = self.hand_model_paths[selected_index]
        self._update_video_player()
        self._update_3d_model()

    def _update_video_player(self):
        # This method's logic remains the same
        ...

    def _update_3d_model(self):
        # This method's logic remains the same
        ...
        
    def validate_and_close(self):
        """
        Validates the selections, generates the metadata dictionary, stores it,
        and closes the application.
        """
        # 1. Instantiate the manager with the final state from the GUI
        manager = HandMetadataManager(
            source_video_path=self.rgb_video_path,
            selected_hand_model_path=self.selected_model_path,
            selected_frame_number=self.current_frame_number,
            is_left_hand=self.is_left,
            selected_points=self.selected_points
        )

        # 2. Generate the final dictionary and store it in an instance attribute
        try:
            self.result_metadata = manager.generate_output()
            logging.info("Metadata successfully generated. Closing window.")
        except ValueError as e:
            logging.error(f"Could not generate metadata: {e}")
            # Here you could show an error dialog to the user
            return # Don't close if validation fails

        # 3. Close the window. The application event loop will then terminate.
        self.close()

    def closeEvent(self, event):
        """Ensures resources are released when the window is closed."""
        self.video_capture.release()
        if self.selector_window:
            self.selector_window.close()
        super().closeEvent(event)