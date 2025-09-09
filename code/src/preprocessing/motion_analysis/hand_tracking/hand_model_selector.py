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

from .hand_mesh_processor import HandMeshProcessor
from .point_selector_window import PointSelectorWindow


class HandModelSelectorGUI(QMainWindow):
    """
    An interactive GUI for selecting a 3D hand model, orientation, video frame,
    and specific points on the mesh.
    """
    def __init__(self, rgb_video_path: Path, hand_models_dir: Path, point_labels: list[str], metadata_path: Path):
        super().__init__()
        self.setWindowTitle("Hand Model Selector")
        self.setGeometry(100, 100, 1200, 600)

        self.point_labels = point_labels
        self.selected_points = {label: None for label in self.point_labels}
        self.pv_mesh = None

        self.selector_window = None

        self.is_left = False
        self.rgb_video_path = rgb_video_path
        self.hand_models_dir = hand_models_dir
        self.metadata_path = metadata_path
        self.mesh_cache = {}

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

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

        # --- Left Panel (Video) ---
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

        # --- Right Panel (3D Model Controls) ---
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

        self.validate_button = QPushButton("Validate and Save")
        self.validate_button.clicked.connect(self.validate_and_close)

        self.right_layout.addWidget(self.orientation_groupbox)
        self.right_layout.addWidget(self.model_dropdown)
        self.right_layout.addWidget(self.plotter)
        self.right_layout.addWidget(self.select_points_button)
        self.right_layout.addWidget(self.validate_button)

        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.right_panel, 1)

        self.update_display()
        self.raise_()
        self.activateWindow()

    # MODIFIED: Replaced the real-time update slot with one that accepts the final dictionary.
    def _on_selections_validated(self, validated_points: dict):
        """
        This slot connects to the `selectionsValidated` signal from the selector window
        and updates the main GUI's state in one atomic operation.
        """
        print(f"SLOT: Received validated points: {validated_points}")
        self.selected_points.update(validated_points)

    def open_point_selector_window(self):
        """
        Opens a non-modal window for selecting points. If a window is already
        open, it brings it to the front.
        """
        if self.pv_mesh is None:
            print("A 3D model must be loaded first.")
            return

        # MODIFIED: This check is now robust. It no longer calls a method
        # on a potentially deleted object.
        if self.selector_window is not None:
            self.selector_window.raise_()
            self.selector_window.activateWindow()
        else:
            self.selector_window = PointSelectorWindow(
                mesh=self.pv_mesh,
                point_labels=self.point_labels,
                existing_points=self.selected_points
            )
            self.selector_window.selectionsValidated.connect(self._on_selections_validated)
            
            # This ensures self.selector_window is set to None when the user
            # closes the window.
            self.selector_window.destroyed.connect(self._on_selector_window_destroyed)
            
            self.selector_window.show()
        
    def _on_selector_window_destroyed(self):
        """
        Slot connected to the selector window's `destroyed` signal.
        This sets the reference to None, preventing calls on a deleted C++ object.
        """
        self.selector_window = None

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
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video_capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

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
        Validates the selections, saves the data to a metadata file with a
        clear structure for selected points, and closes the application.
        """
        selected_points_list = [
            {"label": label, "vertex_id": vertex_id}
            for label, vertex_id in self.selected_points.items()
            if vertex_id is not None
        ]

        metadata = {
            "source_video_name": str(self.rgb_video_path.name),
            "selected_hand_model_name": str(self.selected_model_path.name),
            "selected_frame_number": self.current_frame_number,
            "hand_orientation": "left" if self.is_left else "right",
            "selected_points": selected_points_list
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {self.metadata_path}")
        self.close()

    def closeEvent(self, event):
        self.video_capture.release()
        if self.selector_window:
            self.selector_window.close()
        super().closeEvent(event)