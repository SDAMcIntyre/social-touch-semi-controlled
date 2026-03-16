# Standard library imports
import bisect
import sys
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# Third-party imports
import numpy as np
import open3d as o3d
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSlider, QVBoxLayout, QWidget,
                             QFileDialog, QProgressDialog, QMessageBox)
from pyvistaqt import QtInteractor

# SceneViewer and all other classes
# Assuming these are available in the local package context as provided in the original file
from .scene_viewer import SceneViewer
from .scene_viewer import *

class SceneViewerVideoMaker(SceneViewer):
    """
    A subclass of SceneViewer that includes functionality to export the current
    sequence as a high-resolution 1920x1080 @ 30Hz video.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Scene Navigator + Video Export")

    def _setup_frame_controls(self, parent_layout: QVBoxLayout):
        """Overrides the control setup to add the Export Video button."""
        frame_controls_layout = QHBoxLayout()
        
        self.recenter_button = QPushButton("Recenter View")
        self.recenter_button.clicked.connect(self._recenter_view)
        
        self.export_button = QPushButton("Export Video (1080p)")
        self.export_button.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        self.export_button.clicked.connect(self._on_export_video_clicked)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(self.current_index)
        self.slider.valueChanged.connect(self._on_slider_change)
        
        self.label = QLabel()
        self.label.setFixedWidth(150)
        
        frame_controls_layout.addWidget(QLabel("Frame:"))
        frame_controls_layout.addWidget(self.slider)
        frame_controls_layout.addWidget(self.label)
        frame_controls_layout.addWidget(self.recenter_button)
        frame_controls_layout.addWidget(self.export_button)
        
        parent_layout.addLayout(frame_controls_layout)
        
        self._update_label()

    def _on_export_video_clicked(self):
        """Handles the video export workflow."""
        # 1. Check for imageio-ffmpeg dependency before starting
        try:
            import imageio_ffmpeg
        except ImportError:
            QMessageBox.critical(
                self, 
                "Missing Dependency", 
                "The 'imageio-ffmpeg' library is required to export videos.\n\n"
                "Please install it via:\n"
                "pip install imageio-ffmpeg\n"
                "OR\n"
                "conda install -c conda-forge imageio-ffmpeg"
            )
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4);;AVI Files (*.avi)"
        )
        
        if not filename:
            return

        # 2. Enforce File Extension
        # Users often type "myvideo" without the .mp4 extension.
        # This logic ensures the extension exists, preventing imageio from defaulting to TIFF.
        valid_extensions = ['.mp4', '.avi']
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            if "AVI" in selected_filter:
                filename += ".avi"
            else:
                filename += ".mp4"

        self._generate_video(filename)

    def _generate_video(self, output_path: str):
        """
        Generates a 1920x1080 video at 30Hz using an off-screen plotter.
        """
        total_frames = self.num_frames
        if total_frames <= 0:
            return

        # Create progress dialog
        progress = QProgressDialog("Rendering Video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        # 1. Setup Off-screen Plotter (1920x1080)
        # Note: 'off_screen=True' creates a hidden window.
        render_plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)
        render_plotter.set_background('midnightblue')

        # 2. Sync Camera from Main View
        # Deep copy camera parameters to ensure exact match of the current user view
        current_cam = self.plotter.camera
        render_plotter.camera.position = current_cam.position
        render_plotter.camera.focal_point = current_cam.focal_point
        render_plotter.camera.up = current_cam.up
        render_plotter.camera.view_angle = current_cam.view_angle
        render_plotter.camera.clipping_range = current_cam.clipping_range

        try:
            # Attempt to open the movie file
            render_plotter.open_movie(output_path, framerate=30)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to initialize video writer: {e}\n\nEnsure valid write permissions and file path.")
            render_plotter.close()
            return

        # 3. Render Loop
        axis_length = 250
        
        try:
            for i in range(total_frames):
                if progress.wasCanceled():
                    break

                render_plotter.clear()
                
                # Re-add static elements
                render_plotter.add_axes(interactive=False, line_width=axis_length, box=True)
                render_plotter.add_mesh(pv.Sphere(radius=2.0), color='yellow')

                # Add dynamic objects for this specific frame
                for obj in self.scene_objects.values():
                    # add_to_plotter respects the obj.visible flag internally
                    if obj.visible:
                        obj.add_to_plotter(render_plotter, i)

                # Force render and write to file
                render_plotter.write_frame()
                
                progress.setValue(i + 1)
                QApplication.processEvents() # Keep UI responsive
        
        except Exception as e:
            QMessageBox.critical(self, "Rendering Error", f"An error occurred during rendering:\n{str(e)}")
        
        finally:
            # Cleanup
            render_plotter.close()
            progress.close()
        
        if not progress.wasCanceled() and os.path.exists(output_path):
            QMessageBox.information(self, "Success", f"Video saved to:\n{output_path}")

def main():
    app = QApplication(sys.argv)

    # --- 1. Create Data (Unchanged) ---
    point_clouds_data: Dict[int, PointCloudData] = {}
    for i in range(40):
        angle = 2 * np.pi * i / 50
        center_offset = np.array([np.cos(angle) * 50, np.sin(angle) * 50, 0])
        points = np.random.rand(100, 3) * 50 - 25 + center_offset
        point_clouds_data[i] = PointCloudData(points)
    angle = 2 * np.pi * 49 / 50
    center_offset = np.array([np.cos(angle) * 50, np.sin(angle) * 50, 0])
    points = np.random.rand(100, 3) * 50 - 25 + center_offset
    point_clouds_data[49] = PointCloudData(points)
    
    max_frame_index = 70
    blue_sticker_data: Dict[int, np.ndarray] = {}
    green_sticker_data: Dict[int, np.ndarray] = {}
    yellow_sticker_data: Dict[int, np.ndarray] = {}
    t = np.linspace(0, 2 * np.pi, max_frame_index)
    green_y = np.linspace(-50, 50, max_frame_index)
    green_z = np.linspace(-40, 40, max_frame_index)
    for i in range(max_frame_index):
        blue_sticker_data[i] = np.array([np.cos(t[i]) * 80, np.sin(t[i]) * 80, 20])
        green_sticker_data[i] = np.array([-30, green_y[i], green_z[i]])
        yellow_sticker_data[i] = np.array([np.sin(t[i] * 2) * 60, 60, np.cos(t[i] * 2) * 30])
        
    forearms_dict: Dict[int, PointCloudData] = {}
    points_0 = np.random.rand(5000, 3) * 20 + np.array([-60, -60, 0])
    forearms_dict[0] = PointCloudData(points_0)
    points_25 = np.random.rand(5000, 3) * 20 + np.array([0, 0, 20])
    forearms_dict[25] = PointCloudData(points_25)
    points_50 = np.random.rand(5000, 3) * 20 + np.array([60, 60, 0])
    forearms_dict[50] = PointCloudData(points_50)

    mesh_dict: Dict[int, o3d.geometry.TriangleMesh] = {}
    mesh_10 = o3d.geometry.TriangleMesh.create_sphere(radius=15)
    mesh_10.compute_vertex_normals()
    mesh_10.paint_uniform_color([1.0, 0.2, 0.2]) 
    mesh_10.translate([-50, 0, 30])
    mesh_dict[10] = mesh_10
    mesh_40 = o3d.geometry.TriangleMesh.create_sphere(radius=15)
    mesh_40.compute_vertex_normals()
    mesh_40.paint_uniform_color([0.2, 0.2, 1.0]) 
    mesh_40.translate([50, 0, -10])
    mesh_dict[40] = mesh_40

    # --- 2. Instantiate the SceneViewerVideoMaker (Modified) ---
    viewer = SceneViewerVideoMaker()

    # --- 3. Create SceneObjects and add them to the viewer ---
    main_cloud = PointCloudSequence(name="main_cloud", frame_data=point_clouds_data, point_size=8)
    blue_sticker = Trajectory(name="sticker_blue", frame_data=blue_sticker_data, color='blue', radius=3)
    green_sticker = Trajectory(name="sticker_green", frame_data=green_sticker_data, color='lime', radius=5)
    yellow_sticker = Trajectory(name="sticker_yellow", frame_data=yellow_sticker_data, color='yellow', radius=4)

    forearms_cloud = PersistentPointCloudSequence(name="forearms", frame_data=forearms_dict, point_size=10)
    moving_mesh = PersistentOpen3DTriangleMeshSequence(name="moving_sphere", frame_data=mesh_dict)

    viewer.add_object(main_cloud)
    viewer.add_object(blue_sticker)
    viewer.add_object(green_sticker)
    viewer.add_object(yellow_sticker)
    viewer.add_object(forearms_cloud)
    viewer.add_object(moving_mesh)

    # --- 4. Show the application ---
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()