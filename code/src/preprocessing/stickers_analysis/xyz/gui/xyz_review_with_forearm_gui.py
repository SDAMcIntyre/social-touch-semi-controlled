import sys
import numpy as np
import pandas as pd
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# BASE VISUALIZER (UNCHANGED)
# ==============================================================================

class ObjectsInteractionVisualizer(QMainWindow):
    """
    Manages the visualization of 3D object interactions with a timeline slider
    using PyQt5 for the UI and PyVista for 3D rendering.
    
    This version has a generalized `_update_pyvista_scene` to handle both
    geometry updates (changing mesh vertices) and transform updates (changing
    an actor's position).
    """
    def __init__(self,
                 window_title: str = "3D Interaction Visualizer",
                 width: int = 1600,
                 height: int = 900):
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(width, height)
        self.all_frames_data = []
        self.actors = {}
        self.is_playing = False
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(1000 // 30)  # ~30 FPS
        self.play_timer.timeout.connect(self._advance_frame)
        self.fig = None
        self.axes = []
        self.canvas = None
        self.v_lines = []
        self.plot_configs = []
        self.plot_window_half_width = 30
        self._setup_ui()

    # --- UI and Core Logic Methods (largely unchanged) ---

    def _setup_ui(self):
        """Creates the main layout and all GUI widgets."""
        from pyvistaqt import QtInteractor
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.plotter = QtInteractor()
        left_layout.addWidget(self.plotter, 1)
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self._toggle_playback)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._update_frame)
        self.frame_slider.sliderPressed.connect(self._pause_playback)
        self.frame_entry = QLineEdit("0")
        self.frame_entry.setFixedWidth(80)
        self.frame_entry.returnPressed.connect(self._update_from_text_entry)
        self.frame_label = QLabel("/ 0")
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_entry)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(close_button)
        left_layout.addWidget(controls_widget)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(450)
        self.fig = Figure(dpi=100)
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel)
    
    def add_geometry(self, name: str, geometry: o3d.geometry.Geometry, **kwargs):
        """Adds a geometry to the scene, creating a PyVista actor for it."""
        if name in self.actors:
            self.plotter.remove_actor(self.actors[name])

        try:
            vertices = np.asarray(geometry.points)
        except AttributeError:
            vertices = np.asarray(geometry.vertices)
        
        is_empty = vertices.shape[0] == 0

        if isinstance(geometry, o3d.geometry.TriangleMesh):
            plot_verts = vertices if not is_empty else np.array([[0,0,-9999]]*3)
            plot_faces = np.hstack((np.full((len(geometry.triangles), 1), 3),
                                    np.asarray(geometry.triangles))).flatten() if not is_empty else np.array([3,0,1,2])
            pv_object = pv.PolyData(plot_verts, faces=plot_faces)
            actor = self.plotter.add_mesh(pv_object, smooth_shading=True, **kwargs)
            actor.SetVisibility(not is_empty)
        elif isinstance(geometry, o3d.geometry.PointCloud):
            plot_verts = vertices if not is_empty else np.array([[0,0,-9999]])
            pv_object = pv.PolyData(plot_verts)
            actor = self.plotter.add_mesh(pv_object, point_size=5.0, render_points_as_spheres=True, **kwargs)
            actor.SetVisibility(not is_empty)
        else:
            return

        self.actors[name] = actor
        if len(self.actors) == 1:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
    
    def recenter_view_on_point(self, point: list):
        if self.plotter:
            self.plotter.set_focus(point)
            self.plotter.reset_camera_clipping_range()

    def _toggle_playback(self):
        if self.is_playing:
            self._pause_playback()
        else:
            if self.frame_slider.value() == self.frame_slider.maximum():
                self.frame_slider.setValue(0)
            self.play_timer.start()
            self.is_playing = True
            self.play_pause_button.setText("Pause")

    def _pause_playback(self):
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_pause_button.setText("Play")

    def _advance_frame(self):
        current_frame = self.frame_slider.value()
        if current_frame < self.frame_slider.maximum():
            self.frame_slider.setValue(current_frame + 1)
        else:
            self._pause_playback()

    def add_plot(self, title: str, data_vector: list, **kwargs):
        if not self.all_frames_data or len(data_vector) != len(self.all_frames_data):
            print("Error: Plot data length must match frame data length.")
            return
        plot_config = {'title': title, 'data': data_vector}
        plot_config.update(kwargs)
        self.plot_configs.append(plot_config)

    def _setup_plots(self):
        if not self.plot_configs: return
        with plt.style.context('seaborn-v0_8-whitegrid'):
            plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})
            num_plots = len(self.plot_configs)
            self.fig.clear()
            self.axes = self.fig.subplots(nrows=num_plots, ncols=1, sharex=True)
            self.axes = np.atleast_1d(self.axes)
            self.fig.suptitle(self.plot_configs[0].get('title', 'Data'), fontsize=16, fontweight='bold')
            time_axis = range(len(self.all_frames_data))
            self.v_lines.clear()
            for ax, config in zip(self.axes, self.plot_configs):
                ax.plot(time_axis, config['data'], color=config.get('color', '#007ACC'), linewidth=1.5)
                ax.set_ylabel(config.get('y_label', ''), fontsize=12)
                v_line = ax.axvline(0, color='#D32F2F', linestyle='--', linewidth=1)
                self.v_lines.append(v_line)
            self.axes[-1].set_xlabel("Frame ID", fontsize=12)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
    def _update_frame(self, frame_index: int):
        self.frame_entry.setText(str(frame_index))
        self._update_plot_cursor(frame_index)
        if 0 <= frame_index < len(self.all_frames_data):
            self._update_pyvista_scene(self.all_frames_data[frame_index])

    def _update_plot_cursor(self, frame_index: int):
        if not self.axes.any(): return
        for v_line in self.v_lines:
            v_line.set_xdata([frame_index, frame_index])
        total_frames = len(self.all_frames_data)
        half_width = self.plot_window_half_width
        x_min, x_max = frame_index - half_width, frame_index + half_width
        if x_min < 0: x_max, x_min = x_max - x_min, 0
        if x_max >= total_frames: x_min, x_max = x_min - (x_max - (total_frames - 1)), total_frames - 1
        for ax in self.axes: ax.set_xlim(max(0, x_min), x_max)
        self.canvas.draw_idle()

    def _update_from_text_entry(self):
        self._pause_playback()
        try:
            val = int(self.frame_entry.text())
            max_val = len(self.all_frames_data) - 1
            self.frame_slider.setValue(max(0, min(val, max_val)))
        except (ValueError, IndexError):
            self.frame_entry.setText(str(self.frame_slider.value()))
        self.frame_entry.clearFocus()

    def set_frame_data(self, all_frames_data: list):
        if not all_frames_data: return
        self.all_frames_data = all_frames_data
        num_frames = len(all_frames_data) - 1
        self.frame_slider.setRange(0, num_frames)
        self.frame_label.setText(f"/ {num_frames}")

    def run(self):
        if not self.all_frames_data:
            print("Error: No frame data loaded. Call `set_frame_data()` before `run()`.")
            self.close()
            return
        self._setup_plots()
        self._update_frame(0)
        self.show()

    def _update_pyvista_scene(self, frame_data: dict):
        """
        Updates PyVista actors based on the current frame's data.
        """
        for name, actor in self.actors.items():
            if name not in frame_data:
                continue
            
            update_info = frame_data[name]

            if 'position' in update_info:
                actor.SetPosition(update_info['position'])
                actor.SetVisibility(True)

            elif 'geometry' in update_info:
                geometry = update_info['geometry']
                try:
                    new_points = np.asarray(geometry.points)
                except AttributeError:
                    new_points = np.asarray(geometry.vertices)

                is_visible = new_points.shape[0] > 0
                if is_visible:
                    points_to_set = new_points if new_points.size > 0 else np.array([[0,0,-9999]])
                    actor.mapper.dataset.points = points_to_set
                actor.SetVisibility(is_visible)


# ==============================================================================
# ✨ REVISED TRAJECTORY VISUALIZER CLASS ✨
# ==============================================================================

class XYZReviewWithForearmGUI:
    """
    A high-level controller to visualize object trajectories around a
    central reference point cloud.
    """
    def __init__(self,
                 ref_pcd: o3d.geometry.PointCloud,
                 trajectories: dict[str, pd.DataFrame],
                 trajectory_colors: dict[str, str] = None,
                 radius: float = 2.0,
                 window_title: str = "Object Trajectory Visualizer"):
        """
        Initializes the Trajectory Visualizer.

        Args:
            ref_pcd (o3d.geometry.PointCloud): A static point cloud to display.
            trajectories (dict[str, pd.DataFrame]): A dictionary where keys are
                object names and values are pandas DataFrames containing 'x_mm',
                'y_mm', and 'z_mm' columns.
            trajectory_colors (dict[str, str], optional): A dictionary mapping
                object names (from trajectories keys) to color strings
                (e.g., '#FF0000', 'red'). Defaults to None, which triggers
                automatic color assignment.
        """
        if not trajectories:
            raise ValueError("Trajectories dictionary cannot be empty.")

        self.ref_pcd = ref_pcd
        self.trajectories = trajectories
        self.trajectory_colors = trajectory_colors
        self.radius = radius
        self.view = ObjectsInteractionVisualizer(window_title=window_title)

        # Define default colors for fallback
        self.default_colors = ['#FF5733', '#33FF57', '#3357FF', '#F1C40F', '#9B59B6', '#E74C3C']

    def _prepare_frame_data(self) -> list[dict]:
        """
        Converts the trajectory DataFrames into the list-of-dicts format
        required by the ObjectsInteractionVisualizer.
        """
        print("Preparing frame data from trajectories...")
        first_key = next(iter(self.trajectories))
        num_frames = len(self.trajectories[first_key])
        
        all_frames_data = []
        for i in range(num_frames):
            frame_data = {}
            for name, df in self.trajectories.items():
                if i < len(df):
                    row = df.iloc[i]
                    position = [row['x_mm'], row['y_mm'], row['z_mm']]
                    frame_data[name] = {'position': position}
            all_frames_data.append(frame_data)
        
        print(f"Prepared {len(all_frames_data)} frames of data.")
        return all_frames_data

    def setup_scene(self):
        """
        Configures the visualizer by preparing data, adding geometries,
        and setting up plots.
        """
        # 1. Prepare data and load it into the view
        frame_data = self._prepare_frame_data()
        self.view.set_frame_data(frame_data)

        # 2. Add the static reference point cloud
        self.view.add_geometry('ref_pcd', self.ref_pcd, color='lightgrey')

        # 3. Add actors for each trajectory (as small spheres)
        for i, name in enumerate(self.trajectories.keys()):
            marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
            
            # Use provided color if available, otherwise fall back to default list
            if self.trajectory_colors and name in self.trajectory_colors:
                color_hex = self.trajectory_colors[name]
            else:
                color_hex = self.default_colors[i % len(self.default_colors)]
            
            self.view.add_geometry(name, marker_sphere, color=color_hex)

        # 4. (Optional) Add plots for the coordinates of the first object
        first_name, first_df = next(iter(self.trajectories.items()))
        self.view.add_plot(f"{first_name} Trajectory", first_df['x_mm'].tolist(), color='#D32F2F', y_label='X (mm)')
        self.view.add_plot(f"{first_name} Trajectory", first_df['y_mm'].tolist(), color='#1976D2', y_label='Y (mm)')
        self.view.add_plot(f"{first_name} Trajectory", first_df['z_mm'].tolist(), color='#388E3C', y_label='Z (mm)')

        # 5. Center the camera on the reference object
        center_point = self.ref_pcd.get_center()
        self.view.recenter_view_on_point(center_point)

    def run(self):
        """
        Shows the visualizer window and starts the application event loop.
        """
        print("Starting visualization...")
        self.setup_scene()
        self.view.run()

# ==============================================================================
# MOCKUP MAIN FUNCTION FOR TESTING
# ==============================================================================
def create_mock_trajectories(num_frames=300, num_objects=3):
    """Generates a reference PCD and a dictionary of trajectory DataFrames."""
    print("Generating mock trajectory data...")
    ref_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=50).sample_points_poisson_disk(5000)
    ref_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    stickers_df_dict = {}
    names = ['sticker_blue', 'sticker_green', 'sticker_yellow', 'sticker_red']
    
    for i in range(num_objects):
        t = np.linspace(0, 2 * np.pi, num_frames)
        x = 100 * np.cos(t + i * np.pi / 2)
        y = 75 * np.sin(t + i * np.pi / 2)
        z = 50 * np.sin(2 * t + i * np.pi)
        
        df = pd.DataFrame({'x_mm': x, 'y_mm': y, 'z_mm': z})
        df.index.name = 'frame'
        stickers_df_dict[names[i]] = df

    print("Mock data generation complete.")
    return ref_pcd, stickers_df_dict


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    
    # 1. Generate the data
    reference_point_cloud, trajectories_dict = create_mock_trajectories(num_frames=400, num_objects=3)

    # 2. ✨ Define specific colors for the trajectories by name
    # The names must match the keys in the trajectories_dict
    custom_colors = {
        'sticker_blue': '#007ACC',  # A nice blue
        'sticker_green': 'green',     # Standard green
        'sticker_yellow': '#FFC300'  # A vibrant yellow
    }
    
    # 3. Instantiate the visualizer, passing the custom colors
    visualizer = XYZReviewWithForearmGUI(
        ref_pcd=reference_point_cloud,
        trajectories=trajectories_dict,
        trajectory_colors=custom_colors
    )

    # 4. Run the visualizer
    visualizer.run()

    # 5. Execute the Qt application event loop
    sys.exit(app.exec_())