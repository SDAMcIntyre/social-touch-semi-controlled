import os
import sys

# --- ARCHITECTURAL FIX: Enforce Qt Binding Consistency ---
# We must force both PyVista (via QT_API) and Matplotlib to use PyQt5.
# This prevents the "unexpected type" error where libraries try to mix 
# PyQt5, PyQt6, or PySide2 widgets.
os.environ["QT_API"] = "pyqt5"

import matplotlib
# Force Matplotlib to use the Qt5 backend before pyplot is imported.
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import open3d as o3d

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt, QTimer

# Explicitly import the correct backend canvas for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ObjectsInteractionVisualizer(QMainWindow):
    """
    Manages the visualization of 3D object interactions with a timeline slider
    using PyQt5 for the UI and PyVista for 3D rendering.
    """
    def __init__(self,
                 window_title: str = "3D Interaction Visualizer",
                 width: int = 1600,
                 height: int = 900):
        """
        Initializes the visualizer window and its components.
        """
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(width, height)

        # --- Data & State ---
        self.all_frames_data = []
        self.actors = {}
        self.is_playing = False

        # --- Playback Timer ---
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(1000 // 30)  # ~30 FPS
        self.play_timer.timeout.connect(self._advance_frame)

        # --- Plotting Attributes ---
        self.fig = None
        self.axes = []
        self.canvas = None
        self.v_lines = []
        self.plot_configs = []
        self.plot_window_half_width = 30

        # --- GUI Element and Layout Creation ---
        self._setup_ui()

    def _setup_ui(self):
        """Creates the main layout and all GUI widgets."""
        # Import QtInteractor locally to ensure it picks up the forced QT_API
        from pyvistaqt import QtInteractor
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (3D View and Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Create the plotter and add it to the layout.
        # The '1' stretch factor makes the 3D view expand to fill available space.
        self.plotter = QtInteractor()
        left_layout.addWidget(self.plotter, 1)

        # --- Control Panel ---
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self._toggle_playback)

        self.frame_slider = QSlider(Qt.Horizontal)
        # Connect valueChanged for live updates while dragging.
        self.frame_slider.valueChanged.connect(self._update_frame)
        # Pause playback if the user interacts with the slider
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

        # --- Right Panel (Data Plot) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(450)

        self.fig = Figure(dpi=100)
        # Initialize the canvas with the Figure
        self.canvas = FigureCanvas(self.fig)
        
        # Add the canvas to the layout. 
        # With QT_API='pyqt5' and matplotlib.use('Qt5Agg'), this is now type-safe.
        right_layout.addWidget(self.canvas)

        # Add panels to the main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel)

    def add_geometry(self, name: str, geometry: o3d.geometry.Geometry, point_size: float = 4.0):
        if name in self.actors:
            print(f"Warning: Geometry with name '{name}' already exists. It will be replaced.")
            self.plotter.remove_actor(self.actors[name])

        try:
            vertices = np.asarray(geometry.points)
        except AttributeError:
            vertices = np.asarray(geometry.vertices)
        
        is_empty = vertices.shape[0] == 0

        # ROBUST COLOR EXTRACTION
        # Defaults to lightgrey. Only attempts to access index [0] if the array has data.
        color = 'lightgrey'
        if hasattr(geometry, 'vertex_colors') and len(geometry.vertex_colors) > 0:
            color = np.asarray(geometry.vertex_colors)[0]
        elif hasattr(geometry, 'colors') and len(geometry.colors) > 0:
            color = np.asarray(geometry.colors)[0]

        if isinstance(geometry, o3d.geometry.TriangleMesh):
            if is_empty:
                # Create a placeholder mesh that is invisible.
                plot_verts = np.array([[0, 0, -9999]] * 3)
                plot_faces = np.array([3, 0, 1, 2])
            else:
                plot_verts = vertices
                triangles = np.asarray(geometry.triangles)
                plot_faces = np.hstack((np.full((len(triangles), 1), 3), triangles)).flatten()

            pv_object = pv.PolyData(plot_verts, faces=plot_faces)
            actor = self.plotter.add_mesh(pv_object, color=color, style='wireframe', point_size=point_size)
            actor.SetVisibility(not is_empty)

        elif isinstance(geometry, o3d.geometry.PointCloud):
            # Create a placeholder if empty
            plot_verts = vertices if not is_empty else np.array([[0, 0, -9999]])
            pv_object = pv.PolyData(plot_verts)
            
            actor = self.plotter.add_mesh(
                pv_object, 
                color=color, 
                point_size=point_size, 
                render_points_as_spheres=True
            )
            actor.SetVisibility(not is_empty)
        else:
            print(f"Warning: Geometry '{name}' has an unsupported type: {type(geometry)}")
            return
        
        self.actors[name] = actor
        
        if len(self.actors) == 1:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
    
    def is_existing_geometry(self, name: str):
        return name in self.actors

    def update_geometry(self, name:str, geometry: o3d.geometry.Geometry):
        if self.is_existing_geometry(name):
            self.plotter.remove_actor(self.actors[name])
        self.add_geometry(name, geometry)

    def add_plot(self, title: str, data_vector: list, color: str = 'r', y_label: str = 'Value'):
        """
        Registers a data vector to be plotted.
        """
        if not self.all_frames_data:
            print("Warning: Set frame data with `set_frame_data()` before adding a plot.")
            return

        if len(data_vector) != len(self.all_frames_data):
            print(f"Error: Plot data length ({len(data_vector)}) differs from frame count "
                  f"({len(self.all_frames_data)}). The plot may be incorrect.")
            return

        self.plot_configs.append({
            'title': title,
            'data': data_vector,
            'color': color,
            'y_label': y_label
        })

    def recenter_view_on_point(self, point: list):
        """Recenters the 3D view's camera to focus on a specific point."""
        if self.plotter:
            self.plotter.set_focus(point)
            self.plotter.reset_camera_clipping_range()

    def _toggle_playback(self):
        """Starts or stops the animation playback."""
        if self.is_playing:
            self._pause_playback()
        else:
            if self.frame_slider.value() == self.frame_slider.maximum():
                self.frame_slider.setValue(0)
            
            self.play_timer.start()
            self.is_playing = True
            self.play_pause_button.setText("Pause")

    def _pause_playback(self):
        """Stops the animation playback."""
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_pause_button.setText("Play")

    def _advance_frame(self):
        """Advances the animation by one frame if playing."""
        current_frame = self.frame_slider.value()
        if current_frame < self.frame_slider.maximum():
            self.frame_slider.setValue(current_frame + 1)
        else:
            self._pause_playback()

    def _setup_plots(self):
        """Configures and renders subplots."""
        if not self.plot_configs:
            return

        with plt.style.context('seaborn-v0_8-whitegrid'):
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.edgecolor': 'darkgray',
                'axes.labelcolor': 'black',
                'axes.titlecolor': 'black',
                'xtick.color': 'dimgray',
                'ytick.color': 'dimgray',
                'grid.color': 'lightgray',
                'grid.linestyle': '--',
                'font.family': 'serif'
            })

            num_plots = len(self.plot_configs)
            self.fig.clear()
            
            self.axes = self.fig.subplots(
                nrows=num_plots, 
                ncols=1, 
                sharex=True
            )
            self.axes = np.atleast_1d(self.axes)

            self.fig.suptitle(
                self.plot_configs[0].get('title', 'Sensor Data'), 
                fontsize=16, 
                fontweight='bold'
            )
            
            time_axis = range(len(self.all_frames_data))
            self.v_lines.clear()

            for ax, config in zip(self.axes, self.plot_configs):
                ax.plot(
                    time_axis, 
                    config['data'], 
                    color=config.get('color', '#007ACC'),
                    linewidth=1.5
                )

                ax.set_ylabel(config.get('y_label', ''), fontsize=12)
                v_line = ax.axvline(0, color='#D32F2F', linestyle='--', linewidth=1)
                self.v_lines.append(v_line)

            self.axes[-1].set_xlabel("Frame ID", fontsize=12)
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    def _update_frame(self, frame_index: int):
        """Updates all visual components."""
        self.frame_entry.setText(str(frame_index))
        self._update_plot_cursor(frame_index)
        if 0 <= frame_index < len(self.all_frames_data):
            self._update_pyvista_scene(self.all_frames_data[frame_index])

    def _update_plot_cursor(self, frame_index: int):
        """Updates the vertical cursor and slides the plot's x-axis window."""
        if not hasattr(self.axes, 'any') or not self.axes.any():
            return

        for v_line in self.v_lines:
            v_line.set_xdata([frame_index, frame_index])

        total_frames = len(self.all_frames_data)
        half_width = self.plot_window_half_width
        
        x_min = frame_index - half_width
        x_max = frame_index + half_width

        if x_min < 0:
            x_max -= x_min
            x_min = 0
        
        if x_max >= total_frames:
            x_min -= (x_max - (total_frames - 1))
            x_max = total_frames - 1

        for ax in self.axes:
            ax.set_xlim(max(0, x_min), x_max)

        if self.canvas:
            self.canvas.draw_idle()

    def _update_from_text_entry(self):
        """Validates the text entry and updates the scene and slider."""
        self._pause_playback()
        try:
            val = int(self.frame_entry.text())
            max_val = len(self.all_frames_data) - 1
            clamped_val = max(0, min(val, max_val))
            self.frame_slider.setValue(clamped_val)
        except (ValueError, IndexError):
            self.frame_entry.setText(str(self.frame_slider.value()))
        self.frame_entry.clearFocus()

    def _update_pyvista_scene(self, frame_data: dict):
            """Updates PyVista actors based on the current frame's data."""
            # Update hand mesh
            hand_actor = self.actors.get('hand')
            hand_mesh_data = frame_data.get('transformed_hand_mesh')
            if hand_actor and hand_mesh_data:
                try:
                    new_points = np.asarray(hand_mesh_data.points)
                except AttributeError:
                    new_points = np.asarray(hand_mesh_data.vertices)

                is_visible = new_points.shape[0] > 0
                if is_visible:
                    hand_actor.mapper.dataset.points = new_points
                hand_actor.SetVisibility(is_visible)

            # Update contact points
            contacts_actor = self.actors.get('contacts')
            raw_contacts = frame_data.get('contact_points')

            if isinstance(raw_contacts, (int, float)) or raw_contacts is None:
                points_data = np.array([])
            elif isinstance(raw_contacts, np.ndarray):
                points_data = raw_contacts
            else:
                points_data = np.array([])

            if contacts_actor:
                has_contacts = points_data.size > 0
                if has_contacts:
                    render_points = points_data
                else:
                    render_points = np.array([[0, 0, -9999]])

                new_mesh = pv.PolyData(render_points)
                contacts_actor.mapper.dataset.shallow_copy(new_mesh)
                contacts_actor.SetVisibility(has_contacts)
    
    def set_frame_data(self, all_frames_data: list):
        """Loads the frame-by-frame data."""
        if not all_frames_data:
            print("Error: No frame data provided.")
            return

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