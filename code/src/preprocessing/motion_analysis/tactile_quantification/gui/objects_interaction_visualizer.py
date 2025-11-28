import sys
import numpy as np
import pyvista as pv
import open3d as o3d

import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ObjectsInteractionVisualizer(QMainWindow):
    """
    Manages the visualization of 3D object interactions with a timeline slider
    using PyQt6 for the UI and PyVista for 3D rendering.
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
        # Import QtInteractor locally to prevent import-time side effects
        # that require a QApplication instance before it's created.
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
        # Subplots are now created dynamically in _setup_plots
        self.canvas = FigureCanvas(self.fig)
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
                # Create a placeholder mesh that is invisible. This allows the actor
                # to exist for future updates. A single degenerate triangle is used.
                plot_verts = np.array([[0, 0, -9999]] * 3)
                plot_faces = np.array([3, 0, 1, 2])
            else:
                plot_verts = vertices
                triangles = np.asarray(geometry.triangles)
                plot_faces = np.hstack((np.full((len(triangles), 1), 3), triangles)).flatten()

            pv_object = pv.PolyData(plot_verts, faces=plot_faces)
            # Passed point_size, though it primarily affects vertices if style='points' is used.
            actor = self.plotter.add_mesh(pv_object, color=color, style='wireframe', point_size=point_size)
            actor.SetVisibility(not is_empty)

        elif isinstance(geometry, o3d.geometry.PointCloud):
            # Create a placeholder if empty, so the actor exists for updates.
            plot_verts = vertices if not is_empty else np.array([[0, 0, -9999]])
            pv_object = pv.PolyData(plot_verts)
            
            # MODIFICATION: explicitly using the passed point_size argument
            actor = self.plotter.add_mesh(
                pv_object, 
                color=color, 
                point_size=point_size, 
                render_points_as_spheres=True
            )
            # Visibility for dynamic point clouds like 'contacts' is managed
            # by the _update_pyvista_scene method. Set initial visibility here.
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
        # If it exists, remove the old geometry from the viewer.
        # Using reset_bounding_box=False prevents the camera from auto-resizing.
        if self.is_existing_geometry(name):
            self.plotter.remove_actor(self.actors[name])
        self.add_geometry(name, geometry)

    def add_plot(self, title: str, data_vector: list, color: str = 'r', y_label: str = 'Value'):
        """
        Registers a data vector to be plotted. Call this for each plot you want.
        The plots will be created with shared X-axes when .run() is called.

        Args:
            title (str): The title for the entire plot area (only used from the first call).
            data_vector (list): The list of numerical data to plot.
            color (str, optional): The color for the plot line. Defaults to 'r'.
            y_label (str, optional): The label for the Y-axis of this specific plot.
        """
        if not self.all_frames_data:
            print("Warning: Set frame data with `set_frame_data()` before adding a plot.")
            return

        if len(data_vector) != len(self.all_frames_data):
            print(f"Error: Plot data length ({len(data_vector)}) differs from frame count "
                  f"({len(self.all_frames_data)}). The plot may be incorrect.")
            return

        # Store plot configuration to be processed later in _setup_plots
        self.plot_configs.append({
            'title': title,
            'data': data_vector,
            'color': color,
            'y_label': y_label
        })

    def recenter_view_on_point(self, point: list):
        """
        Recenters the 3D view's camera to focus on a specific point.

        This makes subsequent rotations, panning, and zooming operations
        centered around the provided point, improving user interaction when
        inspecting a specific area of interest.

        Args:
            point (list): The [x, y, z] coordinates of the new focal point.
                          Can also be a tuple or NumPy array.
        """
        if self.plotter:
            # Set the camera's focal point to the specified coordinates
            self.plotter.set_focus(point)
            # Adjust the clipping range to ensure the whole scene is visible
            self.plotter.reset_camera_clipping_range()

    def _toggle_playback(self):
        """Starts or stops the animation playback."""
        if self.is_playing:
            self._pause_playback()
        else:
            # If at the end of the timeline, restart from the beginning
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
            # This will trigger the `valueChanged` signal, which calls `_update_frame`
            self.frame_slider.setValue(current_frame + 1)
        else:
            self._pause_playback()  # Stop playback at the end

    def _setup_plots(self):
        """
        Configures and renders subplots with a clean, professional aesthetic.

        This method creates a series of vertically stacked subplots, applying
        a minimalistic style suitable for publications. It removes unnecessary
        chart elements ("chart junk") like top and right spines and uses a
        subtle grid to guide the eye without distracting.
        """
        if not self.plot_configs:
            return

        # --- Style Configuration for a Professional Look ---
        # Using a context manager ensures these style changes are local to this plot
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
            
            # Create subplots that share the X-axis
            self.axes = self.fig.subplots(
                nrows=num_plots, 
                ncols=1, 
                sharex=True
            )
            
            # Ensure self.axes is always iterable, even for a single subplot
            self.axes = np.atleast_1d(self.axes)

            # Use the title from the first plot config as the main figure title
            self.fig.suptitle(
                self.plot_configs[0].get('title', 'Sensor Data'), 
                fontsize=16, 
                fontweight='bold'
            )
            
            time_axis = range(len(self.all_frames_data))
            self.v_lines.clear() # Ensure the list of lines is reset

            # Iterate through axes and their corresponding configurations
            for ax, config in zip(self.axes, self.plot_configs):
                # Plot the primary data
                ax.plot(
                    time_axis, 
                    config['data'], 
                    color=config.get('color', '#007ACC'), # A professional blue
                    linewidth=1.5
                )

                # Set labels with specific font sizes
                ax.set_ylabel(config.get('y_label', ''), fontsize=12)

                # Add a vertical cursor line
                v_line = ax.axvline(0, color='#D32F2F', linestyle='--', linewidth=1) # A subtle red
                self.v_lines.append(v_line)

            # Configure the shared X-axis on the bottom-most plot
            self.axes[-1].set_xlabel("Frame ID", fontsize=12)

            # Adjust layout to prevent titles/labels from overlapping
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    def _update_frame(self, frame_index: int):
        """Updates all visual components (3D scene, plot cursor, text) for a given frame."""
        self.frame_entry.setText(str(frame_index))
        self._update_plot_cursor(frame_index)
        if 0 <= frame_index < len(self.all_frames_data):
            self._update_pyvista_scene(self.all_frames_data[frame_index])

    def _update_plot_cursor(self, frame_index: int):
        """Updates the vertical cursor and slides the plot's x-axis window."""
        if not self.axes.any():
            return

        # Update vertical cursor lines to stay centered
        for v_line in self.v_lines:
            v_line.set_xdata([frame_index, frame_index])

        # Calculate the boundaries for the sliding window
        total_frames = len(self.all_frames_data)
        half_width = self.plot_window_half_width
        
        x_min = frame_index - half_width
        x_max = frame_index + half_width

        # Adjust window at the beginning of the data range
        if x_min < 0:
            x_max -= x_min  # Keep window size constant
            x_min = 0
        
        # Adjust window at the end of the data range
        if x_max >= total_frames:
            x_min -= (x_max - (total_frames - 1))
            x_max = total_frames - 1

        # Update x-axis limits for all subplots to create the sliding effect
        for ax in self.axes:
            ax.set_xlim(max(0, x_min), x_max)

        # Use draw_idle for better performance during rapid slider dragging
        self.canvas.draw_idle()

    def _update_from_text_entry(self):
        """Validates the text entry and updates the scene and slider."""
        self._pause_playback()  # Stop playback on manual frame entry
        try:
            val = int(self.frame_entry.text())
            max_val = len(self.all_frames_data) - 1
            # Clamp value to valid range
            clamped_val = max(0, min(val, max_val))

            # Update slider position. This emits `valueChanged`, which triggers `_update_frame`.
            self.frame_slider.setValue(clamped_val)
        except (ValueError, IndexError):
            # On error, reset text to current slider value
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
                    # For meshes where topology (triangles) is constant,
                    # directly updating points is performant and correct.
                    hand_actor.mapper.dataset.points = new_points
                hand_actor.SetVisibility(is_visible)

            # Update contact points with type safety for int(0)
            contacts_actor = self.actors.get('contacts')
            raw_contacts = frame_data.get('contact_points')

            # 1. Sanitize the input to ensure we have a valid numpy array
            if isinstance(raw_contacts, (int, float)) or raw_contacts is None:
                # Handle the '0' case or None
                points_data = np.array([])
            elif isinstance(raw_contacts, np.ndarray):
                points_data = raw_contacts
            else:
                # Fallback for unexpected types
                points_data = np.array([])

            if contacts_actor:
                # 2. Determine if we have data to show
                has_contacts = points_data.size > 0
                
                # 3. Handle the data update
                if has_contacts:
                    render_points = points_data
                else:
                    # PyVista/VTK prevents empty datasets in some contexts,
                    # so we use a dummy point hidden far away.
                    render_points = np.array([[0, 0, -9999]])

                # 4. Update the actor using shallow_copy
                # PROBLEM FIX: Previously, we only updated .points. If the number of points changed,
                # the 'verts' (topology) array in PolyData was not updated, causing only the 
                # first N vertices (where N is the old count) to be rendered.
                # creating a new PolyData object automatically generates the correct 'verts' topology.
                new_mesh = pv.PolyData(render_points)
                
                # shallow_copy updates the geometry AND topology pointers of the existing dataset
                # efficiently without breaking the actor/mapper pipeline.
                contacts_actor.mapper.dataset.shallow_copy(new_mesh)
                
                contacts_actor.SetVisibility(has_contacts)
    
    def set_frame_data(self, all_frames_data: list):
        """
        Loads the frame-by-frame data into the visualizer and sets up controls.
        """
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

        # Setup plots now that all data and configs have been added
        self._setup_plots()

        # --- Trigger initial visualization for the first frame ---
        self._update_frame(0) # This updates all visual components

        self.show()

# ==============================================================================
# MOCKUP MAIN FUNCTION FOR TESTING (NO CHANGES NEEDED HERE)
# ==============================================================================
def create_mock_data(num_frames=200):
    # (Same as your original code)
    print("Generating mock data...")
    object_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    object_mesh.compute_vertex_normals()
    object_mesh.paint_uniform_color([0.2, 0.4, 0.8])
    hand_mesh_initial = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.8, tube_radius=0.25)
    hand_mesh_initial.compute_vertex_normals()
    hand_mesh_initial.paint_uniform_color([0.8, 0.2, 0.2])
    all_frames = []
    for i in range(num_frames):
        current_hand_mesh = o3d.geometry.TriangleMesh(hand_mesh_initial)
        dx = 2.5 * np.sin(i / num_frames * 2 * np.pi)
        transform = np.identity(4)
        transform[0, 3] = dx
        current_hand_mesh.transform(transform)
        contact_points = np.array([])
        contact_normals = np.array([])
        avg_force = 0
        contact_area = 0
        if abs(dx) < 1.0:
            num_points = int(np.random.uniform(5, 50))
            contact_cloud = object_mesh.sample_points_uniformly(number_of_points=num_points)
            contact_points = np.asarray(contact_cloud.points)
            contact_normals = np.asarray(contact_cloud.normals)
            avg_force = np.random.uniform(0.5, 5.0)
            contact_area = num_points * 0.15
        frame_data = {
            'transformed_hand_mesh': current_hand_mesh,
            'contact_points': contact_points,
            'contact_normals': contact_normals,
            'avg_force': avg_force,
            'contact_area': contact_area,
        }
        all_frames.append(frame_data)
    print("Mock data generation complete.")
    return all_frames, object_mesh, hand_mesh_initial

if __name__ == "__main__":
    if QApplication.instance() is None:
        app = QApplication(sys.argv)
    
    visualizer = ObjectsInteractionVisualizer(width=1600)

    frames_data, static_object, initial_hand = create_mock_data(num_frames=200)

    # 1. Load the frame data into the visualizer
    visualizer.set_frame_data(frames_data)

    # 2. Extract data vectors and add them as individual plots
    avg_force_data = [frame.get('avg_force', 0) for frame in frames_data]
    contact_area_data = [frame.get('contact_area', 0) for frame in frames_data]

    visualizer.add_plot(
        title="Contact Properties",
        data_vector=avg_force_data,
        color='g',
        y_label='Average Force (N)'
    )
    visualizer.add_plot(
        title="Contact Properties", # This title is ignored, only the first one is used
        data_vector=contact_area_data,
        color='m',
        y_label='Contact Area (cm^2)'
    )

    # 3. Add the 3D geometries
    initial_contacts = o3d.geometry.PointCloud()
    if 'contact_points' in frames_data[0] and len(frames_data[0]['contact_points']) > 0:
        points = frames_data[0]['contact_points']
        initial_contacts.points = o3d.utility.Vector3dVector(points)
    initial_contacts.paint_uniform_color([0.1, 0.9, 0.1])

    visualizer.add_geometry(name="forearm", geometry=static_object)
    visualizer.add_geometry(name="hand", geometry=initial_hand)
    visualizer.add_geometry(name="contacts", geometry=initial_contacts)

    # To recenter the view on a reference object (like a point cloud or mesh),
    # you first calculate its center point. The .get_center() method from
    # Open3D can be used on most geometry types.
    ref_pcd = static_object  # Using the static object as our reference
    center_point = ref_pcd.get_center()
    visualizer.recenter_view_on_point(center_point)

    # 4. Run the application
    visualizer.run()
    sys.exit(app.exec_())