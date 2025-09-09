# Standard library imports
import bisect
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# Third-party imports
import numpy as np
import open3d as o3d
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QSlider, QVBoxLayout, QWidget
)
from pyvistaqt import QtInteractor

class PointCloudData:
    """
    A simple data container for a single point cloud, separating geometry from color.
    This class remains unchanged.
    """
    def __init__(self, points: Optional[np.ndarray], color: Optional[np.ndarray] = None):
        if points is None:
            if color is not None:
                raise ValueError("Color data cannot be provided if points are None.")
            self.points = np.empty((0, 3), dtype=np.float32)
            self.color = None
            return
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be an (N, 3) numpy array, but got shape {points.shape}")
        if color is not None:
            if not isinstance(color, np.ndarray) or color.shape != (points.shape[0], 3):
                raise ValueError(
                    f"Color array shape mismatch. Expected ({points.shape[0]}, 3), "
                    f"but got {color.shape}."
                )
        self.points = points
        self.color = color


class SceneObject(ABC):
    """
    Abstract base class for any object that can be visualized in the scene.
    """
    def __init__(self, name: str):
        if not name:
            raise ValueError("SceneObject must have a non-empty name.")
        self.name = name
        self.visible = True
        
        # --- ✨ NEW: State for color override ---
        self.color_override_active = False
        self.override_color = 'magenta' # The color to use when highlighted
        
        self.update_callback: Optional[Callable[[], None]] = None

    @abstractmethod
    def add_to_plotter(self, plotter: pv.Plotter, frame_index: int):
        """Adds the object's mesh for a specific frame to the plotter."""
        pass

    @abstractmethod
    def get_max_frame(self) -> int:
        """Returns the maximum frame index for which this object has data."""
        pass
        
    def create_controls(self, update_callback: Callable[[], None]) -> Optional[QWidget]:
        """Creates a QWidget containing controls for this object's parameters."""
        self.update_callback = update_callback
        return None

# --- REFACTORED HIERARCHY START ---

class FrameSequenceObject(SceneObject):
    """
    Abstract base class for any time-series object based on a frame dictionary.
    """
    def __init__(self, name: str, frame_data: Dict[int, Any], **kwargs):
        super().__init__(name)
        if not isinstance(frame_data, dict):
            raise TypeError("The 'frame_data' must be a dictionary.")
        self.frame_data = frame_data
        self.actor_settings = kwargs

    def get_max_frame(self) -> int:
        """Returns the maximum frame index from the frame_data dictionary keys."""
        if not self.frame_data:
            return -1
        return max(self.frame_data.keys())

    @abstractmethod
    def _get_frame_data(self, frame_index: int) -> Optional[Any]:
        """Abstract method for retrieving data for a specific frame."""
        pass


class PointCloudSequence(FrameSequenceObject):
    """
    Class for non-persistent point clouds.
    """     
    def __init__(self, name: str, frame_data: Dict[int, PointCloudData], **kwargs):
        super().__init__(name, frame_data, **kwargs)
        self.actor_settings.setdefault('render_points_as_spheres', True)
        self.actor_settings.setdefault('point_size', 5.0)

    def _get_frame_data(self, frame_index: int) -> Optional[PointCloudData]:
        """Implements non-persistent data retrieval (exact key match)."""
        return self.frame_data.get(frame_index)

    def add_to_plotter(self, plotter: pv.Plotter, frame_index: int):
        """Renders a `PointCloudData` object."""
        if not self.visible:
            return

        pc_data = self._get_frame_data(frame_index)
        if pc_data and pc_data.points is not None and pc_data.points.shape[0] > 0:
            cloud = pv.PolyData(pc_data.points)
            
            # --- ✅ MODIFIED: Check for color override before applying original colors ---
            if self.color_override_active:
                plotter.add_mesh(cloud, color=self.override_color, name=self.name, **self.actor_settings)
            elif pc_data.color is not None:
                cloud['colors'] = pc_data.color.astype(np.uint8)
                plotter.add_mesh(cloud, scalars='colors', rgb=True, name=self.name, **self.actor_settings)
            else:
                plotter.add_mesh(cloud, color='cyan', name=self.name, **self.actor_settings)

    def create_controls(self, update_callback: Callable[[], None]) -> Optional[QWidget]:
        """Creates the point size slider control."""
        super().create_controls(update_callback)
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        label = QLabel("Point Size:")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(20)
        slider.setValue(int(self.actor_settings.get('point_size', 5)))
        slider.valueChanged.connect(self._set_point_size)
        layout.addWidget(label)
        layout.addWidget(slider)
        return widget

    def _set_point_size(self, size: int):
        self.actor_settings['point_size'] = size
        if self.update_callback:
            self.update_callback()

class LazyPointCloudSequence(PointCloudSequence):
    """
    A scene object for point clouds that loads frame data on-demand.
    """
    def __init__(self, name: str, data_source: Any, **kwargs):
        super().__init__(name, frame_data={}, **kwargs)
        self.data_source = data_source
        if not (hasattr(data_source, '__getitem__') and hasattr(data_source, '__len__')):
            raise TypeError("The 'data_source' must support indexing `[]` and `len()`.")

    def get_max_frame(self) -> int:
        """Returns the max frame index based on the length of the data source."""
        num_frames = len(self.data_source)
        return num_frames - 1 if num_frames > 0 else -1

    def _get_frame_data(self, frame_index: int) -> Optional[PointCloudData]:
        """Retrieves data on-demand from the data source."""
        try:
            return self.data_source[frame_index]
        except IndexError:
            return None

class PersistentPointCloudSequence(PointCloudSequence):
    """
    A PointCloudSequence that displays data from the most recent available frame.
    """
    def __init__(self, name: str, frame_data: Dict[int, PointCloudData], **kwargs):
        super().__init__(name, frame_data, **kwargs)
        self._sorted_keys = sorted(self.frame_data.keys())

    def _get_frame_data(self, frame_index: int) -> Optional[PointCloudData]:
        """Overrides to retrieve data with a "hold-last-frame" logic."""
        if not self._sorted_keys:
            return None

        insertion_point = bisect.bisect_right(self._sorted_keys, frame_index)

        if insertion_point == 0:
            return None

        key_to_use = self._sorted_keys[insertion_point - 1]
        return self.frame_data.get(key_to_use)

class PersistentOpen3DPointCloudSequence(PersistentPointCloudSequence):
    """
    A persistent point cloud sequence for `open3d.geometry.PointCloud` objects.
    """
    def add_to_plotter(self, plotter: pv.Plotter, frame_index: int):
        """Overrides to handle `open3d.geometry.PointCloud` objects."""
        if not self.visible:
            return

        o3d_pc: o3d.geometry.PointCloud = self._get_frame_data(frame_index)

        if o3d_pc and o3d_pc.has_points():
            points_np = np.asarray(o3d_pc.points)
            cloud = pv.PolyData(points_np)
            
            # --- ✅ MODIFIED: Check for color override ---
            if self.color_override_active:
                plotter.add_mesh(cloud, color=self.override_color, name=self.name, **self.actor_settings)
            elif o3d_pc.has_colors():
                colors_np = np.asarray(o3d_pc.colors)
                colors_uint8 = (colors_np * 255).astype(np.uint8)
                cloud['colors'] = colors_uint8
                plotter.add_mesh(cloud, scalars='colors', rgb=True, name=self.name, **self.actor_settings)
            else:
                plotter.add_mesh(cloud, name=self.name, **self.actor_settings)

class Open3DTriangleMeshSequence(FrameSequenceObject):
    """
    A non-persistent sequence for `o3d.geometry.TriangleMesh`.
    """
    def _get_frame_data(self, frame_index: int) -> Optional[o3d.geometry.TriangleMesh]:
        """Implements non-persistent data retrieval (exact key match)."""
        return self.frame_data.get(frame_index)

    def add_to_plotter(self, plotter: pv.Plotter, frame_index: int):
        """Renders an `o3d.geometry.TriangleMesh` object."""
        if not self.visible:
            return

        o3d_mesh = self._get_frame_data(frame_index)

        if o3d_mesh and o3d_mesh.has_triangles() and o3d_mesh.has_vertices():
            vertices_np = np.asarray(o3d_mesh.vertices)
            triangles_np = np.asarray(o3d_mesh.triangles)
            faces_np = np.hstack((
                np.full((triangles_np.shape[0], 1), 3, dtype=triangles_np.dtype),
                triangles_np
            ))
            mesh = pv.PolyData(vertices_np, faces_np)

            # --- ✅ MODIFIED: Check for color override ---
            if self.color_override_active:
                plotter.add_mesh(mesh, color=self.override_color, name=self.name, smooth_shading=True, **self.actor_settings)
            elif o3d_mesh.has_vertex_colors():
                colors_np = np.asarray(o3d_mesh.vertex_colors)
                colors_uint8 = (colors_np * 255).astype(np.uint8)
                mesh['colors'] = colors_uint8
                plotter.add_mesh(mesh, scalars='colors', rgb=True, name=self.name, smooth_shading=True, **self.actor_settings)
            else:
                plotter.add_mesh(mesh, name=self.name, smooth_shading=True, **self.actor_settings)


class PersistentOpen3DTriangleMeshSequence(Open3DTriangleMeshSequence):
    """
    The persistent version for `o3d.geometry.TriangleMesh`.
    """
    def __init__(self, name: str, frame_data: Dict[int, o3d.geometry.TriangleMesh], **kwargs):
        super().__init__(name, frame_data, **kwargs)
        self._sorted_keys = sorted(self.frame_data.keys())

    def _get_frame_data(self, frame_index: int) -> Optional[o3d.geometry.TriangleMesh]:
        """Overrides the base method to implement persistent data retrieval."""
        if not self._sorted_keys:
            return None
        
        insertion_point = bisect.bisect_right(self._sorted_keys, frame_index)
        if insertion_point == 0:
            return None
            
        key_to_use = self._sorted_keys[insertion_point - 1]
        return self.frame_data.get(key_to_use)

class Trajectory(SceneObject):
    """A scene object representing a moving point (sphere) over time."""
    def __init__(self, name: str, frame_data: Dict[int, np.ndarray], color: Any = 'gray', radius: float = 2.0, **kwargs):
        super().__init__(name)
        if not isinstance(frame_data, dict):
            raise TypeError("The 'frame_data' must be a dictionary.")
        self.frame_data = frame_data
        self.color = color
        self.radius = radius
        self.actor_settings = kwargs

    def get_max_frame(self) -> int:
        if not self.frame_data:
            return -1
        return max(self.frame_data.keys())

    def add_to_plotter(self, plotter: pv.Plotter, frame_index: int):
        if not self.visible:
            return

        position = self.frame_data.get(frame_index)
        if position is not None:
            sphere = pv.Sphere(radius=self.radius, center=position)
            
            # --- ✅ MODIFIED: Choose color based on override state ---
            render_color = self.override_color if self.color_override_active else self.color
            plotter.add_mesh(sphere, color=render_color, name=self.name, **self.actor_settings)
            
    def create_controls(self, update_callback: Callable[[], None]) -> Optional[QWidget]:
        super().create_controls(update_callback)
        
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)

        label = QLabel("Radius:")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(20)
        slider.setValue(int(self.radius))
        slider.valueChanged.connect(self._set_radius)

        layout.addWidget(label)
        layout.addWidget(slider)
        
        return widget

    def _set_radius(self, radius: int):
        """Updates the sphere radius and triggers a scene refresh."""
        self.radius = float(radius)
        if self.update_callback:
            self.update_callback()


class SceneViewer(QMainWindow):
    """
    The main viewer window.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_objects: Dict[str, SceneObject] = {}
        self.current_index = 0
        
        self.setWindowTitle("3D Scene Navigator")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)

        plotter_container = QWidget()
        plotter_layout = QVBoxLayout(plotter_container)
        self.plotter = QtInteractor(self)
        self.plotter.set_background('midnightblue')
        plotter_layout.addWidget(self.plotter.interactor)
        self._setup_frame_controls(plotter_layout)
        
        self.object_controls_layout = self._setup_object_controls_panel()
        
        main_layout.addWidget(plotter_container, 4)
        main_layout.addLayout(self.object_controls_layout, 1)

        self._update_plot()

    @property
    def num_frames(self) -> int:
        if not self.scene_objects:
            return 1
        max_frames = [obj.get_max_frame() for obj in self.scene_objects.values()]
        valid_max_frames = [f for f in max_frames if f >= 0]
        if not valid_max_frames:
            return 1
        return max(valid_max_frames) + 1

    def add_object(self, obj: SceneObject):
        if obj.name in self.scene_objects:
            print(f"Warning: Overwriting scene object with name '{obj.name}'")
        self.scene_objects[obj.name] = obj

        object_groupbox = QGroupBox(obj.name)
        object_groupbox_layout = QVBoxLayout(object_groupbox)

        # Visibility Checkbox
        checkbox = QCheckBox("Visible")
        checkbox.setChecked(obj.visible)
        checkbox.stateChanged.connect(
            lambda state, name=obj.name: self._on_visibility_changed(name, state)
        )
        object_groupbox_layout.addWidget(checkbox)

        # --- ✨ NEW: Add Highlight checkbox to the UI for this object ---
        highlight_checkbox = QCheckBox("Highlight")
        highlight_checkbox.setChecked(obj.color_override_active)
        highlight_checkbox.stateChanged.connect(
            lambda state, name=obj.name: self._on_color_override_changed(name, state)
        )
        object_groupbox_layout.addWidget(highlight_checkbox)

        # Add any other custom controls
        custom_controls = obj.create_controls(self._update_plot)
        if custom_controls:
            object_groupbox_layout.addWidget(custom_controls)

        self.object_controls_layout.insertWidget(self.object_controls_layout.count() - 1, object_groupbox)
        
        self._update_slider_range()
        self._update_plot()

    def _on_visibility_changed(self, name: str, state: int):
        is_visible = (state == Qt.Checked)
        if name in self.scene_objects:
            self.scene_objects[name].visible = is_visible
            self._update_plot()
            
    # --- ✨ NEW: Handler for the highlight checkbox ---
    def _on_color_override_changed(self, name: str, state: int):
        """Toggles the color override state for an object and refreshes the plot."""
        is_active = (state == Qt.Checked)
        if name in self.scene_objects:
            self.scene_objects[name].color_override_active = is_active
            self._update_plot()

    def _setup_object_controls_panel(self) -> QVBoxLayout:
        controls_layout = QVBoxLayout()
        controls_layout.addStretch()
        return controls_layout
        
    def _update_slider_range(self):
        max_frames = self.num_frames
        self.slider.setMaximum(max_frames - 1 if max_frames > 0 else 0)
        self._update_label()

    def _setup_frame_controls(self, parent_layout: QVBoxLayout):
        frame_controls_layout = QHBoxLayout()
        self.recenter_button = QPushButton("Recenter View")
        self.recenter_button.clicked.connect(self._recenter_view)
        
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
        parent_layout.addLayout(frame_controls_layout)
        
        self._update_label()

    def _recenter_view(self):
        if not self.scene_objects:
            self.plotter.reset_camera()
            return
        
        all_bounds = []
        for name, scene_obj in self.scene_objects.items():
            if scene_obj.visible and name in self.plotter.actors:
                actor = self.plotter.actors[name]
                if actor.bounds:
                    all_bounds.append(actor.bounds)

        if not all_bounds:
            self.plotter.reset_camera()
            return
            
        bounds_array = np.array(all_bounds)
        min_point = np.min(bounds_array[:, ::2], axis=0)
        max_point = np.max(bounds_array[:, 1::2], axis=0)
        combined_bounds = [min_point[0], max_point[0], min_point[1], max_point[1], min_point[2], max_point[2]]
        self.plotter.reset_camera(bounds=combined_bounds)
        self.plotter.camera.zoom(1.3)

    def _on_slider_change(self, value: int):
        self.current_index = value
        self._update_plot()
        self._update_label()

    def _update_label(self):
        total_frames = self.num_frames
        text = f"Frame: {self.current_index + 1} / {total_frames}"
        self.label.setText(text)

    def _update_plot(self):
        self.plotter.clear()
        
        axis_length = 250
        self.plotter.add_axes(interactive=False, line_width=axis_length, box=True)
        self.plotter.add_mesh(pv.Sphere(radius=2.0), color='yellow', pickable=False)

        for obj in self.scene_objects.values():
            obj.add_to_plotter(self.plotter, self.current_index)

        if not self.plotter.camera_set:
            self._recenter_view()


def main():
    app = QApplication(sys.argv)

    # --- 1. Create Data ---
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
    mesh_10.paint_uniform_color([1.0, 0.2, 0.2]) # Original color is red
    mesh_10.translate([-50, 0, 30])
    mesh_dict[10] = mesh_10
    mesh_40 = o3d.geometry.TriangleMesh.create_sphere(radius=15)
    mesh_40.compute_vertex_normals()
    mesh_40.paint_uniform_color([0.2, 0.2, 1.0]) # Original color is blue
    mesh_40.translate([50, 0, -10])
    mesh_dict[40] = mesh_40

    # --- 2. Instantiate the Viewer ---
    viewer = SceneViewer()

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