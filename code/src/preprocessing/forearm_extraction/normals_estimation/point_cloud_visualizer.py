# point_cloud_visualizer.py
import pyvista as pv
from pyvistaqt.plotting import BackgroundPlotter

import numpy as np
import os
import sys

print("-" * 50)
print(f"🐍 Python Executable: {sys.executable}")
print(f"📜 Using PyVista Version: {pv.__version__}")

# Using a string literal for the forward reference is cleaner and standard practice.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .point_cloud_controller import PointCloudController


class PointCloudVisualizer:
    """
    Manages the PyVista visualization. It is the "View" in MVC.
    It sends all user interaction events to the Controller.
    """
    def __init__(self, controller: 'PointCloudController', source_file: str):
        self.controller = controller
        self.model = controller.model  # Keep a reference for easy access to data
        # This plotter runs in a background thread and does not block.
        self.plotter = BackgroundPlotter(
            title=f"Normals for {os.path.basename(source_file)}",
            window_size=(1600, 1000),
            off_screen=False,
        )
        # Add key bindings for quitting the application
        self.plotter.add_key_event('q', self.controller.request_save_and_close)
        self.plotter.add_key_event('Return', self.controller.request_save_and_close)

        self.plotter.title = f"Normals for {os.path.basename(source_file)}"
        
        self.point_size = 8.0
        self.glyph_factor = 4.0

        self.point_actor = None
        self.glyph_actor = None
        self.bounds_text_actor = None
        
        # Store references to sliders for dynamic range updates
        self.vp_x_slider = None
        self.vp_y_slider = None
        self.vp_z_slider = None

    def setup_scene(self):
        """
        Configures the plotter and adds all initial actors and widgets.
        This is a non-blocking operation.
        """
        print("Setting up visualizer scene...")
        self.plotter.set_background('white')
        self._add_widgets()
        self.update_plot() # Render the initial state

    def close(self):
        """Closes the plotter window. This will allow the main script to exit."""
        if self.plotter:
            print("Visualizer closing background plotter.")
            self.plotter.close()
            print("plotter closed!")
    
    def is_active(self) -> bool:
        """Check if the plotter window is still open."""
        # The BackgroundPlotter tracks its actors. An empty list means it's closed.
        return self.plotter.renderer is not None and len(self.plotter.actors) > 0

    def _setup_plotter(self, width=1600, height=1000):
        """Configures the plotter's window size and background color."""
        self.plotter.window_size = [width, height]
        self.plotter.set_background('white')

    def _create_slider_row(self, y_pos_px, slider_configs, length_to_height_ratio=10.0):
        """Creates a horizontal row of sliders and calculates its total vertical height."""
        slider_length_px = 250
        slider_spacing_px = 30
        row_start_x_px = 10
        title_height_norm = 0.015

        window_width, window_height = self.plotter.window_size
        
        safe_ratio = length_to_height_ratio if length_to_height_ratio > 0 else 10.0
        slider_tube_height_px = slider_length_px / safe_ratio
        title_height_px = title_height_norm * window_height
        total_row_height_px = title_height_px + slider_tube_height_px
        
        y_norm = y_pos_px / window_height
        len_norm = slider_length_px / window_width
        spacing_norm = slider_spacing_px / window_width
        current_x_norm = row_start_x_px / window_width
        tube_width_ratio = 1.0 / (2.0 * safe_ratio)

        for config in slider_configs:
            pointa = (current_x_norm, y_norm)
            pointb = (current_x_norm + len_norm, y_norm)
            
            slider_widget = self.plotter.add_slider_widget(
                callback=config['callback'],
                rng=config['rng'],
                value=config['value'],
                title=config['title'],
                style='modern',
                title_height=title_height_norm,
                pointa=pointa,
                pointb=pointb,
                tube_width=tube_width_ratio
            )
            
            if config.get('save_as'):
                setattr(self, config['save_as'], slider_widget)

            current_x_norm += len_norm + spacing_norm
                
        return total_row_height_px
    
    def _add_widgets(self):
        """Adds all UI widgets to the plotter window."""
        x_padding = 10
        y_padding_header = 75
        y_padding_widget = 25
        y_padding_group = 30
        font_size_header = 10
        font_size_label = 8
        checkbox_size = 20
        checkbox_label_spacing = 5 
        
        # --- TOP-LEFT BLOCK: Processing Controls ---
        current_y = self.plotter.window_size[1] - 50
        self.plotter.add_text("--- Processing Controls ---", position=(x_padding, current_y), font_size=font_size_header, color="darkred")
        current_y -= y_padding_header

        proc_sliders_1 = [
            {'callback': self._set_k_neighbors_callback, 'rng': [3, 100], 'value': self.model.k_neighbors, 'title': "K-Neighbors"},
            {'callback': self._set_radius_callback, 'rng': [0.01, 2.0], 'value': self.model.radius, 'title': "Radius"}
        ]
        total_row_height_px = self._create_slider_row(current_y, proc_sliders_1)
        current_y -= (total_row_height_px + y_padding_widget)
        
        cb_pos_hybrid = (x_padding, current_y)
        self.plotter.add_checkbox_button_widget(self._toggle_hybrid_callback, value=self.model.hybrid_tree, position=cb_pos_hybrid, size=checkbox_size)
        self.plotter.add_text("Hybrid Search", position=(cb_pos_hybrid[0] + checkbox_size + checkbox_label_spacing, current_y + 2), font_size=font_size_label)
        current_y -= (checkbox_size + y_padding_widget)

        cb_pos_align = (x_padding, current_y)
        self.plotter.add_checkbox_button_widget(self._toggle_viewpoint_align_callback, value=self.model.align_with_viewpoint, position=cb_pos_align, size=checkbox_size)
        self.plotter.add_text("Align to Viewpoint (if activated: adjust VPX, VPY, VPZ below)", position=(cb_pos_align[0] + checkbox_size + checkbox_label_spacing, current_y + 2), font_size=font_size_label)
        current_y -= (checkbox_size + y_padding_widget + y_padding_group)

        vp_sliders = [
            {'callback': self._set_viewpoint_x_callback, 'rng': [-1, 1], 'value': self.model.viewpoint[0], 'title': "VP X", 'save_as': 'vp_x_slider'},
            {'callback': self._set_viewpoint_y_callback, 'rng': [-1, 1], 'value': self.model.viewpoint[1], 'title': "VP Y", 'save_as': 'vp_y_slider'},
            {'callback': self._set_viewpoint_z_callback, 'rng': [-1, 1], 'value': self.model.viewpoint[2], 'title': "VP Z", 'save_as': 'vp_z_slider'}
        ]
        total_row_height_px = self._create_slider_row(current_y, vp_sliders)
        current_y -= (total_row_height_px + y_padding_widget)

        cb_pos_recompute = (x_padding, current_y)
        self.plotter.add_checkbox_button_widget(self._recompute_normals_callback, value=False, position=cb_pos_recompute, size=checkbox_size, color_on='dodgerblue', color_off='dimgrey')
        self.plotter.add_text("Recompute Normals", position=(cb_pos_recompute[0] + checkbox_size + checkbox_label_spacing, current_y + 2), font_size=font_size_label)

        # --- BOTTOM-LEFT BLOCK: Visual Controls ---
        current_y = 140
        self.plotter.add_text("--- Visual Controls ---", position=(x_padding, current_y), font_size=font_size_header, color="darkblue")
        current_y -= y_padding_header

        vis_sliders = [
            {'callback': self._set_arrow_length_callback, 'rng': [0.1, 10], 'value': self.glyph_factor, 'title': "Arrow Len"},
            {'callback': self._set_point_size_callback, 'rng': [1, 20], 'value': self.point_size, 'title': "Pt Size"},
            {'callback': self._set_scale_factor_callback, 'rng': [0.1, 5.0], 'value': self.model.scale_factor, 'title': "Scale"}
        ]
        total_row_height_px = self._create_slider_row(current_y, vis_sliders)
        current_y -= (total_row_height_px + y_padding_widget)

        cb_pos_flip = (x_padding, current_y)
        self.plotter.add_checkbox_button_widget(self._flip_normals_callback, value=self.model.normals_flipped, position=cb_pos_flip, size=checkbox_size)
        self.plotter.add_text("Flip Normals", position=(cb_pos_flip[0] + checkbox_size + checkbox_label_spacing, current_y + 2), font_size=font_size_label)
        current_y -= y_padding_widget

        cb_pos_center = (x_padding, current_y)
        self.plotter.add_checkbox_button_widget(self._toggle_centering_callback, value=self.model.is_centered, position=cb_pos_center, size=checkbox_size)
        self.plotter.add_text("Center Cloud", position=(cb_pos_center[0] + checkbox_size + checkbox_label_spacing, current_y + 2), font_size=font_size_label)

        # --- The "Save & Exit" button and its label have been removed. ---

    def update_viewpoint_slider_ranges(self, bounds: tuple[np.ndarray, np.ndarray]):
        """Dynamically updates the viewpoint slider ranges based on cloud bounds."""
        min_bound, max_bound = bounds
        center = (min_bound + max_bound) / 2
        span = (max_bound - min_bound) * 2.0
        
        sliders = [self.vp_x_slider, self.vp_y_slider, self.vp_z_slider]
        for i, slider in enumerate(sliders):
            if slider:
                rep = slider.GetRepresentation()
                rep.SetMinimumValue(center[i] - span[i])
                rep.SetMaximumValue(center[i] + span[i])
                if self.model.viewpoint[i] == 0.0:
                    self.controller.set_viewpoint(i, center[i])
                    slider.SetValue(center[i])

    # --- Callbacks (delegate to the controller) ---
    def _recompute_normals_callback(self, state: bool):
        self.controller.recompute_normals()
        
    def _set_k_neighbors_callback(self, k: float):
        self.controller.set_k_neighbors(int(k))

    def _set_radius_callback(self, radius: float):
        self.controller.set_radius(radius)
    
    def _toggle_hybrid_callback(self, state: bool):
        self.controller.toggle_hybrid(state)

    def _toggle_viewpoint_align_callback(self, state: bool):
        self.controller.toggle_viewpoint_align(state)

    def _set_viewpoint_x_callback(self, val: float): self.controller.set_viewpoint(0, val)
    def _set_viewpoint_y_callback(self, val: float): self.controller.set_viewpoint(1, val)
    def _set_viewpoint_z_callback(self, val: float): self.controller.set_viewpoint(2, val)
    
    def _toggle_centering_callback(self, state: bool): self.controller.toggle_centering(state)
    def _set_scale_factor_callback(self, scale: float): self.controller.set_scale_factor(scale)
    def _flip_normals_callback(self, state: bool): self.controller.flip_normals(state)
    
    def _set_point_size_callback(self, size: float):
        self.point_size = size
        if self.point_actor:
            self.point_actor.GetProperty().SetPointSize(self.point_size)

    def _set_arrow_length_callback(self, length: float):
        self.glyph_factor = length
        self.update_glyphs()
        
    # --- Plot Update Methods (called by controller) ---
    def update_plot(self):
        """Updates or creates the main point cloud actor."""
        points = self.model.get_transformed_points()
        cloud = pv.PolyData(points)
        cloud['normals'] = self.model.normals
        
        if self.point_actor:
            # More efficient update by modifying the existing dataset
            self.point_actor.mapper.dataset.points = cloud.points
            self.point_actor.mapper.dataset.Modified()
        else:
            self.point_actor = self.plotter.add_mesh(cloud, color='skyblue', point_size=self.point_size, render_points_as_spheres=True, name="points")
            
        self.update_glyphs()
        self.update_bounds_display()

    def update_glyphs(self):
        """Updates or creates the normal vector glyphs."""
        if self.glyph_actor:
            self.plotter.remove_actor(self.glyph_actor, render=False)
        
        points = self.model.get_transformed_points()
        normals = self.model.normals
        
        if points.size == 0 or normals is None or normals.size == 0:
            return

        cloud_for_glyphs = pv.PolyData(points)
        cloud_for_glyphs['normals'] = normals
        
        arrow = pv.Arrow()
        glyphs = cloud_for_glyphs.glyph(orient='normals', scale=False, factor=self.glyph_factor, geom=arrow)
        self.glyph_actor = self.plotter.add_mesh(glyphs, color='gold')

    def update_bounds_display(self):
        """Updates the text display for the point cloud's bounding box."""
        if self.bounds_text_actor:
            self.plotter.remove_actor(self.bounds_text_actor, render=False)
            
        points = self.model.get_transformed_points()
        if points.size == 0:
            bounds_text = "No points to display."
        else:
            min_vals, max_vals = np.min(points, axis=0), np.max(points, axis=0)
            bounds_text = (
                f"Bounds (X): [{min_vals[0]:.2f}, {max_vals[0]:.2f}]\n"
                f"Bounds (Y): [{min_vals[1]:.2f}, {max_vals[1]:.2f}]\n"
                f"Bounds (Z): [{min_vals[2]:.2f}, {max_vals[2]:.2f}]"
            )
        self.bounds_text_actor = self.plotter.add_text(bounds_text, position='upper_right', font_size=12, color='black')

