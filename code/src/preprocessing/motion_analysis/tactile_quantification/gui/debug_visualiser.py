import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import ttk


class DebugVisualizer:
    """
    A dedicated class for visualizing the state of a tactile simulation frame for debugging.
    """
    def __init__(self, ref_pcd: o3d.geometry.PointCloud):
        """
        Initializes the visualizer with the static reference point cloud.
        
        Args:
            ref_pcd (o3d.geometry.PointCloud): The reference point cloud to be used as a base for visualization.
        """
        if not isinstance(ref_pcd, o3d.geometry.PointCloud):
            raise TypeError("ref_pcd must be an Open3D PointCloud object.")
        self.ref_pcd = ref_pcd
        self._highlight_geoms = []

    def show(self,
             transformed_vertices: np.ndarray,
             contact_points: np.ndarray,
             min_indices: np.ndarray,
             dot_products: np.ndarray,
             inside_mask: np.ndarray,
             normals_at_closest_points: np.ndarray,
             unique_contact_indices: np.ndarray,
             show_normals: bool = True,
             normal_length: float = 1.00):
        """
        Pops up a window showing the current frame's state with interactive controls.
        
        Displays:
        - A 3D scene with the point clouds.
        - A separate control window with a slider to select and highlight a vertex.
        """
        # --- 1. Setup Open3D Visualizer ---
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Debug Visualizer")

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15])
        opt.point_size = 3.0

        # --- 2. Create and Add Geometries ---
        transformed_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_vertices))
        transformed_pcd.paint_uniform_color([1, 0, 0])  # Red
        vis.add_geometry(transformed_pcd)

        if contact_points.size > 0:
            contact_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(contact_points))
            contact_pcd.paint_uniform_color([0, 1, 0])  # Green
            vis.add_geometry(contact_pcd)

        ref_pcd_copy = o3d.geometry.PointCloud(self.ref_pcd)
        if not ref_pcd_copy.has_colors():
            ref_pcd_copy.paint_uniform_color([0.5, 0.5, 0.5])  # Grey
        vis.add_geometry(ref_pcd_copy)

        if show_normals and self.ref_pcd.has_normals():
            normal_lines = self._create_normal_lines(normal_length)
            vis.add_geometry(normal_lines)

        scene_bbox = ref_pcd_copy.get_axis_aligned_bounding_box()
        highlight_radius = scene_bbox.get_max_extent() / 150.0

        # --- 3. Setup Tkinter Controls ---
        root = tk.Tk()
        root.title("Controls")
        # root.resizable(False, False) # Allow resizing

        selected_index = tk.IntVar(value=0)

        def update_highlight(slider_val):
            idx = selected_index.get()

            for geom in self._highlight_geoms:
                vis.remove_geometry(geom, reset_bounding_box=False)
            self._highlight_geoms.clear()

            transformed_pt = transformed_vertices[idx]
            ref_pt_idx = min_indices[idx]
            ref_pt = np.asarray(self.ref_pcd.points)[ref_pt_idx]

            # A sphere mesh is used to make the highlight visibly larger than a point.
            # Highlight for transformed vertex (Blue)
            sphere_transformed = o3d.geometry.TriangleMesh.create_sphere(radius=highlight_radius)
            sphere_transformed.translate(transformed_pt)
            sphere_transformed.paint_uniform_color([0.2, 0.5, 1.0])

            # Highlight for reference vertex (Cyan)
            sphere_ref = o3d.geometry.TriangleMesh.create_sphere(radius=highlight_radius)
            sphere_ref.translate(ref_pt)
            sphere_ref.paint_uniform_color([0.0, 1.0, 1.0])

            self._highlight_geoms.extend([sphere_transformed, sphere_ref])

            for geom in self._highlight_geoms:
                vis.add_geometry(geom, reset_bounding_box=False)

            # Update text labels
            slider_label.config(text=f"Selected vertex: {idx}")
            dot_product_label.config(text=f"Dot Product: {dot_products[idx]:.4f}")
            inside_mask_label.config(text=f"Inside Mask: {inside_mask[idx]}")
            normal = normals_at_closest_points[idx]
            normal_label.config(text=f"Closest Normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Slider Frame ---
        slider_frame = ttk.Frame(main_frame)
        slider_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        slider_frame.columnconfigure(0, weight=1)

        slider_label = ttk.Label(slider_frame, text="Selected vertex: 0")
        slider_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=len(transformed_vertices) - 1,
            orient='horizontal',
            variable=selected_index,
            command=update_highlight,
            length=350
        )
        slider.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # --- Focus Point Info Frame ---
        info_frame = ttk.LabelFrame(main_frame, text="Focus Point Details", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)

        dot_product_label = ttk.Label(info_frame, text="Dot Product: N/A")
        dot_product_label.grid(row=0, column=0, sticky=tk.W)

        inside_mask_label = ttk.Label(info_frame, text="Inside Mask: N/A")
        inside_mask_label.grid(row=1, column=0, sticky=tk.W)

        normal_label = ttk.Label(info_frame, text="Closest Normal: N/A")
        normal_label.grid(row=2, column=0, sticky=tk.W)

        # --- Contact Info Frame ---
        contact_frame = ttk.LabelFrame(main_frame, text="Unique Contact Indices", padding="10")
        contact_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        contact_frame.columnconfigure(0, weight=1)
        contact_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Allow this row to expand

        contact_indices_text = tk.Text(contact_frame, height=8, wrap=tk.WORD)
        contact_indices_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(contact_frame, orient=tk.VERTICAL, command=contact_indices_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        contact_indices_text['yscrollcommand'] = scrollbar.set

        # Populate the text box and make it read-only
        indices_str = ", ".join(map(str, unique_contact_indices)) if unique_contact_indices.size > 0 else "None"
        contact_indices_text.insert(tk.END, indices_str)
        contact_indices_text.config(state=tk.DISABLED)

        # --- 4. Run Main Loop ---
        print("Opening debug visualizer. Close the window to continue execution.")

        update_highlight(0)

        try:
            # This loop runs both the Open3D event polling and the Tkinter event loop
            while vis.poll_events():
                root.update()
        finally:
            vis.destroy_window()
            if root.winfo_exists():
                root.destroy()

    def _create_normal_lines(self, normal_length: float, max_normals: int = 2000) -> o3d.geometry.LineSet:
        """Helper method to create a LineSet representing normals to avoid clutter."""
        num_points = len(self.ref_pcd.points)
        
        if num_points > max_normals:
            pcd_for_normals = self.ref_pcd.random_down_sample(sampling_ratio=max_normals / float(num_points))
        else:
            pcd_for_normals = self.ref_pcd

        points = np.asarray(pcd_for_normals.points)
        normals = np.asarray(pcd_for_normals.normals)
        
        line_set_points = np.vstack((points, points + normals * normal_length))
        line_indices = np.array([[i, i + len(points)] for i in range(len(points))])
        
        normal_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_set_points),
            lines=o3d.utility.Vector2iVector(line_indices),
        )
        normal_lines.paint_uniform_color([1.0, 0.8, 0.0])  # Yellow
        return normal_lines