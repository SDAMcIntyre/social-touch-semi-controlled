import numpy as np
import open3d as o3d
import colorsys
from typing import Dict, Callable, Tuple, Any

# Add these imports at the top of your file
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Matplotlib is used for generating distinct colors for clusters
import matplotlib.pyplot as plt
import copy
import collections.abc


def deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Recursively updates a dictionary.
    Sub-dictionaries are updated instead of being replaced.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and key in source:
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source


class ArmSegmentation:
    """
    A class providing stateful methods for arm segmentation using Open3D and NumPy.
    It processes 3D points and their corresponding colors to isolate the largest cluster,
    assumed to be the user's arm.

    The processing parameters are provided during initialization. An optional interactive
    mode allows for real-time parameter tuning and visualization at each step.
    """
    # --- Default parameters for all processing steps ---
    _DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'down_sampling': {
            'enabled': True,
            'leaf_size': 5.0
        },
        'box_filter': {
            'enabled': True,
            'min_z': -np.inf,
            'max_z': +np.inf,
        },
        'color_skin_filter': {
            'enabled': True,
            'hsv_lower_bound': [0, 0, 0],   # H: 0-360, S: 0-1, V: 0-1
            'hsv_upper_bound': [25, 1.0, 1.0]
        },
        'region_growing': {
            'dbscan_eps': 10.0,
            'min_cluster_size': 100
        }
    }

    # --- Slider configuration: min, max, type ('int' or 'float'), and optional
    #     per-component labels for list parameters.
    #
    #     Ranges are derived from the physical scale of the data:
    #       H  : 0–360°  (full hue circle; skin ≈ 0–50°)
    #       S,V: 0–1.0   (normalised saturation / brightness)
    #       dbscan_eps      : 1–100 mm  (point-cloud units; arm spans ~100 mm)
    #       min_cluster_size: 10–2000   (points after voxel downsampling)
    #       leaf_size       : 0.5–20 mm (voxel size)
    # ---
    _SLIDER_CONFIGS: Dict[str, Dict] = {
        'down_sampling': {
            'leaf_size': {'min': 0.5, 'max': 20.0, 'type': 'float'},
        },
        'color_skin_filter': {
            'hsv_lower_bound': [
                {'min': 0,   'max': 360, 'type': 'int',   'label': 'H'},
                {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S'},
                {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V'},
            ],
            'hsv_upper_bound': [
                {'min': 0,   'max': 360, 'type': 'int',   'label': 'H'},
                {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S'},
                {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V'},
            ],
        },
        'region_growing': {
            'dbscan_eps':       {'min': 1.0,  'max': 100.0, 'type': 'float'},
            'min_cluster_size': {'min': 10,   'max': 2000,  'type': 'int'},
        },
    }

    def __init__(self, params: Dict = None, interactive: bool = True):
        """
        Initializes the ArmSegmentation instance.

        Args:
            params (Dict): A dictionary containing parameters to override the defaults.
            interactive (bool): If True, enables a GUI for real-time parameter adjustment.
        """
        # Start with a deep copy of the default parameters
        self.params = copy.deepcopy(self._DEFAULT_PARAMS)
        # Recursively update with user-provided parameters
        if params:
            deep_update(self.params, params)

        self.interactive = interactive
        # Store original colors for clustering visualization
        self._original_colors = None

    def preprocess(self,
                   pcd: o3d.geometry.PointCloud,
                   box_corners: np.ndarray,
                   show: bool = False) -> o3d.geometry.PointCloud:
        """
        Applies a series of filters to the input point cloud data.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud with points and colors.
            box_corners (np.ndarray): A (2, 2) array [[min_x, min_y], [max_x, max_y]] for cropping.
            show (bool): If True (and not in interactive mode), visualizes the point cloud at each step.

        Returns:
            o3d.geometry.PointCloud: The processed point cloud.
        """
        if show and not self.interactive: self._display_pointcloud(pcd, "1. Input Point Cloud")

        # --- 1. Downsampling ---
        if self.params['down_sampling']['enabled']:
            pcd = self._display_pointcloud(
                pcd, "Step 1: Voxel Downsampling",
                params_key='down_sampling',
                processing_func=lambda p, pa: p.voxel_down_sample(voxel_size=pa['leaf_size'])
            )
            print(f"Downsampled cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "2. After Downsampling")

        # --- 2. Box Filter ---
        if self.params['box_filter']['enabled']:
            min_b = np.array([box_corners[0, 0], box_corners[0, 1], self.params['box_filter']['min_z']])
            max_b = np.array([box_corners[1, 0], box_corners[1, 1], self.params['box_filter']['max_z']])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
            pcd = pcd.crop(bbox)
            print(f"Box-filtered cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "3. After Box Filter")

        # --- 3. Skin Color Filter ---
        if self.params['color_skin_filter']['enabled'] and len(pcd.points) > 0:
            pcd = self._display_pointcloud(
                pcd, "Step 2: Skin Color Filter (HSV)",
                params_key='color_skin_filter',
                processing_func=self._apply_skin_color_filter
            )
            print(f"Color-filtered cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "4. After Skin Color Filter")

        return pcd

    def extract_arm(self,
                    pcd: o3d.geometry.PointCloud,
                    show: bool = False) -> o3d.geometry.PointCloud:
        """
        Extracts the largest cluster (assumed to be the arm) via DBSCAN clustering.

        Args:
            pcd (o3d.geometry.PointCloud): The preprocessed point cloud.
            show (bool): If True (and not in interactive mode), visualizes the clustering results.

        Returns:
            o3d.geometry.PointCloud: A point cloud containing only the largest cluster.
        """
        if len(pcd.points) < self.params['region_growing']['min_cluster_size']:
            print("WARNING: Not enough points to process for arm extraction.")
            return o3d.geometry.PointCloud()

        self._original_colors = np.asarray(pcd.colors)

        arm_pcd = self._display_pointcloud(
            pcd, "Step 3: DBSCAN Clustering",
            params_key='region_growing',
            processing_func=self._apply_clustering,
            is_cluster_step=True
        )

        if arm_pcd and len(arm_pcd.points) > 0:
            print(f"Extracted arm with {len(arm_pcd.points)} points.")
            if show and not self.interactive:
                self._display_pointcloud(arm_pcd, "6. Segmented Arm")

        return arm_pcd

    def _apply_clustering(self, pcd: o3d.geometry.PointCloud, params: Dict) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Helper function to perform DBSCAN and extract the largest cluster."""
        labels = np.array(pcd.cluster_dbscan(
            eps=params['dbscan_eps'],
            min_points=int(params['min_cluster_size']),
            print_progress=False
        ))

        pcd_all_clusters = o3d.geometry.PointCloud(pcd)
        # Visualization for all clusters
        max_label = labels.max()
        if max_label >= 0:
            # Use a perceptually uniform colormap
            colors = plt.get_cmap("viridis")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0  # noise points are black
            pcd_all_clusters.colors = o3d.utility.Vector3dVector(colors[:, :3])

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:
            print("No clusters found.")
            return pcd_all_clusters, o3d.geometry.PointCloud()

        largest_cluster_label = unique_labels[counts.argmax()]
        mask = (labels == largest_cluster_label)

        arm_pcd = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        # Restore original colors for the final output
        if self._original_colors is not None and self._original_colors.shape[0] == np.asarray(pcd.points).shape[0]:
            arm_pcd.colors = o3d.utility.Vector3dVector(self._original_colors[mask])

        return pcd_all_clusters, arm_pcd

    def _apply_skin_color_filter(self, pcd: o3d.geometry.PointCloud, params: Dict) -> o3d.geometry.PointCloud:
        """Applies an HSV-based color filter to isolate skin tones."""
        points_color_rgb = np.asarray(pcd.colors)
        if points_color_rgb.shape[0] == 0:
            return o3d.geometry.PointCloud()

        hsv = np.array([colorsys.rgb_to_hsv(c[0], c[1], c[2]) for c in points_color_rgb])

        # scale hue to 0-360 for user-friendliness, but internal logic often uses 0-179 or 0-255
        hsv[:, 0] *= 360

        lb, ub = params['hsv_lower_bound'], params['hsv_upper_bound']
        mask = (hsv[:, 0] >= lb[0]) & (hsv[:, 0] <= ub[0]) & \
               (hsv[:, 1] >= lb[1]) & (hsv[:, 1] <= ub[1]) & \
               (hsv[:, 2] >= lb[2]) & (hsv[:, 2] <= ub[2])

        return pcd.select_by_index(np.where(mask)[0])

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_slider(cfg: dict, initial_value) -> 'gui.Slider':
        """Creates a gui.Slider from a slider-config dict and an initial value."""
        slider_type = gui.Slider.INT if cfg.get('type') == 'int' else gui.Slider.DOUBLE
        s = gui.Slider(slider_type)
        s.set_limits(cfg['min'], cfg['max'])
        if slider_type == gui.Slider.INT:
            s.int_value = int(round(initial_value))
        else:
            s.double_value = float(initial_value)
        return s

    @staticmethod
    def _get_screen_size() -> Tuple[int, int]:
        """Returns (width, height) of the primary screen, falling back to 1920×1080."""
        try:
            import tkinter as _tk
            _r = _tk.Tk()
            _r.withdraw()
            w, h = _r.winfo_screenwidth(), _r.winfo_screenheight()
            _r.destroy()
            return w, h
        except Exception:
            return 1920, 1080

    # ------------------------------------------------------------------
    # Display / interactive loop
    # ------------------------------------------------------------------

    def _display_pointcloud(self,
                            pcd_input: o3d.geometry.PointCloud,
                            window_name: str,
                            params_key: str = None,
                            processing_func: Callable = None,
                            is_cluster_step: bool = False) -> o3d.geometry.PointCloud:
        """
        The single method for displaying point clouds using the modern Open3D GUI framework.
        - If not interactive, it shows a static visualization.
        - If interactive, it launches a single window with the 3D scene and parameter controls.

        Parameter controls are rendered as sliders (with a companion value label) when a
        slider configuration exists for the parameter, and as plain NumberEdit widgets
        otherwise.  Sliders expose both a named component label (e.g. 'H', 'S', 'V' for
        HSV lists) and a live numeric readout that updates as the handle is dragged.
        """
        # --- Non-Interactive (or simple view) Path ---
        if not self.interactive or params_key is None or processing_func is None:
            if len(pcd_input.points) > 0:
                o3d.visualization.draw_geometries([pcd_input], window_name=window_name)
            else:
                print(f"Skipping visualization for '{window_name}': No points to show.")
            return pcd_input

        # --- Modern Interactive Path ---
        # Obtain screen dimensions before initialising the GUI toolkit so that
        # the window is created at full-screen size (maximised by default).
        screen_w, screen_h = self._get_screen_size()
        gui.Application.instance.initialize()

        w = gui.Application.instance.create_window(window_name, screen_w, screen_h)

        # Use a dictionary to hold state that needs to be modified by callbacks
        state = {
            'pcd_processed': o3d.geometry.PointCloud(),
            'camera_set':    False,   # True after the first setup_camera call
            'pcd_colors':    None,    # float RGB array matching the current cloud
            'pcd_points':    None,    # float XYZ array matching the current cloud
        }

        # --- 3D Scene Widget ---
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(w.renderer)
        scene.scene.set_background([0.1, 0.2, 0.3, 1.0]) # Dark background
        w.add_child(scene)

        # --- GUI Controls Layout ---
        em = w.theme.font_size
        layout = gui.Vert(0.25 * em, gui.Margins(0.5 * em))
        w.add_child(layout)

        # --- Retrieve slider config for this step (may be empty) ---
        step_params = self.params[params_key]
        slider_cfg = self._SLIDER_CONFIGS.get(params_key, {})

        # widgets maps: key -> (widget, get_value_fn)
        #                   or list of (widget, get_value_fn)   for list parameters
        widgets = {}

        def _make_paired_row(label_text: str, cfg: dict, initial_value):
            """
            Builds a horizontal row: [label] [slider] [textbox].
            The slider and textbox are kept in bidirectional sync so the user
            can either drag the handle or type a value directly.
            Returns (row, get_value_fn).
            """
            type_str = cfg.get('type', 'float')
            slider = self._make_slider(cfg, initial_value)

            textbox = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            textbox.double_value = float(int(round(initial_value)) if type_str == 'int' else initial_value)
            textbox.set_preferred_width(4 * em)

            # Guard prevents the two callbacks from triggering each other.
            syncing = [False]

            def _on_slider(new_v):
                if syncing[0]:
                    return
                syncing[0] = True
                textbox.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
                syncing[0] = False

            def _on_textbox(new_v):
                if syncing[0]:
                    return
                syncing[0] = True
                clamped = max(cfg['min'], min(cfg['max'], new_v))
                if type_str == 'int':
                    slider.int_value = int(round(clamped))
                else:
                    slider.double_value = clamped
                syncing[0] = False

            slider.set_on_value_changed(_on_slider)
            textbox.set_on_value_changed(_on_textbox)

            row = gui.Horiz(0.25 * em)
            row.add_child(gui.Label(label_text))
            row.add_child(slider)
            row.add_child(textbox)

            if type_str == 'int':
                get_value = lambda s=slider: s.int_value
            else:
                get_value = lambda s=slider: s.double_value

            return row, get_value

        for key, value in step_params.items():
            # Skip boolean flags — they are not tunable via numeric widgets.
            if isinstance(value, bool):
                continue

            key_cfg = slider_cfg.get(key)

            if isinstance(value, (float, int)):
                if key_cfg is not None:
                    # --- Slider ---
                    row, get_value = _make_paired_row(f"{key}:", key_cfg, value)
                    layout.add_child(row)
                    widgets[key] = (None, get_value)
                else:
                    # --- Fallback: NumberEdit ---
                    row = gui.Horiz(0.25 * em)
                    row.add_child(gui.Label(f"{key}:"))
                    row.add_stretch()
                    widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                    widget.double_value = value
                    row.add_child(widget)
                    layout.add_child(row)
                    widgets[key] = (widget, lambda w=widget: w.double_value)

            elif isinstance(value, list) and all(isinstance(i, (float, int)) for i in value):
                layout.add_child(gui.Label(f"{key}:"))
                widgets[key] = []

                for i, v in enumerate(value):
                    if key_cfg is not None and isinstance(key_cfg, list) and i < len(key_cfg):
                        item_cfg = key_cfg[i]
                        component_label = item_cfg.get('label', str(i))
                        row, get_value = _make_paired_row(f"  {component_label}:", item_cfg, v)
                        layout.add_child(row)
                        widgets[key].append((None, get_value))
                    else:
                        # --- Fallback: NumberEdit ---
                        row = gui.Horiz(0.25 * em)
                        row.add_child(gui.Label(f"  [{i}]"))
                        row.add_stretch()
                        widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                        widget.double_value = v
                        row.add_child(widget)
                        layout.add_child(row)
                        widgets[key].append((widget, lambda w=widget: w.double_value))

        # --- HSV hover readout label (updated by on_hover) ---
        hsv_label = gui.Label("HSV: —")
        layout.add_child(hsv_label)

        def on_process():
            """Callback to update parameters and re-run processing."""
            # 1. Update params from widgets
            for key, widget_or_list in widgets.items():
                if isinstance(widget_or_list, list):
                    self.params[params_key][key] = [get_v() for _, get_v in widget_or_list]
                else:
                    _, get_v = widget_or_list
                    self.params[params_key][key] = get_v()

            # 2. Rerun the processing function
            result = processing_func(pcd_input, self.params[params_key])

            # 3. Update the scene
            scene.scene.clear_geometry()
            material = rendering.MaterialRecord()

            if is_cluster_step:
                pcd_to_show, state['pcd_processed'] = result
            else:
                pcd_to_show = state['pcd_processed'] = result

            if len(pcd_to_show.points) > 0:
                scene.scene.add_geometry("processed_pcd", pcd_to_show, material)

                # Cache point data for the hover callback.
                state['pcd_colors'] = np.asarray(pcd_to_show.colors).copy()
                state['pcd_points'] = np.asarray(pcd_to_show.points).copy()

                # Only initialise the camera on the very first run; subsequent
                # runs preserve whatever orientation the user has navigated to.
                if not state['camera_set']:
                    scene.setup_camera(
                        60,
                        pcd_to_show.get_axis_aligned_bounding_box(),
                        pcd_to_show.get_center(),
                    )
                    state['camera_set'] = True
            else:
                state['pcd_colors'] = None
                state['pcd_points'] = None

        # --- HSV hover callback ---
        # Strategy: project every cloud point forward into screen space, then
        # find the 2D-nearest to the cursor.  This is simpler and correct:
        # the old "unproject cursor → 3D ray → KD-tree" approach always sampled
        # at centroid depth, so hovering over background gave the same result as
        # hovering over a point.  The screen-space approach naturally returns
        # "no match" when the cursor is over empty background.
        _HOVER_MAX_PX = 15   # screen-pixel radius that counts as "over a point"

        def on_hover(event):
            if event.type != gui.MouseEvent.MOVE:
                return gui.Widget.EventCallbackResult.IGNORED

            pcd_colors = state['pcd_colors']
            pcd_points = state['pcd_points']

            if pcd_colors is None or pcd_points is None or len(pcd_colors) == 0:
                return gui.Widget.EventCallbackResult.IGNORED

            # Pixel coordinates relative to the scene widget.
            mx = event.x - scene.frame.x
            my = event.y - scene.frame.y
            W  = scene.frame.width
            H  = scene.frame.height
            if W <= 0 or H <= 0 or mx < 0 or my < 0 or mx >= W or my >= H:
                return gui.Widget.EventCallbackResult.IGNORED

            # Forward-project all cloud points into screen space (vectorised).
            camera    = scene.scene.camera
            proj_view = np.array(camera.get_projection_matrix()) @ \
                        np.array(camera.get_view_matrix())          # 4×4

            pts_h = np.column_stack([pcd_points,
                                     np.ones(len(pcd_points))])     # N×4
            clip  = pts_h @ proj_view.T                             # N×4

            # Keep only points in front of the camera (positive w and z).
            visible = clip[:, 3] > 0
            if not np.any(visible):
                return gui.Widget.EventCallbackResult.IGNORED

            clip_v = clip[visible]
            ndc_x  =  clip_v[:, 0] / clip_v[:, 3]
            ndc_y  =  clip_v[:, 1] / clip_v[:, 3]

            sx = (ndc_x + 1.0) * 0.5 * W
            sy = (1.0 - ndc_y) * 0.5 * H

            # Nearest visible point to the cursor in screen space.
            dx, dy   = sx - mx, sy - my
            dists_sq = dx * dx + dy * dy
            best_local = int(np.argmin(dists_sq))

            if dists_sq[best_local] > _HOVER_MAX_PX ** 2:
                # Cursor is over background — clear the readout.
                hsv_label.text = "HSV: —"
                return gui.Widget.EventCallbackResult.IGNORED

            actual_idx = int(np.where(visible)[0][best_local])
            rgb = pcd_colors[actual_idx]
            h, s, v = colorsys.rgb_to_hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
            hsv_label.text = f"H: {h * 360:.1f}°  S: {s:.3f}  V: {v:.3f}"

            return gui.Widget.EventCallbackResult.IGNORED

        scene.set_on_mouse(on_hover)

        # --- Add Buttons ---
        process_button = gui.Button("Process")
        process_button.set_on_clicked(on_process)
        layout.add_child(process_button)

        continue_button = gui.Button("Continue")
        continue_button.set_on_clicked(gui.Application.instance.quit)
        layout.add_child(continue_button)

        # --- Space-bar shortcut → Process ---
        def on_key(key_event):
            if (key_event.key == gui.KeyName.SPACE
                    and key_event.type == gui.KeyEvent.DOWN):
                on_process()
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED

        scene.set_on_key(on_key)
        layout.set_on_key(on_key)   # also fires when a panel widget has focus

        # --- Set window layout and run ---
        # The right-hand panel always occupies exactly 1/5 of the window width.
        # The 3D scene fills the remaining 4/5.
        def on_layout(layout_context):
            r = w.content_rect
            panel_w = max(1, r.width // 5)
            scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
            layout.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)

        w.set_on_layout(on_layout)

        on_process()  # Initial run

        gui.Application.instance.run()

        return state['pcd_processed']


# Example usage:
if __name__ == '__main__':
    # Create a dummy point cloud for demonstration
    dummy_points = np.random.rand(50000, 3) * 100
    dummy_points[:, 2] *= 0.2 # Make it flatter
    # Add a denser "arm" cluster with skin-like color
    arm_points = np.random.rand(5000, 3) * 20 + np.array([40, 40, 5])
    skin_color_rgb = np.array([234, 192, 183]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([dummy_points, arm_points]))

    # Assign random colors and skin color to the arm
    colors = np.random.rand(50000, 3)
    arm_colors = np.tile(skin_color_rgb, (5000, 1)) + np.random.randn(5000, 3) * 0.05
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, arm_colors]))

    # --- Define parameters to OVERRIDE the defaults ---
    # You no longer need to define every single parameter.
    # For example, we can omit the 'box_filter' if the default is acceptable.
    user_defined_params = {
        'down_sampling': {
            'leaf_size': 2.5  # Override default leaf size
        },
        'color_skin_filter': {
            # Widen the hue range slightly
            'hsv_upper_bound': [35, 255, 255]
        },
        'region_growing': { # Changed from 'clustering' to match internal key
            'dbscan_eps': 6.0,
            'min_cluster_size': 150
        }
    }

    # Bounding box for filtering
    box_corners = np.array([[0, 0], [100, 100]])

    # --- Run in INTERACTIVE mode ---
    print("--- Starting INTERACTIVE segmentation ---")

    # The class now handles merging user params with defaults internally
    segmenter_interactive = ArmSegmentation(params=user_defined_params, interactive=True)

    # 1. Preprocessing
    preprocessed_pcd = segmenter_interactive.preprocess(pcd, box_corners)

    # 2. Arm Extraction
    if len(preprocessed_pcd.points) > 0:
        arm_pcd = segmenter_interactive.extract_arm(preprocessed_pcd)
        if arm_pcd and len(arm_pcd.points) > 0:
            print("\n✅ Interactive segmentation complete. Final arm point cloud:")
            segmenter_interactive._display_pointcloud(arm_pcd, "Final Result from Interactive Mode")
        else:
            print("\n❌ Interactive segmentation did not yield a result.")
    else:
        print("\n❌ Preprocessing removed all points. Cannot extract arm.")
