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

    Skin-colour filtering uses a cyclic hue range (``hsv_h_range``) so that
    wrap-around ranges spanning the 0°/360° boundary — e.g. ``[330, 30]`` for
    pink/red hues — are fully supported without any special-casing by the caller.
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
            'hsv_h_range': [335, 25],         # [H_start, H_end] degrees, cyclic (0-360)
            'hsv_s_range': [0.1, 1.0],      # [S_low, S_high]
            'hsv_v_range': [0.0, 1.0],      # [V_low, V_high]
        },
        'region_growing': {
            'dbscan_eps': 18.0,
            'min_cluster_size': 50
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
            'hsv_h_range': {'is_hue_circle': True},
            'hsv_s_range': {
                'is_range': True,
                'cfgs': [
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S low'},
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S high'},
                ],
            },
            'hsv_v_range': {
                'is_range': True,
                'cfgs': [
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V low'},
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V high'},
                ],
            },
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
        # Warn callers that pass the old flat-bound keys
        if params:
            csf = params.get('color_skin_filter', {})
            if 'hsv_lower_bound' in csf or 'hsv_upper_bound' in csf:
                import warnings
                warnings.warn(
                    "ArmSegmentation: 'hsv_lower_bound' and 'hsv_upper_bound' were removed. "
                    "Use 'hsv_h_range', 'hsv_s_range', and 'hsv_v_range' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            deep_update(self.params, params)

        self.interactive = interactive
        # Store original colors for clustering visualization
        self._original_colors = None
        # Set to True if the operator clicked "Process" at least once during an
        # interactive session; remains False if the window was closed without
        # applying any changes (so the caller can skip re-saving unchanged outputs).
        self.was_modified: bool = False

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
        """Applies an HSV-based color filter to isolate skin tones.

        Reads hsv_h_range, hsv_s_range, and hsv_v_range from params.
        Hue is handled cyclically so wrap-around ranges (e.g. [330, 30]) work correctly.
        """
        points_color_rgb = np.asarray(pcd.colors)
        if points_color_rgb.shape[0] == 0:
            return o3d.geometry.PointCloud()

        hsv = np.array([colorsys.rgb_to_hsv(c[0], c[1], c[2]) for c in points_color_rgb])
        hsv[:, 0] *= 360  # scale hue to 0-360 degrees

        h_start, h_end = params['hsv_h_range']
        s_lo,   s_hi   = params['hsv_s_range']
        v_lo,   v_hi   = params['hsv_v_range']

        mask = self._hue_in_range(hsv[:, 0], h_start, h_end) & \
               (hsv[:, 1] >= s_lo) & (hsv[:, 1] <= s_hi) & \
               (hsv[:, 2] >= v_lo) & (hsv[:, 2] <= v_hi)

        return pcd.select_by_index(np.where(mask)[0])

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hue_in_range(H: np.ndarray, h_start: float, h_end: float) -> np.ndarray:
        """Returns a boolean mask: True where H (degrees, 0–360) falls within the
        clockwise arc from h_start to h_end.  Handles wrap-around correctly, so
        _hue_in_range(H, 330, 30) selects red hues that span the 0°/360° boundary."""
        if h_start <= h_end:
            return (H >= h_start) & (H <= h_end)
        else:  # wrap-around (e.g. 330°–30° spans the red/pink end)
            return (H >= h_start) | (H <= h_end)

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
    def _make_range_slider_row(
            label_text: str,
            lo_cfg: dict,
            hi_cfg: dict,
            lo_init,
            hi_init,
            em: float,
    ) -> Tuple['gui.Widget', Callable[[], list]]:
        """Creates a coupled lo/hi range control enforcing lo ≤ hi.

        Builds a vertical group labelled *label_text* containing two paired-slider
        rows.  Dragging lo above hi clamps hi upward; dragging hi below lo clamps
        lo downward.  Direct text entry is similarly clamped.

        Returns ``(container, get_range_fn)`` where
        ``get_range_fn() -> [lo_value, hi_value]``.
        """
        type_str = lo_cfg.get('type', 'float')
        lo_slider = ArmSegmentation._make_slider(lo_cfg, lo_init)
        lo_text   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        lo_text.double_value = float(lo_init)
        lo_text.set_preferred_width(4 * em)

        hi_slider = ArmSegmentation._make_slider(hi_cfg, hi_init)
        hi_text   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        hi_text.double_value = float(hi_init)
        hi_text.set_preferred_width(4 * em)

        syncing = [False]

        def _get_lo_raw():
            return lo_slider.int_value if type_str == 'int' else lo_slider.double_value

        def _get_hi_raw():
            return hi_slider.int_value if type_str == 'int' else hi_slider.double_value

        def _set_lo(v):
            if type_str == 'int':
                lo_slider.int_value = int(round(v))
            else:
                lo_slider.double_value = v
            lo_text.double_value = float(v)

        def _set_hi(v):
            if type_str == 'int':
                hi_slider.int_value = int(round(v))
            else:
                hi_slider.double_value = v
            hi_text.double_value = float(v)

        def _on_lo_slider(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            lo_text.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
            if new_v > _get_hi_raw():
                _set_hi(new_v)
            syncing[0] = False

        def _on_hi_slider(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            hi_text.double_value = float(int(round(new_v)) if type_str == 'int' else new_v)
            if new_v < _get_lo_raw():
                _set_lo(new_v)
            syncing[0] = False

        def _on_lo_text(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            clamped = max(lo_cfg['min'], min(lo_cfg['max'], new_v))
            _set_lo(clamped)
            if clamped > _get_hi_raw():
                _set_hi(clamped)
            syncing[0] = False

        def _on_hi_text(new_v):
            if syncing[0]:
                return
            syncing[0] = True
            clamped = max(hi_cfg['min'], min(hi_cfg['max'], new_v))
            _set_hi(clamped)
            if clamped < _get_lo_raw():
                _set_lo(clamped)
            syncing[0] = False

        lo_slider.set_on_value_changed(_on_lo_slider)
        hi_slider.set_on_value_changed(_on_hi_slider)
        lo_text.set_on_value_changed(_on_lo_text)
        hi_text.set_on_value_changed(_on_hi_text)

        lo_row = gui.Horiz(0.25 * em)
        lo_row.add_child(gui.Label(f"  {lo_cfg.get('label', 'lo')}:"))
        lo_row.add_child(lo_slider)
        lo_row.add_child(lo_text)

        hi_row = gui.Horiz(0.25 * em)
        hi_row.add_child(gui.Label(f"  {hi_cfg.get('label', 'hi')}:"))
        hi_row.add_child(hi_slider)
        hi_row.add_child(hi_text)

        container = gui.Vert(0.25 * em)
        container.add_child(gui.Label(label_text))
        container.add_child(lo_row)
        container.add_child(hi_row)

        def get_range() -> list:
            lo = lo_slider.int_value if type_str == 'int' else lo_slider.double_value
            hi = hi_slider.int_value if type_str == 'int' else hi_slider.double_value
            return [lo, hi]

        return container, get_range

    @staticmethod
    def _render_hue_wheel(size_px: int) -> np.ndarray:
        """Renders a full hue wheel as an RGB uint8 image of *size_px* × *size_px* pixels.

        Layout:
        - Ring occupying radii [32 %, 48 %] × size_px, coloured HSV(h, 1, 1).
        - Inner disc (r < 32 %) in dark grey — provides contrast for the handles.
        - Exterior corners (r > 48 %) in black.

        Hue 0° (red) is placed at the top (12 o'clock), increasing clockwise.
        """
        cx = cy = size_px / 2.0
        ring_outer = size_px * 0.48
        ring_inner = size_px * 0.32

        ys, xs = np.mgrid[0:size_px, 0:size_px]
        dx = (xs + 0.5) - cx
        dy = (ys + 0.5) - cy
        r = np.sqrt(dx * dx + dy * dy)

        # Hue angle: atan2 with -dy gives standard-math angle (CCW, 0 = right).
        # Subtracting from 90° flips to CW with 0° at top.
        theta_deg = np.degrees(np.arctan2(-dy, dx))
        hue_deg   = (90.0 - theta_deg) % 360.0
        hue_norm  = hue_deg / 360.0

        # Vectorised HSV(h, 1, 1) → RGB.
        h6 = hue_norm * 6.0
        i  = h6.astype(int) % 6
        f  = h6 - np.floor(h6)
        one  = np.ones_like(f)
        zero = np.zeros_like(f)
        R = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [one,     1 - f,  zero,   zero,   f,      one])
        G = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [f,      one,    one,    1 - f,  zero,   zero])
        B = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [zero,   zero,   f,      one,    one,    1 - f])

        ring_mask  = (r >= ring_inner) & (r <= ring_outer)
        inner_mask = r < ring_inner

        img = np.zeros((size_px, size_px, 3), dtype=np.uint8)
        img[ring_mask,  0] = (R[ring_mask]  * 255).astype(np.uint8)
        img[ring_mask,  1] = (G[ring_mask]  * 255).astype(np.uint8)
        img[ring_mask,  2] = (B[ring_mask]  * 255).astype(np.uint8)
        img[inner_mask]    = 45  # dark grey inner disc
        return img

    @staticmethod
    def _render_hue_arc_overlay(
            base_img: np.ndarray,
            h_start: float,
            h_end: float,
            handle_radius: int = 8,
    ) -> np.ndarray:
        """Returns a copy of *base_img* (RGB uint8) with the selected arc at full
        saturation and the rest of the ring darkened, plus two circular drag
        handles drawn on the mid-ring track.

        The selected arc is the clockwise arc from *h_start* to *h_end*; wrap-around
        ranges (e.g. 330°–30°) are fully supported.
        h_start handle → white fill; h_end handle → light-grey fill.
        """
        img = base_img.copy()
        size_px = img.shape[0]
        cx = cy = size_px / 2.0
        ring_outer    = size_px * 0.48
        ring_inner    = size_px * 0.32
        handle_track_r = (ring_inner + ring_outer) / 2.0

        ys, xs = np.mgrid[0:size_px, 0:size_px]
        dx = (xs + 0.5) - cx
        dy = (ys + 0.5) - cy
        r  = np.sqrt(dx * dx + dy * dy)

        theta_deg = np.degrees(np.arctan2(-dy, dx))
        hue_deg   = (90.0 - theta_deg) % 360.0
        ring_mask = (r >= ring_inner) & (r <= ring_outer)

        if h_start <= h_end:
            in_arc = ring_mask & (hue_deg >= h_start) & (hue_deg <= h_end)
        else:  # wrap-around
            in_arc = ring_mask & ((hue_deg >= h_start) | (hue_deg <= h_end))

        # Darken out-of-arc ring pixels; the selected arc keeps its full HSV colours.
        out_arc = ring_mask & ~in_arc
        img[out_arc] = (img[out_arc].astype(np.float32) * 0.25).astype(np.uint8)

        # Draw handles (h_start = white, h_end = light grey), both with black border.
        for h_angle, fill_col in ((h_start, (255, 255, 255)), (h_end, (200, 200, 200))):
            theta_rad = np.radians(90.0 - h_angle)
            hx = cx + handle_track_r * np.cos(theta_rad)
            hy = cy - handle_track_r * np.sin(theta_rad)
            dist_sq     = (xs + 0.5 - hx) ** 2 + (ys + 0.5 - hy) ** 2
            fill_mask   = dist_sq <= handle_radius ** 2
            border_mask = (dist_sq > handle_radius ** 2) & \
                          (dist_sq <= (handle_radius + 1.5) ** 2)
            img[fill_mask]   = fill_col
            img[border_mask] = (0, 0, 0)

        return img

    @staticmethod
    def _make_hue_range_circle(
            h_start_init: float,
            h_end_init: float,
            em: float,
            renderer: 'rendering.Renderer',
            size_px: int = 200,
    ) -> Tuple[Dict[str, 'gui.Widget'], Callable[[], list]]:
        """Creates a circular hue-range selector widget.

        Displays a hue wheel with two drag handles — one for *h_start* (white)
        and one for *h_end* (grey) — defining a clockwise arc.  Wrap-around
        ranges (e.g. 330°–30°) are fully supported.  Two companion
        ``NumberEdit`` boxes provide a precise text-entry alternative.

        Implementation note: the wheel is rendered as a 2-D background image on
        a ``SceneWidget`` (empty 3-D scene).  ``ImageWidget`` was not used
        because it never delivers ``DRAG`` events in Open3D 0.19.
        ``SceneWidget`` does deliver them; the 3-D camera is locked out by
        returning ``HANDLED`` for every mouse event so it never receives input.

        Returns ``(fragments_dict, get_hue_range_fn)`` where fragments_dict contains
        the independent UI elements ("top", "scene", "bottom") so the caller can place
        them in a split-panel layout, avoiding Open3D's nested auto-layout bugs.
        ``fragments_dict["refresh"]`` is the ``_refresh`` closure; call it after any
        layout change that resizes ``scene`` to re-apply the correct hue overlay.
        """
        state = {
            'h_start':  float(h_start_init),
            'h_end':    float(h_end_init),
            'dragging': None,   # None | 'start' | 'end'
        }

        base_wheel   = ArmSegmentation._render_hue_wheel(size_px)
        ring_outer_r = size_px * 0.48
        ring_inner_r = size_px * 0.32
        handle_r     = max(4, int(size_px * 0.040))
        handle_tr    = (ring_inner_r + ring_outer_r) / 2.0
        cx = cy      = size_px / 2.0

        def _to_bg_image(rgb_arr: np.ndarray) -> 'o3d.geometry.Image':
            rgba = np.dstack([rgb_arr, np.full(rgb_arr.shape[:2], 255, dtype=np.uint8)])
            return o3d.geometry.Image(np.ascontiguousarray(rgba))

        def _fresh_overlay() -> np.ndarray:
            return ArmSegmentation._render_hue_arc_overlay(
                base_wheel, state['h_start'], state['h_end'],
                handle_radius=handle_r,
            )

        # SceneWidget with an empty 3-D scene — display is via set_background,
        # which renders a flat 2-D quad unaffected by camera position.
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(renderer)
        scene_widget.enable_scene_caching(False)
        scene_widget.scene.set_background([0, 0, 0, 1], _to_bg_image(_fresh_overlay()))

        # --- Companion NumberEdit boxes ---
        start_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        start_edit.double_value = float(h_start_init)
        end_edit   = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        end_edit.double_value   = float(h_end_init)

        syncing = [False]

        def _refresh() -> None:
            scene_widget.scene.set_background([0, 0, 0, 1], _to_bg_image(_fresh_overlay()))

        def _on_start_edit(v: float):
            if syncing[0]:
                return
            syncing[0] = True
            state['h_start'] = float(max(0.0, min(360.0, v)))
            start_edit.double_value = state['h_start']
            _refresh()
            syncing[0] = False

        def _on_end_edit(v: float):
            if syncing[0]:
                return
            syncing[0] = True
            state['h_end'] = float(max(0.0, min(360.0, v)))
            end_edit.double_value = state['h_end']
            _refresh()
            syncing[0] = False

        start_edit.set_on_value_changed(_on_start_edit)
        end_edit.set_on_value_changed(_on_end_edit)

        # --- Mouse handler ---
        def _handle_image_pos(h_angle: float):
            """Return (hx, hy) in image-space pixels for hue angle h_angle."""
            theta_rad = np.radians(90.0 - h_angle)
            return (cx + handle_tr * np.cos(theta_rad),
                    cy - handle_tr * np.sin(theta_rad))

        def _on_mouse(event):
            fr = scene_widget.frame
            px = event.x - fr.x
            py = event.y - fr.y
            fw, fh = fr.width, fr.height

            if event.type == gui.MouseEvent.BUTTON_DOWN:
                if fw > 0 and fh > 0:
                    ix = px / fw * size_px
                    iy = py / fh * size_px
                    if 0 <= ix <= size_px and 0 <= iy <= size_px:
                        hx_s, hy_s = _handle_image_pos(state['h_start'])
                        hx_e, hy_e = _handle_image_pos(state['h_end'])
                        d_start = (ix - hx_s) ** 2 + (iy - hy_s) ** 2
                        d_end   = (ix - hx_e) ** 2 + (iy - hy_e) ** 2
                        grab_r2 = (handle_r * 2.5) ** 2
                        if d_start <= grab_r2 or d_end <= grab_r2:
                            state['dragging'] = 'start' if d_start <= d_end else 'end'
                # Re-apply background: clicking on a SceneWidget triggers an
                # internal scene reset that clears the set_background image.
                _refresh()
                # Always HANDLED — the 3-D camera must never receive mouse input.
                return gui.Widget.EventCallbackResult.HANDLED

            if event.type in (gui.MouseEvent.DRAG, gui.MouseEvent.MOVE):
                if state['dragging'] is not None and fw > 0 and fh > 0:
                    ix = px / fw * size_px
                    iy = py / fh * size_px
                    ddx = ix - cx
                    ddy = iy - cy
                    if ddx * ddx + ddy * ddy >= 1.0:
                        hue = (90.0 - np.degrees(np.arctan2(-ddy, ddx))) % 360.0
                        if state['dragging'] == 'start':
                            state['h_start'] = hue
                            syncing[0] = True
                            start_edit.double_value = round(hue, 1)
                            syncing[0] = False
                        else:
                            state['h_end'] = hue
                            syncing[0] = True
                            end_edit.double_value = round(hue, 1)
                            syncing[0] = False
                        _refresh()
                return gui.Widget.EventCallbackResult.HANDLED

            if event.type == gui.MouseEvent.BUTTON_UP:
                state['dragging'] = None
                _refresh()
                return gui.Widget.EventCallbackResult.HANDLED

            # Catch scroll, right-drag, etc. — camera must not respond.
            return gui.Widget.EventCallbackResult.HANDLED

        scene_widget.set_on_mouse(_on_mouse)

        # --- Layout ---
        edit_row = gui.Horiz(0.25 * em)
        edit_row.add_child(gui.Label(" H start:"))
        edit_row.add_child(start_edit)
        edit_row.add_child(gui.Label(" end:"))
        edit_row.add_child(end_edit)

        title_lbl = gui.Label("H range (drag handles, wrap-around supported):")

        def get_hue_range() -> list:
            return [state['h_start'], state['h_end']]

        return {
            "top": title_lbl,
            "scene": scene_widget,
            "bottom": edit_row,
            "refresh": _refresh,
        }, get_hue_range

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
            # Batch path: if a processing function was provided, run it silently
            # using the saved params. No window, no blocking. Mirror the
            # interactive path's cluster-step unpacking so callers get the
            # selected cluster (not the all-clusters preview).
            if processing_func is not None and params_key is not None:
                result = processing_func(pcd_input, self.params[params_key])
                if is_cluster_step:
                    _all_clusters, arm_pcd = result
                    return arm_pcd
                return result
            # Display-only call — caller has already gated on `show`.
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

        # --- GUI Controls Layout (Strategy C: Split Panel) ---
        em = w.theme.font_size
        panel_top = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.0))
        panel_bottom = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.0, 0.5 * em, 0.5 * em))
        hue_scene_widget = None

        current_panel = panel_top
        
        # Add the split panels directly to the window (hue_scene_widget added conditionally later)
        w.add_child(panel_top)
        w.add_child(panel_bottom)

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
                    current_panel.add_child(row)
                    widgets[key] = (None, get_value)
                else:
                    # --- Fallback: NumberEdit ---
                    row = gui.Horiz(0.25 * em)
                    row.add_child(gui.Label(f"{key}:"))
                    row.add_stretch()
                    widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                    widget.double_value = value
                    row.add_child(widget)
                    current_panel.add_child(row)
                    widgets[key] = (widget, lambda w=widget: w.double_value)

            elif isinstance(value, list) and all(isinstance(i, (float, int)) for i in value):
                # --- Circular hue range selector ---
                if (key_cfg is not None and isinstance(key_cfg, dict)
                        and key_cfg.get('is_hue_circle') and len(value) == 2):
                    hue_elements, get_range = self._make_hue_range_circle(
                        value[0], value[1], em=em, renderer=w.renderer
                    )
                    
                    panel_top.add_child(hue_elements["top"])
                    
                    hue_scene_widget = hue_elements["scene"]
                    hue_refresh_fn = hue_elements["refresh"]
                    w.add_child(hue_scene_widget) # Added directly to window

                    # Redirect any future components into the bottom panel
                    current_panel = panel_bottom
                    current_panel.add_child(hue_elements["bottom"])

                    widgets[key] = (None, get_range)

                # --- Coupled range slider (lo ≤ hi enforced) ---
                elif (key_cfg is not None and isinstance(key_cfg, dict)
                        and key_cfg.get('is_range') and len(value) == 2):
                    cfgs = key_cfg['cfgs']
                    container, get_range = self._make_range_slider_row(
                        f"{key}:", cfgs[0], cfgs[1], value[0], value[1], em
                    )
                    current_panel.add_child(container)
                    widgets[key] = (None, get_range)
                else:
                    # --- Individual sliders (or fallback NumberEdits) per element ---
                    current_panel.add_child(gui.Label(f"{key}:"))
                    widgets[key] = []

                    for i, v in enumerate(value):
                        if key_cfg is not None and isinstance(key_cfg, list) and i < len(key_cfg):
                            item_cfg = key_cfg[i]
                            component_label = item_cfg.get('label', str(i))
                            row, get_value = _make_paired_row(f"  {component_label}:", item_cfg, v)
                            current_panel.add_child(row)
                            widgets[key].append((None, get_value))
                        else:
                            # --- Fallback: NumberEdit ---
                            row = gui.Horiz(0.25 * em)
                            row.add_child(gui.Label(f"  [{i}]"))
                            row.add_stretch()
                            widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                            widget.double_value = v
                            row.add_child(widget)
                            current_panel.add_child(row)
                            widgets[key].append((widget, lambda w=widget: w.double_value))

        # --- HSV hover readout label (updated by on_hover) ---
        hsv_label = gui.Label("HSV: —")
        current_panel.add_child(hsv_label)

        # Tracks whether the initial programmatic on_process() call has completed.
        # Only calls triggered after that (button click, space bar) count as
        # user edits and should set self.was_modified.
        _gui_initialized = [False]

        def on_process():
            """Callback to update parameters and re-run processing."""
            # Mark that the operator applied at least one change in this session,
            # but only for user-triggered calls (not the initial programmatic run).
            if _gui_initialized[0]:
                self.was_modified = True

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
            h_deg = h * 360.0
            if params_key == 'color_skin_filter':
                h_start, h_end = self.params['color_skin_filter']['hsv_h_range']
                in_h = bool(self._hue_in_range(np.array([h_deg]), h_start, h_end)[0])
                indicator = "\u2713" if in_h else "\u2717"
                hsv_label.text = f"H: {h_deg:.1f}\u00b0 ({indicator})  S: {s:.3f}  V: {v:.3f}"
            else:
                hsv_label.text = f"H: {h_deg:.1f}\u00b0  S: {s:.3f}  V: {v:.3f}"

            return gui.Widget.EventCallbackResult.IGNORED

        scene.set_on_mouse(on_hover)

        # --- Add Buttons ---
        process_button = gui.Button("Process")
        process_button.set_on_clicked(on_process)
        current_panel.add_child(process_button)

        continue_button = gui.Button("Continue")
        continue_button.set_on_clicked(gui.Application.instance.quit)
        current_panel.add_child(continue_button)

        # --- Space-bar shortcut → Process ---
        def on_key(key_event):
            if (key_event.key == gui.KeyName.SPACE
                    and key_event.type == gui.KeyEvent.DOWN):
                on_process()
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED

        scene.set_on_key(on_key)

        # --- Set window layout and run ---
        def on_layout(layout_context):
            r = w.content_rect
            panel_w = max(1, r.width // 5)
            scene_w = r.width - panel_w
            px = r.get_right() - panel_w
            
            scene.frame = gui.Rect(r.x, r.y, scene_w, r.height)

            if hue_scene_widget is not None:
                # 1. Top Panel
                try:
                    top_pref = panel_top.calc_preferred_size(layout_context, gui.Widget.Constraints())
                    top_h = top_pref.height
                except Exception:
                    top_h = int(3.5 * em)
                    
                panel_top.frame = gui.Rect(px, r.y, panel_w, top_h)

                # 2. Hue Circle (Perfect Square)
                avail_h = max(10, r.height - top_h)
                bottom_min_h = int(10 * em) # Reserve space for bottom sliders/buttons
                max_sq = max(10, avail_h - bottom_min_h)
                
                sq = min(panel_w, max_sq)
                x_offset = px + (panel_w - sq) // 2
                
                hue_scene_widget.frame = gui.Rect(x_offset, r.y + top_h, sq, sq)

                # 3. Bottom Panel
                bottom_y = r.y + top_h + sq
                panel_bottom.frame = gui.Rect(px, bottom_y, panel_w, max(1, r.height - bottom_y))

                hue_refresh_fn()
            else:
                # Fallback to standard layout if no hue circle is rendered for this step
                try:
                    top_pref = panel_top.calc_preferred_size(layout_context, gui.Widget.Constraints())
                    top_h = top_pref.height
                except Exception:
                    top_h = r.height // 2
                
                panel_top.frame = gui.Rect(px, r.y, panel_w, r.height)
                # Bottom panel stays at 0-size so it doesn't conflict
                panel_bottom.frame = gui.Rect(px, r.y, 0, 0)

        w.set_on_layout(on_layout)

        on_process()  # Initial run — does NOT set was_modified
        _gui_initialized[0] = True  # Subsequent on_process calls are user-triggered

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
            # Widen the hue range slightly (H 0°–35°, full S/V range)
            'hsv_h_range': [0, 35],
        },
        'region_growing': { # Changed from 'clustering' to match internal key
            'dbscan_eps': 6.0,
            'min_cluster_size': 50
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