import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable

# Import provided modules
# Note: Ensure these modules are available in your PYTHONPATH
from preprocessing.motion_analysis import HandTrackingDataManager
from preprocessing.common import KinectMKV, KinectFrame

class HandMeshScalerApp:
    """
    Refactored High-Performance GUI for mesh calibration.
    
    Modifications:
    - Rendering: Converted Mesh to Red LineSet (Wireframe) for transparency.
    - Camera: Added State management for Camera Reset and Mesh Centering.
    - I/O: Added JSON export functionality for calibration parameters.
    """
    MESH_UNIT_CONVERSION_FACTOR = 1000.0
    
    def __init__(self, mesh_path: str, mkv_path: str):
        """
        Initialize Application Resources and State.
        """
        self.mesh_path = Path(mesh_path)
        self.mkv_path = Path(mkv_path)
        
        # Resources
        self.mesh_loader: Optional[HandTrackingDataManager] = None
        self.kinect_mkv: Optional[KinectMKV] = None
        
        # --- Active State (Currently Rendered) ---
        self.current_frame_idx = 0
        self.max_frames = 0
        self.scale_factor = 1.0
        self.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # --- Pending State (UI Values not yet applied) ---
        self._pending_scale = 1.0
        self._pending_translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.is_running = True
        
        # --- Final Output State ---
        self.final_scale_factor: Optional[float] = None 
        self.final_translation: Optional[np.ndarray] = None
        self.final_frame_idx: Optional[int] = None
        
        # Scene Objects
        self._mesh_geometry: Optional[o3d.geometry.TriangleMesh] = None
        self._pcd_geometry: Optional[o3d.geometry.PointCloud] = None
        
        # Material Setup: Wireframe (Lines)
        self._mesh_material = rendering.MaterialRecord()
        self._mesh_material.shader = "unlitLine" # Optimized for lines
        self._mesh_material.line_width = 2.0     # Thicker lines for visibility
        self._mesh_material.base_color = [1.0, 0.0, 0.0, 1.0] # RED
        
        self._pcd_material = rendering.MaterialRecord()
        self._pcd_material.shader = "defaultUnlit"
        
        self.SCENE_MESH = "target_mesh"
        self.SCENE_PCD = "reference_pcd"
        
        # Camera State
        self._initial_bounds = None # Stores the global scene bounds on startup

    def _init_ui(self):
        """Constructs the Open3D GUI Layout with separated logic panels."""
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window("3D Mesh Scale Calibration", 1280, 800)
        
        # 3D Scene Widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Main Layout container
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        
        # =================================================================================
        # SECTION 1: Frame Navigation (Reactive)
        # =================================================================================
        self.panel.add_child(gui.Label("Temporal Navigation"))
        
        # Frame Control Group
        self.frame_ctrl = self._create_input_group(
            label="Frame Index",
            min_val=0,
            max_val=max(1, self.max_frames - 1),
            init_val=self.current_frame_idx,
            is_int=True,
            callback=self._on_frame_ui_change
        )
        self.panel.add_child(self.frame_ctrl['layout'])
        
        self.panel.add_fixed(1.0 * em)
        
        # =================================================================================
        # SECTION 2: Spatial Transformation (Deferred)
        # =================================================================================
        self.panel.add_child(gui.Label("Spatial Transformation"))
        
        # Scale Control
        self.scale_ctrl = self._create_input_group(
            label="Scale Factor",
            min_val=0.1,
            max_val=5.0,
            init_val=self.scale_factor,
            is_int=False,
            callback=self._on_pending_scale_change
        )
        self.panel.add_child(self.scale_ctrl['layout'])
        
        self.panel.add_fixed(0.5 * em)
        
        # Translation Controls
        labels = ["Translation X (mm)", "Translation Y (mm)", "Translation Z (mm)"]
        self.trans_ctrls = []
        for i, lbl in enumerate(labels):
            ctrl = self._create_input_group(
                label=lbl,
                min_val=-1000.0,
                max_val=1000.0,
                init_val=self.translation[i],
                is_int=False,
                callback=lambda v, axis=i: self._on_pending_trans_change(axis, v)
            )
            self.panel.add_child(ctrl['layout'])
            self.trans_ctrls.append(ctrl)
        
        self.panel.add_fixed(1.0 * em)
        
        # =================================================================================
        # SECTION 3: Camera Controls
        # =================================================================================
        self.panel.add_child(gui.Label("Camera Controls"))
        
        # Horizontal layout for camera buttons
        cam_layout = gui.Horiz(0.5 * em)
        
        self.btn_recenter = gui.Button("Center on Mesh")
        self.btn_recenter.set_on_clicked(self._on_center_mesh)
        self.btn_recenter.tooltip = "Focus camera on the current mesh position"
        
        self.btn_reset_cam = gui.Button("Default View")
        self.btn_reset_cam.set_on_clicked(self._on_reset_camera)
        self.btn_reset_cam.tooltip = "Reset camera to initial global view"
        
        cam_layout.add_child(self.btn_recenter)
        cam_layout.add_child(self.btn_reset_cam)
        self.panel.add_child(cam_layout)

        self.panel.add_fixed(1.0 * em)
        
        # =================================================================================
        # SECTION 4: Actions
        # =================================================================================
        
        # Update Mesh Button
        self.btn_update = gui.Button("Process & Update Mesh")
        self.btn_update.set_on_clicked(self._on_click_process)
        self.btn_update.background_color = gui.Color(0.2, 0.4, 0.8)
        self.panel.add_child(self.btn_update)
        
        self.panel.add_fixed(1.0 * em)

        # Validate Button
        self.btn_validate = gui.Button("Validate & Close")
        self.btn_validate.set_on_clicked(self._on_validate)
        self.btn_validate.background_color = gui.Color(0.2, 0.6, 0.2)
        self.panel.add_child(self.btn_validate)

        # Layout Composition
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

    def _create_input_group(self, label: str, min_val: float, max_val: float, 
                            init_val: float, is_int: bool, callback: Callable) -> Dict[str, Any]:
        """Factory method for synchronized input widgets."""
        layout = gui.Vert()
        layout.add_child(gui.Label(label))
        
        h_layout = gui.Horiz(0.5 * self.window.theme.font_size)
        
        num_edit = gui.NumberEdit(gui.NumberEdit.INT) if is_int else gui.NumberEdit(gui.NumberEdit.DOUBLE)
        if is_int:
            num_edit.int_value = int(init_val)
        else:
            num_edit.double_value = float(init_val)
            num_edit.set_limits(min_val, max_val)
            
        slider = gui.Slider(gui.Slider.INT) if is_int else gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(min_val, max_val)
        if is_int:
            slider.int_value = int(init_val)
        else:
            slider.double_value = float(init_val)
            
        def on_slider_changed(new_val):
            if is_int:
                new_val = int(new_val)
                if num_edit.int_value != new_val:
                    num_edit.int_value = new_val
                    callback(new_val)
            else:
                if abs(num_edit.double_value - new_val) > 1e-5:
                    num_edit.double_value = new_val
                    callback(new_val)

        def on_edit_changed(new_val):
            if is_int:
                new_val = int(new_val)
                new_val = max(int(min_val), min(int(max_val), new_val))
                if slider.int_value != new_val:
                    slider.int_value = new_val
                    callback(new_val)
            else:
                new_val = max(min_val, min(max_val, new_val))
                if abs(slider.double_value - new_val) > 1e-5:
                    slider.double_value = new_val
                    callback(new_val)

        slider.set_on_value_changed(on_slider_changed)
        num_edit.set_on_value_changed(on_edit_changed)
        
        h_layout.add_child(num_edit)
        h_layout.add_child(slider)
        layout.add_child(h_layout)
        
        return {'layout': layout, 'slider': slider, 'num_edit': num_edit}

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 20 * layout_context.theme.font_size
        height = min(r.height, self.panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _load_frame_data(self, frame_idx: int):
        """Loads data for the specific frame."""
        if self.mesh_loader is not None and 0 <= frame_idx < len(self.mesh_loader):
            try:
                raw_data = self.mesh_loader[frame_idx]
                if 'hands' in raw_data and len(raw_data['hands']) > 0:
                    hand_data = raw_data['hands'][0]
                    faces = hand_data['faces']
                    # Retrieve vertices (Meters) and normalize Units: Convert Meters to Millimeters
                    vertices_m = hand_data['vertices_3d']
                    vertices_mm = np.asarray(vertices_m) * self.MESH_UNIT_CONVERSION_FACTOR

                    if self._mesh_geometry is None:
                        self._mesh_geometry = o3d.geometry.TriangleMesh()
                    
                    self._mesh_geometry.vertices = o3d.utility.Vector3dVector(vertices_mm)
                    self._mesh_geometry.triangles = o3d.utility.Vector3iVector(faces)
                    self._mesh_geometry.compute_vertex_normals()
                else:
                    if self._mesh_geometry:
                        self._mesh_geometry.clear()
            except (KeyError, IndexError, TypeError) as e:
                print(f"Data error frame {frame_idx}: {e}")
                if self._mesh_geometry:
                    self._mesh_geometry.clear()
        
        if self.kinect_mkv:
            try:
                kinect_frame = self.kinect_mkv[frame_idx]
                pcd = kinect_frame.generate_o3d_point_cloud()
                self._pcd_geometry = pcd if pcd is not None else o3d.geometry.PointCloud()
            except Exception as e:
                print(f"MKV error frame {frame_idx}: {e}")

    def _apply_transforms(self):
        """
        Applies Scale and Translation.
        Refactored: Converts TriangleMesh to LineSet (Wireframe) in RED.
        """
        if self._mesh_geometry is None or self._mesh_geometry.is_empty():
            if self.scene_widget.scene.has_geometry(self.SCENE_MESH):
                self.scene_widget.scene.remove_geometry(self.SCENE_MESH)
            return

        # 1. Clone and Transform Mesh
        vis_mesh = o3d.geometry.TriangleMesh(self._mesh_geometry)
        center = vis_mesh.get_center()
        vis_mesh.scale(self.scale_factor, center)
        vis_mesh.translate(self.translation)
        
        # 2. Convert to LineSet for Transparency + Wireframe
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(vis_mesh)
        line_set.paint_uniform_color([1.0, 0.0, 0.0]) # Force RED Color
        
        # 3. Update Scene
        self.scene_widget.scene.remove_geometry(self.SCENE_MESH)
        self.scene_widget.scene.add_geometry(self.SCENE_MESH, line_set, self._mesh_material)

    def _update_scene_pcd(self):
        if self._pcd_geometry is not None:
            self.scene_widget.scene.remove_geometry(self.SCENE_PCD)
            self.scene_widget.scene.add_geometry(self.SCENE_PCD, self._pcd_geometry, self._pcd_material)

    # --- Callbacks ---

    def _on_frame_ui_change(self, new_val):
        idx = int(new_val)
        if idx != self.current_frame_idx:
            self.current_frame_idx = idx
            self._load_frame_data(idx)
            self._update_scene_pcd()
            self._apply_transforms()

    def _on_pending_scale_change(self, new_val):
        self._pending_scale = float(new_val)

    def _on_pending_trans_change(self, axis_idx, new_val):
        self._pending_translation[axis_idx] = float(new_val)

    def _on_click_process(self):
        self.scale_factor = self._pending_scale
        self.translation = np.copy(self._pending_translation)
        print(f"Processing... Applying Scale: {self.scale_factor}, Trans: {self.translation}")
        self._apply_transforms()

    def _on_validate(self):
        """
        Validates the current configuration, saves state, and closes the application.
        """
        self.final_scale_factor = self.scale_factor
        self.final_translation = np.copy(self.translation)
        self.final_frame_idx = self.current_frame_idx
        
        print(f"Validated Settings -> Scale: {self.final_scale_factor}, Trans: {self.final_translation}, Frame: {self.final_frame_idx}")
        
        self.window.close()
        self.is_running = False

    # --- Camera Callbacks ---

    def _on_center_mesh(self):
        """Recenter camera on the active mesh."""
        if self._mesh_geometry and not self._mesh_geometry.is_empty():
            vis_mesh = o3d.geometry.TriangleMesh(self._mesh_geometry)
            center = vis_mesh.get_center()
            vis_mesh.scale(self.scale_factor, center)
            vis_mesh.translate(self.translation)
            
            bounds = vis_mesh.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60, bounds, bounds.get_center())
            
            self.scene_widget.force_redraw()
            print("Camera centered on mesh.")
        else:
            print("No mesh to center on.")

    def _on_reset_camera(self):
        """Reset camera to initial global state."""
        if self._initial_bounds is not None:
            self.scene_widget.setup_camera(60, self._initial_bounds, self._initial_bounds.get_center())
            self.scene_widget.force_redraw()
            print("Camera reset to default.")

    def run(self) -> Dict[str, Any]:
        """
        Runs the application loop and returns the final configuration dictionary.
        """
        results = {}
        try:
            self.mesh_loader = HandTrackingDataManager(self.mesh_path)
            
            with KinectMKV(self.mkv_path) as mkv:
                self.kinect_mkv = mkv
                
                len_mesh = len(self.mesh_loader)
                len_mkv = len(self.kinect_mkv)
                self.max_frames = min(len_mesh, len_mkv)
                
                print(f"Sync: Mesh={len_mesh}, MKV={len_mkv}. Usable={self.max_frames}")

                if self.max_frames == 0:
                    raise ValueError("No overlapping frames found.")

                self._init_ui()
                
                # Initial Load
                self._load_frame_data(0)
                self._update_scene_pcd()
                self._apply_transforms()
                
                # Camera Setup
                self._initial_bounds = self.scene_widget.scene.bounding_box
                self.scene_widget.setup_camera(60, self._initial_bounds, self._initial_bounds.get_center())
                
                self.app.run()
                
            # Construct result dictionary after run loop finishes
            if self.final_scale_factor is not None:
                results = {
                    "success": True,
                    "input_files": {
                        "mesh_file": str(self.mesh_path),
                        "mkv_file": str(self.mkv_path)
                    },
                    "calibration": {
                        "scale_factor": self.final_scale_factor,
                        "translation_vector": self.final_translation.tolist() if self.final_translation is not None else [0,0,0],
                        "validation_frame_id": self.final_frame_idx
                    }
                }
            else:
                results = {"success": False, "error": "User closed without validating"}
                
        except Exception as e:
            print(f"Execution Error: {e}")
            import traceback
            traceback.print_exc()
            results = {"success": False, "error": str(e)}
            
        return results

if __name__ == "__main__":
    # Example paths - replace with actual paths in production
    MESH_FILE = r"F:/_tmp/scaling_factor/2022-06-17_ST16-05_semicontrolled_block-order03_kinect_handmodel_tracked_hands.pkl"
    MKV_FILE = r"F:/_tmp/scaling_factor/2022-06-17_ST16-05_semicontrolled_block-order03_kinect.mkv"
    OUTPUT_JSON = r"F:/_tmp/scaling_factor/calibration_results.json"
    
    # Check existence before running to prevent crash in example mode
    if Path(MESH_FILE).exists() and Path(MKV_FILE).exists():
        app = HandMeshScalerApp(MESH_FILE, MKV_FILE)
        results = app.run()
        
        print("Final Results:", results)
        
        if results.get("success"):
            try:
                with open(OUTPUT_JSON, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Calibration data successfully saved to: {OUTPUT_JSON}")
            except IOError as e:
                print(f"Failed to write JSON output: {e}")
    else:
        print("Please provide valid paths to run the example.")