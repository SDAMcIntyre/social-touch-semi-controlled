import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

# --- Standalone Visualization Function ---

def visualize_point_cloud_comparison(
    pcd_left: o3d.geometry.Geometry,
    pcd_right: o3d.geometry.Geometry,
    title: str = "Point Cloud Comparison",
    left_label: str = "Original",
    right_label: str = "Modified"
) -> None:
    """
    Launches a standalone Open3D GUI window with split-screen synchronized views
    and controls to recenter or flip the visualization.
    
    Args:
        pcd_left: The geometry object to display in the left viewport.
        pcd_right: The geometry object to display in the right viewport.
        title: Window title.
        left_label: Label for the left geometry.
        right_label: Label for the right geometry.
    """
    print(f"👀 Interactive Mode: Launching Split-Screen Synchronized View.")
    
    # Initialize the Application
    app = gui.Application.instance
    try:
        app.initialize()
    except Exception as e:
        # If running in a persistent environment (like Jupyter) or loop, 
        # it might already be initialized. We log and proceed.
        print(f"Open3D Application initialization notice: {e}")

    # Create the window
    window = gui.Application.instance.create_window(title, 1280, 768) # Increased height slightly for controls

    # --- UI Control Panel ---
    em = window.theme.font_size
    panel = gui.Horiz(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    
    btn_recenter = gui.Button("Recenter Origin")
    btn_recenter.horizontal_padding_em = 0.5
    btn_recenter.vertical_padding_em = 0
    
    btn_flip = gui.Button("Flip Direction")
    btn_flip.horizontal_padding_em = 0.5
    btn_flip.vertical_padding_em = 0
    
    panel.add_child(btn_recenter)
    panel.add_child(btn_flip)
    
    # --- Scene Widgets ---
    widget_left = gui.SceneWidget()
    widget_left.scene = rendering.Open3DScene(window.renderer)
    widget_left.scene.set_background([0.1, 0.1, 0.1, 1.0])

    widget_right = gui.SceneWidget()
    widget_right.scene = rendering.Open3DScene(window.renderer)
    widget_right.scene.set_background([0.1, 0.1, 0.1, 1.0])

    # Add Geometry
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = 3.0 

    widget_left.scene.add_geometry(left_label, pcd_left, mat)
    widget_right.scene.add_geometry(right_label, pcd_right, mat)

    # Calculate Bounding Box (used for Recenter and Flip logic)
    bbox = pcd_left.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()

    # --- Callback Logic ---

    def on_recenter_click():
        """Resets the camera view to the default bounding box center."""
        # Setup camera on the left widget; sync loop will handle the right one.
        widget_left.setup_camera(60.0, bbox, bbox_center)
        # Force a redraw to ensure immediate visual feedback
        widget_left.force_redraw()

    def on_flip_click():
        """Flips the camera to look at the object from the opposite side."""
        # Get current camera pose
        model_matrix = widget_left.scene.camera.get_model_matrix()
        
        # Decompose matrix
        eye = model_matrix[:3, 3]
        R = model_matrix[:3, :3]
        up = R[:, 1]
        
        # Calculate vector from center to current eye
        center_to_eye = eye - bbox_center
        
        # Invert the vector to find the new eye position (mirrored across center)
        new_eye = bbox_center - center_to_eye
        
        # Apply new look_at to the left widget
        widget_left.scene.camera.look_at(bbox_center, new_eye, up)
        widget_left.force_redraw()

    # Connect buttons
    btn_recenter.set_on_clicked(on_recenter_click)
    btn_flip.set_on_clicked(on_flip_click)

    # --- Layout Management ---
    def on_layout(layout_context):
        r = window.content_rect
        
        # Calculate Panel Height (based on preferred height of content)
        panel_height = panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height
        
        # Set Panel Frame (Top strip)
        panel.frame = gui.Rect(r.x, r.y, r.width, panel_height)
        
        # Calculate Viewport Area
        view_y = r.y + panel_height
        view_height = r.height - panel_height
        
        # Split width 50/50 for the scene widgets
        widget_left.frame = gui.Rect(r.x, view_y, r.width // 2, view_height)
        widget_right.frame = gui.Rect(r.x + r.width // 2, view_y, r.width // 2, view_height)

    window.set_on_layout(on_layout)
    window.add_child(panel)
    window.add_child(widget_left)
    window.add_child(widget_right)

    # --- Synchronization Logic (Polling Implementation) ---
    class CameraState:
        def __init__(self):
            self.left_matrix = np.eye(4)
            self.right_matrix = np.eye(4)
            
    state = CameraState()

    def apply_pose_to_camera(target_widget, source_matrix):
        """
        Decomposes the source matrix into eye, center, and up vectors 
        and applies them to the target camera using look_at.
        """
        eye = source_matrix[:3, 3]
        R = source_matrix[:3, :3]
        up = R[:, 1]
        forward = -R[:, 2]
        center = eye + forward
        target_widget.scene.camera.look_at(center, eye, up)

    def sync_loop():
        # --- Safety Check: Wait for Layout ---
        if widget_left.frame.height <= 0 or widget_right.frame.height <= 0:
            gui.Application.instance.post_to_main_thread(window, sync_loop)
            return
        
        # Get current matrices
        current_left = widget_left.scene.camera.get_model_matrix()
        current_right = widget_right.scene.camera.get_model_matrix()

        # Check if Left changed
        if not np.allclose(current_left, state.left_matrix, atol=1e-6):
            apply_pose_to_camera(widget_right, current_left)
            
            cam_left = widget_left.scene.camera
            widget_right.scene.camera.set_projection(
                cam_left.get_field_of_view(),
                widget_right.frame.width / widget_right.frame.height,
                0.1, 1000.0,
                rendering.Camera.FovType.Vertical
            )
            widget_right.force_redraw()
            
            state.left_matrix = current_left
            state.right_matrix = widget_right.scene.camera.get_model_matrix()

        # Check if Right changed
        elif not np.allclose(current_right, state.right_matrix, atol=1e-6):
            apply_pose_to_camera(widget_left, current_right)

            cam_right = widget_right.scene.camera
            widget_left.scene.camera.set_projection(
                cam_right.get_field_of_view(),
                widget_left.frame.width / widget_left.frame.height,
                0.1, 1000.0,
                rendering.Camera.FovType.Vertical
            )
            widget_left.force_redraw()
            
            state.right_matrix = current_right
            state.left_matrix = widget_left.scene.camera.get_model_matrix()

        # Re-schedule this function to run on the next frame
        gui.Application.instance.post_to_main_thread(window, sync_loop)

    # Initial Camera Setup based on Left Geometry
    widget_left.setup_camera(60.0, bbox, bbox.get_center())
    
    # Trigger initial sync
    gui.Application.instance.post_to_main_thread(window, sync_loop)

    # Run
    gui.Application.instance.run()