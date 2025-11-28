import sys
import pandas as pd
import open3d as o3d
import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation as R

from PyQt5.QtWidgets import QApplication

from ..model.objects_interaction_processor import ObjectsInteractionProcessor
from ..gui.objects_interaction_visualizer import ObjectsInteractionVisualizer


class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model)
    and ObjectsInteractionVisualizer (View). This is the 'Controller'.

    Supports two modes:
    1. Real-time simulation and visualization (`monitor=False`).
    2. Monitor mode (`monitor=True`): Pre-computes all frames, then launches an
       interactive visualizer to inspect the results on demand.
    """
    def __init__(self,
                 hand_motion_data: dict,
                 references_pcd: Dict[int, o3d.geometry.PointCloud],
                 selected_points: dict = None,
                 *,
                 visualize: bool = True,
                 fps: int = 30,
                 visualizer_width_sec: int = 3
    ):
        self.processor = ObjectsInteractionProcessor(
            references_pcd=references_pcd,
            base_vertices=hand_motion_data['vertices'],
            tracked_points_groups_indices=selected_points.values() if selected_points else [],
            tracked_points_groups_labels=list(selected_points.keys()) if selected_points else [],
            fps=fps
        )
        self.trajectory_data = {
            'time_points': hand_motion_data['time_points'], 
            'translations': hand_motion_data['translations'], 
            'rotations': hand_motion_data['rotations']
        }

        hand_color = [228/255, 178/255, 148/255]
        base_mesh = o3d.geometry.TriangleMesh()
        base_mesh.vertices = o3d.utility.Vector3dVector(hand_motion_data['vertices'])
        base_mesh.triangles = o3d.utility.Vector3iVector(hand_motion_data['faces'])
        base_mesh.paint_uniform_color(hand_color)
        base_mesh.compute_vertex_normals()
        self.base_mesh = base_mesh
        # Store base triangles to reconstruct meshes efficiently later
        self.base_triangles = base_mesh.triangles
    
        self.visualize = visualize
        self.view = None
        self.hand_motion_data = hand_motion_data  
        self.view_width_in_frame = visualizer_width_sec * fps       

    def run(self) -> pd.DataFrame:
        """
        Executes the simulation.
        
        If `monitor` is True, it pre-computes all frame data and then launches an
        interactive session where results for any frame can be fetched and viewed.
        
        Otherwise, it runs the simulation frame-by-frame, visualizing in real-time
        if `visualize` is True.
        
        Returns:
            pd.DataFrame: A DataFrame containing the simulation metrics for each frame.
        """
        results = []
        num_frames = len(self.trajectory_data['time_points'])

        # --- MONITOR MODE: Pre-compute all, then visualize interactively ---
        print(f"Pre-computing {num_frames} frames for monitoring...")
        visualization_frames = []

        for frame_id, time in enumerate(self.trajectory_data['time_points']):
            if self.processor.is_valid_pcd_key(frame_id):
                self.processor.set_current_pcd(frame_id)
            
            use_debug = (frame_id == 625)
            
            frame_result = self.processor.process_single_frame(
                translation=self.trajectory_data['translations'][frame_id],
                rotation_quat=self.trajectory_data['rotations'][frame_id],
                _debug=use_debug
            )
            
            # Store metrics
            metrics = frame_result["metrics"]
            metrics["time"] = time
            metrics["frame_index"] = frame_id
            results.append(metrics)
            
            # Prepare and store visualization data for this frame
            if self.visualize:
                vis_data = frame_result["visualization"]
                transformed_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vis_data["transformed_vertices"]),
                    self.base_triangles
                )
                vis_data["transformed_hand_mesh"] = transformed_mesh
                visualization_frames.append(vis_data)

            # 3. Log progress
            if (frame_id + 1) % self.processor.fps == 0 or (frame_id + 1) == num_frames:
                print(f"  Computed frame {frame_id + 1}/{num_frames}")
        print("Pre-computation finished.")
        
        # 4. Launch interactive session with all pre-computed data
        if self.visualize:
            self.interactive_view(visualization_frames, results)

        print("Simulation finished.")
        
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        all_cols = list(df.columns)
        
        # Define the desired primary order
        ordered_prefix = ['frame_index', 'time', 'contact_detected']
        
        # Dynamically find columns matching patterns
        contact_loc_cols = sorted([c for c in all_cols if c.startswith('contact_location_')])
        velocity_cols = sorted([c for c in all_cols if c.startswith('velocity_')])
        
        # Combine the ordered and patterned columns
        new_order_start = ordered_prefix + contact_loc_cols + velocity_cols
        
        # Get the remaining columns, ensuring no duplicates and maintaining a sorted order
        remaining_cols = sorted([c for c in all_cols if c not in new_order_start])
        
        # Create the final column list and reorder the DataFrame
        final_order = new_order_start + remaining_cols
        
        return df[final_order]
    
    def interactive_view(self, visualization_frames, results):
        print("Starting interactive monitoring session...")
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        
        # We create an empty PointCloud with the desired red color.
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector([[0, 0, -9999]])
        contact_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red

        self.view = ObjectsInteractionVisualizer(width=self.view_width_in_frame)
        self.view.add_geometry('forearm', self.processor.ref_pcd)
        self.view.add_geometry('hand', self.base_mesh)
        self.view.add_geometry('contacts', contact_pcd, point_size=10.0)
        self.view.set_frame_data(visualization_frames)

        # --- DATA PROCESSING START ---
        # Extract metric data
        contact_depth_data = [r.get('contact_depth', 0) for r in results]
        contact_area_data = [r.get('contact_area', 0) for r in results]
        self.view.add_plot(
            title="Contact Properties",
            data_vector=contact_depth_data,
            color='g',
            y_label='Depth (NA)'
        )
        self.view.add_plot(
            title="Contact Properties",
            data_vector=contact_area_data,
            color='m',
            y_label='Contact Area (cm^2)'
        )

        center_point = self.processor.ref_pcd.get_center()
        self.view.recenter_view_on_point(center_point)

        self.view.run()

        app = QApplication.instance()
        app.exec()

def create_mock_data(num_frames=200, fps=30):
    """
    Generates mock data for testing the ObjectsInteractionOrchestrator.

    Args:
        num_frames (int): The number of frames to generate for the simulation.
        fps (int): The frames per second rate for the simulation.

    Returns:
        tuple: A tuple containing:
            - dict: hand_motion_data for the controller.
            - o3d.geometry.PointCloud: The static reference object.
    """
    # 1. Create a simple hand mesh (e.g., a sphere)
    hand_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    hand_mesh.compute_vertex_normals()
    hand_vertices = np.asarray(hand_mesh.vertices)
    hand_faces = np.asarray(hand_mesh.triangles)

    # 2. Create a static object (e.g., a flat plane point cloud)
    x = np.linspace(-0.5, 0.5, 20)
    y = np.linspace(-0.5, 0.5, 20)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    static_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    static_object = o3d.geometry.PointCloud()
    static_object.points = o3d.utility.Vector3dVector(static_points)
    static_object.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    # Estimate normals for the point cloud, which is crucial for contact analysis.
    # For a flat plane, we expect normals to point along the Z-axis.
    static_object.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    static_object.orient_normals_to_align_with_direction([0, 0, 1])

    # 3. Generate a simple trajectory for the hand
    time_points = np.linspace(0, num_frames / fps, num_frames)
    translations = []
    rotations = []

    # The hand will move from z=0.5 down to z=-0.1 and rotate
    start_z, end_z = 0.5, -0.1
    z_path = np.linspace(start_z, end_z, num_frames)

    for i in range(num_frames):
        # Translation: move along the z-axis
        trans = np.array([0.0, 0.0, z_path[i]])
        translations.append(trans)

        # Rotation: rotate slowly around the y-axis
        angle = np.pi / 2 * (i / num_frames) # 90-degree rotation over the whole sequence
        rotation = R.from_euler('y', angle)
        rotations.append(rotation.as_quat()) # as [x, y, z, w]

    hand_motion_data = {
        'vertices': hand_vertices,
        'faces': hand_faces,
        'time_points': time_points,
        'translations': np.array(translations),
        'rotations': np.array(rotations)
    }

    return hand_motion_data, static_object

if __name__ == '__main__':
    # This main block demonstrates how to use the ObjectsInteractionOrchestrator
    # and serves as a test harness.

    # 1. Create the QApplication instance FIRST. This is crucial for any Qt-based UI.
    # It must be done before any widgets (like our visualizer) are instantiated,
    # even if that instantiation happens inside another class.
    if QApplication.instance() is None:
        app = QApplication(sys.argv)

    # 2. Generate the mock data for the simulation
    print("Generating mock data...")
    mock_hand_data, mock_static_object = create_mock_data(num_frames=200, fps=30)
    print("Mock data generated.")

    # 3. Instantiate the controller with the mock data
    # The controller will manage the simulation logic and the UI.
    controller = ObjectsInteractionController(
        hand_motion_data=mock_hand_data,
        references_pcd={0: mock_static_object}, # Updated to match Dict type hint
        selected_points=None,
        visualize=True,
        fps=30,
        visualizer_width_sec=5 # The plot will show a 5-second window
    )

    # 4. Run the controller. This will start the pre-computation, then launch the visualizer.
    # The script will block here until the visualizer window is closed.
    results_df = controller.run()

    # 5. Post-simulation processing. This executes after the user closes the Qt window.
    print("\n--- Simulation Complete ---")
    print("Metrics collected for all frames:")
    print(results_df.head())
    print("---------------------------\n")