import matplotlib
# [Architectural Fix] Force 'Agg' backend to decouple simulation from UI event loops.
# This prevents WinError 10038/QSocketNotifier errors by disabling Qt hooks in Matplotlib.
matplotlib.use('Agg')

import pandas as pd
import open3d as o3d
import numpy as np
from typing import Dict, Union, Tuple, Optional, Any, List
from scipy.spatial.transform import Rotation as R

from ..model.objects_interaction_processor import ObjectsInteractionProcessor

# Note: ObjectsInteractionVisualizer import is removed to decouple View from Controller
# Note: PyQt5 imports are removed to decouple UI lifecycle from Simulation logic

class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model).
    
    Refactored:
    - Handles geometric transformations (Trajectory application) internally.
    - Passes full TriangleMesh object to the Processor.
    """
    def __init__(self,
                 hand_motion_data: dict,
                 references_pcd: Dict[int, Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]],
                 selected_points: dict = None,
                 *,
                 fps: int = 30,
                 visualizer_width_sec: int = 3
    ):
        # Processor now only needs indices/labels, not base vertices
        self.processor = ObjectsInteractionProcessor(
            references_dict=references_pcd,
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
        # Store base vertices as numpy array for calculations
        self.base_vertices = np.asarray(base_mesh.vertices)
        # Store base triangles to reconstruct meshes efficiently later
        self.base_triangles = base_mesh.triangles
    
        self.hand_motion_data = hand_motion_data  
        self.view_width_in_frame = visualizer_width_sec * fps       

    def _compute_hand_mesh_for_frame(self, frame_id: int) -> Tuple[o3d.geometry.TriangleMesh, bool]:
        """
        Private method to fetch and compute the current hand mesh based on frame ID.

        Args:
            frame_id (int): The index of the frame to compute.

        Returns:
            Tuple[o3d.geometry.TriangleMesh, bool]: 
                - The transformed mesh for the specific frame.
                - A boolean flag indicating if the frame data was valid.
        """
        # --- Transformation Logic ---
        translation = self.trajectory_data['translations'][frame_id]
        rotation_quat = self.trajectory_data['rotations'][frame_id]
        
        # Check for invalid data (zero quaternion)
        is_valid_frame = True
        if np.all(rotation_quat == 0) or np.all(translation == 0):
            is_valid_frame = False
            current_vertices = np.copy(self.base_vertices) # Use rest pose for visualization, metrics will be NaN
        else:
            # Apply Rigid Body Transformation
            # R.from_quat expects [x, y, z, w]
            rot = R.from_quat(rotation_quat)
            # Apply rotation and translation: v' = R * v + t
            current_vertices = rot.apply(self.base_vertices) + translation

        # Create the mesh for the current frame
        current_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(current_vertices),
            self.base_triangles
        )
        # Ensure color/normals are consistent (copying from base)
        # Correction: TriangleMesh uses vertex_colors, not colors (which is for PointCloud)
        current_mesh.vertex_colors = self.base_mesh.vertex_colors
        
        return current_mesh, is_valid_frame

    def run(self) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Executes the simulation.
        
        Returns:
            Tuple[pd.DataFrame, Optional[Dict]]: 
                - DataFrame containing simulation metrics.
                - Dictionary containing visualization assets (frames, meshes, references).
        """
        results = []
        num_frames = len(self.trajectory_data['time_points'])

        print(f"Computing {num_frames} frames...")
        visualization_frames = []

        for frame_id, time in enumerate(self.trajectory_data['time_points']):
            if self.processor.is_valid_pcd_key(frame_id):
                self.processor.set_current_pcd(frame_id)
            
            use_debug = False #(frame_id == 625)
            
            # Fetch the transformed mesh via the new private method
            current_mesh, is_valid_frame = self._compute_hand_mesh_for_frame(frame_id)

            # Pass transformed mesh to processor
            frame_result = self.processor.process_single_frame(
                current_mesh=current_mesh,
                data_valid=is_valid_frame,
                _debug=use_debug
            )
            
            # Store metrics
            metrics = frame_result["metrics"]
            metrics["time"] = time
            metrics["frame_index"] = frame_id
            results.append(metrics)
            
            # Prepare and store visualization data for this frame
            vis_data = frame_result["visualization"]
            # Rename key to match expected output for visualizer compatibility, if needed
            vis_data["transformed_hand_mesh"] = vis_data.pop("transformed_mesh")
            visualization_frames.append(vis_data)

            # Log progress
            if (frame_id + 1) % self.processor.fps == 0 or (frame_id + 1) == num_frames:
                print(f"  Computed frame {frame_id + 1}/{num_frames}")
        
        print("Simulation finished.")
        
        # Compile DataFrame
        if not results:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(results)
            all_cols = list(df.columns)
            
            # Define the desired primary order
            ordered_prefix = ['frame_index', 'time', 'contact_detected']
            
            # Dynamically find columns matching patterns
            contact_loc_cols = sorted([c for c in all_cols if c.startswith('contact_location_')])
            velocity_cols = sorted([c for c in all_cols if c.startswith('velocity_')])
            
            # Combine the ordered and patterned columns
            new_order_start = ordered_prefix + contact_loc_cols + velocity_cols
            
            # Get the remaining columns
            remaining_cols = sorted([c for c in all_cols if c not in new_order_start])
            
            # Create the final column list and reorder the DataFrame
            final_order = new_order_start + remaining_cols
            df = df[final_order]

        # Bundle visualization artifacts
        vis_artifacts = {
            "frames": visualization_frames,
            "reference_pcd": self.processor.ref_pcd,
            "base_mesh": self.base_mesh,
            "view_width": self.view_width_in_frame
        }

        return df, vis_artifacts