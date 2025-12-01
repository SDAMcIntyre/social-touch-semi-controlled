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

class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model).
    
    Refactored:
    - Handles geometric transformations (Trajectory application) internally.
    - Manages the reference PointClouds/Meshes and updates the Processor.
    - Handles calculation of the second mesh (Hand).
    """
    def __init__(self,
                 hand_motion_data: dict,
                 references_pcd: Dict[int, Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]],
                 selected_points: dict = None,
                 *,
                 fps: int = 30,
                 visualizer_width_sec: int = 3
    ):
        self.fps = fps
        self.references_pcd = references_pcd
        
        # Initialize Processor with the first available reference in the dict
        # We assume the dict is not empty based on typical flow
        if not references_pcd:
            raise ValueError("References dictionary cannot be empty.")
            
        initial_key = next(iter(references_pcd))
        
        # Processor initialized with a SINGLE reference geometry. 
        # No indices or tracking passed to processor.
        self.processor = ObjectsInteractionProcessor(reference_geometry=references_pcd[initial_key])
        
        self.trajectory_data = {
            'time_points': hand_motion_data['time_points'], 
            'translations': hand_motion_data['translations'], 
            'rotations': hand_motion_data['rotations']
        }

        self.tracked_points_groups_indices = selected_points.values() if selected_points else []
        self.tracked_points_groups_labels = list(selected_points.keys()) if selected_points else []

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
        This handles the calculation of the position of the second mesh (the hand).

        Args:
            frame_id (int): The index of the frame to compute.

        Returns:
            Tuple[o3d.geometry.TriangleMesh, bool]: 
                - The transformed mesh for the specific frame.
                - A boolean flag indicating if the frame data was valid.
        """
        # --- Transformation Logic ---
        # Ensure we don't go out of bounds if trajectory data is shorter than expected
        if frame_id >= len(self.trajectory_data['translations']):
             return self.base_mesh, False

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
        current_mesh.vertex_colors = self.base_mesh.vertex_colors
        
        return current_mesh, is_valid_frame

    def _calculate_proprioceptive_data(
            self,
            v_transformed: np.ndarray
        ) -> tuple[dict, dict]:
        """Calculates motion-related metrics for a single frame using provided vertices."""
        if not self.tracked_points_groups_labels:
            return {}, {}
        
        proprio_keys = []
        for label in self.tracked_points_groups_labels:
            proprio_keys.extend([
                f"{label}_position_x", f"{label}_position_y", f"{label}_position_z"
            ])
        proprio_data = {key: np.nan for key in proprio_keys}

        for label, indices in zip(self.tracked_points_groups_labels, self.tracked_points_groups_indices):
            position_xyz = np.mean(v_transformed[np.atleast_1d(indices), :], axis=0)
            proprio_data[f"{label}_position_x"], proprio_data[f"{label}_position_y"], proprio_data[f"{label}_position_z"] = position_xyz
        return proprio_data

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
            
            # --- Reference Management (Controller Logic) ---
            # If there is a specific reference point cloud defined for this frame ID,
            # update the processor's reference.
            if frame_id in self.references_pcd:
                self.processor.update_reference(self.references_pcd[frame_id])
            
            # --- Hand Mesh Calculation (Controller Logic) ---
            # Fetch the transformed mesh via the private method
            current_mesh, is_valid_frame = self._compute_hand_mesh_for_frame(frame_id)

            # --- Processing (Processor Logic) ---
            proprio_data = self._calculate_proprioceptive_data(np.asarray(current_mesh.vertices))
            # Pass transformed mesh to processor.
            if is_valid_frame:
                contact_data, viz = self.processor.process_single_frame(current_mesh=current_mesh, _debug=False)
            else:
                contact_data, viz = self.processor.empty_structure()

            # Store metrics
            merged_data = {**proprio_data, **contact_data}
            merged_data["time"] = time
            merged_data["frame_index"] = frame_id
            results.append(merged_data)
            
            # Prepare and store visualization data for this frame
            viz["transformed_hand_mesh"] = current_mesh
            visualization_frames.append(viz)

            # Log progress
            if (frame_id + 1) % self.fps == 0 or (frame_id + 1) == num_frames:
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
            
            # Combine the ordered and patterned columns
            # Velocity columns are removed from calculation, so we don't search for them here
            new_order_start = ordered_prefix + contact_loc_cols
            
            # Get the remaining columns
            remaining_cols = sorted([c for c in all_cols if c not in new_order_start])
            
            # Create the final column list and reorder the DataFrame
            final_order = new_order_start + remaining_cols
            df = df[final_order]

        # Bundle visualization artifacts
        vis_artifacts = {
            "frames": visualization_frames,
            # Note: We return the current ref_pcd of the processor. 
            # If visualizer needs per-frame reference updates, that logic would reside in the visualizer.
            "reference_pcd": self.processor.ref_pcd,
            "base_mesh": self.base_mesh,
            "view_width": self.view_width_in_frame
        }

        return df, vis_artifacts