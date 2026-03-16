import pandas as pd
import open3d as o3d
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Union

from ..model.objects_interaction_processor import ObjectsInteractionProcessor

class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model).
    
    Architectural Update:
    - Decoupled from HandMotionManager/GLB data structures.
    - Expects pre-transformed World Space meshes (Sequence).
    - Implements input normalization for proprioception landmarks.
    - Implements vertex filtering based on exclusion metadata.
    """
    def __init__(self,
                 hand_meshes: List[o3d.geometry.TriangleMesh],
                 timestamps: List[float],
                 references_mesh: Dict[int, o3d.geometry.TriangleMesh],
                 selected_points: Optional[Dict[str, Union[List[int], int]]] = None,
                 excluded_vertex_ids: Optional[List[int]] = None,
                 *,
                 fps: int = 30,
                 visualizer_width_sec: int = 3
    ):
        """
        Args:
            hand_meshes: A time-ordered sequence of Open3D meshes (World Space).
            timestamps: A time-ordered sequence of floating point timestamps.
            references_mesh: Dictionary mapping frame indices to environmental point clouds/meshes.
            selected_points: Dictionary of anatomical landmarks for proprioception tracking. 
                             Accepts Dict[str, int] or Dict[str, List[int]].
            excluded_vertex_ids: List of vertex indices to remove from the mesh before contact processing.
        """
        if len(hand_meshes) != len(timestamps):
            raise ValueError(f"Data Mismatch: {len(hand_meshes)} meshes vs {len(timestamps)} timestamps.")

        self.fps = fps
        self.hand_meshes = hand_meshes
        self.timestamps = timestamps
        self.references_mesh = references_mesh
        self.excluded_vertex_ids = excluded_vertex_ids
        
        # Initialize Processor with the first available reference in the dict
        if not references_mesh:
            raise ValueError("References dictionary cannot be empty.")
            
        initial_key = next(iter(references_mesh))
        self.contact_processor = ObjectsInteractionProcessor(reference_geometry=references_mesh[initial_key])

        # Architectural Fix: Input Normalization
        # Converts scalar integers to lists to ensure consistent iteration downstream.
        self.tracked_points_groups_labels: List[str] = []
        self.tracked_points_groups_indices: List[List[int]] = []

        if selected_points:
            self.tracked_points_groups_labels = list(selected_points.keys())
            raw_values = selected_points.values()
            normalized_indices = []
            
            for val in raw_values:
                if isinstance(val, int):
                    normalized_indices.append([val])
                elif isinstance(val, list):
                    normalized_indices.append(val)
                else:
                    # Fallback for other iterables
                    normalized_indices.append(list(val))
            
            self.tracked_points_groups_indices = normalized_indices

        # Store base mesh for static visualization reference (Frame 0)
        self.base_mesh = hand_meshes[0]
        self.view_width_in_frame = visualizer_width_sec * fps       

    def _calculate_proprioceptive_data(
            self,
            v_transformed: np.ndarray
        ) -> Tuple[Dict[str, float], Dict[str, Any]]:
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
            # indices is guaranteed to be a List[int] due to __init__ normalization
            
            # Ensure indices are valid for current mesh
            valid_indices = [i for i in indices if i < len(v_transformed)]
            if valid_indices:
                position_xyz = np.mean(v_transformed[valid_indices, :], axis=0)
                proprio_data[f"{label}_position_x"] = position_xyz[0]
                proprio_data[f"{label}_position_y"] = position_xyz[1]
                proprio_data[f"{label}_position_z"] = position_xyz[2]
        
        return proprio_data, {}

    def run(self) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Executes the simulation iterating over the injected mesh sequence.
        """
        results = []
        num_frames = len(self.timestamps)

        print(f"Computing {num_frames} frames...")
        visualization_frames = []

        # Iterate directly over the sequence of meshes and times
        for frame_id, (time, current_mesh) in enumerate(zip(self.timestamps, self.hand_meshes)):
            
            # --- Reference Management ---
            if frame_id in self.references_mesh:
                self.contact_processor.update_reference(self.references_mesh[frame_id])
            
            # --- Processing ---
            # 1. Proprioception Calculation
            # MUST happen before vertex removal to ensure indices in `selected_points` remain valid.
            current_vertices = np.asarray(current_mesh.vertices)
            proprio_data, _ = self._calculate_proprioceptive_data(current_vertices)
            
            # 2. Vertex Filtering (Exclusion)
            # If excluded_vertex_ids are present, remove them from the mesh.
            # Open3D's remove_vertices_by_index modifies the mesh in-place.
            if self.excluded_vertex_ids:
                current_mesh.remove_vertices_by_index(self.excluded_vertex_ids)

            # 3. Process Contact
            # The contact processor now receives the mesh with excluded vertices removed.
            contact_data, viz = self.contact_processor.process_single_frame(current_mesh=current_mesh, _debug=False)

            # Store metrics
            merged_data = {**proprio_data, **contact_data}
            merged_data["time"] = time
            merged_data["frame_index"] = frame_id
            results.append(merged_data)
            
            # Prepare visualization data
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
            
            # Reorder Logic
            ordered_prefix = ['frame_index', 'time', 'contact_detected']
            contact_loc_cols = sorted([c for c in all_cols if c.startswith('contact_location_')])
            new_order_start = ordered_prefix + contact_loc_cols
            remaining_cols = sorted([c for c in all_cols if c not in new_order_start])
            final_order = new_order_start + remaining_cols
            df = df[final_order]

        # Bundle visualization artifacts
        vis_artifacts = {
            "frames": visualization_frames,
            "reference_mesh": self.references_mesh[next(iter(self.references_mesh))],
            "base_mesh": self.base_mesh,
            "view_width": self.view_width_in_frame
        }

        return df, vis_artifacts