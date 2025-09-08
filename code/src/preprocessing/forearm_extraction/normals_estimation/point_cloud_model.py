# point_cloud_model.py
import numpy as np
import open3d as o3d
import os
import sys
import json
import copy # <-- Added import for deepcopy

class PointCloudModel:
    """
    Manages the data and processing logic for a point cloud.
    This class is UI-agnostic and serves as the "Model" in an MVC architecture.
    It maintains an original, unmodified point cloud as a reference
    and applies transformations non-destructively for processing and saving.
    """
    def __init__(
        self,
        k_neighbors: int = 100,
        radius: float = 0.1,
        hybrid_tree: bool = False,
        align_with_viewpoint: bool = False,
        viewpoint: np.ndarray = np.array([0.0, 0.0, 0.0])
    ):
        # Processing parameters
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.hybrid_tree = hybrid_tree
        self.align_with_viewpoint = align_with_viewpoint
        self.viewpoint = viewpoint
        
        # Transformation parameters
        self.is_centered: bool = False
        self.scale_factor: float = 1.0
        self.normals_flipped: bool = False

        # --- Core Data ---
        # The original point cloud is kept untouched as a reference.
        self.pcd_original: o3d.geometry.PointCloud | None = None
        
        # Derived data: Normals are computed from the original and cached here.
        self.normals: np.ndarray | None = None

        # Helper attributes derived from the original cloud
        self.cloud_center: np.ndarray = np.array([0.0, 0.0, 0.0])

    def load(self, filepath: str) -> bool:
        """Loads a point cloud from a file, storing it as the unmodified reference."""
        print(f"Loading point cloud from: {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
            return False
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if not pcd.has_points():
                print(f"Error: No points found in {filepath}", file=sys.stderr)
                return False
            
            # Store the original point cloud as the immutable reference
            self.pcd_original = pcd
            
            # Calculate properties from the original cloud
            self.cloud_center = self.pcd_original.get_center()
            print(f"Cloud center calculated at: {self.cloud_center}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            return False

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns the min and max corner points of the original cloud's bounding box."""
        if self.pcd_original is None:
            return None
        return self.pcd_original.get_min_bound(), self.pcd_original.get_max_bound()

    def compute_normals(self):
        """Computes normals based on the original point cloud and current parameters."""
        if self.pcd_original is None:
            print("Error: Point cloud not loaded, cannot compute normals.", file=sys.stderr)
            return
        
        print(f"Computing normals with params: k={self.k_neighbors}, radius={self.radius}, hybrid={self.hybrid_tree}")
        
        # Work on a temporary copy to avoid modifying the original
        pcd_for_normals = o3d.geometry.PointCloud(self.pcd_original)

        if self.hybrid_tree:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.k_neighbors)
        else:
            search_param = o3d.geometry.KDTreeSearchParamKNN(knn=self.k_neighbors)
        
        pcd_for_normals.estimate_normals(search_param=search_param)
        
        if self.align_with_viewpoint:
            pcd_for_normals.orient_normals_to_align_with_direction(orientation_reference=self.viewpoint)
        else:
            pcd_for_normals.orient_normals_consistent_tangent_plane(self.k_neighbors)
        
        # Cache the computed normals as a numpy array
        self.normals = np.asarray(pcd_for_normals.normals)
        if self.normals_flipped:
            self.normals *= -1
            
        print("Normals computed and oriented.")

    def flip_normals(self):
        """Flips the direction of the cached normals."""
        if self.normals is not None:
            self.normals *= -1
            self.normals_flipped = not self.normals_flipped
            print("Normals flipped.")
    
    # This helper function is no longer needed with the new get_processed_pcd logic
    # but is kept here for potential use in other contexts.
    def get_transformed_points(self) -> np.ndarray:
        """
        Applies centering and scaling transformations to a copy of the original points.
        The original point cloud data is never modified.
        """
        if self.pcd_original is None:
            return np.array([])
        
        points = np.asarray(self.pcd_original.points).copy()
        if self.is_centered:
            points -= self.cloud_center
        return points * self.scale_factor
        
    def get_processed_pcd(self) -> o3d.geometry.PointCloud | None:
        """
        Creates a deep copy of the original point cloud and applies all configured
        transformations (centering, scaling) and computed data (normals) to it.
        
        Returns:
            A new, fully processed Open3D PointCloud object, or None if data is missing.
        """
        if self.pcd_original is None or self.normals is None:
            print("Error: Original points or normals are not available.", file=sys.stderr)
            return None
        
        # 1. Create a deep copy to ensure the original is never altered.
        pcd_processed = copy.deepcopy(self.pcd_original)
        
        # 2. Apply transformations directly to the copied object.
        if self.is_centered:
            # Translate the cloud so its original center moves to the origin (0,0,0).
            pcd_processed.translate(-self.cloud_center, relative=True)

        # Scale the cloud relative to the origin. If centered, this scales
        # it around its center. If not, it scales around the world origin.
        pcd_processed.scale(self.scale_factor, center=np.array([0.0, 0.0, 0.0]))

        # 3. Assign the computed normals to the processed cloud.
        pcd_processed.normals = o3d.utility.Vector3dVector(self.normals)
        
        return pcd_processed

    def save(self, output_path: str, metadata_path: str):
        """Saves the fully processed point cloud and its corresponding metadata."""
        print("Saving final results...")
        
        # Generate the final point cloud object with all modifications
        pcd_to_save = self.get_processed_pcd()
        
        if pcd_to_save is None:
            print("Error: Could not generate processed point cloud. Nothing to save.", file=sys.stderr)
            return
            
        # Write the processed point cloud to a PLY file
        try:
            o3d.io.write_point_cloud(output_path, pcd_to_save, write_ascii=True)
            print(f"Successfully saved point cloud to: {output_path}")
        except Exception as e:
            print(f"Error saving point cloud to {output_path}: {e}", file=sys.stderr)
            return # Abort if point cloud saving fails
        
        # Save the metadata describing the processing
        self._save_metadata(output_path, metadata_path)
    
    def _save_metadata(self, output_file: str, metadata_path: str):
        """Saves processing and transformation parameters to a JSON file."""
        print(f"Saving metadata to: {metadata_path}")
        metadata = {
            "output_file": str(output_file),
            "processing_parameters": {
                "k_neighbors_for_normals": self.k_neighbors,
                "radius_for_hybrid": self.radius,
                "used_hybrid_tree": self.hybrid_tree,
                "aligned_with_viewpoint": self.align_with_viewpoint,
                "viewpoint_vector": self.viewpoint.tolist(),
            },
            "final_transformations": {
                "is_centered": self.is_centered,
                "cloud_center_vector": self.cloud_center.tolist(),
                "scale_factor": self.scale_factor,
                "normals_flipped": self.normals_flipped,
            }
        }
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print("Successfully saved metadata.")
        except Exception as e:
            print(f"Error saving metadata file to {metadata_path}: {e}", file=sys.stderr)