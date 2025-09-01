# point_cloud_model.py
import numpy as np
import open3d as o3d
import os
import sys
import json

class PointCloudModel:
    """
    Manages the data and processing logic for a point cloud.
    This class is UI-agnostic. It is the "Model" in MVC.
    """
    def __init__(
        self,
        k_neighbors: int = 100,
        radius: float = 0.1,
        hybrid_tree: bool = False,
        align_with_viewpoint: bool = False,
        viewpoint: np.ndarray = np.array([0.0, 0.0, 0.0])
    ):
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.hybrid_tree = hybrid_tree
        self.align_with_viewpoint = align_with_viewpoint
        self.viewpoint = viewpoint
        self.original_points: np.ndarray | None = None
        self.normals: np.ndarray | None = None
        self.cloud_center: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.is_centered: bool = False
        self.scale_factor: float = 1.0
        self.normals_flipped: bool = False

    def load(self, filepath: str) -> bool:
        print(f"Loading point cloud from: {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
            return False
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if not pcd.has_points():
                print(f"Error: No points found in {filepath}", file=sys.stderr)
                return False
            self.original_points = np.asarray(pcd.points)
            self.cloud_center = self.original_points.mean(axis=0)
            print(f"Cloud center calculated at: {self.cloud_center}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            return False

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns the min and max corner points of the bounding box."""
        if self.original_points is None or self.original_points.size == 0:
            return None
        min_bound = np.min(self.original_points, axis=0)
        max_bound = np.max(self.original_points, axis=0)
        return min_bound, max_bound

    def compute_normals(self):
        if self.original_points is None:
            print("Error: Points not loaded, cannot compute normals.", file=sys.stderr)
            return
        print(f"Computing normals with params: k={self.k_neighbors}, radius={self.radius}, hybrid={self.hybrid_tree}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.original_points)
        if self.hybrid_tree:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.k_neighbors)
        else:
            search_param = o3d.geometry.KDTreeSearchParamKNN(knn=self.k_neighbors)
        pcd.estimate_normals(search_param=search_param)
        if self.align_with_viewpoint:
            pcd.orient_normals_to_align_with_direction(orientation_reference=self.viewpoint)
        else:
            pcd.orient_normals_consistent_tangent_plane(self.k_neighbors)
        self.normals = np.asarray(pcd.normals)
        if self.normals_flipped:
            self.normals *= -1
        print("Normals computed and oriented.")

    def flip_normals(self):
        if self.normals is not None:
            self.normals *= -1
            self.normals_flipped = not self.normals_flipped
            print("Normals flipped.")

    def get_transformed_points(self) -> np.ndarray:
        if self.original_points is None:
            return np.array([])
        points = np.copy(self.original_points)
        if self.is_centered:
            points -= self.cloud_center
        return points * self.scale_factor

    def save(self, output_path: str, metadata_path: str):
        print("Saving final results...")
        if self.original_points is None or self.normals is None:
            print("Error: No data to save.", file=sys.stderr)
            return
        transformed_points = self.get_transformed_points()
        self._write_ply(output_path, transformed_points, self.normals)
        self._save_metadata(output_path, metadata_path)
    
    @staticmethod
    def _write_ply(filepath: str, points: np.ndarray, normals: np.ndarray):
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(points)
        pcd_out.normals = o3d.utility.Vector3dVector(normals)
        try:
            o3d.io.write_point_cloud(filepath, pcd_out, write_ascii=True)
            print(f"Successfully saved point cloud to: {filepath}")
        except Exception as e:
            print(f"Error saving file to {filepath}: {e}", file=sys.stderr)

    def _save_metadata(self, output_file: str, metadata_path: str):
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