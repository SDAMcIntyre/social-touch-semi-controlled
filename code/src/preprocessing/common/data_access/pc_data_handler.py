import open3d as o3d
import pathlib
from typing import Optional

class PointCloudDataHandler:
    """
    A utility class for handling data operations (saving and loading)
    for Open3D point clouds.

    This class uses static methods as the operations are stateless.
    """

    @staticmethod
    def save(point_cloud: o3d.geometry.PointCloud, output_path: str):
        """
        Saves an Open3D point cloud to a file if it contains points.
        This method creates the parent directory if it does not exist.

        Args:
            point_cloud (o3d.geometry.PointCloud): The point cloud object to save.
            output_path (str): The path where the point cloud file will be saved.
                               The file format is inferred from the extension (e.g., .ply, .pcd).
        """
        if not point_cloud or not point_cloud.has_points():
            print("⚠️ WARNING: Point cloud is empty. Nothing to save.")
            return

        output_file = pathlib.Path(output_path)
        # Ensure the parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            o3d.io.write_point_cloud(str(output_file), point_cloud)
            print(f"✅ Saved point cloud to {output_file}")
        except Exception as e:
            print(f"❌ ERROR: Failed to save point cloud to {output_file}. Reason: {e}")

    @staticmethod
    def load(input_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Loads a point cloud from a file.

        Args:
            input_path (str): The path to the point cloud file.

        Returns:
            Optional[o3d.geometry.PointCloud]: The loaded point cloud object,
                                                or None if the file cannot be found
                                                or read.
        """
        input_file = pathlib.Path(input_path)

        if not input_file.exists():
            print(f"❌ ERROR: File not found at {input_file}")
            return None

        try:
            point_cloud = o3d.io.read_point_cloud(str(input_file))
            if not point_cloud.has_points():
                print(f"⚠️ WARNING: Loaded point cloud from {input_file} is empty.")
            else:
                print(f"✅ Loaded point cloud from {input_file}")
            return point_cloud
        except Exception as e:
            print(f"❌ ERROR: Failed to load point cloud from {input_file}. Reason: {e}")
            return None

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a sample Open3D point cloud
    #    (This is just for demonstration purposes)
    import numpy as np
    
    print("--- Creating a sample point cloud ---")
    points = np.random.rand(100, 3)  # 100 points in 3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Sample point cloud created with {len(pcd.points)} points.")

    # 2. Define a file path for saving and loading
    file_path = "temp/sample_cloud.ply"

    # 3. Use the handler to save the point cloud
    print("\n--- Testing the save method ---")
    PointCloudDataHandler.save(pcd, file_path)
    
    # 4. Use the handler to load the point cloud
    print("\n--- Testing the load method ---")
    loaded_pcd = PointCloudDataHandler.load(file_path)

    if loaded_pcd:
        print(f"Successfully loaded point cloud with {len(loaded_pcd.points)} points.")
        # Optional: Visualize the loaded point cloud
        # o3d.visualization.draw_geometries([loaded_pcd])

    # --- Test edge cases ---
    print("\n--- Testing edge cases ---")
    # Attempt to save an empty point cloud
    empty_pcd = o3d.geometry.PointCloud()
    PointCloudDataHandler.save(empty_pcd, "temp/empty.ply")

    # Attempt to load a non-existent file
    PointCloudDataHandler.load("temp/non_existent_file.ply")