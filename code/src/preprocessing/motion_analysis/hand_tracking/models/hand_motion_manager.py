import numpy as np
import os
import trimesh
import open3d as o3d
from typing import Optional, List, Tuple, Union
import copy

class HandMotionManager:
    """
    Domain entity for 3D Hand Motion data.

    Roles:
    1. Builder: Accumulates frames, calculates rigid alignment, and manages vertex history.
    2. Container: Stores motion data and allows indexed access to transformed meshes.
    3. Serializer: Handles Native NumPy (.npz) I/O for high-performance storage.
    4. Debugger: Visualizes algorithmic alignment steps via Open3D.
    """

    def __init__(self, fps: float = 30.0):
        # State containers
        # Stored as list during accumulation for O(1) appends, converted to array on save.
        self.vertices_sequence: List[np.ndarray] = []
        # Invariant: self.faces must be (N, 3) or None.
        self.faces: Optional[np.ndarray] = None
        
        # Motion vectors
        self.translations: List[np.ndarray] = []  # [(3,), ...]
        self.rotations: List[np.ndarray] = []     # [(4,), ...] (x, y, z, w)
        self.timestamps: List[float] = []
        
        self.fps = fps

    # -------------------------------------------------------------------------
    # Public API: Data Generation (Write Mode)
    # -------------------------------------------------------------------------

    def process_frame(
        self, 
        mesh_vertices: np.ndarray, 
        mesh_faces: np.ndarray,
        target_sticker_coords: np.ndarray,
        sticker_vertex_indices: List[int],
        timestamp: float,
        debug: bool = False
    ) -> None:
        """
        Ingests a single frame of data:
        1. Stores mesh topology (if first frame).
        2. Calculates rigid body transformation based on sticker alignment.
        3. Stores the deformed vertices and the calculated transform.

        Args:
            debug (bool): If True, triggers a blocking GUI to visualize alignment steps.
        """ 
        # Store Topology
        if self.faces is None:
            if mesh_faces.ndim == 1:
                self.faces = mesh_faces.reshape(-1, 3)
            else:
                self.faces = mesh_faces
        
        # Store Vertices (Morph Target source)
        clean_verts = np.nan_to_num(mesh_vertices).astype(np.float32)
        self.vertices_sequence.append(clean_verts)
        self.timestamps.append(timestamp)

        # Calculate Alignment
        source_points = clean_verts[sticker_vertex_indices]
        
        if np.any(np.isnan(target_sticker_coords)):
            # Handling missing tracking data: hold last known transform or identity
            if len(self.translations) > 0:
                self.translations.append(self.translations[-1])
                self.rotations.append(self.rotations[-1])
            else:
                self.translations.append(np.zeros(3, dtype=np.float32))
                self.rotations.append(np.array([0, 0, 0, 1], dtype=np.float32))
        else:
            trans, rot_xyzw = self._calculate_alignment(source_points, target_sticker_coords, debug=debug)
            self.translations.append(trans)
            self.rotations.append(rot_xyzw)

    def save(self, output_path: str) -> None:
        """
        Serializes the internal state to a compressed NumPy archive (.npz).
        This replaces the verbose GLB serialization.
        """
        if not self.vertices_sequence:
            raise ValueError("No data to save. Use process_frame first.")
        
        print(f"💾 Saving {len(self.vertices_sequence)} frames to {output_path}...")
        
        # Convert lists to arrays for storage
        # Optimization: Stack creates a contiguous memory block (Frames, Vertices, 3)
        vertices_arr = np.array(self.vertices_sequence, dtype=np.float32)
        trans_arr = np.array(self.translations, dtype=np.float32)
        rot_arr = np.array(self.rotations, dtype=np.float32)
        time_arr = np.array(self.timestamps, dtype=np.float64)
        
        save_dict = {
            "vertices": vertices_arr,
            "translations": trans_arr,
            "rotations": rot_arr,
            "timestamps": time_arr,
            "fps": self.fps
        }

        if self.faces is not None:
            save_dict["faces"] = self.faces.astype(np.uint32)

        # compress=True significantly reduces file size for time-series vertex data
        np.savez_compressed(output_path, **save_dict)
        print(f"✅ Save complete.")

    # -------------------------------------------------------------------------
    # Public API: Playback (Read Mode)
    # -------------------------------------------------------------------------

    def load(self, input_path: str) -> None:
        """
        Loads a .npz file and populates internal state.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} does not exist.")
        
        print(f"📂 Loading from {input_path}...")
        
        try:
            with np.load(input_path) as data:
                # Load topology
                if "faces" in data:
                    self.faces = data["faces"]
                else:
                    self.faces = None

                # Load frame data
                # Converting back to lists allows for further appending if the user desires
                verts_packed = data["vertices"] # (T, V, 3)
                self.vertices_sequence = [v for v in verts_packed]
                
                self.translations = [t for t in data["translations"]]
                self.rotations = [r for r in data["rotations"]]
                self.timestamps = data["timestamps"].tolist()
                
                if "fps" in data:
                    self.fps = float(data["fps"])

            # Validation
            n_verts = len(self.vertices_sequence)
            n_times = len(self.timestamps)
            
            if n_verts != n_times:
                print(f"⚠️  CRITICAL MISMATCH: {n_verts} mesh frames vs {n_times} timestamps.")
                min_len = min(n_verts, n_times)
                self.vertices_sequence = self.vertices_sequence[:min_len]
                self.timestamps = self.timestamps[:min_len]
                self.translations = self.translations[:min_len]
                self.rotations = self.rotations[:min_len]

            print(f"✅ Successfully loaded {len(self.timestamps)} frames.")

        except Exception as e:
            raise RuntimeError(f"Failed to load HandMotionManager data: {e}")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> o3d.geometry.TriangleMesh:
        """
        Returns the fully transformed mesh (World Space) for the given frame index.
        """
        if index >= len(self.vertices_sequence):
            raise IndexError("Frame index out of range.")

        vertices = self.vertices_sequence[index]
        trans = np.array(self.translations[index])
        rot = np.array(self.rotations[index]) # x, y, z, w

        # Quaternions in trimesh/numpy are usually [w, x, y, z], 
        # but common storage often uses [x, y, z, w].
        # The original code rolled by 1 implying input was [x, y, z, w] 
        # and trimesh expects [w, x, y, z].
        matrix = trimesh.transformations.quaternion_matrix(np.roll(rot, 1))
        matrix[:3, 3] = trans

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        
        if self.faces is not None:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
        
        mesh.transform(matrix)
        return mesh

    # -------------------------------------------------------------------------
    # Internal Domain Logic: Math & Visualization
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_basis(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        v1 = p1 - p0
        v2 = p2 - p0
        x_axis = v1 / (np.linalg.norm(v1) + 1e-8)
        z_axis_unnorm = np.cross(x_axis, v2)
        
        if np.linalg.norm(z_axis_unnorm) < 1e-6:
            temp_vec = np.array([0, 1, 0]) if np.allclose(np.abs(x_axis), [1, 0, 0]) else np.array([1, 0, 0])
            z_axis_unnorm = np.cross(x_axis, temp_vec)
            
        z_axis = z_axis_unnorm / (np.linalg.norm(z_axis_unnorm) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    def _calculate_alignment(self, source_points: np.ndarray, target_points: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the rigid transform to align source_points to target_points.
        Includes optional debug visualization.
        """
        # 1. Basis Calculation
        source_basis = self._create_basis(*source_points[:3])
        target_basis = self._create_basis(*target_points[:3])
        
        # 2. Rotation Calculation
        rotation_mat = target_basis @ source_basis.T
        
        # 3. Translation Calculation
        source_origin = source_points[0]
        target_origin = target_points[0]
        rotated_source_origin = rotation_mat @ source_origin
        translation_vec = target_origin - rotated_source_origin
        
        # 4. Matrix Composition
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_mat
        transform_matrix[0:3, 3] = translation_vec
        
        # 5. Decomposition
        _, _, angles, trans, _ = trimesh.transformations.decompose_matrix(transform_matrix)
        quat_wxyz = trimesh.transformations.quaternion_from_euler(*angles)
        
        if debug:
            self._visualize_debug_alignment(
                source_points, 
                target_points, 
                source_basis, 
                target_basis, 
                rotation_mat, 
                translation_vec,
                source_origin,
                target_origin
            )

        # Return translation and Quaternion (x, y, z, w)
        return trans.astype(np.float32), np.roll(quat_wxyz, -1).astype(np.float32)
    
    def _visualize_debug_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_basis: np.ndarray,
        target_basis: np.ndarray,
        rotation_mat: np.ndarray,
        translation_vec: np.ndarray,
        source_origin: np.ndarray,
        target_origin: np.ndarray
    ) -> None:
        """
        Helper method to create a blocking GUI window showing alignment steps.
        Visualizes vertices connected as closed-loop polygons (triangles).
        """
        # Define Legend Schema
        legend_map = {
            "Source": ([1, 0, 0], "Red"),
            "Target": ([0, 1, 0], "Green"),
            "Rotated": ([1, 1, 0], "Yellow"),
            "Final": ([0, 0, 1], "Blue")
        }

        print("\n" + "="*40)
        print("🛠️  DEBUG: ALIGNMENT VISUALIZATION")
        print("="*40)
        print(f"{'GROUP':<15} | {'COLOR':<10}")
        print("-" * 28)
        for name, (_, color_name) in legend_map.items():
            print(f"{name:<15} | {color_name:<10}")
        print("="*40 + "\n")

        geoms = []

        def _create_visual_group(points_array: np.ndarray, color: list) -> list:
            """Factory to create both PointCloud and connecting LineSet (Triangle)."""
            group_geoms = []
            
            # 1. Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_array)
            pcd.paint_uniform_color(color)
            group_geoms.append(pcd)
            
            # 2. LineSet (Connect points to form a loop/triangle)
            num_points = len(points_array)
            if num_points >= 2:
                # Create edges: (0,1), (1,2), ..., (N-1, 0)
                lines_indices = [[i, (i + 1) % num_points] for i in range(num_points)]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points_array)
                line_set.lines = o3d.utility.Vector2iVector(lines_indices)
                line_set.paint_uniform_color(color)
                group_geoms.append(line_set)
                
            return group_geoms

        # 1. Source Points (RED)
        geoms.extend(_create_visual_group(source, legend_map["Source"][0]))

        # 2. Target Points (GREEN)
        geoms.extend(_create_visual_group(target, legend_map["Target"][0]))

        # 3. Rotated Intermediate Source (YELLOW)
        # Math: p_rot = R @ p
        rotated_source_pts = (rotation_mat @ source.T).T
        geoms.extend(_create_visual_group(rotated_source_pts, legend_map["Rotated"][0]))

        # 4. Final Aligned Source (BLUE)
        # Math: Final point = (R @ p) + T
        final_source_pts = rotated_source_pts + translation_vec
        geoms.extend(_create_visual_group(final_source_pts, legend_map["Final"][0]))

        # 5. Basis Visualizations (Coordinate Frames)
        def create_basis_vis(origin, basis_mat, scale=0.05):
            # Create a coordinate frame representing the basis at the specific origin
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0, 0, 0])
            tf = np.eye(4)
            tf[:3, :3] = basis_mat
            tf[:3, 3] = origin
            frame.transform(tf)
            return frame

        geoms.append(create_basis_vis(source_origin, source_basis, scale=0.1)) # Source Basis
        geoms.append(create_basis_vis(target_origin, target_basis, scale=0.1)) # Target Basis

        # Render
        # Note: draw_geometries is blocking. Window title serves as the GUI legend.
        o3d.visualization.draw_geometries(
            geoms, 
            window_name="Legend: Red=Source | Grn=Target | Ylw=Rotated | Blu=Final", 
            width=1024, 
            height=768,
            left=50,
            top=50
        )