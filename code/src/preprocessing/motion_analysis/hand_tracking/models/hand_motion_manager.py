import numpy as np
import os
import trimesh
import open3d as o3d
from typing import Optional, List, Tuple, Union

class HandMotionManager:
    """
    Domain entity for 3D Hand Motion data.

    Roles:
    1. Builder: Accumulates frames, calculates rigid alignment, and manages vertex history.
    2. Container: Stores motion data and allows indexed access to transformed meshes.
    3. Serializer: Handles Native NumPy (.npz) I/O for high-performance storage.
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
        timestamp: float
    ) -> None:
        """
        Ingests a single frame of data:
        1. Stores mesh topology (if first frame).
        2. Calculates rigid body transformation based on sticker alignment.
        3. Stores the deformed vertices and the calculated transform.
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
            trans, rot_xyzw = self._calculate_alignment(source_points, target_sticker_coords)
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
                # If pure playback is required, keeping them as arrays is faster, 
                # but we stick to the class contract (List[np.ndarray]).
                
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
    # Internal Domain Logic: Math
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

    def _calculate_alignment(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        source_basis = self._create_basis(*source_points[:3])
        target_basis = self._create_basis(*target_points[:3])
        rotation_mat = target_basis @ source_basis.T
        
        source_origin = source_points[0]
        target_origin = target_points[0]
        rotated_source_origin = rotation_mat @ source_origin
        translation_vec = target_origin - rotated_source_origin
        
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_mat
        transform_matrix[0:3, 3] = translation_vec
        
        _, _, angles, trans, _ = trimesh.transformations.decompose_matrix(transform_matrix)
        quat_wxyz = trimesh.transformations.quaternion_from_euler(*angles)
        
        # Return translation and Quaternion (x, y, z, w)
        return trans.astype(np.float32), np.roll(quat_wxyz, -1).astype(np.float32)