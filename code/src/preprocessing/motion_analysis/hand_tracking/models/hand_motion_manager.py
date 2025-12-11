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
    1. Builder: Accumulates frames, calculates alignment (Rigid or Scaled), and manages vertex history.
    2. Container: Stores motion data and allows indexed access to transformed meshes.
    3. Serializer: Handles Native NumPy (.npz) I/O for high-performance storage.
    4. Debugger: Visualizes algorithmic alignment steps via Open3D with mesh context.
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
        self.scales: List[float] = []             # [s, ...] Uniform scale factor
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
        alignment_mode: str = "rigid_basis",
        debug: bool = False
    ) -> None:
        """
        Ingests a single frame of data:
        1. Stores mesh topology (if first frame).
        2. Calculates transformation based on sticker alignment using the selected strategy.
        3. Stores the deformed vertices and the calculated transform.

        Args:
            mesh_vertices: (V, 3) array of source vertices.
            mesh_faces: (F, 3) array of face indices.
            target_sticker_coords: (K, 3) array of target points.
            sticker_vertex_indices: List of K indices in mesh_vertices corresponding to stickers.
            timestamp: float time in seconds.
            alignment_mode: Strategy selector. 
                            'rigid_basis' (default) - 3-point rigid alignment.
                            'scaled_procrustes' - All-points least squares alignment with uniform scaling.
            debug: If True, triggers a blocking GUI to visualize alignment steps.
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
        
        # Handle Missing Tracking
        trans = np.zeros(3, dtype=np.float32)
        rot_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)
        scale = 1.0

        if np.any(np.isnan(target_sticker_coords)):
            if len(self.translations) > 0:
                trans = self.translations[-1]
                rot_xyzw = self.rotations[-1]
                scale = self.scales[-1]
            # Else defaults remain
        else:
            # Dispatch Alignment Strategy
            if alignment_mode == "scaled_procrustes":
                trans, rot_xyzw, scale = self._calculate_alignment_procrustes(
                    source_points, target_sticker_coords
                )
            else:
                # Default to rigid basis
                trans, rot_xyzw, scale = self._calculate_alignment_basis(
                    source_points, target_sticker_coords
                )

        self.translations.append(trans)
        self.rotations.append(rot_xyzw)
        self.scales.append(scale)

        if debug:
            self._visualize_mesh_alignment(
                source_verts=clean_verts,
                target_points=target_sticker_coords,
                anchor_indices=sticker_vertex_indices,
                trans=trans,
                rot_xyzw=rot_xyzw,
                scale=scale,
                mode_name=alignment_mode
            )

    def save(self, output_path: str) -> None:
        """
        Serializes the internal state to a compressed NumPy archive (.npz).
        """
        if not self.vertices_sequence:
            raise ValueError("No data to save. Use process_frame first.")
        
        print(f"💾 Saving {len(self.vertices_sequence)} frames to {output_path}...")
        
        vertices_arr = np.array(self.vertices_sequence, dtype=np.float32)
        trans_arr = np.array(self.translations, dtype=np.float32)
        rot_arr = np.array(self.rotations, dtype=np.float32)
        scale_arr = np.array(self.scales, dtype=np.float32)
        time_arr = np.array(self.timestamps, dtype=np.float64)
        
        save_dict = {
            "vertices": vertices_arr,
            "translations": trans_arr,
            "rotations": rot_arr,
            "scales": scale_arr,
            "timestamps": time_arr,
            "fps": self.fps
        }

        if self.faces is not None:
            save_dict["faces"] = self.faces.astype(np.uint32)

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
                if "faces" in data:
                    self.faces = data["faces"]
                else:
                    self.faces = None

                verts_packed = data["vertices"]
                self.vertices_sequence = [v for v in verts_packed]
                
                self.translations = [t for t in data["translations"]]
                self.rotations = [r for r in data["rotations"]]
                self.timestamps = data["timestamps"].tolist()
                
                # Load scales if present, else default to 1.0
                if "scales" in data:
                    self.scales = [s for s in data["scales"]]
                else:
                    self.scales = [1.0] * len(self.timestamps)
                
                if "fps" in data:
                    self.fps = float(data["fps"])

            # Validation
            n_verts = len(self.vertices_sequence)
            n_times = len(self.timestamps)
            
            if n_verts != n_times:
                print(f"⚠️  CRITICAL MISMATCH: {n_verts} mesh vs {n_times} times.")
                min_len = min(n_verts, n_times)
                self.vertices_sequence = self.vertices_sequence[:min_len]
                self.timestamps = self.timestamps[:min_len]
                self.translations = self.translations[:min_len]
                self.rotations = self.rotations[:min_len]
                self.scales = self.scales[:min_len]

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
        scale = self.scales[index]

        # Construct Transform Matrix
        # M = T * R * S
        matrix = trimesh.transformations.quaternion_matrix(np.roll(rot, 1)) # to w, x, y, z
        
        # Inject Uniform Scale into the 3x3 rotation block
        matrix[:3, :3] *= scale
        
        # Set Translation
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

    def _calculate_alignment_basis(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Original Strategy: Rigid alignment based on the first 3 points.
        Returns: (Translation, Rotation_XYZW, Scale=1.0)
        """
        # 1. Basis Calculation
        source_basis = self._create_basis(*source_points[:3])
        target_basis = self._create_basis(*target_points[:3])
        
        # 2. Rotation
        rotation_mat = target_basis @ source_basis.T
        
        # 3. Translation
        source_origin = source_points[0]
        target_origin = target_points[0]
        rotated_source_origin = rotation_mat @ source_origin
        translation_vec = target_origin - rotated_source_origin
        
        # 4. Decompose to Quat
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_mat
        transform_matrix[0:3, 3] = translation_vec
        
        _, _, angles, trans, _ = trimesh.transformations.decompose_matrix(transform_matrix)
        quat_wxyz = trimesh.transformations.quaternion_from_euler(*angles)
        
        return trans.astype(np.float32), np.roll(quat_wxyz, -1).astype(np.float32), 1.0

    def _calculate_alignment_procrustes(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        New Strategy: Scaled Procrustes (Least Squares) alignment using ALL points.
        Constraint: Keeps source_origin (Index 0) mapped to target_origin (Index 0).
        Returns: (Translation, Rotation_XYZW, Scale)
        """
        # 1. Shift to Origin (Fix Point 0)
        s0 = source_points[0]
        t0 = target_points[0]
        
        P = source_points - s0  # Source shifted
        Q = target_points - t0  # Target shifted

        # 2. Compute Rotation (Kabsch / SVD)
        # H = Covariance matrix
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        
        # R = V @ U.T
        R = Vt.T @ U.T
        
        # Handle Reflection case (det < 0)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 3. Compute Uniform Scale
        P_rotated = P @ R.T # (N, 3)
        
        numerator = np.sum(P_rotated * Q)
        denominator = np.sum(P * P)
        
        scale = numerator / (denominator + 1e-8)
        
        # 4. Compute Translation
        # t = T0 - s * R * S0
        translation_vec = t0 - (scale * (R @ s0))

        # 5. Convert to format
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = R
        transform_matrix[0:3, 3] = translation_vec
        
        _, _, angles, _, _ = trimesh.transformations.decompose_matrix(transform_matrix)
        quat_wxyz = trimesh.transformations.quaternion_from_euler(*angles)
            
        return translation_vec.astype(np.float32), np.roll(quat_wxyz, -1).astype(np.float32), float(scale)
    
    def _visualize_mesh_alignment(
        self,
        source_verts: np.ndarray,
        target_points: np.ndarray,
        anchor_indices: List[int],
        trans: np.ndarray,
        rot_xyzw: np.ndarray,
        scale: float,
        mode_name: str
    ) -> None:
        """
        Advanced Visual Debugger using Mesh context.
        Optimized for White Background visualization.

        Architectural Changes:
        1. Source/Target polygons rendered as volumetric cylinders for thickness control.
        2. Displacement vectors removed.
        3. Source Anchor spheres scaled 10x.
        4. Result Anchor spheres and connectivity added (Red).
        """
        print(f"\n⚡ DEBUG: Visualizing '{mode_name}' | Scale: {scale:.3f}")
        
        geoms = []

        # --- Helper: Cylindrical Line Generation ---
        # Generates a 3D cylinder to simulate a "thick line" between two points.
        # Uses Trimesh for robust alignment math, then converts to Open3D.
        def _create_thick_line(p1: np.ndarray, p2: np.ndarray, radius: float, color: list) -> o3d.geometry.TriangleMesh:
            # 1. Validation: Check for degenerate segments (zero length)
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 1e-5: # Skip if points are too close
                return None
            
            try:
                # 2. Generation: Create cylinder via trimesh
                # segment=[p1, p2] creates a cylinder running from p1 to p2
                t_cyl = trimesh.creation.cylinder(radius=radius, segment=[p1, p2])
                
                # 3. Conversion: Trimesh -> Open3D
                o_cyl = o3d.geometry.TriangleMesh()
                o_cyl.vertices = o3d.utility.Vector3dVector(t_cyl.vertices)
                o_cyl.triangles = o3d.utility.Vector3iVector(t_cyl.faces)
                o_cyl.compute_vertex_normals()
                o_cyl.paint_uniform_color(color)
                return o_cyl
                
            except (ValueError, np.linalg.LinAlgError) as e:
                # Swallow geometry errors to prevent debugger crash
                # Common cause: Alignment matrix SVD failure on edge cases
                return None

        # --- 1. Construct Transformation Matrix ---
        # Assuming input rot_xyzw is [x, y, z, w], rolling 1 gives [w, x, y, z]
        matrix = trimesh.transformations.quaternion_matrix(np.roll(rot_xyzw, 1)) 
        matrix[:3, :3] *= scale
        matrix[:3, 3] = trans

        # --- 2. Mesh Visualization (Wireframes) ---
        # A. Source Mesh (Initial State)
        src_mesh = o3d.geometry.TriangleMesh()
        src_mesh.vertices = o3d.utility.Vector3dVector(source_verts.astype(np.float64))
        if self.faces is not None:
            src_mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
        
        # Wireframe: Dark Charcoal for visibility on White
        src_lines = o3d.geometry.LineSet.create_from_triangle_mesh(src_mesh)
        src_lines.paint_uniform_color([0.2, 0.2, 0.2]) 
        geoms.append(src_lines)

        # B. Transformed Mesh (Result State)
        res_mesh = copy.deepcopy(src_mesh)
        res_mesh.transform(matrix)
        
        # Wireframe: Deep Blue
        res_lines = o3d.geometry.LineSet.create_from_triangle_mesh(res_mesh)
        res_lines.paint_uniform_color([0.0, 0.3, 0.8]) 
        geoms.append(res_lines)

        # Define visual properties for "Thick Lines"
        # Cylinder radius: 0.02 * scale provides a distinct volumetric look compared to wireframe
        thick_line_radius = 0.5 * scale 

        # --- 3. Result Anchors Analysis (New) ---
        # Extract transformed vertices to visualize anchors on the result mesh
        res_verts = np.asarray(res_mesh.vertices)
        result_anchors = res_verts[anchor_indices]

        # A. Highlight Result Anchors (Red Spheres)
        for ra in result_anchors:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0 * scale)
            sphere.paint_uniform_color([0.8, 0.0, 0.0]) # Deep Red
            sphere.compute_vertex_normals()
            sphere.translate(ra)
            geoms.append(sphere)

        # B. Result Anchor Connectivity (Red Cylinders)
        if len(result_anchors) > 1:
            for i in range(len(result_anchors)):
                p1 = result_anchors[i]
                p2 = result_anchors[(i + 1) % len(result_anchors)] # Wrap around
                
                # Using Deep Red [0.8, 0.0, 0.0] as requested
                cyl = _create_thick_line(p1, p2, radius=thick_line_radius, color=[0.8, 0.0, 0.0])
                if cyl:
                    geoms.append(cyl)

        # --- 4. Target Visualization ---
        # A. Target Points (Dark Green Spheres)
        for tp in target_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0 * scale)
            sphere.paint_uniform_color([0.0, 0.6, 0.0]) 
            sphere.compute_vertex_normals()
            sphere.translate(tp)
            geoms.append(sphere)

        # B. Target Geometry (Polygon connection connecting targets)
        # Requirement: 2x Thicker -> Implemented as Cylinders
        if len(target_points) > 1:
            for i in range(len(target_points)):
                p1 = target_points[i]
                p2 = target_points[(i + 1) % len(target_points)] # Wrap around to close loop
                
                cyl = _create_thick_line(p1, p2, radius=thick_line_radius, color=[0.0, 0.6, 0.0])
                if cyl:
                    geoms.append(cyl)

        # --- 5. Source Anchor Analysis ---
        source_anchors = source_verts[anchor_indices]
        
        # A. Highlight Source Anchors (Deep Red Spheres)
        for sa in source_anchors:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0 * scale) # Updated Radius
            sphere.paint_uniform_color([0.8, 0.0, 0.0]) # Deep Red
            sphere.compute_vertex_normals()
            sphere.translate(sa)
            geoms.append(sphere)

        # B. Anchor Connectivity (Source Anchor Polygon)
        # Requirement: Draw lines between anchor_indices instead of displacement lines
        # Requirement: 2x Thicker -> Implemented as Cylinders
        if len(source_anchors) > 1:
            for i in range(len(source_anchors)):
                p1 = source_anchors[i]
                p2 = source_anchors[(i + 1) % len(source_anchors)] # Wrap around
                
                # Using Magenta [0.8, 0.0, 0.8] to distinguish from Result Anchors
                cyl = _create_thick_line(p1, p2, radius=thick_line_radius, color=[0.8, 0.0, 0.8])
                if cyl:
                    geoms.append(cyl)

        # --- 6. Render Configuration (White Background & Thicker Mesh Lines) ---
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Debug: {mode_name}", 
            width=1024, 
            height=768, 
            left=50, 
            top=50
        )
        
        for geom in geoms:
            vis.add_geometry(geom)
            
        # Access RenderOptions to modify global view settings
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0]) # Pure White Background
        opt.line_width = 3.0       # Base thickness for wireframes
        opt.point_size = 5.0       # Larger points for better visibility
        opt.show_coordinate_frame = True
        
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    print("🧪 Running HandMotionManager Integration Test...")
    
    # 1. Setup Mock Data
    manager = HandMotionManager(fps=30.0)
    
    # Create a simple pyramid/triangle mesh
    # Vertex 0 is at origin
    initial_verts = np.array([
        [0.0, 0.0, 0.0],  # V0 (Origin)
        [1.0, 0.0, 0.0],  # V1
        [0.0, 1.0, 0.0],  # V2
        [0.0, 0.0, 1.0],  # V3 (Tip)
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])
    
    # Target: Translated by (5, 5, 5), Scaled by 2.0
    target_stickers = np.array([
        [5.0, 5.0, 5.0],  # V0 target
        [7.0, 5.0, 5.0],  # V1 target (x+2)
        [5.0, 7.0, 5.0],  # V2 target (y+2)
    ], dtype=np.float32)
    
    indices = [0, 1, 2] # We only track the base triangle, V3 is untracked

    # 2. Test Scaled Procrustes with Mesh Viz
    print("\n--- Test: Scaled Procrustes Alignment ---")
    print("Expectation: Gray wireframe at origin. Cyan wireframe at (5,5,5) scaled x2.")
    print("Yellow lines connect origin->target.")
    
    manager.process_frame(
        initial_verts, faces, target_stickers, indices, 
        timestamp=0.033, alignment_mode="scaled_procrustes", debug=True
    )
    
    print("\n✅ Integration Test Complete.")