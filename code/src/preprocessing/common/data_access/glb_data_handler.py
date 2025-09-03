import numpy as np
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Accessor, Buffer,
    BufferView, Asset, Animation, AnimationChannel, AnimationChannelTarget,
    AnimationSampler
)
import io
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import io
from typing import Optional

import numpy as np
from pygltflib import (
    GLTF2,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    Mesh,
    Node,
    Primitive,
    Scene,
)

# GLTF constants for clarity and type mapping
GL_ELEMENT_ARRAY_BUFFER = 34963
GL_ARRAY_BUFFER = 34962
GL_UNSIGNED_SHORT = 5123
GL_UNSIGNED_INT = 5125
GL_FLOAT = 5126

# Primitive rendering modes
MODE_POINTS = 0
MODE_TRIANGLES = 4

# Mapping from GLTF component types to numpy dtypes
COMPONENT_TYPE_MAP = {
    GL_UNSIGNED_SHORT: np.uint16,
    GL_UNSIGNED_INT: np.uint32,
    GL_FLOAT: np.float32,
}

# Mapping from GLTF type strings to number of components
TYPE_COMPONENT_MAP = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


class GLBDataHandler:
    """
    A class to handle saving and loading of 3D data to and from the binary
    GLTF format (.glb).

    It can handle an animated mesh (vertices, faces, and animation) and an
    optional static point cloud, combining them into a single file.
    """

    def __init__(self):
        """Initializes the handler with empty data fields."""
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.time_points: Optional[np.ndarray] = None
        self.translations: Optional[np.ndarray] = None
        self.rotations: Optional[np.ndarray] = None
        self.point_cloud: Optional[np.ndarray] = None

    def _decode_accessor(
        self, gltf: GLTF2, accessor_index: int, binary_blob: bytes
    ) -> np.ndarray:
        """Decodes binary data from a GLTF accessor into a numpy array."""
        accessor = gltf.accessors[accessor_index]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        dtype = COMPONENT_TYPE_MAP.get(accessor.componentType)
        num_components = TYPE_COMPONENT_MAP.get(accessor.type)

        start_byte = buffer_view.byteOffset + accessor.byteOffset
        byte_length = accessor.count * num_components * np.dtype(dtype).itemsize
        end_byte = start_byte + byte_length

        data = np.frombuffer(binary_blob[start_byte:end_byte], dtype=dtype)
        if num_components > 1:
            return data.reshape(-1, num_components)
        return data
    
    def _add_data_chunk(
        self,
        gltf: GLTF2,
        buffer_stream: io.BytesIO,
        data: np.ndarray,
        gltf_type: str,
        component_type: int,
        buffer_target: Optional[int] = None,
    ) -> int:
        """
        Writes a numpy array to the binary buffer and creates the corresponding
        BufferView and Accessor in the GLTF structure.
        """
        # Ensure data is contiguous in memory for tobytes() to work correctly
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
            
        byte_offset = buffer_stream.tell()
        buffer_stream.write(data.tobytes())
        byte_length = data.nbytes

        buffer_view = BufferView(
            buffer=0, byteOffset=byte_offset, byteLength=byte_length
        )
        if buffer_target:
            buffer_view.target = buffer_target
        gltf.bufferViews.append(buffer_view)
        buffer_view_idx = len(gltf.bufferViews) - 1

        count = data.size if gltf_type == "SCALAR" else len(data)

        if count == 0:
            # pygltflib doesn't require max/min, so we can omit for empty arrays
            min_val = None
            max_val = None
        else:
            if gltf_type == "SCALAR":
                max_val = [float(np.max(data))]
                min_val = [float(np.min(data))]
            else:
                max_val = np.max(data, axis=0).tolist()
                min_val = np.min(data, axis=0).tolist()

        accessor = Accessor(
            bufferView=buffer_view_idx,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=gltf_type,
            max=max_val,
            min=min_val,
        )
        gltf.accessors.append(accessor)
        return len(gltf.accessors) - 1


    def save(
        self,
        output_path: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        time_points: np.ndarray,
        translations: np.ndarray,
        rotations: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
    ):
        """
        Creates and saves a GLB file with an animated mesh and an optional
        point cloud.
        """
        gltf = GLTF2(
            scene=0,
            nodes=[],
            meshes=[],
            animations=[],
            accessors=[],
            bufferViews=[],
            buffers=[Buffer()],
            asset=Asset(version="2.0"),
        )
        buffer_stream = io.BytesIO()

        # --- Add all data chunks to the buffer first ---
        vertex_idx = self._add_data_chunk(
            gltf, buffer_stream, vertices.astype(np.float32), "VEC3", GL_FLOAT, GL_ARRAY_BUFFER
        )
        face_idx = self._add_data_chunk(
            gltf, buffer_stream, faces.astype(np.uint16), "SCALAR", GL_UNSIGNED_SHORT, GL_ELEMENT_ARRAY_BUFFER
        )
        time_idx = self._add_data_chunk(
            gltf, buffer_stream, time_points.astype(np.float32), "SCALAR", GL_FLOAT
        )
        translation_idx = self._add_data_chunk(
            gltf, buffer_stream, translations.astype(np.float32), "VEC3", GL_FLOAT
        )
        rotation_idx = self._add_data_chunk(
            gltf, buffer_stream, rotations.astype(np.float32), "VEC4", GL_FLOAT
        )

        # 1. Always create the animated mesh and its node
        animated_mesh = Mesh(
            primitives=[
                Primitive(
                    attributes=Attributes(POSITION=vertex_idx),
                    indices=face_idx,
                    mode=MODE_TRIANGLES,
                )
            ]
        )
        gltf.meshes.append(animated_mesh)
        animated_node = Node(mesh=0)  # Mesh index is 0
        gltf.nodes.append(animated_node)
        
        scene_node_indices = [0]  # Start scene with the animated node (index 0)

        # 2. If a point cloud exists, add it as a new, *indexed* mesh and node
        if point_cloud is not None and point_cloud.size > 0:
            pc_vertex_idx = self._add_data_chunk(
                gltf, buffer_stream, point_cloud.astype(np.float32), "VEC3", GL_FLOAT, GL_ARRAY_BUFFER
            )
            
            # --- CHANGE 1: Create a sequential index buffer for the point cloud ---
            num_points = len(point_cloud)
            pc_indices = np.arange(num_points, dtype=np.uint32)
            
            # --- CHANGE 2: Add the new index buffer to the GLB data ---
            pc_indices_idx = self._add_data_chunk(
                gltf, buffer_stream, pc_indices, "SCALAR", GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER
            )

            pc_mesh = Mesh(
                primitives=[
                    Primitive(
                        attributes=Attributes(POSITION=pc_vertex_idx),
                        # --- CHANGE 3: Add the indices accessor to the primitive ---
                        indices=pc_indices_idx,
                        mode=MODE_POINTS
                    )
                ]
            )
            gltf.meshes.append(pc_mesh)
            pc_node = Node(mesh=len(gltf.meshes) - 1) # Use dynamic index for robustness
            gltf.nodes.append(pc_node)
            
            scene_node_indices.append(len(gltf.nodes) - 1) # Add new node to scene

        # 3. Define the main scene with all required nodes
        gltf.scenes = [Scene(nodes=scene_node_indices)]

        # 4. Define animation, always targeting the first node (the animated mesh)
        if len(time_points) > 0:
            animation = Animation(
                samplers=[
                    AnimationSampler(
                        input=time_idx, output=translation_idx, interpolation="LINEAR"
                    ),
                    AnimationSampler(
                        input=time_idx, output=rotation_idx, interpolation="LINEAR"
                    ),
                ],
                channels=[
                    AnimationChannel(
                        sampler=0, target=AnimationChannelTarget(node=0, path="translation")
                    ),
                    AnimationChannel(
                        sampler=1, target=AnimationChannelTarget(node=0, path="rotation")
                    ),
                ],
            )
            gltf.animations.append(animation)

        # --- Finalize buffer and save the file ---
        gltf.buffers[0].byteLength = buffer_stream.tell()
        gltf.set_binary_blob(buffer_stream.getvalue())
        buffer_stream.close()
        gltf.save(output_path)
        print(f"✅ GLB file saved to: {output_path}")

    def load(self, input_path: str):
        """
        Loads data from a GLB file into the handler instance.
        Detects and loads an animated mesh and an optional static point cloud.

        Args:
            input_path (str): The path to the .glb file to load.
        """
        gltf = GLTF2.load(input_path)
        binary_blob = gltf.binary_blob()

        # 1. Load primary animated mesh (always expected to be mesh 0)
        if not gltf.meshes:
            raise ValueError("GLB file contains no meshes.")
        primitive = gltf.meshes[0].primitives[0]
        self.vertices = self._decode_accessor(gltf, primitive.attributes.POSITION, binary_blob)
        
        # Decode the flat list of indices from the buffer
        flat_indices = self._decode_accessor(gltf, primitive.indices, binary_blob)
        if primitive.mode == MODE_TRIANGLES:
            # The `-1` tells NumPy to automatically calculate the number of rows
            self.faces = flat_indices.reshape(-1, 3)
        else:
            # If not triangles, store them as is or handle other modes (lines, etc.)
            self.faces = flat_indices


        # 2. Load animation data
        animation = gltf.animations[0]
        for channel in animation.channels:
            sampler = animation.samplers[channel.sampler]
            self.time_points = self._decode_accessor(gltf, sampler.input, binary_blob)
            if channel.target.path == "translation":
                self.translations = self._decode_accessor(gltf, sampler.output, binary_blob)
            elif channel.target.path == "rotation":
                self.rotations = self._decode_accessor(gltf, sampler.output, binary_blob)

        # 3. Check for and load an optional point cloud (expected to be mesh 1)
        if len(gltf.meshes) > 1:
            point_cloud_primitive = gltf.meshes[1].primitives[0]
            # Verify it's a point cloud by checking the mode
            if point_cloud_primitive.mode == MODE_POINTS:
                self.point_cloud = self._decode_accessor(gltf, point_cloud_primitive.attributes.POSITION, binary_blob)
                print("✅ Optional point cloud found and loaded.")

        print(f"✅ GLB file loaded successfully from: {input_path}")
        
    def get_data(self) -> Dict[str, Optional[np.ndarray]]:
        """Returns all loaded data as a dictionary."""
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "time_points": self.time_points,
            "translations": self.translations,
            "rotations": self.rotations,
            "point_cloud": self.point_cloud,
        }





def _print_mesh_stats(vertices: np.ndarray, faces: np.ndarray):
    """Prints useful statistical information about the mesh data."""
    print("\n--- Mesh Statistics ---")
    print("Vertices:")
    print(f"  array shape: {vertices.shape}")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Min vertex values (x, y, z): {np.min(vertices, axis=0)}")
    print(f"  Max vertex values (x, y, z): {np.max(vertices, axis=0)}")
    print(f"  Data type of vertices: {vertices.dtype}")
    print("\nFaces:")
    print(f"  array shape: {faces.shape}")
    print(f"  Number of faces (triangles): {len(faces)}")
    print(f"  Number of unique vertices referenced by faces: {len(np.unique(faces))}")
    print(f"  Min index value used: {np.min(faces)}")
    print(f"  Max index value used: {np.max(faces)}")
    print(f"  Data type of faces: {faces.dtype}")
    print("-----------------------\n")





if __name__ == "__main__":
    output_filename = "generated_animated_model_refactored.glb"

    handmesh_vertices_path = Path("F:/GitHub/social-touch-semi-controlled/singleFingerVertices0.txt")

    if os.path.exists(handmesh_vertices_path):
        from ..hand_tracking.hand_mesh_processor import HandMeshProcessor
        # This part would need the actual HandMeshProcessor class
        print("File found! (This part is not runnable without the class)")
        parameters = {'left': False}
        handmesh = HandMeshProcessor.create_mesh(handmesh_vertices_path, parameters, SCALE_M_TO_MM=True)
        example_vertices = np.asarray(handmesh.vertices)
        example_faces = np.asarray(handmesh.triangles)
    else:
        # Mock HandMeshProcessor for demonstration if not available
        class MockHandMeshProcessor:
            class MockMesh:
                def __init__(self, vertices, triangles):
                    self.vertices = vertices
                    self.triangles = triangles
            @staticmethod
            def create_mesh(*args, **kwargs):
                # This mock will create a simple tetrahedron
                verts = np.array([
                    [1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]
                ], dtype=np.float64)
                tris = np.array([
                    [0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]
                ], dtype=np.int32)
                return MockHandMeshProcessor.MockMesh(verts, tris)
        
        print("File not found. Using default mock shape (tetrahedron).")
        handmesh = MockHandMeshProcessor.create_mesh()
        example_vertices = np.asarray(handmesh.vertices)
        example_faces = np.asarray(handmesh.triangles)

    _print_mesh_stats(example_vertices, example_faces)

    # Define animation data
    example_time_points = np.array([0.0, 1.5, 3.0], dtype=np.float32)

    example_translations = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    example_rotations = np.array([
        [0.0, 0.0, 0.0, 1.0],  # 0 degrees
        [0.0, 0.707, 0.0, 0.707],  # 90 degrees around Y
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    print("\n--- 2. Saving Data to GLB File ---")
    saver = GLBDataHandler()
    saver.save(
        output_path=output_filename,
        vertices=example_vertices,
        faces=example_faces,
        time_points=example_time_points,
        translations=example_translations,
        rotations=example_rotations
    )
    # --- 3. Load Data from GLB ---
    print("\n--- 3. Loading Data from GLB File ---")
    loader = GLBDataHandler()
    loader.load(output_filename)
    loaded_data = loader.get_data()

    # --- 4. Verify Loaded Data ---
    print("\n--- 4. Verifying Data Integrity ---")
    try:
        # Using np.testing.assert_allclose for robust float comparison
        print("✅ Vertices:")
        np.testing.assert_allclose(example_vertices, loaded_data["vertices"], rtol=1e-6)
        print("✅ Vertices match.\n")
        
        print("✅ Faces:")
        np.testing.assert_array_equal(example_faces, loaded_data["faces"])
        print("✅ Faces match.\n")
        
        print("✅ Time:")
        np.testing.assert_allclose(example_time_points, loaded_data["time_points"], rtol=1e-6)
        print("✅ Time points match.\n")
        
        print("✅ Translations:")
        np.testing.assert_allclose(example_translations, loaded_data["translations"], rtol=1e-6)
        print("✅ Translations match.\n")

        print("✅ Rotations:")
        np.testing.assert_allclose(example_rotations, loaded_data["rotations"], rtol=1e-6)
        print("✅ Rotations match.\n")
        
        print("\n🎉 Success! All original and loaded data are identical.")
        
    except AssertionError as e:
        print(f"\n❌ Verification Failed: {e}")
    

    # --- 5. Test Case: Saving and Loading WITH a Point Cloud ---
    print("--- Test Case 1: Saving and Loading WITH a Point Cloud ---")
    source_point_cloud = (np.random.rand(50, 3) - 0.5) * 4 # 50 random points
    output_path_with_pc = "test_with_point_cloud.glb"

    handler_with_pc = GLBDataHandler()
    handler_with_pc.save(
        output_path=output_path_with_pc,
        vertices=example_translations,
        faces=example_faces,
        time_points=example_time_points,
        translations=example_translations,
        rotations=example_rotations,
        point_cloud=source_point_cloud # Providing the point cloud data
    )

    loader_with_pc = GLBDataHandler()
    loader_with_pc.load(output_path_with_pc)
    loaded_data_with_pc = loader_with_pc.get_data()

    # Verification
    assert loaded_data_with_pc["point_cloud"] is not None
    np.testing.assert_allclose(loaded_data_with_pc["point_cloud"], source_point_cloud, rtol=1e-6)
    print("✅ Verification successful: Loaded point cloud matches original.\n")
    