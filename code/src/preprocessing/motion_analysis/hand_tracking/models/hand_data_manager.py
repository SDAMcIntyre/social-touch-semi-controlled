import pickle
import bisect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class HandTrackingDataManager:
    """
    Manages the loading and extraction of kinematic analysis data.
    Includes pre-calculation of global spatial bounds for visualization stability.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: List[Dict[str, Any]] = []
        self.global_max_range: float = 1.0  # Default fallback
        self._load_data()
        self._calculate_global_bounds()

    def _load_data(self):
        """Loads the heavy pickle file into memory."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        logging.info(f"Loading heavy data file: {self.file_path} (This may take a moment)...")
        try:
            with open(self.file_path, 'rb') as f:
                self.data = pickle.load(f)
            logging.info(f"Successfully loaded {len(self.data)} data frames.")
        except Exception as e:
            logging.error(f"Failed to load pickle file: {e}")
            raise

    def _calculate_global_bounds(self):
        """
        Iterates through all frames to find the maximum spatial extent (bounding box)
        of the hand across the entire recording.
        """
        logging.info("Calculating global spatial bounds...")
        max_extent = 0.0
        valid_frames = 0

        for frame_data in self.data:
            api_resp = frame_data.get('api_response', {})
            if not api_resp or api_resp.get('error'):
                continue

            hands = api_resp.get('hands', [])
            if not hands:
                continue
            
            # We specifically look at planar_z0 as that is what is plotted in 3D
            hand_data = hands[0]
            if 'vertices_planar_z0' in hand_data:
                verts = np.array(hand_data['vertices_planar_z0'], dtype=np.float32)
                if verts.size > 0:
                    ptp = np.ptp(verts, axis=0)  # Peak-to-peak (max - min)
                    current_max = ptp.max()
                    if current_max > max_extent:
                        max_extent = current_max
                    valid_frames += 1

        if max_extent > 0:
            self.global_max_range = max_extent
            logging.info(f"Global max range calculated: {self.global_max_range:.2f}mm based on {valid_frames} valid frames.")
        else:
            logging.warning("No valid 3D geometry found in data to calculate bounds. Defaulting to 1.0.")

    def get_hand_geometry(self, frame_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts 2D vertices (pixels) and faces for a specific frame.
        """
        if frame_index < 0 or frame_index >= len(self.data):
            return None

        frame_data = self.data[frame_index]
        api_resp = frame_data.get('api_response', {})
        if not api_resp or api_resp.get('error'):
            return None

        hands = api_resp.get('hands', [])
        if not hands or not isinstance(hands, list):
            return None

        hand_data = hands[0]
        
        if 'vertices_pixel' not in hand_data or 'faces' not in hand_data:
            return None

        vertices = np.array(hand_data['vertices_pixel'], dtype=np.float32)
        faces = np.array(hand_data['faces'], dtype=np.int32)

        if vertices.size == 0 or faces.size == 0:
            return None

        return vertices, faces

    def get_hand_3d_geometry(self, frame_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts 3D vertices (planar_z0) and faces for a specific frame.
        """
        if frame_index < 0 or frame_index >= len(self.data):
            return None

        frame_data = self.data[frame_index]
        api_resp = frame_data.get('api_response', {})
        if not api_resp or api_resp.get('error'):
            return None

        hands = api_resp.get('hands', [])
        if not hands or not isinstance(hands, list):
            return None

        hand_data = hands[0]
        
        if 'vertices_planar_z0' not in hand_data or 'faces' not in hand_data:
            return None

        vertices = np.array(hand_data['vertices_planar_z0'], dtype=np.float32)
        faces = np.array(hand_data['faces'], dtype=np.int32)

        if vertices.size == 0 or faces.size == 0:
            return None

        return vertices, faces

    def extract_clean_mesh_data(self, frame_index: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Extracts cleaned NumPy arrays for storage without API wrappers.
        Returns dictionary with 'vertices' and 'faces' keys if data exists.
        """
        geometry = self.get_hand_3d_geometry(frame_index)
        if geometry is None:
            return None
        
        vertices, faces = geometry
        return {
            'vertices': vertices,
            'faces': faces
        }


class MeshSequenceLoader:
    """
    Data structure that acts as a read-only array for mesh sequences.
    Implements a Hybrid Fill logic:
    1. Forward-Fill (Sample-and-Hold): If frame N is missing, return frame N-k.
    2. Back-Fill: If frame N is before the start of the sequence, return the first available frame.
    """
    def __init__(self, data_source: Union[str, Path, Dict]):
        self.meshes = {}
        self.max_frames = 0
        self._sorted_keys = []

        # Load data based on input type
        if isinstance(data_source, (str, Path)):
            p = Path(data_source)
            if not p.exists():
                raise FileNotFoundError(f"Mesh file not found: {p}")
            with open(p, 'rb') as f:
                raw_data = pickle.load(f)
                self.meshes = raw_data.get('meshes', {})
                self.max_frames = raw_data.get('max_frames', 0)
        elif isinstance(data_source, dict):
            self.meshes = data_source.get('meshes', {})
            self.max_frames = data_source.get('max_frames', 0)
        else:
            raise TypeError("data_source must be a file path (str/Path) or a dictionary.")

        # Cache sorted keys for O(log N) lookup using bisect
        self._sorted_keys = sorted(self.meshes.keys())

    def __getitem__(self, frame_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve mesh data for the given frame_idx.
        
        Logic:
        1. Exact Match: Return mesh.
        2. Left Neighbor Exists (Forward-Fill): Return closest previous mesh.
        3. No Left Neighbor (Start of Sequence): Return closest right mesh (first available).
        """
        if frame_idx < 0:
            raise IndexError("Frame index cannot be negative.")
            
        if not self._sorted_keys:
            return None

        # 1. Exact match (O(1))
        if frame_idx in self.meshes:
            return self.meshes[frame_idx]

        # 2. Find insertion point (O(log N))
        idx = bisect.bisect_right(self._sorted_keys, frame_idx)
        
        # Case A: No Left Neighbor (frame_idx < start of data)
        # Search on the "right side" (return the first available frame)
        if idx == 0:
            first_key = self._sorted_keys[0]
            return self.meshes[first_key]
        
        # Case B: Left Neighbor Exists (Standard Forward-Fill)
        # Return the mesh corresponding to the key immediately preceding frame_idx
        prev_key = self._sorted_keys[idx - 1]
        return self.meshes[prev_key]

    def __len__(self) -> int:
        return self.max_frames