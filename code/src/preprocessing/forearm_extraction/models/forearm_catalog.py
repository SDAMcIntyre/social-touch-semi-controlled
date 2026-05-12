import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import open3d as o3d
# from bisect import bisect_left # For a highly optimized search

from .forearm_parameters import ForearmParameters

from preprocessing.common import (
    PointCloudDataHandler
)

# Configure a basic logger instead of using print()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@dataclass(frozen=True)
class VideoIdentifier:
    """A structured representation of a video filename with a block number."""
    prefix: str
    block_number: int
    original_stem: str

    @classmethod
    def from_filename(cls, filename: str) -> Optional['VideoIdentifier']:
        """Parses a filename to extract the block prefix and number."""
        stem = Path(filename).stem
        match = re.match(r"(.*_block-order)(\d+)", stem)
        if match:
            prefix, number_str = match.groups()
            return cls(prefix, int(number_str), stem)
        return None


class ForearmCatalog:
    """
    Manages and provides access to a collection of forearm point clouds and meshes.

    This class preprocesses forearm metadata to allow for efficient lookups
    of both specific video data and fallback references.
    """
    def __init__(self, forearm_params: List[ForearmParameters], pointcloud_dir: Path):
        self._pointcloud_dir = pointcloud_dir
        self._params_by_video: Dict[str, List[ForearmParameters]] = {}
        self._refs_by_prefix: Dict[str, List[Tuple[int, ForearmParameters]]] = {}

        self._build_indexes(forearm_params)

    def _build_indexes(self, forearm_params: List[ForearmParameters]):
        """Processes the raw list of params into efficient lookup dictionaries."""
        for params in forearm_params:
            # Index for direct lookup
            self._params_by_video.setdefault(params.video_filename, []).append(params)
            
            # Index for fallback reference lookup
            identifier = VideoIdentifier.from_filename(params.video_filename)
            if identifier:
                self._refs_by_prefix.setdefault(identifier.prefix, []).append(
                    (identifier.block_number, params)
                )

        # Sort the reference lists by block number for efficient searching
        for prefix in self._refs_by_prefix:
            self._refs_by_prefix[prefix].sort(key=lambda x: x[0])

    def _load_pointcloud(self, params: ForearmParameters) -> Optional[o3d.geometry.PointCloud]:
        """Loads a single point cloud, handling file existence and errors."""
        video_stem = Path(params.video_filename).stem
        stem = params.build_output_stem(video_stem)
        pointcloud_filename = f"{stem}_with_normals.ply"
        path = self._pointcloud_dir / pointcloud_filename
        return PointCloudDataHandler.load(path)

    def _load_mesh(self, params: ForearmParameters) -> Optional[o3d.geometry.TriangleMesh]:
        """Loads a single mesh, handling file existence and errors."""
        video_stem = Path(params.video_filename).stem
        stem = params.build_output_stem(video_stem)
        mesh_filename = f"{stem}_mesh.obj"
        path = self._pointcloud_dir / mesh_filename
        
        if not path.exists():
            return None
            
        try:
            # Use Open3D standard mesh loader
            mesh = o3d.io.read_triangle_mesh(str(path))
            # Basic validation to ensure the mesh isn't empty
            if mesh.is_empty():
                return None
            return mesh
        except Exception as e:
            logging.error(f"Failed to load mesh {path}: {e}")
            return None

    def get_pointclouds_for_video(self, video_filename: str) -> Dict[int, o3d.geometry.PointCloud]:
        """
        Finds and loads all forearm point clouds for the specified video.
        """
        params_for_video = self._params_by_video.get(video_filename, [])
        if not params_for_video:
            return {}
            
        pointclouds = {}
        for params in params_for_video:
            pc = self._load_pointcloud(params)
            if pc:
                pointclouds[params.frame_id] = pc
        
        return pointclouds

    def get_meshes_for_video(self, video_filename: str) -> Dict[int, o3d.geometry.TriangleMesh]:
        """
        Finds and loads all forearm meshes for the specified video.
        """
        params_for_video = self._params_by_video.get(video_filename, [])
        if not params_for_video:
            return {}
            
        meshes = {}
        for params in params_for_video:
            mesh = self._load_mesh(params)
            if mesh:
                meshes[params.frame_id] = mesh
        
        return meshes

    def get_first_pointcloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Retrieves the first available point cloud from the entire catalog.
        """
        for params_list in self._params_by_video.values():
            for params in params_list:
                pc = self._load_pointcloud(params)
                if pc:
                    logging.info(f"Fetched first available point cloud: {params.video_filename} (Frame {params.frame_id})")
                    return pc
        
        logging.warning("Catalog is empty or no point cloud files could be loaded.")
        return None

    def get_first_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Retrieves the first available mesh from the entire catalog.
        """
        for params_list in self._params_by_video.values():
            for params in params_list:
                mesh = self._load_mesh(params)
                if mesh:
                    logging.info(f"Fetched first available mesh: {params.video_filename} (Frame {params.frame_id})")
                    return mesh
        
        logging.warning("Catalog is empty or no mesh files could be loaded.")
        return None

    # ------------------------------------------------------------------
    # Registration-aware accessors
    # ------------------------------------------------------------------

    def get_unified_pointcloud(
        self, session_id: str
    ) -> Optional[o3d.geometry.PointCloud]:
        """Load the unified registered point cloud for *session_id*.

        Returns ``None`` when the file does not exist (single-forearm
        session or registration has not been run yet).
        """
        path = self._pointcloud_dir / f"{session_id}_unified_registered.ply"
        if not path.exists():
            return None
        return PointCloudDataHandler.load(path)

    def load_registration_transforms(
        self, session_id: str
    ) -> Optional[Dict[str, Tuple[np.ndarray, float]]]:
        """Load persisted per-snapshot registration transforms.

        Returns a mapping ``{snapshot_key: (4x4_matrix, fitness)}`` where
        each key is a composite string ``"video_stem:frame_id"``, or
        ``None`` if the transforms file does not exist.
        """
        path = self._pointcloud_dir / f"{session_id}_registration_transforms.json"
        if not path.exists():
            return None

        with open(path) as fh:
            raw = json.load(fh)

        transforms: Dict[str, Tuple[np.ndarray, float]] = {}
        for key, entry in raw["transforms"].items():
            transforms[key] = (
                np.asarray(entry["matrix_4x4"]),
                entry["fitness"],
            )
        return transforms

    def find_closest_reference(
        self, 
        video_filename: str, 
        use_mesh: bool = False
    ) -> Optional[Tuple[int, Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]]]:
        """
        Finds a single reference forearm (PC or Mesh) by looking for the closest block number.

        Args:
            video_filename: The target video filename.
            use_mesh: If True, returns a TriangleMesh; otherwise, returns a PointCloud.
        """
        identifier = VideoIdentifier.from_filename(video_filename)
        if not identifier:
            logging.warning(f"Could not parse block number from '{video_filename}' for reference search.")
            return None

        candidates = self._refs_by_prefix.get(identifier.prefix)
        if not candidates:
            logging.warning(f"No references found with prefix '{identifier.prefix}'.")
            return None

        # Find the candidate with the minimum difference in block number
        closest = min(
            candidates,
            key=lambda x: abs(x[0] - identifier.block_number)
        )
        
        closest_block_num, closest_params = closest
        diff = abs(closest_block_num - identifier.block_number)
        logging.info(f"Found closest reference: '{closest_params.video_filename}' (Block difference: {diff}).")
        
        if use_mesh:
            geometry = self._load_mesh(closest_params)
        else:
            geometry = self._load_pointcloud(closest_params)
            
        if geometry:
            return closest_params.frame_id, geometry
        
        return None
    
    
def get_forearms_with_fallback(
    catalog: ForearmCatalog,
    current_video_filename: str,
    *,
    use_mesh: bool = False,
) -> Dict[int, Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]]:
    """
    Gets forearm geometries for a video based on its block number.

    1. Extracts block number *N* from *current_video_filename*.
    2. Loads every forearm whose video shares that block number, keyed by
       ``representative_frame_id``.
    3. If no entry has ``representative_frame_id == 0``:
       a. Searches blocks *N − 1*, *N − 2*, … down to **1** for the forearm
          with the **highest** ``representative_frame_id`` and inserts it as
          key ``0``.
       b. If no previous block yields a result, the forearm with the lowest
          ``representative_frame_id`` already in the dict is duplicated under
          key ``0`` so that a reference is available from frame 0.

    Args:
        catalog: An initialized ForearmCatalog instance.
        current_video_filename: The filename of the video to process.
        use_mesh: If True, loads meshes; otherwise loads point clouds.

    Returns:
        A dictionary mapping representative frame IDs to Open3D geometry
        objects, with key ``0`` guaranteed when at least one forearm exists.
    """
    identifier = VideoIdentifier.from_filename(current_video_filename)
    if not identifier:
        logging.warning(f"Could not parse block number from '{current_video_filename}'.")
        return {}

    candidates = catalog._refs_by_prefix.get(identifier.prefix)
    if not candidates:
        logging.warning(f"No forearm references found with prefix '{identifier.prefix}'.")
        return {}

    load_fn = catalog._load_mesh if use_mesh else catalog._load_pointcloud

    # -- Step 1: collect all forearms for block N --------------------------
    forearms: Dict[int, Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]] = {}
    for block_num, params in candidates:
        if block_num == identifier.block_number:
            geometry = load_fn(params)
            if geometry:
                forearms[params.representative_frame_id] = geometry

    # -- Step 2: ensure a frame-0 entry exists -----------------------------
    if 0 not in forearms:
        # 2a. Walk backwards through previous blocks (N-1 … 1)
        for prev_block in range(identifier.block_number - 1, 0, -1):
            prev_params_list = [
                params for block_num, params in candidates
                if block_num == prev_block
            ]
            if not prev_params_list:
                continue
            best = max(prev_params_list, key=lambda p: p.representative_frame_id)
            geometry = load_fn(best)
            if geometry:
                logging.info(
                    f"No frame_id 0 in block {identifier.block_number}; "
                    f"using '{best.video_filename}' frame {best.representative_frame_id} "
                    f"from block {prev_block} as fallback."
                )
                forearms[0] = geometry
                logging.warning(
                    "No forearm snapshot found for block %d of '%s'. "
                    "Using forearm from block %d (frame %d) as fallback. "
                    "If the arm repositioned between blocks, capture a new PLY.",
                    identifier.block_number,
                    current_video_filename,
                    prev_block,
                    best.representative_frame_id,
                )
                break

        # 2b. No earlier block had a loadable forearm — duplicate the earliest
        if 0 not in forearms and forearms:
            min_key = min(forearms)
            logging.info(
                f"No previous-block fallback found; duplicating frame {min_key} as frame 0."
            )
            forearms[0] = forearms[min_key]

    if not forearms:
        logging.warning(f"Could not find any forearm data for '{current_video_filename}'.")

    return forearms