import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

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
    Manages and provides access to a collection of forearm point clouds.

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
        pointcloud_filename = f"{video_stem}_frame_{params.frame_id:04}_with_normals.ply"
        path = self._pointcloud_dir / pointcloud_filename
        return PointCloudDataHandler.load(path)

    def get_pointclouds_for_video(self, video_filename: str) -> Dict[int, o3d.geometry.PointCloud]:
        """
        Finds and loads all forearm point clouds for the specified video.

        This method performs an exact match on the video filename.
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

    def find_closest_reference(self, video_filename: str) -> Optional[Tuple[int, o3d.geometry.PointCloud]]:
        """
        Finds a single reference forearm by looking for the closest block number.

        This is the explicit fallback logic. It returns None if no suitable
        reference can be found.
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
        
        pc = self._load_pointcloud(closest_params)
        if pc:
            return closest_params.frame_id, pc
        
        return None
    
    
def get_forearms_with_fallback(
    catalog: ForearmCatalog, 
    current_video_filename: str
) -> Dict[int, o3d.geometry.PointCloud]:
    """
    Gets forearm point clouds for a video, with intelligent fallback.

    This function first attempts to find all forearms that exactly match the
    video filename. If none are found, it uses the catalog's reference-finding
    logic to locate and load the single closest forearm from another video
    based on the 'block-order' number.

    Args:
        catalog: An initialized ForearmCatalog instance.
        current_video_filename: The filename of the video to process.

    Returns:
        A dictionary mapping frame IDs to o3d.geometry.PointCloud objects. This will
        contain all forearms for the video, a single reference forearm,
        or be empty if no data can be found.
    """
    # 1. Attempt the primary, explicit search first.
    forearms = catalog.get_pointclouds_for_video(current_video_filename)
    if not forearms:
        # 2. If the primary search returned nothing, trigger the fallback.
        logging.info(
            f"No direct forearm match for '{current_video_filename}'. "
            "Attempting to find a fallback reference."
        )
        reference_data = catalog.find_closest_reference(current_video_filename)

        # 3. If a reference was found, format it.
        if reference_data:
            frame_id, pointcloud = reference_data
            forearms = {frame_id: pointcloud}

    if forearms:
        # Find the lowest key and rebuild the dict, replacing that key with 0.
        min_key = min(forearms.keys())
        forearms_adjusted = {0 if k == min_key else k: v for k, v in forearms.items()}
        return forearms_adjusted
    
    # 4. If nothing was found, return an empty dictionary.
    logging.warning(f"Could not find any data or suitable reference for '{current_video_filename}'.")
    return {}

