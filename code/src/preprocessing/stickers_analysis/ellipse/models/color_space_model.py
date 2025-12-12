import numpy as np
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

class ColorSpaceStatus(Enum):
    """Provides type-safe status options for colorspace processing."""
    TO_BE_DEFINED = "to_be_defined"
    TO_BE_PROCESSED = "to_be_processed"
    TO_BE_REVIEWED = "pending_review"
    REVIEW_COMPLETED = "review_completed"


class ColorSpaceDefault(Enum):
    """Provides default values for colorspace properties."""
    Threshold = 250
    Status = ColorSpaceStatus.TO_BE_DEFINED.value


class ColorSpace:
    """
    Represents a single named colorspace object, encapsulating its status,
    threshold, and frame-specific data.
    """
    def __init__(self, data: Dict[str, Any]):
        """
        Initializes the ColorSpace object from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary for a single named colorspace.
        """
        self.status: str = data.get("status", ColorSpaceDefault.Status.value)
        self.threshold: Optional[int] = data.get("threshold")
        self.frames_colorspace: List[Dict[str, Any]] = data.get("colorspaces", [])

    def get_frame_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """
        Finds and returns the data for a specific frame ID.

        Args:
            frame_id (int): The specific frame ID to retrieve.

        Returns:
            The dictionary of data for the matching frame, or None if not found.
        """
        for frame_data in self.frames_colorspace:
            if frame_data.get("frame_id") == frame_id:
                return frame_data
        return None

    def get_frame_ids(self) -> List[int]:
        """
        Retrieves a sorted list of all unique frame IDs present in the colorspace.

        Returns:
            A sorted list of integers representing the unique frame IDs.
        """
        # Using a set for efficiency in finding unique IDs, then sorting.
        frame_ids = {
            frame_data.get("frame_id")
            for frame_data in self.frames_colorspace
            if frame_data.get("frame_id") is not None
        }
        return sorted(list(frame_ids))

    def parse(self) -> Tuple[List[int], List[Dict], str]:
        """
        Parses the internal data into separate lists for frame IDs and
        colorspace dictionaries, along with the status.

        Returns:
            A tuple containing (frame_ids, colorspaces, status).
        """
        valid_entries = [
            (item.get('frame_id'), item.get('colorspace'))
            for item in self.frames_colorspace
            if item.get('frame_id') is not None and item.get('colorspace') is not None
        ]

        if not valid_entries:
            return ([], [], self.status)

        frame_ids, colorspaces = zip(*valid_entries)
        return list(frame_ids), list(colorspaces), self.status

    def to_dict(self) -> Dict[str, Any]:
        """Converts the object back to its dictionary representation for JSON serialization."""
        data = {
            "status": self.status,
            "threshold": self.threshold,
            "colorspaces": self.frames_colorspace,
        }
        return data

    def update(self,
               frame_ids: List[int],
               adjusted_colorspaces: List[Dict[str, Any]],
               status: str,
               threshold: Optional[int] = None,
               merge: bool = True) -> None:
        """
        Updates the colorspace's data from new lists of frames and colorspaces.

        Args:
            frame_ids: List of frame IDs.
            adjusted_colorspaces: List of corresponding colorspace dicts.
            status: The new review status to assign.
            threshold: The new threshold value. If None, the existing value is kept.
            merge: If True, merge with existing frames. If False, overwrite all frames.
        """
        self.status = status
        if threshold is not None:
            self.threshold = threshold

        new_frames_list = [
            {"frame_id": frame_id, "colorspace": colorspace}
            for frame_id, colorspace in zip(frame_ids, adjusted_colorspaces)
        ]

        if not merge:
            self.frames_colorspace = new_frames_list
        else:
            existing_frames_map = {item['frame_id']: item for item in self.frames_colorspace}
            new_frames_map = {item['frame_id']: item for item in new_frames_list}
            existing_frames_map.update(new_frames_map)
            self.frames_colorspace = list(existing_frames_map.values())

    def extract_rgb_triplets(
        self,
        output_format: str = 'array'
    ) -> Union[np.ndarray, list]:
        """
        Extracts all RGB color triplets from the nested 'freehand_pixels'
        dictionary across all frames in this colorspace.

        Args:
            output_format (str, optional): The desired output format ('array' or 'list').

        Returns:
            A collection of all extracted RGB triplets as a NumPy array or a list.

        Raises:
            ValueError: If an unsupported output_format is provided.
        """
        all_rgb_triplets = []
        for frame_data in self.frames_colorspace:
            freehand_pixels_data = frame_data.get('colorspace', {}).get('freehand_pixels', {})
            if freehand_pixels_data:
                rgb_list = freehand_pixels_data.get('rgb')
                if rgb_list:
                    all_rgb_triplets.extend(rgb_list)

        if not all_rgb_triplets:
            return np.empty((0, 3), dtype=np.uint8) if output_format == 'array' else []

        if output_format == 'array':
            return np.array(all_rgb_triplets, dtype=np.uint8)
        elif output_format == 'list':
            return all_rgb_triplets
        else:
            raise ValueError(f"Invalid output_format: '{output_format}'. Please choose 'array' or 'list'.")