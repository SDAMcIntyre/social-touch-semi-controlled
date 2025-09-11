from typing import Any, Dict, List, Optional, Tuple

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
        self.status: str = data.get("status", "pending")
        self.threshold: Optional[int] = data.get("threshold")
        self.colorspaces: List[Dict[str, Any]] = data.get("colorspaces", [])

    def get_frame_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """
        Finds and returns the data for a specific frame ID.

        Args:
            frame_id (int): The specific frame ID to retrieve.

        Returns:
            The dictionary of data for the matching frame, or None if not found.
        """
        for frame_data in self.colorspaces:
            if frame_data.get("frame_id") == frame_id:
                return frame_data
        return None

    def parse(self) -> Tuple[List[int], List[Dict], str]:
        """
        Parses the internal data into separate lists for frame IDs and
        colorspace dictionaries, along with the status.

        Returns:
            A tuple containing (frame_ids, colorspaces, status).
        """
        valid_entries = [
            (item.get('frame_id'), item.get('colorspace'))
            for item in self.colorspaces
            if item.get('frame_id') is not None and item.get('colorspace') is not None
        ]

        if not valid_entries:
            return ([], [], self.status)

        frame_ids, colorspaces = zip(*valid_entries)
        return list(frame_ids), list(colorspaces), self.status

    def to_dict(self) -> Dict[str, Any]:
        """Converts the object back to its dictionary representation."""
        data = {
            "status": self.status,
            "colorspaces": self.colorspaces,
        }
        if self.threshold is not None:
            data["threshold"] = self.threshold
        return data