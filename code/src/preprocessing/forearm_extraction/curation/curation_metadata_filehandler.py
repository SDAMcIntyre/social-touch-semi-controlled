import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


class CurationMetadataFileHandler:
    """JSON persistence for point cloud curation metadata (removed indices and counts)."""

    @staticmethod
    def save(removed_indices: List[int], total_original: int, path: Union[str, Path]) -> None:
        """
        Saves curation metadata to a JSON file.

        Args:
            removed_indices: List of point indices that were removed during curation.
            total_original: Total number of points in the original (uncurated) cloud.
            path: Destination path for the JSON file.
        """
        data = {
            "removed_point_indices": removed_indices,
            "total_points_original": total_original,
            "total_points_remaining": total_original - len(removed_indices),
            "timestamp": datetime.now().isoformat(timespec='seconds'),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load(path: Union[str, Path]) -> Optional[Dict]:
        """
        Loads curation metadata from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The metadata dict, or None if the file does not exist or cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: could not read curation metadata from {path}: {e}")
            return None
