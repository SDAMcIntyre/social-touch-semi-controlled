import json
from pathlib import Path
import logging
from typing import Optional

# Import the dataclass to be used for object generation
from ..models.hand_metadata import HandMetadataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandMetadataFileHandler:
    """
    Handles file I/O operations, such as saving and loading metadata.
    """
    @staticmethod
    def save_json(data: dict, path: Path) -> None:
        """Saves a dictionary to a specified file path in JSON format."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Metadata successfully saved to {path}")
        except IOError as e:
            logging.error(f"Failed to save metadata to {path}: {e}")

    @staticmethod
    def load(json_path: Path) -> Optional[HandMetadataManager]:
        """
        Loads a JSON file and generates a MetadataManager object from its content.

        Args:
            json_path (Path): The path to the JSON metadata file.

        Returns:
            An instance of MetadataManager if successful, otherwise None.
        """
        try:
            # Step 1: Read the JSON file into a dictionary
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Step 2: Transform the dictionary data to match the MetadataManager constructor
            
            # Reconstruct Path objects from filenames (assuming relative to some base, 
            # but storing just the name as per previous structure)
            video_path = Path(data.get("source_video_name", ""))
            model_path = Path(data.get("selected_hand_model_name", ""))

            # Convert hand orientation string ("left"/"right") to a boolean
            is_left = data.get("hand_orientation") == "left"

            # Convert the list of point dictionaries into a single {label: id} dictionary
            points_list = data.get("selected_points", [])
            selected_points = {item["label"]: item["vertex_id"] for item in points_list}

            # Retrieve excluded vertices (default to empty list if not present)
            excluded_vertex_ids = data.get("excluded_vertex_ids", [])

            # Step 3: Instantiate and return the MetadataManager object
            return HandMetadataManager(
                source_video_path=video_path,
                selected_hand_model_path=model_path,
                selected_frame_number=data.get("selected_frame_number", 0),
                is_left_hand=is_left,
                selected_points=selected_points,
                excluded_vertex_ids=excluded_vertex_ids
            )

        except FileNotFoundError:
            logging.error(f"Metadata file not found at: {json_path}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse {json_path}. File may be corrupt or missing key. Error: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {json_path}: {e}")
            return None