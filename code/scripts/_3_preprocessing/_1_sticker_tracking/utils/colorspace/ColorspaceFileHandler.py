import json
import os
from typing import Any, Dict, List, Optional, Tuple, Generator

class ColorspaceFileHandler:
    """
    Manages loading, parsing, and updating of colorspace data in a JSON file.

    This class encapsulates the file I/O and data manipulation logic for a 
    specific JSON file containing colorspace objects. It provides a structured
    and error-resistant way to interact with the data.

    Attributes:
        file_path (str): The path to the JSON file being managed.
        data (Dict[str, Any]): The in-memory representation of the JSON content.
    """

    def __init__(self, file_path: str):
        """
        Initializes the handler with a file path and loads existing data.

        Args:
            file_path: The path to the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist upon initial load attempt.
            ValueError: If the file contains invalid JSON.
        """
        self.file_path = file_path

        if not os.path.exists(file_path):
            # If the file doesn't exist, we can start with an empty state
            # without raising an error immediately. It will be created on save().
            print(f"⚠️ Warning: File '{file_path}' not found. Will be created on save.")
            self.data = {}
        else:
            self.load()
            

    def load(self) -> None:
        """
        Loads and decodes the JSON data from the file into the instance.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains malformed JSON.
        """
        try:
            with open(self.file_path, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: File not found at '{self.file_path}'")
            # Re-raise to allow calling code to handle it
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Could not decode JSON from '{self.file_path}'")
            # Re-raise with a more informative message
            raise ValueError(f"Invalid JSON in {self.file_path}: {e}") from e

    def save(self) -> None:
        """
        Writes the current in-memory data back to the JSON file.

        Raises:
            IOError: If an error occurs during file writing.
        """
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.data, f, indent=4)
            print(f"💾 Successfully saved data to '{os.path.basename(self.file_path)}'")
        except IOError as e:
            print(f"❌ An error occurred while writing to file: {e}")
            raise
    
    def get_object_names(self) -> Optional[List[str]]:
        return list(self.data.keys())

    def get_parsed_object(self, object_name: str) -> Optional[Tuple[List[int], List[Dict], str]]:
        """
        Parses a specific top-level object from the loaded data.

        Args:
            object_name: The key of the top-level object to parse.

        Returns:
            A tuple containing (frame_ids, colorspaces, status) or None if
            the object is not found or is malformed.
        """
        payload = self.data.get(object_name)
        if not isinstance(payload, dict) or 'status' not in payload or 'colorspaces' not in payload:
            print(f"⚠️ Warning: Object '{object_name}' is missing required keys or is not a dictionary.")
            return None

        status = payload['status']
        colorspace_items = payload.get('colorspaces', [])
        
        # Filter and unzip in one pass for efficiency
        valid_entries = [
            (item.get('frame_id'), item.get('colorspace')) 
            for item in colorspace_items 
            if item.get('frame_id') is not None and item.get('colorspace') is not None
        ]

        if not valid_entries:
            return ([], [], status)

        frame_ids, colorspaces = zip(*valid_entries)
        return list(frame_ids), list(colorspaces), status

    def get_all_parsed_objects(self) -> Generator[Tuple[str, List[int], List[Dict], str], None, None]:
        """
        A generator that yields all parsed top-level objects from the data.

        Yields:
            A tuple of (object_name, frame_ids, colorspaces, status) for each
            valid object in the file.
        """
        for object_name in self.data.keys():
            parsed_data = self.get_parsed_object(object_name)
            if parsed_data:
                frame_ids, colorspaces, status = parsed_data
                yield object_name, frame_ids, colorspaces, status

    def update_object(self, 
                      object_name: str, 
                      frame_ids: List[int], 
                      adjusted_colorspaces: List[Dict[str, Any]], 
                      status: str = "pending", 
                      overwrite: bool = False) -> None:
        """
        Updates or creates a top-level object with new colorspace data.

        This method first prepares the payload and then merges or overwrites it
        into the in-memory data store. Call save() to persist changes.

        Args:
            object_name: The key of the top-level object to update.
            frame_ids: List of frame IDs.
            adjusted_colorspaces: List of corresponding colorspace dicts.
            status: The review status to assign.
            overwrite: If True, replace the object. If False, merge with existing.
        """
        new_content = self._prepare_payload(frame_ids, adjusted_colorspaces, status)

        if overwrite or object_name not in self.data:
            print(f"✅ Overwriting or creating object '{object_name}'.")
            self.data[object_name] = new_content
        else:
            # The original `update_json_object` had merging logic for dicts.
            # We can replicate that here for the top-level content.
            existing_object = self.data.get(object_name, {})
            if isinstance(existing_object, dict) and isinstance(new_content, dict):
                print(f"✅ Merging new content into object '{object_name}'.")
                existing_object.update(new_content)
                self.data[object_name] = existing_object
            else:
                print(f"⚠️ Cannot merge due to incompatible types. Overwriting '{object_name}'.")
                self.data[object_name] = new_content

    @staticmethod
    def _prepare_payload(frame_ids: List[int], 
                         adjusted_colorspaces: List[Dict[str, Any]], 
                         status: str) -> Dict[str, Any]:
        """
        Formats colorspace data into a dictionary payload. (Internal helper)
        """
        payload = {'status': status, 'colorspaces': []}
        for frame_id, colorspace in zip(frame_ids, adjusted_colorspaces):
            frame_content = {"frame_id": frame_id, "colorspace": colorspace}
            payload['colorspaces'].append(frame_content)
        return payload


