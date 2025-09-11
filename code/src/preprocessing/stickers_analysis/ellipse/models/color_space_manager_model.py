from typing import Any, Dict, List, Optional, Tuple, Generator

from .color_space_model import ColorSpace


class ColorSpaceManager:
    """
    A class to hold and provide easy access to colorspace data by managing
    a collection of ColorSpace objects.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initializes the ColorSpaceManager object.

        Args:
            data (Dict[str, Any]): The raw dictionary data loaded from a JSON file.
        """
        self._data: Dict[str, ColorSpace] = {
            name: ColorSpace(properties) for name, properties in data.items()
        }

    @property
    def colorspace_names(self) -> List[str]:
        """Returns a list of all top-level colorspace names (e.g., 'sticker_yellow')."""
        return list(self._data.keys())

    def get_colorspace(self, name: str) -> Optional[ColorSpace]:
        """
        Fetches the ColorSpace object for a specific name.

        Args:
            name (str): The name of the colorspace (e.g., "sticker_yellow").

        Returns:
            The ColorSpace object, or None if the name is not found.
        """
        return self._data.get(name)

    def get_status(self, name: str) -> Optional[str]:
        """Fetches the status for a specific colorspace."""
        colorspace = self.get_colorspace(name)
        return colorspace.status if colorspace else None

    def get_threshold(self, name: str) -> Optional[int]:
        """Fetches the threshold value for a specific colorspace."""
        colorspace = self.get_colorspace(name)
        return colorspace.threshold if colorspace else None

    def get_all_frame_data(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Fetches the list of all frame data for a specific colorspace."""
        colorspace = self.get_colorspace(name)
        return colorspace.colorspaces if colorspace else None

    def get_frame_by_id(self, colorspace_name: str, frame_id: int) -> Optional[Dict[str, Any]]:
        """Finds data for a specific frame ID within a named colorspace."""
        colorspace = self.get_colorspace(colorspace_name)
        return colorspace.get_frame_by_id(frame_id) if colorspace else None

    def get_parsed_object(self, object_name: str) -> Optional[Tuple[List[int], List[Dict], str]]:
        """
        Parses a specific colorspace object into its core components.

        Args:
            object_name: The key of the top-level object to parse.

        Returns:
            A tuple containing (frame_ids, colorspaces, status) or None if not found.
        """
        colorspace = self.get_colorspace(object_name)
        if not colorspace:
            print(f"⚠️ Warning: Object '{object_name}' not found.")
            return None
        return colorspace.parse()

    def get_all_parsed_objects(self) -> Generator[Tuple[str, List[int], List[Dict], str], None, None]:
        """A generator that yields all parsed top-level objects from the data."""
        for object_name in self.colorspace_names:
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
        Updates or creates a top-level colorspace object with new data.

        Args:
            object_name: The key of the object to update.
            frame_ids: List of frame IDs.
            adjusted_colorspaces: List of corresponding colorspace dicts.
            status: The review status to assign.
            overwrite: If True, replace the object. If False, merge with existing.
        """
        new_payload = self._prepare_payload(frame_ids, adjusted_colorspaces, status)

        if overwrite or object_name not in self._data:
            print(f"✅ Overwriting or creating object '{object_name}'.")
            self._data[object_name] = ColorSpace(data=new_payload)
        else:
            print(f"✅ Merging new content into object '{object_name}'.")
            existing_cs = self._data[object_name]
            existing_cs.status = status  # New status overwrites old

            # Merge frames based on frame_id to update existing or add new ones
            existing_frames = {item['frame_id']: item for item in existing_cs.colorspaces}
            new_frames = {item['frame_id']: item for item in new_payload.get('colorspaces', [])}
            existing_frames.update(new_frames)
            
            existing_cs.colorspaces = list(existing_frames.values())

    @staticmethod
    def _prepare_payload(frame_ids: List[int],
                         adjusted_colorspaces: List[Dict[str, Any]],
                         status: str) -> Dict[str, Any]:
        """Formats colorspace data into a dictionary payload. (Internal helper)"""
        payload = {'status': status, 'colorspaces': []}
        for frame_id, colorspace in zip(frame_ids, adjusted_colorspaces):
            frame_content = {"frame_id": frame_id, "colorspace": colorspace}
            payload['colorspaces'].append(frame_content)
        return payload