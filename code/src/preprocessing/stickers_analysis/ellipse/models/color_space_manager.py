from typing import Any, Dict, List, Optional, Tuple, Generator

from .color_space_model import ColorSpace, ColorSpaceDefault, ColorSpaceStatus


class ColorSpaceManager:
    """
    A class to hold and provide easy access to colorspace data by managing
    a collection of ColorSpace objects.
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """
        Initializes the ColorSpaceManager object.

        Args:
            data (Optional[Dict[str, Any]], optional): 
                The raw dictionary data loaded from a file.
                If None, an empty manager is created. Defaults to None.
        """
        self._data: Dict[str, ColorSpace] = {
            name: ColorSpace(properties) for name, properties in (data or {}).items()
        }

    @property
    def colorspace_names(self) -> List[str]:
        """Returns a list of all top-level colorspace names."""
        return list(self._data.keys())

    def are_no_objects_with_status(self, status: ColorSpaceStatus) -> bool:
        return not self.any_objects_with_status(status)
    
    def any_objects_with_status(self, status: ColorSpaceStatus) -> bool:
        return any(cs.status == status.value for cs in self._data.values())
    
    def all_objects_with_status(self, status: ColorSpaceStatus) -> bool:
        return all(cs.status == status.value for cs in self._data.values())
    
    def not_all_objects_with_status(self, status: ColorSpaceStatus) -> bool:
        return not self.all_objects_with_status(status)

    # An alternative, more explicit implementation:
    # return any(cs.status != status.value for cs in self._data.values())
    
    def update_status(self, name: str, status: str) -> None:
        """
        Updates the status of a specific named colorspace.

        Args:
            name (str): The name of the colorspace to update.
            status (str): The new status to set.

        Raises:
            KeyError: If the colorspace name is not found.
        """
        if name in self._data:
            self._data[name].status = status
            print(f"✅ Status for '{name}' updated to '{status}'.")
        else:
            raise KeyError(f"ColorSpace with name '{name}' not found.")
            
    def update_threshold(self, name: str, threshold: int) -> None:
        """
        Updates the threshold of a specific named colorspace.

        Args:
            name (str): The name of the colorspace to update.
            threshold (str): The new threshold to set.

        Raises:
            KeyError: If the colorspace name is not found.
        """
        if name in self._data:
            self._data[name].threshold = threshold
            print(f"✅ Threshold for '{name}' updated to '{threshold}'.")
        else:
            raise KeyError(f"ColorSpace with name '{name}' not found.")
            
    def get_colorspace(self, name: str) -> Optional[ColorSpace]:
        """Fetches the ColorSpace object for a specific name."""
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
        return colorspace.frames_colorspace if colorspace else None

    def get_frame_by_id(self, colorspace_name: str, frame_id: int) -> Optional[Dict[str, Any]]:
        """Finds data for a specific frame ID within a named colorspace."""
        colorspace = self.get_colorspace(colorspace_name)
        return colorspace.get_frame_by_id(frame_id) if colorspace else None

    def get_parsed_object(self, object_name: str) -> Optional[Tuple[List[int], List[Dict], str]]:
        """Parses a specific colorspace object by delegating to its `parse` method."""
        colorspace = self.get_colorspace(object_name)
        if not colorspace:
            print(f"⚠️ Warning: Object '{object_name}' not found.")
            return None
        return colorspace.parse()

    def get_all_parsed_objects(self) -> Generator[Tuple[str, List[int], List[Dict], str], None, None]:
        """A generator that yields all parsed top-level objects from the data."""
        for object_name, colorspace in self._data.items():
            frame_ids, colorspaces, status = colorspace.parse()
            yield object_name, frame_ids, colorspaces, status

    def add_object(self,
                   object_name: str,
                   frame_ids: List[int],
                   adjusted_colorspaces: List[Dict[str, Any]],
                   status: ColorSpaceStatus = ColorSpaceStatus.TO_BE_PROCESSED,
                   threshold: int = ColorSpaceDefault.Threshold.value,
                   *,
                   replace: bool = False) -> bool:
        """
        Adds a new colorspace object to the manager.

        Args:
            object_name: The key of the new object to create.
            frame_ids: List of frame IDs.
            adjusted_colorspaces: List of corresponding colorspace dicts.
            status: The review status to assign.
            threshold: The threshold value to assign.

        Returns:
            True if the object was added, False if it already exists.
        """
        if object_name in self._data and not replace:
            # This is the key change: raise an exception instead of returning False.
            raise ValueError(
                f"Object '{object_name}' already exists. "
                "Use update_object() or set replace=True."
            )

        new_payload = {
            'status': status.value,
            'threshold': threshold,
            'colorspaces': [
                {"frame_id": fid, "colorspace": cs}
                for fid, cs in zip(frame_ids, adjusted_colorspaces)
            ]
        }
        self._data[object_name] = ColorSpace(data=new_payload)
        print(f"✅ Successfully added new object '{object_name}'.")
        return True

    def update_object(self,
                      object_name: str,
                      frame_ids: List[int],
                      adjusted_colorspaces: List[Dict[str, Any]],
                      status: ColorSpaceStatus = ColorSpaceStatus.TO_BE_PROCESSED,
                      threshold: Optional[int] = None,
                      *,
                      merge: bool = True) -> None:
        """
        Updates or creates a top-level colorspace object with new data.

        Args:
            object_name: The key of the object to update or create.
            frame_ids: List of frame IDs.
            adjusted_colorspaces: List of corresponding colorspace dicts.
            status: The review status to assign.
            threshold: The threshold value to assign. If None, existing is kept on update.
            merge: If False, replace frames. If True, merge them.
        """
        if not object_name in self._data:
            # This is the key change: raise an exception instead of returning False.
            raise ValueError(
                f"Object '{object_name}' do not exists. "
                "Use add_object() or update_or_add_object()."
            )
        colorspace_obj = self.get_colorspace(object_name)

        print(f"✅ Updating object '{object_name}'. Merge={merge}.")
        colorspace_obj.update(
            frame_ids=frame_ids,
            adjusted_colorspaces=adjusted_colorspaces,
            status=status.value,
            threshold=threshold,
            merge=merge
        )
    
    def add_or_update_object(self,
                         object_name: str,
                         frame_ids: List[int],
                         adjusted_colorspaces: List[Dict[str, Any]],
                         status: ColorSpaceStatus = ColorSpaceStatus.TO_BE_PROCESSED,
                         threshold: Optional[int] = None,
                         *,
                         merge: bool = True) -> None:
        """
        Adds a new colorspace object or updates an existing one (upsert).

        If an object with the given 'object_name' already exists, its data is
        updated. If it does not exist, a new object is created.

        Args:
            object_name: The key of the object to add or update.
            frame_ids: A list of frame IDs.
            adjusted_colorspaces: A list of colorspace dictionaries corresponding
                                to the frame_ids.
            status: The review status to assign to the object.
            threshold: The threshold value. If updating and set to None, the
                    existing threshold is kept. If adding and set to None, a
                    default is used.
            merge: If True (default) and the object exists, new frame data is
                merged with existing data. If False, existing frame data is
                replaced. This has no effect when creating a new object.
        """
        # Check if the object already exists to decide whether to update or add.
        if object_name in self._data:
            # --- UPDATE PATH ---
            print(f"✅ Updating existing object '{object_name}'. Merge={merge}.")
            colorspace_obj = self.get_colorspace(object_name)
            
            # Ensure the object was retrieved successfully before updating
            if colorspace_obj:
                colorspace_obj.update(
                    frame_ids=frame_ids,
                    adjusted_colorspaces=adjusted_colorspaces,
                    status=status.value,
                    threshold=threshold, # Pass None to preserve existing on update
                    merge=merge
                )
            else:
                # This case handles potential inconsistencies where a key might
                # exist in self._data but get_colorspace returns nothing.
                raise KeyError(f"Object '{object_name}' key exists but object could not be retrieved.")
        else:
            # --- ADD PATH ---
            print(f"✨ Adding new object '{object_name}'.")
            
            # For a new object, use a default threshold if one isn't provided.
            create_threshold = threshold if threshold is not None else ColorSpaceDefault.Threshold.value
            
            self.add_object(
                object_name=object_name,
                frame_ids=frame_ids,
                adjusted_colorspaces=adjusted_colorspaces,
                status=status,
                threshold=create_threshold
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the entire manager's state into a JSON-friendly dictionary."""
        return {
            name: cs_object.to_dict()
            for name, cs_object in self._data.items()
        }