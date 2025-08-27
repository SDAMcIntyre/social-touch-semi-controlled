import logging
from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

# Import pandas
import pandas as pd

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ROIProcessingStatus(Enum):
    """Defines the allowed processing stages for a tracked object."""
    TO_BE_PROCESSED = "to be processed"
    TO_BE_REVIEWED = "to be reviewed"
    COMPLETED = "completed"

# ======================= Data Models =======================

@dataclass
class TrackedObject:
    """Represents a single object being tracked with its status and ROIs."""
    status: ROIProcessingStatus = ROIProcessingStatus.TO_BE_PROCESSED.value
    # 💡 MODIFIED: rois DataFrame now has a flat, explicit schema for ROI components.
    # This is more robust and efficient for data analysis and validation.
    rois: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height']
    ))

@dataclass
class AnnotationData:
    """Top-level container for all annotation data."""
    objects_to_track: Dict[str, TrackedObject] = field(default_factory=dict)


# ======================= Business Logic Layer =======================

class ROIAnnotationManager:
    """Manages annotation data in memory using a structured data model."""
    def __init__(self, data: AnnotationData):
        self._data = data
    
    @property
    def data(self) -> AnnotationData:
        return self._data

    def add_object(self, obj_name: str, status: ROIProcessingStatus = ROIProcessingStatus.TO_BE_PROCESSED):
        if obj_name in self._data.objects_to_track:
            raise ValueError(f"Object '{obj_name}' already exists.")
        # Note: We store the string value for consistency
        self._data.objects_to_track[obj_name] = TrackedObject(status=status.value)
        logger.info("Added new object: '%s'", obj_name)

    def set_roi(self, obj_name: str, frame_id: int, x: int, y: int, width: int, height: int):
        """Sets an ROI by taking individual components as arguments."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None:
            raise KeyError(f"Object '{obj_name}' not found.")
        
        if frame_id in obj.rois['frame_id'].values:
            # Update the columns for the existing frame_id
            roi_columns = ['roi_x', 'roi_y', 'roi_width', 'roi_height']
            new_values = [x, y, width, height]
            obj.rois.loc[obj.rois['frame_id'] == frame_id, roi_columns] = new_values
        else:
            # Append a new row with the flat ROI data
            new_row = pd.DataFrame([{
                'frame_id': frame_id, 'roi_x': x, 'roi_y': y,
                'roi_width': width, 'roi_height': height
            }])
            obj.rois = pd.concat([obj.rois, new_row], ignore_index=True)

    def get_roi(self, obj_name: str, frame_id: int) -> Optional[Dict[str, int]]:
        """Retrieves ROI components and returns them in a dictionary."""
        obj = self.get_object(obj_name)
        if obj is None: return None
        
        result = obj.rois[obj.rois['frame_id'] == frame_id]
        if not result.empty:
            # Select the ROI columns, get the first row, and convert to a dictionary
            roi_data = result[['roi_x', 'roi_y', 'roi_width', 'roi_height']].iloc[0].to_dict()
            return roi_data
        return None

    def get_object_names(self) -> List[str]:
        return list(self._data.objects_to_track.keys())

    def get_object(self, obj_name: str) -> Optional[TrackedObject]:
        return self._data.objects_to_track.get(obj_name)

    def update_status(self, obj_name: str, status: Union[ROIProcessingStatus, str]):
        """
        Updates the status of an object.
        This method correctly handles both Enum members and their string representations.
        """
        if obj_name not in self._data.objects_to_track:
            raise KeyError(f"Object '{obj_name}' not found.")

        try:
            # 1. Normalize the input to an Enum member.
            #    This handles both cases:
            #    - If 'status' is ROIProcessingStatus.COMPLETED, it returns that member.
            #    - If 'status' is "completed", it finds the corresponding member.
            valid_status_member = ROIProcessingStatus(status)

            # 2. Assign the canonical string value.
            self._data.objects_to_track[obj_name].status = valid_status_member.value
        except ValueError:
            # This block catches invalid string values.
            raise ValueError(f"'{status}' is not a valid processing status.")
        
    def update_all_status(self, status: ROIProcessingStatus):
        for object_name in self.get_object_names():
            self.update_status(object_name, status)

    def remove_object(self, obj_name: str):
        if obj_name in self._data.objects_to_track:
            del self._data.objects_to_track[obj_name]
            logger.info("Removed object: '%s'", obj_name)
        else:
            raise KeyError(f"Object '{obj_name}' not found.")

    def remove_roi(self, obj_name: str, frame_id: int):
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None: raise KeyError(f"Object '{obj_name}' not found.")
        
        initial_len = len(obj.rois)
        obj.rois = obj.rois[obj.rois['frame_id'] != frame_id].reset_index(drop=True)
        
        if len(obj.rois) == initial_len:
            raise KeyError(f"Frame ID '{frame_id}' not found for object '{obj_name}'.")
        
    def remove_roi_ifexists(self, obj_name: str, frame_id: int):
        if self.get_roi(obj_name, frame_id) is not None:
            self.remove_roi(obj_name, frame_id)

    def are_all_objects_with_status(self, status: ROIProcessingStatus) -> bool:
        """
        Checks if all tracked objects have a specific status.

        Args:
            status (ROIProcessingStatus): The status to check for.

        Returns:
            bool: True if all objects have the given status, False otherwise.
                  Returns False for an empty collection of objects.
        """
        objects = self._data.objects_to_track.values()
        if not objects:
            # The all() function returns True for an empty iterable.
            # We explicitly return False to indicate that not "all" objects
            # are in this state, because there are no objects.
            # This can be changed to True depending on business logic.
            return False
            
        return all(obj.status == status.value for obj in objects)

    def is_any_object_with_status(self, status: ROIProcessingStatus) -> bool:
        """
        Checks if at least one tracked object has a specific status.

        Args:
            status (ROIProcessingStatus): The status to check for.

        Returns:
            bool: True if any object has the given status, False otherwise.
        """
        return any(obj.status == status.value for obj in self._data.objects_to_track.values())

    def are_no_objects_with_status(self, status: ROIProcessingStatus) -> bool:
        """
        Checks if no tracked objects have a specific status.

        Args:
            status (ROIProcessingStatus): The status to check for.

        Returns:
            bool: True if no objects have the given status, False otherwise.
        """
        # This is more efficient than iterating again. It reuses the 'any' check.
        return not self.is_any_object_with_status(status)