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

# ======================= Persistence Layer =======================



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
        self._data.objects_to_track[obj_name] = TrackedObject(status=status)
        logger.info("Added new object: '%s'", obj_name)

    def set_roi(self, obj_name: str, frame_id: int, x: int, y: int, width: int, height: int):
        """
        💡 MODIFIED: Sets an ROI by taking individual components as arguments.
        """
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
        """
        💡 MODIFIED: Retrieves ROI components and returns them in a dictionary.
        """
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

    def update_status(self, obj_name: str, status: ROIProcessingStatus):
        if obj_name not in self._data.objects_to_track:
            raise KeyError(f"Object '{obj_name}' not found.")
        self._data.objects_to_track[obj_name].status = status.value

    def update_status(self, obj_name: str, status: Union[ROIProcessingStatus, str]):
        if obj_name not in self._data.objects_to_track:
            raise KeyError(f"Object '{obj_name}' not found.")

        try:
            # 1. Normalize the input.
            # This line elegantly handles both cases:
            # - If 'status' is ROIProcessingStatus.COMPLETED, it returns that member.
            # - If 'status' is "completed", it finds the corresponding member.
            # - If 'status' is an invalid string like "done", it raises a ValueError.
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
        """This method's logic remains correct as it only depends on frame_id."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None: raise KeyError(f"Object '{obj_name}' not found.")
        
        initial_len = len(obj.rois)
        obj.rois = obj.rois[obj.rois['frame_id'] != frame_id].reset_index(drop=True)
        
        if len(obj.rois) == initial_len:
            raise KeyError(f"Frame ID '{frame_id}' not found for object '{obj_name}'.")

    def is_all_tracking_completed(self) -> bool:
        """
        Checks if all tracked objects have the status 'COMPLETED'.

        This implementation uses the all() function for a more concise and
        readable check. It returns True if the dictionary is empty.

        Returns:
            bool: True if all objects are completed, False otherwise.
        """
        if not self._data.objects_to_track:
            return False
            
        return all(
            obj.status == ROIProcessingStatus.COMPLETED 
            for obj in self._data.objects_to_track.values()
        )