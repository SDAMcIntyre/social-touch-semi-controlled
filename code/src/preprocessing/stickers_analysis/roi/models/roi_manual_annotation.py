import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Set
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
    
    # ROI DataFrame: ['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height']
    rois: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height']
    ))
    
    # Ignore Start DataFrame: ['frame_id']
    ignore_starts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=['frame_id']
    ))
    
    # Ignore Stop DataFrame: ['frame_id']
    ignore_stops: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=['frame_id']
    ))

@dataclass
class AnnotationData:
    """Top-level container for all annotation data."""
    objects_to_track: Dict[str, TrackedObject] = field(default_factory=dict)


# ======================= Business Logic Layer =======================

class ROIAnnotationManager:
    """Manages annotation data in memory using a structured data model."""

    def __init__(self, data: Optional[AnnotationData] = None):
        """
        Initializes the manager.

        If `data` is None, a new AnnotationData instance is created.
        Otherwise, the provided instance is used.
        """
        self._data = data or AnnotationData()

    @property
    def data(self) -> AnnotationData:
        return self._data

    def add_object(self, obj_name: str, status: ROIProcessingStatus = ROIProcessingStatus.TO_BE_PROCESSED):
        if obj_name in self._data.objects_to_track:
            raise ValueError(f"Object '{obj_name}' already exists.")
        # Note: We store the string value for consistency
        self._data.objects_to_track[obj_name] = TrackedObject(status=status.value)
        logger.info("Added new object: '%s'", obj_name)

    # --- ROI Methods ---

    def set_roi(self, obj_name: str, frame_id: int, x: int, y: int, width: int, height: int):
        """Sets an ROI by taking individual components as arguments."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None:
            raise KeyError(f"Object '{obj_name}' not found.")

        # Check if the frame_id already exists in the DataFrame
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
            return {k: int(v) for k, v in roi_data.items()} # Ensure values are int
        return None

    def remove_roi(self, obj_name: str, frame_id: int):
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None: raise KeyError(f"Object '{obj_name}' not found.")

        initial_len = len(obj.rois)
        obj.rois = obj.rois[obj.rois['frame_id'] != frame_id].reset_index(drop=True)

        if len(obj.rois) == initial_len:
            raise KeyError(f"Frame ID '{frame_id}' not found for object '{obj_name}'.")

    def remove_roi_if_exists(self, obj_name: str, frame_id: int):
        """Safely removes an ROI if it exists, without raising an error."""
        if self.get_roi(obj_name, frame_id) is not None:
            self.remove_roi(obj_name, frame_id)

    # --- Ignore Start Methods ---

    def set_ignore_start(self, obj_name: str, frame_id: int):
        """Marks a frame as the start of an ignored sequence for an object."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None:
            raise KeyError(f"Object '{obj_name}' not found.")

        if frame_id not in obj.ignore_starts['frame_id'].values:
            new_row = pd.DataFrame([{'frame_id': frame_id}])
            obj.ignore_starts = pd.concat([obj.ignore_starts, new_row], ignore_index=True)

    def remove_ignore_start(self, obj_name: str, frame_id: int):
        """Removes an ignore start marker."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None: raise KeyError(f"Object '{obj_name}' not found.")
        
        obj.ignore_starts = obj.ignore_starts[obj.ignore_starts['frame_id'] != frame_id].reset_index(drop=True)

    # --- Ignore Stop Methods ---

    def set_ignore_stop(self, obj_name: str, frame_id: int):
        """Marks a frame as the end of an ignored sequence for an object."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None:
            raise KeyError(f"Object '{obj_name}' not found.")

        if frame_id not in obj.ignore_stops['frame_id'].values:
            new_row = pd.DataFrame([{'frame_id': frame_id}])
            obj.ignore_stops = pd.concat([obj.ignore_stops, new_row], ignore_index=True)

    def remove_ignore_stop(self, obj_name: str, frame_id: int):
        """Removes an ignore stop marker."""
        obj = self._data.objects_to_track.get(obj_name)
        if obj is None: raise KeyError(f"Object '{obj_name}' not found.")
        
        obj.ignore_stops = obj.ignore_stops[obj.ignore_stops['frame_id'] != frame_id].reset_index(drop=True)

    # --- General Object Methods ---

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
            valid_status_member = ROIProcessingStatus(status)
            self._data.objects_to_track[obj_name].status = valid_status_member.value
        except ValueError:
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

    def are_all_objects_with_status(self, status: ROIProcessingStatus) -> bool:
        """
        Checks if all tracked objects have a specific status.
        Returns False for an empty collection of objects.
        """
        objects = self._data.objects_to_track.values()
        if not objects:
            return False
        return all(obj.status == status.value for obj in objects)

    def is_any_object_with_status(self, status: ROIProcessingStatus) -> bool:
        """Checks if at least one tracked object has a specific status."""
        return any(obj.status == status.value for obj in self._data.objects_to_track.values())

    def are_no_objects_with_status(self, status: ROIProcessingStatus) -> bool:
        """Checks if no tracked objects have a specific status."""
        return not self.is_any_object_with_status(status)

    def get_frames_to_process(self, obj_name: str, total_frames: int) -> Set[int]:
        """
        Calculates the set of frame IDs that should be processed for a given object.
        
        Logic:
        - Iterates from 0 to total_frames.
        - Maintains an 'ignoring' state (default False).
        - 'ignore_start' sets ignoring=True.
        - 'ignore_stop' sets ignoring=False.
        - 'roi' definition sets ignoring=False (Priority: ROI > Ignore).
        """
        obj = self.get_object(obj_name)
        if obj is None:
            raise KeyError(f"Object '{obj_name}' not found.")
            
        roi_frames = set(obj.rois['frame_id'].astype(int).values)
        start_ignore_frames = set(obj.ignore_starts['frame_id'].astype(int).values)
        stop_ignore_frames = set(obj.ignore_stops['frame_id'].astype(int).values)
        
        frames_to_process = set()
        is_ignoring = False
        
        for frame_id in range(total_frames):
            # Check for state triggers
            if frame_id in start_ignore_frames:
                is_ignoring = True
            
            if frame_id in stop_ignore_frames:
                is_ignoring = False
                
            if frame_id in roi_frames:
                is_ignoring = False
                
            # Add to set if valid
            if not is_ignoring:
                frames_to_process.add(frame_id)
                
        return frames_to_process