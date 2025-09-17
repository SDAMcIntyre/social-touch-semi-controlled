# data_model.py

from typing import Dict, Iterator, Tuple, List
import pandas as pd
from enum import Enum

from ...common.models.tracked_data_interface import TrackedDataInterface


# --- 1. Status Enum ---
class ROITrackedStatus(Enum):
    """Defines status constants for tracking and annotation."""
    LABELED = "Labeled"
    INITIAL_ROI = "Initial ROI"
    TRACKING = "Tracking"

# --- 2. Data Schema and Type Alias ---
ROI_TRACKED_SCHEMA = {
    'frame_id': 'Int64',
    'roi_x': 'Int64',
    'roi_y': 'Int64',
    'roi_width': 'Int64',
    'roi_height': 'Int64',
    'status': str
}

# Represents the data for a single tracked object (remains a DataFrame).
ROITrackedObject = pd.DataFrame

# --- 4. Concrete Data Structure ---
class ROITrackedObjects(Dict, TrackedDataInterface):
    """
    Represents the main collection of all tracked object data.

    This class is the concrete implementation. It acts as both the data 
    container (by inheriting from dict) and the data source provider 
    (by implementing TrackedDataInterface). This bundles the data and its
    access logic together.
    """
    def get_items_for_frame(self, frame_index: int) -> Iterator[Tuple[str, pd.Series]]:
        """
        Retrieves the data for all tracked objects at a specific frame index.
        """
        for name in self.keys():
            try:
                # Accesses its own data to yield the specific row
                yield name, self[name].loc[frame_index]
            except KeyError:
                # Gracefully skips objects that do not have data for the requested frame.
                continue
    
    @property
    def object_names(self) -> List[str]:
        """Returns the list of tracked object names."""
        return list(self.keys())