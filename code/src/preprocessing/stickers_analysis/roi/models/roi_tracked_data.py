# data_model.py

from typing import Dict
import pandas as pd
from enum import Enum

# --- 1. Status Enum (replaces "magic strings") ---
class ROITrackedStatus(Enum):
    """Defines status constants for tracking and annotation."""
    LABELED = "Labeled"
    INITIAL_ROI = "Initial ROI"
    TRACKING = "Tracking"

# Define the schema for our tracking data.
# We use pandas' nullable dtypes ('Int64') to gracefully handle missing data
# in integer columns without causing crashes during type conversion.
ROI_TRACKED_SCHEMA = {
    'frame_id': 'Int64',
    'roi_x': 'Int64',
    'roi_y': 'Int64',
    'roi_width': 'Int64',
    'roi_height': 'Int64',
    'status': str
}

# Define a type alias for clarity.
# This represents the data for a single tracked object.
ROITrackedObject = pd.DataFrame

# This represents the main dictionary holding all object data.
ROITrackedObjects = Dict[str, ROITrackedObject]
