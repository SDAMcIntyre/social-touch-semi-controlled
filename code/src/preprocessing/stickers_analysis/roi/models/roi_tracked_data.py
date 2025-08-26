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

# Define the schema for our tracking data
# This helps enforce consistency.
ROI_TRACKED_SCHEMA = {
    'frame_id': int,
    'roi_x': int,
    'roi_y': int,
    'roi_width': int,
    'roi_height': int,
    'status': str
}

# Define a type alias for clarity.
# This represents the data for a single tracked object.
ROITrackedObject = pd.DataFrame

# This represents the main dictionary holding all object data.
ROITrackedObjects = Dict[str, ROITrackedObject]
