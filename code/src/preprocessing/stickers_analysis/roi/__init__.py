# preprocessing/stickers_analysis/roi/__init__.py

"""
This file exposes the public API for the ROI analysis sub-package.

By importing the key classes here, we allow users to access them directly
from the 'roi' namespace, decoupling their code from our internal file structure.
"""

# Import from sub-modules to "lift" them into this package's namespace

from .data_access.roi_manual_annotation_filehandler import ROIAnnotationFileHandler
from .models.roi_manual_annotation import ROIAnnotationManager, ROIProcessingStatus

from .data_access.roi_tracked_filehandler import ROITrackedFileHandler
from .models.roi_tracked_data import ROITrackedObjects, ROITrackedObject, ROITrackedStatus

from .core.review_tracking_orchestrator import TrackerReviewOrchestrator, TrackerReviewStatus
from .gui.review_tracking_gui import TrackerReviewGUI
from .gui.frame_roi_square import FrameROISquare
from .models.video_review_manager import VideoReviewManager

from .core.roi_tracking_orchestrator import TrackingOrchestrator
from .models.tracking_handlers import InteractiveGUIHandler

from .models.xyz_metadata_model import XYZMetadataModel
from .models.xyz_metadata_model import XYZMetadataConfig
from .models.xyz_metadata_manager import XYZMetadataManager
from .gui.xyz_monitor_gui import XYZVisualizationHandler
from .data_access.xyz_data_filehandler import XYZDataFileHandler
from .core.xyz_extracting_orchestrator import XYZStickerOrchestrator

# Optional: Define what gets imported with 'from . import *'
__all__ = [
    "ROIAnnotationFileHandler",
    "ROIAnnotationManager",
    "ROIProcessingStatus",

    "ROITrackedFileHandler",
    "ROITrackedObjects",
    "ROITrackedObject",
    "ROITrackedStatus",

    "TrackingOrchestrator",
    "InteractiveGUIHandler",

    "TrackerReviewOrchestrator", 
    "TrackerReviewStatus"
    "TrackerReviewGUI", 
    "FrameROISquare",
    "VideoReviewManager", 
]