# preprocessing/stickers_analysis/roi/__init__.py

"""
This file exposes the public API for the ROI analysis sub-package.

By importing the key classes here, we allow users to access them directly
from he 'roi' namespace, decoupling their code from .roiour internal file structure.
"""

from .roi.data_access.roi_manual_annotation_filehandler import ROIAnnotationFileHandler
from .roi.models.roi_manual_annotation import ROIAnnotationManager, ROIProcessingStatus

from .roi.data_access.roi_tracked_filehandler import ROITrackedFileHandler
from .roi.models.roi_tracked_data import ROITrackedObjects, ROITrackedObject, ROITrackedStatus

from .roi.core.review_tracking_orchestrator import TrackerReviewOrchestrator, TrackerReviewStatus
from .roi.gui.review_tracking_gui import TrackerReviewGUI
#from .roi...common.gui.frame_roi_square import FrameROISquare
from .roi.models.video_review_manager import VideoReviewManager

from .roi.core.roi_tracking_orchestrator import TrackingOrchestrator
from .roi.models.tracking_handlers import InteractiveGUIHandler

from .xyz.models.xyz_metadata_model import XYZMetadataModel, XYZMetadataConfig
from .xyz.data_access.xyz_metadata_filehandler import XYZMetadataFileHandler
from .xyz.gui.xyz_monitor_gui import XYZVisualizationHandler
from .xyz.data_access.xyz_data_filehandler import XYZDataFileHandler
from .xyz.core.xyz_extracting_orchestrator import XYZStickerOrchestrator


# Define what gets imported with 'from .roi. import *'
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
    "TrackerReviewStatus",
    "TrackerReviewGUI", 
    #"FrameROISquare",
    "VideoReviewManager", 
]