# preprocessing/stickers_analysis/roi/__init__.py

"""
This file exposes the public API for the ROI analysis sub-package.

By importing the key classes here, we allow users to access them directly
from he 'roi' namespace, decoupling their code from .roiour internal file structure.
"""

from .roi.data_access.roi_manual_annotation_filehandler import ROIAnnotationFileHandler
from .roi.models.roi_manual_annotation import ROIAnnotationManager, ROIProcessingStatus
from .roi.models.video_review_manager import VideoReviewManager

from .roi.data_access.roi_tracked_filehandler import ROITrackedFileHandler
from .roi.models.roi_tracked_data import ROITrackedObjects, ROITrackedObject, ROITrackedStatus

from .roi.core.review_tracking_orchestrator import TrackerReviewOrchestrator, TrackerReviewStatus
from .roi.gui.review_tracking_gui import TrackerReviewGUI
from .roi.models.video_review_manager import VideoReviewManager

from .roi.core.roi_tracking_orchestrator import TrackingOrchestrator
from .roi.models.tracking_handlers import InteractiveGUIHandler

# colorspace definition 
from .ellipse.data_access.color_space_filehandler import ColorSpaceFileHandler
from .ellipse.models.color_space_manager import ColorSpaceManager
from .ellipse.models.color_space_model import ColorSpace, ColorSpaceDefault, ColorSpaceStatus
# correlation map of the colorspace
from .ellipse.models.color_family_model import ColorFamilyModel
from .ellipse.gui.color_correlation_visualiser import ColorCorrelationVisualizer
from .ellipse.gui.frame_roi_color_gui import FrameROIColor
from .ellipse.gui.threshold_selector_tool_gui import ThresholdSelectorTool
# fit ellipses on correlation map
from .ellipse.data_access.fitted_ellipses_filehandler import FittedEllipsesFileHandler
from .ellipse.models.fitted_ellipses_manager import FittedEllipsesManager
from .ellipse.gui.ellipse_fit_view_gui import EllipseFitViewGUI
# Summary of the ellipses and rois
from .common.data_access.consolidated_tracks_filehandler import ConsolidatedTracksFileHandler
from .common.models.consolidated_tracks_manager import ConsolidatedTracksManager
from .common.gui.consolidated_tracks_gui import ConsolidatedTracksReviewGUI

from .xyz.models.xyz_metadata_model import XYZMetadataModel
from .xyz.data_access.xyz_metadata_filehandler import XYZMetadataFileHandler
from .xyz.data_access.xyz_data_filehandler import XYZDataFileHandler
from .xyz.core.xyz_extractor_factory import XYZExtractorFactory
from .xyz.core.xyz_extracting_orchestrator import XYZStickerOrchestrator

from .common.models.tracked_data_interface import TrackedDataInterface

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
    "VideoReviewManager", 
]