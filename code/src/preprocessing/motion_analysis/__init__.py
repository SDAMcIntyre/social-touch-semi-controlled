"""
Motion Analysis Public API.

This module exposes the core classes for hand tracking, motion management,
and tactile quantification analysis. It acts as a facade, decoupling external
consumers from the internal subdirectory structure.
"""

# -----------------------------------------------------------------------------
# Core Domain & Logic (Hand Tracking)
# -----------------------------------------------------------------------------
from .hand_tracking.models.hand_metadata import HandMetadataManager
from .hand_tracking.models.hand_tracking_data_manager import HandTrackingDataManager, MeshSequenceLoader
from .hand_tracking.models.hand_motion_manager import HandMotionManager

# -----------------------------------------------------------------------------
# Core Domain & Logic (Tactile Quantification)
# -----------------------------------------------------------------------------
from .tactile_quantification.core.objects_interaction_controller import (
    ObjectsInteractionController,
)

# -----------------------------------------------------------------------------
# Data Access & Infrastructure
# -----------------------------------------------------------------------------
from .hand_tracking.data_access.hand_metadata_filehandler import HandMetadataFileHandler
from .hand_tracking.hamer_liu_client.hamer_client_api import HamerClientAPI

# -----------------------------------------------------------------------------
# GUI & Visualization (Presentation Layer)
# -----------------------------------------------------------------------------
from .hand_tracking.gui.hand_model_selector import HandModelSelectorGUI
from .hand_tracking.gui.hand_mask_selector_gui import HandMaskSelectorGUI
from .hand_tracking.gui.handmeshes_selection_gui import HamerCheckupSelector
from .tactile_quantification.gui.objects_interaction_visualizer import (
    ObjectsInteractionVisualizer,
)

# -----------------------------------------------------------------------------
# Public API Export
# -----------------------------------------------------------------------------
__all__ = [
    # Hand Tracking - Models
    "HandMetadataManager",
    "HandTrackingDataManager",
    "MeshSequenceLoader",
    "HandMotionManager",
    # Hand Tracking - Infrastructure
    "HandMetadataFileHandler",
    "HamerClientAPI",
    # Hand Tracking - GUI
    "HandModelSelectorGUI",
    "HandMaskSelectorGUI",
    "HamerCheckupSelector",
    # Tactile Quantification - Core
    "ObjectsInteractionController",
    # Tactile Quantification - GUI
    "ObjectsInteractionVisualizer",
]




