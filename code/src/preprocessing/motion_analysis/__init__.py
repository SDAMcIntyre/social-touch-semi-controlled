# preprocessing/motion_analysis/__init__.py

"""
This file exposes the public API for the tactile quantification analysis sub-package.

By importing the key classes here, we allow users to access them directly
from the 'tactile_quantification' namespace, decoupling their code from our internal file structure.
"""

# Import from sub-modules to "lift" them into this package's namespace
from .hand_tracking.models.hand_motion import HandMotion
from .hand_tracking.models.hand_metadata import HandMetadataManager
from .hand_tracking.data_access.hand_metadata_filehandler import HandMetadataFileHandler
from .hand_tracking.gui.hand_model_selector import HandModelSelectorGUI
from .hand_tracking.hamer_liu_client.hamer_client_api import HamerClientAPI

from .tactile_quantification.core.objects_interaction_controller import ObjectsInteractionController
from .tactile_quantification.gui.objects_interaction_visualizer import ObjectsInteractionVisualizer



__all__ = [
    "ObjectsInteractionController",

]
