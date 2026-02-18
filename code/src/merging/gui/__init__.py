"""
merging.gui
-----------
GUI components for visualising merged neural + Kinect recordings.
"""
from .neural_kinect_scene_viewer import NeuralKinectViewer
from .sticker_velocity_compass import StickerVelocityCompass

__all__ = [
    "NeuralKinectViewer",
    "StickerVelocityCompass",
]
