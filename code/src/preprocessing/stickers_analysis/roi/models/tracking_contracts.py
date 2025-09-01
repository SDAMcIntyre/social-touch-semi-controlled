# tracking_contracts.py

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager

@dataclass
class TrackingState:
    """A dataclass to hold the state of tracking at a specific frame."""
    frame_num: int
    frame_rgb: np.ndarray
    roi: Optional[Tuple[int, int, int, int]]
    status: str

@dataclass
class UserInteractionResult:
    """A dataclass to hold the result of a user interaction."""
    rewind_to_index: Optional[int] = None
    new_roi: Optional[Tuple[int, int, int, int]] = None
    new_status: Optional[str] = None
    should_quit: bool = False

class TrackingUIHandler(ABC):
    """
    Abstract base class defining the interface for UI interaction during tracking.
    This allows decoupling the tracking logic from the GUI implementation.
    """
    @abstractmethod
    def display_update(self, state: TrackingState) -> UserInteractionResult:
        """
        Display the current tracking state to the user and handle real-time input.
        
        Returns:
            A UserInteractionResult indicating if the user wants to interrupt or quit.
        """
        pass

    @abstractmethod
    def prompt_for_redefinition(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Prompt the user to define a new ROI when tracking fails.

        Returns:
            The new ROI tuple or None if the user cancels.
        """
        pass

    @abstractmethod
    def handle_interrupt(self, vm: VideoMP4Manager, all_results: Dict[int, Any], current_frame_num: int, frame_list: List[int]) -> UserInteractionResult:
        """
        Handle a major user interruption (e.g., pressing Enter to rewind).
        
        Returns:
            A UserInteractionResult with details to rewind the tracking process.
        """
        pass
        
    def teardown(self):
        """Clean up any resources, like closing windows."""
        pass