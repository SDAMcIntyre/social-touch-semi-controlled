# tracking_handlers.py

import cv2
import tkinter as tk
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from preprocessing.common.rgb_video_manager import RGBVideoManager

from .tracking_contracts import TrackingUIHandler, TrackingState, UserInteractionResult
from ..gui.frame_roi_square import FrameROISquare
from ..gui.video_frame_selector import VideoFrameSelector
from .video_review_manager import VideoReviewManager


class HeadlessUIHandler(TrackingUIHandler):
    """A UI handler that runs without any graphical interface, for batch processing."""
    
    def display_update(self, state: TrackingState) -> UserInteractionResult:
        """Prints status to the console instead of showing a window."""
        print(f"Frame: {state.frame_num}, Status: {state.status}, ROI: {state.roi}")
        return UserInteractionResult() # Default: do nothing

    def prompt_for_redefinition(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Cannot redefine ROI in headless mode. Tracking will continue as 'Failed'."""
        print("Tracking failed. Cannot redefine ROI in headless mode.")
        return None

    def handle_interrupt(self, vm: RGBVideoManager, all_results: Dict[int, Any], current_frame_num: int, frame_list: List[int]) -> UserInteractionResult:
        """Interrupts are not possible in headless mode."""
        print("User interrupt requested, but not supported in headless mode.")
        return UserInteractionResult()


class InteractiveGUIHandler(TrackingUIHandler):
    """
    A UI handler that uses OpenCV and Tkinter for a full interactive experience.
    This contains all the logic from the original class's display methods.
    """
    def __init__(self, title: str = "Object Tracker"):
        self.title = title

    def display_update(self, state: TrackingState) -> UserInteractionResult:
        """Displays the frame with tracking info and listens for key presses."""
        display_frame = cv2.cvtColor(state.frame_rgb, cv2.COLOR_RGB2BGR)
        color = VideoReviewManager.STATUS_COLORS.get(state.status.split(':')[0], VideoReviewManager.STATUS_COLORS["default"])

        if state.roi:
            x, y, w, h = state.roi
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(display_frame, (x, y), (x + w, y + h), color, 1)
            cv2.line(display_frame, (x, y + h), (x + w, y), color, 1)

        cv2.putText(display_frame, state.status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(display_frame, "Press ENTER to interrupt | 'q' to quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.title, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return UserInteractionResult(should_quit=True)
        if key == 13: # Enter key
            return UserInteractionResult(rewind_to_index=-1) # Signal an interrupt
        
        return UserInteractionResult()

    def prompt_for_redefinition(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Opens a GUI window for the user to draw a new ROI."""
        window_name = f"{self.title} - Redefine ROI"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        selector = FrameROISquare(frame_bgr, is_rgb=False, window_title=window_name)
        selector.run()
        roi_data = selector.get_roi_data()
        
        if roi_data and roi_data['width'] > 0 and roi_data['height'] > 0:
            return (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])
        return None

    def handle_interrupt(self, vm: RGBVideoManager, all_results: Dict[int, Any], current_frame_num: int, frame_list: List[int]) -> UserInteractionResult:
        """Handles user interruption to rewind and re-select an ROI."""
        print("\n--- Tracking Interrupted ---")
        
        start = max(0, current_frame_num - 150)
        end = min(current_frame_num + 150, vm.total_frames)
        frames_rgb = vm.get_frames_range(start, end)
        
        self._draw_tracking_on_frames(frames_rgb, start, all_results)

        root = tk.Tk()
        selector = VideoFrameSelector(
            root, frames_rgb,
            first_frame=0, last_frame=len(frames_rgb),
            current_frame_num=current_frame_num - start,
            title=self.title
        )
        root.mainloop()

        if selector.selected_frame_num is not None:
            new_frame_num = start + selector.selected_frame_num
            print(f"Restarting from frame {new_frame_num}. Please select a new ROI.")
            
            new_frame_rgb = vm.get_frame(new_frame_num)
            new_roi = self.prompt_for_redefinition(new_frame_rgb)

            if new_roi:
                try:
                    new_index = frame_list.index(new_frame_num)
                    return UserInteractionResult(rewind_to_index=new_index, new_roi=new_roi, new_status="Manual")
                except ValueError:
                    print(f"Error: Selected frame {new_frame_num} not in current pass.")
        
        return UserInteractionResult()

    def _draw_tracking_on_frames(self, frames_rgb: List[np.ndarray], start_frame_num: int, tracking_results: Dict[int, Any]):
        """Draws existing bounding boxes onto a list of frames."""
        for i, frame in enumerate(frames_rgb):
            frame_id = start_frame_num + i
            if frame_id in tracking_results:
                result = tracking_results[frame_id]
                if result['roi'] and result['status'] != "Failed":
                    x, y, w, h = result['roi']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    def teardown(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()