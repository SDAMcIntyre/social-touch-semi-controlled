import time
import cv2
import pandas as pd
from enum import Enum
from typing import Dict, List, Any

from ..gui.review_tracking_gui import TrackerReviewGUI
from ..models.video_review_manager import VideoReviewManager


class TrackerReviewStatus(Enum):
    """Defines status constants for tracking and annotation."""
    PROCEED = "proceed"
    COMPLETED = "completed"
    UNPERFECT = "unperfect"
    UNDEFINED = "undefined"

ROITrackedObject = pd.DataFrame
ROITrackedObjects = Dict[str, ROITrackedObject]

class TrackerReviewOrchestrator:
    """
    Manages the application's state and logic. Connects the Model and the View.
    This class is completely UI-agnostic.
    """
    def __init__(self, model: VideoReviewManager, view: TrackerReviewGUI, tracking_history: ROITrackedObjects = None):
        """
        Initializes the orchestrator.

        Args:
            model (VideoReviewManager): The data model for video access.
            view (TrackerReviewGUI): The UI view.
            tracking_history (ROITrackedObjects, optional): A dictionary where keys are object IDs (str)
                                                             and values are pandas DataFrames containing
                                                             the tracking history for that object.
                                                             The DataFrame index should be the frame number.
                                                             Defaults to None.
        """
        self.model = model
        self.view = view
        self.view.controller = self

        self.tracking_history = tracking_history if tracking_history is not None else {}
        self.object_names = list(self.tracking_history.keys())

        # --- State variables ---
        self.current_frame_num = 0
        self.is_paused = True
        self.playback_speed = 1.0
        self.base_delay_ms = 30
        
        self.marked_for_labeling: Dict[int, List[str]] = {}
        
        self.status = TrackerReviewStatus.UNDEFINED
        self._update_job = None

    def run(self) -> tuple[str, Dict[int, List[str]]]:
        """
        Starts the application's main loop.

        Returns:
            A tuple containing the final status (str) and a dictionary of
            marked frames, where each key is a frame number and the value is a
            list of associated string reasons.
        """
        self.view.setup_ui()
        self.update_view_full()
        self.view.start_mainloop()
        
        # MODIFIED: Return the dictionary, sorted by frame number for consistency.
        return self.status, dict(sorted(self.marked_for_labeling.items()))

    def toggle_play_pause(self):
        """Toggles the playback state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            if self._update_job:
                self.view.root.after_cancel(self._update_job)
                self._update_job = None
        else:
            if self.current_frame_num >= self.model.total_frames - 1:
                self.seek_to_frame(0)
            self._schedule_next_frame()
        self.view.update_play_pause_button(self.is_paused)

    def seek_to_frame(self, frame_num: int):
        """Jumps to a specific frame."""
        frame_num = int(frame_num)
        self.current_frame_num = max(0, min(frame_num, self.model.total_frames - 1))
        if not self.is_paused:
            self.toggle_play_pause()
        self.update_view_full()

    def change_speed(self, speed: float):
        """Updates the playback speed."""
        self.playback_speed = float(speed)
        self.view.update_speed_label(self.playback_speed)

    def mark_current_frame(self, reasons: List[str]):
        """
        Adds or updates the current frame in the marked list with associated reasons.
        The UI is responsible for collecting and passing the 'reasons'.

        Args:
            reasons (List[str]): A list of strings explaining why the frame is marked.
        """
        self.marked_for_labeling[self.current_frame_num] = reasons
        self.status = TrackerReviewStatus.UNPERFECT
        
        # Pass the sorted list of frame numbers (keys) to the view.
        # This keeps the view's interface simple.
        self.view.update_marked_list(sorted(self.marked_for_labeling.keys()))
        self.view.update_finish_buttons(self.marked_for_labeling)

    # MODIFIED: The logic now works with the dictionary's keys.
    def delete_marked_frame(self, index_in_list: int):
        """Deletes a frame from the marked list by its displayed listbox index."""
        sorted_keys = sorted(self.marked_for_labeling.keys())
        if 0 <= index_in_list < len(sorted_keys):
            frame_to_delete = sorted_keys[index_in_list]
            del self.marked_for_labeling[frame_to_delete]
        
        # Update the view with the new sorted list of marked frame numbers.
        self.view.update_marked_list(sorted(self.marked_for_labeling.keys()))
        self.view.update_finish_buttons(self.marked_for_labeling)

    def finish_as_valid(self):
        """Sets the final status to 'completed' and quits."""
        self.status = TrackerReviewStatus.COMPLETED
        self.view.quit()

    def finish_and_proceed(self):
        """Sets the final status to 'proceed' and quits."""
        self.status = TrackerReviewStatus.PROCEED
        self.view.quit()
        
    def get_frame_with_overlays(self):
        """
        Gets the current frame and draws tracking data for all objects.
        """
        frame = self.model.get_frame(self.current_frame_num).copy()
        for obj_id, df_history in self.tracking_history.items():
            if self.current_frame_num in df_history.index:
                object_data_at_frame = df_history.loc[self.current_frame_num]
                self._draw_overlay(frame, object_data_at_frame.to_dict())
        return frame
        
    def _draw_overlay(self, frame, obj_result: Dict[str, Any]):
        """Helper to draw a single object's bounding box and status."""
        STATUS_COLORS = {
            "Tracking": (0, 255, 0), "Out of Frame": (0, 0, 255),
            "Failure": (0, 165, 255), "Re-initialized": (0, 165, 255),
            "default": (255, 0, 0)
        }
        status_text = obj_result.get("status", "No Data")
        box = obj_result.get("box")
        color = STATUS_COLORS.get(status_text.split(':')[0], STATUS_COLORS["default"])
        if box and len(box) == 4:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            obj_id = obj_result.get("id", "")
            label = f"{obj_id}: {status_text}" if obj_id else status_text
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _schedule_next_frame(self):
        """The main playback loop logic."""
        if self.is_paused:
            return

        self.current_frame_num += 1
        if self.current_frame_num >= self.model.total_frames:
            self.current_frame_num = self.model.total_frames - 1
            self.toggle_play_pause()
            self.update_view_full()
            return

        self.update_view_full()
        
        delay_ms = max(1, int(self.base_delay_ms / self.playback_speed))
        self._update_job = self.view.root.after(delay_ms, self._schedule_next_frame)

    def update_view_full(self):
        """Centralized method to tell the view to update all its components."""
        frame_to_display = self.get_frame_with_overlays()
        self.view.update_video_display(frame_to_display)
        self.view.update_timeline(self.current_frame_num)
        self.view.update_frame_label(self.current_frame_num, self.model.total_frames)