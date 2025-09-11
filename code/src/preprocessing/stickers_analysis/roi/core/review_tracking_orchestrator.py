import cv2
from enum import Enum
from typing import Dict, List, Any, Optional


from preprocessing.common import (
    VideoMP4Manager
)

from ..gui.review_tracking_gui import (
    TrackerReviewGUI,
    FrameAction
)
from ..models.roi_tracked_data import ROITrackedObjects
from ..models.roi_review_shared_types import FrameAction, FrameMark

class TrackerReviewStatus(Enum):
    """Defines status constants for tracking and annotation."""
    PROCEED = "proceed"
    COMPLETED = "completed"
    UNPERFECT = "unperfect"
    UNDEFINED = "undefined"

# -----------------------------------------------------------

class TrackerReviewOrchestrator:
    """
    Manages the application's state and logic. Connects the Model and the View.
    This class is completely UI-agnostic.
    """
    def __init__(self, model: VideoMP4Manager, view: TrackerReviewGUI, tracking_history: Optional[ROITrackedObjects] = None):
        """
        Initializes the orchestrator.

        Args:
            model (VideoMP4Manager): The data model for video access.
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
        
        # --- REFACTORED: Use a single dictionary for all marked frames ---
        self.marked_frames: Dict[int, FrameMark] = {}
        
        self.status = TrackerReviewStatus.UNDEFINED
        self._update_job = None

    def run(self) -> tuple[str, Dict[int, List[str]], Dict[int, List[str]]]:
        """
        Starts the application's main loop.

        Returns:
            A tuple containing the final status (str) and two dictionaries:
            one for marked frames for labeling and one for marked frames for deletion.
            NOTE: The internal architecture is refactored, but this public method's
                  return signature is preserved for backward compatibility.
        """
        self.view.setup_ui()
        self.update_view_full()
        self.view.start_mainloop()
            
        # Case 1: tracking_history IS NOT present. Return simple lists of frame numbers.
        if not self.tracking_history:
            frames_for_labeling = []
            frames_for_deleting = []
            for frame_num, mark in self.marked_frames.items():
                if mark.action == FrameAction.LABEL:
                    frames_for_labeling.append(frame_num)
                elif mark.action == FrameAction.DELETE:
                    frames_for_deleting.append(frame_num)

            return (self.status, 
                    sorted(frames_for_labeling), 
                    sorted(frames_for_deleting))

        # Case 2: tracking_history IS present. Return dicts with object IDs.
        else:
            frames_for_labeling = {}
            frames_for_deleting = {}
            for frame_num, mark in self.marked_frames.items():
                if mark.action == FrameAction.LABEL:
                    frames_for_labeling[frame_num] = mark.object_ids
                elif mark.action == FrameAction.DELETE:
                    frames_for_deleting[frame_num] = mark.object_ids
            
            return (self.status, 
                    dict(sorted(frames_for_labeling.items())), 
                    dict(sorted(frames_for_deleting.items())))
        

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

    def mark_frame(self, action: FrameAction, objects_to_mark: List[str]):
        """
        Marks the current frame for a specific action.

        A frame can only have one action marked at a time. This method will
        show a pop-up if the frame is already marked with a different action.

        Args:
            action (FrameAction): The action to mark the frame with (e.g., LABEL, DELETE).
            objects_to_mark (List[str]): A list of object IDs to associate with the mark.
        """
        frame_num = self.current_frame_num

        if frame_num in self.marked_frames:
            existing_mark = self.marked_frames[frame_num]
            # Allow re-marking with the same action (e.g., to update object_ids),
            # but prevent marking with a different action.
            if existing_mark.action != action:
                self.view.pop_up_window(
                    message=f"Frame {frame_num} is already marked for '{existing_mark.action.value}'.\n"
                            "Please remove that mark first before adding a new one."
                )
                return

        self.marked_frames[frame_num] = FrameMark(action=action, object_ids=objects_to_mark)
        self.status = TrackerReviewStatus.UNPERFECT
        
        # --- Note: The View needs a corresponding unified update method ---
        self.view.update_marked_list(self.marked_frames)

        self.view.update_finish_buttons(bool(self.marked_frames))

    def remove_mark(self, frame_id_to_delete: int):
        """
        Removes a marked frame from the list by its UI index.

        Args:
            index_in_list (int): The index of the item in the displayed listbox,
                                 which is assumed to be sorted by frame number.
        """
        # The list displayed in the UI should be built from the sorted keys
        # of the marked_frames dictionary.
        sorted_keys = sorted(self.marked_frames.keys())
        if frame_id_to_delete in sorted_keys:
            del self.marked_frames[frame_id_to_delete]
        
        # --- Note: The View needs a corresponding unified update method ---
        self.view.update_marked_list(self.marked_frames)
        self.view.update_finish_buttons(bool(self.marked_frames))
        
    def is_marked(self, frame_id: int) -> bool:
        """
        Checks if a specific frame ID is in the list of marked frames.

        Args:
            frame_id (int): The frame number to check.

        Returns:
            bool: True if the frame is marked, False otherwise.
        """
        return frame_id in self.marked_frames

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
        frame = self.model[self.current_frame_num].copy()
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