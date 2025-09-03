# tracking_orchestrator.py
import cv2
import numpy as np
import pandas as pd  # Added pandas import
from typing import List, Dict, Any, Optional, Tuple, Iterable

from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager
from ..models.tracking_contracts import TrackingUIHandler, TrackingState
from ..models.tracker_wrapper import TrackerWrapper
from ..models.tracking_handlers import HeadlessUIHandler, InteractiveGUIHandler

class TrackingOrchestrator:
    """
    Orchestrates the tracking process by managing video I/O and tracker state.
    This class is decoupled from the UI and delegates all user interaction
    to a provided UI handler.
    """
    def __init__(self, video_path: str, use_gui: bool = False, ui_handler: TrackingUIHandler = None):
        self.video_path = video_path

        if ui_handler:
            use_gui = True
            
        if use_gui:
            if ui_handler is None:
                ui_handler = InteractiveGUIHandler(title="Tracker Monitor")
            self.ui_handler = ui_handler
        else:
            self.ui_handler = HeadlessUIHandler()

        self.tracking_results: Dict[int, Dict[str, Any]] = {}

    # MODIFIED: The method now accepts and returns a pandas DataFrame.
    def run(self, labeled_rois: pd.DataFrame, search_area_expansion: Optional[int] = None) -> pd.DataFrame:
        """
        Starts the tracking process using an initial set of labeled ROIs
        provided as a DataFrame.

        Args:
            labeled_rois (pd.DataFrame): A DataFrame with columns 
                ['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height'].
            search_area_expansion (Optional[int]): The pixel amount to expand the search area.

        Returns:
            pd.DataFrame: A DataFrame containing the tracking results with columns
                ['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height', 'status'].
        """
        if labeled_rois.empty:
            print("Error: No labeled ROIs provided.")
            return pd.DataFrame()

        try:
            with VideoMP4Manager(self.video_path) as vm:
                # MODIFICATION: Prepare initial data from the DataFrame.
                # Sort by frame_id to ensure the first row is the earliest annotation.
                labeled_rois = labeled_rois.sort_values(by='frame_id').reset_index(drop=True)
                
                start_roi_row = labeled_rois.iloc[0]
                start_frame_num = int(start_roi_row['frame_id'])
                start_roi = (
                    int(start_roi_row['roi_x']), int(start_roi_row['roi_y']),
                    int(start_roi_row['roi_width']), int(start_roi_row['roi_height'])
                )
                
                start_frame_img_rgb = vm.get_frame(start_frame_num)
                if start_frame_img_rgb is None:
                    print(f"Error: Could not read start frame {start_frame_num}.")
                    return pd.DataFrame()

                self.tracking_results = {start_frame_num: {'roi': start_roi, 'status': 'Initial'}}
                
                # MODIFICATION: Create a map of other labeled ROIs for quick lookup.
                remaining_rois_df = labeled_rois.iloc[1:]
                labeled_rois_map = {
                    int(row['frame_id']): (
                        int(row['roi_x']), int(row['roi_y']),
                        int(row['roi_width']), int(row['roi_height'])
                    ) for _, row in remaining_rois_df.iterrows()
                }

                # --- Forward Pass ---
                print("\n--- Starting Forward Pass ---")
                forward_range = range(start_frame_num + 1, vm.total_frames)
                self._track_pass(vm, start_frame_num, start_roi, labeled_rois_map, forward_range, search_area_expansion)

                # --- Backward Pass ---
                if start_frame_num > 0:
                    print("\n--- Starting Backward Pass ---")
                    backward_range = range(start_frame_num - 1, -1, -1)
                    self._track_pass(vm, start_frame_num, start_roi, labeled_rois_map, backward_range, search_area_expansion)

        except SystemExit as e:
            print(f"Process exited: {e}")
        finally:
            self.ui_handler.teardown()

        # --- Finalize and Return ---
        # MODIFICATION: Convert the results dictionary to a pandas DataFrame.
        results_list = []
        for frame_id in sorted(self.tracking_results.keys()):
            data = self.tracking_results[frame_id]
            roi = data.get('roi')
            status = data.get('status')
            if roi:
                results_list.append({
                    'frame_id': frame_id,
                    'roi_x': roi[0],
                    'roi_y': roi[1],
                    'roi_width': roi[2],
                    'roi_height': roi[3],
                    'status': status
                })
            else:  # Handle cases where tracking failed and ROI is None
                results_list.append({
                    'frame_id': frame_id,
                    'roi_x': np.nan, 'roi_y': np.nan,
                    'roi_width': np.nan, 'roi_height': np.nan,
                    'status': status
                })
        return pd.DataFrame(results_list)

    def _track_pass(self, vm: VideoMP4Manager, start_frame_num: int, start_roi: Tuple, labeled_rois: Dict, frame_range: Iterable[int], expansion: Optional[int]):
        """Performs a single tracking pass, delegating all UI to the handler."""
        tracker = TrackerWrapper()
        initial_frame_rgb = vm.get_frame(start_frame_num)
        tracker.init(cv2.cvtColor(initial_frame_rgb, cv2.COLOR_RGB2BGR), start_roi)
        
        last_successful_roi = start_roi
        frame_list = list(frame_range)
        i = 0
        while i < len(frame_list):
            frame_num = frame_list[i]
            
            current_frame_rgb = vm.get_frame(frame_num)
            if current_frame_rgb is None:
                print(f"Warning: Could not read frame {frame_num}. Skipping.")
                i += 1
                continue

            # --- Core Tracking Logic ---
            if frame_num in labeled_rois:
                roi = labeled_rois[frame_num]
                status = "Labeled"
                tracker.init(cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2BGR), roi)
            else:
                roi, status = self._update_tracker_on_frame(tracker, current_frame_rgb, last_successful_roi, expansion, vm)
            
            if roi:
                last_successful_roi = roi
            self.tracking_results[frame_num] = {'roi': roi, 'status': status}

            # --- Delegate to UI Handler ---
            state = TrackingState(frame_num, current_frame_rgb, roi, status)
            interaction = self.ui_handler.display_update(state)

            if interaction.should_quit:
                raise SystemExit("Tracking stopped by user.")
            
            if interaction.rewind_to_index is not None: # User wants to interrupt
                user_choice = self.ui_handler.handle_interrupt(vm, self.tracking_results, frame_num, frame_list)
                if user_choice.rewind_to_index is not None:
                    i = user_choice.rewind_to_index
                    new_roi = user_choice.new_roi
                    # Prune future results and update state
                    self.tracking_results = {k: v for k, v in self.tracking_results.items() if k < frame_list[i]}
                    self.tracking_results[frame_list[i]] = {'roi': new_roi, 'status': user_choice.new_status}
                    last_successful_roi = new_roi
                    # Re-initialize the tracker
                    new_start_frame_rgb = vm.get_frame(frame_list[i])
                    tracker.init(cv2.cvtColor(new_start_frame_rgb, cv2.COLOR_RGB2BGR), new_roi)
                    continue # Restart loop at the new index

            i += 1
    
    def _update_tracker_on_frame(self, tracker, frame_rgb, last_roi, expansion, vm):
        """Processes a single frame, returning the new ROI and status."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if not frame_bgr.any():
            return last_roi, "Black Frame"

        frame_to_track, (offset_x, offset_y) = frame_bgr, (0, 0)
        if expansion is not None and last_roi is not None:
            frame_to_track, (offset_x, offset_y) = self._crop_search_area(frame_bgr, last_roi, expansion, vm)

        success, box = tracker.update(frame_to_track)

        if success:
            final_box = (box[0] + offset_x, box[1] + offset_y, box[2], box[3])
            status = self._get_status(True, final_box, vm)
            return final_box, status
        else:
            new_roi = self.ui_handler.prompt_for_redefinition(frame_rgb)
            if new_roi:
                tracker.init(frame_bgr, new_roi)
                return new_roi, "Manual"
            return None, "Failed"

    # --- Helper Methods (unchanged, remain in orchestrator) ---
    def _crop_search_area(self, frame, roi, expansion, vm):
        x, y, w, h = roi
        center_x, center_y = x + w // 2, y + h // 2
        size = max(w, h) + 2 * expansion
        offset_x = max(0, center_x - size // 2)
        offset_y = max(0, center_y - size // 2)
        end_x = min(vm.frame_width, offset_x + size)
        end_y = min(vm.frame_height, offset_y + size)
        cropped_frame = frame[offset_y:end_y, offset_x:end_x]
        return cropped_frame, (offset_x, offset_y)

    # REMOVED: The _standardize_roi method is no longer needed as the
    # input DataFrame enforces a standard structure.

    @staticmethod
    def _get_status(is_tracking: bool, box: Tuple, vm: VideoMP4Manager) -> str:
        if not is_tracking: return "Failure"
        x, y, w, h = box
        if x <= 0 or y <= 0 or (x + w) >= vm.frame_width or (y + h) >= vm.frame_height:
            return "Out of Frame"
        return "Tracking"