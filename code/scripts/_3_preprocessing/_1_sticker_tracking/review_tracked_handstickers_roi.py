# Standard library imports
import os
from typing import Optional, List, Dict, Tuple
import pandas as pd

from preprocessing.common.gui.frame_roi_square import FrameROISquare

from preprocessing.stickers_analysis.roi import (
    ROIProcessingStatus,
    ROIAnnotationFileHandler,
    ROIAnnotationManager,

    ROITrackedObjects,
    ROITrackedFileHandler,

    TrackerReviewOrchestrator, 
    TrackerReviewStatus,
    TrackerReviewGUI,
    VideoReviewManager
)

# --- 2. Embedded Color Configuration ---
# The color map is now a simple dictionary within the .py file.
# Format: color_name -> (B, G, R) tuple
COLOR_CONFIG = {
  "yellow": (0, 255, 255),
  "blue": (255, 0, 0),
  "green": (0, 255, 0),
  "red": (0, 0, 255),
  "white": (255, 255, 255),
  "purple": (255, 0, 255),
  "cyan": (255, 255, 0)
}

def get_object_colors(obj_name: str) -> Optional[Dict[str, Tuple[int, int, int]]]:
    """
    Gets live and final drawing colors for a given object name from the embedded config.
    """
    try:
        color_name = obj_name.split('_')[-1]
        if color_name in COLOR_CONFIG:
            live_color = COLOR_CONFIG[color_name]
            # Calculate a darker version for the 'final' color
            final_color = tuple(c // 2 for c in live_color)
            return {'live': live_color, 'final': final_color}
    except IndexError:
        pass  # Handles cases where obj_name has no underscore
    print(f"Warning: Color for '{obj_name}' not found in COLOR_CONFIG.")
    return None

# --- 4. Refactored Functions (Controllers) ---
def annotate_rois_interactively(
    video_manager: VideoReviewManager, 
    annotation_manager: ROIAnnotationManager, 
    marked_frames: Dict[int, List[str]]
):
    """
    Creates or updates a metadata file with ROIs using an interactive UI.
    This function now uses the embedded color configuration.
    """
    
    for frame_id, object_names in marked_frames.items():
        frame = video_manager.get_frame(frame_id, ignore_tracking_history=True)

        for object_name in object_names:
            title = f"Tracker for '{object_name}' on Frame {frame_id}"
            tracker_ui = FrameROISquare(frame, is_rgb=False, window_title=title)
            
            # Get colors from the new helper function
            colors = get_object_colors(object_name)
            if colors:
                tracker_ui.set_color_live(colors['live'])
                tracker_ui.set_color_final(colors['final'])

            tracker_ui.run()
            tracking_data = tracker_ui.get_roi_data()
            if tracking_data:
                x = tracking_data["x"]
                y = tracking_data["y"]
                w = tracking_data["width"]
                h = tracking_data["height"]
                annotation_manager.set_roi(object_name, frame_id, x, y, w, h)
                print(f"--- ✅ ROI data for '{object_name}' on frame {frame_id} extracted ---")
            else:
                print(f"--- ⚠️ No ROI was set for '{object_name}' on this frame. ---")
    

def get_unique_frame_ids(annotation_data):
    """
    Extracts all unique frame_ids from the AnnotationData object.

    Args:
        annotation_data: An object containing a dictionary of TrackedObject instances,
                         where each has a 'rois' DataFrame with a 'frame_id' column.

    Returns:
        A sorted list of unique integer frame IDs.
    """
    # 1. Create a list of all the 'rois' DataFrames.
    # This avoids creating a list if there are no objects to track.
    list_of_dfs = [
        tracked_obj.rois 
        for tracked_obj in annotation_data.objects_to_track.values()
        if not tracked_obj.rois.empty
    ]

    # If there are no DataFrames to process, return an empty list.
    if not list_of_dfs:
        return []

    # 2. Concatenate all DataFrames into a single one.
    # ignore_index=True is good practice for performance when the original
    # indices are not needed.
    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # 3. Get the unique values from the 'frame_id' column.
    # .unique() returns a NumPy array.
    unique_ids_array = combined_df['frame_id'].unique()

    # 4. Sort the array and convert it to a standard Python list.
    unique_ids_list = sorted(unique_ids_array.tolist())
    
    return unique_ids_list


def remove_frames_from_metadata(
        annotation_manager: ROIAnnotationManager, 
        frames_for_deleting: Dict[int, List[str]]
):  
    for frame_id, object_names in frames_for_deleting.items():
        for object_name in object_names:
            annotation_manager.remove_roi_ifexists(object_name, frame_id)

def review_tracking(
        video_manager: VideoReviewManager, 
        annotation_manager: ROIAnnotationManager, 
        tracked_objs: ROITrackedObjects, 
        title: str
) -> tuple[str, Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Identifies frames that require manual review based on object tracking status.

    This function processes a dictionary of pandas DataFrames, where each DataFrame
    contains the tracking history for a specific object. It finds all unique frames
    where any object was manually labeled ('labeled' or 'initial_roi').

    Args:
        video_path: The path to the video file being analyzed.
        tracked_objs: A dictionary where keys are unique object names (str) and
                      values are pandas DataFrames. Each DataFrame must contain
                      at least a 'frame_id' and a 'status' column.

    Returns:
        A tuple containing the original video path and a sorted list of unique
        frame indices that require manual review.
    """
    annotated_frame_ids = get_unique_frame_ids(annotation_manager.data)

    print("\nLaunching Tkinter Video Player...")
    view = TrackerReviewGUI(title=title, landmarks=annotated_frame_ids, windowState='maximized')
    controller = TrackerReviewOrchestrator(model=video_manager, view=view, tracking_history=tracked_objs)
    final_status, frames_for_labeling, frames_for_deleting = controller.run()

    print("\n--- Results from player session ---")
    print(f"Final Status: {final_status}")
    print(f"Frames marked for labeling: {frames_for_labeling}")
    
    return final_status, frames_for_labeling, frames_for_deleting


def review_tracked_objects_in_video(
    video_path: str, 
    metadata_path: str,
    tracked_data_path: str
) -> Optional[str]:
    """
    Main pipeline to review tracking, validate, or trigger re-annotation.
    """
    annotation_data_iohandler = ROIAnnotationFileHandler.load(metadata_path)
    annotation_manager = ROIAnnotationManager(annotation_data_iohandler)

    if annotation_manager.are_no_objects_with_status(ROIProcessingStatus.TO_BE_REVIEWED):
        print(f"No object has been assigned to be reviewed: either ✅ Tracking is completed or automatic algorithm has to process it.")
        return tracked_data_path
    
    tracked_data_iohandler = ROITrackedFileHandler(tracked_data_path)
    tracked_data = tracked_data_iohandler.load_all_data()

    video_manager = VideoReviewManager(video_path, tracking_history=tracked_data)
    
    final_status, frames_for_labeling, frames_for_deleting = review_tracking(video_manager, annotation_manager, tracked_data, title=os.path.basename(video_path))
    
    if final_status == TrackerReviewStatus.UNDEFINED or final_status == TrackerReviewStatus.UNPERFECT:
        print("Review session ended without validation or marking new frames.")
        return None
    
    elif final_status == TrackerReviewStatus.COMPLETED:
        annotation_manager.update_all_status(ROIProcessingStatus.COMPLETED)

    elif final_status == TrackerReviewStatus.PROCEED:
        print(f"⚠️ User marked {len(frames_for_labeling)} frames for re-annotation. Launching interactive tool.")
        remove_frames_from_metadata(annotation_manager, frames_for_deleting)
        annotate_rois_interactively(video_manager, annotation_manager, frames_for_labeling)
        annotation_manager.update_all_status(ROIProcessingStatus.TO_BE_PROCESSED)
        print("\nRe-annotation complete. The process may need to be run again to verify the new tracking.")
    
    ROIAnnotationFileHandler.save(metadata_path, annotation_manager.data)

    return tracked_data_path