# Standard library imports
import os
from typing import Optional, List, Dict, Tuple
import pandas as pd
from pathlib import Path

from preprocessing.common.gui.frame_roi_square import FrameROISquare

from preprocessing.stickers_analysis import (
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
                annotation_manager.update_status(object_name, ROIProcessingStatus.TO_BE_PROCESSED)
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
            annotation_manager.remove_roi_if_exists(object_name, frame_id)

def update_ignore_frames_in_metadata(
    annotation_manager: ROIAnnotationManager,
    ignore_starts: Dict[int, List[str]],
    ignore_stops: Dict[int, List[str]]
):
    """
    Updates the annotation manager with ignore start and stop events.
    """
    # Process Ignore Start
    for frame_id, object_names in ignore_starts.items():
        for object_name in object_names:
            annotation_manager.set_ignore_start(object_name, frame_id)
            print(f"--- 🚫 Ignore START set for '{object_name}' at frame {frame_id}")

    # Process Ignore Stop
    for frame_id, object_names in ignore_stops.items():
        for object_name in object_names:
            annotation_manager.set_ignore_stop(object_name, frame_id)
            print(f"--- 🟢 Ignore STOP set for '{object_name}' at frame {frame_id}")


def review_tracking(
        video_manager: VideoReviewManager, 
        annotation_manager: ROIAnnotationManager, 
        tracked_objs: ROITrackedObjects, 
        title: str
) -> tuple[str, Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Identifies frames that require manual review based on object tracking status.

    Returns:
        A tuple containing:
        1. Final Status (str)
        2. Frames for labeling (Dict)
        3. Frames for deleting (Dict)
        4. Frames for ignore start (Dict)
        5. Frames for ignore stop (Dict)
    """
    if annotation_manager.data:
        annotated_frame_ids = get_unique_frame_ids(annotation_manager.data)
    else:
        annotated_frame_ids = None

    print("\nLaunching Tkinter Video Player...")
    view = TrackerReviewGUI(title=title, landmarks=annotated_frame_ids, windowState='maximized')
    controller = TrackerReviewOrchestrator(model=video_manager, view=view, tracking_history=tracked_objs)
    
    # Updated unpacking to handle 5 return values
    final_status, frames_for_labeling, frames_for_deleting, frames_for_ignore_start, frames_for_ignore_stop = controller.run()

    print("\n--- Results from player session ---")
    print(f"Final Status: {final_status}")
    print(f"Frames marked for labeling: {frames_for_labeling}")
    print(f"Frames marked for deletion: {frames_for_deleting}")
    print(f"Frames marked for ignore start: {frames_for_ignore_start}")
    print(f"Frames marked for ignore stop: {frames_for_ignore_stop}")
    
    return final_status, frames_for_labeling, frames_for_deleting, frames_for_ignore_start, frames_for_ignore_stop


def review_tracked_objects_in_video(
    video_path: Path, 
    objects_to_track: list[str],
    metadata_path: Path,
    tracked_data_path: Path,
    *,
    force_processing: bool = False
) -> Optional[Path]:
    """
    Main pipeline to review tracking, validate, or trigger re-annotation.
    """
    if metadata_path.exists():
        annotation_data_iohandler = ROIAnnotationFileHandler.load(metadata_path)
        annotation_manager = ROIAnnotationManager(annotation_data_iohandler)

        if not force_processing and annotation_data_iohandler is not None and annotation_manager.are_no_objects_with_status(ROIProcessingStatus.TO_BE_REVIEWED):
            print(f"No object has been assigned to be reviewed: either ✅ Tracking is completed or automatic algorithm has to process it.")
            return tracked_data_path
    else:
        annotation_manager = ROIAnnotationManager()
        for object_name in objects_to_track:
            annotation_manager.add_object(object_name)

    if tracked_data_path.exists():
        tracked_data_iohandler = ROITrackedFileHandler(tracked_data_path)
        tracked_data: ROITrackedObjects = tracked_data_iohandler.load_all_data()
        video_manager = VideoReviewManager(video_path, tracking_history=tracked_data)
    else:
        # if there is no file, the data will be empty.
        # This means it is the first time we are seeing the video
        tracked_data = dict.fromkeys(objects_to_track)
        video_manager = VideoReviewManager(video_path)
    
    # Call the updated review_tracking function which returns 5 values
    final_status, frames_for_labeling, frames_for_deleting, frames_for_ignore_start, frames_for_ignore_stop = review_tracking(
        video_manager, annotation_manager, tracked_data, title=os.path.basename(video_path)
    )
    
    if final_status == TrackerReviewStatus.UNDEFINED or final_status == TrackerReviewStatus.UNPERFECT:
        print("Review session ended without validation or marking new frames.")
        return None
    
    elif final_status == TrackerReviewStatus.COMPLETED:
        annotation_manager.update_all_status(ROIProcessingStatus.COMPLETED)

    elif final_status == TrackerReviewStatus.PROCEED:
        print(f"⚠️ User marked {len(frames_for_labeling)} frames for re-annotation. Launching interactive tool.")
        # 1. Handle Deletions and Ignore Events, and Handle Interactive Labeling
        remove_frames_from_metadata(annotation_manager, frames_for_deleting)
        update_ignore_frames_in_metadata(annotation_manager, frames_for_ignore_start, frames_for_ignore_stop)
        annotate_rois_interactively(video_manager, annotation_manager, frames_for_labeling)
        print("\nRe-annotation complete. The process may need to be run again to verify the new tracking.")
    
    ROIAnnotationFileHandler.save(metadata_path, annotation_manager.data)

    return tracked_data_path