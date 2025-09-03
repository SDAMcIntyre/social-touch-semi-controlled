import cv2
import numpy as np
import tkinter as tk
import os
from collections import defaultdict


from preprocessing.common import (
    VideoMP4Manager,
    FrameROISquare
)

from preprocessing.forearm_extraction import (
    MultiVideoFramesSelector,
    ForearmParameters,
    RegionOfInterest,
    Point,
    ForearmFrameParametersFileHandler,
    sort_forearm_parameters_by_video_and_frame
)

# Global variable to hold the root Tkinter instance, ensuring it's a singleton.
_tk_root_instance = None

def _get_or_create_tk_root():
    """
    Manages a singleton Tkinter root instance.

    On the first call, it creates a tk.Tk() instance and hides it.
    On subsequent calls, it returns the existing instance.
    This avoids issues with re-initializing Tkinter, which can lead to errors.
    """
    global _tk_root_instance
    if _tk_root_instance is None:
        _tk_root_instance = tk.Tk()
        _tk_root_instance.withdraw()  # Hide the root window
    return _tk_root_instance

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
def define_forearm_extraction_parameters(rgb_video_paths: list[str], metadata_path: str) -> None:
    """
    Interactively selects frames and Regions of Interest (ROIs) from multiple
    videos and saves the combined metadata to a single JSON file.

    This function orchestrates the following steps:
    1.  Checks if a valid metadata file already exists.
    2.  Launches a UI to select one or more frames from each provided video.
    3.  Iterates through each selected frame. For each one, it:
        a. Loads the corresponding video.
        b. Displays the selected frame.
        c. Launches a UI for the user to draw an ROI on that frame.
    4.  Gathers all metadata into a list of ForearmParameters objects.
    5.  Saves the list to a single JSON file.

    Args:
        rgb_video_paths (list[str]): A list of paths to the input RGB video files.
        metadata_path (str): Path where the output JSON metadata file will be saved.
    """
    parameters_list = None
    if ForearmFrameParametersFileHandler.is_valid_structure(metadata_path):
        print(f"⚠️ Metadata file '{metadata_path}' already exists and has a valid structure. Loading content...")
        parameters_list = ForearmFrameParametersFileHandler.load(metadata_path)

    # --- Step 1: Select frames from all videos ---
    print("🖱️ Step 1: Select frames from all videos. Close the window to continue...")
    # Use a singleton root and a Toplevel window for the UI.
    # This prevents errors from creating and destroying multiple tk.Tk() instances.
    root = _get_or_create_tk_root()
    app_window = tk.Toplevel(root)
    app_window.title("Multi-Video Frame Selector")

    if parameters_list:
        video_frames_dict = defaultdict(list)
        for params in parameters_list:
            video_frames_dict[params.video_filename].append(params.frame_id)
        multi_selector = MultiVideoFramesSelector(app_window, rgb_video_paths, video_frames_dict)
    else:
        multi_selector = MultiVideoFramesSelector(app_window, rgb_video_paths)
    root.wait_window(app_window) # Script pauses here until the Toplevel window is closed.

    if not multi_selector.validated or not multi_selector.all_selected_frames:
        print("\n🟡 Selection window was closed without validation or no frames were selected. Aborting.")
        return

    print("\n--- Frames Selected for Processing ---")
    for video_path, frames in multi_selector.all_selected_frames.items():
        print(f"  - {os.path.basename(video_path)}: Frames {frames}")

    # --- Step 2: Define ROI for each selected frame ---
    print("\n🖱️ Step 2: Define a Region of Interest (ROI) for each selected frame.")
    all_forearm_parameters = []

    # Iterate through each video that has selected frames
    for video_path, frame_indices in multi_selector.all_selected_frames.items():
        print(f"\n▶️ Processing video: '{os.path.basename(video_path)}'")
        try:
            video_manager = VideoMP4Manager(video_path)
        except FileNotFoundError as e:
            print(f"❌ Error loading video: {e}. Skipping this video.")
            continue
        
        video_filename = os.path.basename(video_path)
        # Iterate through each frame selected for the current video
        for frame_idx in frame_indices:
            print(f"  - Defining ROI for frame {frame_idx}...")
            selected_frame = video_manager.get_frame(frame_idx)

            if parameters_list:
                # Find the specific parameter object that matches the video filename and frame index
                matching_param = next(
                    (p for p in parameters_list if p.video_filename == video_filename and p.frame_id == frame_idx),
                    None  # Default to None if no match is found
                )

                # Initialize the dictionary as None
                predefined_roi = None

                # If a matching parameter was found, extract and format the ROI
                if matching_param:
                    roi = matching_param.region_of_interest
                    predefined_roi = {
                        'x': roi.top_left_corner.x,
                        'y': roi.top_left_corner.y,
                        'width': roi.bottom_right_corner.x - roi.top_left_corner.x,
                        'height': roi.bottom_right_corner.y - roi.top_left_corner.y
                    }
                roi_ui = FrameROISquare(
                    selected_frame,
                    is_rgb=True,
                    window_title=f"Draw ROI for {video_filename} - Frame {frame_idx}",
                    predefined_roi=predefined_roi
                )
            else:
                roi_ui = FrameROISquare(
                    selected_frame,
                    is_rgb=True,
                    window_title=f"Draw ROI for {video_filename} - Frame {frame_idx}"
                )
            roi_ui.run()
            roi_data = roi_ui.get_roi_data()

            if not roi_data:
                print(f"    🟡 ROI selection cancelled for frame {frame_idx}. Skipping this frame.")
                continue

            x, y, w, h = roi_data["x"], roi_data["y"], roi_data["width"], roi_data["height"]
            print(f"    ✅ ROI selected at (x={x}, y={y}) with size (w={w}, h={h}).")

            # Compile metadata for this specific frame
            top_left = Point(x=x, y=y)
            bottom_right = Point(x=x + w, y=y + h)
            roi = RegionOfInterest(top_left_corner=top_left, bottom_right_corner=bottom_right)

            metadata = ForearmParameters(
                video_filename=os.path.basename(video_path),
                frame_id=frame_idx,
                region_of_interest=roi,
                frame_width=video_manager.frame_width,
                frame_height=video_manager.frame_height,
                fps=video_manager.fps,
                nframes=video_manager.total_frames,
                fourcc_str=video_manager.fourcc_str
            )
            all_forearm_parameters.append(metadata)

    # --- Step 3: Save all collected metadata ---
    if not all_forearm_parameters:
        print("\n🟡 No ROIs were defined. No metadata file will be created.")
        return

    print("\n💾 Step 3: Saving all compiled metadata...")
    all_forearm_parameters_sorted = sort_forearm_parameters_by_video_and_frame(all_forearm_parameters)
    ForearmFrameParametersFileHandler.save(all_forearm_parameters_sorted, metadata_path)

# ----------------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------------

def create_dummy_video(filename: str, width: int, height: int, num_frames: int, color: tuple):
    """Helper function to create a sample video file."""
    if os.path.exists(filename):
        print(f"Found existing video: '{filename}'")
        return

    print(f"'{filename}' not found. Creating a dummy video for testing...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a moving square with the specified color
        x_pos = int((width/3) + (width/4) * np.sin(i * 0.1))
        y_pos = int((height/3) + (height/4) * np.cos(i * 0.15))
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 50, y_pos + 50), color, -1)
        text = f"Frame {i}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Dummy video '{filename}' created.")


if __name__ == "__main__":
    # --- Configuration ---
    VIDEO_FILENAMES = ["sample_video_1.mp4", "sample_video_2.mp4"]
    METADATA_FILENAME = "forearm_extraction_parameters.json"

    # --- Create dummy videos for demonstration ---
    create_dummy_video(VIDEO_FILENAMES[0], 640, 480, 80, color=(255, 0, 0)) # Blue square
    create_dummy_video(VIDEO_FILENAMES[1], 800, 600, 60, color=(0, 255, 0)) # Green square

    # --- Run the main function ---
    print("\nStarting metadata generation process...")
    try:
        define_forearm_extraction_parameters(
            rgb_video_paths=VIDEO_FILENAMES,
            metadata_path=METADATA_FILENAME
        )
    finally:
        # Ensure the Tkinter root is destroyed when the script finishes to clean up.
        if _tk_root_instance:
            _tk_root_instance.destroy()
            print("\nTkinter instance destroyed.")