import cv2
import numpy as np
import tkinter as tk

from preprocessing.common import (
    VideoMP4Manager,
    VideoFrameSelector,
    FrameROISquare
)

from preprocessing.forearm_extraction import (
    ForearmParameters,
    RegionOfInterest,
    Point,
    ForearmFrameParametersFileHandler
)

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------

def define_forearm_extraction_parameters(rgb_video_path: str, metadata_path: str) -> None:
    """
    Interactively selects a reference frame and a Region of Interest (ROI)
    from a video and saves the metadata to a JSON file.

    This function orchestrates the following steps:
    1. Loads the video using VideoMP4Manager.
    2. Opens a UI with VideoFrameSelector for the user to choose a reference frame.
    3. Opens a second UI with FrameROISquare for the user to draw an ROI on that frame.
    4. Gathers all metadata and saves it to a JSON file in the specified format.

    Args:
        rgb_video_path (str): Path to the input RGB video file.
        metadata_path (str): Path where the output JSON metadata file will be saved.
    """
    if ForearmFrameParametersFileHandler.is_valid_structure(metadata_path):
        print(f"⚠️ Metadata file '{metadata_path}' already exists. Skipping metadata generation.")
        return
    
    print(f"▶️ Step 1: Loading video from '{rgb_video_path}'...")
    try:
        video_manager = VideoMP4Manager(rgb_video_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    print(f"✅ Video loaded successfully.")

    print("\n🖼️ Step 2: Select a reference frame...")
    root = tk.Tk()
    selector = VideoFrameSelector(root, video_manager.capture) # 'root' is not needed for this implementation
    root.mainloop()
    
    ref_frame_idx = selector.selected_frame_num
    if ref_frame_idx is None:
        print("🟡 Operation cancelled by user during frame selection. Aborting.")
        return
    
    print(f"✅ Frame {ref_frame_idx} selected as the reference.")
    selected_frame = video_manager.get_frame(ref_frame_idx)

    print("\n🖱️ Step 3: Select the Region of Interest (ROI)...")
    tracker_ui = FrameROISquare(
        selected_frame, 
        is_rgb=True, # The frames from VideoMP4Manager are RGB
        window_title="Draw a box for the ROI and press ENTER or SPACE"
    )
    tracker_ui.run()
    tracking_data = tracker_ui.get_roi_data()

    if not tracking_data:
        print("🟡 Operation cancelled by user during ROI selection. Aborting.")
        return

    x, y, w, h = tracking_data["x"], tracking_data["y"], tracking_data["width"], tracking_data["height"]
    print(f"✅ ROI selected at (x={x}, y={y}) with size (w={w}, h={h}).")

    print("\n💾 Step 4: Compiling metadata...")
    # Create dataclass instances from the collected data
    top_left = Point(x=x, y=y)
    bottom_right = Point(x=x + w, y=y + h)
    roi = RegionOfInterest(top_left_corner=top_left, bottom_right_corner=bottom_right)

    metadata = ForearmParameters(
        reference_frame_idx=ref_frame_idx,
        region_of_interest=roi,
        frame_width=video_manager.frame_width,
        frame_height=video_manager.frame_height,
        fps=video_manager.fps,
        nframes=video_manager.total_frames,
        fourcc_str=video_manager.fourcc_str
    )

    # Use the file handler to save the metadata
    ForearmFrameParametersFileHandler.save(metadata, metadata_path)

# ----------------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Create a dummy video for demonstration if it doesn't exist ---
    VIDEO_FILENAME = "sample_video.mp4"
    METADATA_FILENAME = "video_metadata.json"
    
    try:
        # Check if the video file exists by trying to open it
        with open(VIDEO_FILENAME, "r") as f:
            pass
        print(f"Found existing video: '{VIDEO_FILENAME}'")
    except FileNotFoundError:
        print(f"'{VIDEO_FILENAME}' not found. Creating a dummy video for testing...")
        width, height = 640, 480
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 files
        out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 20.0, (width, height))
        
        for i in range(60): # 3 seconds of video at 20 fps
            # Create a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add a moving white square
            x_pos = int(100 + 200 * np.sin(i * 0.1))
            y_pos = int(150 + 100 * np.cos(i * 0.15))
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 50, y_pos + 50), (255, 255, 255), -1)
            # Add frame number text
            text = f"Frame {i+1}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        print("Dummy video created.")

    # --- Run the main function ---
    print("\nStarting metadata generation process...")
    define_forearm_extraction_parameters(
        rgb_video_path=VIDEO_FILENAME, 
        metadata_path=METADATA_FILENAME
    )