# main_script.py

from pathlib import Path
from typing import Iterable

# Assuming all files are in a 'preprocessing/common' structure
from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    ConsolidatedTracksFileHandler,
    ConsolidatedTracksManager,
    ConsolidatedTracksReviewGUI
)


def define_custom_colors(string_list: Iterable[str]) -> dict[str, str]:
    """
    Assigns a standard color keyword based on substrings in a list of names.

    Args:
        string_list: An iterable (e.g., list) of object names.

    Returns:
        A dictionary mapping each object name to a found color string.
    """
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet"
    }
    
    found_colors = {}
    
    for item in string_list:
        item_lower = item.lower()
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
                break # Assign the first color found and move to the next item
    
    return found_colors


def view_summary_stickers_on_rgb_data(
    xy_csv_path: Path,
    rgb_video_path: Path
):
    """
    Loads tracking data and a video, then launches a GUI to visualize the results.
    """
    print("Loading tracking data...")
    try:
        # Load the tracking data from the CSV file
        tracked_data: ConsolidatedTracksManager = ConsolidatedTracksFileHandler.load(xy_csv_path)
        sticker_names = tracked_data.object_names
        print(f"Successfully loaded data for objects: {sticker_names}")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return
    
    print("Loading video...")
    try:
        # Initialize the video manager
        video_manager = VideoMP4Manager(rgb_video_path)
        print(f"Successfully loaded video: {rgb_video_path.name}")
    except Exception as e:
        print(f"An error occurred during video loading: {e}")
        return

    # Automatically determine colors for each sticker based on its name
    sticker_colors = define_custom_colors(sticker_names)
    print(f"Assigned colors: {sticker_colors}")

    # Initialize and start the GUI
    print("Launching review GUI...")
    gui = ConsolidatedTracksReviewGUI(
        video_manager=video_manager,
        tracks_manager=tracked_data,
        object_colors=sticker_colors,
        title=f"Reviewing: {rgb_video_path.name}",
        windowState='maximized'
    )
    gui.start()


# --- Example Usage ---
if __name__ == '__main__':
    # Replace with the actual paths to your files
    path_to_csv = Path("F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_summary_2d_coordinates.csv")
    path_to_video = Path("F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-07/2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4")

    if not path_to_csv.exists() or not path_to_video.exists():
        print("="*60)
        print("!! PLEASE UPDATE THE FILE PATHS IN THE `if __name__ == '__main__':` BLOCK !!")
        print(f"CSV Path Check: {'EXISTS' if path_to_csv.exists() else 'NOT FOUND -> ' + str(path_to_csv)}")
        print(f"Video Path Check: {'EXISTS' if path_to_video.exists() else 'NOT FOUND -> ' + str(path_to_video)}")
        print("="*60)
    else:
        view_summary_stickers_on_rgb_data(
            xy_csv_path=path_to_csv,
            rgb_video_path=path_to_video
        )