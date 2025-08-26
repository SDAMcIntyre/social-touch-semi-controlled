# Standard library imports
import json
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application imports
from .utils.roi_to_position_funcs import load_video_frames_bgr
from .utils.ThresholdSelectorTool import ThresholdSelectorTool


def get_obj_name(metadata: dict) -> list[str]:
    keys_list = list(metadata.keys())
    return keys_list


# --- Core Component 1: Loading ---
def _load_metadata(md_path: str) -> Dict[str, Any]:
    """
    Responsibility: Loads and validates metadata and tracking data from files.
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Metadata file not found at '{md_path}'.")

    try:
        with open(md_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Could not read or parse metadata file '{md_path}'.") from e

    print(f"✅ Successfully loaded metadata.")
    return metadata




def update_object_threshold(metadata: Dict[str, Any], object_name: str, threshold: int, md_path: str):
    """
    Adds or updates the threshold for an object and saves the metadata to a file.

    Args:
        metadata (Dict[str, Any]): The full metadata dictionary.
        object_name (str): The key for the object to be updated.
        threshold (int): The integer threshold value to save.
        md_path (str): The path to the .json metadata file to be overwritten.
    """
    if object_name not in metadata:
        print(f"⚠️ Object '{object_name}' not in metadata. Creating new entry.")
        metadata[object_name] = {}

    # Update the threshold value for the specified object
    metadata[object_name]['threshold'] = threshold

    try:
        with open(md_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata file '{md_path}' successfully updated for '{object_name}'.")
    except IOError as e:
        print(f"❌ Critical Error: Could not save metadata to '{md_path}'. Reason: {e}")



def define_handstickers_color_threshold(
    video_path: str,
    md_path: str
):
    """
    Orchestrates the video processing workflow.
    """
    print(f"--- Starting Video Processing ---")
    
    # 1. Load all data
    metadata = _load_metadata(md_path)
    
    # 2. Identify objects to process
    object_names = get_obj_name(metadata)

    # 3. Process each object sequentially
    for name in object_names:
        print(f"\nProcessing '{name}'...")
        video_name = (video_path.stem + f"_{name}.mp4")
        input_video_path = video_path.parent / video_name
        
        print(f"Loading video '{input_video_path}'...")
        frames_bgr = load_video_frames_bgr(str(input_video_path)) # Ensure path is a string
        
        if 'threshold' in metadata[name]:
            threshold = metadata[name]['threshold'] 
        else:
            threshold = 127
        
        # Call the interactive GUI function
        try:
            tool = ThresholdSelectorTool(frames=frames_bgr, 
                                         video_name=video_name,
                                         threshold=threshold,
                                         spot_type='bright')  # 'dark', 'bright'
            threshold = tool.select_threshold()
        except Exception as e:
            print(f"An error occurred while running the threshold tool: {e}")
            # This can happen if tkinter is not available, for example.
            return None
    
        # IMPORTANT: Only update if the user confirmed a value
        if threshold is not None:
            print(f"✅ Finished processing for '{name}'.")
            update_object_threshold(metadata, name, threshold, md_path)
        else:
            print(f"⏩ Skipping metadata update for '{name}' as selection was cancelled.")
            
    # 4. Final confirmation
    print(f"\n--- Processing Complete ---")