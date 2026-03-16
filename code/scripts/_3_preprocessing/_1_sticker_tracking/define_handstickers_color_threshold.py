# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict

# Local application imports
from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    ColorSpaceFileHandler, 
    ColorSpaceManager,
    ColorSpace,
    ColorSpaceStatus,
    ThresholdSelectorTool,
    SelectionState
)


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
    rgb_video_base_path: Path,
    corrmap_video_base_path: Path,
    md_path: Path,
    *,
    force_processing: bool = False
):
    """
    Orchestrates the video processing workflow.
    
    Architectural Note:
    rgb_video_base_path and corrmap_video_base_path are treated as template paths.
    The character '*' within these paths will be replaced by the specific object name during iteration.
    """
    # 1. Load all data
    if not os.path.exists(md_path):
        print("🟡 Skipping: Metadata file do not exist yet.")
        return

    colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(md_path)

    # 2. Early exit: If not forcing and no objects need processing, stop.
    if not force_processing and colorspace_manager.are_no_objects_with_status(ColorSpaceStatus.TO_BE_REVIEWED):
        print(f"✅ No objects are marked '{ColorSpaceStatus.TO_BE_REVIEWED.value}'. Nothing to do.")
        return
    
    object_names = colorspace_manager.colorspace_names
    colorspace_modified = False

    # 3. Process each object sequentially
    for name in object_names:
        current_colorspace: ColorSpace = colorspace_manager.get_colorspace(name)

        # Decide whether to process this specific object
        should_process = force_processing or (current_colorspace.status == ColorSpaceStatus.TO_BE_REVIEWED.value)

        if not should_process:
            print(f"Skipping '{name}' (status: '{current_colorspace.status}').")
            continue
        
        print(f"\nProcessing '{name}'...")
        threshold = current_colorspace.threshold if current_colorspace.threshold is not None else 127

        # Construct paths for both Correlation and RGB videos using wildcard replacement
        # The base paths are converted to strings to perform the substitution, then cast back to Path objects.
        corr_video_path = Path(str(corrmap_video_base_path).replace("*", name))
        rgb_input_path = Path(str(rgb_video_base_path).replace("*", name))

        print(f"Loading Correlation video '{corr_video_path}'...")
        frames_corr = VideoMP4Manager(corr_video_path)

        print(f"Loading RGB video '{rgb_input_path}'...")
        frames_rgb = VideoMP4Manager(rgb_input_path)
        
        # Call the interactive GUI function
        try:
            tool = ThresholdSelectorTool(
                frames_rgb=frames_rgb, 
                frames_corr=frames_corr,
                video_name=corr_video_path.name,
                threshold=threshold,
                spot_type='bright'  # 'dark', 'bright'
            )
            tool.run()
        except Exception as e:
            print(f"An error occurred while running the threshold tool: {e}")
            # This can happen if tkinter is not available, for example.
            return None
    
        # IMPORTANT: Only update if the user confirmed a value
        if tool.selection_state == SelectionState.CONFIRMED:
            print(f"✅ Finished processing for '{name}'.")
            colorspace_manager.update_threshold(name, tool.result)
            colorspace_manager.update_status(name, ColorSpaceStatus.REVIEW_COMPLETED.value)
            colorspace_modified = True
        elif tool.selection_state == SelectionState.REDO_COLORSPACE:
            colorspace_manager.update_status(name, ColorSpaceStatus.TO_BE_DEFINED.value)
            colorspace_modified = True
        else:
            print(f"⏩ Skipping metadata update for '{name}' as selection was cancelled.")
    
    if colorspace_modified:
        ColorSpaceFileHandler.write(md_path, colorspace_manager)

    print(f"\n--- Processing Complete ---")