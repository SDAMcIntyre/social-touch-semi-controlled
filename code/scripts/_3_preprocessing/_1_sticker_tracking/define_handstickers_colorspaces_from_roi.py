# Standard library imports
import json
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np

from preprocessing.common import (
    VideoMP4Manager
)

from preprocessing.stickers_analysis import (
    ROIAnnotationFileHandler,
    ROIAnnotationManager,

    TrackerReviewGUI,
    TrackerReviewOrchestrator,

    FrameROIColor,

    ColorSpaceFileHandler,
    ColorSpaceManager,
    ColorSpaceStatus,
    ColorSpace
)


def generate_color_config(objs_to_track: List[str]) -> Dict[str, Any]:
    """Generates a configuration dictionary for object drawing colors.

    Args:
        objs_to_track: A list of strings, where each string is an object name.

    Returns:
        A config dictionary where each key is an object name and its value is
        a dictionary containing 'live' and 'final' BGR color tuples.
    """
    color_map = {
        'yellow': (0, 255, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'white': (255, 255, 255)
    }
    color_config = {}
    for obj_name in objs_to_track:
        try:
            # Handle potential suffix variations for discarded objects if re-processing
            base_color_name = obj_name.split('_')[-1]
            
            # Fallback if the name is complex (e.g. sticker_yellow_discarded_1)
            found_color = None
            for key in color_map:
                if key in obj_name:
                    found_color = key
                    break
            
            if found_color:
                live_color = color_map[found_color]
                final_color = tuple(c // 2 for c in live_color)
                color_config[obj_name] = {'live': live_color, 'final': final_color}
            else:
                print(f"⚠️ Warning: Color could not be inferred for '{obj_name}'. Skipping.")
        except IndexError:
            print(f"⚠️ Warning: Could not extract color from '{obj_name}'. Skipping.")
            continue
    return color_config


def run_colorspace_definition_tool(
    video_frames: List[np.ndarray], 
    colors_rgb: Dict[str, Tuple[int, int, int]],
    title: str = None
) -> List[Dict[str, Any]]:
    """Runs the ROI drawing tool for each frame to define a colorspace."""
    defined_colorspaces = []
    # Default fallback colors if config is missing
    live = colors_rgb.get('live', (0, 255, 0)) if colors_rgb else (0, 255, 0)
    final = colors_rgb.get('final', (0, 128, 0)) if colors_rgb else (0, 128, 0)

    for i, frame_bgr in enumerate(video_frames):
        print(f"\n🎨 Defining colorspace for frame {i+1}/{len(video_frames)}...")
        tracker = FrameROIColor(
            frame_bgr,
            resize_to=(1024, 768),
            is_bgr=True,
            color_live=live,
            color_final=final,
            window_title=title
        )
        tracker.run()
        tracking_data = tracker.get_tracking_data()

        if tracking_data:
            print("--- ✅ Tracking Data Extracted ---")
            defined_colorspaces.append(tracking_data)

    return defined_colorspaces


class ExtractionPipeline:
    """Helper class to encapsulate the video loading and extraction logic."""
    
    def __init__(self, video_path: Path, color_config: Dict[str, Any]):
        self.video_path = video_path
        self.color_config = color_config

    def execute(self, object_name: str, landmarks: List[int] = None) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Runs the review GUI and FrameROIColor tool for a specific object."""
        
        # Construct filename based on standard convention
        # Note: We rely on the base video path logic from the original script
        # Assuming object_name passed here is the base name (e.g., 'sticker_yellow')
        # If processing a discard, we still load the video for the base object.
        
        # Clean object name to find the video file (remove _discarded suffix if present)
        base_object_name = object_name.split("_discarded")[0]
        
        object_video_filename = f"{self.video_path.stem}_{base_object_name}{self.video_path.suffix}"
        object_video_path = self.video_path.parent / object_video_filename

        if not object_video_path.exists():
            print(f"❌ Video file not found: {object_video_path}")
            return [], []

        print(f"Loading video '{object_video_path.name}'...")
        video_manager = VideoMP4Manager(object_video_path)
        
        # 1. Select Frames
        print("   - Waiting for user to select representative frames...")
        try:
            view = TrackerReviewGUI(
                title=f"Select Frames: {object_name}", 
                landmarks={'label': landmarks} if landmarks else [],
                show_valid_button=False,
                show_rerun_button=False,
                windowState='maximized')
        except:
            view = TrackerReviewGUI(
                title=f"Select Frames: {object_name}",
                show_valid_button=False,
                show_rerun_button=False,
                windowState='maximized')
        
        controller = TrackerReviewOrchestrator(model=video_manager, view=view)
        returned_values = controller.run()
        selected_frame_indices = returned_values[1].keys()
        
        if not selected_frame_indices:
            print(f"   - No frames selected for '{object_name}'.")
            return [], []
            
        selected_frames = [video_manager[i] for i in selected_frame_indices]

        # 2. Define Colorspace
        colorspace_data = run_colorspace_definition_tool(
            selected_frames, 
            colors_rgb=self.color_config.get(base_object_name), # Use base config
            title=f"Define: {object_name}"
        )
        
        return selected_frame_indices, colorspace_data


class ColorDefinitionDialog:
    """GUI Controller for managing Target vs Discard colorspace definition."""
    
    def __init__(self, root, object_name: str, pipeline: ExtractionPipeline, manager: ColorSpaceManager):
        self.root = root
        self.object_name = object_name
        self.pipeline = pipeline
        self.manager = manager
        
        # Determine if this object already exists in the manager
        self.existing_cs = self.manager.get_colorspace(object_name)
        
        # Adjust window height slightly if we need to fit the extra button
        height = 360 if self.existing_cs else 300
        self.root.title(f"Processing: {object_name}")
        self.root.geometry(f"400x{height}")
        
        # State tracking
        self.target_defined = False
        self.discard_count = 0
        self.pending_updates = [] # List of tuples (name, frames, data, status, merge)

        # UI Elements
        self.lbl_info = tk.Label(root, text=f"Object: {object_name}", font=("Arial", 12, "bold"))
        self.lbl_info.pack(pady=10)

        self.btn_target = tk.Button(
            root, 
            text="Define Current Color", 
            command=self.on_define_target,
            height=2, width=30,
            bg="#f0f0f0"
        )
        self.btn_target.pack(pady=10)

        self.btn_discard = tk.Button(
            root, 
            text="Add Discard Color", 
            command=self.on_add_discard,
            height=2, width=30
        )
        self.btn_discard.pack(pady=10)

        self.btn_proceed = tk.Button(
            root, 
            text="Proceed / Save", 
            command=self.on_proceed,
            height=2, width=30,
            bg="#d0e0ff"
        )
        self.btn_proceed.pack(pady=20)
        
        # Conditional Skip Button
        if self.existing_cs:
            self.btn_skip = tk.Button(
                root, 
                text="Skip (Keep Existing)", 
                command=self.on_skip,
                height=2, width=30,
                bg="#fff0f0" # Light tint to differentiate
            )
            self.btn_skip.pack(pady=5)
            
            # Check if existing status implies it's ready, to potentially color the target button
            if self.existing_cs.status != ColorSpaceStatus.TO_BE_REVIEWED.value:
                 pass

    def on_define_target(self):
        """Handler for defining the main object color."""
        if self.target_defined:
            return

        print(f"\n🔵 Starting extraction for TARGET: {self.object_name}")
        self.root.withdraw() # Hide GUI during processing
        
        # Fetch existing landmarks if any
        landmarks = self.existing_cs.get_frame_ids() if self.existing_cs else []

        frame_ids, data = self.pipeline.execute(self.object_name, landmarks)
        
        self.root.deiconify() # Restore GUI

        if frame_ids and data:
            self.target_defined = True
            self.btn_target.config(bg="green", fg="white", state="disabled", text="Target Defined (Saved on Proceed)")
            
            # Queue the update
            self.pending_updates.append({
                "object_name": self.object_name,
                "frame_ids": frame_ids,
                "data": data,
                "status": ColorSpaceStatus.TO_BE_PROCESSED,
                "merge": False # Replace target if redefined
            })
        else:
            print("❌ Target extraction cancelled or empty.")

    def on_add_discard(self):
        """Handler for adding a discarded color."""
        self.discard_count += 1
        discard_name = f"{self.object_name}_discarded_{self.discard_count}"
        
        print(f"\n🟠 Starting extraction for DISCARD: {discard_name}")
        self.root.withdraw()
        
        frame_ids, data = self.pipeline.execute(discard_name)
        
        self.root.deiconify()

        if frame_ids and data:
            print(f"✅ Discard '{discard_name}' prepared.")
            # Queue the update
            self.pending_updates.append({
                "object_name": discard_name,
                "frame_ids": frame_ids,
                "data": data,
                "status": ColorSpaceStatus.TO_BE_PROCESSED, # Or a specific DISCARD status if it existed
                "merge": False # New object
            })
        else:
            self.discard_count -= 1 # Revert counter if cancelled

    def on_proceed(self):
        """Saves data and closes the dialog."""
        if not self.target_defined and not self.existing_cs:
            # Only warn if we don't have an existing definition falling back on
            confirm = messagebox.askyesno(
                "Confirm", 
                "You have not defined the Target Color. Are you sure you want to proceed?"
            )
            if not confirm:
                return

        # Commit changes to Manager
        for update in self.pending_updates:
            try:
                # Check if it's an update or new add
                if self.manager.get_colorspace(update["object_name"]):
                     self.manager.update_object(
                        update["object_name"],
                        update["frame_ids"],
                        update["data"],
                        status=update["status"],
                        merge=update["merge"]
                    )
                else:
                    self.manager.add_object(
                        update["object_name"],
                        update["frame_ids"],
                        update["data"],
                        status=update["status"]
                    )
            except Exception as e:
                print(f"❌ Error saving '{update['object_name']}': {e}")
                
        self.root.quit()
        self.root.destroy()

    def on_skip(self):
        """Skips processing for the current object, preserving existing data."""
        print(f"⏩ Skipping updates for '{self.object_name}' (Existing data preserved).")
        self.pending_updates = [] # Ensure no changes are committed
        self.root.quit()
        self.root.destroy()


def define_handstickers_colorspaces_from_roi(
    video_path: Path,
    roi_metadata_path: Path,
    dest_metadata_path: Path,
    *,
    force_processing: bool = False,
    merge: bool = False
) -> None:
    """Orchestrates the colorspace definition with a GUI selection step."""
    
    print(f"🎬 Starting colorspace definition for: '{Path(video_path).name}'")

    # --- 1. Pre-computation Checks & Validation ---
    if not os.path.exists(roi_metadata_path):
        print("🟡 Skipping: Source metadata not found.")
        return

    metadata_exists = os.path.exists(dest_metadata_path)
    if metadata_exists:
        colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(dest_metadata_path)
    else:
        colorspace_manager = ColorSpaceManager()

    # --- 2. Load Source Data ---
    annotation_data = ROIAnnotationFileHandler.load(roi_metadata_path)
    object_names = ROIAnnotationManager(annotation_data).get_object_names()

    if not object_names:
        print("❌ Halting: 'object_names' is empty in source metadata.")
        return

    color_config = generate_color_config(object_names)
    
    # Initialize Pipeline
    pipeline = ExtractionPipeline(video_path, color_config)

    # --- 3. Processing Loop ---
    # We create a root Tk instance once, but we will show/hide or create dialogs for each object.
    # To avoid mainloop conflicts, we instantiate a root for each object dialog or reuse one.
    # Reusing one root and destroying children is cleaner.
    
    # Check if we have a display (headless check)
    try:
        root = tk.Tk()
        root.withdraw() # Hide the main root window
    except Exception as e:
        print(f"❌ GUI Error: Could not initialize Tkinter ({e}). Is a display available?")
        return

    colorspace_modified = False

    for object_name in object_names:
        current_colorspace: Optional[ColorSpace] = colorspace_manager.get_colorspace(object_name)

        # Logic to skip if already done (unless forced)
        # Note: existing logic checked .status. value.
        status = current_colorspace.status if current_colorspace else None
        should_process = (
            not metadata_exists 
            or force_processing 
            or (status == ColorSpaceStatus.TO_BE_DEFINED.value)
            or (status is None)
        )

        if not should_process:
            print(f"Skipping '{object_name}' (status: '{status}').")
            continue

        print(f"\n➡️ Processing object: '{object_name}'")
        
        # Launch the Selection Dialog
        # We create a Toplevel linked to root, but run a local mainloop or wait_window
        dialog_window = tk.Toplevel(root)
        app = ColorDefinitionDialog(dialog_window, object_name, pipeline, colorspace_manager)
        
        # Wait for this specific window to close before moving to the next object
        root.wait_window(dialog_window)
        
        # If pending updates occurred in the dialog, we flag modification
        if app.pending_updates:
            colorspace_modified = True

    # --- 4. Save Changes ---
    if colorspace_modified:
        print("\n💾 Saving all changes to disk...")
        ColorSpaceFileHandler.write(dest_metadata_path, colorspace_manager)
    else:
        print("\n💤 No changes were made.")

    print(f"\n🎉 Finished processing for: '{Path(video_path).name}'")
    
    # Clean up Tkinter
    try:
        root.destroy()
    except:
        pass