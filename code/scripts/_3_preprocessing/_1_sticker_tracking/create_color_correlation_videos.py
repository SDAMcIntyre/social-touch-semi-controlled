# --------------------------------------------------------------------------- #
# Standard Library Imports
# --------------------------------------------------------------------------- #
from pathlib import Path
from typing import List, Optional, Union

# --------------------------------------------------------------------------- #
# Third-Party Imports
# --------------------------------------------------------------------------- #
import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Local Application Imports
# --------------------------------------------------------------------------- #
from preprocessing.common import ColorFormat, VideoMP4Manager
from preprocessing.stickers_analysis import (
    ColorCorrelationVisualizer,
    ColorFamilyModel,
    ColorSpace,
    ColorSpaceFileHandler,
    ColorSpaceManager,
    ColorSpaceStatus,
)
from utils.should_process_task import should_process_task


# --------------------------------------------------------------------------- #
# Core Processing Functions
# --------------------------------------------------------------------------- #

def generate_correlation_maps(
    frames_bgr: List[np.ndarray],
    metadata: ColorSpace,
    negative_samples_list: List[np.ndarray],
    *,
    show: bool = False,
    conversion_mode: str = 'xyz'
) -> List[np.ndarray]:
    """
    Generates color correlation maps for a sequence of BGR frames.
    
    Now supports discriminative modeling by passing negative sample lists.
    """
    family_colors = metadata.extract_rgb_triplets(output_format='array')
    
    # Initialize model with both positive (target) and negative (discarded) samples
    model = ColorFamilyModel(
        family_colors, 
        color_space='rgb', 
        conversion_mode=conversion_mode,
        negative_samples_list=negative_samples_list
    )

    correlation_maps = [
        model.calculate_probability_map(frame_bgr, image_color_space='bgr')
        for frame_bgr in frames_bgr
    ]

    if show:
        try:
            import matplotlib.pyplot as plt
            visualizer = ColorCorrelationVisualizer(model)
            for frame_bgr in frames_bgr:
                visualizer.update(frame_bgr)
        finally:
            print("Processing finished. Close the plot window to exit.")
            plt.ioff()
            plt.show()

    return correlation_maps


def invert_correlation_maps(corr_maps: List[np.ndarray]) -> List[np.ndarray]:
    """
    Inverts correlation map values for intuitive grayscale visualization.
    """
    if not corr_maps:
        raise ValueError("Input list 'corr_maps' cannot be empty.")

    global_max = max(np.max(frame) for frame in corr_maps)
    inverted_maps = [global_max - frame for frame in corr_maps]

    return inverted_maps


# --------------------------------------------------------------------------- #
# I/O and Utility Functions
# --------------------------------------------------------------------------- #

def save_correlation_results_to_mp4(
    corr_maps: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30
) -> None:
    """
    Saves a list of correlation maps as a grayscale MP4 video.
    """
    if not corr_maps:
        print("⚠️ Warning: The list of correlation maps is empty. No video created.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = corr_maps[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height), isColor=False
    )

    if not video_writer.isOpened():
        print(f"❌ Error: Could not open video writer for path: {output_path}")
        return

    global_min = np.min([np.min(m) for m in corr_maps])
    global_max = np.max([np.max(m) for m in corr_maps])
    scaled_range = global_max - global_min

    print(f"📹 Saving grayscale video ({len(corr_maps)} frames) to '{output_path}'...")
    
    try:
        for corr_map in corr_maps:
            if scaled_range == 0.0:
                normalized_float = np.zeros_like(corr_map, dtype=float)
            else:
                normalized_float = (corr_map - global_min) / scaled_range

            frame_uint8 = (normalized_float * 255).astype(np.uint8)
            video_writer.write(frame_uint8)
    finally:
        video_writer.release()
        print(f"✅ Video successfully saved to '{output_path}'.")


# --------------------------------------------------------------------------- #
# Main Workflow Orchestrator
# --------------------------------------------------------------------------- #

def create_color_correlation_videos(
    video_standard_format_path: Path,
    md_path: Path,
    output_standard_format_path: Path,
    *,
    force_processing: bool = False,
    monitor: bool = False
) -> Optional[Path]:
    """
    Orchestrates the workflow to generate color correlation videos.
    Automatically identifies 'discarded' color variants and uses them to
    penalize false positives in the correlation map.
    """
    colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(md_path)
    metadata_file_modified = False
    last_output_path = None

    print("--- Starting Color Correlation Video Processing ---")

    for name in colorspace_manager.colorspace_names:
        # Skip "discarded" definitions in the main loop; they are only used as helpers
        # for their parent colors.
        if "discarded" in name:
            continue

        current_colorspace = colorspace_manager.get_colorspace(name)
        
        # --- NEW LOGIC: Identify associated discarded colors ---
        # Look for names like "{name}_discarded_1", "{name}_discarded_2"
        discarded_samples_list = []
        possible_discard_prefix = f"{name}_discarded"
        
        for potential_discard_name in colorspace_manager.colorspace_names:
            if potential_discard_name.startswith(possible_discard_prefix):
                print(f"   + Found discarded/negative class for {name}: {potential_discard_name}")
                discarded_cs = colorspace_manager.get_colorspace(potential_discard_name)
                # Extract RGBs and add to list
                discarded_rgb = discarded_cs.extract_rgb_triplets(output_format='array')
                if len(discarded_rgb) > 0:
                    discarded_samples_list.append(discarded_rgb)
        # -----------------------------------------------------

        input_video_path = video_standard_format_path.with_name(
            f"{video_standard_format_path.stem}_{name}.mp4"
        )
        output_video_path = output_standard_format_path.with_name(
            f"{output_standard_format_path.stem}_{name}.mp4"
        )

        is_to_be_processed = (
            current_colorspace.status == ColorSpaceStatus.TO_BE_PROCESSED.value
        )
        needs_processing = should_process_task(
            output_paths=output_video_path,
            input_paths=input_video_path,
            force=force_processing
        )

        if not (is_to_be_processed or needs_processing):
            print(f"Skipping '{name}' (Status: '{current_colorspace.status}', files up-to-date).")
            continue

        print(f"\nProcessing '{name}'...")
        if discarded_samples_list:
            print(f"   -> Using {len(discarded_samples_list)} negative sample sets to improve accuracy.")

        print(f"-> Loading video: '{input_video_path}'")
        video_manager = VideoMP4Manager(input_video_path)
        video_manager.color_format = ColorFormat.BGR

        # Generate maps with negative sample support
        corr_maps = generate_correlation_maps(
            frames_bgr=video_manager,
            metadata=current_colorspace,
            negative_samples_list=discarded_samples_list,
            show=monitor,
            conversion_mode='circular'
        )

        #corr_maps = invert_correlation_maps(corr_maps)
        save_correlation_results_to_mp4(corr_maps, output_video_path)

        colorspace_manager.update_status(name, ColorSpaceStatus.TO_BE_REVIEWED.value)
        metadata_file_modified = True
        last_output_path = output_video_path
        print(f"✅ Finished processing for '{name}'.")

    if metadata_file_modified:
        ColorSpaceFileHandler.write(md_path, colorspace_manager)
        print("\n--- Metadata file updated. Processing complete. ---")
    else:
        print("\n✅ No objects required processing. Nothing to do.")

    return last_output_path