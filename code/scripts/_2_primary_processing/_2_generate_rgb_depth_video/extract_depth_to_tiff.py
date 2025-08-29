import os
import datetime
import numpy as np
import tifffile as tiff

from utils.kinect_mkv_manager import KinectMKV


def extract_depth_to_tiff(
        mkv_path: str, 
        output_dir: str, 
        verbose: bool = True
):
    """
    Extracts point cloud data from an Azure Kinect MKV file and saves each
    frame as a multi-channel TIFF image.

    This function operates sequentially and creates a .SUCCESS marker file in the
    output directory upon successful completion to avoid reprocessing.

    Args:
        mkv_path (str): The full path to the input MKV file.
        output_dir (str): The directory where TIFF files will be saved.
        verbose (bool): If True, prints progress to the console.
    """
    # 1. --- Pre-flight checks and path setup ---
    if not os.path.exists(mkv_path):
        raise FileNotFoundError(f"Input file not found: {mkv_path}")

    base_filename = os.path.splitext(os.path.basename(mkv_path))[0]
    marker_filepath = os.path.join(output_dir, f".SUCCESS_{base_filename}")

    if os.path.exists(marker_filepath):
        if verbose:
            print(f"Skipping '{base_filename}': success marker found.")
        return

    os.makedirs(output_dir, exist_ok=True)


    # 3. --- Process frames and handle cleanup robustly ---
    was_successful = True
    try:
        with KinectMKV(mkv_path) as mkv:
            print(f"Successfully opened video with ~{len(mkv)} frames.")

            if verbose:
                duration_sec = mkv.length / 1_000_000
                print(f"Processing '{base_filename}' ({duration_sec:.2f}s)...")

            nframes_saved = 0
            # Iterate through each frame using a for loop
            for frame_index, frame in enumerate(mkv):
                print(f"--- Processing Frame {frame_index} ---")

                # 2. Safely check for the depth map
                if frame.transformed_depth_point_cloud is None:
                    print("  ❌ No depth map in this frame.")
                    continue
                print(f"  ✅ Depth map found with shape: {frame.transformed_depth_point_cloud.shape}")
                
                point_cloud = frame.transformed_depth_point_cloud
                # Transpose from (H, W, C) to (W, H, C) and ensure memory is contiguous
                # to prevent potential stride issues with some TIFF readers.
                point_cloud_for_tiff = np.ascontiguousarray(point_cloud.transpose(1, 0, 2))
                
                # Save the frame as a TIFF file
                output_filename = f"{base_filename}_point_cloud-{frame_index:04d}.tiff"
                output_filepath = os.path.join(output_dir, output_filename)
                tiff.imwrite(output_filepath, point_cloud_for_tiff)
                
                nframes_saved += 1
                if verbose:
                    print(f"Frames saved: {nframes_saved}", end='\r')

    except EOFError:
        # This is the expected way to end the loop
        if verbose:
            print(f"\nEnd of file reached. A total of {nframes_saved} frames were saved.")
        was_successful = False
        
    finally:
        # 4. --- Create marker file on success and close resources ---
        if was_successful:
            with open(marker_filepath, 'w') as f:
                f.write(f"Completed on: {datetime.datetime.now()}\n")
            if verbose:
                print(f"Successfully created marker file: '{marker_filepath}'")
        else:
            print("\nProcessing failed or was interrupted. No success marker will be created.")
    