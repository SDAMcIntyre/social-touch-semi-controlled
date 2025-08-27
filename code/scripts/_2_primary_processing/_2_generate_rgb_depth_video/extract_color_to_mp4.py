import subprocess
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyk4a import PyK4APlayback, K4AException, PyK4ACapture
from pyk4a.config import ColorResolution
import datetime
import threading
import contextlib

# This maps the stable enum to the hardware's known dimensions (width, height).
CUSTOM_RESOLUTION_MAP = {
    ColorResolution.RES_720P: (1280, 720),
    ColorResolution.RES_1080P: (1920, 1080),
    ColorResolution.RES_1440P: (2560, 1440),
    ColorResolution.RES_1536P: (2048, 1536),
    ColorResolution.RES_2160P: (3840, 2160),
    ColorResolution.RES_3072P: (4096, 3072),
}

def processing_func_color_to_mp4(
    playback: PyK4APlayback,
    output_filepath: str,
    frame_rate: int,
    shared_state: dict = None,
    verbose: bool = False
):
    """
    This function processes an MKV file, extracting the color video stream and
    saving it as an MP4 file.
    """
    video_writer = None
    frame_count = 0

    try:
        color_resolution_enum = playback.configuration['color_resolution']

        if color_resolution_enum not in CUSTOM_RESOLUTION_MAP:
            raise RuntimeError(f"Unsupported color resolution: {color_resolution_enum}")
        width, height = CUSTOM_RESOLUTION_MAP[color_resolution_enum]

        if frame_rate <= 0:
            # BUG FIX: This also needs to use key access for the dictionary.
            camera_fps_enum = playback.configuration['camera_fps']
            frame_rate = camera_fps_enum.get_fps()
            print(f"Using recording FPS from metadata: {frame_rate} FPS")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filepath, fourcc, frame_rate, (width, height))

        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for path: {output_filepath}")

        while True:
            try:
                capture = playback.get_next_capture()
            except EOFError:
                print(f"\nProcessing finished. Reached end of file. A total of {frame_count} frames were saved.")
                break

            if shared_state:
                with shared_state["lock"]:
                    shared_state["latest_capture"] = capture

            frame_bgr = None
            if capture.color is not None:
                color_image = capture.color
                if color_image.ndim == 1:
                    # Compressed MJPEG format
                    frame_bgr = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
                else:
                    # Uncompressed BGRA format
                    frame_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            
            # If a color frame is missing or decoding failed, insert a black frame
            if frame_bgr is None:
                frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            
            video_writer.write(frame_bgr)
            frame_count += 1

            if verbose:
                print(f"Processed frames: {frame_count}", end='\r')

    except Exception as e:
        print(f"\nAn error occurred in the processing function: {e}")
        if shared_state:
            shared_state["is_running"] = False
        raise
    finally:
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print("\nVideo writer released.")
        if shared_state:
            shared_state["is_running"] = False


def plot_capture(capture: PyK4ACapture, fig: plt.Figure):
    """
    Plots available images from a capture onto a given figure object.
    This function is unchanged and handles compressed MJPEG color frames correctly.
    """
    image_specs = {
        "color": "Color", "depth": "Depth", "ir": "Infrared",
        "transformed_depth": "Transformed Depth",
    }
    available_images = []
    if capture:
        for attr, title in image_specs.items():
            image = getattr(capture, attr, None)
            if image is not None:
                available_images.append({"title": title, "image": image})

    if not available_images:
        return

    fig.clf()
    num_images = len(available_images)
    cols = 2
    rows = int(np.ceil(num_images / cols))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()

    for i, spec in enumerate(available_images):
        ax = axes[i]
        title = spec["title"]
        image_data = spec["image"].copy()
        ax.set_title(title)
        ax.axis('off')

        if "Color" in title:
            if image_data.ndim == 1:  # Compressed MJPEG
                decoded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if decoded_image is not None:
                    ax.imshow(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
            else:  # Uncompressed BGRA
                ax.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB))
        else:  # Depth and IR
            im = ax.imshow(image_data, cmap="jet")
            fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Kinect Capture Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def open_kinect_file_safely(long_path, drive_letter = "Z:", force_subst=False):
    """
    Attempts to open a Kinect MKV file, applying a 'subst' workaround if the path is too long.
    """
    if len(long_path) > 240 or force_subst: # A safe threshold below 260
        print("Long path detected. Applying 'subst' workaround...")
        parent_dir = os.path.dirname(long_path)
        file_name = os.path.basename(long_path)
        
        # Unmap the drive first in case it's left over from a previous run
        subprocess.run(f"subst {drive_letter} /D", shell=True, capture_output=True)
        # Map the parent directory to the virtual drive
        subprocess.run(f'subst {drive_letter} "{parent_dir}"', shell=True, check=True)
        
        aliased_path = os.path.join(f"{drive_letter}\\", file_name)
        return PyK4APlayback(aliased_path)
    else:
        return PyK4APlayback(long_path)



def extract_color_to_mp4(
        mkv_path: str, 
        output_filepath: str,
        *,
        show: bool = False, 
        frame_rate: int = 30
):
    """
    Extracts the color stream from an MKV file and saves it as an MP4 video.
    It calls the processing function either directly (for speed) or in a
    separate thread (to allow for visualization).
    """
    if not os.path.exists(mkv_path):
        raise FileNotFoundError(f"Input file not found at: {mkv_path}")

    marker_filepath = f"{output_filepath}.SUCCESS"

    if os.path.exists(marker_filepath):
        print(f"Skipping '{mkv_path}' as output file already validated.")
        return output_filepath

    try:
        playback = open_kinect_file_safely(str(mkv_path))
        playback.open()
    except K4AException as e:   
        raise K4AException(f"Failed to open MKV file. Error: {e}")

    duration_sec = playback.length / 1_000_000
    print(f"Successfully opened MKV. Recording duration: {duration_sec:.2f} seconds.")

    processing_thread = None
    was_successful = False
    fig = None  # Initialize fig to None

    try:
        if show:
            # --- Threaded execution for visualization ---
            print("Running in visualization mode (multi-threaded).")
            shared_state = {
                "latest_capture": None,
                "is_running": True,
                "lock": threading.Lock(),
            }
            
            # BUG FIX: The arguments passed to the thread must match the target 
            # function's signature. We pass the fully constructed 'output_filepath'.
            processing_thread = threading.Thread(
                target=processing_func_color_to_mp4,
                args=(playback, output_filepath, frame_rate, shared_state),
                daemon=True
            )
            processing_thread.start()

            fig = plt.figure(figsize=(15, 8))
            plt.ion()
            # If frame_rate is 0, we'll use a default pause for visualization
            target_pause = 1.0 / frame_rate if frame_rate > 0 else 0.03

            while shared_state["is_running"]:
                with shared_state["lock"]:
                    capture_to_plot = shared_state["latest_capture"]

                if capture_to_plot:
                    plot_capture(capture_to_plot, fig)
                
                if not plt.fignum_exists(fig.number):
                    print("Display window closed by user. Shutting down.")
                    shared_state["is_running"] = False
                    break
                
                plt.pause(target_pause)
            
            # Check if the processing thread encountered an error
            processing_thread.join()
            if not shared_state.get("is_running", True) and not capture_to_plot:
                 # This might indicate the thread exited early due to an error
                 print("Warning: Processing thread may have exited prematurely.")

        else:
            # --- Sequential execution for processing only ---
            print("Running in processing-only mode (single-threaded).")
            # In sequential mode, we pass verbose=True for progress
            processing_func_color_to_mp4(playback, output_filepath, frame_rate, shared_state=None, verbose=True)

        # If we reach here without an exception, processing was successful.
        was_successful = True

    except Exception as e:
        print(f"\nAn error occurred during the session: {e}")
        print("Processing failed. No success marker will be created.")
        if 'shared_state' in locals() and isinstance(shared_state, dict):
            shared_state["is_running"] = False
        raise
    finally:
        print("Cleaning up resources...")
        if was_successful:
            with open(marker_filepath, 'w') as f:
                f.write(f"Completed on: {datetime.datetime.now()}\n")
            print(f"Successfully created marker file: '{marker_filepath}'")

        if processing_thread and processing_thread.is_alive():
            print("Waiting for processing thread to complete...")
            processing_thread.join()

        if fig and plt.fignum_exists(fig.number):
            plt.ioff()
            plt.close(fig)

        playback.close()
        print("Processing session finished.")

    return output_filepath
