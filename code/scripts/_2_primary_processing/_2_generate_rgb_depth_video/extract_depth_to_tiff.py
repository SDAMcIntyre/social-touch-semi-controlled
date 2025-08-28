import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyk4a import PyK4APlayback, K4AException, PyK4ACapture
import datetime
import tifffile as tiff
import threading

def processing_func(playback: PyK4APlayback, 
                    output_dir: str, 
                    base_filename: str, 
                    shared_state: dict = None,
                    verbose: bool = False):
    """
    This function processes an MKV file, extracting and saving the point cloud.
    
    It can run in two modes:
    - Threaded mode (if shared_state is provided): Updates a shared dictionary
      for a viewer thread and can be stopped externally.
    - Sequential mode (if shared_state is None): Runs standalone, processing
      the entire file.
    """
    frame_count = 0
    try:
        while True:
            # In threaded mode, this provides a mechanism to stop the loop from another thread.
            if shared_state and not shared_state.get("is_running", True):
                print("Processing thread received stop signal.")
                break

            # Get the next capture. This will raise EOFError at the end of the file.
            capture = playback.get_next_capture()

            # If in threaded mode, update the shared state for the viewer.
            if shared_state:
                with shared_state["lock"]:
                    shared_state["latest_capture"] = capture
            
            # --- Core processing logic (runs in both modes) ---
            if capture.transformed_depth_point_cloud is not None:
                depth_point_cloud = capture.transformed_depth_point_cloud
                # Transpose to (W, H, C) for export, then make it contiguous to avoid
                # stride issues in some TIFF readers (e.g., OpenCV) that can yield
                # half-black images.
                point_cloud_for_tiff = np.ascontiguousarray(depth_point_cloud.transpose(1, 0, 2))
                output_filename = f"{base_filename}_point_cloud-{frame_count:04d}.tiff"
                output_filepath = os.path.join(output_dir, output_filename)
                
                tiff.imwrite(output_filepath, point_cloud_for_tiff)
                frame_count += 1
                
                # Provide progress feedback in sequential mode
                if verbose:
                    print(f"Processed frames: {frame_count}", end='\r')

    except EOFError:
        print(f"\nProcessing finished. Reached end of file. A total of {frame_count} frames were saved.")
    except Exception as e:
        print(f"An error occurred in the processing function: {e}")
        # Re-raise the exception to allow the main thread to handle it
        raise
    finally:
        # Signal that processing is complete, primarily for the threaded mode.
        if shared_state:
            shared_state["is_running"] = False

def extract_depth_to_tiff(mkv_path: str, output_dir: str, show: bool = False, frame_rate: int = 30):
    """
    Extracts depth frames from an MKV file and saves them as TIFF images.
    It calls a unified processing function either directly (for speed) or in a
    separate thread (to allow for visualization).
    """
    if not os.path.exists(mkv_path):
        raise FileNotFoundError(f"Input file not found at: {mkv_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(mkv_path))[0]
    marker_filename = f".SUCCESS_{base_filename}"
    marker_filepath = os.path.join(output_dir, marker_filename)

    if os.path.exists(marker_filepath):
        print(f"Skipping '{mkv_path}' as it appears to be complete.")
        return

    try:
        playback = PyK4APlayback(mkv_path)
        playback.open()
    except K4AException as e:
        raise K4AException(f"Failed to open MKV file. Error: {e}")

    duration_sec = playback.length / 1_000_000
    print(f"Successfully opened MKV. Recording duration: {duration_sec:.2f} seconds.")

    processing_thread = None
    was_successful = False
    fig = None # Initialize fig to None

    try:
        if show:
            # --- Threaded execution for visualization ---
            print("Running in visualization mode (multi-threaded).")
            shared_state = {
                "latest_capture": None,
                "is_running": True,
                "lock": threading.Lock(),
            }

            processing_thread = threading.Thread(
                target=processing_func,
                args=(playback, output_dir, base_filename, shared_state),
                daemon=True
            )
            processing_thread.start()

            fig = plt.figure(figsize=(15, 8))
            plt.ion()
            target_pause = 1.0 / frame_rate if frame_rate > 0 else 0.01

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
            
            # Wait for the processing thread to finish its work
            processing_thread.join()

        else:
            # --- Sequential execution for processing only ---
            print("Running in processing-only mode (single-threaded).")
            # Call the processing function directly in the main thread.
            processing_func(playback, output_dir, base_filename, shared_state=None)

        # If we reach here without an exception, processing was successful.
        was_successful = True

    except Exception as e:
        print(f"\nAn error occurred during the session: {e}")
        print("Processing failed. No success marker will be created.")
        # Ensure the thread is stopped if an error occurs in the main loop
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


def plot_capture(capture: PyK4ACapture, fig: plt.Figure):
    """
    Plots all available images from a capture onto a given figure object.
    This function now handles compressed MJPEG color frames.
    """
    image_specs = {
        "color": "Color", "depth": "Depth", "ir": "Infrared",
        "transformed_depth": "Transformed Depth",
        "transformed_depth_point_cloud": "Transformed Depth Point Cloud",
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
    cols = 3 if num_images > 4 else 2
    rows = int(np.ceil(num_images / cols))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()

    for i, spec in enumerate(available_images):
        ax = axes[i]
        title = spec["title"]
        image_data = spec["image"].copy()
        ax.set_title(title)
        ax.axis('off')

        if "Point Cloud" in title:
            image_data[np.isinf(image_data)] = 0
            image_data[np.isnan(image_data)] = 0
            normalized_channels = [cv2.normalize(image_data[:, :, ch], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for ch in range(3)]
            pseudo_color_image = cv2.merge(normalized_channels)
            ax.imshow(cv2.cvtColor(pseudo_color_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{title}\n(X,Y,Z mapped to R,G,B)")
            
        elif "Color" in title:
            # --- MODIFICATION START ---
            # Check if the image is compressed (1D array) or uncompressed (3D array)
            if image_data.ndim == 1:
                # It's a compressed MJPEG frame. Decode it from the buffer.
                image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image_data is None:
                    print("Warning: Failed to decode MJPEG color frame.")
                    continue  # Skip plotting this frame
                # Convert the decoded BGR image to RGB for matplotlib
                ax.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                # It's an uncompressed BGRA frame. Convert to RGB as before.
                ax.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB))
            # --- MODIFICATION END ---
            
        else:  # Depth and IR
            norm = None
            if "Depth" in title and image_data.size > 0 and np.any(image_data > 0):
                non_zero_vals = image_data[image_data > 0]
                vmin = np.min(non_zero_vals)
                vmax = np.max(image_data)
                if vmin < vmax:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
            im = ax.imshow(image_data, cmap="jet", norm=norm)
            fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Kinect Capture Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == '__main__':
    try:
        mkv_file_path = "path/to/your/file.mkv"
        output_directory = "./mkv_extraction_output"
        
        if not os.path.exists(mkv_file_path):
             print("="*50)
             print("WARNING: Example MKV file not found.")
             print(f"Please update 'mkv_file_path' in the '__main__' block to a valid .mkv file.")
             print("="*50)
        else:
            extract_depth_to_tiff(mkv_file_path, output_directory, show=True, frame_rate=2)

    except FileNotFoundError as e:
        print(e)
    except K4AException as e:
        print(f"A Kinect SDK error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the session: {e}")
