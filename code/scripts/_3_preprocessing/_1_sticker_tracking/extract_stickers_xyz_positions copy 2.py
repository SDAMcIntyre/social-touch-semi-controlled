import os
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import cv2
from pyk4a import PyK4APlayback, K4AException



# This function remains unchanged.
def _resize_with_padding(image: np.ndarray, target_dims: tuple) -> np.ndarray:
    """
    Resizes an image to a target dimension while preserving the aspect ratio
    by padding the background with black.
    """
    target_w, target_h = target_dims
    if image.shape[0] == 0 or image.shape[1] == 0:
        if len(image.shape) == 3:
            return np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            return np.zeros((target_h, target_w), dtype=image.dtype)

    src_h, src_w = image.shape[:2]
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_w = target_w
        new_h = int(new_w / src_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * src_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    if len(image.shape) == 3:
        padded_image = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded_image = np.zeros((target_h, target_w), dtype=image.dtype)
        
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return padded_image

def _create_monitoring_frame(frame_index: int,
                             color_image: np.ndarray,
                             depth_image: np.ndarray,
                             monitoring_data: dict,
                             target_dims: tuple = (1080, 1920),
                             display_dims: tuple = (1080, 1920)) -> np.ndarray:
    """
    Creates a monitoring visualization frame but does not display it.
    This function generates the canvas, draws images, overlays, and text, then returns the final image.
    """
    # --- 0. Setup Dimensions ---
    display_h, display_w = display_dims
    panel_width = 450
    images_total_width = display_w - panel_width
    if images_total_width <= 0:
        raise ValueError("Display width is too small for the text panel.")

    # --- 1. Prepare Visualizations and Draw Overlays ---
    target_h, target_w = target_dims
    color_present = color_image is not None
    depth_present = depth_image is not None

    # Use a black image as a placeholder if the source is missing
    color_vis = (cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR) if color_present and color_image.shape[:2] == target_dims 
                 else np.zeros((target_h, target_w, 3), dtype=np.uint8))
    
    if depth_present and depth_image.shape[:2] == target_dims:
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    else:
        depth_vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Draw markers on the prepared images
    for data in monitoring_data.values():
        px, py = data.get('px', np.nan), data.get('py', np.nan)
        if not np.isnan(px) and not np.isnan(py):
            center_px = (int(round(px)), int(round(py)))
            if color_present:
                cv2.drawMarker(color_vis, center_px, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            if depth_present:
                cv2.drawMarker(depth_vis, center_px, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # --- 2. Create Final Canvas and Assemble Components ---
    final_canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    
    if color_present and depth_present:
        single_image_width = images_total_width // 2
        image_target_box = (single_image_width, display_h)
        resized_color = _resize_with_padding(color_vis, image_target_box)
        resized_depth = _resize_with_padding(depth_vis, image_target_box)
        final_canvas[0:display_h, 0:single_image_width] = resized_color
        final_canvas[0:display_h, single_image_width:images_total_width] = resized_depth
    elif color_present:
        image_target_box = (images_total_width, display_h)
        resized_color = _resize_with_padding(color_vis, image_target_box)
        final_canvas[0:display_h, 0:images_total_width] = resized_color
    elif depth_present:
        image_target_box = (images_total_width, display_h)
        resized_depth = _resize_with_padding(depth_vis, image_target_box)
        final_canvas[0:display_h, 0:images_total_width] = resized_depth

    # --- 3. Draw Text on the Right Panel ---
    text_x_start = images_total_width + 15
    text_y = 30
    cv2.putText(final_canvas, f"Frame: {frame_index}", (text_x_start, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text_y += 40
    for name, data in monitoring_data.items():
        cv2.putText(final_canvas, f"- Sticker: {name}", (text_x_start, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 25
        px_str = f"Pixel: ({data.get('px', float('nan')):.1f}, {data.get('py', float('nan')):.1f})"
        cv2.putText(final_canvas, px_str, (text_x_start + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += 25
        xyz_str = f"XYZ(mm): ({data.get('x_mm', float('nan')):.1f}, {data.get('y_mm', float('nan')):.1f}, {data.get('z_mm', float('nan')):.1f})"
        cv2.putText(final_canvas, xyz_str, (text_x_start + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += 40

    return final_canvas


def _display_frame(frame: np.ndarray) -> bool:
    """
    Displays a given frame in a window and handles user input.
    Returns True if the user presses 'q', False otherwise.
    """
    cv2.imshow("XYZ Monitoring", frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')


def get_xyz_from_point_cloud(point_cloud: np.ndarray, px: float, py: float) -> tuple:
    """
    Retrieves the (x, y, z) coordinates from a point cloud at a given pixel location.
    """
    if point_cloud is None: return (np.nan, np.nan, np.nan)
    if np.isnan(px) or np.isnan(py): return (np.nan, np.nan, np.nan)

    height, width, _ = point_cloud.shape
    ix, iy = int(round(px)), int(round(py))

    if 0 <= iy < height and 0 <= ix < width:
        x_mm, y_mm, z_mm = point_cloud[iy, ix]
        if x_mm == 0 and y_mm == 0 and z_mm == 0:
            return (np.nan, np.nan, np.nan)
        return x_mm, y_mm, z_mm
    return (np.nan, np.nan, np.nan)


def extract_stickers_xyz_positions(
        source_video: str,
        center_csv_path: str,
        output_csv_path: str,
        metadata_path: str = None,      # NEW: Added metadata path argument
        monitor: bool = False,
        video_path: str = None):
    """
    Extracts 3D sticker positions and optionally shows a monitor or saves it to a video file.

    Args:
        source_video (str): Path to the source MKV file.
        center_csv_path (str): Path to the input CSV with 2D sticker centers.
        output_csv_path (str): Path to save the output CSV with 3D positions.
        metadata_path (str, optional): Path to save processing metadata as a JSON file.
        monitor (bool): If True and video_path is None, displays the monitoring window.
        video_path (str, optional): If provided, saves the monitoring view as an MP4 file.
                                    If a path is given, the window will NOT be displayed,
                                    regardless of the monitor flag.
    """
    if not os.path.exists(source_video): raise FileNotFoundError(f"Source video not found: {source_video}")
    if not os.path.exists(center_csv_path): raise FileNotFoundError(f"Center CSV not found: {center_csv_path}")

    if os.path.exists(video_path) and os.path.exists(output_csv_path):
        print(f"Output csv and video already exists {video_path}")
        return output_csv_path

    # NEW: Initialize a dictionary to hold all metadata
    processing_metadata = {
        "start_time_utc": datetime.utcnow().isoformat(),
        "inputs": {
            "source_video": os.path.abspath(source_video),
            "center_csv_path": os.path.abspath(center_csv_path),
        },
        "outputs": {
            "output_csv_path": os.path.abspath(output_csv_path),
            "video_path": os.path.abspath(video_path) if video_path else None,
            "metadata_path": os.path.abspath(metadata_path) if metadata_path else None,
        },
        "parameters": {"monitor": monitor},
        "mkv_metadata": {},
        "processing_details": {},
        "status": "In Progress",
    }
    
    print("Starting sticker 3D position extraction...")
    centers_df = pd.read_csv(center_csv_path)
    x_cols = [col for col in centers_df.columns if col.endswith('_x')]
    sticker_names = [x[:-2] for x in x_cols if f"{x[:-2]}_y" in centers_df.columns]
    
    if not sticker_names: raise ValueError("No valid sticker columns found in CSV.")
    
    print(f"Found stickers: {sticker_names}")
    processing_metadata["processing_details"]["stickers_found"] = sticker_names
    processing_metadata["processing_details"]["stickers_number_of_rows"] = len(centers_df)

    results_data = []
    playback = None
    video_writer = None
    display_dims = (1080, 1920)
    frame_index = 0
    
    try:
        playback = PyK4APlayback(source_video)
        playback.open()
        print(f"Successfully opened MKV: {source_video}")

        # NEW: Extract and store MKV metadata
        k4a_fps_to_int = {0: 0, 1: 5, 2: 15, 3: 30}
        config = playback.configuration
        fps_int = k4a_fps_to_int.get(config['camera_fps'], 30)
        processing_metadata["mkv_metadata"] = {
            "camera_fps": fps_int,
            "color_format": str(config['color_format']),
            "color_resolution": str(config['color_resolution']),
            "depth_mode": str(config['depth_mode']),
            "wired_sync_mode": str(config['wired_sync_mode']),
            "start_timestamp_offset_usec": config['start_timestamp_offset_usec'],
        }

        if video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps_int, (display_dims[1], display_dims[0]))
            if not video_writer.isOpened():
                raise IOError(f"Could not open video writer for path: {video_path}")
            print(f"Saving monitoring video to: {video_path} at {fps_int} FPS")
            
        while True:
            capture = playback.get_next_capture()
            if capture is None: break

            frame_centers = centers_df[centers_df['frame'] == frame_index]
            if not frame_centers.empty:
                point_cloud = capture.transformed_depth_point_cloud
                frame_results = {'frame': frame_index}
                monitoring_data = {}

                for name in sticker_names:
                    px, py = frame_centers[f"{name}_x"].iloc[0], frame_centers[f"{name}_y"].iloc[0]
                    # This assumes get_xyz_from_point_cloud is defined elsewhere
                    x_mm, y_mm, z_mm = get_xyz_from_point_cloud(point_cloud, px, py)
                    frame_results[f"{name}_x_mm"] = x_mm
                    frame_results[f"{name}_y_mm"] = y_mm
                    frame_results[f"{name}_z_mm"] = z_mm
                    monitoring_data[name] = {'px': px, 'py': py, 'x_mm': x_mm, 'y_mm': y_mm, 'z_mm': z_mm}
                
                results_data.append(frame_results)
                
                should_visualize = monitor or (video_path is not None)
                if should_visualize:
                    # This assumes _create_monitoring_frame is defined elsewhere
                    visual_frame = _create_monitoring_frame(
                        frame_index=frame_index, color_image=None, depth_image=capture.transformed_depth,
                        monitoring_data=monitoring_data, display_dims=display_dims
                    )
                    
                    if video_path:
                        video_writer.write(visual_frame)
                    elif monitor:
                        # This assumes _display_frame is defined elsewhere
                        if _display_frame(visual_frame):
                            print("\nMonitoring stopped by user.")
                            break
            
            frame_index += 1
            print(f"Processing frame: {frame_index}", end='\r')
        
        # NEW: Set status to completed if loop finishes naturally
        processing_metadata["status"] = "Completed"

    except EOFError:
        print(f"\nReached end of video file. Processed {frame_index} frames.")
        # NEW: Update status for a normal end-of-file event
        processing_metadata["status"] = "Completed (End of File)"
    except K4AException as e:
        print(f"\nError processing MKV file: {e}")
        # NEW: Update status and add error message on failure
        processing_metadata["status"] = "Failed"
        processing_metadata["error_message"] = str(e)
        raise
    finally:
        if playback: playback.close()
        if video_writer: video_writer.release()
        cv2.destroyAllWindows()
        
        processing_metadata["end_time_local"] = datetime.now().isoformat()
        processing_metadata["end_time_utc"] = datetime.now(timezone.utc).isoformat()
        processing_metadata["processing_details"]["frames_processed"] = frame_index
        
        if metadata_path:
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(processing_metadata, f, indent=4)
                print(f"\nSuccessfully saved metadata to: {metadata_path}")
            except Exception as e:
                print(f"\nError: Could not write metadata file to {metadata_path}. Reason: {e}")

    if results_data:
        result_df = pd.DataFrame(results_data)
        cols = ['frame'] + [f"{name}_{axis}_mm" for name in sticker_names for axis in ['x', 'y', 'z']]
        result_df = result_df[cols]
        result_df.to_csv(output_csv_path, index=False, float_format='%.2f')
        print(f"\nSuccessfully saved 3D sticker positions to: {output_csv_path}")
    else:
        print("\nNo data was processed. The output file was not created.")

