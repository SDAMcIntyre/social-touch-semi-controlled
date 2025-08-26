import cv2
import os

def generate_foreground_masks(
    video_path: str,
    output_path: str,
    history: int = 2000,
    var_threshold: int = 16,
    detect_shadows: bool = False
) -> str:
    """
    Processes a video to generate grayscale foreground masks using GMM.

    This function reads an input video, applies a Gaussian Mixture Model (GMM)
    background subtractor to each frame, and saves the resulting grayscale
    foreground masks to a new video file.

    Args:
        video_path (str): The full path to the input video file.
        output_path (str): The full path where the output mask video will be saved.
        history (int, optional): The number of last frames that affect the
            background model. If this value is greater than half the total
            frames in the video, it will be automatically capped. Defaults to 2000.
        var_threshold (int, optional): Threshold on the squared Mahalanobis
            distance to decide if a pixel is foreground. A higher value reduces
            the number of pixels detected as foreground. Defaults to 16.
        detect_shadows (bool, optional): If True, the algorithm will detect
            and mark shadows in the mask. Defaults to False.

    Returns:
        str: The path to the generated mask video file.
        
    Raises:
        FileNotFoundError: If the input video file does not exist.
        IOError: If the video file cannot be opened or the output file cannot be created.
    """
    print(f"Starting mask generation for: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # validate and adjust the 'history' parameter
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        max_history = total_frames-1
        if history > max_history:
            print(f"⚠️ Warning: 'history' value ({history}) is too high for a video with {total_frames} frames.")
            history = max_history
            print(f"   -> Adjusting 'history' to the maximum allowed value: {history}")
    
    # Create a GMM background subtractor object with the (potentially adjusted) parameters.
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=history, 
        varThreshold=var_threshold, 
        detectShadows=detect_shadows
    )

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create or write to output file: {output_path}")

    print(f"Processing frames and saving masks to: {output_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)
        out.write(fg_mask)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")

    print(f"Finished processing. Total frames: {frame_count}")

    cap.release()
    out.release()
    
    return output_path



def remove_video_background(
    video_path: str,
    metadata_path: str,
    output_path: str
):
    pass