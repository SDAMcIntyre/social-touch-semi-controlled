import json
import logging
import os
import tempfile
import pickle
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import requests
import urllib3
from tqdm import tqdm

# Internal module imports (Assumed to exist in user environment)
from preprocessing.common import VideoMP4Manager
from preprocessing.motion_analysis import HamerClientAPI
from utils.should_process_task import should_process_task, clean_task_outputs

# --- Configuration & Setup ---

@dataclass
class ProcessingConfig:
    """Immutable configuration for the processing pipeline."""
    api_host: str = os.getenv("API_HOST", "localhost")
    api_port: str = os.getenv("API_PORT", "8080")
    max_retries: int = 10
    retry_delay_base: int = 2
    max_workers: int = 8  # For parallel frame processing
    person_selector: str = "right"  # giver = rightmost person; server can't parse ordinals ("second"->"1") over HTTP. Moot once the hand-tracking ROI is defined.
    hand_side: str = "right"

    @property
    def base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

# Configure Logging
logger = logging.getLogger(__name__)

# --- Utilities ---

def retry_operation(max_retries: int, delay_base: int):
    """Decorator for exponential backoff retries on network operations."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError) as e:
                    last_exception = e
                    sleep_time = delay_base * (attempt + 1)
                    if attempt < max_retries - 1:
                        logger.warning(f"Network error in {func.__name__}: {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
            logger.error(f"Operation {func.__name__} failed after {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator

# --- ROI Helpers ---

def _load_roi_config(roi_path: Path) -> Optional[dict]:
    """Read the sidecar ROI JSON. Returns None if the file does not exist."""
    if not roi_path.exists():
        return None
    try:
        with open(roi_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read ROI config {roi_path}: {e}")
        return None


def _create_cropped_video(source_path: Path, roi: dict, temp_dir: Path) -> Path:
    """
    Write a temporary cropped MP4 from source_path using the given ROI bounds.

    null values in roi are treated as full extent for that edge.
    If the resolved crop covers the entire frame, source_path is returned unchanged.

    Args:
        source_path: Original video path.
        roi:         Dict with keys x_min, x_max, y_min, y_max (int or None).
        temp_dir:    Directory where the temp file will be written.

    Returns:
        Path to the cropped video (or source_path if no cropping is needed).
    """
    cap = cv2.VideoCapture(str(source_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Resolve null to full extent and clamp to actual frame dimensions
    x_min = max(0, int(roi.get("x_min") or 0))
    x_max = min(frame_w, int(roi.get("x_max") or frame_w))
    y_min = max(0, int(roi.get("y_min") or 0))
    y_max = min(frame_h, int(roi.get("y_max") or frame_h))

    # If all null (or bounds equal full frame), no cropping needed
    if x_min == 0 and x_max == frame_w and y_min == 0 and y_max == frame_h:
        cap.release()
        logger.info("ROI covers full frame — skipping crop.")
        return source_path

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    out_path = temp_dir / f"cropped_{source_path.name}"

    writer = cv2.VideoWriter(str(out_path), fourcc_int, fps, (crop_w, crop_h))
    if not writer.isOpened():
        # Fall back to mp4v codec if the source codec is not supported for writing
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h)
        )

    logger.info(f"Cropping video to x=[{x_min},{x_max}] y=[{y_min},{y_max}] → {out_path.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame[y_min:y_max, x_min:x_max])

    cap.release()
    writer.release()
    return out_path


def _offset_roi_coordinates(results_map: Dict[int, Any], roi: dict) -> None:
    """
    Shift pixel coordinates in API results from cropped-frame space back to
    the original full-frame space by adding the ROI origin offset.

    Mutates *results_map* values in-place.  Samples that lack a ``hands``
    key (e.g. error frames) are silently skipped.
    """
    x_off = int(roi.get("x_min") or 0)
    y_off = int(roi.get("y_min") or 0)
    if x_off == 0 and y_off == 0:
        return

    for sample in results_map.values():
        for hand in sample.get("hands", []):
            bbox = hand.get("person_bounding_box_xyxy")
            if bbox is not None and len(bbox) == 4:
                hand["person_bounding_box_xyxy"] = [
                    bbox[0] + x_off,
                    bbox[1] + y_off,
                    bbox[2] + x_off,
                    bbox[3] + y_off,
                ]

            vertices = hand.get("vertices_pixel")
            if vertices is not None:
                hand["vertices_pixel"] = [
                    [v[0] + x_off, v[1] + y_off] for v in vertices
                ]


# --- Core Logic Components ---

class HandTrackingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        # video_timeout raised to 10h: batch (whole-video) mode processes every frame in
        # one request, which at local inference speed exceeds the 1h default.
        self.client = HamerClientAPI(self.config.base_url, video_timeout=(10, 36000))

    @retry_operation(max_retries=10, delay_base=2)
    def _upload_video_safe(self, path: str):
        return self.client.upload_video(
            path, 
            person_selector=self.config.person_selector, 
            hand_side=self.config.hand_side
        )

    @retry_operation(max_retries=5, delay_base=1)
    def _upload_image_safe(self, path: str):
        return self.client.upload_image(
            path,
            person_selector=self.config.person_selector,
            hand_side=self.config.hand_side
        )

    def _process_batch_mode(
        self,
        video_manager: VideoMP4Manager,
        original_video_path: Path,
        roi_video_path: Optional[Path] = None,
    ) -> Dict[int, Any]:
        """
        Executes the Video Upload strategy.
        Uploads the original file directly as a whole.
        If roi_video_path is provided, that cropped video is uploaded instead.
        """
        results_map = {}

        # STRATEGY: Upload Entire Video (Pass-through)
        logger.info("Strategy: Whole Video Processing. Uploading original source.")
        path_to_upload = str(roi_video_path if roi_video_path is not None else original_video_path)
        
        # Mapping is direct identity (0->0, 1->1, etc.)
        total_frames = len(video_manager)
        frame_mapping = list(range(total_frames))
        
        # Perform Upload
        logger.info(f"Uploading video: {path_to_upload}...")
        try:
            response_data = self._upload_video_safe(path_to_upload)
        except Exception as e:
            logger.error(f"Batch upload failed completely: {e}")
            return results_map

        # Normalize API Response
        api_samples = []
        if isinstance(response_data, dict):
            api_samples = response_data.get('samples', response_data.get('results', []))
        elif isinstance(response_data, list):
            api_samples = response_data

        # Map back to original indices
        count = min(len(api_samples), len(frame_mapping))
        if count != len(frame_mapping):
            logger.warning(f"Mismatch: Expecting {len(frame_mapping)} frames, received {len(api_samples)}.")

        for i in range(count):
            original_idx = frame_mapping[i]
            results_map[original_idx] = api_samples[i]

        return results_map

    def _process_single_frame(
        self, 
        index: int, 
        video_manager: VideoMP4Manager, 
        temp_dir: Path
    ) -> Tuple[int, Optional[Any]]:
        """Worker function for thread pool."""
        try:
            frame = video_manager[index]
            # Unique temp file per thread
            temp_path = temp_dir / f"frame_{index}_{os.getpid()}.jpg"
            if not cv2.imwrite(str(temp_path), frame):
                return index, None
            
            response = self._upload_image_safe(str(temp_path))
            
            # Clean up immediately to save disk space
            try:
                os.remove(temp_path)
            except OSError:
                pass
                
            return index, response
        except Exception as e:
            logger.error(f"Frame {index} failed: {e}")
            return index, None

    def _process_parallel_frame_mode(
        self,
        video_manager: VideoMP4Manager,
        temp_dir: str
    ) -> Dict[int, Any]:
        """
        Executes the Parallel Frame-by-Frame strategy.
        Uses ThreadPoolExecutor to handle network I/O latency.
        Processes every frame in the video manager.
        """
        results_map = {}
        total_frames = len(video_manager)
        all_indices = list(range(total_frames))
        
        temp_path_root = Path(temp_dir)
        
        logger.info(f"Processing {total_frames} frames with {self.config.max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Create a partial function to bind static arguments
            worker = functools.partial(
                self._process_single_frame, 
                video_manager=video_manager, 
                temp_dir=temp_path_root
            )
            
            # Submit all tasks
            future_to_idx = {executor.submit(worker, i): i for i in all_indices}
            
            with tqdm(total=total_frames, desc="Parallel Processing", unit="img") as pbar:
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    if result:
                        results_map[idx] = result
                    pbar.update(1)
                    
        return results_map

    def execute(
        self,
        rgb_video_path: Path,
        output_path: Path,
        use_video_api: bool,
        roi: Optional[dict] = None,
    ):
        # 1. Setup Video
        video_manager = VideoMP4Manager(rgb_video_path)
        total_frames = len(video_manager)

        # 2. Processing
        results_list: List[Optional[Any]] = [None] * total_frames

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a cropped video for batch mode when an ROI is configured
            roi_video_path: Optional[Path] = None
            if use_video_api and roi is not None:
                cropped = _create_cropped_video(rgb_video_path, roi, temp_path)
                # _create_cropped_video returns source_path unchanged when no crop needed
                if cropped != rgb_video_path:
                    roi_video_path = cropped

            if use_video_api:
                logger.info("Mode: Batch Video API (Whole Video)")
                processed_map = self._process_batch_mode(
                    video_manager=video_manager,
                    original_video_path=rgb_video_path,
                    roi_video_path=roi_video_path,
                )
            else:
                logger.info("Mode: Parallel Frame Extraction (Whole Video)")
                processed_map = self._process_parallel_frame_mode(
                    video_manager, temp_dir
                )

        # 2b. Offset pixel coordinates when ROI cropping was applied
        if roi_video_path is not None and roi is not None:
            _offset_roi_coordinates(processed_map, roi)

        # 3. Assembly & Serialization
        logger.info("Assembling results...")
        for i, data in processed_map.items():
            if i < total_frames and data:
                results_list[i] = {
                    "frame_index": i,
                    "api_response": data
                }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results_list, f)
        
        logger.info(f"Saved {len(results_list)} records to {output_path}")

# --- Entry Point ---

def track_hands_on_video(
    rgb_video_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
    use_video_api: bool = True,
    roi_path: Optional[Path] = None,
    keep_stale: bool = False,
):
    """
    Entry point for whole-video hand tracking.

    Args:
        rgb_video_path: Path to the input video.
        output_path:    Destination for the pickle file.
        force_processing: Ignore task cache check.
        use_video_api:  Use batch video upload (True) or frame-by-frame (False).
        roi_path:       Optional path to a ``*_handmodel_roi.json`` sidecar file.
                        When present and the file exists, the video is cropped to
                        the specified region before being uploaded to the API.
                        Gracefully ignored if the file does not exist.
    """
    # 0. Check Processing Status
    input_paths = [rgb_video_path]
    if roi_path is not None and roi_path.exists():
        input_paths.append(roi_path)
    if not should_process_task(
        input_paths=input_paths,
        output_paths=[output_path],
        force=force_processing,
        keep_stale=keep_stale,
    ):
        logger.info(f"Skipping: {output_path} is up to date.")
        return
    clean_task_outputs(output_path)

    # 1. Load ROI configuration (None if file absent or unreadable)
    roi = _load_roi_config(roi_path) if roi_path is not None else None
    if roi is not None:
        logger.info(f"ROI config loaded from {roi_path}: {roi}")

    # 2. Initialize Config
    config = ProcessingConfig(
        max_workers=16  # Aggressive I/O threading for HTTP requests
    )

    # 3. Run Pipeline
    pipeline = HandTrackingPipeline(config)

    try:
        pipeline.execute(
            rgb_video_path=rgb_video_path,
            output_path=output_path,
            use_video_api=use_video_api,
            roi=roi,
        )
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise