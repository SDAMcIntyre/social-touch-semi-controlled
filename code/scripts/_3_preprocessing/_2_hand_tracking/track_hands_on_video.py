import json
import logging
import os
import tempfile
import pickle
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import pandas as pd
import requests
import urllib3
from tqdm import tqdm

# Internal module imports (Assumed to exist in user environment)
from preprocessing.common import VideoMP4Manager
from preprocessing.motion_analysis import HamerClientAPI
from utils.should_process_task import should_process_task

# --- Configuration & Setup ---

@dataclass
class ProcessingConfig:
    """Immutable configuration for the processing pipeline."""
    api_host: str = os.getenv("API_HOST", "localhost")
    api_port: str = os.getenv("API_PORT", "8080")
    max_retries: int = 10
    retry_delay_base: int = 2
    max_workers: int = 8  # For parallel frame processing
    person_selector: str = "right"
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

def _is_valid_trial_id(value: Any) -> bool:
    """Validates trial ID: Not None, Not NaN, and > 0."""
    if pd.isna(value):
        return False
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False

# --- Core Logic Components ---

class VideoConsolidator:
    """Handles the creation of temporary batch videos from non-contiguous frames."""
    
    @staticmethod
    def get_valid_chunks(trial_ids: List[Any], limit: int) -> List[Tuple[int, int]]:
        chunks = []
        start_idx = None
        for i in range(limit):
            is_valid = _is_valid_trial_id(trial_ids[i])
            if is_valid and start_idx is None:
                start_idx = i
            elif not is_valid and start_idx is not None:
                chunks.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            chunks.append((start_idx, limit))
        return chunks

    @staticmethod
    def create_batch_video(
        video_manager: VideoMP4Manager, 
        chunks: List[Tuple[int, int]], 
        output_path: str
    ) -> List[int]:
        """
        Writes frames to video. 
        Returns mapping: new_video_frame_index -> original_video_frame_index.
        """
        if not chunks:
            return []

        # Get dimensions from first valid frame
        first_frame = video_manager[chunks[0][0]]
        height, width, _ = first_frame.shape
        
        # Initialize Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        frame_mapping = []
        try:
            for start, end in chunks:
                for i in range(start, end):
                    frame = video_manager[i]
                    writer.write(frame)
                    frame_mapping.append(i)
        finally:
            writer.release()
            
        return frame_mapping

class HandTrackingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = HamerClientAPI(self.config.base_url)

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
        trial_ids: List[Any], 
        process_limit: int, 
        temp_dir: str,
        original_video_path: Path,
        process_whole_video: bool
    ) -> Dict[int, Any]:
        """
        Executes the Video Upload strategy.
        If process_whole_video is True: Uploads the original file directly.
        If process_whole_video is False: Consolidates valid chunks into a temp video first.
        """
        results_map = {}
        path_to_upload = ""
        frame_mapping: List[int] = []

        if process_whole_video:
            # STRATEGY A: Upload Entire Video (Pass-through)
            logger.info("Strategy: Whole Video Processing. Uploading original source.")
            path_to_upload = str(original_video_path)
            # Mapping is direct identity (0->0, 1->1, etc.)
            frame_mapping = list(range(len(video_manager)))
        
        else:
            # STRATEGY B: Consolidate Chunks
            logger.info("Strategy: Chunk Processing. Consolidating valid frames.")
            valid_chunks = VideoConsolidator.get_valid_chunks(trial_ids, process_limit)
            
            if not valid_chunks:
                logger.warning("No valid trial IDs found for batch processing.")
                return results_map

            temp_vid_path = os.path.join(temp_dir, "consolidated_batch.mp4")
            
            frame_mapping = VideoConsolidator.create_batch_video(
                video_manager, valid_chunks, temp_vid_path
            )
            path_to_upload = temp_vid_path

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
        # If we uploaded the whole video, api_samples[i] corresponds to original frame i.
        # If we uploaded chunks, api_samples[i] corresponds to frame_mapping[i].
        
        count = min(len(api_samples), len(frame_mapping))
        if count != len(frame_mapping):
            logger.warning(f"Mismatch: Expecting {len(frame_mapping)} frames, received {len(api_samples)}.")

        for i in range(count):
            original_idx = frame_mapping[i]
            # When processing the whole video, we might process frames that are NOT valid trials.
            # We filter the result map to only include indices that are valid trials, 
            # unless we want to return everything. 
            # Given the context of 'trial_ids', we likely only care about valid ones for the final output,
            # but we populate the map regardless.
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
        trial_ids: List[Any],
        process_limit: int,
        temp_dir: str
    ) -> Dict[int, Any]:
        """
        Executes the Parallel Frame-by-Frame strategy.
        Uses ThreadPoolExecutor to handle network I/O latency.
        """
        results_map = {}
        valid_indices = [
            i for i in range(process_limit) 
            if _is_valid_trial_id(trial_ids[i])
        ]
        
        temp_path_root = Path(temp_dir)
        
        logger.info(f"Processing {len(valid_indices)} frames with {self.config.max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Create a partial function to bind static arguments
            worker = functools.partial(
                self._process_single_frame, 
                video_manager=video_manager, 
                temp_dir=temp_path_root
            )
            
            # Submit all tasks
            future_to_idx = {executor.submit(worker, i): i for i in valid_indices}
            
            with tqdm(total=len(valid_indices), desc="Parallel Processing", unit="img") as pbar:
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    if result:
                        results_map[idx] = result
                    pbar.update(1)
                    
        return results_map

    def execute(
        self,
        rgb_video_path: Path,
        trial_id_path: Path,
        output_path: Path,
        use_video_api: bool,
        process_whole_video: bool
    ):
        # 1. Load Data
        df = pd.read_csv(trial_id_path)
        if "trial_id" not in df.columns:
            raise ValueError(f"CSV {trial_id_path} missing 'trial_id'")
        trial_ids = df["trial_id"].tolist()

        # 2. Setup Video
        video_manager = VideoMP4Manager(rgb_video_path)
        total_frames = len(video_manager)
        process_limit = min(total_frames, len(trial_ids))
        
        # 3. Processing
        results_list: List[Optional[Any]] = [None] * total_frames
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if use_video_api:
                logger.info("Mode: Batch Video API")
                processed_map = self._process_batch_mode(
                    video_manager=video_manager, 
                    trial_ids=trial_ids, 
                    process_limit=process_limit, 
                    temp_dir=temp_dir,
                    original_video_path=rgb_video_path,
                    process_whole_video=process_whole_video
                )
            else:
                logger.info("Mode: Parallel Frame Extraction")
                processed_map = self._process_parallel_frame_mode(
                    video_manager, trial_ids, process_limit, temp_dir
                )

        # 4. Assembly & Serialization
        logger.info("Assembling results...")
        for i, data in processed_map.items():
            # If we processed the whole video, we might have data for frames 
            # that don't have valid trial_ids. Only save if within bounds.
            if i < len(trial_ids) and data:
                # Optional: Strict filtering to only save valid trials
                # if _is_valid_trial_id(trial_ids[i]):
                results_list[i] = {
                    "frame_index": i,
                    "trial_id": int(trial_ids[i]) if _is_valid_trial_id(trial_ids[i]) else -1,
                    "api_response": data
                }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results_list, f)
        
        logger.info(f"Saved {len(results_list)} records to {output_path}")

# --- Entry Point ---

def track_hands_on_video(
    rgb_video_path: Path, 
    trial_id_path: Path, 
    output_path: Path, 
    *, 
    force_processing: bool = False,
    use_video_api: bool = True,
    process_whole_video: bool = True
): 
    """
    Refactored entry point.
    
    Args:
        rgb_video_path: Path to the input video.
        trial_id_path: Path to the CSV containing trial IDs.
        output_path: Destination for the pickle file.
        force_processing: Ignore task cache check.
        use_video_api: Use batch video upload (True) or frame-by-frame (False).
        process_whole_video: If using video API, True uploads the entire file. 
                             False creates a consolidated video of only relevant chunks.
    """
    # 0. Check Processing Status
    if not should_process_task(
        input_paths=[rgb_video_path, trial_id_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logger.info(f"Skipping: {output_path} is up to date.")
        return

    # 1. Initialize Config
    config = ProcessingConfig(
        max_workers=16  # Aggressive I/O threading for HTTP requests
    )

    # 2. Run Pipeline
    pipeline = HandTrackingPipeline(config)
    
    try:
        pipeline.execute(
            rgb_video_path=rgb_video_path,
            trial_id_path=trial_id_path,
            output_path=output_path,
            use_video_api=use_video_api,
            process_whole_video=process_whole_video
        )
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise

    return