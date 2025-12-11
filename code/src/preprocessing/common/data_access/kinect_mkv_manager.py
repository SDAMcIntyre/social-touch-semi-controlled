import csv
import logging
from pathlib import Path
from typing import Iterator, Dict, Literal

import cv2
import numpy as np
import open3d as o3d
from pyk4a import K4AException, PyK4APlayback, PyK4ACapture
from pyk4a.config import ImageFormat, FPS, ColorResolution, DepthMode

# --- Setup & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Structures ---

class KinectFrame:
    """
    A container for a single frame's data with lazy-loading capabilities.
    """
    def __init__(self, capture: PyK4ACapture, color_format: ImageFormat):
        self._capture = capture
        self._color_format = color_format

        # Private attributes for caching the processed data once loaded
        self._color: np.ndarray | None | bool = False
        self._depth: np.ndarray | None | bool = False
        self._transformed_depth: np.ndarray | None | bool = False
        self._transformed_depth_point_cloud: np.ndarray | None | bool = False

    @property
    def color(self) -> np.ndarray | None:
        """Decoded color image (BGR format). Lazily loaded on first access."""
        if self._color is False:
            if self._capture.color is None:
                self._color = None
            elif self._color_format == ImageFormat.COLOR_MJPG:
                self._color = cv2.imdecode(self._capture.color, cv2.IMREAD_COLOR)
            else:
                self._color = self._capture.color
        return self._color

    @property
    def depth(self) -> np.ndarray | None:
        """Depth map in millimeters. Lazily loaded on first access."""
        if self._depth is False:
            self._depth = self._capture.depth
        return self._depth

    @property
    def transformed_depth(self) -> np.ndarray | None:
        """Depth map transformed to the color camera's geometry. Lazily loaded."""
        if self._transformed_depth is False:
            self._transformed_depth = self._capture.transformed_depth
        return self._transformed_depth

    @property
    def transformed_depth_point_cloud(self) -> np.ndarray | None:
        """3D point cloud from transformed depth. Lazily loaded."""
        if self._transformed_depth_point_cloud is False:
            self._transformed_depth_point_cloud = self._capture.transformed_depth_point_cloud
        return self._transformed_depth_point_cloud

    def generate_o3d_point_cloud(self) -> o3d.geometry.PointCloud | None:
        if self.transformed_depth_point_cloud is None or self.color is None:
            return None
        
        xyz, color_img = self.transformed_depth_point_cloud, self.color
        valid_mask = (xyz[:, :, 2] > 0) & ~np.isnan(xyz).any(axis=2)
        points_xyz = xyz[valid_mask]
        points_rgb = color_img[valid_mask][:, ::-1] / 255.0  # BGR to RGB
        
        if len(points_xyz) == 0:
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb)
        return pcd

    def convert_xy_to_xyz(self, xy: tuple[int, int]) -> np.ndarray:
        x, y = xy
        point_cloud_map = self.transformed_depth_point_cloud
        if point_cloud_map is not None and 0 <= y < point_cloud_map.shape[0] and 0 <= x < point_cloud_map.shape[1]:
            xyz_point = point_cloud_map[y, x]
            if np.any(np.isnan(xyz_point)) or np.all(xyz_point == 0):
                    return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return xyz_point
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_depth_for_viewing(self) -> np.ndarray | None:
        depth_map = self.depth
        if depth_map is None:
            return None
        depth_clipped = np.clip(depth_map, 0, 5000)
        norm_depth = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)

# --- Internal Implementation ---

class _KinectPlayback:
    """[Internal] A low-level context manager for a PyK4APlayback resource."""
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.playback: PyK4APlayback | None = None
        self.is_open = False

    def open(self) -> PyK4APlayback:
        try:
            self.playback = PyK4APlayback(self.video_path)
            self.playback.open()
            self.is_open = True
            return self.playback
        except K4AException as e:
            raise K4AException(f"Failed to open MKV file '{self.video_path}'.") from e

    def close(self):
        if self.playback and self.is_open:
            self.playback.close()
            self.is_open = False

class _KinectReader:
    """[Internal] Handles list-like reading and navigation."""
    def __init__(self, playback: PyK4APlayback, seek_strategy: Literal['timestamp', 'sequential'] = 'timestamp'):
        self._playback = playback
        self._seek_strategy = seek_strategy
        fps_map = {FPS.FPS_5: 5, FPS.FPS_15: 15, FPS.FPS_30: 30}
        self.fps = fps_map.get(self._playback.configuration['camera_fps'], 30)
        self.total_frames = int((self._playback.length / 1_000_000) * self.fps)
        self._color_format = self._playback.configuration['color_format']
        self._current_frame_index = -1

    def _get_frame_from_capture(self, capture: PyK4ACapture) -> KinectFrame | None:
        if capture is None:
            return None
        return KinectFrame(capture, self._color_format)

    def seek(self, frame_index: int):
        """
        Moves the internal pointer to the position *before* the requested frame_index.
        """
        if not (0 <= frame_index < self.total_frames):
            raise IndexError(f"Frame index {frame_index} out of bounds (Total: {self.total_frames}).")

        if self._seek_strategy == 'timestamp':
            timestamp_usec = int(frame_index * (1_000_000 / self.fps))
            self._playback.seek(timestamp_usec)
            # We approximate the index; read_once will confirm logic by fetching the next frame
            self._current_frame_index = frame_index - 1
            
        elif self._seek_strategy == 'sequential':
            # Check if requested frame is behind current
            if frame_index <= self._current_frame_index:
                dist_back = self._current_frame_index - frame_index
                dist_from_start = frame_index
                
                # OPTIMIZATION: Use get_previous_capture if target is closer to current than to 0.
                if dist_back < dist_from_start:
                    # Backward traversal:
                    # We need to decrement until we are just before frame_index.
                    # Since read_once() increments current_index, we need the state 
                    # where current_index == frame_index - 1.
                    while self._current_frame_index >= frame_index:
                        try:
                            self._playback.get_previous_capture()
                            self._current_frame_index -= 1
                        except EOFError:
                            # Fallback if reverse seek hits limit unexpectedly
                            self._playback.seek(0)
                            self._current_frame_index = -1
                            break
                else:
                    # Standard reset to beginning
                    self._playback.seek(0)
                    self._current_frame_index = -1
            
            # Fast forward (if necessary) by consuming captures until we are just before the target
            while self._current_frame_index < frame_index - 1:
                try:
                    self._playback.get_next_capture()
                    self._current_frame_index += 1
                except EOFError:
                    raise IndexError(f"Seek failed: End of stream reached at index {self._current_frame_index}")

    def read_once(self) -> KinectFrame | None:
        """
        Reads exactly one capture from the stream.
        Returns None if the capture is empty/invalid but stream continues.
        Raises EOFError if stream ends.
        """
        # Let PyK4A raise EOFError if we hit the end
        capture = self._playback.get_next_capture()
        self._current_frame_index += 1
        
        frame_data = self._get_frame_from_capture(capture)
        # Verify frame has data
        if frame_data and (frame_data.color is not None or frame_data.depth is not None):
            return frame_data
        
        return None

    def read(self) -> KinectFrame | None:
        """
        Reads the next *valid* frame, skipping empty captures.
        """
        try:
            while True:
                # Requirement 1: Uses read_once
                frame = self.read_once()
                if frame is not None:
                    return frame
        except EOFError:
            return None

    def __getitem__(self, frame_index: int) -> KinectFrame:
        # Requirement 2: Optimized getitem
        # Only seek if we are NOT at the immediate predecessor
        if frame_index != self._current_frame_index + 1:
            self.seek(frame_index)
        
        try:
            frame = self.read_once()
            if frame is None:
                # We successfully grabbed a capture, but it was empty/invalid
                raise ValueError(f"Frame at index {frame_index} contains no valid data.")
            return frame
        except EOFError:
            raise IndexError(f"Frame index {frame_index} is out of bounds (EOF).")

# --- The Public Facade Class ---

class KinectMKV:
    """
    A unified, high-level interface for interacting with Azure Kinect MKV recordings.
    """
    def __init__(self, video_path: str | Path, seek_strategy: Literal['timestamp', 'sequential'] = 'timestamp'):
        """
        Args:
            video_path: Path to the MKV file.
            seek_strategy: 'timestamp' (fast, default) or 'sequential' (slower, more accurate).
                           Sequential mode resets to start if seeking backwards.
        """
        self._playback_manager = _KinectPlayback(video_path)
        self._playback: PyK4APlayback | None = None
        self._reader: _KinectReader | None = None
        self._seek_strategy = seek_strategy
        
    def __enter__(self) -> 'KinectMKV':
        self._playback = self._playback_manager.open()
        # Initialize reader with the configured strategy
        self._reader = _KinectReader(self._playback, seek_strategy=self._seek_strategy)
        logging.info(f"Opened '{self._playback_manager.video_path}'. Frames: ~{len(self)}, FPS: {self._reader.fps}, Mode: {self._seek_strategy}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._playback_manager.close()
        logging.info(f"Closed '{self._playback_manager.video_path}'.")
    
    @property
    def length(self) -> int:
        if not self._playback:
            raise RuntimeError("MKV not opened. Use within a 'with' block.")
        return self._playback.length
       
    # --- Delegated Reader Methods for Interactive Use ---
    def __len__(self) -> int:
        if not self._reader: raise RuntimeError("MKV not opened. Use within a 'with' block.")
        return self._reader.total_frames

    def __getitem__(self, frame_index: int) -> KinectFrame:
        if not self._reader: raise RuntimeError("MKV not opened. Use within a 'with' block.")
        # Requirement 2: Delegate to Reader's optimized getitem
        return self._reader[frame_index]
        
    def __iter__(self) -> Iterator[KinectFrame]:
        if not self._reader: raise RuntimeError("MKV not opened. Use within a 'with' block.")
        self._reader.seek(0)
        while (frame := self._reader.read()) is not None:
            yield frame

    # --- High-Level Analysis Method ---
    def run_analysis_report(self, report_path: str | Path):
        if not self._playback: raise RuntimeError("MKV not opened. Use within a 'with' block.")
        
        logging.info(f"Starting stream analysis for '{self._playback_manager.video_path}'...")
        validator = _KinectValidator(self._playback)
        report_generator = validator.analyze()
        self._save_report_to_csv(report_generator, report_path)
        logging.info(f"Analysis complete. Report saved to '{report_path}'.")

    def _save_report_to_csv(self, report_iterator: Iterator[Dict], output_path: str | Path):
        try:
            first_item = next(report_iterator)
        except StopIteration:
            logging.warning("No frames to analyze, CSV report will be empty.")
            return

        with open(output_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=first_item.keys())
            writer.writeheader()
            writer.writerow(first_item)
            processed_count = 1
            for row_data in report_iterator:
                writer.writerow(row_data)
                processed_count += 1
                print(f"Frames processed for report: {processed_count}", end='\r')
        print()

class _KinectValidator:
    """[Internal] Analyzes an MKV file for stream integrity."""
    def __init__(self, playback: PyK4APlayback):
        CUSTOM_RESOLUTION_MAP = {
            ColorResolution.RES_720P: (1280, 720), ColorResolution.RES_1080P: (1920, 1080),
            ColorResolution.RES_1440P: (2560, 1440), ColorResolution.RES_1536P: (2048, 1536),
            ColorResolution.RES_2160P: (3840, 2160), ColorResolution.RES_3072P: (4096, 3072),
        }
        CUSTOM_DEPTH_MODE_MAP = {
            DepthMode.NFOV_2X2BINNED: (320, 288), DepthMode.NFOV_UNBINNED: (640, 576),
            DepthMode.WFOV_2X2BINNED: (512, 512), DepthMode.WFOV_UNBINNED: (1024, 1024),
            DepthMode.PASSIVE_IR: (1024, 1024),
        }
        self._playback = playback
        self._playback.seek(0)
        self._config = self._playback.configuration
        self.expected_color_shape = CUSTOM_RESOLUTION_MAP.get(self._config['color_resolution'], (0,0))
        self.expected_depth_shape = CUSTOM_DEPTH_MODE_MAP.get(self._config['depth_mode'], (0,0))[:2]
        
    def analyze(self) -> Iterator[Dict]:
        frame_index = 0
        while True:
            try:
                capture = self._playback.get_next_capture()
                if capture is None: continue 
                yield {
                    'frame_index': frame_index,
                    'has_rgb': self._validate_rgb(capture),
                    'has_depth': self._validate_depth(capture)
                }
                frame_index += 1
            except EOFError:
                break
    
    def _validate_rgb(self, capture) -> bool:
        if capture.color is None: return False
        img = (cv2.imdecode(capture.color, cv2.IMREAD_COLOR) 
               if self._config['color_format'] == ImageFormat.COLOR_MJPG 
               else capture.color)
        return (img.shape[1], img.shape[0]) == self.expected_color_shape if img is not None else False

    def _validate_depth(self, capture) -> bool:
        if capture.depth is None: return False
        return (capture.depth.shape[1], capture.depth.shape[0]) == self.expected_depth_shape