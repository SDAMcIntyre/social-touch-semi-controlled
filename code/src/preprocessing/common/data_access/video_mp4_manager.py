# Standard library imports
import logging
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, List, Optional, Union

# Third-party imports
import cv2
import numpy as np
# tqdm is an optional dependency for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ColorFormat(Enum):
    """Provides type-safe color format options."""
    BGR = auto()
    RGB = auto()

@contextmanager
def video_capture(video_path: Union[str, Path]):
    """A context manager for cv2.VideoCapture to ensure resources are always released."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {video_path}")
    try:
        yield cap
    finally:
        cap.release()
        logging.debug(f"Video capture for '{video_path}' released.")

class VideoMP4Manager:
    """
    A robust and efficient manager for reading video files using OpenCV.

    This class provides a lazy-loaded, list-like interface to a video's frames.
    It opens the video file only when frames are requested, ensuring efficient
    resource management. It also supports optional pre-loading of all frames
    into memory for performance-critical applications.

    Features:
    - Lazy loading by default (low memory usage).
    - Efficient slicing and indexing (`video[10]`, `video[100:200]`).
    - On-the-fly color conversion (RGB/BGR).
    - Optional pre-loading into memory.
    - Context-manager-based file handling for safety.
    """
    def __init__(self, video_path: Union[str, Path], color_format: ColorFormat = ColorFormat.BGR):
        """
        Initializes the VideoMP4Manager by reading video metadata.

        Args:
            video_path (Union[str, Path]): Path to the video file.
            color_format (ColorFormat): The desired color format for frames.
                                        Defaults to ColorFormat.BGR.
        """
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        self._color_format = color_format
        self._frames: Optional[List[np.ndarray]] = None  # Cache for pre-loaded frames

        # Read metadata without holding the file open
        with video_capture(self.video_path) as cap:
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = cap.get(cv2.CAP_PROP_FPS)

    @property
    def color_format(self) -> ColorFormat:
        """Gets the current color format (ColorFormat.RGB or ColorFormat.BGR)."""
        return self._color_format

    @color_format.setter
    def color_format(self, new_format: ColorFormat):
        """
        Sets the color format. If frames are pre-loaded, this will trigger
        a re-load or in-memory conversion. For simplicity, this implementation
        clears the cache.
        """
        if not isinstance(new_format, ColorFormat):
            raise TypeError("color_format must be an instance of ColorFormat Enum")
        if new_format != self._color_format:
            logging.info(f"Color format changed to {new_format.name}. Clearing frame cache if it exists.")
            self._color_format = new_format
            self._frames = None # Invalidate cache

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Centralized method to handle color conversion."""
        if self._color_format == ColorFormat.RGB:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def preload(self, progress_bar: Optional[tqdm] = None):
        """
        Loads the entire video into memory for fast access.

        Args:
            progress_bar (tqdm, optional): An instance of tqdm for progress visualization.
                                           If None, no progress bar is shown.
        """
        if self.is_preloaded:
            logging.info("Video is already pre-loaded in memory.")
            return

        logging.info(f"Pre-loading all {self.total_frames} frames into memory...")
        # Use the efficient slicing implementation to load all frames
        frame_iterator = self[0:self.total_frames]
        
        # Use provided tqdm instance or create a default one if available
        if progress_bar:
            progress_bar.reset(total=self.total_frames)
            self._frames = [frame for frame in frame_iterator if progress_bar.update(1) or True]
        else:
            self._frames = list(frame_iterator)
        
        logging.info("Pre-loading complete.")

    @property
    def is_preloaded(self) -> bool:
        """Returns True if the video frames are cached in memory."""
        return self._frames is not None

    def __len__(self) -> int:
        """Returns the total number of frames in the video."""
        return self._total_frames

    def __getitem__(self, index: Union[int, slice]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Retrieves frames by index or slice, providing list-like access.

        This method is highly efficient. For slices, it seeks only once and
        reads frames sequentially.
        """
        if self.is_preloaded:
            logging.debug(f"Fetching frame(s) {index} from memory cache.")
            # Let the list handle the slice or index error
            return self._frames[index]

        if isinstance(index, int):
            if index < 0:
                index += self._total_frames
            if not (0 <= index < self._total_frames):
                raise IndexError("Frame index out of range")
            
            with video_capture(self.video_path) as cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = cap.read()
                if not success:
                    raise IOError(f"Failed to read frame {index} from {self.video_path}")
                return self._process_frame(frame)

        elif isinstance(index, slice):
            start, stop, step = index.indices(self._total_frames)
            frames = []
            
            with video_capture(self.video_path) as cap:
                # Efficiently read a range of frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                pos = start
                while pos < stop:
                    success, frame = cap.read()
                    if not success:
                        break # End of video
                    
                    if (pos - start) % step == 0:
                        frames.append(self._process_frame(frame))
                    pos += 1
            return frames
        
        else:
            raise TypeError("Index must be an integer or slice")
            
    def __iter__(self) -> Iterator[np.ndarray]:
        """Allows iterating through all frames of the video sequentially."""
        if self.is_preloaded:
            yield from self._frames
        else:
            with video_capture(self.video_path) as cap:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    yield self._process_frame(frame)

    def get_frames_range(self, start: int, end: int) -> List[np.ndarray]:
        """
        Retrieves a range of frames efficiently.

        This method is a convenient wrapper around the class's slice access (`video[start:end]`).
        If pre-loaded, it returns a slice from memory; otherwise, it seeks to the
        start frame and reads sequentially from the file.

        Args:
            start: The starting frame index (inclusive).
            end: The ending frame index (exclusive).

        Returns:
            A list of frames as NumPy arrays. Returns an empty list if the
            range is invalid or out of bounds.
        """
        # The slice notation `self[start:end]` inherently handles bounds gracefully,
        # returning an empty list for invalid ranges (e.g., start >= end or out of bounds).
        start = max(0, start)
        return self[start:end]

    def get_frames(self) -> List[np.ndarray]:
        """
        Retrieves all frames from the video.

        This is equivalent to `get_frames_range(0, total_frames)`. If the video
        is already pre-loaded, it returns the cached list directly.

        Returns:
            A list containing all frames as NumPy arrays.
        """
        if self.is_preloaded:
            return self._frames
        return self.get_frames_range(0, self.total_frames)

    @property
    def total_frames(self):
        return self._total_frames
    
    @property
    def width(self) -> int:
        return self._frame_width

    @property
    def height(self) -> int:
        return self._frame_height

    @property
    def fps(self) -> float:
        return self._fps

    def __repr__(self) -> str:
        return (f"VideoMP4Manager(path='{self.video_path.name}', frames={self._total_frames}, "
                f"resolution=({self.width}x{self.height}), preloaded={self.is_preloaded})")