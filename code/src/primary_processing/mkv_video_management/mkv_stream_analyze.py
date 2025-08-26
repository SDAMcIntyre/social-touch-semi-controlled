import csv
from pathlib import Path
import cv2
import numpy as np

# PyK4A is the library for Azure Kinect DK
from pyk4a import PyK4APlayback, K4AException
from pyk4a.config import ColorResolution, DepthMode

# This maps the stable enum to the hardware's known dimensions (width, height).
CUSTOM_RESOLUTION_MAP = {
    ColorResolution.RES_720P: (1280, 720),
    ColorResolution.RES_1080P: (1920, 1080),
    ColorResolution.RES_1440P: (2560, 1440),
    ColorResolution.RES_1536P: (2048, 1536),
    ColorResolution.RES_2160P: (3840, 2160),
    ColorResolution.RES_3072P: (4096, 3072),
}

# This maps the stable DepthMode enum to the hardware's known dimensions (width, height).
CUSTOM_DEPTH_MODE_MAP = {
    DepthMode.NFOV_2X2BINNED: (320, 288),    # Narrow Field of View, 2x2 Binned
    DepthMode.NFOV_UNBINNED: (640, 576),     # Narrow Field of View, Unbinned
    DepthMode.WFOV_2X2BINNED: (512, 512),    # Wide Field of View, 2x2 Binned
    DepthMode.WFOV_UNBINNED: (1024, 1024),   # Wide Field of View, Unbinned
    DepthMode.PASSIVE_IR: (1024, 1024),      # Passive Infrared
}

class MKVStreamAnalyzer:
    """
    Analyzes an Azure Kinect MKV file for stream integrity.

    This class encapsulates the logic for opening an MKV file, reading its
    metadata, and iterating through its frames to validate the color and depth streams.
    
    It is designed to be used as a context manager (with a 'with' statement).
    """

    def __init__(self, video_path: str):
        """
        Initializes the analyzer with the path to the video file.

        Args:
            video_path (str): The path to the input MKV video file.

        Raises:
            FileNotFoundError: If the video_path does not exist.
        """
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Error: The file '{self.video_path}' was not found.")
        
        self.playback = None
        self.metadata = {}

    def __enter__(self):
        """
        Opens the MKV file and prepares for analysis. Context manager entry.

        Returns:
            MKVStreamAnalyzer: The instance of the analyzer.

        Raises:
            PyK4AException: If the file cannot be opened or is corrupted.
        """
        try:
            print(f"Opening '{self.video_path}'...")
            self.playback = PyK4APlayback(self.video_path)
            self.playback.open()
            self._load_metadata()
            print("✅ File opened successfully.")
            return self
        except K4AException as e:
            raise K4AException(f"Failed to open or read MKV file '{self.video_path}'. It may be corrupted or not a valid Kinect recording.") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the MKV file. Context manager exit."""
        if self.playback:
            self.playback.close()
            print("File closed.")

    def _load_metadata(self):
        """Internal method to extract and store expected stream resolutions."""
        config = self.playback.configuration
        
        color_res_enum = config['color_resolution']
        depth_mode_enum = config['depth_mode']

        self.metadata = {
            'expected_color_shape': CUSTOM_RESOLUTION_MAP.get(color_res_enum, (0, 0)),
            'expected_depth_shape': CUSTOM_DEPTH_MODE_MAP.get(depth_mode_enum, (0, 0))[:2]
        }
        print(f"-> Expected Color Resolution (WxH): {self.metadata['expected_color_shape']}")
        print(f"-> Expected Depth Resolution (WxH): {self.metadata['expected_depth_shape']}")

    def analyze_frames(self):
        """
        Analyzes frames and yields a result for each one.

        This is a generator function that processes the MKV frame-by-frame,
        performing validation checks without loading the entire file into memory.

        Yields:
            dict: A dictionary containing the analysis result for a single frame,
                  e.g., {'frame_index': 0, 'has_rgb': True, 'has_depth': False}.
        """
        if not self.playback:
            raise RuntimeError("Playback is not open. Use this method within a 'with' block.")
            
        frame_index = 0
        while True:
            try:
                capture = self.playback.get_next_capture()
                
                rgb_ok = self._validate_rgb(capture)
                depth_ok = self._validate_depth(capture)

                yield {
                    'frame_index': frame_index,
                    'has_rgb': rgb_ok,
                    'has_depth': depth_ok
                }
                
                frame_index += 1

            except EOFError:
                print(f"\nAnalysis finished. Reached end of file. A total of {frame_index} frames were analyzed.")
                break # Graceful exit from the loop

    def _validate_rgb(self, capture) -> bool:
        """Checks if the capture's color image is valid and matches metadata."""
        if capture.color is None:
            return False
            
        actual_color_image = None
        if capture.color.ndim == 1:
            actual_color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
        else:
            actual_color_image = capture.color

        if actual_color_image is not None:
            actual_shape = (actual_color_image.shape[1], actual_color_image.shape[0])
            return actual_shape == self.metadata['expected_color_shape']
        return False

    def _validate_depth(self, capture) -> bool:
        """Checks if the capture's depth data is valid and matches metadata."""
        if capture.depth is None or capture.transformed_depth_point_cloud is None:
            return False
            
        actual_shape = (capture.depth.shape[1], capture.depth.shape[0])
        return actual_shape == self.metadata['expected_depth_shape']

# --- Separated Concerns: The CSV Writer ---

def save_report_to_csv(report_iterator, output_csv_path: str):
    """
    Saves the report data from an iterator to a CSV file.

    Args:
        report_iterator: An iterator that yields dictionaries with frame data.
        output_csv_path (str): The path to save the CSV file.
    """
    print(f"Writing report to '{output_csv_path}'...")
    with open(output_csv_path, 'w', newline='') as csv_file:
        # Define header based on the keys of the first item yielded
        first_item = next(report_iterator, None)
        if not first_item:
            print("Warning: No frames to analyze.")
            return

        writer = csv.DictWriter(csv_file, fieldnames=first_item.keys())
        writer.writeheader()
        writer.writerow(first_item) # Write the first item we already fetched
        
        # Write the rest of the items
        processed_count = 1
        for row_data in report_iterator:
            writer.writerow(row_data)
            processed_count += 1
            print(f"Processed frame: {processed_count}", end='\r')

    print(f"\n✅ Success! Report saved.")


## How to Run the Refactored Code

if __name__ == "__main__":
    # Example usage:
    # Replace with the actual path to your MKV file
    input_video = "path/to/your/kinect_recording.mkv"
    
    # Replace with the desired path for your output CSV report
    output_report = "path/to/your/report.csv"

    try:
        # Use the class as a context manager
        with MKVStreamAnalyzer(input_video) as analyzer:
            # 1. The analyzer generates the results
            frame_report_generator = analyzer.analyze_frames()
            
            # 2. The separate writer function consumes the results and saves the file
            save_report_to_csv(frame_report_generator, output_report)

    except (FileNotFoundError, K4AException, RuntimeError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")