import os
import json
import cv2
import numpy as np
from pathlib import Path

class LEDROIFilesHandler:
    """Handles saving and loading of processing results."""

    def save_metadata(self, file_path: str, metadata: dict):
        """
        Saves a dictionary of metadata to a file in JSON format.
        
        Args:
            file_path (str): The absolute path to the output file.
            metadata (dict): The dictionary containing result metadata.
        """
        serializable_metadata = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in metadata.items()
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metadata, f, indent=4)


    def load_metadata(self, file_path: str) -> dict:
        """
        Loads metadata from a JSON file.
        
        Args:
            file_path (str): The absolute path to the metadata file.
            
        Returns:
            dict: The loaded metadata dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found at: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_video(self, file_path: str, frames: list[np.ndarray], fps: float):
        """
        Saves a list of frames as a video file.
        
        Args:
            file_path (str): The absolute path to the output video file.
            frames (list[np.ndarray]): A list of frames (as NumPy arrays).
            fps (float): The frames per second for the output video.
        """
        if not frames:
            print("Warning: No frames provided to save_video. Nothing to save.")
            return
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        frame_height, frame_width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
        size = (frame_width, frame_height)
        
        out = cv2.VideoWriter(file_path, fourcc, fps, size)
        if not out.isOpened():
            raise IOError(f"Could not open video writer for path: {file_path}")

        for frame in frames:
            out.write(frame)
        
        out.release()