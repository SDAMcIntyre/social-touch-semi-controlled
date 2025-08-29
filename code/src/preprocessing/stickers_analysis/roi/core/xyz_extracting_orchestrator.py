import os
import pandas as pd
from typing import Dict, Any

from pyk4a import PyK4APlayback, K4AException

# Import the new processor and other required modules
from .xyz_extractor import Sticker3DPositionExtractor

# Type hints for injected dependencies
from ..gui.xyz_monitor_gui import XYZVisualizationHandler
from ..models.roi_tracked_data import ROITrackedObjects


class XYZStickerOrchestrator:
    """
    Orchestrates the process of extracting 3D sticker positions by delegating
    responsibilities to specialized components.
    """

    def __init__(
        self,
        visualizer: XYZVisualizationHandler
    ):
        """
        Initializes the orchestrator with its dependencies.

        Args:
            config: The configuration model object.
            metadata_manager: The manager for handling metadata I/O.
            result_writer: The writer for saving final 3D data.
            visualizer: The handler for GUI visualization.
        """
        self.visualizer = visualizer

    def run(self, source_video_path: str, tracked_data: ROITrackedObjects):
        """Executes the entire extraction and processing pipeline."""
        print("Starting sticker 3D position extraction...")
        object_names = list(tracked_data.keys())

        results_data = {}
        for object_name in object_names:
            results_data[object_name] = []

        try:
            with PyK4APlayback(source_video_path) as playback:
                # Wrap the playback object with our iterator
                for frame_index, capture in enumerate(self.playback_to_iterator(playback)):
                    print(f"Processing frame: {frame_index}")

                    for object_name in object_names:
                        tracked_obj_row = tracked_data[object_name].loc[frame_index]
                        if tracked_obj_row["status"] != "Failed":
                            (coords_3d, monitor_data) = Sticker3DPositionExtractor.get_xyz(
                                tracked_obj_row, 
                                capture.transformed_depth_point_cloud)
                        else:
                            (coords_3d, monitor_data) = Sticker3DPositionExtractor.get_empty_xyz()
                        results_data[object_name].append({**coords_3d, **monitor_data})

                print("Finished processing all frames.")

        except K4AException as e:
            print(f"Failed to open or process playback file: {e}")
        except Exception as e:
            print(f"\nProcessing failed: {e}")
            raise
        finally:
            if self.visualizer:
                self.visualizer.release()
        
        return results_data, frame_index

    def playback_to_iterator(self, playback: PyK4APlayback):
        """A generator to make PyK4APlayback iterable."""
        while True:
            try:
                yield playback.get_next_capture()
            except EOFError:
                # End of the recording, the generator will stop.
                break
            
    def _visualize_frame_callback(self, frame_index: int, capture: 'Capture', monitoring_data: Dict[str, Any]) -> bool:
        """
        A callback function passed to the processor to handle visualization for each frame.

        Args:
            frame_index: The index of the current frame.
            capture: The K4A capture object for the frame.
            monitoring_data: The processed 3D data for visualization.

        Returns:
            True if the user has requested to stop the visualization, otherwise False.
        """
        visual_frame = self.visualizer.create_frame(
            frame_index, capture.transformed_depth, monitoring_data
        )
        should_stop = self.visualizer.process_frame(visual_frame)
        return should_stop