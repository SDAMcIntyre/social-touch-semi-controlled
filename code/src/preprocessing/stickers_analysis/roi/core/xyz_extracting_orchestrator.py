import os
from typing import Dict, Any, Optional
from pathlib import Path

from utils.package_utils import load_pyk4a

# Import the new processor and other required modules
from .xyz_extractor import BasePositionExtractor, MKVPositionExtractor, TIFFPositionExtractor
from ..data_access.roi_tracked_filehandler import ROITrackedFileHandler

# Type hints for injected dependencies
from ..models.xyz_metadata_manager import XYZMetadataManager
from ..gui.xyz_monitor_gui import XYZVisualizationHandler
from ..data_access.xyz_data_filehandler import XYZDataFileHandler
from ..models.xyz_metadata_model import XYZMetadataModel


class XYZStickerOrchestrator:
    """
    Orchestrates the process of extracting 3D sticker positions by delegating
    responsibilities to specialized components.
    """

    def __init__(
        self,
        config: XYZMetadataModel,
        metadata_manager: XYZMetadataManager,
        result_writer: XYZDataFileHandler,
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
        self.config = config
        self.metadata = metadata_manager
        self.visualizer = visualizer
        self.result_writer = result_writer

    def _create_extractor(self, tracked_data) -> BasePositionExtractor:
        """Creates the appropriate position extractor based on input type."""
        if self.config.input_type == 'mkv':
            PyK4APlayback = load_pyk4a()
            if PyK4APlayback is None:
                raise ImportError("pyk4a is required for MKV processing")
            playback = PyK4APlayback(str(self.config.source_path))
            return MKVPositionExtractor(playback, tracked_data)
        else:  # tiff
            return TIFFPositionExtractor(self.config.source_path, tracked_data)

    def run(self):
        """Executes the entire extraction and processing pipeline."""
        print("Starting sticker 3D position extraction...")
        
        processed_count = 0
        try:
            # 1. Load Data
            tracked_data_iohandler = ROITrackedFileHandler(self.config.input_csv_path)
            tracked_data = tracked_data_iohandler.load_all_data()
            sticker_names = list(tracked_data.keys())
            self.metadata.update_processing_detail("stickers_found", sticker_names)

            # 2. Create appropriate extractor and process
            extractor = self._create_extractor(tracked_data)
            
            # Setup visualization if enabled
            callback = None
            if self.visualizer and self.visualizer.is_enabled:
                callback = self._visualize_frame_callback
                
            # Process frames
            results_data, processed_count = extractor.extract_positions(
                on_frame_processed=callback
            )

            # 3. Save Results
            self.result_writer.save(results_data, sticker_names, self.config.output_csv_path)
            self.metadata.set_status("Completed")

        except Exception as e:
            self.metadata.set_status("Failed", str(e))
            print(f"\nProcessing failed: {e}")
            raise
        finally:
            self.metadata.finalize(processed_count)
            if self.visualizer:
                self.visualizer.release()

    def _visualize_frame_callback(self, frame_index: int, frame_data: Any, monitoring_data: Dict[str, Any]) -> bool:
        """
        A callback function passed to the processor to handle visualization for each frame.

        Args:
            frame_index: The index of the current frame.
            frame_data: Either a K4A capture object (MKV) or numpy array (TIFF).
            monitoring_data: The processed 3D data for visualization.

        Returns:
            True if the user has requested to stop the visualization, otherwise False.
        """
        if self.config.input_type == 'mkv':
            depth_data = frame_data.transformed_depth
        else:  # tiff
            depth_data = frame_data
            
        visual_frame = self.visualizer.create_frame(
            frame_index, depth_data, monitoring_data
        )
        should_stop = self.visualizer.process_frame(visual_frame)
        return should_stop