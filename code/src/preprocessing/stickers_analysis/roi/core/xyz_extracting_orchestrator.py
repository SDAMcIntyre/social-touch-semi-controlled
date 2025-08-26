import os
from typing import Dict, Any

from pyk4a import PyK4APlayback, K4AException

# Import the new processor and other required modules
from .xyz_extractor import Sticker3DPositionExtractor
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

    def _validate_inputs(self):
        """Checks for the existence of the source video."""
        if not os.path.exists(self.config.source_video):
            raise FileNotFoundError(f"Source video not found: {self.config.source_video}")

    def run(self):
        """Executes the entire extraction and processing pipeline."""
        print("Starting sticker 3D position extraction...")
        self._validate_inputs()
        
        processed_count = 0
        try:
            # 1. Load Data
            tracked_data_iohandler = ROITrackedFileHandler(self.config.center_csv_path)
            tracked_data = tracked_data_iohandler.load_all_data()
            sticker_names = list(tracked_data.keys())
            self.metadata.update_processing_details("stickers_found", sticker_names)

            # 2. Process Video using a context manager for the playback resource
            with PyK4APlayback(self.config.source_video) as playback:
                print(f"Successfully opened MKV: {self.config.source_video}")
                
                # Setup visualization if enabled
                if self.visualizer and self.visualizer.is_enabled:
                    fps = self.metadata.add_mkv_metadata(playback)
                    self.visualizer.setup_writer(fps)
                    callback = self._visualize_frame_callback
                else:
                    callback = None
                
                # 3. Delegate Core Logic to the extractor
                processor = Sticker3DPositionExtractor(playback, tracked_data)
                results_data, processed_count = processor.extract_positions(
                    on_frame_processed=callback
                )

            # 4. Save Results
            self.result_writer.save(results_data, sticker_names, self.config.output_csv_path)
            self.metadata.set_status("Completed")

        except (K4AException, FileNotFoundError, ValueError) as e:
            self.metadata.set_status("Failed", str(e))
            print(f"\nProcessing failed: {e}")
            raise
        finally:
            self.metadata.save(processed_count)
            if self.visualizer:
                self.visualizer.release()

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