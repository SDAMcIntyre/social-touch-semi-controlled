import os
from typing import Dict, Any, Optional
from pathlib import Path

# Make pyk4a import optional
try:
    from pyk4a import PyK4APlayback, K4AException
    HAVE_PYK4A = True
except ImportError:
    HAVE_PYK4A = False
    K4AException = Exception

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
        """Checks for the existence of input sources."""
        source = Path(self.config.source_path)
        if self.config.input_type == 'mkv':
            if not source.is_file():
                raise FileNotFoundError(f"Source video not found: {source}")
            if not HAVE_PYK4A:
                raise ImportError("pyk4a is required for MKV processing")
        else:  # tiff
            if not source.is_dir():
                raise FileNotFoundError(f"Source TIFF directory not found: {source}")

    def _process_mkv(self, tracked_data, sticker_names):
        """Process MKV video input."""
        processed_count = 0
        with PyK4APlayback(self.config.source_path) as playback:
            print(f"Successfully opened MKV: {self.config.source_path}")
            
            # Setup visualization if enabled
            if self.visualizer and self.visualizer.is_enabled:
                # fps = self.metadata.add_mkv_metadata(playback)
                fps = self.metadata.populate_from_mkv(playback)
                self.visualizer.setup_writer(fps)
                callback = self._visualize_frame_callback
            else:
                callback = None
            
            processor = Sticker3DPositionExtractor(playback, tracked_data)
            results_data, processed_count = processor.extract_positions(
                on_frame_processed=callback
            )
            
        return results_data, processed_count
    
    def _process_tiff(self, tracked_data, sticker_names):
        """Process TIFF frames input."""
        from PIL import Image
        import numpy as np
        
        source_dir = Path(self.config.source_path)
        tiff_files = sorted(list(source_dir.glob("*.tiff")))
        if not tiff_files:
            tiff_files = sorted(list(source_dir.glob("*.tif")))
            
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {source_dir}")
            
        # Get metadata from first frame
        self.metadata.populate_from_tiff(source_dir)
        
        results_data = {name: [] for name in sticker_names}
        processed_count = 0
        
        for frame_idx, tiff_path in enumerate(tiff_files):
            with Image.open(tiff_path) as img:
                depth_frame = np.array(img, dtype=np.float32)
                
                # Process frame
                frame_results = self._process_tiff_frame(
                    depth_frame, 
                    frame_idx, 
                    tracked_data
                )
                
                # Store results
                for name in sticker_names:
                    if name in frame_results:
                        results_data[name].append(frame_results[name])
                
                processed_count += 1
                
                if self.visualizer and self.visualizer.is_enabled:
                    should_stop = self._visualize_frame_callback(
                        frame_idx, depth_frame, frame_results
                    )
                    if should_stop:
                        break
                        
        return results_data, processed_count

    def _process_tiff_frame(self, depth_frame, frame_idx, tracked_data):
        """Process a single TIFF frame."""
        # Implementation depends on your specific processing needs
        # This is a placeholder - you'll need to implement the actual processing
        frame_results = {}
        for sticker_name, positions in tracked_data.items():
            if frame_idx < len(positions):
                x, y = positions[frame_idx]
                z = depth_frame[int(y), int(x)]
                frame_results[sticker_name] = [x, y, z]
        return frame_results

    def run(self):
        """Executes the entire extraction and processing pipeline."""
        print("Starting sticker 3D position extraction...")
        self._validate_inputs()
        
        processed_count = 0
        try:
            # 1. Load Data
            tracked_data_iohandler = ROITrackedFileHandler(self.config.input_csv_path)
            tracked_data = tracked_data_iohandler.load_all_data()
            sticker_names = list(tracked_data.keys())
            self.metadata.update_processing_detail("stickers_found", sticker_names)

            # 2. Process input based on type
            if self.config.input_type == 'mkv':
                results_data, processed_count = self._process_mkv(tracked_data, sticker_names)
            else:  # tiff
                results_data, processed_count = self._process_tiff(tracked_data, sticker_names)
            
            # 3. Save Results
            self.result_writer.save(results_data, sticker_names, self.config.output_csv_path)
            self.metadata.set_status("Completed")

        except (K4AException, FileNotFoundError, ValueError) as e:
            self.metadata.set_status("Failed", str(e))
            print(f"\nProcessing failed: {e}")
            raise
        finally:
            self.metadata.finalize(processed_count)
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