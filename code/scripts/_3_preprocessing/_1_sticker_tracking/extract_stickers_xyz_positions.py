import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import cv2
from pyk4a import PyK4APlayback, K4AException


from preprocessing.stickers_analysis.roi import (
    XYZMetadataModel,
    XYZMetadataManager,
    XYZVisualizationHandler,
    XYZDataFileHandler,
    XYZStickerOrchestrator
)

def extract_stickers_xyz_positions(
        source_video: str,
        center_csv_path: str,
        output_csv_path: str,
        metadata_path: str = None,      # NEW: Added metadata path argument
        monitor: bool = False,
        video_path: str = None):
    """
    Extracts 3D sticker positions and optionally shows a standard, non-distorted monitor window.
    """
    if not os.path.exists(source_video): raise FileNotFoundError(f"Source video not found: {source_video}")
    if not os.path.exists(center_csv_path): raise FileNotFoundError(f"Center CSV not found: {center_csv_path}")
    print("Starting sticker 3D position extraction...")

    # 1. Define the configuration parameters for the job.
    config_data = {
        "source_video_path": source_video,
        "center_csv_path": center_csv_path,
        "output_csv_path": output_csv_path,
        "metadata_path": metadata_path,
        "monitor": monitor,
        "video_path": video_path
    }
    
    # Create the typed config object.
    config = XYZMetadataModel(**config_data)

    # 2. Instantiate each dependency.
    #    These are the "services" the orchestrator will use.
    metadata_manager = XYZMetadataManager(config)
    result_writer = XYZDataFileHandler()
    visualizer = XYZVisualizationHandler(config)

    # 3. Instantiate the main orchestrator with its dependencies.
    orchestrator = XYZStickerOrchestrator(
        config=config,
        metadata_manager=metadata_manager,
        result_writer=result_writer,
        visualizer=visualizer
    )

    # 4. Execute the process! 🚀
    try:
        orchestrator.run()
        print("✅ Processing finished successfully!")
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
