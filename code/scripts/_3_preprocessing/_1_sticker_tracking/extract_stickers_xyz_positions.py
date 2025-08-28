import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import glob
import cv2
import importlib
from utils.package_utils import load_pyk4a

from preprocessing.stickers_analysis.roi import (
    XYZMetadataModel,
    XYZMetadataManager,
    XYZVisualizationHandler,
    XYZDataFileHandler,
    XYZStickerOrchestrator
)

def extract_stickers_xyz_positions(
        source: Union[str, Path],
        input_csv_path: str,
        output_csv_path: str,
        metadata_path: str = None,      # NEW: Added metadata path argument
        monitor: bool = False,
        video_path: str = None,
        input_type: str = 'mkv'):
    """
    Extracts 3D sticker positions from either MKV video or TIFF frames.
    
    Args:
        source: Path to either MKV file or directory containing TIFF frames
        input_csv_path: Path to input CSV with sticker positions
        output_csv_path: Path to save output CSV
        metadata_path: Optional path to save metadata
        monitor: Whether to show visualization window
        video_path: Optional path to save visualization video
        input_type: Either 'mkv' or 'tiff'
    """
    source = Path(source)
    
    # Validate input type and source
    if input_type == 'mkv':
        if not source.is_file():
            raise FileNotFoundError(f"Source video not found: {source}")
        # Load pyk4a only if needed
        PyK4APlayback = load_pyk4a()
        if PyK4APlayback is None:
            raise ImportError("pyk4a is required for MKV processing but could not be imported")
            
    elif input_type == 'tiff':
        if not source.is_dir():
            raise FileNotFoundError(f"Source TIFF directory not found: {source}")
    else:
        raise ValueError("input_type must be either 'mkv' or 'tiff'")

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Center CSV not found: {input_csv_path}")

    print(f"Starting sticker 3D position extraction using {input_type} input...")

    # 1. Define the configuration parameters for the job.
    config_data = {
        "source_path": str(source),
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "metadata_path": metadata_path,
        "monitor": monitor,
        "video_path": video_path,
        "input_type": input_type
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

def read_tiff_frames(tiff_folder: Union[str, Path]) -> list:
    """Read all TIFF frames from a folder in sorted order."""
    tiff_folder = Path(tiff_folder)
    tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tiff")))
    if not tiff_files:
        tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {tiff_folder}")
    
    return tiff_files

def process_tiff_frame(tiff_path: str) -> np.ndarray:
    """Read and process a single TIFF frame."""
    with Image.open(tiff_path) as img:
        depth_frame = np.array(img, dtype=np.float32)
        return depth_frame

