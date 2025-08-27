import os
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image
import glob

# Import the required function and dependencies
from _3_preprocessing._1_sticker_tracking import extract_stickers_xyz_positions
from preprocessing.stickers_analysis.roi import (
    XYZMetadataModel,
    XYZMetadataManager,
    XYZVisualizationHandler,
    XYZDataFileHandler,
    XYZStickerOrchestrator
)


if __name__ == "__main__":
    tiff_folder = Path("/path/to/your/tiff/frames")
    stickers_roi_csv_path = Path("/path/to/handstickers_roi_tracking.csv")
    result_csv_path = Path("/path/to/output/xyz_tracked.csv")
    result_md_path = Path("/path/to/output/xyz_tracked_metadata.json")
    result_video_path = Path("/path/to/output/xyz_tracked.mp4")

    """
    # For MKV input
    extract_stickers_xyz_positions(
        source='path/to/video.mkv',
        input_csv_path='path/to/input.csv',
        output_csv_path='path/to/output.csv',
        input_type='mkv'
    ) 
    """
    # For TIFF input
    extract_stickers_xyz_positions(
        source=tiff_folder,
        input_csv_path=stickers_roi_csv_path,
        output_csv_path=result_md_path,
        metadata_path=result_md_path,
        monitor=True,
        video_path=result_video_path,
        input_type='tiff'
    )
