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

tiff_folder = Path("~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/TEAMS/Documents - Social touch Kinect MNG/data/semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-01/2022-06-15_ST14-01_semicontrolled_block-order01_kinect_depth")
stickers_roi_csv_path = Path("~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/TEAMS/Documents - Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-01/handstickers/2022-06-15_ST14-01_semicontrolled_block-order01_kinect_handstickers_roi_tracking.csv")
result_csv_path = Path("~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/TEAMS/Documents - Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-01/handstickers/2022-06-15_ST14-01_semicontrolled_block-order01_kinect_xyz_tracked_new.csv")
result_md_path = Path("~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/TEAMS/Documents - Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-01/handstickers/2022-06-15_ST14-01_semicontrolled_block-order01_kinect_xyz_tracked_metadata_new.json")
result_video_path = Path("~/Library/CloudStorage/OneDrive-Linköpingsuniversitet/TEAMS/Documents - Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-01/handstickers/2022-06-15_ST14-01_semicontrolled_block-order01_kinect_xyz_tracked.mp4")

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
