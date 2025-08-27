import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from _3_preprocessing._1_sticker_tracking import (
    track_objects_in_video,
    ensure_tracking_is_valid,
    convert_sticker_roi_to_center,

    extract_stickers_xyz_positions,

    unify_objects_rois_size,
    create_windowed_videos,
    fit_ellipses_on_correlation_videos,
    adjust_ellipses_coord_to_frame,
    generate_ellipses_on_frame_video
)


result_csv_path = output_dir / (name_baseline + "_xyz_tracked.csv")
result_video_path = output_dir / (name_baseline + "_xyz_tracked.mp4")
result_md_path = output_dir / (name_baseline + "_xyz_tracked_metadata.json")

extract_stickers_xyz_positions(
    source_video, 
    stickers_roi_csv_path, 
    
    result_csv_path,
    metadata_path=result_md_path,
    video_path=result_video_path,
    monitor=True)