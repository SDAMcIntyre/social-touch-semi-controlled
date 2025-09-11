# File: _3_preprocessing/_1_sticker_tracking/__init__.py

"""
This file makes key sticker tracking functions available at the package level,
simplifying imports elsewhere in the project.
"""
from .is_2d_stickers_tracking_valid import is_2d_stickers_tracking_valid

# automatic computation
# - 2d tracking process
from .track_handstickers_roi import track_objects_in_video
# - ellipses processes
from .standardize_handstickers_roi import generate_standard_roi_size_dataset
from .create_standardized_roi_videos import create_standardized_roi_videos
from .create_color_correlation_videos import create_color_correlation_videos
from .define_handstickers_colorspaces_from_roi import define_handstickers_colorspaces_from_roi
from .define_handstickers_color_threshold import define_handstickers_color_threshold
from .fit_ellipses_on_correlation_videos import fit_ellipses_on_correlation_videos

# - XYZ stickers location process
from .extract_stickers_xyz_positions import extract_stickers_xyz_positions

# user review or tasks
from .review_tracked_handstickers_roi import review_tracked_objects_in_video
from .view_xyz_stickers_with_depth_data import view_xyz_stickers_on_depth_data

