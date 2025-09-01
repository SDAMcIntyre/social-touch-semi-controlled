from dataclasses import dataclass

# ----------------------------------------------------------------------------
# 1. Data Storage Classes
# These classes define the structure of your data. Using dataclasses makes
# them lightweight and automatically provides useful methods like __init__.
# ----------------------------------------------------------------------------

@dataclass
class Point:
    """Represents a 2D coordinate."""
    x: int
    y: int

@dataclass
class RegionOfInterest:
    """Defines a rectangular region using two corner points."""
    top_left_corner: Point
    bottom_right_corner: Point

@dataclass
class ForearmParameters:
    """A comprehensive container for all video-related metadata."""
    reference_frame_idx: int
    region_of_interest: RegionOfInterest
    frame_width: int
    frame_height: int
    fps: float
    nframes: int
    fourcc_str: str
    