from dataclasses import dataclass
from typing import List, TypeVar

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
    video_filename: str
    frame_id: int
    region_of_interest: RegionOfInterest
    frame_width: int
    frame_height: int
    fps: float
    nframes: int
    fourcc_str: str


# Define a generic TypeVar bound to the base class
T = TypeVar('T', bound=ForearmParameters)

def sort_forearm_parameters_by_video_and_frame(
    parameters_list: List[T]
) -> List[T]:
    """
    Sorts a list of ForearmParameters objects by video_filename and frame_id.
    """
    return sorted(parameters_list, key=lambda p: (p.video_filename, p.frame_id))