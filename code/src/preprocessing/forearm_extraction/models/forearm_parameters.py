from dataclasses import dataclass
from typing import List, Optional, TypeVar

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
    """Defines a rectangular region using two corner points (axis-aligned bounding box) and an optional rotation angle.

    The AABB fields (``top_left_corner``, ``bottom_right_corner``) are the authoritative
    source for all downstream extraction code (e.g. ``get_3d_cuboid_from_roi``).

    The optional centre-based fields (``center_x``, ``center_y``, ``width``, ``height``)
    store the original rotated rectangle drawn by the user.  They are populated by the
    annotation script and used exclusively to pre-fill the ROI GUI on reprocessing.
    When absent (``None``) the GUI falls back to the AABB-based pre-fill.
    """
    top_left_corner: Point
    bottom_right_corner: Point
    angle_deg: float = 0.0
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

@dataclass
class ForearmParameters:
    """A comprehensive container for all video-related metadata."""
    video_filename: str
    frame_ids: List[int]
    representative_frame_id: int
    region_of_interest: RegionOfInterest
    frame_width: int
    frame_height: int
    fps: float
    nframes: int
    fourcc_str: str

    @property
    def frame_id(self) -> int:
        """Backward-compatible accessor — returns representative_frame_id."""
        return self.representative_frame_id

    @property
    def is_averaged(self) -> bool:
        """True when this capture was derived from more than one depth frame."""
        return len(self.frame_ids) > 1

    def build_output_stem(self, video_stem: str) -> str:
        """Returns the filename stem for outputs produced from this capture.

        Single-frame:  {video_stem}_frame_{id:04d}
        Averaged:      {video_stem}_frames_{lo:04d}-{hi:04d}_avg_N{count}
        """
        if self.is_averaged:
            lo, hi = min(self.frame_ids), max(self.frame_ids)
            return f"{video_stem}_frames_{lo:04d}-{hi:04d}_avg_N{len(self.frame_ids)}"
        return f"{video_stem}_frame_{self.representative_frame_id:04d}"


# Define a generic TypeVar bound to the base class
T = TypeVar('T', bound=ForearmParameters)

def sort_forearm_parameters_by_video_and_frame(
    parameters_list: List[T]
) -> List[T]:
    """
    Sorts a list of ForearmParameters objects by video_filename and representative_frame_id.
    """
    return sorted(parameters_list, key=lambda p: (p.video_filename, p.representative_frame_id))