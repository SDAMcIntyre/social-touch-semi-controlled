"""Backward-compatibility shim — imports from surface/tangent_plane_alignment.

All code has moved to ``analysis.receptive_field_mapping.surface.tangent_plane_alignment``.
This file exists only so that existing ``from analysis.receptive_field_mapping.tangent_plane_alignment
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.tangent_plane_alignment import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.tangent_plane_alignment import (
    align_points,
    camera_settings_to_rotation,
)
