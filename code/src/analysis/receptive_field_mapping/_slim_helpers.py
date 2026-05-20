"""Backward-compatibility shim — imports from surface/slim_helpers.

All code has moved to ``analysis.receptive_field_mapping.surface.slim_helpers``.
This file exists only so that existing ``from analysis.receptive_field_mapping._slim_helpers
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.slim_helpers import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.slim_helpers import (
    clean_mesh,
    boundary_loop,
    canonicalise_uv,
    flatten_slim,
    compute_face_distortion,
    _find_boundary_loops,
    _fill_interior_holes,
    _has_flipped_triangles,
    _tutte_uniform_map,
)
