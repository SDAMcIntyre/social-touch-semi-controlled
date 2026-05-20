"""Backward-compatibility shim — imports from surface/rf_surface_utils.

All code has moved to ``analysis.receptive_field_mapping.surface.rf_surface_utils``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_surface_utils
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.rf_surface_utils import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.rf_surface_utils import (
    load_or_build_forearm_mesh,
    map_scalars_to_mesh,
    mesh_to_pyvista,
    build_delaunay_mesh,
    apply_rotation_to_mesh,
)
