"""Backward-compatibility shim — imports from surface/rf_projection.

All code has moved to ``analysis.receptive_field_mapping.surface.rf_projection``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_projection
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.rf_projection import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.rf_projection import (
    project_tangent_plane,
    fit_cylinder_axis,
    project_cylindrical_unwrap,
    project_slim,
    project_to_2d,
)
