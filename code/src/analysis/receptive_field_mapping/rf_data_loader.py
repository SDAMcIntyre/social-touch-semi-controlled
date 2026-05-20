"""Backward-compatibility shim — imports from data/rf_data_loader.

All code has moved to ``analysis.receptive_field_mapping.data.rf_data_loader``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_data_loader
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.data.rf_data_loader import *  # noqa: F401,F403
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertices,
    load_forearm_vertex_colors,
    resolve_forearm_ply,
    parse_contact_points,
    load_grouped_spatial_data,
    _discretize_column,
    _build_group_lookup,
)
