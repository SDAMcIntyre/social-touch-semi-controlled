"""Backward-compatibility shim — imports from rendering/rf_population_map_renderer.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_population_map_renderer``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_map_renderer
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_interpolated_grid,
    render_population_rf_map,
    render_population_rf_composite,
    _PANEL_ORDER,
    _fill_interior_face_holes,
    _interpolate_on_mesh,
    _draw_inflection_boundary,
)
