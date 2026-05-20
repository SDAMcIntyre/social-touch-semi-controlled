"""Backward-compatibility shim — imports from rendering/rf_2d_renderer.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_2d_renderer``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_2d_renderer
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_2d_renderer import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_2d_renderer import (
    render_2d_heatmap,
    _AXIS_LABELS,
    _draw_hull_2d,
)
