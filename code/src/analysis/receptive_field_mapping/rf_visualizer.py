"""Backward-compatibility shim — imports from rendering/rf_visualizer.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_visualizer``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_visualizer
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_visualizer import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_visualizer import (
    RFVisualizer,
    _BASE_HUES,
)
