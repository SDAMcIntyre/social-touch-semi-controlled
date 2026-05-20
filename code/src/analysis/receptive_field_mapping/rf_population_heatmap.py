"""Backward-compatibility shim — imports from data/rf_population_heatmap.

All code has moved to ``analysis.receptive_field_mapping.data.rf_population_heatmap``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_heatmap
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.data.rf_population_heatmap import *  # noqa: F401,F403
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    GESTURE_TYPES,
    compute_rf_heatmap,
    compute_unique_touch_count,
    compute_threshold_from_ratio,
    apply_vertex_threshold,
    build_gesture_touch_indices,
)
