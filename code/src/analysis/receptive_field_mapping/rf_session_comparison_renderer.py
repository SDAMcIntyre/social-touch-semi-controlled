"""Backward-compatibility shim — imports from rendering/rf_session_comparison_renderer.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_session_comparison_renderer``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_session_comparison_renderer
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_session_comparison_renderer import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_session_comparison_renderer import (
    render_session_comparison_heatmap,
    run_session_comparison_visualization,
    _build_session_feature_matrix,
    _nan_safe_correlation_distance,
    _cluster_session_rows,
)
