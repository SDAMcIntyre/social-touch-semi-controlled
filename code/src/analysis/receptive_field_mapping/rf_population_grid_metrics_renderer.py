"""Backward-compatibility shim — imports from rendering/rf_population_grid_metrics_renderer.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_population_grid_metrics_renderer``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_grid_metrics_renderer
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_population_grid_metrics_renderer import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_population_grid_metrics_renderer import (
    IFF_METRICS,
    DEVIATION_METRICS,
    render_grid_metric_heatmap,
    run_population_rf_grid_metrics_visualization,
    _TOUCH_COUNT_VMAX,
    _n_gradient,
    _touch_count_colors,
    _TOUCH_COUNT_CMAP,
    _TOUCH_COUNT_NORM,
    _get_shared_range,
)
