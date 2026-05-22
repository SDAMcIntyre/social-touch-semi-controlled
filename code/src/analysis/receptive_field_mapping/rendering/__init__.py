"""Rendering sub-package: matplotlib and PyVista image output for RF mapping.

Note: rf_session_comparison_renderer imports pipeline modules, which creates a
cross-layer dependency. It is not eagerly imported here to avoid circular imports.
Import it directly: from analysis.receptive_field_mapping.rendering.rf_session_comparison_renderer import ...
"""

from .rf_2d_renderer import render_2d_heatmap
from .rf_cluster_visualizer import RFRenderContext, render_forearm_heatmap
from .rf_population_map_renderer import (
    compute_interpolated_grid,
    render_population_rf_map,
    render_population_rf_composite,
    render_population_rf_standalone_interpolated,
)
from .rf_population_grid_metrics_renderer import (
    render_grid_metric_heatmap,
    run_population_rf_grid_metrics_visualization,
)
from .rf_simple_diagnostics import (
    diagnose_population_data,
    diagnose_spike_extraction,
    diagnose_aggregation,
    diagnose_projection,
    run_diagnostics,
)
from .rf_visualizer import RFVisualizer

__all__ = [
    # rf_2d_renderer
    "render_2d_heatmap",
    # rf_cluster_visualizer
    "RFRenderContext",
    "render_forearm_heatmap",
    # rf_population_map_renderer
    "compute_interpolated_grid",
    "render_population_rf_map",
    "render_population_rf_composite",
    "render_population_rf_standalone_interpolated",
    # rf_population_grid_metrics_renderer
    "render_grid_metric_heatmap",
    "run_population_rf_grid_metrics_visualization",
    # rf_simple_diagnostics
    "diagnose_population_data",
    "diagnose_spike_extraction",
    "diagnose_aggregation",
    "diagnose_projection",
    "run_diagnostics",
    # rf_visualizer
    "RFVisualizer",
    # rf_session_comparison_renderer: not eagerly imported (see module docstring)
    # "render_session_comparison_heatmap",
    # "run_session_comparison_visualization",
]
