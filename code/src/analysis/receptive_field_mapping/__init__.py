"""Receptive field mapping.

Cluster-based spike-count pipeline split across four focused modules:
- ``pipelines/rf_cluster_pipeline``                — extraction orchestration
- ``pipelines/rf_cluster_metrics_pipeline``        — RF metrics computation
- ``pipelines/rf_cluster_visualization_pipeline``  — heatmap rendering
- ``pipelines/rf_cluster_gui_launchers``           — GUI launcher functions

Simple per-neuron spike-position pipeline (rf_simple_pipeline).
"""

from .pipelines.rf_cluster_pipeline import (
    run_cluster_rf_extraction,
    run_cluster_rf_mapping,
)
from .pipelines.rf_cluster_metrics_pipeline import run_cluster_rf_metrics_computation
from .pipelines.rf_cluster_visualization_pipeline import run_cluster_rf_visualization
from .pipelines.rf_cluster_gui_launchers import (
    precompute_explorer_caches,
    launch_feature_space_explorer,
    launch_single_touch_rf_explorer,
    launch_touch_playback_explorer,
    launch_touch_population_explorer,
    launch_gallery_viewer,
    launch_rf_camera_settings_viewer,
)
from .rf_metrics import RFMetrics, compute_rf_metrics
from .rf_projection import project_to_2d
from .rf_simple_pipeline import run_simple_rf_mapping
from .rf_single_touch_pipeline import run_single_touch_rf_mapping
from .rf_population_grid_pipeline import run_population_rf_grid, PopulationRFGridConfig
from .rf_population_grid_metrics_pipeline import run_population_rf_grid_metrics, PopulationRFGridMetricsConfig
from .rf_population_grid_metrics_renderer import run_population_rf_grid_metrics_visualization
from .rf_session_comparison_renderer import run_session_comparison_visualization
from .rf_baseline_deviation import BaselineDeviationMetrics, compute_baseline_deviation
from .rf_population_map_pipeline import run_population_rf_maps

__all__ = [
    "project_to_2d",
    "run_cluster_rf_extraction",
    "run_cluster_rf_metrics_computation",
    "run_cluster_rf_mapping",
    "run_cluster_rf_visualization",
    "precompute_explorer_caches",
    "launch_feature_space_explorer",
    "launch_single_touch_rf_explorer",
    "launch_touch_playback_explorer",
    "launch_touch_population_explorer",
    "launch_gallery_viewer",
    "launch_rf_camera_settings_viewer",
    "RFMetrics",
    "compute_rf_metrics",
    "run_simple_rf_mapping",
    "run_single_touch_rf_mapping",
    "run_population_rf_grid",
    "PopulationRFGridConfig",
    "run_population_rf_grid_metrics",
    "PopulationRFGridMetricsConfig",
    "run_population_rf_grid_metrics_visualization",
    "run_session_comparison_visualization",
    "BaselineDeviationMetrics",
    "compute_baseline_deviation",
    "run_population_rf_maps",
]
