"""Backward-compatibility shim for rf_cluster_pipeline.

The cluster pipeline has been split into focused modules under
``receptive_field_mapping/pipelines/``:

- ``pipelines/rf_cluster_pipeline.py``           — extraction orchestration
- ``pipelines/rf_cluster_metrics_pipeline.py``   — RF metrics computation
- ``pipelines/rf_cluster_visualization_pipeline.py`` — heatmap rendering
- ``pipelines/rf_cluster_gui_launchers.py``      — GUI launcher functions

All public symbols are re-exported here so existing imports continue to work
without modification.
"""

from .pipelines.rf_cluster_pipeline import (
    run_cluster_rf_extraction,
    run_cluster_rf_mapping,
    _build_pairs,
    _format_cluster_folder,
    _build_cluster_description,
    _resolve_explorer_session_paths,
)
from .pipelines.rf_cluster_metrics_pipeline import run_cluster_rf_metrics_computation
from .pipelines.rf_cluster_visualization_pipeline import run_cluster_rf_visualization
from .pipelines.rf_cluster_gui_launchers import (
    precompute_explorer_caches,
    launch_feature_space_explorer,
    launch_touch_playback_explorer,
    launch_single_touch_rf_explorer,
    launch_touch_population_explorer,
    launch_gallery_viewer,
    launch_rf_camera_settings_viewer,
)

__all__ = [
    "run_cluster_rf_extraction",
    "run_cluster_rf_metrics_computation",
    "run_cluster_rf_visualization",
    "run_cluster_rf_mapping",
    "precompute_explorer_caches",
    "launch_feature_space_explorer",
    "launch_touch_playback_explorer",
    "launch_single_touch_rf_explorer",
    "launch_touch_population_explorer",
    "launch_gallery_viewer",
    "launch_rf_camera_settings_viewer",
    # Private helpers re-exported for any internal consumers
    "_build_pairs",
    "_format_cluster_folder",
    "_build_cluster_description",
    "_resolve_explorer_session_paths",
]
