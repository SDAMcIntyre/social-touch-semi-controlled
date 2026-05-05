"""Receptive field mapping.

Cluster-based spike-count pipeline (rf_cluster_pipeline / rf_cluster_visualizer).
Simple per-neuron spike-position pipeline (rf_simple_pipeline).
"""

from .rf_cluster_pipeline import (
    run_cluster_rf_extraction,
    run_cluster_rf_mapping,
    run_cluster_rf_visualization,
    precompute_explorer_caches,
    launch_feature_space_explorer,
)
from .rf_metrics import RFMetrics, compute_rf_metrics
from .rf_projection import project_to_2d
from .rf_simple_pipeline import run_simple_rf_mapping

# rf_camera_angle_task is intentionally NOT imported here.
# It chains through preprocessing.forearm_extraction → pyk4a (Windows-only),
# which breaks macOS runs of map_receptive_fields_simple. Import it directly
# in the clustered-RF flow functions that actually need it.

__all__ = [
    "project_to_2d",
    "run_cluster_rf_extraction",
    "run_cluster_rf_mapping",
    "run_cluster_rf_visualization",
    "precompute_explorer_caches",
    "launch_feature_space_explorer",
    "RFMetrics",
    "compute_rf_metrics",
    "run_simple_rf_mapping",
]
