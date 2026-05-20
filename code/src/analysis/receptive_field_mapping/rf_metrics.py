"""Backward-compatibility shim — imports from metrics/rf_metrics.

All code has moved to ``analysis.receptive_field_mapping.metrics.rf_metrics``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_metrics
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.metrics.rf_metrics import *  # noqa: F401,F403
from analysis.receptive_field_mapping.metrics.rf_metrics import (
    RFMetrics,
    compute_rf_metrics,
    metrics_to_dict,
    metrics_to_row,
    _compute_weighted_centroid,
    _compute_convex_hull_area,
    _compute_threshold_boundary,
    _compute_hotspot,
    _fit_ellipse,
    _gaussian_2d_model,
    _fit_2d_gaussian,
    _to_python,
)
