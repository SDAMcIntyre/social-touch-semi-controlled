"""Backward-compatibility shim — imports from metrics/rf_baseline_deviation.

All code has moved to ``analysis.receptive_field_mapping.metrics.rf_baseline_deviation``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_baseline_deviation
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.metrics.rf_baseline_deviation import *  # noqa: F401,F403
from analysis.receptive_field_mapping.metrics.rf_baseline_deviation import (
    BaselineDeviationMetrics,
    compute_baseline_deviation,
    _safe_ratio,
    _centroid_3d_distance,
    _vertex_set_overlap,
)
