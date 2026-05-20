"""Backward-compatibility shim — imports from metrics/rf_grid_cell_metrics.

All code has moved to ``analysis.receptive_field_mapping.metrics.rf_grid_cell_metrics``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_grid_cell_metrics
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.metrics.rf_grid_cell_metrics import *  # noqa: F401,F403
from analysis.receptive_field_mapping.metrics.rf_grid_cell_metrics import (
    GridCellRFMetrics,
    compute_iff_intensity_metrics,
    compute_topographic_metrics,
    compute_distribution_metrics,
    compute_boundary_shape_metrics,
    compute_grid_cell_metrics,
    _gini_coefficient,
    _RF_METRICS_KEPT_FIELDS,
    _RF_METRICS_INT_FIELDS,
    _RF_METRICS_BOOL_FIELDS,
    _empty_rf_metrics_dict,
    _empty_grid_cell_metrics_dict,
    _empty_row_dict,
    _rf_metrics_to_kept_dict,
)
