"""Backward-compatibility shim — imports from metrics/rf_inflection_boundary.

All code has moved to ``analysis.receptive_field_mapping.metrics.rf_inflection_boundary``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_inflection_boundary
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import *  # noqa: F401,F403
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    InflectionBoundary,
    compute_inflection_boundary,
    inflection_boundary_to_dict,
    _compute_masked_laplacian,
    _find_peak_location,
    _is_contour_closed,
    _contour_area_pixels,
    _select_enclosing_contour,
    _select_peak_basin_contour,
    _contour_pixels_to_uv,
    _compute_polygon_area,
    _compute_polygon_perimeter,
    _compute_polygon_centroid,
    _compute_contour_pca,
    _sample_grid_along_contour,
    _diag_save_laplacian_png,
    _to_json_safe,
)
