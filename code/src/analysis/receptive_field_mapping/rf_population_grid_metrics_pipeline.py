"""Backward-compatibility shim — imports from pipelines/rf_population_grid_metrics_pipeline.

All code has moved to ``analysis.receptive_field_mapping.pipelines.rf_population_grid_metrics_pipeline``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_grid_metrics_pipeline
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.pipelines.rf_population_grid_metrics_pipeline import *  # noqa: F401,F403
from analysis.receptive_field_mapping.pipelines.rf_population_grid_metrics_pipeline import (
    PopulationRFGridMetricsConfig,
    run_population_rf_grid_metrics,
    _REQUIRED_NPZ_KEYS,
    _REQUIRED_BASELINE_NPZ_KEYS,
    _DEVIATION_IDENTITY,
    _load_grid_npz,
    _load_baseline_npz,
    _compute_baseline_metrics,
    _build_metrics_dataframe,
    _build_baseline_comparison_dataframe,
)
