"""Backward-compatibility shim — imports from pipelines/rf_population_grid_pipeline.

All code has moved to ``analysis.receptive_field_mapping.pipelines.rf_population_grid_pipeline``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_grid_pipeline
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.pipelines.rf_population_grid_pipeline import *  # noqa: F401,F403
from analysis.receptive_field_mapping.pipelines.rf_population_grid_pipeline import (
    PopulationRFGridConfig,
    build_feature_grid,
    filter_touches_for_cell,
    compute_cell_rf,
    run_population_rf_grid,
    _VALID_NEURON_MODES,
    _save_grid_results,
    _sweep_grid,
    _compute_and_save_baseline,
)
