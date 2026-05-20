"""Backward-compatibility shim — imports from pipelines/rf_population_map_pipeline.

All code has moved to ``analysis.receptive_field_mapping.pipelines.rf_population_map_pipeline``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_population_map_pipeline
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.pipelines.rf_population_map_pipeline import *  # noqa: F401,F403
from analysis.receptive_field_mapping.pipelines.rf_population_map_pipeline import (
    run_population_rf_maps,
    _SessionCompositeData,
    _save_vertex_data_npz,
    _write_sentinel,
)
