"""Backward-compatibility shim — imports from data/touch_population_data.

All code has moved to ``analysis.receptive_field_mapping.data.touch_population_data``.
This file exists only so that existing ``from analysis.receptive_field_mapping.touch_population_data
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.data.touch_population_data import *  # noqa: F401,F403
from analysis.receptive_field_mapping.data.touch_population_data import (
    PopulationData,
    ViewerSessionData,
    PopulationRFData,
    load_viewer_session_data,
    load_population_data,
    load_population_rf_data,
    _CACHE_SCHEMA_VERSION,
    _REQUIRED_COLUMNS,
    _bracket_re,
    _ts,
    _population_cache_path,
    _save_population_cache,
    _load_population_cache,
    _merge_stage3_features,
)
