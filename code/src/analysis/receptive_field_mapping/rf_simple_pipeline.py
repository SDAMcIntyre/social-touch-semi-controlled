"""Backward-compatibility shim — imports from pipelines/rf_simple_pipeline.

All code has moved to ``analysis.receptive_field_mapping.pipelines.rf_simple_pipeline``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_simple_pipeline
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.pipelines.rf_simple_pipeline import *  # noqa: F401,F403
from analysis.receptive_field_mapping.pipelines.rf_simple_pipeline import (
    run_simple_rf_mapping,
)
