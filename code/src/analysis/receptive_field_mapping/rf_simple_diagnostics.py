"""Backward-compatibility shim — imports from rendering/rf_simple_diagnostics.

All code has moved to ``analysis.receptive_field_mapping.rendering.rf_simple_diagnostics``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_simple_diagnostics
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.rendering.rf_simple_diagnostics import *  # noqa: F401,F403
from analysis.receptive_field_mapping.rendering.rf_simple_diagnostics import (
    diagnose_population_data,
    diagnose_spike_extraction,
    diagnose_aggregation,
    diagnose_projection,
    run_diagnostics,
    _BG,
    _CMAP,
    _apply_dark_style,
)
