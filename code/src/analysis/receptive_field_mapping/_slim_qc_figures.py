"""Backward-compatibility shim — imports from surface/slim_qc_figures.

All code has moved to ``analysis.receptive_field_mapping.surface.slim_qc_figures``.
This file exists only so that existing ``from analysis.receptive_field_mapping._slim_qc_figures
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.slim_qc_figures import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.slim_qc_figures import (
    plot_slim_uv_panel,
    plot_slim_distortion_panel,
    save_slim_qc_figures,
)
