"""Backward-compatibility shim — imports from surface/forearm_slim_uv.

All code has moved to ``analysis.receptive_field_mapping.surface.forearm_slim_uv``.
This file exists only so that existing ``from analysis.receptive_field_mapping.forearm_slim_uv
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.surface.forearm_slim_uv import *  # noqa: F401,F403
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    SlimUvCache,
    precompute_forearm_slim_uv,
    load_slim_uv_cache,
    barycentric_uv_lookup,
)
