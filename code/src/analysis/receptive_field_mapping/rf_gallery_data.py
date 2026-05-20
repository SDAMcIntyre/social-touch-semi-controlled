"""Backward-compatibility shim — imports from data/rf_gallery_data.

All code has moved to ``analysis.receptive_field_mapping.data.rf_gallery_data``.
This file exists only so that existing ``from analysis.receptive_field_mapping.rf_gallery_data
import ...`` call-sites continue to work without modification.
"""
from analysis.receptive_field_mapping.data.rf_gallery_data import *  # noqa: F401,F403
from analysis.receptive_field_mapping.data.rf_gallery_data import (
    GalleryCell,
    GalleryData,
    load_gallery_data,
    _is_noise_label,
    _cluster_folder_to_label,
    _sort_key_for_cluster_label,
    _load_cluster_description,
    _load_neuron_cluster_touches_safe,
    _load_rf_metrics,
    _load_forearm_geometry,
)
