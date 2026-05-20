"""Shared bootstrap helpers for the analysis workflow entry scripts.

Provides session-discovery utilities and a generic stage-runner dispatcher
so ``analysis_workflow_processing.py`` and ``analysis_workflow_viewers.py``
can each be self-contained without duplicating infrastructure code.

Also re-exports domain constants and utility functions from
``shared_constants`` so callers can do ``from analysis.pipeline import
GESTURE_TYPES`` etc.
"""

from .session_discovery import collect_unique_session_dirs, discover_input_items
from .stage_runner import run_pipeline_stages
from .shared_constants import (
    GESTURE_TYPES,
    TOUCH_ID_COLS,
    TOUCH_ID_COLS_WITH_SESSION,
    NERVE_SPIKE_COL,
    CONTACT_POINTS_COL,
    NERVE_FREQ_COL,
    LOCATION_SHARED_COLS,
    LOCATION_BASE_COLS,
    NEURON_MODES,
    session_id_from_path,
    filter_enabled_profiles,
)

__all__ = [
    "collect_unique_session_dirs",
    "discover_input_items",
    "run_pipeline_stages",
    "GESTURE_TYPES",
    "TOUCH_ID_COLS",
    "TOUCH_ID_COLS_WITH_SESSION",
    "NERVE_SPIKE_COL",
    "CONTACT_POINTS_COL",
    "NERVE_FREQ_COL",
    "LOCATION_SHARED_COLS",
    "LOCATION_BASE_COLS",
    "NEURON_MODES",
    "session_id_from_path",
    "filter_enabled_profiles",
]
